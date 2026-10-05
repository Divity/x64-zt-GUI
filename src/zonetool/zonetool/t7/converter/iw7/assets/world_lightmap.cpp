#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "world_lightmap.hpp"

#include "probe_lighting.hpp"
#include "zonetool/t7/converter/iw7/map_common.hpp"
#include "../parallel.hpp"
#include "../gpu.hpp"
#include "../triangle_bvh.hpp"
#include "comworld.hpp"
#include "world_sky.hpp"
#include "world_material.hpp"
#include "xmodel.hpp"
#include "xmodel_mesh.hpp"
#include "world_material_bake.hpp"
#include "material_texture.hpp"
#include "static_model_clusters.hpp"

#include "zonetool/iw7/assets/gfximage.hpp"
#include "zonetool/iw7/assets/gfxlightmap.hpp"

#include <DirectXTex.h>
#include <DirectXPackedVector.h>
#include <utils/string.hpp>

// IW7 side (stock lmap_specenv_..._allprim_sundfd_ssao world shader and stock lightmaps):
// * one lightmap per map: "*lightmap0" with *_primary BC4 W x H (sun shadow distance field, read
//   only while the sun is on), *_secondary BC6H_UF16 W x 2H (rows [0, H) ambient colour, [H, 2H)
//   directional colour) and *_secondunorm R8G8 W x H (the direction in the vertex tangent frame,
//   octahedral: x = (r + g) / 2, y = (r - g) / 2 after * 2 - 1, z = 1 - |x| - |y|); image flags
//   0x3B, 2D, semantic 1, category 2 (lightmap), one level.
// * the world vertex shader passes lmapCoord through; the pixel shader lights with
//   ambient + ssao * directional * saturate(dot(N, dir)) with N normal-mapped and
//   dir = T * x + B * y + N * z, B = cross(N, T) * (tangent.w > 0 ? -1 : 1).
// * primary lights are added at runtime, so the lightmap only carries BO3's indirect light, which
//   BO3 takes from its reflection probes (probe_lighting.cpp).
// * stock atlases pack their charts edge to edge, not aligned to BC blocks.

namespace zonetool::t7
{
	namespace converter::iw7::world_lightmap
	{
		namespace
		{
			constexpr unsigned char no_lightmap = 31;

			constexpr std::uint32_t atlas_width = 4096;
			// the secondary image is twice the atlas' height: 8192 keeps it at D3D11's 16384
			constexpr std::uint32_t atlas_max_height = 8192;
			// a chart's content runs from the centre of its first texel to the centre of its last, so
			// bilinear taps inside it stay on its own texels; one texel of its own values around it keeps
			// rounding in the interpolated coordinates and block compression off its neighbours
			constexpr std::uint32_t chart_margin = 1;
			// the sun's visibility: sun_grid x sun_grid rays over each texel, a signed distance over sun_range texels
			constexpr std::uint32_t sun_grid = 4;
			constexpr double sun_range = 2.0;
			constexpr std::uint32_t max_chart_texels = 1024;
			// a placed model part's texels against the world's: small props took the light grid's few samples as static models
			constexpr float part_texel_factor = 4.0f;
			constexpr float crease_cos = 0.5f; // a triangle more than 60 degrees off its chart's normal starts another
			// the finest lightmap texel, in probe volume texels; the atlas decides how coarse it gets. The sun's visibility
			// shares the atlas and wants it fine: thin bars shadow strips a few units wide
			constexpr float finest_scale = 0.125f;
			constexpr float min_texel_size = 2.0f;
			// a bilinear lookup reads the texels whose centres are less than a texel away on both axes, so
			// every texel within sqrt(2) of a triangle needs that triangle's lighting
			constexpr float sample_reach_squared = 2.0f;
			// projected triangles of one chart overlapping deeper than this (world units) would share texels
			constexpr double overlap_tolerance = 1e-3;

			constexpr float fit_cap_degrees[2] = { 30.0f, 60.0f }; // normals the direction is fitted over
			constexpr auto fit_ring = 8u;
			constexpr auto fit_normals = 1u + 2u * fit_ring;

			using vec3 = std::array<float, 3>;

			vec3 sub(const float* a, const float* b) { return { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; }
			float dot(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
			vec3 cross(const float* a, const float* b)
			{
				return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
			}
			float length(const float* a) { return std::sqrt(dot(a, a)); }
			vec3 normalize(vec3 a)
			{
				const auto l = length(a.data());
				if (l > 0.0f)
				{
					for (auto& c : a)
					{
						c /= l;
					}
				}
				return a;
			}

			// IW7 world vertex attributes (gfxworld.cpp packs them): component = k / 1023 * 2 - 1
			vec3 unpack_unit(const std::uint32_t packed)
			{
				return { (packed & 0x3FF) / 1023.0f * 2.0f - 1.0f, ((packed >> 10) & 0x3FF) / 1023.0f * 2.0f - 1.0f,
					((packed >> 20) & 0x3FF) / 1023.0f * 2.0f - 1.0f };
			}

			constexpr auto unassigned = std::numeric_limits<unsigned int>::max();

			std::uint64_t edge_key(unsigned int p, unsigned int q)
			{
				if (p > q)
				{
					std::swap(p, q);
				}
				return (static_cast<std::uint64_t>(p) << 32) | q;
			}

			// every triangle of the lightmapped static surfaces, in surface order
			struct triangle_set
			{
				std::vector<unsigned int> surface;
				std::vector<unsigned int> local; // triangle index within its surface
				std::vector<std::array<vec3, 3>> corners;
				std::vector<std::array<unsigned int, 3>> welded; // equal ids for equal positions
				std::vector<vec3> normals;
				std::vector<float> areas;
				std::vector<std::pair<std::uint64_t, unsigned int>> edges; // (welded edge, triangle), sorted

				std::size_t size() const
				{
					return surface.size();
				}

				// every triangle sharing a welded edge with t (t itself included)
				template <typename F>
				void for_neighbours(const unsigned int t, F&& fn) const
				{
					for (auto e = 0; e < 3; e++)
					{
						const auto p = welded[t][e], q = welded[t][(e + 1) % 3];
						if (p == q)
						{
							continue;
						}
						const auto key = edge_key(p, q);
						for (auto it = std::lower_bound(edges.begin(), edges.end(), std::make_pair(key, 0u));
							it != edges.end() && it->first == key; ++it)
						{
							fn(it->second);
						}
					}
				}
			};

			triangle_set collect_triangles(const zonetool::iw7::GfxWorld* world, const zonetool::iw7::GfxWorldTransientZone* zone,
				std::vector<unsigned int>& first_triangle)
			{
				triangle_set set;
				const auto static_count = world->dpvs.staticSurfaceCount;
				first_triangle.assign(world->surfaceCount, unassigned);
				for (auto s = 0u; s < static_count; s++)
				{
					const auto& surface = world->dpvs.surfaces[s];
					if (surface.lightmapIndex == no_lightmap || !surface.tris.triCount)
					{
						continue;
					}
					first_triangle[s] = static_cast<unsigned int>(set.size());
					const auto* indices = world->draw.indices + surface.tris.baseIndex;
					const auto* vertices = zone->vd.vertices + surface.tris.firstVertex;
					for (auto t = 0u; t < surface.tris.triCount; t++)
					{
						set.surface.push_back(s);
						set.local.push_back(t);
						auto& corners = set.corners.emplace_back();
						for (auto k = 0; k < 3; k++)
						{
							const auto* p = vertices[indices[t * 3 + k]].xyz;
							corners[k] = { p[0], p[1], p[2] };
						}
					}
				}

				struct key_hash
				{
					std::size_t operator()(const std::array<std::uint32_t, 3>& k) const
					{
						return (static_cast<std::size_t>(k[0]) * 73856093u) ^ (static_cast<std::size_t>(k[1]) * 19349663u)
							^ (static_cast<std::size_t>(k[2]) * 83492791u);
					}
				};

				const auto count = set.size();
				std::unordered_map<std::array<std::uint32_t, 3>, unsigned int, key_hash> weld;
				weld.reserve(count * 2);
				set.welded.resize(count);
				set.normals.resize(count);
				set.areas.resize(count);
				set.edges.reserve(count * 3);
				for (auto t = 0u; t < count; t++)
				{
					const auto& c = set.corners[t];
					for (auto k = 0; k < 3; k++)
					{
						std::array<std::uint32_t, 3> key;
						for (auto i = 0; i < 3; i++)
						{
							const auto value = c[k][i] == 0.0f ? 0.0f : c[k][i]; // -0 and 0 are one position
							std::memcpy(&key[i], &value, sizeof(value));
						}
						set.welded[t][k] = weld.emplace(key, static_cast<unsigned int>(weld.size())).first->second;
					}
					const auto ab = sub(c[1].data(), c[0].data()), ac = sub(c[2].data(), c[0].data());
					const auto n = cross(ab.data(), ac.data());
					const auto len = length(n.data());
					set.areas[t] = 0.5f * len;
					set.normals[t] = len > 0.0f ? vec3{ n[0] / len, n[1] / len, n[2] / len } : vec3{ 0.0f, 0.0f, 0.0f };
					for (auto e = 0; e < 3; e++)
					{
						const auto p = set.welded[t][e], q = set.welded[t][(e + 1) % 3];
						if (p != q)
						{
							set.edges.emplace_back(edge_key(p, q), t);
						}
					}
				}
				std::sort(set.edges.begin(), set.edges.end());
				return set;
			}

			struct chart
			{
				std::vector<unsigned int> triangles; // triangle_set indices
				vec3 normal{}, axis_u{}, axis_v{};
				float min_u = 0.0f, min_v = 0.0f, max_u = 0.0f, max_v = 0.0f;
				float texel = 1.0f; // world units per texel
				float probe_texel = 1.0f;
				bool part = false; // a placed model part's box projected chart: part_texel_factor coarser, one sun ray a texel
				std::uint32_t width = 0, height = 0; // texels, margins included
				std::uint32_t x = 0, y = 0;
			};

			// projection plane: the seed's normal (no triangle of the chart is more than 60 degrees off it)
			void set_projection(chart& c)
			{
				const auto& n = c.normal;
				const vec3 helper = std::fabs(n[2]) < 0.9f ? vec3{ 0.0f, 0.0f, 1.0f } : vec3{ 1.0f, 0.0f, 0.0f };
				c.axis_u = normalize(cross(helper.data(), n.data()));
				c.axis_v = cross(n.data(), c.axis_u.data());
			}

			struct flat_triangle
			{
				double u[3], v[3];
			};

			flat_triangle flatten(const triangle_set& set, const chart& c, const unsigned int t)
			{
				flat_triangle f{};
				for (auto k = 0; k < 3; k++)
				{
					const auto& p = set.corners[t][k];
					f.u[k] = static_cast<double>(p[0]) * c.axis_u[0] + static_cast<double>(p[1]) * c.axis_u[1]
						+ static_cast<double>(p[2]) * c.axis_u[2];
					f.v[k] = static_cast<double>(p[0]) * c.axis_v[0] + static_cast<double>(p[1]) * c.axis_v[1]
						+ static_cast<double>(p[2]) * c.axis_v[2];
				}
				return f;
			}

			// separating axes: the triangles overlap unless an edge normal of either separates them (to
			// within the tolerance, so triangles sharing an edge or a corner only touch); a degenerate
			// triangle has no interior and overlaps nothing
			bool overlap(const flat_triangle& a, const flat_triangle& b)
			{
				const flat_triangle* both[2] = { &a, &b };
				for (const auto* t : both)
				{
					for (auto e = 0; e < 3; e++)
					{
						const auto du = t->u[(e + 1) % 3] - t->u[e];
						const auto dv = t->v[(e + 1) % 3] - t->v[e];
						const auto len = std::sqrt(du * du + dv * dv);
						if (!(len > 0.0))
						{
							continue;
						}
						const auto nu = -dv / len, nv = du / len;
						auto a_min = std::numeric_limits<double>::max(), a_max = -a_min, b_min = a_min, b_max = -a_min;
						for (auto k = 0; k < 3; k++)
						{
							const auto pa = a.u[k] * nu + a.v[k] * nv;
							const auto pb = b.u[k] * nu + b.v[k] * nv;
							a_min = std::min(a_min, pa);
							a_max = std::max(a_max, pa);
							b_min = std::min(b_min, pb);
							b_max = std::max(b_max, pb);
						}
						if (std::min(a_max, b_max) - std::max(a_min, b_min) <= overlap_tolerance)
						{
							return false;
						}
					}
				}
				return true;
			}

			// one chart's projected triangles, bucketed on a uniform grid
			struct flat_grid
			{
				double cell = 1.0;
				std::vector<flat_triangle> triangles;
				std::unordered_map<std::uint64_t, std::vector<unsigned int>> cells;

				template <typename F>
				void for_cells(const flat_triangle& f, F&& fn) const
				{
					const auto i0 = static_cast<std::int64_t>(std::floor(std::min({ f.u[0], f.u[1], f.u[2] }) / cell));
					const auto i1 = static_cast<std::int64_t>(std::floor(std::max({ f.u[0], f.u[1], f.u[2] }) / cell));
					const auto j0 = static_cast<std::int64_t>(std::floor(std::min({ f.v[0], f.v[1], f.v[2] }) / cell));
					const auto j1 = static_cast<std::int64_t>(std::floor(std::max({ f.v[0], f.v[1], f.v[2] }) / cell));
					for (auto i = i0; i <= i1; i++)
					{
						for (auto j = j0; j <= j1; j++)
						{
							fn((static_cast<std::uint64_t>(static_cast<std::uint32_t>(i)) << 32) | static_cast<std::uint32_t>(j));
						}
					}
				}

				bool overlaps(const flat_triangle& f) const
				{
					auto hit = false;
					for_cells(f, [&](const std::uint64_t key)
					{
						if (hit)
						{
							return;
						}
						const auto found = cells.find(key);
						if (found == cells.end())
						{
							return;
						}
						for (const auto other : found->second)
						{
							if (overlap(f, triangles[other]))
							{
								hit = true;
								return;
							}
						}
					});
					return hit;
				}

				void insert(const flat_triangle& f)
				{
					const auto index = static_cast<unsigned int>(triangles.size());
					triangles.push_back(f);
					for_cells(f, [&](const std::uint64_t key) { cells[key].push_back(index); });
				}
			};

			// Grows a chart from seed over the unassigned triangles connected to it through welded edges
			// (positions shared, so the source mesh's texture seams and material boundaries do not cut
			// charts) that lie within the crease angle of the seed's normal. With a grid, triangles that
			// would overlap the chart's projection so far are left for another chart.
			void grow(const triangle_set& set, const unsigned int seed, const unsigned int id, std::vector<unsigned int>& chart_of,
				chart& c, flat_grid* grid)
			{
				c.normal = set.areas[seed] > 0.0f ? set.normals[seed] : vec3{ 0.0f, 0.0f, 1.0f };
				set_projection(c);
				chart_of[seed] = id;
				if (grid)
				{
					grid->insert(flatten(set, c, seed));
				}
				std::vector<unsigned int> stack{ seed };
				while (!stack.empty())
				{
					const auto t = stack.back();
					stack.pop_back();
					c.triangles.push_back(t);
					set.for_neighbours(t, [&](const unsigned int other)
					{
						if (chart_of[other] != unassigned)
						{
							return;
						}
						// degenerate triangles have no direction and go with any neighbour
						if (set.areas[other] > 0.0f && dot(set.normals[other].data(), c.normal.data()) < crease_cos)
						{
							return;
						}
						if (grid)
						{
							const auto f = flatten(set, c, other);
							if (grid->overlaps(f))
							{
								return;
							}
							grid->insert(f);
						}
						chart_of[other] = id;
						stack.push_back(other);
					});
				}
			}

			// grid cell for a chart's overlap tests: a few triangles per cell on average, and no triangle
			// over more than 9 x 9 cells
			double grid_cell(const triangle_set& set, const chart& c)
			{
				auto area = 0.0;
				auto extent = 0.0;
				for (const auto t : c.triangles)
				{
					const auto f = flatten(set, c, t);
					area += 0.5 * std::fabs((f.u[1] - f.u[0]) * (f.v[2] - f.v[0]) - (f.u[2] - f.u[0]) * (f.v[1] - f.v[0]));
					extent = std::max({ extent, std::max({ f.u[0], f.u[1], f.u[2] }) - std::min({ f.u[0], f.u[1], f.u[2] }),
						std::max({ f.v[0], f.v[1], f.v[2] }) - std::min({ f.v[0], f.v[1], f.v[2] }) });
				}
				return std::max({ 2.0 * std::sqrt(area / static_cast<double>(c.triangles.size())), extent / 8.0, 1e-2 });
			}

			// Charts over all lightmapped triangles, largest triangles seeding first. Charts that fold over
			// themselves in their projection (spiral ramps and the like) would share texels between their
			// layers: their triangles are charted again with the overlap test.
			// `charts` and `chart_of` come with the box projected part charts (bake), which keep their triangles: a part's
			// overlapping projections share texels
			std::vector<chart> build_charts(const triangle_set& set, std::vector<chart> charts, std::vector<unsigned int>& chart_of,
				unsigned int& folded_count)
			{
				const auto count = static_cast<unsigned int>(set.size());
				std::vector<unsigned int> order(count);
				std::iota(order.begin(), order.end(), 0u);
				std::stable_sort(order.begin(), order.end(), [&](const unsigned int a, const unsigned int b) { return set.areas[a] > set.areas[b]; });

				const auto fixed_count = charts.size();
				for (const auto seed : order)
				{
					if (chart_of[seed] == unassigned)
					{
						const auto id = static_cast<unsigned int>(charts.size());
						grow(set, seed, id, chart_of, charts.emplace_back(), nullptr);
					}
				}

				// two triangles sharing an edge can only both face the chart's normal side by side, so a
				// fold needs three or more
				std::vector<double> folded_cell(charts.size(), 0.0);
				parallel_for(static_cast<std::uint32_t>(charts.size()), [&](const std::uint32_t i, std::uint32_t)
				{
					const auto& c = charts[i];
					if (c.triangles.size() < 3 || i < fixed_count)
					{
						return;
					}
					flat_grid grid;
					grid.cell = grid_cell(set, c);
					for (const auto t : c.triangles)
					{
						const auto f = flatten(set, c, t);
						if (grid.overlaps(f))
						{
							folded_cell[i] = grid.cell;
							return;
						}
						grid.insert(f);
					}
				});

				std::vector<std::pair<unsigned int, double>> again; // triangle, grid cell of its folded chart
				folded_count = 0;
				for (auto i = 0u; i < charts.size(); i++)
				{
					if (folded_cell[i] > 0.0)
					{
						folded_count++;
						for (const auto t : charts[i].triangles)
						{
							chart_of[t] = unassigned;
							again.emplace_back(t, folded_cell[i]);
						}
						charts[i].triangles.clear();
					}
				}
				if (again.empty())
				{
					return charts;
				}

				std::stable_sort(again.begin(), again.end(), [&](const auto& a, const auto& b) { return set.areas[a.first] > set.areas[b.first]; });
				for (const auto& [seed, cell] : again)
				{
					if (chart_of[seed] == unassigned)
					{
						flat_grid grid;
						grid.cell = cell;
						const auto id = static_cast<unsigned int>(charts.size());
						grow(set, seed, id, chart_of, charts.emplace_back(), &grid);
					}
				}

				// drop the emptied charts
				std::vector<chart> kept;
				std::vector<unsigned int> renumber(charts.size(), unassigned);
				for (auto i = 0u; i < charts.size(); i++)
				{
					if (!charts[i].triangles.empty())
					{
						renumber[i] = static_cast<unsigned int>(kept.size());
						kept.push_back(std::move(charts[i]));
					}
				}
				for (auto& id : chart_of)
				{
					id = renumber[id];
				}
				return kept;
			}

			void chart_bounds(const triangle_set& set, chart& c)
			{
				c.min_u = c.min_v = std::numeric_limits<float>::max();
				c.max_u = c.max_v = -std::numeric_limits<float>::max();
				for (const auto t : c.triangles)
				{
					for (const auto& p : set.corners[t])
					{
						const auto u = dot(p.data(), c.axis_u.data());
						const auto v = dot(p.data(), c.axis_v.data());
						c.min_u = std::min(c.min_u, u);
						c.max_u = std::max(c.max_u, u);
						c.min_v = std::min(c.min_v, v);
						c.max_v = std::max(c.max_v, v);
					}
				}
			}

			// scale: lightmap texel size in probe volume texels
			void size_chart(chart& c, const float scale)
			{
				c.texel = std::max(min_texel_size, c.probe_texel * scale * (c.part ? part_texel_factor : 1.0f));
				const auto extent = std::max(c.max_u - c.min_u, c.max_v - c.min_v);
				// a chart may not outgrow max_chart_texels: its texels get coarser instead
				if (extent / c.texel > static_cast<float>(max_chart_texels))
				{
					c.texel = extent / static_cast<float>(max_chart_texels);
				}
				const auto span = [&](const float lo, const float hi)
				{
					return static_cast<std::uint32_t>(std::ceil((hi - lo) / c.texel)) + 1 + 2 * chart_margin;
				};
				c.width = span(c.min_u, c.max_u);
				c.height = span(c.min_v, c.max_v);
			}

			// bottom-left skyline, tallest charts first; false when the charts do not fit max_height
			bool pack(std::vector<chart>& charts, std::uint32_t& used_height)
			{
				std::vector<unsigned int> order(charts.size());
				std::iota(order.begin(), order.end(), 0u);
				std::stable_sort(order.begin(), order.end(), [&](const unsigned int a, const unsigned int b)
				{
					return charts[a].height != charts[b].height ? charts[a].height > charts[b].height : charts[a].width > charts[b].width;
				});

				std::vector<std::uint32_t> skyline(atlas_width, 0);
				std::vector<std::uint32_t> window(atlas_width); // columns of a sliding window maximum
				used_height = 0;
				for (const auto i : order)
				{
					auto& c = charts[i];
					const auto w = c.width;
					if (w > atlas_width)
					{
						return false;
					}
					auto best_x = 0u;
					auto best_y = std::numeric_limits<std::uint32_t>::max();
					std::size_t head = 0, tail = 0;
					for (auto x = 0u; x < atlas_width; x++)
					{
						while (tail > head && skyline[window[tail - 1]] <= skyline[x])
						{
							tail--;
						}
						window[tail++] = x;
						if (window[head] + w <= x)
						{
							head++;
						}
						if (x + 1 >= w)
						{
							const auto y = skyline[window[head]];
							if (y < best_y)
							{
								best_y = y;
								best_x = x + 1 - w;
							}
						}
					}
					if (best_y + c.height > atlas_max_height)
					{
						return false;
					}
					c.x = best_x;
					c.y = best_y;
					for (auto x = best_x; x < best_x + w; x++)
					{
						skyline[x] = best_y + c.height;
					}
					used_height = std::max(used_height, best_y + c.height);
				}
				return true;
			}

			// where a world position lands in the atlas (texel units, texel centres at + 0.5)
			void atlas_position(const chart& c, const float* p, float& x, float& y)
			{
				x = static_cast<float>(c.x + chart_margin) + (dot(p, c.axis_u.data()) - c.min_u) / c.texel + 0.5f;
				y = static_cast<float>(c.y + chart_margin) + (dot(p, c.axis_v.data()) - c.min_v) / c.texel + 0.5f;
			}

			// the point of triangle (px, py) closest to (x, y): squared distance, barycentrics
			float closest_point(const float px[3], const float py[3], const float x, const float y, float bary[3])
			{
				const auto area2 = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]);
				if (area2 != 0.0f)
				{
					float w[3];
					for (auto k = 0; k < 3; k++)
					{
						const auto a = (k + 1) % 3, b = (k + 2) % 3;
						w[k] = ((px[b] - px[a]) * (y - py[a]) - (py[b] - py[a]) * (x - px[a])) / area2;
					}
					if (w[0] >= 0.0f && w[1] >= 0.0f && w[2] >= 0.0f)
					{
						std::memcpy(bary, w, sizeof(w));
						return 0.0f;
					}
				}
				auto best = std::numeric_limits<float>::max();
				for (auto i = 0; i < 3; i++)
				{
					const auto j = (i + 1) % 3;
					const auto edge_x = px[j] - px[i], edge_y = py[j] - py[i];
					const auto len2 = edge_x * edge_x + edge_y * edge_y;
					const auto t = len2 > 0.0f ? std::clamp(((x - px[i]) * edge_x + (y - py[i]) * edge_y) / len2, 0.0f, 1.0f) : 0.0f;
					const auto off_x = px[i] + t * edge_x - x, off_y = py[i] + t * edge_y - y;
					const auto d2 = off_x * off_x + off_y * off_y;
					if (d2 < best)
					{
						best = d2;
						bary[0] = bary[1] = bary[2] = 0.0f;
						bary[i] = 1.0f - t;
						bary[j] = t;
					}
				}
				return best;
			}

			// one texel to bake: the lighting at a point of a triangle
			struct sample
			{
				std::uint32_t texel; // y * width + x
				unsigned int triangle; // triangle_set index
				float bary[2]; // weights of corners 1 and 2
			};

			struct surface_point
			{
				vec3 position;
				vec3 normal;
				vec3 tangent;
				float binormal_sign;
			};

			struct fitted
			{
				float ambient[3];
				float directional[3];
				std::uint8_t direction[2]; // octahedral, tangent space
			};

			// the fixed set of normals around n the fit uses: n itself and two rings
			void fit_directions(const vec3& n, const vec3& t, std::array<vec3, fit_normals>& out)
			{
				const auto u = normalize(cross(n.data(), (std::fabs(dot(t.data(), n.data())) < 0.99f ? t : vec3{ n[1], n[2], n[0] }).data()));
				const auto v = cross(n.data(), u.data());
				out[0] = n;
				for (auto ring = 0u; ring < 2; ring++)
				{
					const auto theta = fit_cap_degrees[ring] * 3.14159265f / 180.0f;
					for (auto k = 0u; k < fit_ring; k++)
					{
						const auto phi = (static_cast<float>(k) + (ring ? 0.5f : 0.0f)) * 2.0f * 3.14159265f / static_cast<float>(fit_ring);
						const auto st = std::sin(theta), ct = std::cos(theta);
						out[1 + ring * fit_ring + k] = normalize({ n[0] * ct + (u[0] * std::cos(phi) + v[0] * std::sin(phi)) * st,
							n[1] * ct + (u[1] * std::cos(phi) + v[1] * std::sin(phi)) * st,
							n[2] * ct + (u[2] * std::cos(phi) + v[2] * std::sin(phi)) * st });
					}
				}
			}

			float luminance(const float* c)
			{
				return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
			}

			// IW7's lightmap term, ambient + directional * saturate(dot(n, dir)), fitted to BO3's
			// probe lighting over the normals around the surface normal
			fitted fit(const surface_point& s, const std::array<vec3, fit_normals>& normals, const std::array<vec3, fit_normals>& lighting)
			{
				// direction: the gradient of a linear fit of the luminance over the normals
				double ata[4][4]{}, atb[4]{};
				for (auto i = 0u; i < fit_normals; i++)
				{
					const double row[4] = { 1.0, normals[i][0], normals[i][1], normals[i][2] };
					const auto l = luminance(lighting[i].data());
					for (auto r = 0; r < 4; r++)
					{
						for (auto c = 0; c < 4; c++)
						{
							ata[r][c] += row[r] * row[c];
						}
						atb[r] += row[r] * l;
					}
				}
				// Gaussian elimination, 4x4
				double m[4][5];
				for (auto r = 0; r < 4; r++)
				{
					for (auto c = 0; c < 4; c++)
					{
						m[r][c] = ata[r][c];
					}
					m[r][4] = atb[r];
				}
				auto solvable = true;
				for (auto c = 0; c < 4 && solvable; c++)
				{
					auto pivot = c;
					for (auto r = c + 1; r < 4; r++)
					{
						if (std::fabs(m[r][c]) > std::fabs(m[pivot][c]))
						{
							pivot = r;
						}
					}
					if (std::fabs(m[pivot][c]) < 1e-12)
					{
						solvable = false;
						break;
					}
					std::swap(m[c], m[pivot]);
					for (auto r = 0; r < 4; r++)
					{
						if (r != c)
						{
							const auto f = m[r][c] / m[c][c];
							for (auto k = c; k < 5; k++)
							{
								m[r][k] -= f * m[c][k];
							}
						}
					}
				}
				vec3 dir = s.normal;
				if (solvable)
				{
					const vec3 g{ static_cast<float>(m[1][4] / m[1][1]), static_cast<float>(m[2][4] / m[2][2]),
						static_cast<float>(m[3][4] / m[3][3]) };
					if (length(g.data()) > 1e-6f)
					{
						dir = normalize(g);
					}
				}
				// keep it on the surface's side, where IW7's tangent-space direction can point
				if (dot(dir.data(), s.normal.data()) < 0.0f)
				{
					const auto d = dot(dir.data(), s.normal.data());
					dir = normalize({ dir[0] - d * s.normal[0], dir[1] - d * s.normal[1], dir[2] - d * s.normal[2] });
					if (length(dir.data()) < 0.5f)
					{
						dir = s.normal;
					}
				}

				fitted out{};
				std::array<float, fit_normals> term{};
				for (auto i = 0u; i < fit_normals; i++)
				{
					term[i] = std::clamp(dot(normals[i].data(), dir.data()), 0.0f, 1.0f);
				}

				// the ambient / directional split, fitted to the luminance (a, b >= 0). Every channel
				// keeps that shape, scaled to its own data: over normals this close together the constant
				// and the cosine term are nearly collinear, so separate per-channel fits split each channel
				// arbitrarily, and IW7 applies SSAO to the directional part alone (hue shifts in crevices)
				double s1 = 0, st = 0, stt = 0, sy = 0, sty = 0;
				for (auto i = 0u; i < fit_normals; i++)
				{
					const auto l = static_cast<double>(luminance(lighting[i].data()));
					s1 += 1.0;
					st += term[i];
					stt += term[i] * term[i];
					sy += l;
					sty += term[i] * l;
				}
				const auto split_det = s1 * stt - st * st;
				double split_a = sy / s1, split_b = 0.0;
				if (std::fabs(split_det) > 1e-9)
				{
					split_a = (stt * sy - st * sty) / split_det;
					split_b = (s1 * sty - st * sy) / split_det;
				}
				if (split_b < 0.0)
				{
					split_b = 0.0;
					split_a = sy / s1;
				}
				if (split_a < 0.0)
				{
					split_a = 0.0;
					split_b = stt > 0.0 ? std::max(0.0, sty / stt) : 0.0;
				}
				double shape_norm = 0.0;
				for (auto i = 0u; i < fit_normals; i++)
				{
					const auto f = split_a + split_b * term[i];
					shape_norm += f * f;
				}
				for (auto c = 0; c < 3; c++)
				{
					auto k = 0.0;
					if (shape_norm > 0.0)
					{
						for (auto i = 0u; i < fit_normals; i++)
						{
							k += (split_a + split_b * term[i]) * lighting[i][c];
						}
						k /= shape_norm;
					}
					out.ambient[c] = static_cast<float>(k * split_a);
					out.directional[c] = static_cast<float>(k * split_b);
				}

				// tangent-space direction, solving the shader's T x + B y + N z = dir
				const auto& n = s.normal;
				const auto& t = s.tangent;
				const auto bn = cross(n.data(), t.data());
				const vec3 b{ bn[0] * s.binormal_sign, bn[1] * s.binormal_sign, bn[2] * s.binormal_sign };
				const float mtx[3][3] = { { t[0], b[0], n[0] }, { t[1], b[1], n[1] }, { t[2], b[2], n[2] } };
				const auto det = mtx[0][0] * (mtx[1][1] * mtx[2][2] - mtx[1][2] * mtx[2][1])
					- mtx[0][1] * (mtx[1][0] * mtx[2][2] - mtx[1][2] * mtx[2][0])
					+ mtx[0][2] * (mtx[1][0] * mtx[2][1] - mtx[1][1] * mtx[2][0]);
				vec3 ts{ 0.0f, 0.0f, 1.0f };
				if (std::fabs(det) > 1e-8f)
				{
					// Cramer's rule
					const auto solve = [&](const int column)
					{
						float m2[3][3];
						std::memcpy(m2, mtx, sizeof(m2));
						for (auto r = 0; r < 3; r++)
						{
							m2[r][column] = dir[r];
						}
						return (m2[0][0] * (m2[1][1] * m2[2][2] - m2[1][2] * m2[2][1])
							- m2[0][1] * (m2[1][0] * m2[2][2] - m2[1][2] * m2[2][0])
							+ m2[0][2] * (m2[1][0] * m2[2][1] - m2[1][1] * m2[2][0])) / det;
					};
					ts = { solve(0), solve(1), solve(2) };
				}
				ts[2] = std::max(ts[2], 0.0f);
				auto sum = std::fabs(ts[0]) + std::fabs(ts[1]) + ts[2];
				if (sum <= 0.0f)
				{
					ts = { 0.0f, 0.0f, 1.0f };
					sum = 1.0f;
				}
				const auto px = ts[0] / sum, py = ts[1] / sum;
				const auto to_byte = [](const float v)
				{
					return static_cast<std::uint8_t>(std::clamp(std::lround((v * 0.5f + 0.5f) * 255.0f), 0L, 255L));
				};
				out.direction[0] = to_byte(px + py);
				out.direction[1] = to_byte(px - py);
				return out;
			}

			std::vector<std::uint8_t> compress(const DirectX::Image& src, const DXGI_FORMAT format)
			{
				DirectX::ScratchImage out;
				HRESULT hr;
				if ((format == DXGI_FORMAT_BC6H_UF16 || format == DXGI_FORMAT_BC7_UNORM) && gpu_device())
				{
					std::lock_guard _(gpu_mutex);
					hr = DirectX::Compress(gpu_device(), src, format, DirectX::TEX_COMPRESS_DEFAULT, 1.0f, out);
				}
				else
				{
					hr = DirectX::Compress(src, format, DirectX::TEX_COMPRESS_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, out);
				}
				if (FAILED(hr))
				{
					throw std::runtime_error(utils::string::va("lightmap block compression failed (0x%08X)", static_cast<unsigned int>(hr)));
				}
				return { out.GetPixels(), out.GetPixels() + out.GetPixelsSize() };
			}

			void write_lightmap_image(const std::string& name, const std::uint32_t width, const std::uint32_t height,
				const DXGI_FORMAT format, std::vector<std::uint8_t>& pixels)
			{
				zonetool::iw7::GfxImage image{};
				image.imageFormat = format;
				image.flags = 0x3B;
				image.mapType = zonetool::iw7::MAPTYPE_2D;
				image.semantic = zonetool::iw7::TS_FUNCTION;
				image.category = zonetool::iw7::IMG_CATEGORY_LIGHTMAP;
				image.dataLen1 = static_cast<unsigned int>(pixels.size());
				image.dataLen2 = image.dataLen1;
				image.width = static_cast<unsigned short>(width);
				image.height = static_cast<unsigned short>(height);
				image.depth = 1;
				image.numElements = 1;
				image.levelCount = 1;
				image.streamed = 0;
				image.pixelData = pixels.data();
				image.name = name.data();
				zonetool::iw7::gfx_image::dump(&image);
			}

			// Gives every texel of a chart's rectangle that was not baked the values of the nearest baked
			// texel (breadth first from the baked ones), so the chart's border and the gaps between its
			// triangles hold its own lighting. Charts are disjoint rectangles.
			void dilate(std::vector<float>& ambient, std::vector<float>& directional, std::vector<std::uint8_t>& direction,
				std::vector<std::uint8_t>& sun, std::vector<std::uint8_t>& known, const std::uint32_t width, const chart& c)
			{
				std::vector<std::uint32_t> queue;
				for (auto y = c.y; y < c.y + c.height; y++)
				{
					for (auto x = c.x; x < c.x + c.width; x++)
					{
						if (known[static_cast<std::size_t>(y) * width + x])
						{
							queue.push_back(y * width + x);
						}
					}
				}
				for (std::size_t head = 0; head < queue.size(); head++)
				{
					const auto from = queue[head];
					const auto fx = from % width, fy = from / width;
					const std::int32_t offsets[4][2] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
					for (const auto& o : offsets)
					{
						const auto nx = static_cast<std::int32_t>(fx) + o[0], ny = static_cast<std::int32_t>(fy) + o[1];
						if (nx < static_cast<std::int32_t>(c.x) || ny < static_cast<std::int32_t>(c.y)
							|| nx >= static_cast<std::int32_t>(c.x + c.width) || ny >= static_cast<std::int32_t>(c.y + c.height))
						{
							continue;
						}
						const auto to = static_cast<std::uint32_t>(ny) * width + static_cast<std::uint32_t>(nx);
						if (known[to])
						{
							continue;
						}
						known[to] = 1;
						for (auto k = 0; k < 3; k++)
						{
							ambient[static_cast<std::size_t>(to) * 3 + k] = ambient[static_cast<std::size_t>(from) * 3 + k];
							directional[static_cast<std::size_t>(to) * 3 + k] = directional[static_cast<std::size_t>(from) * 3 + k];
						}
						direction[static_cast<std::size_t>(to) * 2 + 0] = direction[static_cast<std::size_t>(from) * 2 + 0];
						direction[static_cast<std::size_t>(to) * 2 + 1] = direction[static_cast<std::size_t>(from) * 2 + 1];
						// the sun's distance too: a texel left at its default (lit) lets the bilinear lookup blend light into
						// the edge of a shadowed chart
						sun[to] = sun[from];
						queue.push_back(to);
					}
				}
			}
		}

		struct sun_blockers::impl
		{
			struct sun_mask
			{
				material_texture::texture texture;
				std::uint32_t channel;
				world_material::bake::sample_map map;
			};
			std::optional<triangle_bvh> opaque, alpha;
			std::vector<std::uint32_t> opaque_owner, alpha_owner, alpha_mask_index;
			std::vector<float> alpha_uvs;
			std::vector<sun_mask> alpha_masks;

			// an alpha tested triangle blocks where its mask passes IW7's test (alpha 0.5: every BO3 alpha-test template
			// discards below it, make_plan), read where its shader reads it
			bool alpha_blocks(const std::uint32_t triangle, const double u, const double v) const
			{
				const auto& mask = alpha_masks[alpha_mask_index[triangle]];
				const auto* uv = &alpha_uvs[static_cast<std::size_t>(triangle) * 6];
				const auto w = 1.0 - u - v;
				const auto s = static_cast<float>(uv[0] * w + uv[2] * u + uv[4] * v);
				const auto t = static_cast<float>(uv[1] * w + uv[3] * u + uv[5] * v);
				const auto& m = mask.map.m;
				float texel[4];
				mask.texture.sample(m[0] * s + m[1] * t + m[2], m[3] * s + m[4] * t + m[5], 0.0f, mask.map.offset[0], mask.map.offset[1], texel);
				return texel[mask.channel] >= 0.5f;
			}
		};

		// The sun's blockers for both baked visibilities: the lightmap's primary image and the light grid's coefficient 27.
		// IW7 shadows its sun in real time only near the camera (sun shadow cascades) and past them takes these, so a gap
		// lets the sun light everything out of the cascades, indoors too.
		sun_blockers::sun_blockers(const GfxWorld* asset, const zonetool::iw7::GfxWorld* world,
			const zonetool::iw7::GfxWorldTransientZone* zone, const std::vector<std::uint8_t>& occluders)
			: impl_(std::make_unique<impl>())
		{
			this->build(asset, world, zone, &occluders);
		}

		sun_blockers::sun_blockers(const GfxWorld* asset)
			: impl_(std::make_unique<impl>())
		{
			this->build(asset, nullptr, nullptr, nullptr);
		}

		void sun_blockers::build(const GfxWorld* asset, const zonetool::iw7::GfxWorld* world,
			const zonetool::iw7::GfxWorldTransientZone* zone, const std::vector<std::uint8_t>* occluders)
		{
			std::vector<float> sun_triangles;
			// the alpha tested ones: per triangle its three uv0 and its mask
			std::vector<float> alpha_triangles, alpha_uvs;
			std::vector<std::uint32_t> alpha_mask_index;
			std::vector<impl::sun_mask> alpha_masks;
			// per triangle, the IW7 static model drawing it (no_owner: a world surface)
			std::vector<std::uint32_t> sun_owner, alpha_owner;
			constexpr auto no_owner = ~0u;
			const auto& clusters = static_model_clusters::current();
			std::unordered_map<const Material*, std::uint32_t> mask_of_material;
			auto alpha_model_triangles = 0u;
			if (!world)
			{
				// BO3's static surfaces: those that cast (flag 1, as the IW7 surfaces' sun flag) with an opaque caster material
				for (auto s = 0u; s < asset->dpvs.staticSurfaceCount; s++)
				{
					const auto& surface = asset->dpvs.surfaces[s];
					const auto* material = surface.material;
					if (!(surface.flags & 1) || !world_material::model_blocks_sun(material) || world_material::bake::alpha_mask_of(material))
					{
						continue;
					}
					const auto name = world_material::bake::bo3_template(material);
					if (name.find("alpha") != std::string::npos || name.find("atest") != std::string::npos || name.find("foliage") != std::string::npos)
					{
						continue;
					}
					const auto first = static_cast<unsigned int>(surface.tris.vertexDataOffset0) / 12u;
					const auto* indices = asset->draw.indices + surface.tris.baseIndex;
					for (auto t = 0u; t < surface.tris.triCount * 3u; t++)
					{
						const auto* xyz = reinterpret_cast<const float*>(asset->draw.vd0.data + (first + indices[t]) * 12u);
						sun_triangles.insert(sun_triangles.end(), xyz, xyz + 3);
					}
					// a BO3 static surface: 0x80000000 | its index (no static model's)
					sun_owner.insert(sun_owner.end(), surface.tris.triCount, 0x80000000u | s);
				}
			}
			{
				const auto sky = world ? world_sky::surfaces(world) : std::vector<std::uint8_t>{};
				for (auto s = 0u; world && s < world->dpvs.staticSurfaceCount; s++)
				{
					if (sky[s] || !(*occluders)[s])
					{
						continue;
					}
					const auto& surface = world->dpvs.surfaces[s];
					for (auto t = 0u; t < surface.tris.triCount * 3u; t++)
					{
						const auto index = world->draw.indices[surface.tris.baseIndex + t];
						const auto* xyz = zone->vd.vertices[surface.tris.firstVertex + index].xyz;
						sun_triangles.insert(sun_triangles.end(), xyz, xyz + 3);
					}
					sun_owner.insert(sun_owner.end(), surface.tris.triCount, no_owner);
				}
				// and the static models that cast (the real-time shadow draws them too), first LOD of the rigid ones, placed.
				// Alpha tested surfaces (leaves, bars, fences) block it where their alpha passes IW7's test, as the real-time
				// shadow draws them: a separate set with their texture coordinates, read through the texture channel their
				// shader discards by (bake::alpha_mask_of)
				auto model_triangles = 0u;
				// what is left out of the sun's blockers and why, the largest first (a left-out ceiling lets the sun in)
				std::map<std::string, std::pair<float, std::uint32_t>> left_out;
				const auto leave_out = [&](const char* why, const XModel* model, const float scale, const Material* material)
				{
					auto key = utils::string::va("%s: %s%s%s", why, model->name, material ? " / " : "", material && material->name ? material->name : "");
					auto& [size, count] = left_out[key];
					size = std::max(size, model->radius * scale);
					count++;
				};
				for (auto i = 0u; i < asset->dpvs.smodelCount; i++)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					const auto owner = !world ? i : i < clusters.iw7_index.size() ? clusters.iw7_index[i] : no_owner;
					if (!src.model)
					{
						continue;
					}
					if (src.flags & 0x1)
					{
						leave_out("BO3 no shadow (flag 0x1)", src.model, src.placement.scale, nullptr);
						continue;
					}
					const auto meshes = xmodel::lod_meshes(src.model);
					if (meshes.empty())
					{
						continue;
					}
					const auto& [mesh, materials] = meshes.front();
					constexpr float no_offset[3] = { 0.0f, 0.0f, 0.0f };
					for (auto s = 0u; s < materials.size(); s++)
					{
						if (!world_material::model_blocks_sun(materials[s]))
						{
							leave_out("material blocks no sun", src.model, src.placement.scale, materials[s]);
							continue;
						}
						const auto mask = world_material::bake::alpha_mask_of(materials[s]);
						if (!mask)
						{
							// an alpha test whose coverage is not one texture channel: left out
							const auto name = world_material::bake::bo3_template(materials[s]);
							if (name.find("alpha") != std::string::npos || name.find("atest") != std::string::npos
								|| name.find("foliage") != std::string::npos)
							{
								leave_out("alpha test without a one-channel mask", src.model, src.placement.scale, materials[s]);
								continue;
							}
						}
						std::vector<zonetool::iw7::GfxPackedVertex> verts;
						std::vector<zonetool::iw7::Face> faces;
						if (!xmodel_mesh::append_placed(mesh, s, {}, src.placement.origin, src.placement.axis, src.placement.scale,
							no_offset, verts, faces))
						{
							continue;
						}
						std::uint32_t mask_index = 0;
						if (mask)
						{
							const auto found = mask_of_material.find(materials[s]);
							if (found != mask_of_material.end())
							{
								mask_index = found->second;
							}
							else
							{
								mask_index = static_cast<std::uint32_t>(alpha_masks.size());
								alpha_masks.push_back({ material_texture::texture(mask->image, mask->u, mask->v), mask->channel, mask->map });
								mask_of_material.emplace(materials[s], mask_index);
							}
						}
						for (const auto& f : faces)
						{
							for (const auto v : { f.v1, f.v2, f.v3 })
							{
								auto& out = mask ? alpha_triangles : sun_triangles;
								out.insert(out.end(), verts[v].xyz, verts[v].xyz + 3);
								if (mask)
								{
									// append_placed with no transform keeps BO3's half precision uv0
									alpha_uvs.push_back(DirectX::PackedVector::XMConvertHalfToFloat(static_cast<std::uint16_t>(verts[v].texCoord.packed)));
									alpha_uvs.push_back(DirectX::PackedVector::XMConvertHalfToFloat(static_cast<std::uint16_t>(verts[v].texCoord.packed >> 16)));
								}
							}
							(mask ? alpha_owner : sun_owner).push_back(owner);
							if (mask)
							{
								alpha_mask_index.push_back(mask_index);
							}
						}
						(mask ? alpha_model_triangles : model_triangles) += static_cast<unsigned int>(faces.size());
					}
				}
				std::vector<std::pair<std::string, std::pair<float, std::uint32_t>>> sorted(left_out.begin(), left_out.end());
				std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
				for (const auto& [key, value] : sorted)
				{
					ZONETOOL_INFO("lightmap: sun blockers leave out %s (%u placements, radius up to %.0f)", key.data(), value.second, value.first);
				}
				ZONETOOL_INFO("lightmap: sun visibility against %zu triangles (%u of static models) and %u alpha tested ones of %zu materials",
					sun_triangles.size() / 9, model_triangles, alpha_model_triangles, alpha_masks.size());
			}
			impl_->opaque.emplace(std::move(sun_triangles));
			impl_->alpha.emplace(std::move(alpha_triangles));
			impl_->opaque_owner = std::move(sun_owner);
			impl_->alpha_owner = std::move(alpha_owner);
			impl_->alpha_mask_index = std::move(alpha_mask_index);
			impl_->alpha_uvs = std::move(alpha_uvs);
			impl_->alpha_masks = std::move(alpha_masks);
		}

		sun_blockers::~sun_blockers() = default;

		bool sun_blockers::blocked(const double from[3], const double to[3], const std::uint32_t* ignore_model) const
		{
			const auto& d = *impl_;
			const auto counts = [&](const std::vector<std::uint32_t>& owner, const std::uint32_t triangle)
			{
				return !ignore_model || owner[triangle] != *ignore_model;
			};
			if (d.opaque->blocked(from, to, 1e-7, 1.0, [&](const std::uint32_t triangle, double, double)
			{
				return counts(d.opaque_owner, triangle);
			}))
			{
				return true;
			}
			return d.alpha->blocked(from, to, 1e-7, 1.0, [&](const std::uint32_t triangle, const double u, const double v)
			{
				return counts(d.alpha_owner, triangle) && d.alpha_blocks(triangle, u, v);
			});
		}

		std::pair<int, std::uint32_t> sun_blockers::blocker(const double from[3], const double to[3], const std::uint32_t* ignore_model) const
		{
			const auto& d = *impl_;
			std::pair<int, std::uint32_t> found{ 0, ~0u };
			if (d.opaque->blocked(from, to, 1e-7, 1.0, [&](const std::uint32_t triangle, double, double)
			{
				if (ignore_model && d.opaque_owner[triangle] == *ignore_model)
				{
					return false;
				}
				found = { 1, d.opaque_owner[triangle] };
				return true;
			}))
			{
				return found;
			}
			if (d.alpha->blocked(from, to, 1e-7, 1.0, [&](const std::uint32_t triangle, const double u, const double v)
			{
				if ((ignore_model && d.alpha_owner[triangle] == *ignore_model) || !d.alpha_blocks(triangle, u, v))
				{
					return false;
				}
				found = { 2, d.alpha_owner[triangle] };
				return true;
			}))
			{
				return found;
			}
			return { 0, ~0u };
		}

		float sun_fraction(const sun_blockers& blockers, const float sun_dir[3], const float p[3], const std::uint32_t* ignore_model)
		{
			double d[3] = { sun_dir[0], sun_dir[1], sun_dir[2] };
			const auto len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
			if (len <= 0.0)
			{
				return 1.0f;
			}
			for (auto& c : d)
			{
				c /= len;
			}
			// two axes across the sun direction
			double u[3] = { 0.0, 0.0, 0.0 };
			u[std::fabs(d[2]) < 0.9 ? 2 : 0] = 1.0;
			double t1[3] = { d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0] };
			const auto l1 = std::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
			for (auto& c : t1)
			{
				c /= l1;
			}
			const double t2[3] = { d[1] * t1[2] - d[2] * t1[1], d[2] * t1[0] - d[0] * t1[2], d[0] * t1[1] - d[1] * t1[0] };
			constexpr double spread = 0.00872665; // tan(0.5 degrees)
			constexpr double offsets[5][2] = { { 0, 0 }, { spread, 0 }, { -spread, 0 }, { 0, spread }, { 0, -spread } };
			constexpr double reach = 131072.0;
			auto open = 0;
			for (const auto& o : offsets)
			{
				const double from[3] = { p[0] + d[0], p[1] + d[1], p[2] + d[2] };
				double to[3];
				for (auto k = 0; k < 3; k++)
				{
					to[k] = from[k] + (d[k] + t1[k] * o[0] + t2[k] * o[1]) * reach;
				}
				if (!blockers.blocked(from, to, ignore_model))
				{
					open++;
				}
			}
			return static_cast<float>(open) / 5.0f;
		}

		void bake(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, zonetool::iw7::GfxWorldTransientZone* zone,
			const sun_blockers& blockers, const std::unordered_map<unsigned int, std::vector<unsigned int>>& pieces,
			utils::memory::allocator& allocator)
		{
			const auto start = std::chrono::steady_clock::now();
			probe_lighting::evaluator lighting(asset, map::lighting_state());

			const auto* com = comworld::converted();
			const auto& sun = com->primaryLights[world->lastSunPrimaryLightIndex];
			const auto sun_lit = sun.color[0] + sun.color[1] + sun.color[2] > 0.0f;
			double sun_dir[3] = { sun.dir[0], sun.dir[1], sun.dir[2] };
			{
				const auto len = std::sqrt(sun_dir[0] * sun_dir[0] + sun_dir[1] * sun_dir[1] + sun_dir[2] * sun_dir[2]);
				for (auto& c : sun_dir)
				{
					c = len > 0.0 ? c / len : 0.0;
				}
			}
			const auto sun_on = sun_lit && (sun_dir[0] != 0.0 || sun_dir[1] != 0.0 || sun_dir[2] != 0.0);
			// whether the sun reaches p: one ray toward it, off the surface along its normal so it does not shadow itself
			const auto sun_open = [&](const double* p, const float* n) -> bool
			{
				constexpr double reach = 131072.0;
				const double from[3] = { p[0] + n[0] + sun_dir[0], p[1] + n[1] + sun_dir[1], p[2] + n[2] + sun_dir[2] };
				const double to[3] = { from[0] + sun_dir[0] * reach, from[1] + sun_dir[1] * reach, from[2] + sun_dir[2] * reach };
				return !blockers.blocked(from, to);
			};


			const auto surface_count = world->surfaceCount;
			const auto static_count = world->dpvs.staticSurfaceCount;

			// ---- charts --------------------------------------------------------------------------
			std::vector<unsigned int> first_triangle; // per surface, its first triangle_set index
			const auto set = collect_triangles(world, zone, first_triangle);
			// the placed parts' charts: per part, one a box axis (+x -x +y -y +z -z) its triangles' normals lean to
			std::vector<unsigned int> chart_of(set.size(), unassigned);
			std::vector<chart> part_charts;
			for (const auto& [s, starts] : pieces)
			{
				if (first_triangle[s] == unassigned)
				{
					continue;
				}
				const auto triangles = world->dpvs.surfaces[s].tris.triCount;
				for (auto p = 0u; p < starts.size(); p++)
				{
					const auto end = p + 1 < starts.size() ? starts[p + 1] : static_cast<unsigned int>(triangles);
					unsigned int axis_chart[6];
					std::fill(std::begin(axis_chart), std::end(axis_chart), unassigned);
					for (auto local = starts[p]; local < end; local++)
					{
						const auto t = first_triangle[s] + local;
						const auto& n = set.normals[t];
						auto k = 2;
						if (std::fabs(n[0]) >= std::fabs(n[1]) && std::fabs(n[0]) >= std::fabs(n[2]))
						{
							k = 0;
						}
						else if (std::fabs(n[1]) >= std::fabs(n[2]))
						{
							k = 1;
						}
						const auto a = k * 2 + (n[k] < 0.0f ? 1 : 0);
						if (axis_chart[a] == unassigned)
						{
							axis_chart[a] = static_cast<unsigned int>(part_charts.size());
							auto& c = part_charts.emplace_back();
							c.part = true;
							c.normal = { 0.0f, 0.0f, 0.0f };
							c.normal[k] = (a & 1) ? -1.0f : 1.0f;
							set_projection(c);
						}
						part_charts[axis_chart[a]].triangles.push_back(t);
						chart_of[t] = axis_chart[a];
					}
				}
			}
			const auto part_chart_count = part_charts.size();
			auto folded = 0u;
			auto charts = build_charts(set, std::move(part_charts), chart_of, folded);
			ZONETOOL_INFO("lightmap: %zu charts of placed models' parts (box projected), %zu of the world's", part_chart_count,
				charts.size() - part_chart_count);

			// the charts' extents, and how finely the probe lighting varies at each
			parallel_for(static_cast<std::uint32_t>(charts.size()), [&](const std::uint32_t i, std::uint32_t)
			{
				auto& c = charts[i];
				chart_bounds(set, c);
				vec3 centroid{};
				auto area = 0.0f;
				for (const auto t : c.triangles)
				{
					const auto w = set.areas[t] + 1e-6f;
					for (const auto& p : set.corners[t])
					{
						for (auto k = 0; k < 3; k++)
						{
							centroid[k] += w * p[k] / 3.0f;
						}
					}
					area += w;
				}
				for (auto& k : centroid)
				{
					k /= area;
				}
				c.probe_texel = lighting.texel_size(lighting.volume_at(centroid.data()), centroid.data());
			});

			// ---- the finest scale that fits one atlas ----------------------------------------------
			const auto atlas_texels = static_cast<double>(atlas_width) * atlas_max_height;
			const auto size_all = [&](const float s)
			{
				auto sum = 0.0;
				for (auto& c : charts)
				{
					size_chart(c, s);
					sum += static_cast<double>(c.width) * c.height;
				}
				return sum;
			};
			// the rectangles' total area only bounds the scale from below, the packing decides
			auto scale = finest_scale;
			if (size_all(scale) > atlas_texels)
			{
				auto lo = scale, hi = scale * 2.0f;
				while (size_all(hi) > atlas_texels)
				{
					lo = hi;
					hi *= 2.0f;
					if (hi > 65536.0f)
					{
						ZONETOOL_FATAL("lightmap: %zu charts need more than %ux%u texels at any texel size", charts.size(),
							atlas_width, atlas_max_height);
					}
				}
				for (auto i = 0; i < 40; i++)
				{
					const auto mid = 0.5f * (lo + hi);
					(size_all(mid) > atlas_texels ? lo : hi) = mid;
				}
				scale = hi;
			}
			const auto area_bound = scale;
			std::uint32_t used_height = 0;
			for (auto attempt = 0;; attempt++)
			{
				size_all(scale);
				if (pack(charts, used_height))
				{
					break;
				}
				if (attempt >= 200)
				{
					ZONETOOL_FATAL("lightmap: %zu charts do not fit %ux%u even at %g probe texels per lightmap texel", charts.size(),
						atlas_width, atlas_max_height, scale);
				}
				scale *= 1.01f;
			}
			const auto atlas_height = (used_height + 3) & ~3u;
			const auto texel_count = static_cast<std::size_t>(atlas_width) * atlas_height;
			{
				// the packing must leave the charts' rectangles disjoint
				std::vector<std::uint8_t> taken(texel_count, 0);
				auto used = 0.0;
				for (const auto& c : charts)
				{
					for (auto y = c.y; y < c.y + c.height; y++)
					{
						for (auto x = c.x; x < c.x + c.width; x++)
						{
							auto& t = taken[static_cast<std::size_t>(y) * atlas_width + x];
							if (t)
							{
								ZONETOOL_FATAL("lightmap: the packing put two charts on texel %u,%u", x, y);
							}
							t = 1;
						}
					}
					used += static_cast<double>(c.width) * c.height;
				}
				ZONETOOL_INFO("lightmap: %zu charts (%u folded ones charted again) packed in %ux%u at %.3f probe texels per texel "
					"(area bound %.3f), %.1f%% of the atlas", charts.size(), folded, atlas_width, atlas_height, scale, area_bound,
					100.0 * used / static_cast<double>(texel_count));
			}

			// ---- vertices: one per (chart, source vertex), lightmap coordinates set ----------------
			std::vector<zonetool::iw7::GfxWorldVertex> vertices;
			vertices.reserve(zone->vertexCount + zone->vertexCount / 4);
			for (auto s = 0u; s < surface_count; s++)
			{
				auto& surface = world->dpvs.surfaces[s];
				auto& tris = surface.tris;
				const auto first = static_cast<unsigned int>(vertices.size());
				auto* indices = world->draw.indices + tris.baseIndex;
				const auto* source = zone->vd.vertices + tris.firstVertex;

				if (first_triangle[s] == unassigned)
				{
					vertices.insert(vertices.end(), source, source + tris.vertexCount);
					tris.firstVertex = first;
					continue;
				}

				std::unordered_map<std::uint64_t, std::uint32_t> remap;
				for (auto t = 0u; t < tris.triCount; t++)
				{
					const auto chart_index = chart_of[first_triangle[s] + t];
					const auto& c = charts[chart_index];
					for (auto k = 0; k < 3; k++)
					{
						auto& index = indices[t * 3 + k];
						const auto key = (static_cast<std::uint64_t>(chart_index) << 32) | index;
						auto found = remap.find(key);
						if (found == remap.end())
						{
							const auto local = static_cast<std::uint32_t>(vertices.size() - first);
							if (local > 0xFFFF)
							{
								ZONETOOL_FATAL("lightmap: surface %u needs more than 65536 vertices once split into charts", s);
							}
							auto v = source[index];
							float x, y;
							atlas_position(c, v.xyz, x, y);
							v.lmapCoord[0] = x / static_cast<float>(atlas_width);
							v.lmapCoord[1] = y / static_cast<float>(atlas_height);
							vertices.push_back(v);
							found = remap.emplace(key, local).first;
						}
						index = static_cast<unsigned short>(found->second);
					}
				}
				tris.firstVertex = first;
				tris.vertexCount = static_cast<unsigned short>(vertices.size() - first);
			}

			const auto source_vertices = zone->vertexCount;
			zone->vertexCount = static_cast<unsigned int>(vertices.size());
			zone->vd.vertices = allocator.allocate_array<zonetool::iw7::GfxWorldVertex>(vertices.size());
			std::memcpy(zone->vd.vertices, vertices.data(), sizeof(zonetool::iw7::GfxWorldVertex) * vertices.size());

			// ---- texels to bake: every texel a bilinear lookup on a chart's triangles reads, lit at the
			// closest point of the closest triangle (texel centres inside a triangle at the centre) ------
			std::vector<std::vector<sample>> chart_samples(charts.size());
			std::atomic<std::size_t> inside_count{ 0 };
			parallel_for(static_cast<std::uint32_t>(charts.size()), [&](const std::uint32_t i, std::uint32_t)
			{
				const auto& c = charts[i];
				std::vector<float> best(static_cast<std::size_t>(c.width) * c.height, std::numeric_limits<float>::max());
				std::vector<sample> at(best.size());
				for (const auto t : c.triangles)
				{
					float px[3], py[3];
					for (auto k = 0; k < 3; k++)
					{
						atlas_position(c, set.corners[t][k].data(), px[k], py[k]);
					}
					const auto x0 = std::max(static_cast<std::int32_t>(c.x), static_cast<std::int32_t>(std::floor(std::min({ px[0], px[1], px[2] }) - 1.5f)));
					const auto x1 = std::min(static_cast<std::int32_t>(c.x + c.width) - 1, static_cast<std::int32_t>(std::floor(std::max({ px[0], px[1], px[2] }) + 1.5f)));
					const auto y0 = std::max(static_cast<std::int32_t>(c.y), static_cast<std::int32_t>(std::floor(std::min({ py[0], py[1], py[2] }) - 1.5f)));
					const auto y1 = std::min(static_cast<std::int32_t>(c.y + c.height) - 1, static_cast<std::int32_t>(std::floor(std::max({ py[0], py[1], py[2] }) + 1.5f)));
					for (auto y = y0; y <= y1; y++)
					{
						for (auto x = x0; x <= x1; x++)
						{
							float bary[3];
							const auto d2 = closest_point(px, py, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, bary);
							const auto local = static_cast<std::size_t>(y - static_cast<std::int32_t>(c.y)) * c.width
								+ static_cast<std::size_t>(x - static_cast<std::int32_t>(c.x));
							if (d2 <= sample_reach_squared && d2 < best[local])
							{
								best[local] = d2;
								at[local] = { static_cast<std::uint32_t>(y) * atlas_width + static_cast<std::uint32_t>(x), t, { bary[1], bary[2] } };
							}
						}
					}
				}
				auto inside = 0u;
				for (auto k = 0u; k < best.size(); k++)
				{
					if (best[k] <= sample_reach_squared)
					{
						chart_samples[i].push_back(at[k]);
						inside += best[k] == 0.0f;
					}
				}
				if (chart_samples[i].empty())
				{
					throw std::runtime_error(utils::string::va("lightmap: chart %u (%zu triangles) has no texel to sample", i, c.triangles.size()));
				}
				inside_count += inside;
			});
			std::vector<sample> samples;
			for (auto& list : chart_samples)
			{
				samples.insert(samples.end(), list.begin(), list.end());
				list = {};
			}

			// the surface at a sample (the rebuilt vertices keep each triangle's corners in order)
			const auto point_of = [&](const sample& s)
			{
				const auto& tris = world->dpvs.surfaces[set.surface[s.triangle]].tris;
				const auto* indices = world->draw.indices + tris.baseIndex + set.local[s.triangle] * 3;
				const auto* verts = zone->vd.vertices + tris.firstVertex;
				const float w[3] = { 1.0f - s.bary[0] - s.bary[1], s.bary[0], s.bary[1] };
				surface_point p{};
				vec3 n{}, tn{};
				for (auto k = 0; k < 3; k++)
				{
					const auto& v = verts[indices[k]];
					const auto cn = unpack_unit(v.normal.packed);
					const auto ct = unpack_unit(v.tangent.packed);
					for (auto i = 0; i < 3; i++)
					{
						p.position[i] += w[k] * v.xyz[i];
						n[i] += w[k] * cn[i];
						tn[i] += w[k] * ct[i];
					}
				}
				p.normal = normalize(n);
				p.tangent = normalize(tn);
				p.binormal_sign = (verts[indices[0]].tangent.packed >> 30) ? -1.0f : 1.0f;
				return p;
			};

			// ---- bake, one sun volume at a time -------------------------------------------------------
			std::vector<float> ambient(texel_count * 3, 0.0f), directional(texel_count * 3, 0.0f);
			std::vector<std::uint8_t> direction(texel_count * 2, 128);
			std::vector<std::uint8_t> known(texel_count, 0);
			std::vector<std::uint8_t> sun_lit_texel(texel_count, 255);
			std::vector<std::uint16_t> sun_bits(samples.size(), 0xFFFF);
			std::vector<unsigned int> volume_of(samples.size());
			parallel_for(static_cast<std::uint32_t>(samples.size()), [&](const std::uint32_t k, std::uint32_t)
			{
				const auto p = point_of(samples[k]);
				volume_of[k] = lighting.volume_at(p.position.data());
			});
			std::vector<std::vector<std::uint32_t>> by_volume(lighting.volume_count());
			for (auto k = 0u; k < samples.size(); k++)
			{
				by_volume[volume_of[k]].push_back(k);
			}
			std::atomic<std::size_t> baked{ 0 };
			for (auto v = 0u; v < by_volume.size(); v++)
			{
				if (by_volume[v].empty())
				{
					continue;
				}
				lighting.load(v);
				parallel_for(static_cast<std::uint32_t>(by_volume[v].size()), [&](const std::uint32_t k, std::uint32_t)
				{
					const auto& smp = samples[by_volume[v][k]];
					const auto p = point_of(smp);
					std::array<vec3, fit_normals> normals, values;
					fit_directions(p.normal, p.tangent, normals);
					static_assert(sizeof(vec3) == sizeof(float[3]));
					lighting.diffuse(v, p.position.data(), reinterpret_cast<const float(*)[3]>(normals.data()), fit_normals,
						reinterpret_cast<float(*)[3]>(values.data()));
					const auto f = fit(p, normals, values);
					const auto t = smp.texel;
					for (auto c = 0; c < 3; c++)
					{
						ambient[static_cast<std::size_t>(t) * 3 + c] = f.ambient[c];
						directional[static_cast<std::size_t>(t) * 3 + c] = f.directional[c];
					}
					direction[static_cast<std::size_t>(t) * 2 + 0] = f.direction[0];
					direction[static_cast<std::size_t>(t) * 2 + 1] = f.direction[1];
					known[t] = 1;
					// the sun at sun_grid x sun_grid points over the texel (the chart's axes, flattened onto the surface)
					std::uint16_t bits = 0xFFFF;
					if (sun_on && charts[chart_of[smp.triangle]].part)
					{
						// a part's texel: one ray from its point
						const double q[3] = { p.position[0], p.position[1], p.position[2] };
						bits = sun_open(q, p.normal.data()) ? 0xFFFF : 0;
					}
					else if (sun_on)
					{
						bits = 0;
						const auto& c = charts[chart_of[smp.triangle]];
						for (auto sy = 0u; sy < sun_grid; sy++)
						{
							for (auto sx = 0u; sx < sun_grid; sx++)
							{
								const auto du = ((static_cast<double>(sx) + 0.5) / sun_grid - 0.5) * c.texel;
								const auto dv = ((static_cast<double>(sy) + 0.5) / sun_grid - 0.5) * c.texel;
								double off[3], q[3];
								for (auto i = 0; i < 3; i++)
								{
									off[i] = c.axis_u[i] * du + c.axis_v[i] * dv;
								}
								const auto along = off[0] * p.normal[0] + off[1] * p.normal[1] + off[2] * p.normal[2];
								for (auto i = 0; i < 3; i++)
								{
									q[i] = p.position[i] + off[i] - p.normal[i] * along;
								}
								if (sun_open(q, p.normal.data()))
								{
									bits |= static_cast<std::uint16_t>(1u << (sy * sun_grid + sx));
								}
							}
						}
					}
					sun_bits[by_volume[v][k]] = bits;
					const auto n = ++baked;
					if (n * 20 / samples.size() != (n - 1) * 20 / samples.size())
					{
						ZONETOOL_INFO("lightmap: %zu of %zu texels baked", n, samples.size());
					}
				});
				lighting.unload(v);
			}
			{
				std::string per_volume;
				for (auto v = 0u; v < by_volume.size(); v++)
				{
					std::size_t black = 0;
					for (const auto k : by_volume[v])
					{
						const auto t = static_cast<std::size_t>(samples[k].texel);
						black += ambient[t * 3] + ambient[t * 3 + 1] + ambient[t * 3 + 2]
							+ directional[t * 3] + directional[t * 3 + 1] + directional[t * 3 + 2] <= 0.0f;
					}
					per_volume += utils::string::va("%s%u: %zu (%zu black)", v ? ", " : "", v, by_volume[v].size(), black);
				}
				ZONETOOL_INFO("lightmap: texels per sun volume %s", per_volume.data());
			}

			// ---- ZT_SUN_PROBE="x y z radius": what BO3 has between the sun and the lit texels near a point ------------
			// Every static surface and every static model surface (first LOD), whether or not it blocks the sun here, is
			// traced; the first one each lit texel's ray crosses is reported, so a patch of sun where BO3 has none names
			// the geometry the blocker rules leave out.
			if (const char* probe = std::getenv("ZT_SUN_PROBE"); probe && sun_on)
			{
				double at[3]{}, radius = 0.0;
				if (std::sscanf(probe, "%lf %lf %lf %lf", &at[0], &at[1], &at[2], &radius) == 4)
				{
					std::vector<float> all;
					std::vector<std::string> labels;
					std::vector<std::uint32_t> label_of;
					for (auto s = 0u; s < world->surfaceCount; s++)
					{
						const auto& surface = world->dpvs.surfaces[s];
						const auto label = static_cast<std::uint32_t>(labels.size());
						labels.push_back(utils::string::va("world surface %u (%s, %s, flags 0x%X)", s, surface.material ? surface.material->name : "?",
							s < world->dpvs.staticSurfaceCount ? "static" : "not static", surface.flags));
						for (auto t = 0u; t < surface.tris.triCount * 3u; t++)
						{
							const auto index = world->draw.indices[surface.tris.baseIndex + t];
							const auto* xyz = zone->vd.vertices[surface.tris.firstVertex + index].xyz;
							all.insert(all.end(), xyz, xyz + 3);
						}
						label_of.insert(label_of.end(), surface.tris.triCount, label);
					}
					for (auto i = 0u; i < asset->dpvs.smodelCount; i++)
					{
						const auto& src = asset->dpvs.smodelDrawInsts[i];
						if (!src.model)
						{
							continue;
						}
						const auto meshes = xmodel::lod_meshes(src.model);
						if (meshes.empty())
						{
							continue;
						}
						const auto& [mesh, materials] = meshes.front();
						constexpr float no_offset[3] = { 0.0f, 0.0f, 0.0f };
						for (auto s = 0u; s < materials.size(); s++)
						{
							std::vector<zonetool::iw7::GfxPackedVertex> verts;
							std::vector<zonetool::iw7::Face> faces;
							if (!xmodel_mesh::append_placed(mesh, s, {}, src.placement.origin, src.placement.axis, src.placement.scale,
								no_offset, verts, faces))
							{
								continue;
							}
							const auto label = static_cast<std::uint32_t>(labels.size());
							labels.push_back(utils::string::va("static model %u %s (BO3 flags 0x%X) / %s (blocks sun: %s)", i, src.model->name, src.flags,
								materials[s] && materials[s]->name ? materials[s]->name : "?", materials[s] && world_material::model_blocks_sun(materials[s]) ? "yes" : "no"));
							for (const auto& f : faces)
							{
								for (const auto v : { f.v1, f.v2, f.v3 })
								{
									all.insert(all.end(), verts[v].xyz, verts[v].xyz + 3);
								}
							}
							label_of.insert(label_of.end(), faces.size(), label);
						}
					}
					const triangle_bvh everything(std::move(all));
					std::map<std::string, std::uint32_t> first_hits;
					auto lit = 0u, open = 0u;
					for (auto k = 0u; k < samples.size(); k++)
					{
						const auto p = point_of(samples[k]);
						const double d[3] = { p.position[0] - at[0], p.position[1] - at[1], p.position[2] - at[2] };
						if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > radius * radius || !sun_bits[k])
						{
							continue;
						}
						lit++;
						constexpr double reach = 131072.0;
						const double from[3] = { p.position[0] + p.normal[0] + sun_dir[0], p.position[1] + p.normal[1] + sun_dir[1],
							p.position[2] + p.normal[2] + sun_dir[2] };
						const double to[3] = { from[0] + sun_dir[0] * reach, from[1] + sun_dir[1] * reach, from[2] + sun_dir[2] * reach };
						std::uint32_t hit = ~0u;
						everything.blocked(from, to, 1e-7, 1.0, [&](const std::uint32_t triangle, double, double)
						{
							hit = triangle;
							return true;
						});
						if (hit == ~0u)
						{
							open++;
						}
						else
						{
							first_hits[labels[label_of[hit]]]++;
						}
					}
					ZONETOOL_INFO("sun probe (%g %g %g, radius %g): %u lit texels, %u open to the sky through everything", at[0], at[1], at[2],
						radius, lit, open);
					for (const auto& [label, count] : first_hits)
					{
						ZONETOOL_INFO("sun probe: %u lit texels' rays cross %s", count, label.data());
					}
				}
			}

			// ---- the sun's visibility as the distance field IW7 reads ---------------------------------------------
			// The lmap_..._sundfd world shader takes d = texel * 2 - 1 and shows the sun by smoothstep(sat((d + w) / 2w))
			// (w the runtime softness): d is a signed distance, 0 on the shadow's edge, so the bilinear lookup puts the
			// edge between texel centres where a plain lit fraction left it blurred over a texel. Per chart, the sub-texel
			// points' visibility on a grid sun_grid times finer; a texel's d = (distance to the nearest shadowed point -
			// distance to the nearest lit one) / 2, in texels, over sun_range texels.
			if (sun_on)
			{
				std::vector<std::vector<std::uint32_t>> of_chart(charts.size());
				for (auto k = 0u; k < samples.size(); k++)
				{
					of_chart[chart_of[samples[k].triangle]].push_back(k);
				}
				parallel_for(static_cast<std::uint32_t>(charts.size()), [&](const std::uint32_t i, std::uint32_t)
				{
					const auto& c = charts[i];
					const auto fw = c.width * sun_grid, fh = c.height * sun_grid;
					std::vector<std::int8_t> fine(static_cast<std::size_t>(fw) * fh, -1);
					for (const auto k : of_chart[i])
					{
						const auto t = samples[k].texel;
						const auto lx = t % atlas_width - c.x, ly = t / atlas_width - c.y;
						for (auto sy = 0u; sy < sun_grid; sy++)
						{
							for (auto sx = 0u; sx < sun_grid; sx++)
							{
								fine[static_cast<std::size_t>(ly * sun_grid + sy) * fw + lx * sun_grid + sx] =
									static_cast<std::int8_t>((sun_bits[k] >> (sy * sun_grid + sx)) & 1);
							}
						}
					}
					const auto reach = static_cast<int>(std::ceil(sun_range * sun_grid));
					for (const auto k : of_chart[i])
					{
						const auto t = samples[k].texel;
						const auto lx = static_cast<int>(t % atlas_width - c.x), ly = static_cast<int>(t / atlas_width - c.y);
						// the texel's centre in fine cells; cell centres at + 0.5
						const auto centre_x = static_cast<double>(lx * sun_grid) + sun_grid * 0.5;
						const auto centre_y = static_cast<double>(ly * sun_grid) + sun_grid * 0.5;
						double near_lit = DBL_MAX, near_dark = DBL_MAX;
						for (auto y = std::max(0, static_cast<int>(centre_y) - reach); y < std::min(static_cast<int>(fh), static_cast<int>(centre_y) + reach); y++)
						{
							for (auto x = std::max(0, static_cast<int>(centre_x) - reach); x < std::min(static_cast<int>(fw), static_cast<int>(centre_x) + reach); x++)
							{
								const auto v = fine[static_cast<std::size_t>(y) * fw + x];
								if (v < 0)
								{
									continue;
								}
								const auto off_x = x + 0.5 - centre_x, off_y = y + 0.5 - centre_y;
								const auto d2 = off_x * off_x + off_y * off_y;
								auto& best = v ? near_lit : near_dark;
								best = std::min(best, d2);
							}
						}
						double d;
						if (near_dark == DBL_MAX)
						{
							d = 1.0;
						}
						else if (near_lit == DBL_MAX)
						{
							d = -1.0;
						}
						else
						{
							d = (std::sqrt(near_dark) - std::sqrt(near_lit)) * 0.5 / sun_grid / sun_range;
						}
						d = std::clamp(d, -1.0, 1.0);
						sun_lit_texel[t] = static_cast<std::uint8_t>(std::lround((d + 1.0) * 0.5 * 255.0));
					}
				});
			}

			// ---- fill the rest of each chart (disjoint rectangles, so in parallel), encode ------------
			parallel_for(static_cast<std::uint32_t>(charts.size()), [&](const std::uint32_t i, std::uint32_t)
			{
				dilate(ambient, directional, direction, sun_lit_texel, known, atlas_width, charts[i]);
			});

			const std::string base = "*lightmap0";
			{
				// secondary: ambient rows on top, directional below
				std::vector<float> rgba(texel_count * 2 * 4, 0.0f);
				for (std::size_t i = 0; i < texel_count; i++)
				{
					for (auto c = 0; c < 3; c++)
					{
						rgba[i * 4 + c] = ambient[i * 3 + c] * map::bo3_light_scale;
						rgba[(texel_count + i) * 4 + c] = directional[i * 3 + c] * map::bo3_light_scale;
					}
					rgba[i * 4 + 3] = 1.0f;
					rgba[(texel_count + i) * 4 + 3] = 1.0f;
				}
				DirectX::Image src{};
				src.width = atlas_width;
				src.height = atlas_height * 2;
				src.format = DXGI_FORMAT_R32G32B32A32_FLOAT;
				src.rowPitch = static_cast<std::size_t>(atlas_width) * 16;
				src.slicePitch = src.rowPitch * src.height;
				src.pixels = reinterpret_cast<std::uint8_t*>(rgba.data());
				auto data = compress(src, DXGI_FORMAT_BC6H_UF16);
				write_lightmap_image(base + "_secondary", atlas_width, atlas_height * 2, DXGI_FORMAT_BC6H_UF16, data);
			}
			{
				std::vector<std::uint8_t> data(direction.begin(), direction.end());
				write_lightmap_image(base + "_secondunorm", atlas_width, atlas_height, DXGI_FORMAT_R8G8_UNORM, data);
			}
			{
				// the sun's visibility (see sun_visibility above); IW7 reads it only while its sun is on
				auto& lit = sun_lit_texel;
				DirectX::Image src{};
				src.width = atlas_width;
				src.height = atlas_height;
				src.format = DXGI_FORMAT_R8_UNORM;
				src.rowPitch = atlas_width;
				src.slicePitch = texel_count;
				src.pixels = lit.data();
				auto data = compress(src, DXGI_FORMAT_BC4_UNORM);
				write_lightmap_image(base + "_primary", atlas_width, atlas_height, DXGI_FORMAT_BC4_UNORM, data);
			}

			auto* lightmap = allocator.allocate<zonetool::iw7::GfxLightMap>();
			lightmap->name = allocator.duplicate_string(base);
			const char* suffixes[3] = { "_primary", "_secondary", "_secondunorm" };
			for (auto i = 0; i < 3; i++)
			{
				lightmap->textures[i] = allocator.allocate<zonetool::iw7::GfxImage>();
				lightmap->textures[i]->name = allocator.duplicate_string(base + suffixes[i]);
			}
			zonetool::iw7::gfx_light_map::dump(lightmap);

			world->draw.lightMapCount = 1;
			world->draw.lightMaps = allocator.allocate_array<zonetool::iw7::GfxLightMap*>(1);
			world->draw.lightMaps[0] = lightmap;
			for (auto s = 0u; s < static_count; s++)
			{
				auto& surface = world->dpvs.surfaces[s];
				if (surface.lightmapIndex != no_lightmap)
				{
					surface.lightmapIndex = 0;
				}
			}

			const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
			ZONETOOL_INFO("lightmap: %zu texels baked (%zu at a texel centre inside a triangle, %zu at the closest point of one) "
				"over %u sun volumes, vertices %u -> %u, %.0f s", samples.size(), inside_count.load(),
				samples.size() - inside_count.load(), lighting.volume_count(), source_vertices, zone->vertexCount, seconds);
		}
	}
}
