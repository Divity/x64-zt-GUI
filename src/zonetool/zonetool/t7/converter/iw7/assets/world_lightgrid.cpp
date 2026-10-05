#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "world_lightgrid.hpp"

#include "comworld.hpp"
#include "probe_lighting.hpp"
#include "static_model_clusters.hpp"
#include "world_sky.hpp"
#include "zonetool/t7/converter/iw7/map_common.hpp"
#include "../delaunay.hpp"
#include "../parallel.hpp"
#include "../triangle_bvh.hpp"

#include "zonetool/iw7/common/lightgrid/LightGridProbes.hpp"
#include "zonetool/iw7/common/lightgrid/LightGridSH.hpp"

#include <utils/string.hpp>

// IW7 (stock data; the sampling compute shader cs_del_sh_sample_lightgrid):
// * the legacy palette / tree light grid is the same stub in every stock map (below, as the IW5 -> IW7 port);
// * the GPU grid: SH probes (27 coefficients = 9 per channel in IW7's basis, [27] sun visibility 0..1, [28] 0) on a
//   32-unit lattice offset by 1/32, placed adaptively and joined
//   into positively oriented tetrahedra whose neighbour k lies across the face opposite vertex k;
// * the voxel tree (root 512, internal 128, leaf 32 units) holds per leaf a start tetrahedron for the walk
//   (none where no tetrahedron is) and a light list; its zoneBound is the world
//   bounds shrunk by one unit;
// * probe visibility: a tetrahedron vertex index with the high bit set has 16 bytes at tetrahedron * 64 + 16 * k
//   of tetrahedronVisibility: 15 values on the face opposite vertex k split four ways (node (i, j), i + j <= 4, is
//   byte j (11 - j) / 2 + i at face coordinates a = i / 4, b = j / 4). The shader fades the vertex out for sample
//   points closer to the face than 1 - value / 255 of the way to the vertex, i.e. behind the last occluder
//   between that face point and the probe. Tetrahedra with visibility come first (tetrahedronCountVisible).
// * static models light from gpuVisibleProbes: model i's samples start at unk0 | unk1 << 16, unk3 of them
//   (2 at the bounds centre; stock also uses up to 65 on large models, a layout not reverse engineered);
// * the voxel tree's leaves address per-voxel light lists for particles: a header (spots << 7) | omnis, then the
//   spot indices, then the omni indices, both ascending; the array starts { 16384, 1, 0 } and address 2 is the
//   empty list.
// The SH is BO3's probe lighting as a function of the normal (probe_lighting), fitted over IW7's 56 grid basis
// directions. Probes go where BO3 has local probe lighting (its local probes' boxes), finer where the lighting
// changes. The sun's visibility (coefficient 27) is traced from each probe (1 where the sun is black) against the
// lightmap's sun blockers, static models included (model roofs and walls must keep the sun off indoor probes).
// The probe visibility's occluders are the opaque static surfaces.

namespace zonetool::t7
{
	namespace converter::iw7::world_lightgrid
	{
		namespace
		{
			constexpr float cell = 32.0f;
			constexpr int cells_per_root = 16; // 512 / 32
			constexpr float root_size = cell * cells_per_root;
			constexpr int max_level = 4;       // octree levels: 512, 256, 128, 64, 32 units
			constexpr float fine_texel = 64.0f; // BO3 local probes this detailed mark where IW7 probes go
			constexpr unsigned int max_leaves = 16u << 20;
			constexpr float probe_offset = 1.0f / 32.0f;
			constexpr float probe_jitter = 0.25f; // breaks the lattice's Delaunay degeneracies
			constexpr unsigned int probe_budget = 120000;
			constexpr float refine_error = 0.02f; // relative SH error below which a cell is not split
			constexpr float local_margin = 256.0f; // probes reach this far past BO3's local probe boxes
			constexpr float small_smodel = 1024.0f; // larger static models mark only their box's shell
			constexpr unsigned int smodel_samples = 2;
			constexpr unsigned int runtime_probe_slots = 0x2000; // stock gpuVisibleProbesData = positions + 0x2000
			constexpr unsigned short voxel_list_empty = 2;
			constexpr unsigned int start_chunk = 4096;

			using sh_coeffs = std::array<float, 28>;

			std::uint64_t root_key(const int rx, const int ry, const int rz)
			{
				constexpr std::int64_t bias = 1 << 20;
				return (static_cast<std::uint64_t>(rx + bias) << 42) | (static_cast<std::uint64_t>(ry + bias) << 21)
					| static_cast<std::uint64_t>(rz + bias);
			}

			void root_coords(const std::uint64_t key, int& rx, int& ry, int& rz)
			{
				constexpr std::int64_t bias = 1 << 20;
				rx = static_cast<int>(static_cast<std::int64_t>((key >> 42) & 0x1FFFFF) - bias);
				ry = static_cast<int>(static_cast<std::int64_t>((key >> 21) & 0x1FFFFF) - bias);
				rz = static_cast<int>(static_cast<std::int64_t>(key & 0x1FFFFF) - bias);
			}

			int cell_bit(const int cx, const int cy, const int cz)
			{
				return ((cz & 15) << 8) | ((cy & 15) << 4) | (cx & 15);
			}

			int cell_of(const float v)
			{
				return static_cast<int>(std::floor(v / cell));
			}

			// 32-unit cells in world cell coordinates, 4096 bits per 512-unit root
			struct cell_set
			{
				std::unordered_map<std::uint64_t, std::array<std::uint64_t, 64>> roots;

				void set(const int cx, const int cy, const int cz)
				{
					auto& words = roots[root_key(cx >> 4, cy >> 4, cz >> 4)];
					const auto b = cell_bit(cx, cy, cz);
					words[b >> 6] |= 1ull << (b & 63);
				}

				bool test(const int cx, const int cy, const int cz) const
				{
					const auto found = roots.find(root_key(cx >> 4, cy >> 4, cz >> 4));
					if (found == roots.end())
					{
						return false;
					}
					const auto b = cell_bit(cx, cy, cz);
					return (found->second[b >> 6] >> (b & 63)) & 1;
				}

				template <typename F>
				void for_each(F&& fn) const
				{
					for (const auto& [key, words] : roots)
					{
						int rx, ry, rz;
						root_coords(key, rx, ry, rz);
						for (auto w = 0; w < 64; w++)
						{
							auto bits = words[w];
							while (bits)
							{
								unsigned long b;
								_BitScanForward64(&b, bits);
								bits &= bits - 1;
								const auto index = (w << 6) | static_cast<int>(b);
								fn(rx * cells_per_root + (index & 15), ry * cells_per_root + ((index >> 4) & 15), rz * cells_per_root + (index >> 8));
							}
						}
					}
				}

				std::size_t count() const
				{
					std::size_t total = 0;
					for (const auto& [key, words] : roots)
					{
						for (const auto w : words)
						{
							total += __popcnt64(w);
						}
					}
					return total;
				}
			};

			void mark_triangle(cell_set& out, const float* a, const float* b, const float* c)
			{
				const auto length = [](const float* p, const float* q)
				{
					const float d[3] = { q[0] - p[0], q[1] - p[1], q[2] - p[2] };
					return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
				};
				const auto edge = std::max({ length(a, b), length(b, c), length(c, a) });
				const auto n = std::max(1, static_cast<int>(std::ceil(edge / (cell * 0.5f))));
				for (auto i = 0; i <= n; i++)
				{
					for (auto j = 0; i + j <= n; j++)
					{
						const auto u = static_cast<float>(i) / static_cast<float>(n);
						const auto v = static_cast<float>(j) / static_cast<float>(n);
						const auto w = 1.0f - u - v;
						out.set(cell_of(a[0] * w + b[0] * u + c[0] * v), cell_of(a[1] * w + b[1] * u + c[1] * v),
							cell_of(a[2] * w + b[2] * u + c[2] * v));
					}
				}
			}

			void legacy_stub(zonetool::iw7::GfxLightGrid& grid, utils::memory::allocator& allocator)
			{
				constexpr int unk_values[] = { 0, 0, 5, 5, 6, 32, 32, 64, 0 };
				static constexpr int palette_addresses[3] = { 0, 30, 86 };
				static constexpr unsigned char palette_bitstream[116] = {
					0xE7, 0x1C, 0x00, 0xF8, 0x08, 0x80, 0x80, 0x80, 0x80, 0x80, 0xF1, 0x00, 0x08, 0x80, 0xF8, 0x80,
					0x80, 0x80, 0xB8, 0x48, 0x00, 0x80, 0xF8, 0x08, 0x80, 0x80, 0x80, 0x48, 0x48, 0x00, 0x00, 0x00,
					0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
					0x80, 0x80, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x5C, 0x5E, 0x4A, 0x3F,
					0xFF, 0xFF, 0x7F, 0x7F, 0xFF, 0xFF, 0x7F, 0x7F, 0xFF, 0xFF, 0x7F, 0xFF, 0xFF, 0xFF, 0x7F, 0xFF,
					0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
					0x80, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80,
					0x80, 0x80, 0x80, 0x00,
				};
				static constexpr unsigned int node_table[2] = { 16777217u, 2147483648u };
				static constexpr unsigned char leaf_table[6] = { 0x01, 0x83, 0x00, 0x04, 0x06, 0x11 };

				std::memcpy(grid.unk, unk_values, sizeof(unk_values));
				grid.tableVersion = 1;
				grid.paletteVersion = 1;
				grid.rangeExponent8BitsEncoding = 0;
				grid.rangeExponent12BitsEncoding = 4;
				grid.rangeExponent16BitsEncoding = 23;
				grid.stageCount = 0;
				grid.stageLightingContrastGain = nullptr;
				grid.paletteEntryCount = 3;
				grid.paletteEntryAddress = allocator.allocate_array<int>(3);
				std::memcpy(grid.paletteEntryAddress, palette_addresses, sizeof(palette_addresses));
				grid.paletteBitstreamSize = sizeof(palette_bitstream);
				grid.paletteBitstream = allocator.allocate_array<unsigned char>(sizeof(palette_bitstream));
				std::memcpy(grid.paletteBitstream, palette_bitstream, sizeof(palette_bitstream));
				std::memset(&grid.skyLightGridColors, 0, sizeof(grid.skyLightGridColors));
				for (auto i = 0; i < 56; i++)
				{
					grid.defaultLightGridColors.rgb[i][0] = 0.0f;
					grid.defaultLightGridColors.rgb[i][1] = 0.0f;
					grid.defaultLightGridColors.rgb[i][2] = 0.21875f;
				}
				auto& tree = grid.tree;
				tree.maxDepth = 1;
				tree.nodeCount = 2;
				tree.leafCount = 1;
				const int mins[3] = { 4092, 4092, 2047 }, maxs[3] = { 4100, 4100, 2049 }, halves[3] = { 4, 4, 1 };
				std::memcpy(tree.coordMinGridSpace, mins, sizeof(mins));
				std::memcpy(tree.coordMaxGridSpace, maxs, sizeof(maxs));
				std::memcpy(tree.coordHalfSizeGridSpace, halves, sizeof(halves));
				tree.defaultColorIndexBitCount = 2;
				tree.defaultLightIndexBitCount = 32;
				tree.p_nodeTable = allocator.allocate_array<unsigned int>(2);
				std::memcpy(tree.p_nodeTable, node_table, sizeof(node_table));
				tree.leafTableSize = sizeof(leaf_table);
				tree.p_leafTable = allocator.allocate_array<unsigned char>(sizeof(leaf_table));
				std::memcpy(tree.p_leafTable, leaf_table, sizeof(leaf_table));
			}

			// BO3's probe lighting at p over IW7's 56 grid basis directions, fitted in IW7's SH basis, and the sun's
			// visibility from p (coefficient 27, 0..1)
			void probe_sh(const probe_lighting::evaluator& lighting, const float p[3], const float sun_visibility, sh_coeffs& out)
			{
				float samples[56][3];
				lighting.diffuse(lighting.volume_at(p), p, lightgrid_sh::grid_basis_dirs, 56, samples);
				lightgrid_probes::project_sh(samples, lightgrid_sh::grid_basis_dirs, 56, map::bo3_light_scale, out.data());
				out[27] = sun_visibility;
			}

			std::uint64_t mix(std::uint64_t x)
			{
				x += 0x9E3779B97F4A7C15ull;
				x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
				x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
				return x ^ (x >> 31);
			}

			struct local_light
			{
				unsigned short index;
				bool spot;
				float origin[3];
				float axis[3];
				float radius;
				float half_angle;
				float hull_min[3], hull_max[3];
			};

			// the light reaches the box: its hull's box overlaps it, and its sphere / cone reaches the box's
			// bounding sphere
			bool reaches(const local_light& l, const float lo[3], const float hi[3])
			{
				for (auto k = 0; k < 3; k++)
				{
					if (hi[k] < l.hull_min[k] || lo[k] > l.hull_max[k])
					{
						return false;
					}
				}
				float centre[3], half[3];
				for (auto k = 0; k < 3; k++)
				{
					centre[k] = (lo[k] + hi[k]) * 0.5f;
					half[k] = (hi[k] - lo[k]) * 0.5f;
				}
				const auto r = std::sqrt(half[0] * half[0] + half[1] * half[1] + half[2] * half[2]);
				const float d[3] = { centre[0] - l.origin[0], centre[1] - l.origin[1], centre[2] - l.origin[2] };
				const auto dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
				if (dist - r > l.radius)
				{
					return false;
				}
				if (l.spot && dist > 0.001f)
				{
					const auto c = (d[0] * l.axis[0] + d[1] * l.axis[1] + d[2] * l.axis[2]) / dist;
					if (std::acos(std::clamp(c, -1.0f, 1.0f)) > l.half_angle + std::asin(std::min(1.0f, r / dist)))
					{
						return false;
					}
				}
				return true;
			}

			// face coordinates of the visibility grid per vertex k (the shader's): a weights vertex face_a[k],
			// b weights face_b[k], the rest goes to face_c[k]
			constexpr int face_a[4] = { 3, 0, 0, 0 };
			constexpr int face_b[4] = { 1, 3, 1, 1 };
			constexpr int face_c[4] = { 2, 2, 3, 2 };
		}

		void build(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, const zonetool::iw7::GfxWorldTransientZone* zone,
			const std::vector<std::uint8_t>& occluders, const world_lightmap::sun_blockers& sun_blockers,
			utils::memory::allocator& allocator)
		{
			const auto start = std::chrono::steady_clock::now();
			const auto* com = comworld::converted();
			// the sun (possibly black): its direction toward the sun, for coefficient 27
			const auto& sun = com->primaryLights[world->lastSunPrimaryLightIndex];
			const auto sun_lit = sun.color[0] + sun.color[1] + sun.color[2] > 0.0f;
			if (occluders.size() != world->dpvs.staticSurfaceCount)
			{
				ZONETOOL_FATAL("light grid: %zu occluder flags for %u static surfaces", occluders.size(), world->dpvs.staticSurfaceCount);
			}

			auto& grid = world->lightGrid;
			legacy_stub(grid, allocator);

			// ---- geometry: occupied cells and occluding triangles ----------------------------------------------------
			cell_set geometry;
			std::vector<float> occluder_triangles;
			const auto sky = world_sky::surfaces(world);
			for (auto s = 0u; s < world->surfaceCount; s++)
			{
				if (sky[s])
				{
					continue; // the box around the world
				}
				const auto& surface = world->dpvs.surfaces[s];
				const auto occludes = s < world->dpvs.staticSurfaceCount && occluders[s];
				for (auto t = 0u; t < surface.tris.triCount; t++)
				{
					const float* p[3];
					for (auto k = 0; k < 3; k++)
					{
						const auto index = world->draw.indices[surface.tris.baseIndex + t * 3 + k];
						p[k] = zone->vd.vertices[surface.tris.firstVertex + index].xyz;
					}
					mark_triangle(geometry, p[0], p[1], p[2]);
					if (occludes)
					{
						for (const auto* v : p)
						{
							occluder_triangles.insert(occluder_triangles.end(), v, v + 3);
						}
					}
				}
			}
			const auto& clusters = static_model_clusters::current();
			for (auto m = 0u; m < world->dpvs.smodelCount; m++)
			{
				// a shadow proxy's box spans the models it casts for, which mark their own
				if (m >= clusters.proxy_begin && m < clusters.proxy_begin + clusters.proxies.size())
				{
					continue;
				}
				const auto& b = world->dpvs.smodelInsts[m].bounds;
				int c0[3], c1[3];
				auto largest = 0.0f;
				for (auto k = 0; k < 3; k++)
				{
					c0[k] = cell_of(b.midPoint[k] - b.halfSize[k]);
					c1[k] = cell_of(b.midPoint[k] + b.halfSize[k]);
					largest = std::max(largest, b.halfSize[k] * 2.0f);
				}
				const auto shell_only = largest > small_smodel;
				for (auto x = c0[0]; x <= c1[0]; x++)
				{
					for (auto y = c0[1]; y <= c1[1]; y++)
					{
						for (auto z = c0[2]; z <= c1[2]; z++)
						{
							if (!shell_only || x == c0[0] || x == c1[0] || y == c0[1] || y == c1[1] || z == c0[2] || z == c1[2])
							{
								geometry.set(x, y, z);
							}
						}
					}
				}
			}
			const auto occluder_count = occluder_triangles.size() / 9;
			const triangle_bvh occluder_bvh(std::move(occluder_triangles));

			// the share of five rays toward the sun (its direction and four 0.5 degrees around it, a soft edge) that leave
			// the map past every sun blocker; 1 where the sun is dark. ignore_model: the static model lit at p (p is inside it)
			const auto sun_visibility = [&](const float p[3], const std::uint32_t* ignore_model) -> float
			{
				if (!sun_lit)
				{
					return 1.0f;
				}
				return world_lightmap::sun_fraction(sun_blockers, sun.dir, p, ignore_model);
			};

			// ---- where probes go: roots with geometry inside BO3's detailed local probes ---------------------------------
			// (BO3's coarse probes, texels of hundreds of units, may cover the whole map including its vistas)
			probe_lighting::evaluator lighting(asset, map::lighting_state());
			for (auto v = 0u; v < lighting.volume_count(); v++)
			{
				lighting.load(v);
			}
			std::vector<std::array<float, 6>> fine_boxes;
			for (const auto& box : lighting.local_boxes())
			{
				if (box[6] <= fine_texel)
				{
					fine_boxes.push_back({ box[0] - local_margin, box[1] - local_margin, box[2] - local_margin,
						box[3] + local_margin, box[4] + local_margin, box[5] + local_margin });
				}
			}
			std::unordered_set<std::uint64_t> probe_roots;
			for (const auto& [key, words] : geometry.roots)
			{
				int r[3];
				root_coords(key, r[0], r[1], r[2]);
				const float lo[3] = { static_cast<float>(r[0]) * root_size, static_cast<float>(r[1]) * root_size, static_cast<float>(r[2]) * root_size };
				for (const auto& b : fine_boxes)
				{
					if (lo[0] <= b[3] && lo[0] + root_size >= b[0] && lo[1] <= b[4] && lo[1] + root_size >= b[1]
						&& lo[2] <= b[5] && lo[2] + root_size >= b[2])
					{
						probe_roots.insert(key);
						break;
					}
				}
			}
			if (probe_roots.empty())
			{
				ZONETOOL_FATAL("light grid: no geometry lies in BO3's %zu detailed local probes", fine_boxes.size());
			}

			// ---- voxel leaves (stock layout): 128-unit nodes within one node of geometry, split into 32-unit leaves in
			// the probe roots and kept whole as 128-unit leaves elsewhere -----------------------------------------------
			std::unordered_set<std::uint64_t> near_nodes; // 128-unit nodes, root_key-encoded node coordinates
			{
				std::unordered_set<std::uint64_t> geometry_nodes;
				geometry.for_each([&](const int cx, const int cy, const int cz)
				{
					geometry_nodes.insert(root_key(cx >> 2, cy >> 2, cz >> 2));
				});
				for (const auto key : geometry_nodes)
				{
					int n[3];
					root_coords(key, n[0], n[1], n[2]);
					for (auto dx = -1; dx <= 1; dx++)
					{
						for (auto dy = -1; dy <= 1; dy++)
						{
							for (auto dz = -1; dz <= 1; dz++)
							{
								near_nodes.insert(root_key(n[0] + dx, n[1] + dy, n[2] + dz));
							}
						}
					}
				}
			}
			std::unordered_set<std::uint64_t> tree_roots;
			for (const auto key : near_nodes)
			{
				int n[3];
				root_coords(key, n[0], n[1], n[2]);
				tree_roots.insert(root_key(n[0] >> 2, n[1] >> 2, n[2] >> 2));
			}

			int rmin[3] = { INT_MAX, INT_MAX, INT_MAX }, rmax[3] = { INT_MIN, INT_MIN, INT_MIN };
			for (const auto key : tree_roots)
			{
				int r[3];
				root_coords(key, r[0], r[1], r[2]);
				for (auto k = 0; k < 3; k++)
				{
					rmin[k] = std::min(rmin[k], r[k]);
					rmax[k] = std::max(rmax[k], r[k]);
				}
			}
			int dims[3];
			float origin[3];
			for (auto k = 0; k < 3; k++)
			{
				dims[k] = rmax[k] - rmin[k] + 1;
				origin[k] = static_cast<float>(rmin[k]) * root_size;
			}

			// the tree: dense top-down columns, root nodes, 128-unit nodes, leaves (the layout the shader walks)
			std::vector<zonetool::iw7::GfxVoxelTopDownViewNode> top_down(static_cast<std::size_t>(dims[0]) * dims[1]);
			std::vector<std::array<int, 3>> root_nodes; // world root coordinates
			{
				std::vector<std::vector<int>> columns(top_down.size());
				for (const auto key : tree_roots)
				{
					int r[3];
					root_coords(key, r[0], r[1], r[2]);
					columns[static_cast<std::size_t>(r[1] - rmin[1]) * dims[0] + (r[0] - rmin[0])].push_back(r[2]);
				}
				for (auto ry = 0; ry < dims[1]; ry++)
				{
					for (auto rx = 0; rx < dims[0]; rx++)
					{
						const auto col = static_cast<std::size_t>(ry) * dims[0] + rx;
						auto& zs = columns[col];
						if (zs.empty())
						{
							top_down[col] = { -1, 0x7FFFFFFF, static_cast<int>(0x80000000) };
							continue;
						}
						std::sort(zs.begin(), zs.end());
						const auto z_lo = zs.front() - rmin[2], z_hi = zs.back() - rmin[2];
						top_down[col] = { static_cast<int>(root_nodes.size()), z_lo, z_hi };
						for (auto z = z_lo; z <= z_hi; z++)
						{
							root_nodes.push_back({ rx + rmin[0], ry + rmin[1], z + rmin[2] });
						}
					}
				}
			}

			// nodes and leaves root by root, so each root's leaves are contiguous
			struct voxel_leaf
			{
				int cell[3]; // world 32-unit cell of the low corner
				int size;    // in cells: 1 or 4
			};
			std::vector<zonetool::iw7::GfxVoxelInternalNode> internal(root_nodes.size());
			std::vector<zonetool::iw7::GfxVoxelInternalNode> node_level;
			std::vector<voxel_leaf> leaves;
			std::vector<std::uint32_t> root_first_leaf(root_nodes.size() + 1, 0);
			std::vector<std::uint8_t> root_fine(root_nodes.size(), 0);
			auto fine_roots = 0u, coarse_roots = 0u;
			for (auto r = 0u; r < root_nodes.size(); r++)
			{
				root_first_leaf[r] = static_cast<std::uint32_t>(leaves.size());
				const auto& rc = root_nodes[r];
				const auto fine = probe_roots.contains(root_key(rc[0], rc[1], rc[2]));
				root_fine[r] = fine;
				std::uint64_t mask = 0;
				const auto first_node = static_cast<unsigned int>(node_level.size());
				const auto first_leaf = static_cast<unsigned int>(leaves.size());
				for (auto i = 0; i < 64; i++)
				{
					const int node[3] = { rc[0] * 4 + (i & 3), rc[1] * 4 + ((i >> 2) & 3), rc[2] * 4 + ((i >> 4) & 3) };
					if (!near_nodes.contains(root_key(node[0], node[1], node[2])))
					{
						continue;
					}
					mask |= 1ull << i;
					if (fine)
					{
						auto& n = node_level.emplace_back();
						n.firstNodeIndex[0] = static_cast<int>(static_cast<unsigned int>(leaves.size()) | 0x80000000u);
						n.childNodeMask[0] = -1;
						n.childNodeMask[1] = -1;
						for (auto k = 0; k < 64; k++)
						{
							leaves.push_back({ { node[0] * 4 + (k & 3), node[1] * 4 + ((k >> 2) & 3), node[2] * 4 + ((k >> 4) & 3) }, 1 });
						}
					}
					else
					{
						leaves.push_back({ { node[0] * 4, node[1] * 4, node[2] * 4 }, 4 });
					}
				}
				if (mask)
				{
					(fine ? fine_roots : coarse_roots)++;
				}
				auto& n = internal[r];
				n.firstNodeIndex[0] = fine ? static_cast<int>(first_node) : static_cast<int>(first_leaf | 0x80000000u);
				n.childNodeMask[0] = static_cast<int>(mask & 0xFFFFFFFF);
				n.childNodeMask[1] = static_cast<int>(mask >> 32);
				if (leaves.size() > max_leaves)
				{
					ZONETOOL_FATAL("light grid: more than %u voxel leaves", max_leaves);
				}
			}
			root_first_leaf[root_nodes.size()] = static_cast<std::uint32_t>(leaves.size());
			const auto node_base = static_cast<int>(internal.size());
			for (auto r = 0u; r < root_nodes.size(); r++)
			{
				if (root_fine[r])
				{
					internal[r].firstNodeIndex[0] += node_base;
				}
			}
			internal.insert(internal.end(), node_level.begin(), node_level.end());
			for (auto& n : internal)
			{
				n.firstNodeIndex[1] = static_cast<int>(static_cast<unsigned int>(n.firstNodeIndex[0])
					+ __popcnt(static_cast<unsigned int>(n.childNodeMask[0])));
			}
			const auto leaf_count = static_cast<std::uint32_t>(leaves.size());

			// ---- probes: an octree over the probe roots, split where interpolation misses BO3's lighting --------------
			struct probe_point
			{
				float position[3];
				sh_coeffs sh{};
				bool probe = false;
			};
			std::vector<probe_point> points;
			std::unordered_map<std::uint64_t, std::uint32_t> point_of;
			std::vector<std::uint32_t> pending;
			auto probe_count = 0u;

			// lattice coordinates (32-unit steps from the tree origin)
			const auto point_at = [&](const int lx, const int ly, const int lz)
			{
				const auto key = (static_cast<std::uint64_t>(lx) << 42) | (static_cast<std::uint64_t>(ly) << 21) | static_cast<std::uint64_t>(lz);
				const auto [it, inserted] = point_of.try_emplace(key, static_cast<std::uint32_t>(points.size()));
				if (inserted)
				{
					auto& p = points.emplace_back();
					const auto h = mix(key);
					const int l[3] = { lx, ly, lz };
					for (auto k = 0; k < 3; k++)
					{
						const auto unit = static_cast<float>((h >> (k * 21)) & 0x1FFFFF) / static_cast<float>(0x1FFFFF) * 2.0f - 1.0f;
						p.position[k] = origin[k] + static_cast<float>(l[k]) * cell + probe_offset + unit * probe_jitter;
					}
					pending.push_back(it->second);
				}
				return it->second;
			};
			const auto make_probe = [&](const std::uint32_t index)
			{
				if (!points[index].probe)
				{
					points[index].probe = true;
					probe_count++;
				}
			};
			const auto evaluate_pending = [&]()
			{
				parallel_for(static_cast<std::uint32_t>(pending.size()), [&](const std::uint32_t k, std::uint32_t)
				{
					auto& p = points[pending[k]];
					probe_sh(lighting, p.position, sun_visibility(p.position, nullptr), p.sh);
				});
				pending.clear();
			};

			struct octree_cell
			{
				int x, y, z; // lattice coordinates of the low corner
				int level;
				float priority;
				bool operator<(const octree_cell& o) const
				{
					return priority < o.priority;
				}
			};
			static constexpr float sample_at[7][3] = {
				{ 0.5f, 0.5f, 0.5f }, { 0.0f, 0.5f, 0.5f }, { 1.0f, 0.5f, 0.5f }, { 0.5f, 0.0f, 0.5f },
				{ 0.5f, 1.0f, 0.5f }, { 0.5f, 0.5f, 0.0f }, { 0.5f, 0.5f, 1.0f },
			};
			const auto size_of = [](const int level)
			{
				return cells_per_root >> level;
			};
			const auto add_samples = [&](const octree_cell& c)
			{
				const auto h = size_of(c.level) / 2;
				for (const auto& s : sample_at)
				{
					point_at(c.x + static_cast<int>(s[0] * 2.0f) * h, c.y + static_cast<int>(s[1] * 2.0f) * h, c.z + static_cast<int>(s[2] * 2.0f) * h);
				}
			};
			const auto lookup = [&](const int lx, const int ly, const int lz) -> const probe_point&
			{
				return points[point_of.at((static_cast<std::uint64_t>(lx) << 42) | (static_cast<std::uint64_t>(ly) << 21) | static_cast<std::uint64_t>(lz))];
			};
			const auto has_geometry = [&](const octree_cell& c)
			{
				const auto s = size_of(c.level);
				const int base[3] = { rmin[0] * cells_per_root + c.x, rmin[1] * cells_per_root + c.y, rmin[2] * cells_per_root + c.z };
				if (c.level == 0)
				{
					const auto found = geometry.roots.find(root_key(base[0] >> 4, base[1] >> 4, base[2] >> 4));
					return found != geometry.roots.end() && std::any_of(found->second.begin(), found->second.end(), [](const std::uint64_t w) { return w != 0; });
				}
				for (auto x = 0; x < s; x++)
				{
					for (auto y = 0; y < s; y++)
					{
						for (auto z = 0; z < s; z++)
						{
							if (geometry.test(base[0] + x, base[1] + y, base[2] + z))
							{
								return true;
							}
						}
					}
				}
				return false;
			};
			auto error_floor = 0.0f;
			const auto cell_error = [&](const octree_cell& c)
			{
				const auto s = size_of(c.level);
				const probe_point* corner[8];
				for (auto i = 0; i < 8; i++)
				{
					corner[i] = &lookup(c.x + (i & 1) * s, c.y + ((i >> 1) & 1) * s, c.z + ((i >> 2) & 1) * s);
				}
				auto level = 0.0f;
				for (auto i = 0; i < 8; i++)
				{
					for (auto ch = 0; ch < 3; ch++)
					{
						level += std::fabs(corner[i]->sh[ch * 9]) / 8.0f;
					}
				}
				auto worst = 0.0f;
				const auto h = s / 2;
				for (const auto& sp : sample_at)
				{
					const auto& truth = lookup(c.x + static_cast<int>(sp[0] * 2.0f) * h, c.y + static_cast<int>(sp[1] * 2.0f) * h,
						c.z + static_cast<int>(sp[2] * 2.0f) * h).sh;
					auto error = 0.0f;
					for (auto ch = 0; ch < 3; ch++)
					{
						for (auto b = 0; b < 4; b++)
						{
							auto predicted = 0.0f;
							for (auto i = 0; i < 8; i++)
							{
								const auto wx = (i & 1) ? sp[0] : 1.0f - sp[0];
								const auto wy = ((i >> 1) & 1) ? sp[1] : 1.0f - sp[1];
								const auto wz = ((i >> 2) & 1) ? sp[2] : 1.0f - sp[2];
								predicted += wx * wy * wz * corner[i]->sh[ch * 9 + b];
							}
							error += std::fabs(truth[ch * 9 + b] - predicted) * (b ? 0.5f : 1.0f);
						}
					}
					worst = std::max(worst, error);
				}
				return worst / (level + error_floor);
			};

			std::priority_queue<octree_cell> queue;
			std::vector<octree_cell> roots;
			for (const auto key : probe_roots)
			{
				int r[3];
				root_coords(key, r[0], r[1], r[2]);
				roots.push_back({ (r[0] - rmin[0]) * cells_per_root, (r[1] - rmin[1]) * cells_per_root, (r[2] - rmin[2]) * cells_per_root, 0, 0.0f });
			}
			for (const auto& c : roots)
			{
				for (auto i = 0; i < 8; i++)
				{
					make_probe(point_at(c.x + (i & 1) * cells_per_root, c.y + ((i >> 1) & 1) * cells_per_root, c.z + ((i >> 2) & 1) * cells_per_root));
				}
				add_samples(c);
			}
			evaluate_pending();
			{
				// errors are relative to the local level, floored at a tenth of the mean level of the root corners
				double total = 0.0;
				auto n = 0u;
				for (const auto& p : points)
				{
					if (p.probe)
					{
						total += std::fabs(p.sh[0]) + std::fabs(p.sh[9]) + std::fabs(p.sh[18]);
						n++;
					}
				}
				error_floor = static_cast<float>(0.1 * total / std::max(1u, n));
			}
			const auto push = [&](octree_cell c)
			{
				const auto error = cell_error(c);
				if (!(error > refine_error))
				{
					return;
				}
				c.priority = error * static_cast<float>(size_of(c.level)) * (has_geometry(c) ? 1.0f : 0.5f);
				queue.push(c);
			};
			for (const auto& c : roots)
			{
				push(c);
			}

			unsigned int splits[max_level] = {};
			std::vector<octree_cell> batch, children;
			while (!queue.empty() && probe_count < probe_budget)
			{
				batch.clear();
				children.clear();
				const auto batch_size = std::max<std::size_t>(256, queue.size() / 16);
				while (!queue.empty() && batch.size() < batch_size && probe_count + 19 * (batch.size() + 1) <= probe_budget)
				{
					batch.push_back(queue.top());
					queue.pop();
				}
				if (batch.empty())
				{
					break;
				}
				for (const auto& c : batch)
				{
					splits[c.level]++;
					const auto h = size_of(c.level) / 2;
					for (auto i = 0; i < 27; i++)
					{
						make_probe(point_at(c.x + (i % 3) * h, c.y + ((i / 3) % 3) * h, c.z + (i / 9) * h));
					}
					for (auto i = 0; i < 8; i++)
					{
						const octree_cell child{ c.x + (i & 1) * h, c.y + ((i >> 1) & 1) * h, c.z + ((i >> 2) & 1) * h, c.level + 1, 0.0f };
						if (child.level < max_level)
						{
							add_samples(child);
							children.push_back(child);
						}
					}
				}
				evaluate_pending();
				for (const auto& c : children)
				{
					push(c);
				}
			}
			if (probe_count > lightgrid_probes::max_probes)
			{
				ZONETOOL_FATAL("light grid: %u probes exceed IW7's %u", probe_count, lightgrid_probes::max_probes);
			}

			std::vector<std::uint32_t> probe_points;
			for (auto i = 0u; i < points.size(); i++)
			{
				if (points[i].probe)
				{
					probe_points.push_back(i);
				}
			}
			std::vector<delaunay::point> positions(probe_points.size());
			for (auto i = 0u; i < probe_points.size(); i++)
			{
				const auto& p = points[probe_points[i]];
				positions[i] = { p.position[0], p.position[1], p.position[2] };
			}

			// ---- tetrahedra --------------------------------------------------------------------------------------------
			delaunay::mesh mesh;
			try
			{
				mesh = delaunay::tetrahedralize(positions);
			}
			catch (const std::exception& e)
			{
				ZONETOOL_FATAL("light grid: %s", e.what());
			}
			const auto tet_count = static_cast<std::uint32_t>(mesh.tets.size());
			std::vector<std::uint32_t> tet_of_probe(positions.size(), delaunay::none);
			for (auto t = 0u; t < tet_count; t++)
			{
				for (const auto v : mesh.tets[t])
				{
					tet_of_probe[v] = t;
				}
			}

			// start tetrahedra: a walk to each leaf's centre, from the previous leaf's; the probes' box bounds the hull
			double hull_lo[3] = { DBL_MAX, DBL_MAX, DBL_MAX }, hull_hi[3] = { -DBL_MAX, -DBL_MAX, -DBL_MAX };
			for (const auto& p : positions)
			{
				for (auto k = 0; k < 3; k++)
				{
					hull_lo[k] = std::min(hull_lo[k], p[k]);
					hull_hi[k] = std::max(hull_hi[k], p[k]);
				}
			}
			std::vector<std::uint32_t> start_tet(leaf_count, delaunay::none);
			const auto chunks = (leaf_count + start_chunk - 1) / start_chunk;
			parallel_for(chunks, [&](const std::uint32_t chunk, std::uint32_t)
			{
				auto seed = delaunay::none;
				for (auto l = chunk * start_chunk; l < std::min(leaf_count, (chunk + 1) * start_chunk); l++)
				{
					const auto& leaf = leaves[l];
					const auto half = leaf.size * 0.5;
					const delaunay::point centre = { (leaf.cell[0] + half) * cell, (leaf.cell[1] + half) * cell, (leaf.cell[2] + half) * cell };
					if (centre[0] < hull_lo[0] || centre[1] < hull_lo[1] || centre[2] < hull_lo[2]
						|| centre[0] > hull_hi[0] || centre[1] > hull_hi[1] || centre[2] > hull_hi[2])
					{
						continue;
					}
					if (seed == delaunay::none)
					{
						// the probe at the leaf's root corner, when its root holds probes
						const int lx = (leaf.cell[0] >> 4) * cells_per_root - rmin[0] * cells_per_root;
						const int ly = (leaf.cell[1] >> 4) * cells_per_root - rmin[1] * cells_per_root;
						const int lz = (leaf.cell[2] >> 4) * cells_per_root - rmin[2] * cells_per_root;
						const auto found = point_of.find((static_cast<std::uint64_t>(lx) << 42) | (static_cast<std::uint64_t>(ly) << 21) | static_cast<std::uint64_t>(lz));
						if (found != point_of.end() && points[found->second].probe)
						{
							const auto index = static_cast<std::uint32_t>(std::lower_bound(probe_points.begin(), probe_points.end(), found->second) - probe_points.begin());
							seed = tet_of_probe[index];
						}
						if (seed == delaunay::none)
						{
							seed = 0;
						}
					}
					const auto t = delaunay::locate(mesh, positions, centre, seed);
					start_tet[l] = t;
					if (t != delaunay::none)
					{
						seed = t;
					}
				}
			});
			auto started = 0u;
			for (const auto t : start_tet)
			{
				started += t != delaunay::none;
			}

			// ---- probe visibility: the last occluder between each face point and each probe ---------------------------
			std::vector<std::uint8_t> flagged(tet_count, 0);
			std::vector<std::array<std::uint8_t, 64>> visibility(tet_count);
			parallel_for(tet_count, [&](const std::uint32_t t, std::uint32_t)
			{
				const auto& verts = mesh.tets[t];
				float lo[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, hi[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
				for (const auto v : verts)
				{
					for (auto k = 0; k < 3; k++)
					{
						lo[k] = std::min(lo[k], static_cast<float>(positions[v][k]));
						hi[k] = std::max(hi[k], static_cast<float>(positions[v][k]));
					}
				}
				if (!occluder_bvh.any_in_box(lo, hi))
				{
					return;
				}
				auto& record = visibility[t];
				record.fill(0xFF);
				for (auto k = 0; k < 4; k++)
				{
					const auto& va = positions[verts[face_a[k]]];
					const auto& vb = positions[verts[face_b[k]]];
					const auto& vc = positions[verts[face_c[k]]];
					const auto& target = positions[verts[k]];
					auto occluded = false;
					for (auto j = 0; j <= 4; j++)
					{
						for (auto i = 0; i + j <= 4; i++)
						{
							const auto a = i / 4.0, b = j / 4.0, c = 1.0 - a - b;
							const double from[3] = { va[0] * a + vb[0] * b + vc[0] * c, va[1] * a + vb[1] * b + vc[1] * c, va[2] * a + vb[2] * b + vc[2] * c };
							const double to[3] = { target[0], target[1], target[2] };
							const auto hit = occluder_bvh.last_crossing(from, to, 1e-4, 1.0 - 1e-6);
							if (!(hit > 0.0f))
							{
								continue;
							}
							const auto value = static_cast<std::uint8_t>(std::clamp(std::lround(255.0f * (1.0f - hit)), 0l, 255l));
							record[16 * k + j * (11 - j) / 2 + i] = value;
							occluded |= value < 254;
						}
					}
					if (occluded)
					{
						flagged[t] |= static_cast<std::uint8_t>(1u << k);
					}
				}
			});

			// flagged tetrahedra first (the visibility array covers only them)
			std::vector<std::uint32_t> new_index(tet_count), old_of(tet_count);
			auto visible_count = 0u;
			for (auto t = 0u; t < tet_count; t++)
			{
				if (flagged[t])
				{
					old_of[visible_count] = t;
					new_index[t] = visible_count++;
				}
			}
			for (auto t = 0u, next = visible_count; t < tet_count; t++)
			{
				if (!flagged[t])
				{
					old_of[next] = t;
					new_index[t] = next++;
				}
			}

			// ---- SH per probe, static model samples ---------------------------------------------------------------
			auto& pd = grid.probeData;
			pd = {};
			const auto probes = static_cast<std::uint32_t>(probe_points.size());
			pd.probeCount = probes;
			pd.probes = allocator.allocate_array<zonetool::iw7::GfxSHProbeData>(probes);
			pd.probePositions = allocator.allocate_array<zonetool::iw7::GfxGpuLightGridProbePosition>(probes);
			double l0[3] = { 0.0, 0.0, 0.0 };
			for (auto i = 0u; i < probes; i++)
			{
				const auto& p = points[probe_points[i]];
				unsigned short coeffs[32];
				lightgrid_probes::encode_probe_sh(p.sh.data(), coeffs);
				std::memcpy(&pd.probes[i], coeffs, sizeof(coeffs));
				std::memcpy(pd.probePositions[i].origin, p.position, sizeof(p.position));
				for (auto c = 0; c < 3; c++)
				{
					l0[c] += p.sh[c * 9];
				}
			}

			const auto smodels = world->dpvs.smodelCount;
			std::unordered_map<std::uint32_t, float> cluster_sun;
			{
				const auto& plan = static_model_clusters::current();
				for (const auto& c : plan.clusters)
				{
					if (c.sun >= 0.0f && !c.members.empty() && c.members.front() < plan.iw7_index.size())
					{
						cluster_sun.emplace(plan.iw7_index[c.members.front()], c.sun);
					}
				}
			}
			std::vector<sh_coeffs> model_sh(smodels);
			parallel_for(smodels, [&](const std::uint32_t m, std::uint32_t)
			{
				// at its lighting origin: a merged model's brightest member (static_model_clusters)
				// a merged model's members share one sun visibility (static_model_clusters groups them by it): theirs
				const auto* p = world->dpvs.smodelInsts[m].lightingOrigin;
				const auto cluster = cluster_sun.find(m);
				probe_sh(lighting, p, cluster != cluster_sun.end() ? cluster->second : sun_visibility(p, &m), model_sh[m]);
			});
			{
				// the sun's reach (coefficient 27 above 0): a probe or static model it reaches is sunlit past the cascades
				auto probes_lit = 0u, models_lit = 0u;
				for (auto i = 0u; i < probes; i++)
				{
					probes_lit += points[probe_points[i]].sh[27] > 0.0f;
				}
				for (const auto& sh : model_sh)
				{
					models_lit += sh[27] > 0.0f;
				}
				ZONETOOL_INFO("light grid: the sun reaches %u of %u probes and %u of %u static models", probes_lit, probes, models_lit, smodels);
			}
			// a cluster's own probes (static_model_clusters::light_probes), each vertex blending four of them (level 3, as
			// Spaceland's large models), then its fade probe (the model's sample); any other model two copies of its sample
			std::vector<std::uint32_t> first_slot(smodels + 1, 0u);
			for (auto m = 0u; m < smodels; m++)
			{
				const auto& own = static_model_clusters::light_probes(m);
				first_slot[m + 1] = first_slot[m] + (own.empty() ? smodel_samples : static_cast<unsigned int>(own.size()) + 1);
			}
			pd.gpuVisibleProbesCount = first_slot[smodels];
			pd.gpuVisibleProbePositions = allocator.allocate_array<zonetool::iw7::GfxGpuLightGridProbePosition>(std::max(1u, pd.gpuVisibleProbesCount));
			pd.gpuVisibleProbesData = allocator.allocate_array<zonetool::iw7::GfxSHProbeData>(pd.gpuVisibleProbesCount + runtime_probe_slots);
			std::atomic<unsigned int> own_probes = 0, own_lit = 0;
			parallel_for(smodels, [&](const std::uint32_t m, std::uint32_t)
			{
				const auto& own = static_model_clusters::light_probes(m);
				const auto put = [&](const std::uint32_t slot, const float* origin, const sh_coeffs& sh)
				{
					unsigned short coeffs[32];
					lightgrid_probes::encode_probe_sh(sh.data(), coeffs);
					std::memcpy(pd.gpuVisibleProbePositions[slot].origin, origin, sizeof(float) * 3);
					std::memcpy(&pd.gpuVisibleProbesData[slot], coeffs, sizeof(coeffs));
				};
				const auto first = first_slot[m];
				for (auto k = 0u; k < own.size(); k++)
				{
					sh_coeffs sh;
					probe_sh(lighting, own[k].data(), sun_visibility(own[k].data(), nullptr), sh);
					put(first + k, own[k].data(), sh);
					own_lit += sh[27] >= 0.5f;
				}
				own_probes += static_cast<unsigned int>(own.size());
				const auto samples = first_slot[m + 1] - first;
				for (auto k = static_cast<std::uint32_t>(own.size()); k < samples; k++)
				{
					put(first + k, world->dpvs.smodelInsts[m].lightingOrigin, model_sh[m]);
				}
				auto& inst = world->dpvs.smodelDrawInsts[m];
				inst.unk0 = static_cast<unsigned short>(first & 0xFFFF);
				inst.unk1 = static_cast<unsigned short>(first >> 16);
				inst.unk2 = own.empty() ? 0 : 3;
				inst.unk3 = static_cast<unsigned short>(samples);
			});
			ZONETOOL_INFO("light grid: %u static model probes of their own, the sun reaching %u", own_probes.load(), own_lit.load());

			// ---- the tetrahedral grid ------------------------------------------------------------------------------
			pd.tetrahedronCount = tet_count;
			pd.tetrahedrons = allocator.allocate_array<zonetool::iw7::GfxGpuLightGridTetrahedron>(tet_count);
			pd.tetrahedronNeighbors = allocator.allocate_array<zonetool::iw7::GfxGpuLightGridTetrahedronNeighbors>(tet_count);
			for (auto n = 0u; n < tet_count; n++)
			{
				const auto t = old_of[n];
				for (auto k = 0; k < 4; k++)
				{
					pd.tetrahedrons[n].indexFlags[k] = mesh.tets[t][k] | (((flagged[t] >> k) & 1) ? 0x80000000u : 0u);
					const auto u = mesh.neighbours[t][k];
					pd.tetrahedronNeighbors[n].neighbors[k] = u == delaunay::none ? delaunay::none : new_index[u];
				}
			}
			pd.tetrahedronCountVisible = visible_count;
			pd.tetrahedronVisibility = visible_count ? allocator.allocate_array<zonetool::iw7::GfxGpuLightGridTetrahedronVisibility>(visible_count) : nullptr;
			for (auto n = 0u; n < visible_count; n++)
			{
				std::memcpy(pd.tetrahedronVisibility[n].visibility, visibility[old_of[n]].data(), 64);
			}
			pd.voxelStartTetrahedronCount = leaf_count;
			pd.voxelStartTetrahedron = allocator.allocate_array<zonetool::iw7::GfxGpuLightGridVoxelStartTetrahedron>(std::max(1u, leaf_count));
			for (auto l = 0u; l < leaf_count; l++)
			{
				pd.voxelStartTetrahedron[l].index = start_tet[l] == delaunay::none ? delaunay::none : new_index[start_tet[l]];
			}

			float fallback[28] = {};
			for (auto c = 0; c < 3; c++)
			{
				fallback[c * 9] = static_cast<float>(l0[c] / std::max(1u, probes));
			}
			fallback[27] = 1.0f;
			pd.zoneCount = 1;
			pd.zones = allocator.allocate_array<zonetool::iw7::GfxGpuLightGridZone>(1);
			auto& zone_data = pd.zones[0];
			zone_data.numProbes = probes;
			zone_data.firstProbe = 0;
			zone_data.numTetrahedrons = tet_count;
			zone_data.firstTetrahedron = 0;
			zone_data.firstVoxelTetrahedronIndex = 0;
			zone_data.numVoxelTetrahedronIndices = leaf_count;
			{
				unsigned short coeffs[32];
				lightgrid_probes::encode_probe_sh(fallback, coeffs);
				std::memcpy(zone_data.fallbackProbeData.coeffs, coeffs, sizeof(zone_data.fallbackProbeData.coeffs));
				std::memset(zone_data.fallbackProbeData.pad, 0, sizeof(zone_data.fallbackProbeData.pad));
			}

			// ---- the voxel light lists -------------------------------------------------------------------------------
			std::vector<local_light> lights;
			for (auto i = world->lastSunPrimaryLightIndex + 1; i < world->primaryLightCount; i++)
			{
				const auto& src = com->primaryLights[i];
				const auto& hull = world->frustumLights[i];
				if ((src.type != zonetool::iw7::GFX_LIGHT_TYPE_SPOT && src.type != zonetool::iw7::GFX_LIGHT_TYPE_OMNI) || !hull.vertexCount)
				{
					continue;
				}
				local_light l{};
				l.index = static_cast<unsigned short>(i);
				std::memcpy(l.origin, src.origin, sizeof(l.origin));
				const float axis[3] = { -src.dir[0], -src.dir[1], -src.dir[2] };
				const auto len = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
				l.spot = src.type == zonetool::iw7::GFX_LIGHT_TYPE_SPOT && len > 0.5f;
				for (auto k = 0; k < 3; k++)
				{
					l.axis[k] = len > 0.0f ? axis[k] / len : 0.0f;
					l.hull_min[k] = FLT_MAX;
					l.hull_max[k] = -FLT_MAX;
				}
				l.radius = src.radius;
				l.half_angle = std::acos(std::clamp(src.cosHalfFovOuter, -1.0f, 1.0f));
				for (auto v = 0u; v < hull.vertexCount; v++)
				{
					float p[3];
					std::memcpy(p, &hull.vertices[32 * v], sizeof(p));
					for (auto k = 0; k < 3; k++)
					{
						l.hull_min[k] = std::min(l.hull_min[k], p[k]);
						l.hull_max[k] = std::max(l.hull_max[k], p[k]);
					}
				}
				lights.push_back(l);
			}

			// the lights that can reach each root node, then each of its leaves
			std::unordered_map<std::uint64_t, std::uint32_t> root_index;
			for (auto r = 0u; r < root_nodes.size(); r++)
			{
				root_index.emplace(root_key(root_nodes[r][0], root_nodes[r][1], root_nodes[r][2]), r);
			}
			std::vector<std::vector<std::uint32_t>> root_lights(root_nodes.size());
			for (auto i = 0u; i < lights.size(); i++)
			{
				const auto& l = lights[i];
				int r0[3], r1[3];
				for (auto k = 0; k < 3; k++)
				{
					r0[k] = static_cast<int>(std::floor(l.hull_min[k] / (cell * cells_per_root)));
					r1[k] = static_cast<int>(std::floor(l.hull_max[k] / (cell * cells_per_root)));
				}
				for (auto x = r0[0]; x <= r1[0]; x++)
				{
					for (auto y = r0[1]; y <= r1[1]; y++)
					{
						for (auto z = r0[2]; z <= r1[2]; z++)
						{
							const auto found = root_index.find(root_key(x, y, z));
							if (found != root_index.end())
							{
								root_lights[found->second].push_back(i);
							}
						}
					}
				}
			}
			// each lit leaf's list: [spots << 7 | omnis, spots, omnis], interned. IW7 addresses them with 16 bits; a map
			// whose exact lists do not fit gives every leaf the list of the aligned block it lies in (the lights reaching
			// the block: all of the leaf's and some that add nothing to it), the block doubling until they fit
			std::vector<std::vector<unsigned short>> leaf_hits(leaf_count);
			std::vector<unsigned short> light_list;
			std::vector<unsigned short> leaf_address(leaf_count, voxel_list_empty);
			std::map<std::vector<unsigned short>, unsigned short> interned;
			auto lit_leaves = 0u;
			auto most_spots = 0u, most_omnis = 0u;
			auto block = 0.0f;
			for (;;)
			{
				std::atomic<bool> over_cap = false;
				parallel_for(static_cast<std::uint32_t>(root_nodes.size()), [&](const std::uint32_t r, std::uint32_t)
				{
					const auto& candidates = root_lights[r];
					std::vector<unsigned short> spots, omnis;
					for (auto l = root_first_leaf[r]; l < root_first_leaf[r + 1]; l++)
					{
						leaf_hits[l].clear();
						if (candidates.empty())
						{
							continue;
						}
						const auto& leaf = leaves[l];
						const auto size = static_cast<float>(leaf.size) * cell;
						float lo[3] = { leaf.cell[0] * cell, leaf.cell[1] * cell, leaf.cell[2] * cell };
						float hi[3] = { lo[0] + size, lo[1] + size, lo[2] + size };
						if (block > 0.0f)
						{
							for (auto k = 0; k < 3; k++)
							{
								lo[k] = std::floor(lo[k] / block) * block;
								hi[k] = std::max(std::ceil(hi[k] / block) * block, lo[k] + block);
							}
						}
						spots.clear();
						omnis.clear();
						for (const auto i : candidates)
						{
							if (reaches(lights[i], lo, hi))
							{
								(lights[i].spot ? spots : omnis).push_back(lights[i].index);
							}
						}
						if (spots.size() > 0x1FF || omnis.size() > 0x7F)
						{
							if (block == 0.0f)
							{
								throw std::runtime_error(utils::string::va("light grid: voxel %u is reached by %zu spot and %zu omni lights", l, spots.size(), omnis.size()));
							}
							over_cap = true;
							continue;
						}
						if (spots.empty() && omnis.empty())
						{
							continue;
						}
						auto& list = leaf_hits[l];
						list.push_back(static_cast<unsigned short>((spots.size() << 7) | omnis.size()));
						list.insert(list.end(), spots.begin(), spots.end());
						list.insert(list.end(), omnis.begin(), omnis.end());
					}
				});
				if (over_cap)
				{
					ZONETOOL_FATAL("light grid: the voxel light lists outgrow 16-bit addresses and %g unit blocks are reached by more "
						"than 511 spot or 127 omni lights", block);
				}
				light_list = { 16384, 1, 0 };
				std::fill(leaf_address.begin(), leaf_address.end(), voxel_list_empty);
				interned.clear();
				lit_leaves = most_spots = most_omnis = 0;
				auto fits = true;
				for (auto l = 0u; l < leaf_count; l++)
				{
					const auto& list = leaf_hits[l];
					if (list.empty())
					{
						continue;
					}
					lit_leaves++;
					most_spots = std::max(most_spots, static_cast<unsigned int>(list[0] >> 7));
					most_omnis = std::max(most_omnis, static_cast<unsigned int>(list[0] & 0x7F));
					auto found = interned.find(list);
					if (found == interned.end())
					{
						if (light_list.size() + list.size() > 0xFFFF)
						{
							fits = false;
							break;
						}
						found = interned.emplace(list, static_cast<unsigned short>(light_list.size())).first;
						light_list.insert(light_list.end(), list.begin(), list.end());
					}
					leaf_address[l] = found->second;
				}
				if (fits)
				{
					break;
				}
				block = block == 0.0f ? 256.0f : block * 2.0f;
			}
			if (block > 0.0f)
			{
				ZONETOOL_INFO("light grid: the exact voxel light lists outgrow 16-bit addresses; each voxel lists the lights of its %g unit block", block);
			}

			world->voxelTreeCount = 1;
			world->voxelTree = allocator.allocate_array<zonetool::iw7::GfxVoxelTree>(1);
			auto& tree = world->voxelTree[0];
			for (auto k = 0; k < 3; k++)
			{
				tree.zoneBound.midPoint[k] = world->bounds.midPoint[k];
				tree.zoneBound.halfSize[k] = std::max(0.0f, world->bounds.halfSize[k] - 1.0f);
			}
			tree.voxelTopDownViewNodeCount = static_cast<int>(top_down.size());
			tree.voxelInternalNodeCount = static_cast<int>(internal.size());
			tree.voxelLeafNodeCount = static_cast<int>(leaf_count);
			tree.lightListArraySize = static_cast<int>(light_list.size());
			tree.voxelTreeHeader = allocator.allocate<zonetool::iw7::GfxVoxelTreeHeader>();
			for (auto k = 0; k < 3; k++)
			{
				tree.voxelTreeHeader->rootNodeDimension[k] = dims[k];
				tree.voxelTreeHeader->boundMin[k] = origin[k];
				tree.voxelTreeHeader->boundMax[k] = origin[k] + static_cast<float>(dims[k]) * cell * cells_per_root;
			}
			tree.voxelTreeHeader->nodeCoordBitShift[0] = 9;
			tree.voxelTreeHeader->nodeCoordBitShift[1] = 7;
			tree.voxelTreeHeader->nodeCoordBitShift[2] = 5;
			tree.voxelTopDownViewNodeArray = allocator.allocate_array<zonetool::iw7::GfxVoxelTopDownViewNode>(top_down.size());
			std::memcpy(tree.voxelTopDownViewNodeArray, top_down.data(), sizeof(zonetool::iw7::GfxVoxelTopDownViewNode) * top_down.size());
			tree.voxelInternalNodeArray = allocator.allocate_array<zonetool::iw7::GfxVoxelInternalNode>(internal.size());
			std::memcpy(tree.voxelInternalNodeArray, internal.data(), sizeof(zonetool::iw7::GfxVoxelInternalNode) * internal.size());
			tree.voxelLeafNodeArray = allocator.allocate_array<zonetool::iw7::GfxVoxelLeafNode>(std::max(1u, leaf_count));
			std::memcpy(tree.voxelLeafNodeArray, leaf_address.data(), sizeof(unsigned short) * leaf_count);
			tree.lightListArray = allocator.allocate_array<unsigned short>(light_list.size());
			std::memcpy(tree.lightListArray, light_list.data(), sizeof(unsigned short) * light_list.size());
			tree.voxelInternalNodeDynamicLightList = allocator.allocate_array<unsigned int>(2 * std::max<std::size_t>(1, internal.size()));

			const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
			ZONETOOL_INFO("light grid: %u probes in %zu roots (%zu detailed BO3 probes; splits 512:%u 256:%u 128:%u 64:%u), %u tetrahedra "
				"(%u with visibility, %zu occluding triangles), %u leaves in %dx%dx%d roots (%u fine, %u coarse; %u with a start tetrahedron, "
				"%u lit by %zu local lights, %zu distinct lists in %zu entries, at most %u spots %u omnis), %u static model samples, %.0f s",
				probes, roots.size(), fine_boxes.size(), splits[0], splits[1], splits[2], splits[3], tet_count, visible_count, occluder_count,
				leaf_count, dims[0], dims[1], dims[2], fine_roots, coarse_roots, started, lit_leaves, lights.size(), interned.size(),
				light_list.size(), most_spots, most_omnis, pd.gpuVisibleProbesCount, seconds);
		}
	}
}
