#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "world_lights.hpp"

#include "comworld.hpp"
#include "world_sky.hpp"
#include "static_model_clusters.hpp"
#include "../convex.hpp"
#include "../parallel.hpp"

#include <utils/string.hpp>

// IW7 (stock data):
// * frustumLights[i]: the hull light i is binned into clusters with, 32-byte vertices (position, then zeros),
//   triangles clockwise seen from outside. The IW5 -> IW7 port builds a cone (spot) or sphere (omni) around the
//   light's radius and clips it to where the light may reach; BO3 bounds every light by its wldCullMin / Max box,
//   so the hull is clipped to that box.
// * lightViewFrustums[i]: the shadow frustum of a spot light that can use a shadow map (IW5 port).
// * shadowGeomOptimized[i]: the light's casters, ascending surface indices from the opaque range (never sorted
//   slots) and static models, all within the light's reach.
// * lightAABB: a tree over the lights with canUseShadowMap and needsDynamicShadows; a leaf's box is its lights'
//   hull bounds, firstChild | 0x8000 marks a leaf and indexes lightArray.
// * lightLists: count-prefixed runs of ascending light indices, [0] the empty list; a static surface's run
//   offset is the u16 at GfxSurface +36 (unk1, unk2), a static model's is GfxStaticModelDrawInst::unk11. The
//   offset tables themselves are not part of the zone.
// The sun is left out of every list.

namespace zonetool::t7
{
	namespace converter::iw7::world_lights
	{
		namespace
		{
			constexpr float pi = 3.14159265358979f;
			constexpr unsigned int proxy_segments = 8;
			constexpr unsigned int spot_rings = 3;
			constexpr unsigned int omni_rings = 5;
			// spots wider than 80 degrees get the sphere (IW5 port)
			constexpr float wide_spot_cutoff = 1.3962634f;
			constexpr float proxy_margin = 1.20f;
			constexpr float view_frustum_near_fraction = 0.01f;
			constexpr unsigned int aabb_leaf_lights = 4;

			using vec3 = std::array<float, 3>;

			float dot(const float* a, const float* b)
			{
				return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
			}

			vec3 cross(const float* a, const float* b)
			{
				return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
			}

			vec3 normalized(vec3 v)
			{
				const auto len = std::sqrt(dot(v.data(), v.data()));
				if (len > 0.0f)
				{
					for (auto& c : v)
					{
						c /= len;
					}
				}
				return v;
			}

			// ---- proxy hulls (IW5 -> IW7 port) -------------------------------------------------------

			struct proxy
			{
				std::vector<vec3> vertices;
				std::vector<std::array<unsigned int, 3>> triangles; // counter-clockwise seen from outside
			};

			void proxy_basis(const vec3& axis, vec3& u, vec3& v)
			{
				const vec3 helper = std::fabs(axis[2]) > 0.9f ? vec3{ 1.0f, 0.0f, 0.0f } : vec3{ 0.0f, 0.0f, 1.0f };
				u = normalized(cross(helper.data(), axis.data()));
				v = normalized(cross(axis.data(), u.data()));
			}

			// the facets must stay outside the true sphere / cone
			float circumscribe(const float polar_gap)
			{
				return 1.0f / (std::cos(pi / proxy_segments) * std::cos(polar_gap * 0.5f));
			}

			std::vector<unsigned int> ring(proxy& p, const float origin[3], const vec3& axis, const vec3& u, const vec3& v,
				const float theta, const float dist)
			{
				std::vector<unsigned int> out;
				for (auto s = 0u; s < proxy_segments; s++)
				{
					const auto phi = 2.0f * pi * static_cast<float>(s) / proxy_segments;
					vec3 q;
					for (auto c = 0; c < 3; c++)
					{
						q[c] = origin[c] + (axis[c] * std::cos(theta) + (u[c] * std::cos(phi) + v[c] * std::sin(phi)) * std::sin(theta)) * dist;
					}
					out.push_back(static_cast<unsigned int>(p.vertices.size()));
					p.vertices.push_back(q);
				}
				return out;
			}

			void bridge(proxy& p, const std::vector<unsigned int>& inner, const std::vector<unsigned int>& outer)
			{
				for (auto s = 0u; s < proxy_segments; s++)
				{
					const auto n = (s + 1) % proxy_segments;
					p.triangles.push_back({ inner[s], outer[s], outer[n] });
					p.triangles.push_back({ inner[s], outer[n], inner[n] });
				}
			}

			void cap(proxy& p, const unsigned int pole, const std::vector<unsigned int>& r, const bool flip)
			{
				for (auto s = 0u; s < proxy_segments; s++)
				{
					const auto n = (s + 1) % proxy_segments;
					p.triangles.push_back(flip ? std::array<unsigned int, 3>{ pole, r[n], r[s] } : std::array<unsigned int, 3>{ pole, r[s], r[n] });
				}
			}

			proxy spot_proxy(const float origin[3], const vec3& axis, const float half_angle, const float range)
			{
				proxy p;
				vec3 u, v;
				proxy_basis(axis, u, v);
				const auto step = std::atan(std::tan(half_angle) / std::cos(pi / proxy_segments)) / spot_rings;
				const auto dist = range * circumscribe(step) * proxy_margin;
				const auto apex = static_cast<unsigned int>(p.vertices.size());
				p.vertices.push_back({ origin[0], origin[1], origin[2] });
				const auto tip = static_cast<unsigned int>(p.vertices.size());
				p.vertices.push_back({ origin[0] + axis[0] * dist, origin[1] + axis[1] * dist, origin[2] + axis[2] * dist });
				std::vector<std::vector<unsigned int>> rings;
				for (auto k = 1u; k <= spot_rings; k++)
				{
					rings.push_back(ring(p, origin, axis, u, v, step * static_cast<float>(k), dist));
				}
				cap(p, tip, rings.front(), false);
				for (auto k = 0u; k + 1 < rings.size(); k++)
				{
					bridge(p, rings[k], rings[k + 1]);
				}
				cap(p, apex, rings.back(), true);
				return p;
			}

			proxy omni_proxy(const float origin[3], const float range)
			{
				proxy p;
				const vec3 axis{ 0.0f, 0.0f, 1.0f };
				vec3 u, v;
				proxy_basis(axis, u, v);
				const auto step = pi / (omni_rings + 1);
				const auto dist = range * circumscribe(step) * proxy_margin;
				const auto north = static_cast<unsigned int>(p.vertices.size());
				p.vertices.push_back({ origin[0], origin[1], origin[2] + dist });
				const auto south = static_cast<unsigned int>(p.vertices.size());
				p.vertices.push_back({ origin[0], origin[1], origin[2] - dist });
				std::vector<std::vector<unsigned int>> rings;
				for (auto k = 1u; k <= omni_rings; k++)
				{
					rings.push_back(ring(p, origin, axis, u, v, step * static_cast<float>(k), dist));
				}
				cap(p, north, rings.front(), false);
				for (auto k = 0u; k + 1 < rings.size(); k++)
				{
					bridge(p, rings[k], rings[k + 1]);
				}
				cap(p, south, rings.back(), true);
				return p;
			}

			// the proxy's facets and the box as planes, rebuilt as one convex hull
			bool clip_to_box(proxy& p, const float lo[3], const float hi[3])
			{
				vec3 centre{};
				for (const auto& v : p.vertices)
				{
					for (auto k = 0; k < 3; k++)
					{
						centre[k] += v[k] / static_cast<float>(p.vertices.size());
					}
				}
				std::vector<convex::plane> planes;
				const auto add = [&](vec3 n, float d)
				{
					for (const auto& q : planes)
					{
						if (dot(q.data(), n.data()) > 0.99999f && std::fabs(q[3] - d) < 0.01f)
						{
							return;
						}
					}
					planes.push_back({ n[0], n[1], n[2], d });
				};
				for (const auto& t : p.triangles)
				{
					const auto& a = p.vertices[t[0]];
					const auto& b = p.vertices[t[1]];
					const auto& c = p.vertices[t[2]];
					const vec3 ab{ b[0] - a[0], b[1] - a[1], b[2] - a[2] }, ac{ c[0] - a[0], c[1] - a[1], c[2] - a[2] };
					auto n = cross(ab.data(), ac.data());
					const auto len = std::sqrt(dot(n.data(), n.data()));
					if (!(len > 1e-6f))
					{
						continue;
					}
					for (auto& k : n)
					{
						k /= len;
					}
					auto d = dot(n.data(), a.data());
					if (dot(n.data(), centre.data()) > d)
					{
						for (auto& k : n)
						{
							k = -k;
						}
						d = -d;
					}
					add(n, d);
				}
				for (auto k = 0; k < 3; k++)
				{
					vec3 n{};
					n[k] = 1.0f;
					add(n, hi[k]);
					n[k] = -1.0f;
					add(n, -lo[k]);
				}
				convex::hull hull;
				if (!convex::from_planes(planes, hull))
				{
					return false;
				}
				proxy clipped;
				clipped.vertices = hull.vertices;
				for (const auto& f : hull.faces)
				{
					for (auto k = 1u; k + 1 < f.indices.size(); k++)
					{
						clipped.triangles.push_back({ f.indices[0], f.indices[k], f.indices[k + 1] });
					}
				}
				p = std::move(clipped);
				return true;
			}

			struct light_info
			{
				bool local = false; // spot or omni with a radius
				bool spot = false;
				vec3 origin{};
				vec3 axis{}; // the direction a spot shines
				float radius = 0.0f;
				float half_angle = 0.0f;
				vec3 hull_min{}, hull_max{};
			};

			// the IW5 port's reach test: the object's bounding sphere against the light's sphere and cone
			bool reaches(const light_info& l, const zonetool::iw7::Bounds& b)
			{
				const auto r = std::sqrt(dot(b.halfSize, b.halfSize));
				const float delta[3] = { b.midPoint[0] - l.origin[0], b.midPoint[1] - l.origin[1], b.midPoint[2] - l.origin[2] };
				const auto dist = std::sqrt(dot(delta, delta));
				if (dist - r > l.radius)
				{
					return false;
				}
				// and the light's hull box (BO3's cull box)
				for (auto k = 0; k < 3; k++)
				{
					if (b.midPoint[k] + b.halfSize[k] < l.hull_min[k] || b.midPoint[k] - b.halfSize[k] > l.hull_max[k])
					{
						return false;
					}
				}
				if (l.spot && dist > 0.001f)
				{
					const auto theta = std::acos(std::clamp(dot(delta, l.axis.data()) / dist, -1.0f, 1.0f));
					if (theta > l.half_angle + std::asin(std::min(1.0f, r / dist)))
					{
						return false;
					}
				}
				return true;
			}

			// ---- light view frustums (IW5 port) --------------------------------------------------------

			bool view_frustum(const zonetool::iw7::ComPrimaryLight& src, zonetool::iw7::GfxLightViewFrustum& dest, utils::memory::allocator& allocator)
			{
				auto axis = normalized({ -src.dir[0], -src.dir[1], -src.dir[2] });
				if (!(std::sqrt(dot(axis.data(), axis.data())) > 0.5f))
				{
					return false;
				}
				const auto axial = std::fabs(axis[2]) > 0.9f;
				const vec3 helper{ axial ? 1.0f : 0.0f, 0.0f, axial ? 0.0f : 1.0f };
				const auto hd = dot(helper.data(), axis.data());
				auto right = normalized({ helper[0] - axis[0] * hd, helper[1] - axis[1] * hd, helper[2] - axis[2] * hd });
				const auto up = cross(axis.data(), right.data());
				const auto cos_outer = std::max(0.017452f, src.cosHalfFovOuter);
				const auto tan_half = std::sqrt(std::max(0.0f, 1.0f - cos_outer * cos_outer)) / cos_outer;
				const auto near_dist = std::max(1.0f, src.radius * view_frustum_near_fraction);
				const auto far_dist = std::max(near_dist + 1.0f, src.radius);

				vec3 verts[8];
				for (auto slice = 0; slice < 2; slice++)
				{
					const auto depth = slice ? far_dist : near_dist;
					const auto extent = depth * tan_half;
					for (auto corner = 0; corner < 4; corner++)
					{
						const auto sx = (corner == 0 || corner == 3) ? -extent : extent;
						const auto sy = corner < 2 ? -extent : extent;
						for (auto k = 0; k < 3; k++)
						{
							verts[slice * 4 + corner][k] = src.origin[k] + axis[k] * depth + right[k] * sx + up[k] * sy;
						}
					}
				}
				static constexpr unsigned short quads[6][4] = { { 0, 1, 2, 3 }, { 4, 5, 6, 7 }, { 0, 1, 5, 4 }, { 1, 2, 6, 5 }, { 2, 3, 7, 6 }, { 3, 0, 4, 7 } };
				std::vector<unsigned short> indices;
				for (const auto& q : quads)
				{
					indices.insert(indices.end(), { q[0], q[1], q[2], q[0], q[2], q[3] });
				}
				auto volume = 0.0f;
				for (auto t = 0u; t + 2 < indices.size(); t += 3)
				{
					const auto c = cross(verts[indices[t + 1]].data(), verts[indices[t + 2]].data());
					volume += dot(verts[indices[t]].data(), c.data());
				}
				if (volume < 0.0f)
				{
					for (auto t = 0u; t + 2 < indices.size(); t += 3)
					{
						std::swap(indices[t + 1], indices[t + 2]);
					}
				}
				vec3 centre{};
				for (const auto& v : verts)
				{
					for (auto k = 0; k < 3; k++)
					{
						centre[k] += v[k] * 0.125f;
					}
				}
				dest.vertexCount = 8;
				dest.vertices = allocator.allocate_array<std::remove_pointer_t<decltype(dest.vertices)>>(8);
				std::memcpy(dest.vertices, verts, sizeof(verts));
				dest.indexCount = static_cast<unsigned int>(indices.size());
				dest.indices = allocator.allocate_array<unsigned short>(indices.size());
				std::memcpy(dest.indices, indices.data(), sizeof(unsigned short) * indices.size());
				dest.planeCount = 6;
				dest.planes = allocator.allocate_array<std::remove_pointer_t<decltype(dest.planes)>>(6);
				for (auto f = 0; f < 6; f++)
				{
					const auto& a = verts[quads[f][0]];
					const auto& b = verts[quads[f][1]];
					const auto& c = verts[quads[f][2]];
					const vec3 ab{ b[0] - a[0], b[1] - a[1], b[2] - a[2] }, ac{ c[0] - a[0], c[1] - a[1], c[2] - a[2] };
					auto n = normalized(cross(ab.data(), ac.data()));
					auto d = -dot(n.data(), a.data());
					if (dot(n.data(), centre.data()) + d < 0.0f)
					{
						for (auto& k : n)
						{
							k = -k;
						}
						d = -d;
					}
					dest.planes[f][0] = n[0];
					dest.planes[f][1] = n[1];
					dest.planes[f][2] = n[2];
					dest.planes[f][3] = d;
				}
				return true;
			}

			// ---- the dynamic shadow light tree -------------------------------------------------------------

			void build_light_tree(const std::vector<light_info>& lights, const std::vector<unsigned int>& members, zonetool::iw7::GfxWorld* world,
				utils::memory::allocator& allocator)
			{
				auto& aabb = world->lightAABB;
				if (members.empty())
				{
					aabb = {};
					return;
				}
				struct node
				{
					vec3 lo, hi;
					unsigned short first = 0, count = 0;
				};
				std::vector<node> nodes;
				std::vector<unsigned short> order;
				const auto bounds_of = [&](const std::vector<unsigned int>& list, node& n)
				{
					n.lo = { FLT_MAX, FLT_MAX, FLT_MAX };
					n.hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
					for (const auto i : list)
					{
						for (auto k = 0; k < 3; k++)
						{
							n.lo[k] = std::min(n.lo[k], lights[i].hull_min[k]);
							n.hi[k] = std::max(n.hi[k], lights[i].hull_max[k]);
						}
					}
				};
				// breadth first: a node's children are consecutive
				struct pending
				{
					std::vector<unsigned int> list;
					std::size_t node;
				};
				std::deque<pending> queue;
				nodes.emplace_back();
				queue.push_back({ members, 0 });
				while (!queue.empty())
				{
					auto item = std::move(queue.front());
					queue.pop_front();
					bounds_of(item.list, nodes[item.node]);
					if (item.list.size() <= aabb_leaf_lights)
					{
						nodes[item.node].first = static_cast<unsigned short>(0x8000 | order.size());
						nodes[item.node].count = static_cast<unsigned short>(item.list.size());
						for (const auto i : item.list)
						{
							order.push_back(static_cast<unsigned short>(i));
						}
						continue;
					}
					// split at the median of the hull centres along the widest axis
					const auto& n = nodes[item.node];
					auto axis = 0;
					for (auto k = 1; k < 3; k++)
					{
						if (n.hi[k] - n.lo[k] > n.hi[axis] - n.lo[axis])
						{
							axis = k;
						}
					}
					std::sort(item.list.begin(), item.list.end(), [&](const unsigned int a, const unsigned int b)
					{
						return lights[a].hull_min[axis] + lights[a].hull_max[axis] < lights[b].hull_min[axis] + lights[b].hull_max[axis];
					});
					const auto half = item.list.size() / 2;
					const auto first = nodes.size();
					nodes[item.node].first = static_cast<unsigned short>(first);
					nodes[item.node].count = 2;
					nodes.emplace_back();
					nodes.emplace_back();
					queue.push_back({ { item.list.begin(), item.list.begin() + half }, first });
					queue.push_back({ { item.list.begin() + half, item.list.end() }, first + 1 });
				}
				if (nodes.size() > 0x7FFF || order.size() > 0x7FFF)
				{
					ZONETOOL_FATAL("world lights: %zu tree nodes / %zu lights do not fit the light AABB tree", nodes.size(), order.size());
				}
				aabb.nodeCount = static_cast<unsigned short>(nodes.size());
				aabb.lightCount = static_cast<unsigned short>(order.size());
				aabb.nodeArray = allocator.allocate_array<zonetool::iw7::GfxLightAABBNode>(nodes.size());
				for (auto i = 0u; i < nodes.size(); i++)
				{
					auto& dst = aabb.nodeArray[i];
					for (auto k = 0; k < 3; k++)
					{
						dst.bound.midPoint[k] = (nodes[i].lo[k] + nodes[i].hi[k]) * 0.5f;
						dst.bound.halfSize[k] = (nodes[i].hi[k] - nodes[i].lo[k]) * 0.5f;
					}
					dst.firstChild = nodes[i].first;
					dst.childCount = nodes[i].count;
				}
				aabb.lightArray = allocator.allocate_array<unsigned short>(order.size());
				std::memcpy(aabb.lightArray, order.data(), sizeof(unsigned short) * order.size());
			}
		}

		namespace
		{
			std::vector<char> sun_only_casters;
		}

		void set_sun_only_casters(std::vector<char> surfaces)
		{
			sun_only_casters = std::move(surfaces);
		}

		void build(zonetool::iw7::GfxWorld* world, utils::memory::allocator& allocator)
		{
			const auto* com = comworld::converted();
			const auto* source = comworld::source();
			if (!com || !source)
			{
				ZONETOOL_FATAL("world lights: the ComWorld has to be converted first");
			}
			const auto light_count = world->primaryLightCount;

			// the BO3 light each IW7 light came from
			std::vector<int> source_of(light_count, -1);
			for (auto i = 0u; i < source->primaryLightCount; i++)
			{
				const auto k = comworld::remap_light(i);
				if (k && k < light_count)
				{
					source_of[k] = static_cast<int>(i);
				}
			}

			// ---- hulls ------------------------------------------------------------------------------------
			std::vector<light_info> lights(light_count);
			world->frustumLights = allocator.allocate_array<zonetool::iw7::GfxFrustumLights>(light_count);
			auto hulls = 0u, unclipped = 0u, spots_as_spheres = 0u;
			for (auto i = world->lastSunPrimaryLightIndex + 1; i < light_count; i++)
			{
				const auto& src = com->primaryLights[i];
				auto& l = lights[i];
				if ((src.type != zonetool::iw7::GFX_LIGHT_TYPE_SPOT && src.type != zonetool::iw7::GFX_LIGHT_TYPE_OMNI) || !(src.radius > 0.0f))
				{
					continue;
				}
				if (source_of[i] < 0)
				{
					ZONETOOL_FATAL("world lights: IW7 light %u has no BO3 source light", i);
				}
				const auto& cfg = source->primaryLights[source_of[i]].config;
				l.local = true;
				l.origin = { src.origin[0], src.origin[1], src.origin[2] };
				l.radius = src.radius;
				l.axis = normalized({ -src.dir[0], -src.dir[1], -src.dir[2] });
				l.half_angle = std::acos(std::clamp(src.cosHalfFovOuter, -1.0f, 1.0f));
				l.spot = src.type == zonetool::iw7::GFX_LIGHT_TYPE_SPOT && dot(l.axis.data(), l.axis.data()) > 0.5f;

				auto p = l.spot && l.half_angle < wide_spot_cutoff ? spot_proxy(src.origin, l.axis, l.half_angle, l.radius)
					: omni_proxy(src.origin, l.radius);
				spots_as_spheres += l.spot && l.half_angle >= wide_spot_cutoff;
				if (!clip_to_box(p, cfg.wldCullMin, cfg.wldCullMax))
				{
					unclipped++;
				}

				auto& dest = world->frustumLights[i];
				dest.vertexCount = static_cast<unsigned int>(p.vertices.size());
				dest.vertices = allocator.allocate_array<char>(32 * p.vertices.size());
				l.hull_min = { FLT_MAX, FLT_MAX, FLT_MAX };
				l.hull_max = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
				for (auto v = 0u; v < p.vertices.size(); v++)
				{
					std::memcpy(&dest.vertices[32 * v], p.vertices[v].data(), sizeof(float) * 3);
					for (auto k = 0; k < 3; k++)
					{
						l.hull_min[k] = std::min(l.hull_min[k], p.vertices[v][k]);
						l.hull_max[k] = std::max(l.hull_max[k], p.vertices[v][k]);
					}
				}
				// clockwise seen from outside, as stock
				dest.indexCount = static_cast<unsigned int>(p.triangles.size() * 3);
				dest.indices = allocator.allocate_array<unsigned short>(p.triangles.size() * 3);
				for (auto t = 0u; t < p.triangles.size(); t++)
				{
					dest.indices[t * 3 + 0] = static_cast<unsigned short>(p.triangles[t][0]);
					dest.indices[t * 3 + 1] = static_cast<unsigned short>(p.triangles[t][2]);
					dest.indices[t * 3 + 2] = static_cast<unsigned short>(p.triangles[t][1]);
				}
				hulls++;
			}
			if (unclipped)
			{
				ZONETOOL_FATAL("world lights: %u light hulls do not reach their BO3 cull box", unclipped);
			}

			// ---- shadow frustums --------------------------------------------------------------------------
			world->lightViewFrustums = allocator.allocate_array<zonetool::iw7::GfxLightViewFrustum>(light_count);
			auto frustums = 0u;
			for (auto i = world->lastSunPrimaryLightIndex + 1; i < light_count; i++)
			{
				const auto& src = com->primaryLights[i];
				if (src.type == zonetool::iw7::GFX_LIGHT_TYPE_SPOT && src.canUseShadowMap && src.radius > 0.0f
					&& src.cosHalfFovOuter > 0.0f && src.cosHalfFovOuter < 1.0f)
				{
					frustums += view_frustum(src, world->lightViewFrustums[i], allocator);
				}
			}

			// ---- shadow casters ----------------------------------------------------------------------------
			const auto opaque_begin = world->dpvs.litOpaqueSurfsBegin, opaque_end = world->dpvs.litOpaqueSurfsEnd;
			world->shadowGeomOptimized = allocator.allocate_array<zonetool::iw7::GfxShadowGeometry>(light_count);
			std::vector<std::vector<unsigned int>> caster_surfaces(light_count);
			std::vector<std::vector<unsigned short>> caster_models(light_count);
			parallel_for(light_count, [&](const std::uint32_t i, std::uint32_t)
			{
				const auto& l = lights[i];
				if (!l.local)
				{
					return;
				}
				for (auto s = opaque_begin; s < opaque_end; s++)
				{
					if ((world->dpvs.surfaces[s].flags & 1) && !(s < sun_only_casters.size() && sun_only_casters[s])
						&& reaches(l, world->dpvs.surfacesBounds[s].bounds))
					{
						caster_surfaces[i].push_back(s);
					}
				}
				for (auto m = 0u; m < world->dpvs.smodelCount; m++)
				{
					if (!(world->dpvs.smodelDrawInsts[m].flags & zonetool::iw7::STATIC_MODEL_FLAG_NO_CAST_SHADOW)
						&& reaches(l, world->dpvs.smodelInsts[m].bounds))
					{
						caster_models[i].push_back(static_cast<unsigned short>(m));
					}
				}
			});
			// ZT_LIGHT_CASTERS="x y z": every shadow caster of the light at that origin (surface bounds and material, static
			// model bounds), then exit
			if (const auto* at = std::getenv("ZT_LIGHT_CASTERS"))
			{
				float p[3]{};
				std::sscanf(at, "%f %f %f", &p[0], &p[1], &p[2]);
				for (auto i = 0u; i < light_count; i++)
				{
					const auto& src = com->primaryLights[i];
					const float d[3] = { src.origin[0] - p[0], src.origin[1] - p[1], src.origin[2] - p[2] };
					if (dot(d, d) > 4.0f)
					{
						continue;
					}
					ZONETOOL_INFO("casters of light %u: %zu surfaces, %zu models, shadow %d dynamic %d", i, caster_surfaces[i].size(),
						caster_models[i].size(), src.canUseShadowMap, src.needsDynamicShadows);
					for (const auto s : caster_surfaces[i])
					{
						const auto& b = world->dpvs.surfacesBounds[s].bounds;
						const auto* m = world->dpvs.surfaces[s].material;
						ZONETOOL_INFO("caster surface %u: mid %.1f %.1f %.1f half %.1f %.1f %.1f %s", s, b.midPoint[0], b.midPoint[1], b.midPoint[2],
							b.halfSize[0], b.halfSize[1], b.halfSize[2], m && m->name ? m->name : "?");
					}
					for (const auto m : caster_models[i])
					{
						const auto& b = world->dpvs.smodelInsts[m].bounds;
						ZONETOOL_INFO("caster model %u: mid %.1f %.1f %.1f half %.1f %.1f %.1f", m, b.midPoint[0], b.midPoint[1], b.midPoint[2],
							b.halfSize[0], b.halfSize[1], b.halfSize[2]);
					}
				}
				std::fflush(stdout);
				std::_Exit(0);
			}

			std::size_t caster_total = 0;
			for (auto i = 0u; i < light_count; i++)
			{
				if (caster_surfaces[i].size() > 0xFFFF || caster_models[i].size() > 0xFFFF)
				{
					ZONETOOL_FATAL("world lights: light %u has %zu caster surfaces and %zu caster models", i, caster_surfaces[i].size(),
						caster_models[i].size());
				}
				auto& dst = world->shadowGeomOptimized[i];
				dst.surfaceCount = static_cast<unsigned short>(caster_surfaces[i].size());
				dst.smodelCount = static_cast<unsigned short>(caster_models[i].size());
				if (dst.surfaceCount)
				{
					dst.sortedSurfIndex = allocator.allocate_array<unsigned int>(dst.surfaceCount);
					std::memcpy(dst.sortedSurfIndex, caster_surfaces[i].data(), sizeof(unsigned int) * dst.surfaceCount);
				}
				if (dst.smodelCount)
				{
					dst.smodelIndex = allocator.allocate_array<unsigned short>(dst.smodelCount);
					std::memcpy(dst.smodelIndex, caster_models[i].data(), sizeof(unsigned short) * dst.smodelCount);
				}
				caster_total += caster_surfaces[i].size() + caster_models[i].size();
			}

			// ---- the tree of lights with dynamic shadows ---------------------------------------------------
			std::vector<unsigned int> dynamic;
			for (auto i = 0u; i < light_count; i++)
			{
				const auto& src = com->primaryLights[i];
				if (lights[i].local && src.canUseShadowMap && src.needsDynamicShadows)
				{
					dynamic.push_back(i);
				}
			}
			build_light_tree(lights, dynamic, world, allocator);

			// ---- light lists -----------------------------------------------------------------------------------
			std::vector<unsigned short> lists{ 0 };
			std::map<std::vector<unsigned short>, unsigned int> interned;
			const auto intern = [&](std::vector<unsigned short> hits) -> unsigned int
			{
				if (hits.empty())
				{
					return 0;
				}
				const auto found = interned.find(hits);
				if (found != interned.end())
				{
					return found->second;
				}
				const auto offset = static_cast<unsigned int>(lists.size());
				lists.push_back(static_cast<unsigned short>(hits.size()));
				lists.insert(lists.end(), hits.begin(), hits.end());
				interned.emplace(std::move(hits), offset);
				return offset;
			};
			const auto hits_of = [&](const zonetool::iw7::Bounds& bounds, std::vector<unsigned short>& hits)
			{
				for (auto i = 0u; i < light_count; i++)
				{
					if (lights[i].local && reaches(lights[i], bounds))
					{
						hits.push_back(static_cast<unsigned short>(i));
					}
				}
			};
			// a merged static model (static_model_clusters): the lights reaching one of its members, not its whole box
			std::unordered_map<unsigned int, const static_model_clusters::cluster*> cluster_at;
			{
				const auto& plan = static_model_clusters::current();
				for (const auto& c : plan.clusters)
				{
					if (!c.members.empty() && c.members.front() < plan.iw7_index.size())
					{
						cluster_at[plan.iw7_index[c.members.front()]] = &c;
					}
				}
			}
			const auto cluster_hits = [&](const static_model_clusters::cluster& c)
			{
				std::vector<unsigned short> hits;
				for (const auto& mb : c.member_bounds)
				{
					zonetool::iw7::Bounds b{};
					for (auto k = 0; k < 3; k++)
					{
						b.midPoint[k] = (mb[k] + mb[k + 3]) * 0.5f;
						b.halfSize[k] = (mb[k + 3] - mb[k]) * 0.5f;
					}
					hits_of(b, hits);
				}
				std::ranges::sort(hits);
				hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
				return hits;
			};
			const auto static_count = world->dpvs.staticSurfaceCount;
			std::vector<unsigned int> surface_offsets(static_count), model_offsets(world->dpvs.smodelCount);
			auto longest = 0u;
			// the sky box spans the world: lights never light it (stock skies have a short list)
			const auto sky = world_sky::surfaces(world);
			// the static surfaces and the static models, shortest list first: while the 16-bit offsets have room each gets its
			// own list, past that the shortest stored list holding all its lights (a light that does not reach it adds nothing
			// to it; the list of every local light is stored first so one always exists)
			std::vector<std::vector<unsigned short>> surface_hits(static_count);
			parallel_for(static_count, [&](const std::uint32_t s, std::uint32_t)
			{
				if (!sky[s])
				{
					hits_of(world->dpvs.surfacesBounds[s].bounds, surface_hits[s]);
				}
			});
			std::vector<std::vector<unsigned short>> model_hits(world->dpvs.smodelCount);
			parallel_for(world->dpvs.smodelCount, [&](const std::uint32_t m, std::uint32_t)
			{
				const auto c = cluster_at.find(m);
				if (c != cluster_at.end())
				{
					model_hits[m] = cluster_hits(*c->second);
				}
				else
				{
					hits_of(world->dpvs.smodelInsts[m].bounds, model_hits[m]);
				}
			});
			std::vector<unsigned short> every_light;
			for (auto i = 0u; i < light_count; i++)
			{
				if (lights[i].local)
				{
					every_light.push_back(static_cast<unsigned short>(i));
				}
			}
			intern(every_light);
			// entries below static_count: surfaces; the rest: static models
			const auto hits_at = [&](const unsigned int k) -> std::vector<unsigned short>&
			{
				return k < static_count ? surface_hits[k] : model_hits[k - static_count];
			};
			const auto offset_at = [&](const unsigned int k) -> unsigned int&
			{
				return k < static_count ? surface_offsets[k] : model_offsets[k - static_count];
			};
			std::vector<unsigned int> order(static_count + world->dpvs.smodelCount);
			std::iota(order.begin(), order.end(), 0u);
			std::ranges::stable_sort(order, [&](const unsigned int a, const unsigned int b) { return hits_at(a).size() < hits_at(b).size(); });
			auto shared = 0u, shared_surfaces = 0u;
			for (const auto m : order)
			{
				auto& hits = hits_at(m);
				if (hits.empty() || interned.contains(hits) || lists.size() + hits.size() + 1 <= 0xFFFF)
				{
					offset_at(m) = intern(std::move(hits));
					continue;
				}
				const std::vector<unsigned short>* best = nullptr;
				for (const auto& [list, offset] : interned)
				{
					if ((!best || list.size() < best->size()) && list.size() >= hits.size()
						&& std::ranges::includes(list, hits))
					{
						best = &list;
					}
				}
				offset_at(m) = interned.at(*best);
				(m < static_count ? shared_surfaces : shared)++;
			}
			if (shared || shared_surfaces)
			{
				ZONETOOL_INFO("world lights: %u static surfaces and %u static models share a longer light list holding all their lights "
					"(16-bit list offsets)", shared_surfaces, shared);
			}
			for (const auto& [list, offset] : interned)
			{
				longest = std::max(longest, static_cast<unsigned int>(list.size()));
			}
			if (lists.size() > 0xFFFF)
			{
				std::size_t models_longest = 0;
				for (auto m = 0u; m < world->dpvs.smodelCount; m++)
				{
					models_longest = std::max<std::size_t>(models_longest, model_offsets[m] ? lists[model_offsets[m]] : 0);
				}
				ZONETOOL_FATAL("world lights: %zu light list entries do not fit 16-bit offsets (longest list %u lights, a static model's %zu; "
					"%zu lists)", lists.size(), longest, models_longest, interned.size());
			}
			for (auto s = 0u; s < static_count; s++)
			{
				const auto offset = static_cast<unsigned short>(surface_offsets[s]);
				std::memcpy(&world->dpvs.surfaces[s].unk1, &offset, sizeof(offset));
			}
			for (auto m = 0u; m < world->dpvs.smodelCount; m++)
			{
				world->dpvs.smodelDrawInsts[m].unk11 = static_cast<unsigned short>(model_offsets[m]);
			}
			auto& ll = world->lightLists;
			ll.surfaceListOffsetCount = static_count;
			ll.surfaceListOffsets = allocator.allocate_array<unsigned int>(std::max(1u, static_count));
			std::memcpy(ll.surfaceListOffsets, surface_offsets.data(), sizeof(unsigned int) * surface_offsets.size());
			ll.smodelListOffsetCount = world->dpvs.smodelCount;
			ll.smodelListOffsets = allocator.allocate_array<unsigned int>(std::max(1u, world->dpvs.smodelCount));
			std::memcpy(ll.smodelListOffsets, model_offsets.data(), sizeof(unsigned int) * model_offsets.size());
			ll.listsSize = static_cast<unsigned int>(lists.size());
			ll.lists = allocator.allocate_array<unsigned short>(lists.size());
			std::memcpy(ll.lists, lists.data(), sizeof(unsigned short) * lists.size());

			ZONETOOL_INFO("world lights: %u hulls (%u wide spots as spheres), %u shadow frustums, %zu shadow casters, %zu lights with "
				"dynamic shadows in %u tree nodes, light lists %zu distinct (longest %u) in %zu entries", hulls, spots_as_spheres, frustums,
				caster_total, dynamic.size(), world->lightAABB.nodeCount, interned.size(), longest, lists.size());
		}
	}
}
