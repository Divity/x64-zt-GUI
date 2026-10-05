#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "gfxworld.hpp"

#include "zonetool/t7/converter/iw7/map_common.hpp"
#include "comworld.hpp"
#include "probe_lighting.hpp"
#include "world_material.hpp"
#include "world_lightmap.hpp"
#include "reflection_probes.hpp"
#include "world_lights.hpp"
#include "world_lightgrid.hpp"
#include "world_sky.hpp"
#include "gfximage.hpp"
#include "xmodel.hpp"
#include "xmodel_mesh.hpp"
#include "static_model_clusters.hpp"
#include "zonetool/t7/converter/iw7/map_entities.hpp"

#include "zonetool/iw7/assets/gfximage.hpp"
#include "zonetool/iw7/assets/gfxworld.hpp"
#include "zonetool/t7/functions.hpp"
#include "zonetool/iw7/assets/gfxworld_tr.hpp"
#include "zonetool/iw7/common/umbra/umbra_tome.hpp"

#include <array>
#include <numeric>
#include <unordered_map>

// Sources for the non-obvious conversions below (BO3 = BlackOps3 client/dedi, IW7 = iw7_ship):
// * vertex streams: BO3 keeps positions (float3) and attributes (20 bytes: R8G8B8A8 colour,
//   float2 uv, R10G10B10A2 normal, R10G10B10A2 tangent) in two streams; a surface's vertices start
//   at vertexDataOffset0 / 12 (= vertexDataOffset1 / 20) and its indices are relative to that. The
//   colour format is R8G8B8A8_UNORM in BO3's vertex format table (0x142F77BD8).
// * normals: BO3's gbuffer_lit vertex shader decodes a component as k * 2.001957 - 1.001957 of the UNORM value,
//   i.e. (k - 512) / 511, IW7's world shader as 2k / 1023 - 1.
// * bitangent sign: both games build B = cross(N, T) * s. BO3 takes s from the tangent's 2-bit
//   alpha as w * 2 - 1 (0 -> -1, 3 -> +1); IW7's world vertex shader uses s = (w > 0) ? -1 : +1.
//   So BO3's alpha 3 becomes IW7's alpha 0 and BO3's 0 becomes IW7's 3.
// * vertex colour: BO3 linearises it as c^2.2 in the vertex shader; IW7 multiplies the square
//   root of albedo by it and squares the result, i.e. applies c^2. The IW7 byte is c^1.1.
// * visibility BSP: BO3 and IW7 walk the node array the same way (0x141CC4590 / IW5 layout): a
//   value above cellCount is a plane (value - cellCount - 1) followed by the offset of its second
//   child, children at +2 and +offset; otherwise it is a leaf, 0 solid, else cell + 1. The planes
//   array only carries the planes the tree uses. Stock IW7 planes have no signbits (byte 17 is 0).
// * AABB trees: in BO3 a node's surface range and static model list are the union of its
//   children's; ranges are sortedSurfIndex slots.
// * static surface ranges: stock IW7 keeps every static surface in exactly one of the four ranges
//   (opaque incl. region 11 casters, decal, trans, emissive), ordered by material sort key.
// * header and counts: bspVersion 159, the sort key table, materialLod0SizeThreshold 0.5, 134
//   entity motion words, the vis-data word counts and the runtime buffer sizes match stock maps
//   with the formulas used here.
// * static models: BO3 flag 0x1 keeps a model out of the sun (0x141C73D70) and spot (0x141C74740)
//   shadow passes; IW7 skips NO_CAST_SHADOW (0x10) in both (0xDCC700, 0xDCCE40, 0xDCDC90).
//   sunShadowFlags bit k / GfxSurface flags bit 3+k select sun light k when IW7's shadow geometry
//   optimisation is on (r_useShadowGeomOpt, 0xDCAF40); stock sunShadowOptCount is the sun count.

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace gfxworld
		{
			namespace
			{
				utils::memory::allocator persistent_allocator;

				constexpr auto t7_position_stride = 12u;
				constexpr auto t7_attribute_stride = 20u;
				constexpr auto t7_aabb_node_size = 48u;

				constexpr unsigned int sort_key_lit_decal = 7;
				constexpr unsigned int sort_key_effect_decal = 14;
				constexpr unsigned int sort_key_top_decal = 17;
				constexpr unsigned int sort_key_effect_auto = 35;
				constexpr unsigned int sort_key_distortion = 24;
				constexpr unsigned int sort_key_effect_distortion = 36;
				constexpr unsigned int sort_key_2d = 41;
				constexpr unsigned int sort_key_opaque_begin = 1;
				constexpr unsigned int sort_key_opaque_end = 6;
				constexpr unsigned int sort_key_decal_begin = 7;
				constexpr unsigned int sort_key_decal_end = 17;
				constexpr unsigned int sort_key_trans_begin = 18;
				constexpr unsigned int sort_key_trans_end = 34;
				constexpr unsigned int sort_key_emissive_begin = 35;
				constexpr unsigned int sort_key_emissive_end = 40;

				constexpr unsigned char no_lightmap = 31;

				constexpr unsigned short t7_smodel_no_shadow = 0x1;

				float umbra_smallest_hole = 0.0f;

				// ---- vertices -------------------------------------------------------------------

				float t7_unit_component(const std::uint32_t k)
				{
					return (static_cast<float>(k) - 512.0f) / 511.0f;
				}

				void t7_unpack_unit(const std::uint32_t packed, float* out)
				{
					out[0] = t7_unit_component(packed & 0x3FF);
					out[1] = t7_unit_component((packed >> 10) & 0x3FF);
					out[2] = t7_unit_component((packed >> 20) & 0x3FF);
				}

				std::uint32_t iw7_unit_component(const float v)
				{
					const auto clamped = std::clamp(v, -1.0f, 1.0f);
					return static_cast<std::uint32_t>(std::floor((clamped + 1.0f) * 0.5f * 1023.0f + 0.5f));
				}

				std::uint32_t iw7_pack_unit(const float* v, const std::uint32_t alpha_bits)
				{
					return iw7_unit_component(v[0]) | (iw7_unit_component(v[1]) << 10)
						| (iw7_unit_component(v[2]) << 20) | ((alpha_bits & 3) << 30);
				}

				void normalize(float* v)
				{
					const auto len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
					if (len > 0.0f)
					{
						v[0] /= len;
						v[1] /= len;
						v[2] /= len;
					}
				}

				std::uint8_t convert_vertex_color(const std::uint8_t c)
				{
					static const auto table = []()
					{
						std::array<std::uint8_t, 256> t{};
						for (auto i = 0; i < 256; i++)
						{
							const auto v = std::pow(static_cast<double>(i) / 255.0, 1.1);
							t[i] = static_cast<std::uint8_t>(std::clamp(std::lround(v * 255.0), 0l, 255l));
						}
						return t;
					}();
					return table[c];
				}

				void convert_vertex(const std::uint8_t* position, const std::uint8_t* attributes,
					zonetool::iw7::GfxWorldVertex& dst)
				{
					std::memcpy(dst.xyz, position, sizeof(float) * 3);

					dst.color.array[0] = convert_vertex_color(attributes[0]);
					dst.color.array[1] = convert_vertex_color(attributes[1]);
					dst.color.array[2] = convert_vertex_color(attributes[2]);
					dst.color.array[3] = attributes[3];

					std::memcpy(dst.texCoord, attributes + 4, sizeof(float) * 2);
					dst.lmapCoord[0] = 0.0f;
					dst.lmapCoord[1] = 0.0f;

					std::uint32_t normal_packed, tangent_packed;
					std::memcpy(&normal_packed, attributes + 12, sizeof(normal_packed));
					std::memcpy(&tangent_packed, attributes + 16, sizeof(tangent_packed));

					float normal[3], tangent[3];
					t7_unpack_unit(normal_packed, normal);
					t7_unpack_unit(tangent_packed, tangent);
					normalize(normal);
					normalize(tangent);

					const auto t7_sign_bits = tangent_packed >> 30;
					const auto sign = t7_sign_bits * 2.0f / 3.0f - 1.0f;

					dst.normal.packed = iw7_pack_unit(normal, 3);
					dst.tangent.packed = iw7_pack_unit(tangent, sign > 0.0f ? 0 : 3);
					dst.binormalSign = sign > 0.0f ? 1.0f : -1.0f;
				}

				const float* t7_position(const GfxWorld* asset, const unsigned int vertex)
				{
					return reinterpret_cast<const float*>(asset->draw.vd0.data + vertex * t7_position_stride);
				}

				void set_bounds(zonetool::iw7::Bounds& bounds, const float* mins, const float* maxs)
				{
					for (auto c = 0; c < 3; c++)
					{
						bounds.midPoint[c] = (mins[c] + maxs[c]) * 0.5f;
						bounds.halfSize[c] = (maxs[c] - mins[c]) * 0.5f;
					}
				}

				// ---- visibility BSP ---------------------------------------------------------------

				void convert_planes_and_nodes(const GfxWorld* asset, zonetool::iw7::GfxWorld* world,
					utils::memory::allocator& allocator)
				{
					const auto cell_count = static_cast<unsigned int>(asset->dpvsPlanes.cellCount);
					const auto node_count = static_cast<unsigned int>(asset->nodeCount);
					const auto* nodes = asset->dpvsPlanes.nodes;

					std::vector<char> visited(node_count, 0);
					std::vector<unsigned int> internal;
					std::set<unsigned int> used_planes;

					std::vector<unsigned int> pending{ 0 };
					while (!pending.empty())
					{
						const auto i = pending.back();
						pending.pop_back();
						if (i >= node_count || visited[i])
						{
							ZONETOOL_FATAL("gfxworld: BSP node %u is out of range or reached twice (%u entries)", i, node_count);
						}
						visited[i] = 1;

						if (nodes[i] >= cell_count + 1)
						{
							if (i + 1 >= node_count)
							{
								ZONETOOL_FATAL("gfxworld: BSP node %u has no child offset", i);
							}
							visited[i + 1] = 1;
							internal.push_back(i);
							used_planes.insert(nodes[i] - (cell_count + 1));
							pending.push_back(i + 2);
							pending.push_back(i + nodes[i + 1]);
						}
					}

					for (auto i = 0u; i < node_count; i++)
					{
						if (!visited[i])
						{
							ZONETOOL_FATAL("gfxworld: BSP node entry %u is never reached", i);
						}
					}

					std::unordered_map<unsigned int, unsigned int> plane_remap;
					world->planeCount = static_cast<int>(used_planes.size());
					world->dpvsPlanes.planes = allocator.allocate_array<zonetool::iw7::cplane_s>(used_planes.size());
					for (const auto plane : used_planes)
					{
						if (plane >= static_cast<unsigned int>(asset->planeCount))
						{
							ZONETOOL_FATAL("gfxworld: BSP references plane %u of %d", plane, asset->planeCount);
						}

						const auto index = static_cast<unsigned int>(plane_remap.size());
						plane_remap[plane] = index;

						const auto& src = asset->dpvsPlanes.planes[plane];
						auto& dst = world->dpvsPlanes.planes[index];
						std::memcpy(dst.normal, src.normal, sizeof(dst.normal));
						dst.dist = src.dist;
						dst.type = src.type;
						std::memset(dst.pad, 0, sizeof(dst.pad));
					}

					// one IW7 cell: leaves become 0 (solid) or 1, planes move to index + 2
					world->nodeCount = static_cast<int>(node_count);
					world->dpvsPlanes.nodes = allocator.allocate_array<unsigned short>(node_count);
					for (auto i = 0u; i < node_count; i++)
					{
						world->dpvsPlanes.nodes[i] = nodes[i] == 0 ? 0 : 1;
					}
					for (const auto i : internal)
					{
						const auto value = plane_remap[nodes[i] - (cell_count + 1)] + 2;
						const auto offset = nodes[i + 1];
						if (value > 0xFFFF || offset > 0xFFFF)
						{
							ZONETOOL_FATAL("gfxworld: BSP node %u does not fit IW7's 16-bit nodes (%u, %u)", i, value, offset);
						}
						world->dpvsPlanes.nodes[i] = static_cast<unsigned short>(value);
						world->dpvsPlanes.nodes[i + 1] = static_cast<unsigned short>(offset);
					}

					world->dpvsPlanes.cellCount = 1;
					world->dpvsPlanes.sceneEntCellBits = nullptr;

					ZONETOOL_INFO("gfxworld: visibility BSP %u entries (%zu planes used of %d), %u BO3 cells merged into one",
						node_count, used_planes.size(), asset->planeCount, cell_count);
				}

				// ---- cells and AABB tree ----------------------------------------------------------

				void convert_aabb_node(const GfxAabbTree& src, zonetool::iw7::GfxAabbTree& dst, const int child_delta,
					utils::memory::allocator& allocator)
				{
					set_bounds(dst.bounds, src.mins, src.maxs);
					dst.childCount = src.childCount;
					dst.surfaceCount = src.surfaceCount;
					dst.startSurfIndex = src.startSurfIndex;
					// BO3's static model indices as the IW7 static models drawing them (static_model_clusters)
					const auto smodels = static_model_clusters::remap_indices(src.smodelIndexes, src.smodelIndexCount);
					dst.smodelIndexCount = static_cast<unsigned short>(smodels.size());
					dst.smodelIndexes = nullptr;
					if (!smodels.empty())
					{
						dst.smodelIndexes = allocator.allocate_array<unsigned short>(smodels.size());
						std::memcpy(dst.smodelIndexes, smodels.data(), sizeof(unsigned short) * smodels.size());
					}

					dst.childrenOffset = 0;
					if (src.childCount)
					{
						const auto offset = static_cast<std::int64_t>(src.childrenOffset);
						if (offset <= 0 || offset % t7_aabb_node_size)
						{
							ZONETOOL_FATAL("gfxworld: AABB node children offset %lld is not a forward node offset", offset);
						}
						dst.childrenOffset = static_cast<int>((offset / t7_aabb_node_size + child_delta)
							* sizeof(zonetool::iw7::GfxAabbTree));
					}
				}

				// a leaf of the AABB tree under the root: sorted slots [begin, begin + count) within bounds
				struct extra_leaf
				{
					unsigned int begin = 0;
					unsigned int count = 0;
					float mins[3]{};
					float maxs[3]{};
				};

				// with_sky: the sky box surface takes the sorted slot after BO3's, in a leaf of its own under the root; `leaves`
				// (the world groups' surfaces, static_model_clusters world_clusters) follow it
				void convert_cells(const GfxWorld* asset, zonetool::iw7::GfxWorld* world,
					zonetool::iw7::GfxWorldTransientZone* zone, const bool with_sky, const std::vector<extra_leaf>& leaves,
					utils::memory::allocator& allocator)
				{
					const auto cell_count = asset->dpvsPlanes.cellCount;

					float mins[3]{ FLT_MAX, FLT_MAX, FLT_MAX };
					float maxs[3]{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
					std::vector<const GfxCell*> tree_cells;
					auto node_total = 1;
					for (auto c = 0; c < cell_count; c++)
					{
						const auto& cell = asset->cells[c];
						for (auto k = 0; k < 3; k++)
						{
							mins[k] = std::min(mins[k], cell.mins[k]);
							maxs[k] = std::max(maxs[k], cell.maxs[k]);
						}
						if (cell.portalCount)
						{
							ZONETOOL_WARNING("gfxworld: BO3 cell %d has %d portals; the merged IW7 cell has none", c, cell.portalCount);
						}
						if (cell.aabbTreeCount > 0)
						{
							tree_cells.push_back(&cell);
							node_total += cell.aabbTreeCount;
						}
					}

					world->cells = allocator.allocate<zonetool::iw7::GfxCell>();
					set_bounds(world->cells[0].bounds, mins, maxs);
					world->cells[0].portalCount = 0;
					world->cells[0].portals = nullptr;

					world->cellTransientInfos = allocator.allocate<zonetool::iw7::GfxCellTransientInfo>();
					world->cellTransientInfos[0].aabbTreeIndex = 0;
					world->cellTransientInfos[0].transientZone = 0;

					if (with_sky)
					{
						node_total++;
					}
					node_total += static_cast<int>(leaves.size());
					auto* tree = allocator.allocate_array<zonetool::iw7::GfxAabbTree>(node_total);
					const auto slots = static_cast<int>(tree_cells.size());
					const auto extra_first = 1 + slots + (with_sky ? 1 : 0);
					auto next = extra_first + static_cast<int>(leaves.size());

					float root_mins[3]{ FLT_MAX, FLT_MAX, FLT_MAX };
					float root_maxs[3]{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
					std::vector<std::pair<unsigned int, unsigned int>> root_ranges;
					std::set<unsigned short> root_smodels;

					for (auto slot = 0; slot < slots; slot++)
					{
						const auto* cell = tree_cells[slot];
						const auto* src = cell->aabbTree;
						const auto root_index = 1 + slot;

						convert_aabb_node(src[0], tree[root_index], next - 1 - root_index, allocator);
						for (auto j = 1; j < cell->aabbTreeCount; j++)
						{
							convert_aabb_node(src[j], tree[next + j - 1], 0, allocator);
						}
						next += cell->aabbTreeCount - 1;

						for (auto k = 0; k < 3; k++)
						{
							root_mins[k] = std::min(root_mins[k], src[0].mins[k]);
							root_maxs[k] = std::max(root_maxs[k], src[0].maxs[k]);
						}
						if (src[0].surfaceCount)
						{
							root_ranges.emplace_back(src[0].startSurfIndex, src[0].startSurfIndex + src[0].surfaceCount);
						}
						const auto smodels = static_model_clusters::remap_indices(src[0].smodelIndexes, src[0].smodelIndexCount);
						root_smodels.insert(smodels.begin(), smodels.end());
					}

					if (with_sky)
					{
						// the sky box encloses the world
						const auto slot = asset->dpvs.staticSurfaceCount;
						auto& sky = tree[1 + slots];
						set_bounds(sky.bounds, asset->mins, asset->maxs);
						sky.childCount = 0;
						sky.childrenOffset = 0;
						sky.startSurfIndex = slot;
						sky.surfaceCount = 1;
						sky.smodelIndexCount = 0;
						sky.smodelIndexes = nullptr;
						root_ranges.emplace_back(slot, slot + 1);
						for (auto k = 0; k < 3; k++)
						{
							root_mins[k] = std::min(root_mins[k], asset->mins[k]);
							root_maxs[k] = std::max(root_maxs[k], asset->maxs[k]);
						}
					}

					for (auto l = 0u; l < leaves.size(); l++)
					{
						const auto& leaf = leaves[l];
						auto& node = tree[extra_first + l];
						set_bounds(node.bounds, leaf.mins, leaf.maxs);
						node.childCount = 0;
						node.childrenOffset = 0;
						if (leaf.count > 0xFFFF)
						{
							ZONETOOL_FATAL("gfxworld: a world group of %u surfaces does not fit an AABB tree node", leaf.count);
						}
						node.startSurfIndex = leaf.begin;
						node.surfaceCount = static_cast<unsigned short>(leaf.count);
						node.smodelIndexCount = 0;
						node.smodelIndexes = nullptr;
						root_ranges.emplace_back(leaf.begin, leaf.begin + leaf.count);
						for (auto k = 0; k < 3; k++)
						{
							root_mins[k] = std::min(root_mins[k], leaf.mins[k]);
							root_maxs[k] = std::max(root_maxs[k], leaf.maxs[k]);
						}
					}

					// the root's surfaces are the union of the cell roots' ranges, which must touch
					std::sort(root_ranges.begin(), root_ranges.end());
					for (auto i = 1u; i < root_ranges.size(); i++)
					{
						if (root_ranges[i].first != root_ranges[i - 1].second)
						{
							ZONETOOL_FATAL("gfxworld: BO3 cell roots cover sorted surfaces [%u, %u) and [%u, %u), which do not touch",
								root_ranges[i - 1].first, root_ranges[i - 1].second, root_ranges[i].first, root_ranges[i].second);
						}
					}

					auto& root = tree[0];
					set_bounds(root.bounds, root_mins, root_maxs);
					if (slots + (with_sky ? 1 : 0) + leaves.size() > 0xFFFF)
					{
						ZONETOOL_FATAL("gfxworld: %zu AABB tree root children", slots + (with_sky ? 1 : 0) + leaves.size());
					}
					root.childCount = static_cast<unsigned short>(slots + (with_sky ? 1 : 0) + leaves.size());
					root.childrenOffset = static_cast<int>(sizeof(zonetool::iw7::GfxAabbTree));
					root.startSurfIndex = root_ranges.empty() ? 0 : root_ranges.front().first;
					root.surfaceCount = static_cast<unsigned short>(root_ranges.empty() ? 0
						: root_ranges.back().second - root_ranges.front().first);
					root.smodelIndexCount = static_cast<unsigned short>(root_smodels.size());
					root.smodelIndexes = nullptr;
					if (!root_smodels.empty())
					{
						root.smodelIndexes = allocator.allocate_array<unsigned short>(root_smodels.size());
						std::copy(root_smodels.begin(), root_smodels.end(), root.smodelIndexes);
					}

					zone->cellCount = 1;
					zone->aabbTreeCounts = allocator.allocate<zonetool::iw7::GfxCellTreeCount>();
					zone->aabbTreeCounts[0].aabbTreeCount = node_total;
					zone->aabbTrees = allocator.allocate<zonetool::iw7::GfxCellTree>();
					zone->aabbTrees[0].aabbTree = tree;

					ZONETOOL_INFO("gfxworld: AABB tree of %d nodes from %d cell trees, root surfaces [%u, %u), %zu root static models",
						node_total, slots, root.startSurfIndex, root.startSurfIndex + root.surfaceCount, root_smodels.size());
				}

				// IW7 culls an AABB tree node's static model list with the node's bounds, so every node has to enclose the static
				// models it lists (and its children); BO3's node bounds need not, and a merged static model reaches past the node
				// of the member that lists it. Grown bottom-up once the static models are final.
				void enclose_static_models(zonetool::iw7::GfxAabbTree* node, const zonetool::iw7::GfxWorld* world, unsigned int& grown)
				{
					float mins[3], maxs[3];
					for (auto k = 0; k < 3; k++)
					{
						mins[k] = node->bounds.midPoint[k] - node->bounds.halfSize[k];
						maxs[k] = node->bounds.midPoint[k] + node->bounds.halfSize[k];
					}
					const auto add = [&](const zonetool::iw7::Bounds& b)
					{
						for (auto k = 0; k < 3; k++)
						{
							mins[k] = std::min(mins[k], b.midPoint[k] - b.halfSize[k]);
							maxs[k] = std::max(maxs[k], b.midPoint[k] + b.halfSize[k]);
						}
					};
					if (node->childCount)
					{
						auto* children = reinterpret_cast<zonetool::iw7::GfxAabbTree*>(reinterpret_cast<char*>(node) + node->childrenOffset);
						for (auto c = 0; c < node->childCount; c++)
						{
							enclose_static_models(&children[c], world, grown);
							add(children[c].bounds);
						}
					}
					for (auto m = 0; m < node->smodelIndexCount; m++)
					{
						add(world->dpvs.smodelInsts[node->smodelIndexes[m]].bounds);
					}
					zonetool::iw7::Bounds before = node->bounds;
					set_bounds(node->bounds, mins, maxs);
					for (auto k = 0; k < 3; k++)
					{
						if (node->bounds.halfSize[k] > before.halfSize[k] + 0.01f)
						{
							grown++;
							break;
						}
					}
				}

				// ---- surfaces ---------------------------------------------------------------------

				int range_of_sort_key(const unsigned int sort_key)
				{
					if (sort_key >= sort_key_emissive_begin)
					{
						return 3;
					}
					if (sort_key >= sort_key_trans_begin)
					{
						return 2;
					}
					if (sort_key >= sort_key_decal_begin)
					{
						return 1;
					}
					return 0;
				}

				struct surface_geometry
				{
					unsigned int first_vertex = 0;
					unsigned int vertex_count = 0;
					unsigned int min_index = 0;
					float max_edge_length = 0.0f;
				};

				constexpr unsigned int sky_box_vertices = 24;
				constexpr unsigned int sky_box_indices = 36;
				constexpr unsigned short sky_box_triangles[sky_box_indices] = {
					0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11,
					12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
				};

				// a static world surface of a world group's members (static_model_clusters world_clusters): its vertices in the
				// transient zone, its indices (relative to its first vertex) in prop_geometry::indices
				struct prop_surface
				{
					const Material* material = nullptr;
					unsigned int first_vertex = 0;
					unsigned int vertex_count = 0;
					unsigned int first_index = 0;
					unsigned int tri_count = 0;
					float mins[3]{};
					float maxs[3]{};
					float max_edge_length = 0.0f;
					bool casts = false;
					// the first triangle of each placed model's part (world_lightmap charts each part on its own)
					std::vector<unsigned int> pieces;
				};

				struct prop_geometry
				{
					std::vector<prop_surface> surfaces;      // each group's together, in group order
					std::vector<unsigned int> group_begin;   // per group its first surface
					std::vector<unsigned short> indices;
				};

				struct surface_plan
				{
					// IW7 surface i is BO3 surface order[i]; BO3 surface s is IW7 surface remap[s]. The sky box, when
					// there is one, is entry `sky` = BO3's surface count: the last opaque surface, as in stock maps
					std::vector<unsigned int> order;
					std::vector<unsigned int> remap;
					std::vector<const world_material::info*> materials;
					std::vector<surface_geometry> geometry;
					unsigned int range_begin[4]{};
					unsigned int range_end[4]{};
					unsigned int sky = ~0u;
					// the world groups' surfaces: entries [prop_begin, prop_begin + prop count)
					unsigned int prop_begin = ~0u;
				};

				surface_geometry measure_surface(const GfxWorld* asset, const GfxSurface& surface)
				{
					surface_geometry geometry{};
					const auto& tris = surface.tris;
					if (tris.vertexDataOffset0 % t7_position_stride || tris.vertexDataOffset1 % t7_attribute_stride
						|| tris.vertexDataOffset0 / t7_position_stride != tris.vertexDataOffset1 / t7_attribute_stride)
					{
						ZONETOOL_FATAL("gfxworld: surface vertex streams disagree (offsets %d / %d)",
							tris.vertexDataOffset0, tris.vertexDataOffset1);
					}

					const auto base = static_cast<unsigned int>(tris.vertexDataOffset0) / t7_position_stride;
					const auto* indices = asset->draw.indices + tris.baseIndex;

					auto lo = std::numeric_limits<unsigned int>::max();
					auto hi = 0u;
					for (auto i = 0u; i < tris.triCount * 3u; i++)
					{
						lo = std::min<unsigned int>(lo, indices[i]);
						hi = std::max<unsigned int>(hi, indices[i]);
					}
					if (!tris.triCount)
					{
						lo = hi = 0;
					}

					for (auto t = 0u; t < tris.triCount; t++)
					{
						for (auto e = 0; e < 3; e++)
						{
							const auto* a = t7_position(asset, base + indices[t * 3 + e]);
							const auto* b = t7_position(asset, base + indices[t * 3 + (e + 1) % 3]);
							const float d[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
							geometry.max_edge_length = std::max(geometry.max_edge_length,
								std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
						}
					}

					geometry.first_vertex = base + lo;
					geometry.vertex_count = tris.triCount ? hi - lo + 1 : 0;
					geometry.min_index = lo;
					if (geometry.first_vertex + geometry.vertex_count > asset->draw.vertexCount)
					{
						ZONETOOL_FATAL("gfxworld: surface vertices [%u, %u) run past the %u world vertices",
							geometry.first_vertex, geometry.first_vertex + geometry.vertex_count, asset->draw.vertexCount);
					}
					return geometry;
				}

				surface_plan plan_surfaces(const GfxWorld* asset, const world_material::info* sky, const prop_geometry& props)
				{
					surface_plan plan{};
					const auto surface_count = static_cast<unsigned int>(asset->surfaceCount);
					const auto static_count = asset->dpvs.staticSurfaceCount;

					plan.materials.resize(surface_count);
					plan.geometry.resize(surface_count);
					for (auto s = 0u; s < surface_count; s++)
					{
						const auto& surface = asset->dpvs.surfaces[s];
						plan.materials[s] = &world_material::get(surface.material);
						plan.geometry[s] = measure_surface(asset, surface);
					}

					// static surfaces by IW7 sort key, BO3 order kept within a key; brush model
					// surfaces follow unchanged because the brush models address them by range
					plan.order.resize(surface_count);
					std::iota(plan.order.begin(), plan.order.end(), 0u);
					std::stable_sort(plan.order.begin(), plan.order.begin() + static_count,
						[&](const unsigned int a, const unsigned int b)
					{
						return plan.materials[a]->sort_key < plan.materials[b]->sort_key;
					});

					plan.remap.resize(surface_count);
					for (auto i = 0u; i < surface_count; i++)
					{
						plan.remap[plan.order[i]] = i;
					}

					// range r spans the surfaces whose key falls in r; an empty range sits where the
					// next one starts
					for (auto r = 0; r < 4; r++)
					{
						auto begin = 0u;
						while (begin < static_count && range_of_sort_key(plan.materials[plan.order[begin]]->sort_key) < r)
						{
							begin++;
						}
						auto end = begin;
						while (end < static_count && range_of_sort_key(plan.materials[plan.order[end]]->sort_key) == r)
						{
							end++;
						}
						plan.range_begin[r] = begin;
						plan.range_end[r] = end;
					}

					if (sky)
					{
						// the sky box: its vertices follow BO3's, its surface ends the opaque range
						plan.sky = surface_count;
						plan.materials.push_back(sky);
						surface_geometry geometry{};
						geometry.first_vertex = asset->draw.vertexCount;
						geometry.vertex_count = sky_box_vertices;
						geometry.min_index = 0;
						for (auto k = 0; k < 3; k++)
						{
							geometry.max_edge_length = std::max(geometry.max_edge_length, asset->maxs[k] - asset->mins[k]);
						}
						plan.geometry.push_back(geometry);
						plan.order.insert(plan.order.begin() + plan.range_end[0], plan.sky);
						plan.remap.resize(surface_count + 1);
						for (auto i = 0u; i <= surface_count; i++)
						{
							plan.remap[plan.order[i]] = i;
						}
						plan.range_end[0]++;
						for (auto r = 1; r < 4; r++)
						{
							plan.range_begin[r]++;
							plan.range_end[r]++;
						}
					}

					// the world groups' surfaces, each at the end of its range, one material's together (IW7 draws a run of
					// consecutive visible surfaces of one material as one entry)
					plan.prop_begin = static_cast<unsigned int>(plan.materials.size());
					std::vector<unsigned int> by_material;
					for (auto k = 0u; k < props.surfaces.size(); k++)
					{
						const auto& prop = props.surfaces[k];
						plan.materials.push_back(&world_material::get(prop.material));
						surface_geometry geometry{};
						geometry.first_vertex = prop.first_vertex;
						geometry.vertex_count = prop.vertex_count;
						geometry.min_index = 0;
						geometry.max_edge_length = prop.max_edge_length;
						plan.geometry.push_back(geometry);
						by_material.push_back(plan.prop_begin + k);
					}
					std::ranges::stable_sort(by_material, [&](const unsigned int a, const unsigned int b)
					{
						return plan.materials[a]->name < plan.materials[b]->name;
					});
					for (const auto source : by_material)
					{
						const auto r = range_of_sort_key(plan.materials[source]->sort_key);
						plan.order.insert(plan.order.begin() + plan.range_end[r], source);
						plan.range_end[r]++;
						for (auto later = r + 1; later < 4; later++)
						{
							plan.range_begin[later]++;
							plan.range_end[later]++;
						}
					}
					plan.remap.assign(plan.materials.size(), 0);
					for (auto i = 0u; i < plan.order.size(); i++)
					{
						plan.remap[plan.order[i]] = i;
					}

					return plan;
				}

				void convert_surfaces(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, const surface_plan& plan,
					const prop_geometry& props, const unsigned int sun_count, utils::memory::allocator& allocator)
				{
					const auto surface_count = static_cast<unsigned int>(plan.order.size());
					const auto sun_mask = static_cast<unsigned char>((1u << sun_count) - 1u);
					const auto with_sky = plan.sky != ~0u;

					world->dpvs.surfaces = allocator.allocate_array<zonetool::iw7::GfxSurface>(surface_count);
					world->dpvs.surfacesBounds = allocator.allocate_array<zonetool::iw7::GfxSurfaceBounds>(surface_count);

					// indices are rewritten relative to each surface's own first vertex; the sky box's follow BO3's
					const auto source_indices = static_cast<unsigned int>(asset->draw.indexCount);
					const auto prop_indices = source_indices + (with_sky ? sky_box_indices : 0u);
					world->draw.indexCount = prop_indices + static_cast<unsigned int>(props.indices.size());
					world->draw.indices = allocator.allocate_array<unsigned short>(world->draw.indexCount);
					std::memcpy(world->draw.indices, asset->draw.indices, sizeof(unsigned short) * source_indices);
					if (with_sky)
					{
						std::memcpy(world->draw.indices + source_indices, sky_box_triangles, sizeof(sky_box_triangles));
					}
					if (!props.indices.empty())
					{
						std::memcpy(world->draw.indices + prop_indices, props.indices.data(), sizeof(unsigned short) * props.indices.size());
					}

					for (auto i = 0u; i < surface_count; i++)
					{
						const auto t7_index = plan.order[i];
						const auto& geometry = plan.geometry[t7_index];
						const auto& material = *plan.materials[t7_index];
						auto& dst = world->dpvs.surfaces[i];

						if (t7_index == plan.sky)
						{
							dst = {};
							dst.tris.firstVertex = geometry.first_vertex;
							dst.tris.maxEdgeLength = geometry.max_edge_length;
							dst.tris.vertexCount = static_cast<unsigned short>(geometry.vertex_count);
							dst.tris.triCount = static_cast<unsigned short>(sky_box_indices / 3);
							dst.tris.baseIndex = source_indices;
							auto* stub = allocator.allocate<zonetool::iw7::Material>();
							stub->name = allocator.duplicate_string(material.name);
							dst.material = stub;
							dst.lightmapIndex = no_lightmap;
							dst.flags = 0; // the box would shadow the whole world
							auto& bounds = world->dpvs.surfacesBounds[i];
							set_bounds(bounds.bounds, asset->mins, asset->maxs);
							std::memset(bounds.unk, 0, sizeof(bounds.unk));
							continue;
						}

						if (t7_index >= plan.prop_begin)
						{
							const auto& prop = props.surfaces[t7_index - plan.prop_begin];
							dst = {};
							dst.tris.firstVertex = prop.first_vertex;
							dst.tris.maxEdgeLength = prop.max_edge_length;
							dst.tris.vertexCount = static_cast<unsigned short>(prop.vertex_count);
							dst.tris.triCount = static_cast<unsigned short>(prop.tri_count);
							dst.tris.baseIndex = prop_indices + prop.first_index;
							auto* stub = allocator.allocate<zonetool::iw7::Material>();
							stub->name = allocator.duplicate_string(material.name);
							dst.material = stub;
							dst.lightmapIndex = material.lightmapped ? 0 : no_lightmap;
							const auto casts_sun = material.casts_shadow && prop.casts;
							dst.flags = casts_sun ? static_cast<unsigned char>(1 | (sun_mask << 3)) : 0;
							auto& bounds = world->dpvs.surfacesBounds[i];
							set_bounds(bounds.bounds, prop.mins, prop.maxs);
							std::memset(bounds.unk, 0, sizeof(bounds.unk));
							continue;
						}

						const auto& src = asset->dpvs.surfaces[t7_index];

						dst.tris.vertexLayerData = 0;
						dst.tris.firstVertex = geometry.first_vertex;
						dst.tris.maxEdgeLength = geometry.max_edge_length;
						dst.tris.vertexCount = static_cast<unsigned short>(geometry.vertex_count);
						dst.tris.triCount = src.tris.triCount;
						dst.tris.baseIndex = static_cast<unsigned int>(src.tris.baseIndex);

						auto* stub = allocator.allocate<zonetool::iw7::Material>();
						stub->name = allocator.duplicate_string(material.name);
						dst.material = stub;

						dst.lightmapIndex = material.lightmapped ? 0 : no_lightmap;

						const auto casts_sun = material.casts_shadow && (src.flags & 1);
						dst.flags = casts_sun ? static_cast<unsigned char>(1 | (sun_mask << 3)) : 0;

						dst.sortKey = 0;
						dst.unk1 = 0;
						dst.unk2 = 0;
						dst.unk3 = 0;
						dst.unk4 = 0;
						dst.transientZone = 0;

						for (auto k = 0u; k < src.tris.triCount * 3u; k++)
						{
							auto& index = world->draw.indices[src.tris.baseIndex + k];
							index = static_cast<unsigned short>(index - geometry.min_index);
						}

						auto& bounds = world->dpvs.surfacesBounds[i];
						set_bounds(bounds.bounds, src.bounds[0], src.bounds[1]);
						std::memset(bounds.unk, 0, sizeof(bounds.unk));
					}

					const auto source_static = asset->dpvs.staticSurfaceCount;
					const auto static_count = source_static + (with_sky ? 1u : 0u) + static_cast<unsigned int>(props.surfaces.size());
					world->dpvs.staticSurfaceCount = static_count;
					world->dpvs.litOpaqueSurfsBegin = plan.range_begin[0];
					world->dpvs.litOpaqueSurfsEnd = plan.range_end[0];
					world->dpvs.litDecalSurfsBegin = plan.range_begin[1];
					world->dpvs.litDecalSurfsEnd = plan.range_end[1];
					world->dpvs.litTransSurfsBegin = plan.range_begin[2];
					world->dpvs.litTransSurfsEnd = plan.range_end[2];
					world->dpvs.emissiveSurfsBegin = plan.range_begin[3];
					world->dpvs.emissiveSurfsEnd = plan.range_end[3];

					if (plan.range_end[3] != static_count)
					{
						ZONETOOL_FATAL("gfxworld: static surface ranges end at %u, not at %u", plan.range_end[3], static_count);
					}

					// sorted slots keep BO3's order (the AABB tree addresses them), their surfaces move; the sky box
					// takes the slot after them
					world->dpvs.sortedSurfIndex = allocator.allocate_array<unsigned int>(static_count);
					for (auto k = 0u; k < source_static; k++)
					{
						world->dpvs.sortedSurfIndex[k] = plan.remap[asset->dpvs.sortedSurfIndex[k]];
					}
					if (with_sky)
					{
						world->dpvs.sortedSurfIndex[source_static] = plan.remap[plan.sky];
					}
					// the world groups' surfaces take the slots after the sky's, each group's together (prop_slot_begin)
					for (auto k = 0u; k < props.surfaces.size(); k++)
					{
						world->dpvs.sortedSurfIndex[source_static + (with_sky ? 1u : 0u) + k] = plan.remap[plan.prop_begin + k];
					}

					// runtime in the zone (XFILE_BLOCK_RUNTIME), filled so the dump is self consistent
					world->dpvs.surfaceCastsSunShadow = allocator.allocate_array<unsigned int>(world->dpvs.surfaceVisDataCount);
					for (auto i = 0u; i < static_count; i++)
					{
						if (world->dpvs.surfaces[i].flags & 1)
						{
							world->dpvs.surfaceCastsSunShadow[i >> 5] |= 0x80000000u >> (i & 31);
						}
					}
				}

				// A material whose baked textures cover several BO3 texture units (world_material::info::
				// uv_period), or BO3's water (its maps sampled at uv x normalMapScale), needs its surfaces' uv0
				// moved into its texture area: (uv - uv_origin) / uv_span, the period for a baked material. A vertex
				// two areas disagree on cannot be converted.
				void apply_uv_periods(const GfxWorld* asset, zonetool::iw7::GfxWorldTransientZone* zone, const surface_plan& plan)
				{
					const auto same_area = [](const world_material::info& a, const world_material::info& b)
					{
						return a.uv_origin[0] == b.uv_origin[0] && a.uv_origin[1] == b.uv_origin[1] && a.uv_span[0] == b.uv_span[0]
							&& a.uv_span[1] == b.uv_span[1];
					};
					std::vector<const world_material::info*> owner(zone->vertexCount, nullptr);
					auto scaled = 0u;
					for (auto s = 0u; s < plan.materials.size(); s++)
					{
						const auto& material = *plan.materials[s];
						const auto& geometry = plan.geometry[s];
						for (auto v = geometry.first_vertex; v < geometry.first_vertex + geometry.vertex_count; v++)
						{
							if (owner[v] && !same_area(*owner[v], material))
							{
								ZONETOOL_FATAL("gfxworld: vertex %u is used by surfaces whose materials need the texture areas "
									"(%g, %g) + (%g, %g) and (%g, %g) + (%g, %g)", v, owner[v]->uv_origin[0], owner[v]->uv_origin[1],
									owner[v]->uv_span[0], owner[v]->uv_span[1], material.uv_origin[0], material.uv_origin[1],
									material.uv_span[0], material.uv_span[1]);
							}
							if (owner[v])
							{
								continue;
							}
							owner[v] = &material;
							if (material.uv_origin[0] != 0.0f || material.uv_origin[1] != 0.0f || material.uv_span[0] != 1.0f
								|| material.uv_span[1] != 1.0f)
							{
								for (auto axis = 0; axis < 2; axis++)
								{
									auto& uv = zone->vd.vertices[v].texCoord[axis];
									uv = (uv - material.uv_origin[axis]) / material.uv_span[axis];
								}
								scaled++;
							}
						}
					}
					if (scaled)
					{
						ZONETOOL_INFO("gfxworld: moved uv0 of %u vertices into their materials' texture areas (texture periods, "
							"BO3 water's normal map scale)", scaled);
					}
				}

				// white rgb on the vertices of surfaces whose IW7 techset reads the vertex colour where BO3's reads only its
				// alpha (world_material info vertex_alpha: the multiply decals); a vertex another surface also uses would
				// change for it too, which is counted
				void apply_vertex_alpha_white(zonetool::iw7::GfxWorldTransientZone* zone, const surface_plan& plan)
				{
					std::vector<std::uint8_t> use(zone->vertexCount, 0); // bit 0: a white surface's, bit 1: another's
					for (auto s = 0u; s < plan.materials.size(); s++)
					{
						const auto bit = plan.materials[s]->vertex_alpha ? 1u : 2u;
						const auto& geometry = plan.geometry[s];
						for (auto v = geometry.first_vertex; v < geometry.first_vertex + geometry.vertex_count; v++)
						{
							use[v] |= bit;
						}
					}
					auto whitened = 0u, shared = 0u;
					for (auto v = 0u; v < zone->vertexCount; v++)
					{
						if (use[v] & 1u)
						{
							auto& c = zone->vd.vertices[v].color.array;
							c[0] = c[1] = c[2] = 0xFF;
							whitened++;
							shared += (use[v] & 2u) ? 1u : 0u;
						}
					}
					if (whitened)
					{
						ZONETOOL_INFO("gfxworld: vertex rgb white on %u vertices of surfaces that take only BO3's vertex alpha (%u also used by "
							"other surfaces)", whitened, shared);
					}
				}

				// the sky box as stock: one quad per face (-x, -y, -z, +z, +y, +x), triangles 0 1 2 and 0 2 3
				// wound outward, normals inward
				void sky_box(const float* mins, const float* maxs, zonetool::iw7::GfxWorldVertex* out)
				{
					static constexpr int corners[6][4][3] = {
						{ { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 }, { 0, 0, 0 } },
						{ { 1, 0, 0 }, { 1, 0, 1 }, { 0, 0, 1 }, { 0, 0, 0 } },
						{ { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } },
						{ { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 } },
						{ { 0, 1, 0 }, { 0, 1, 1 }, { 1, 1, 1 }, { 1, 1, 0 } },
						{ { 1, 1, 0 }, { 1, 1, 1 }, { 1, 0, 1 }, { 1, 0, 0 } },
					};
					static constexpr float normals[6][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }, { 0, -1, 0 }, { -1, 0, 0 } };
					static constexpr float tangents[6][3] = { { 0, 1, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } };
					for (auto f = 0; f < 6; f++)
					{
						for (auto k = 0; k < 4; k++)
						{
							auto& v = out[f * 4 + k];
							v = {};
							for (auto c = 0; c < 3; c++)
							{
								v.xyz[c] = corners[f][k][c] ? maxs[c] : mins[c];
							}
							v.binormalSign = 1.0f;
							v.color.array[0] = v.color.array[1] = v.color.array[2] = v.color.array[3] = 0xFF;
							v.normal.packed = iw7_pack_unit(normals[f], 3);
							v.tangent.packed = iw7_pack_unit(tangents[f], 0);
						}
					}
				}

				zonetool::iw7::GfxWorldTransientZone* convert_transient_zone(const GfxWorld* asset,
					const std::string& zone_name, const bool with_sky, utils::memory::allocator& allocator)
				{
					auto* zone = allocator.allocate<zonetool::iw7::GfxWorldTransientZone>();
					zone->name = allocator.duplicate_string(zone_name);
					zone->transientZoneIndex = 0;
					const auto source_count = asset->draw.vertexCount;
					zone->vertexCount = source_count + (with_sky ? sky_box_vertices : 0u);
					zone->vd.vertices = allocator.allocate_array<zonetool::iw7::GfxWorldVertex>(zone->vertexCount);

					const auto* positions = asset->draw.vd0.data;
					const auto* attributes = asset->draw.vd1.data;
					for (auto v = 0u; v < source_count; v++)
					{
						convert_vertex(positions + v * t7_position_stride, attributes + v * t7_attribute_stride,
							zone->vd.vertices[v]);
					}
					if (with_sky)
					{
						sky_box(asset->mins, asset->maxs, zone->vd.vertices + source_count);
					}

					zone->vertexLayerDataSize = 0;
					zone->vld.data = nullptr;
					return zone;
				}

				// a world group surface's vertex cap: the lightmap's charts split its vertices (world_lightmap: at most 65536 a
				// surface after the split)
				constexpr std::size_t prop_surface_vertices = 16384;
				// a placed surface's triangles go in chunks this size (at most 3 vertices each)
				constexpr unsigned int prop_chunk_triangles = 4096;

				// the world groups (static_model_clusters world_clusters) at their members' member_lod as static world surfaces:
				// per group one surface a material (another past prop_surface_vertices), the vertices appended to the zone's
				prop_geometry place_world_groups(const GfxWorld* asset, zonetool::iw7::GfxWorldTransientZone* zone,
					utils::memory::allocator& allocator)
				{
					prop_geometry out;
					const auto& clusters = static_model_clusters::current();
					std::vector<zonetool::iw7::GfxWorldVertex> vertices;

					struct open_surface
					{
						const Material* material = nullptr;
						std::vector<zonetool::iw7::GfxWorldVertex> verts;
						std::vector<unsigned short> indices;
						std::vector<unsigned int> pieces;
					};
					const auto close = [&](open_surface& o, const bool casts)
					{
						if (o.indices.empty())
						{
							return;
						}
						prop_surface surface{};
						surface.material = o.material;
						surface.first_vertex = zone->vertexCount + static_cast<unsigned int>(vertices.size());
						surface.vertex_count = static_cast<unsigned int>(o.verts.size());
						surface.first_index = static_cast<unsigned int>(out.indices.size());
						surface.tri_count = static_cast<unsigned int>(o.indices.size() / 3);
						surface.casts = casts;
						for (auto k = 0; k < 3; k++)
						{
							surface.mins[k] = FLT_MAX;
							surface.maxs[k] = -FLT_MAX;
						}
						for (auto& v : o.verts)
						{
							for (auto k = 0; k < 3; k++)
							{
								surface.mins[k] = std::min(surface.mins[k], v.xyz[k]);
								surface.maxs[k] = std::max(surface.maxs[k], v.xyz[k]);
							}
							for (auto c = 0; c < 3; c++)
							{
								v.color.array[c] = convert_vertex_color(v.color.array[c]);
							}
						}
						for (auto t = 0u; t + 2 < o.indices.size(); t += 3)
						{
							for (auto e = 0; e < 3; e++)
							{
								const auto* a = o.verts[o.indices[t + e]].xyz;
								const auto* b = o.verts[o.indices[t + (e + 1) % 3]].xyz;
								const float d[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
								surface.max_edge_length = std::max(surface.max_edge_length, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
							}
						}
						surface.pieces = std::move(o.pieces);
						vertices.insert(vertices.end(), o.verts.begin(), o.verts.end());
						out.indices.insert(out.indices.end(), o.indices.begin(), o.indices.end());
						out.surfaces.push_back(std::move(surface));
						o.verts.clear();
						o.indices.clear();
						o.pieces.clear();
					};

					for (const auto& group : clusters.world_clusters)
					{
						out.group_begin.push_back(static_cast<unsigned int>(out.surfaces.size()));
						std::map<std::string, open_surface> open; // by material name: a group's surfaces in name order
						for (const auto i : group.members)
						{
							const auto& src = asset->dpvs.smodelDrawInsts[i];
							const auto meshes = xmodel::lod_meshes(src.model);
							if (meshes.empty())
							{
								ZONETOOL_FATAL("gfxworld: world group %s member %u (%s) has no mesh", group.name.data(), i, src.model->name);
							}
							const auto lod = std::min<std::size_t>(clusters.member_lod[i], meshes.size() - 1);
							const auto& [mesh, mats] = meshes[lod];
							for (auto s = 0u; s < mats.size(); s++)
							{
								if (!mats[s])
								{
									continue;
								}
								auto& o = open[mats[s]->name];
								o.material = mats[s];
								// in chunks of prop_chunk_triangles: the lightmap's charts split a surface's vertices further
								const auto triangles = static_cast<unsigned int>(mesh->surfs[s].triCount);
								for (auto first = 0u; first < triangles; first += prop_chunk_triangles)
								{
									const auto n = std::min(prop_chunk_triangles, triangles - first);
									const auto piece = static_cast<unsigned int>(o.indices.size() / 3);
									if (!xmodel_mesh::append_placed_world(mesh, s, first, n, src.placement.origin, src.placement.axis,
										src.placement.scale, o.verts, o.indices, prop_surface_vertices))
									{
										close(o, group.casts_shadow);
										if (!xmodel_mesh::append_placed_world(mesh, s, first, n, src.placement.origin, src.placement.axis,
											src.placement.scale, o.verts, o.indices, prop_surface_vertices))
										{
											ZONETOOL_FATAL("gfxworld: world group %s: triangles %u-%u of surface %u of %s do not fit a world surface",
												group.name.data(), first, first + n, s, mesh->name);
										}
										o.pieces.push_back(0);
										continue;
									}
									o.pieces.push_back(piece);
								}
							}
						}
						for (auto& [name, o] : open)
						{
							close(o, group.casts_shadow);
						}
					}

					// the texture area each material's world group surfaces use: the world bake of a material only they use covers it
					{
						std::unordered_set<const Material*> on_world;
						for (auto s = 0; s < asset->surfaceCount; s++)
						{
							on_world.insert(asset->dpvs.surfaces[s].material);
						}
						std::unordered_map<const Material*, world_material::uv_bounds> usage;
						for (const auto& surface : out.surfaces)
						{
							if (on_world.contains(surface.material))
							{
								continue;
							}
							auto& uv = usage[surface.material];
							const auto first = surface.first_vertex - zone->vertexCount;
							for (auto v = first; v < first + surface.vertex_count; v++)
							{
								uv.add(vertices[v].texCoord[0], vertices[v].texCoord[1]);
							}
						}
						std::vector<const Material*> props;
						for (const auto& [material, uv] : usage)
						{
							world_material::set_world_usage(material, uv);
							props.push_back(material);
						}
						// IW7 loads at most 15616 images across its zones: the placed models' materials share atlases as models do
						std::ranges::sort(props, [](const Material* a, const Material* b) { return std::strcmp(a->name, b->name) < 0; });
						world_material::plan_world_atlases(props);
					}

					if (!vertices.empty())
					{
						const auto before = zone->vertexCount;
						auto* grown = allocator.allocate_array<zonetool::iw7::GfxWorldVertex>(before + vertices.size());
						std::memcpy(grown, zone->vd.vertices, sizeof(zonetool::iw7::GfxWorldVertex) * before);
						std::memcpy(grown + before, vertices.data(), sizeof(zonetool::iw7::GfxWorldVertex) * vertices.size());
						zone->vd.vertices = grown;
						zone->vertexCount = before + static_cast<unsigned int>(vertices.size());
						ZONETOOL_INFO("gfxworld: %zu world groups of static models as %zu static world surfaces (%zu vertices, %zu triangles)",
							clusters.world_clusters.size(), out.surfaces.size(), vertices.size(), out.indices.size() / 3);
					}
					return out;
				}

				// ---- static models ----------------------------------------------------------------

				void convert_static_models(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, const unsigned int sun_count,
					utils::memory::allocator& allocator)
				{
					// BO3's static models, some merged (static_model_clusters): the unmerged ones keep their order, the merged
					// models follow
					const auto& clusters = static_model_clusters::current();
					const auto bo3_count = asset->dpvs.smodelCount;
					const auto count = clusters.iw7_count;

					const auto sun_mask = static_cast<unsigned char>((1u << sun_count) - 1u);

					world->dpvs.smodelCount = count;
					world->dpvs.smodelDrawInsts = allocator.allocate_array<zonetool::iw7::GfxStaticModelDrawInst>(count);
					world->dpvs.smodelInsts = allocator.allocate_array<zonetool::iw7::GfxStaticModelInst>(count);

					std::unordered_map<const XModel*, zonetool::iw7::XModel*> model_stubs;
					unsigned int no_shadow = 0;

					// IW7 lists a static model's surfaces per frame as 16-bit entries, record offset x 4 + surface index
					// (0x140DCE680, read back by 0x140DEBDB0): a LOD of more than 16 surfaces runs into the next record's
					// offset, and the draw reads mid-record (crash 0x140DEBF06); stock places no such static model. Such an
					// instance stays in the list (every static model index holds) but is never drawn (cullDist 1, no shadow), and a script_model draws the model.
					std::unordered_map<const XModel*, bool> too_many_surfaces;
					// cullDist does not keep such an instance out of the lists: 0x140DCB890 skips it only while its scaled distance
					// ((distance x scale + bias) x factor) is at least cullDist, which a close camera undercuts (crash 0x140DEC1C6,
					// 2026-10-02: the entry for surface 16 of a posed cluster read mid-record). So the instance draws an opaque shadow
					// proxy instead: shadow-only materials the camera's lists skip (camera region 11, 0x140DCE680), within 16
					// surfaces (the shadow lists draw every proxy), and with no shadow of its own it draws nothing.
					const char* hidden_model = nullptr;
					{
						const static_model_clusters::shadow_proxy* best = nullptr;
						for (const auto& proxy : clusters.proxies)
						{
							if (!proxy.alpha && (!best || proxy.members.size() < best->members.size()))
							{
								best = &proxy;
							}
						}
						if (best)
						{
							hidden_model = allocator.duplicate_string(best->name);
						}
					}
					const auto hide = [&](zonetool::iw7::GfxStaticModelDrawInst& inst)
					{
						if (!hidden_model)
						{
							ZONETOOL_FATAL("gfxworld: a static model of more than 16 surfaces a LOD and no opaque shadow proxy to stand in for it");
						}
						inst.model = allocator.allocate<zonetool::iw7::XModel>();
						inst.model->name = hidden_model;
					};
					std::vector<map_entities::entity> script_models;
					std::map<std::string, unsigned int> moved;
					unsigned int scaled = 0;

					for (auto i = 0u; i < bo3_count; i++)
					{
						if (clusters.cluster_of[i] >= 0 || clusters.world_of[i] >= 0)
						{
							continue;
						}
						const auto& src = asset->dpvs.smodelDrawInsts[i];
						auto& dst = world->dpvs.smodelDrawInsts[clusters.iw7_index[i]];

						std::memcpy(&dst.placement, &src.placement, sizeof(dst.placement));

						auto& stub = model_stubs[src.model];
						if (!stub)
						{
							stub = allocator.allocate<zonetool::iw7::XModel>();
							stub->name = allocator.duplicate_string(xmodel::iw7_name(src.model->name));
						}
						dst.model = stub;

						dst.modelLightmapInfo.lightmapIndex = -1;
						dst.cullDist = 0;

						// scale modifier 6 = 1.0 (iw7_ship table 0x141568050; 0x140DCB890 divides the LOD distance by it), as most stock
						// static models; LOD distances are xmodel.cpp's
						dst.flags = zonetool::iw7::STATIC_MODEL_FLAG_SCALE_6 | zonetool::iw7::STATIC_MODEL_FLAG_LIGHTGRID_LIGHTING;
						// a shadow proxy draws its shadow (static_model_clusters.hpp)
						const auto casts_shadow = !(src.flags & t7_smodel_no_shadow) && !clusters.proxied[i];
						if (!casts_shadow)
						{
							dst.flags |= zonetool::iw7::STATIC_MODEL_FLAG_NO_CAST_SHADOW;
							no_shadow++;
						}
						dst.sunShadowFlags = casts_shadow ? sun_mask : 0;

						dst.primaryLightEnvIndex = 0;
						dst.reflectionProbeIndex = 0;
						dst.firstMtlSkinIndex = 0;
						dst.transientZone = 0;

						auto& inst = world->dpvs.smodelInsts[clusters.iw7_index[i]];
						set_bounds(inst.bounds, src.mins, src.maxs);
						std::memcpy(inst.lightingOrigin, src.center, sizeof(inst.lightingOrigin));

						auto [known, added] = too_many_surfaces.try_emplace(src.model, false);
						if (added)
						{
							for (const auto& [mesh, materials] : xmodel::lod_meshes(src.model))
							{
								if (mesh && mesh->numSurfs > 16)
								{
									known->second = true;
								}
							}
						}
						if (known->second)
						{
							dst.cullDist = 1;
							hide(dst);
							if (!(dst.flags & zonetool::iw7::STATIC_MODEL_FLAG_NO_CAST_SHADOW))
							{
								dst.flags |= zonetool::iw7::STATIC_MODEL_FLAG_NO_CAST_SHADOW;
								no_shadow++;
							}
							dst.sunShadowFlags = 0;

							// angles from the axis (forward, left, up): IW's yaw, pitch (down positive), roll
							const auto& p = src.placement;
							constexpr auto deg = 180.0 / 3.14159265358979323846;
							const auto yaw = std::atan2(p.axis[0][1], p.axis[0][0]) * deg;
							const auto pitch = std::atan2(-p.axis[0][2], std::hypot(p.axis[0][0], p.axis[0][1])) * deg;
							const auto roll = std::atan2(p.axis[1][2], p.axis[2][2]) * deg;
							map_entities::entity e{};
							e.set("classname", "script_model");
							e.set("model", src.model->name);
							e.set("origin", utils::string::va("%g %g %g", p.origin[0], p.origin[1], p.origin[2]));
							e.set("angles", utils::string::va("%g %g %g", pitch, yaw, roll));
							script_models.push_back(std::move(e));
							moved[src.model->name]++;
							if (std::fabs(p.scale - 1.0f) > 1e-4f)
							{
								scaled++;
							}
						}
					}

					// the merged models (static_model_clusters): placed unrotated at their members' centroid, which their vertices
					// are relative to; a posed one of more than 16 surfaces a LOD is drawn by a script_model, as above
					for (const auto& group : clusters.clusters)
					{
						const auto index = clusters.iw7_index[group.members.front()];
						auto& dst = world->dpvs.smodelDrawInsts[index];
						for (auto k = 0; k < 3; k++)
						{
							dst.placement.origin[k] = group.origin[k];
							for (auto c = 0; c < 3; c++)
							{
								dst.placement.axis[k][c] = k == c ? 1.0f : 0.0f;
							}
						}
						dst.placement.scale = 1.0f;
						dst.model = allocator.allocate<zonetool::iw7::XModel>();
						dst.model->name = allocator.duplicate_string(group.name);
						dst.modelLightmapInfo.lightmapIndex = -1;
						dst.cullDist = group.too_many_surfaces ? 1 : 0;
						dst.flags = zonetool::iw7::STATIC_MODEL_FLAG_SCALE_6 | zonetool::iw7::STATIC_MODEL_FLAG_LIGHTGRID_LIGHTING;
						// a shadow proxy draws its shadow when it does every member's (static_model_clusters.hpp)
						const auto by_proxies = std::ranges::all_of(group.members, [&](const unsigned int i) { return clusters.proxied[i] != 0; });
						const auto shadow = group.casts_shadow && !group.too_many_surfaces && !by_proxies;
						if (!shadow)
						{
							dst.flags |= zonetool::iw7::STATIC_MODEL_FLAG_NO_CAST_SHADOW;
							no_shadow++;
						}
						dst.sunShadowFlags = shadow ? sun_mask : 0;
						dst.primaryLightEnvIndex = 0;
						dst.reflectionProbeIndex = 0;
						dst.firstMtlSkinIndex = 0;
						dst.transientZone = 0;

						auto& inst = world->dpvs.smodelInsts[index];
						set_bounds(inst.bounds, group.mins, group.maxs);
						std::memcpy(inst.lightingOrigin, group.lighting_origin, sizeof(inst.lightingOrigin));

						if (group.too_many_surfaces)
						{
							hide(dst);
							map_entities::entity e{};
							e.set("classname", "script_model");
							e.set("model", group.name);
							e.set("origin", utils::string::va("%g %g %g", group.origin[0], group.origin[1], group.origin[2]));
							e.set("angles", "0 0 0");
							script_models.push_back(std::move(e));
							moved[group.name]++;
						}
					}

					// the shadow proxies (static_model_clusters.hpp): placed unrotated at their members' centroid, which their
					// vertices are relative to; their materials are shadow-only, the camera never draws them
					for (auto k = 0u; k < clusters.proxies.size(); k++)
					{
						const auto& proxy = clusters.proxies[k];
						const auto index = clusters.proxy_begin + k;
						auto& dst = world->dpvs.smodelDrawInsts[index];
						for (auto a = 0; a < 3; a++)
						{
							dst.placement.origin[a] = proxy.origin[a];
							for (auto c = 0; c < 3; c++)
							{
								dst.placement.axis[a][c] = a == c ? 1.0f : 0.0f;
							}
						}
						dst.placement.scale = 1.0f;
						dst.model = allocator.allocate<zonetool::iw7::XModel>();
						dst.model->name = allocator.duplicate_string(proxy.name);
						dst.modelLightmapInfo.lightmapIndex = -1;
						dst.cullDist = 0;
						dst.flags = zonetool::iw7::STATIC_MODEL_FLAG_SCALE_6 | zonetool::iw7::STATIC_MODEL_FLAG_LIGHTGRID_LIGHTING;
						dst.sunShadowFlags = sun_mask;
						dst.primaryLightEnvIndex = 0;
						dst.reflectionProbeIndex = 0;
						dst.firstMtlSkinIndex = 0;
						dst.transientZone = 0;

						auto& inst = world->dpvs.smodelInsts[index];
						set_bounds(inst.bounds, proxy.mins, proxy.maxs);
						std::memcpy(inst.lightingOrigin, proxy.origin, sizeof(inst.lightingOrigin));
					}

					if (!script_models.empty())
					{
						std::string text;
						for (const auto& [name, n] : moved)
						{
							text += utils::string::va(" %s %u", name.data(), n);
						}
						ZONETOOL_INFO("gfxworld: %zu static models of LODs over 16 surfaces placed as script_models:%s", script_models.size(),
							text.data());
						if (scaled)
						{
							ZONETOOL_WARNING("gfxworld: %u of those static models are scaled; IW7's script_model spawns at scale 1", scaled);
						}
					}
					map_entities::set_script_models(std::move(script_models));

					// the Umbra object IDs of static models are index + 1 (see the tome), so lodData is
					// the identity; the IW5 port ships the same and draws its models
					world->dpvs.lodData = allocator.allocate_array<unsigned int>(count + 1);
					for (auto i = 0u; i < count; i++)
					{
						world->dpvs.lodData[i + 1] = i;
					}

					world->dpvs.sortedSmodelIndices = allocator.allocate_array<unsigned short>(count);
					for (auto i = 0u; i < count; i++)
					{
						world->dpvs.sortedSmodelIndices[i] = static_cast<unsigned short>(i);
					}

					ZONETOOL_INFO("gfxworld: %u static models (%zu models placed single, %zu merged ones, %zu shadow proxies, %u without "
						"shadows of their own)", count, model_stubs.size(), clusters.clusters.size(), clusters.proxies.size(), no_shadow);
					{
						std::map<std::string, unsigned int> brush;
						for (auto i = 0u; i < bo3_count; i++)
						{
							const auto* n = asset->dpvs.smodelDrawInsts[i].model->name;
							if (n && n[0] == '*')
							{
								brush[n]++;
							}
						}
						for (const auto& [n, c] : brush)
						{
							ZONETOOL_INFO("gfxworld: %u static models name the brush model %s", c, n.data());
						}
					}
				}

				// ---- brush models -----------------------------------------------------------------

				void convert_brush_models(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, const surface_plan& plan,
					utils::memory::allocator& allocator)
				{
					world->modelCount = asset->modelCount;
					world->models = allocator.allocate_array<zonetool::iw7::GfxBrushModel>(asset->modelCount);
					for (auto m = 0; m < asset->modelCount; m++)
					{
						const auto& src = asset->models[m];
						auto& dst = world->models[m];

						set_bounds(dst.bounds, src.bounds[0], src.bounds[1]);
						dst.radius = std::sqrt(dst.bounds.halfSize[0] * dst.bounds.halfSize[0]
							+ dst.bounds.halfSize[1] * dst.bounds.halfSize[1]
							+ dst.bounds.halfSize[2] * dst.bounds.halfSize[2]);

						if (src.surfaceCount > 0xFFFF)
						{
							ZONETOOL_FATAL("gfxworld: brush model %d has %u surfaces", m, src.surfaceCount);
						}
						dst.surfaceCount = static_cast<unsigned short>(src.surfaceCount);
						dst.startSurfIndex = src.startSurfIndex;

						// model 0 covers the static surfaces, which moved (and the sky box joined them); any other
						// model's surfaces are brush surfaces, which kept their order behind the static ones
						if (m == 0)
						{
							if (src.startSurfIndex == 0 && src.surfaceCount == asset->dpvs.staticSurfaceCount)
							{
								const auto added = (plan.sky != ~0u ? 1u : 0u) + static_cast<unsigned int>(plan.materials.size() - plan.prop_begin);
								if (src.surfaceCount + added > 0xFFFF)
								{
									ZONETOOL_FATAL("gfxworld: brush model 0 would hold %u surfaces", src.surfaceCount + added);
								}
								dst.surfaceCount = static_cast<unsigned short>(src.surfaceCount + added);
							}
						}
						else if (src.surfaceCount && src.startSurfIndex != 0xFFFFFFFF)
						{
							dst.startSurfIndex = plan.remap[src.startSurfIndex];
							for (auto s = src.startSurfIndex; s < src.startSurfIndex + src.surfaceCount; s++)
							{
								const auto moved = plan.remap[s];
								if (moved != dst.startSurfIndex + (s - src.startSurfIndex))
								{
									ZONETOOL_FATAL("gfxworld: brush model %d surface %u moved to %u, out of its run", m, s, moved);
								}
							}
						}
					}
				}

				// ---- Umbra tome -------------------------------------------------------------------

				// The occlusion tome IW7 culls the world with, generated like the IW5 port does
				// (IW5/Converter/IW7/Assets/GfxWorldUmbra.cpp): every static surface is a target
				// named by its sorted slot, the drawn opaque ones also occlude, static models and
				// bounded primary lights are boxes, and anything without bounds is a box around the
				// whole view volume so it is never culled.
				constexpr float umbra_view_half_extent = 262144.0f;
				constexpr auto umbra_box_target = zonetool::iw7::umbra::SCENE_OBJECT_TARGET | zonetool::iw7::umbra::SCENE_OBJECT_VOLUME;

				void generate_umbra_tome(zonetool::iw7::GfxWorld* world, const zonetool::iw7::GfxWorldTransientZone* zone,
					const surface_plan& plan, utils::memory::allocator& allocator)
				{
					namespace umbra = zonetool::iw7::umbra;

					umbra::tome_input input{};
					input.name = world->baseName;
					// BO3 baked the map's visibility with its own smallest hole; a larger one closes the gaps
					// between props, grates and railings, and what is seen through them drops out as the camera moves
					if (umbra_smallest_hole > 0.0f)
					{
						input.params.smallest_hole = umbra_smallest_hole;
						ZONETOOL_INFO("umbra: smallest hole %g (BO3 worldspawn umbraSmallestHole)", umbra_smallest_hole);
					}

					umbra::scene_view_volume volume{};
					for (auto k = 0; k < 3; k++)
					{
						volume.mins[k] = -umbra_view_half_extent;
						volume.maxs[k] = umbra_view_half_extent;
					}
					input.view_volumes.push_back(volume);
					const auto view_box = input.add_box_model(volume.mins, volume.maxs);

					const auto static_count = world->dpvs.staticSurfaceCount;
					std::vector<unsigned int> sorted_slot(static_count);
					for (auto k = 0u; k < static_count; k++)
					{
						sorted_slot[world->dpvs.sortedSurfIndex[k]] = k;
					}

					auto occluders = 0u, triangles = 0u;
					for (auto i = 0u; i < static_count; i++)
					{
						const auto& surface = world->dpvs.surfaces[i];
						const auto& material = *plan.materials[plan.order[i]];
						const auto user_id = umbra::USER_ID_SURFACE | (sorted_slot[i] & umbra::USER_ID_INDEX_MASK);

						if (plan.order[i] == plan.sky)
						{
							// the sky box is seen from everywhere and hides nothing
							input.objects.push_back({ view_box, user_id, umbra_box_target });
							continue;
						}

						umbra::tome_model model{};
						std::vector<int> remap(surface.tris.vertexCount, -1);
						for (auto k = 0u; k < surface.tris.triCount * 3u; k++)
						{
							const auto index = world->draw.indices[surface.tris.baseIndex + k];
							if (remap[index] < 0)
							{
								const auto* xyz = zone->vd.vertices[surface.tris.firstVertex + index].xyz;
								remap[index] = static_cast<int>(model.vertices.size() / 3);
								model.vertices.insert(model.vertices.end(), xyz, xyz + 3);
							}
							model.indices.push_back(static_cast<std::uint32_t>(remap[index]));
						}
						triangles += surface.tris.triCount;
						input.models.emplace_back(std::move(model));

						auto flags = static_cast<std::uint32_t>(umbra::SCENE_OBJECT_TARGET);
						if (material.cls == world_material::surface_class::opaque && !material.alpha_test)
						{
							flags |= umbra::SCENE_OBJECT_OCCLUDER;
							occluders++;
						}
						input.objects.push_back({ static_cast<std::uint32_t>(input.models.size() - 1), user_id, flags });
					}

					// a static model: a target of its members' boxes (a merged one draws where one of its members can be seen, not
					// wherever its whole box can), and its opaque casting triangles as an occluder of their own (props are most of a
					// BO3 map's walls and buildings)
					auto smodel_occluders = 0u, smodel_occluder_triangles = 0u;
					for (auto i = 0u; i < world->dpvs.smodelCount; i++)
					{
						std::vector<std::array<float, 6>> member_bounds;
						umbra::tome_model occluder{};
						static_model_clusters::umbra_geometry(i, member_bounds, occluder.vertices, occluder.indices);
						const auto user_id = umbra::USER_ID_SMODEL | (i + 1);
						if (member_bounds.empty())
						{
							const auto& bounds = world->dpvs.smodelInsts[i].bounds;
							float mins[3], maxs[3];
							for (auto k = 0; k < 3; k++)
							{
								mins[k] = bounds.midPoint[k] - bounds.halfSize[k];
								maxs[k] = bounds.midPoint[k] + bounds.halfSize[k];
							}
							input.objects.push_back({ input.add_box_model(mins, maxs), user_id, umbra_box_target });
						}
						else
						{
							umbra::tome_model boxes{};
							for (const auto& b : member_bounds)
							{
								const auto first = static_cast<std::uint32_t>(boxes.vertices.size() / 3);
								for (auto corner = 0; corner < 8; corner++)
								{
									boxes.vertices.push_back(b[(corner & 1) ? 3 : 0]);
									boxes.vertices.push_back(b[(corner & 2) ? 4 : 1]);
									boxes.vertices.push_back(b[(corner & 4) ? 5 : 2]);
								}
								static constexpr std::uint32_t faces[36] = { 0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4,
									2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5 };
								for (const auto f : faces)
								{
									boxes.indices.push_back(first + f);
								}
							}
							input.models.emplace_back(std::move(boxes));
							input.objects.push_back({ static_cast<std::uint32_t>(input.models.size() - 1), user_id, umbra_box_target });
						}
						if (!occluder.indices.empty())
						{
							smodel_occluders++;
							smodel_occluder_triangles += static_cast<unsigned int>(occluder.indices.size() / 3);
							input.models.emplace_back(std::move(occluder));
							input.objects.push_back({ static_cast<std::uint32_t>(input.models.size() - 1), user_id,
								static_cast<std::uint32_t>(umbra::SCENE_OBJECT_OCCLUDER) });
						}
					}
					ZONETOOL_INFO("umbra: %u static models occlude with %u opaque triangles", smodel_occluders, smodel_occluder_triangles);

					const auto* com_world = comworld::converted();
					auto bounded_lights = 0u;
					for (auto i = 0u; i < world->primaryLightCount; i++)
					{
						auto model = view_box;
						const auto& light = com_world->primaryLights[i];
						if ((light.type == zonetool::iw7::GFX_LIGHT_TYPE_SPOT || light.type == zonetool::iw7::GFX_LIGHT_TYPE_OMNI)
							&& light.radius > 0.0f)
						{
							float mins[3], maxs[3];
							for (auto k = 0; k < 3; k++)
							{
								mins[k] = light.origin[k] - light.radius;
								maxs[k] = light.origin[k] + light.radius;
							}
							model = input.add_box_model(mins, maxs);
							bounded_lights++;
						}
						input.objects.push_back({ model, umbra::USER_ID_PRIMARY_LIGHT | i, umbra_box_target });
					}

					// IW7 takes the reflection probe instances it blends from the tome's visible objects: each is
					// the box around its volume, the sun volume fallbacks the whole view volume
					const auto& probes = world->draw.reflectionProbeData;
					for (auto i = 0u; i < probes.reflectionProbeInstanceCount; i++)
					{
						const auto& obb = probes.reflectionProbeInstances[i].volumeObb;
						auto model = view_box;
						if (std::max({ obb.halfSize[0], obb.halfSize[1], obb.halfSize[2] }) < umbra_view_half_extent)
						{
							const float* axes[3] = { obb.xAxis, obb.yAxis, obb.zAxis };
							float mins[3], maxs[3];
							for (auto k = 0; k < 3; k++)
							{
								const auto extent = std::fabs(axes[0][k]) * obb.halfSize[0] + std::fabs(axes[1][k]) * obb.halfSize[1]
									+ std::fabs(axes[2][k]) * obb.halfSize[2];
								mins[k] = obb.center[k] - extent;
								maxs[k] = obb.center[k] + extent;
							}
							model = input.add_box_model(mins, maxs);
						}
						input.objects.push_back({ model, umbra::USER_ID_REFLECTION_PROBE | i, umbra_box_target });
					}

					ZONETOOL_INFO("umbra: %u surfaces (%u occluders, %u triangles), %u static models, %u of %u primary lights bounded, "
						"%u reflection probe instances", static_count, occluders, triangles, world->dpvs.smodelCount, bounded_lights,
						world->primaryLightCount, probes.reflectionProbeInstanceCount);

					const auto log = [](const std::string& line)
					{
						ZONETOOL_INFO("umbra: %s", line.data());
					};

					// Every object has to come back out of the optimizer, or IW7 never draws it. The
					// optimizer leaves out objects it finds hidden from every view cell, treating gaps under
					// smallest_hole as closed. Those come back as boxes around themselves grown until the
					// optimizer keeps them - a box that reaches past its enclosure is seen from outside it.
					// Boxes around the whole view volume are only the last step: many of them make the
					// optimizer's tile pass very slow.
					constexpr float grow_by[] = { 256.0f, 1024.0f, 4096.0f };
					std::unordered_map<std::uint32_t, std::array<float, 6>> original_bounds; // object index -> mins, maxs
					umbra::tome_result result{};
					for (auto attempt = 0u;; attempt++)
					{
						if (!umbra::generate_tome(input, result, log))
						{
							ZONETOOL_FATAL("umbra: %s", result.error.data());
						}

						const auto ids = umbra::read_user_ids(result.data.data(), result.data.size());
						const std::unordered_set<std::uint32_t> present(ids.begin(), ids.end());
						std::uint32_t dropped[8]{};
						auto missing = 0u;
						for (auto o = 0u; o < input.objects.size(); o++)
						{
							auto& object = input.objects[o];
							if (present.contains(object.user_id) || !(object.flags & umbra::SCENE_OBJECT_TARGET))
							{
								continue;
							}
							missing++;
							dropped[(object.user_id >> 28) & 7]++;
							if (attempt > std::size(grow_by))
							{
								continue;
							}

							auto bounds = original_bounds.find(o);
							if (bounds == original_bounds.end())
							{
								std::array<float, 6> b{ FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
								const auto& v = input.models[object.model].vertices;
								for (auto k = 0u; k + 2 < v.size(); k += 3)
								{
									for (auto c = 0; c < 3; c++)
									{
										b[c] = std::min(b[c], v[k + c]);
										b[3 + c] = std::max(b[3 + c], v[k + c]);
									}
								}
								bounds = original_bounds.emplace(o, b).first;
							}
							if (attempt < std::size(grow_by))
							{
								const auto& b = bounds->second;
								const float mins[3] = { b[0] - grow_by[attempt], b[1] - grow_by[attempt], b[2] - grow_by[attempt] };
								const float maxs[3] = { b[3] + grow_by[attempt], b[4] + grow_by[attempt], b[5] + grow_by[attempt] };
								object.model = input.add_box_model(mins, maxs);
							}
							else
							{
								object.model = view_box;
							}
							object.flags = umbra_box_target;
						}
						if (!missing)
						{
							break;
						}
						if (attempt > std::size(grow_by))
						{
							ZONETOOL_FATAL("umbra: %u objects are still missing from the tome as boxes around the whole view volume", missing);
						}
						const auto how = attempt < std::size(grow_by) ? "their bounds grown by " + std::to_string(static_cast<int>(grow_by[attempt]))
							: std::string("boxes around the view volume");
						ZONETOOL_INFO("umbra: the optimizer left out %u of %zu objects (%u surfaces, %u static models, %u lights, %u reflection "
							"probe instances); generating again with them as %s", missing, input.objects.size(), dropped[0], dropped[1],
							dropped[4], dropped[5], how.data());
					}

					const auto& stats = result.stats;
					ZONETOOL_INFO("umbra: tome version 0x%X, %u bytes, %d objects (%u user IDs), %d tiles (%d leaf), %u cells "
						"(%u without portals), %u cell portals, generated in %.1f s", stats.version, stats.size, stats.object_count,
						stats.user_id_count, stats.tile_count, stats.leaf_tile_count, stats.cell_count, stats.cells_without_portals,
						stats.portal_count, result.generation_seconds);

					world->numUmbraGates = 0;
					world->umbraGates = nullptr;
					world->umbraTomeSize = static_cast<unsigned int>(result.data.size());
					world->umbraTomeData = allocator.allocate_array<char>(result.data.size());
					std::memcpy(world->umbraTomeData, result.data.data(), result.data.size());
				}
			}

			zonetool::iw7::GfxWorld* convert(GfxWorld* asset, utils::memory::allocator& allocator)
			{
				auto* world = allocator.allocate<zonetool::iw7::GfxWorld>();

				const auto* com_world = comworld::converted();
				if (!com_world)
				{
					ZONETOOL_FATAL("gfxworld: the ComWorld has to be converted first (primary lights)");
				}
				comworld::export_flicker(asset);

				// ZT_PROBE_SAMPLE="x y z;x y z;...": BO3's state 0 probe diffuse at each point for +-x, +-y, +z normals, then exit
				if (const auto* points = std::getenv("ZT_PROBE_SAMPLE"))
				{
					probe_lighting::evaluator eval(asset, map::lighting_state());
					for (const auto& text : utils::string::split(points, ';'))
					{
						float p[3]{};
						if (std::sscanf(text.data(), "%f %f %f", &p[0], &p[1], &p[2]) != 3)
						{
							continue;
						}
						const auto v = eval.volume_at(p);
						eval.load(v);
						const float normals[5][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 } };
						float out[5][3]{};
						eval.diffuse(v, p, normals, 5, out);
						ZONETOOL_INFO("probe sample (%g %g %g) volume %u: +x %.4f %.4f %.4f | -x %.4f %.4f %.4f | +y %.4f %.4f %.4f | -y %.4f %.4f %.4f | +z %.4f %.4f %.4f",
							p[0], p[1], p[2], v, out[0][0], out[0][1], out[0][2], out[1][0], out[1][1], out[1][2], out[2][0], out[2][1], out[2][2],
							out[3][0], out[3][1], out[3][2], out[4][0], out[4][1], out[4][2]);
						eval.unload(v);
					}
					std::fflush(stdout);
					std::_Exit(0);
				}

				// primary light 0 is the null light and 1 the sun (comworld.cpp)
				constexpr auto sun_count = 1u;

				const auto base_name = map::map_name(asset->name);
				world->name = allocator.duplicate_string(map::bsp_name(asset->name));
				world->baseName = allocator.duplicate_string(base_name);
				world->bspVersion = 159;
				world->surfaceCount = static_cast<unsigned int>(asset->surfaceCount);

				world->lastSunPrimaryLightIndex = sun_count;
				world->primaryLightCount = com_world->primaryLightCount;
				world->movingScriptablePrimaryLightCount = 0;

				world->sortKeyLitDecal = sort_key_lit_decal;
				world->sortKeyEffectDecal = sort_key_effect_decal;
				world->sortKeyTopDecal = sort_key_top_decal;
				world->sortKeyEffectAuto = sort_key_effect_auto;
				world->sortKeyDistortion = sort_key_distortion;
				world->sortKeyEffectDistortion = sort_key_effect_distortion;
				world->sortKey2D = sort_key_2d;
				world->sortKeyOpaqueBegin = sort_key_opaque_begin;
				world->sortKeyOpaqueEnd = sort_key_opaque_end;
				world->sortKeyDecalBegin = sort_key_decal_begin;
				world->sortKeyDecalEnd = sort_key_decal_end;
				world->sortKeyTransBegin = sort_key_trans_begin;
				world->sortKeyTransEnd = sort_key_trans_end;
				world->sortKeyEmissiveBegin = sort_key_emissive_begin;
				world->sortKeyEmissiveEnd = sort_key_emissive_end;

				convert_planes_and_nodes(asset, world, allocator);

				// the sky: a cube and material from BO3's skybox, drawn on a box around the world (stock IW7)
				const auto sky = world_sky::convert(asset);
				world_material::info sky_info{};
				if (sky)
				{
					sky_info.name = sky->material;
					sky_info.cls = world_material::surface_class::opaque;
					sky_info.sort_key = sky->sort_key;
					sky_info.casts_shadow = false;
					sky_info.lightmapped = false;
				}

				auto* zone = convert_transient_zone(asset, base_name, sky.has_value(), allocator);
				world->draw.transientZoneCount = 1;
				world->draw.transientZones[0] = zone;
				// which static models IW7 draws merged, before the cells name them (static_model_clusters.hpp)
				static_model_clusters::make_plan(asset, map::map_name(asset->name));

				// IW7's lit shaders sample the world's IES lookup (code texture 4) whatever light they draw with; without one
				// the first lit draw fails ("Tried to use codeTexture 4 'iesLookupTexture' when it isn't valid"). The converted
				// lights are all light_point_linear (no IES profile), so the stock "*ieslookup" serves
				world->draw.iesLookupTexture = allocator.allocate<zonetool::iw7::GfxImage>();
				world->draw.iesLookupTexture->name = "*ieslookup";

				// BO3's world fog, for the level's vision (fog and exposure are written there, not in the GfxWorld)
				{
					const auto& f = asset->worldSpawnConfig.defaultFog;
					ZONETOOL_INFO("gfxworld: BO3 default fog: colour %g %g %g opacity %g intensity %g, base/half dist %g %g, base/half height %g %g, "
						"sun colour %g %g %g opacity %g intensity %g inner/outer %g %g, atmosphere fog %g %g %g density %g, haze %g %g %g "
						"base/fade %g %g density %g spread %g; transition %g s; %u fog volumes, %u lit fog volumes; exp EV auto %u",
						f._fogcolor[0], f._fogcolor[1], f._fogcolor[2], f.fogopacity, f.fogintensity, f.basedist, f.halfdist, f.baseheight, f.halfheight,
						f._sunfogcolor[0], f._sunfogcolor[1], f._sunfogcolor[2], f.sunfogopacity, f.sunfogintensity, f.sunfoginner, f.sunfogouter,
						f._atmospherefogcolor[0], f._atmospherefogcolor[1], f._atmospherefogcolor[2], f.atmospherefogdensity,
						f._atmospherehazecolor[0], f._atmospherehazecolor[1], f._atmospherehazecolor[2], f.atmospherehazebasedist,
						f.atmospherehazefadedist, f.atmospherehazedensity, f.atmospherehazespread, asset->worldSpawnConfig.fogTransitionTime,
						asset->worldFogVolumeCount, asset->litFogVolumeCount, asset->worldSpawnConfig.expEvAuto);
					// BO3's exposure pass (the EV -> exposure multiplier the lighting units are exposed with) and its tone mapping /
					// colour grading (the LUT builder, the pass that applies it, the LUT images), written out for the level's vision
					// and colour LUT: dump/shaders_bo3/<set>_<shader>_<method>.cso, <material>_<technique>_<method>_<vs|ps>.cso,
					// img_<image>.raw (u32 width, height, depth, faces, levels, DXGI format, then the pixels in D3D subresource order)
					const auto write_blob = [](const std::string& path, const void* data, const std::size_t size)
					{
						filesystem::file file(path);
						file.open("wb");
						file.write(data, size, 1);
						file.close();
					};
					const auto dump_image = [&](const std::string& image_name)
					{
						const auto* image = gfximage::find_image(image_name);
						gfximage::image_pixels pixels;
						if (!image || !gfximage::get_pixels(image, pixels))
						{
							ZONETOOL_INFO("gfxworld: BO3 image %s is not loaded", image_name.data());
							return;
						}
						std::vector<std::uint8_t> out(24);
						const std::uint32_t header[6] = { pixels.width, pixels.height, pixels.depth, pixels.faces, pixels.levels,
							static_cast<std::uint32_t>(pixels.format) };
						std::memcpy(out.data(), header, sizeof(header));
						out.insert(out.end(), pixels.data.begin(), pixels.data.end());
						write_blob(utils::string::va("shaders_bo3/img_%s.raw", image_name.data()), out.data(), out.size());
						ZONETOOL_INFO("gfxworld: BO3 image %s written (%ux%ux%u, %u faces, %u levels, format %u)", image_name.data(), pixels.width,
							pixels.height, pixels.depth, pixels.faces, pixels.levels, static_cast<unsigned int>(pixels.format));
					};
					// and the materials named in ZT_DUMP_MATERIAL_SHADERS (comma separated), for reading a template the
					// converter does not handle yet
					std::vector<std::string> shader_materials = { "bloom_apply_lut", "luts_t7_scurve", "luts_t7_default", "create_lut2dv" };
					if (const auto* extra = std::getenv("ZT_DUMP_MATERIAL_SHADERS"))
					{
						for (const auto& name : utils::string::split(extra, ','))
						{
							shader_materials.push_back(name);
						}
					}
					for (const auto& material_name_string : shader_materials)
					{
						const auto* material_name = material_name_string.data();
						const auto* entry = zonetool::t7::DB_FindXAssetEntry(ASSET_TYPE_MATERIAL, material_name, false);
						const auto* material = entry ? reinterpret_cast<const Material*>(entry->asset.header.data) : nullptr;
						if (!material || !material->techniqueSet)
						{
							ZONETOOL_INFO("gfxworld: BO3 material %s is not loaded", material_name);
							continue;
						}
						for (auto t = 0; t < 12; t++)
						{
							const auto* technique = material->techniqueSet->techniques[t];
							if (!technique)
							{
								continue;
							}
							for (auto m = 0; m < 8; m++)
							{
								const auto& method = technique->drawMethods[m];
								if (method.vertexShader && method.vertexShader->prog.loadDef.program)
								{
									write_blob(utils::string::va("shaders_bo3/%s_%d_%d_vs.cso", material_name, t, m), method.vertexShader->prog.loadDef.program,
										method.vertexShader->prog.loadDef.programSize);
								}
								if (method.pixelShader && method.pixelShader->prog.loadDef.program)
								{
									write_blob(utils::string::va("shaders_bo3/%s_%d_%d_ps.cso", material_name, t, m), method.pixelShader->prog.loadDef.program,
										method.pixelShader->prog.loadDef.programSize);
									ZONETOOL_INFO("gfxworld: BO3 material %s (techset %s) technique %d method %d: %s", material_name, material->techniqueSet->name,
										t, m, technique->name ? technique->name : "?");
								}
							}
						}
						for (auto i = 0; i < material->textureCount; i++)
						{
							const auto& def = material->textureTable[i];
							const auto* image = def.image;
							ZONETOOL_INFO("gfxworld: BO3 material %s texture %d: slot 0x%08X image %s", material_name, i, def.nameHash,
								image && image->name ? image->name : "?");
							if (image && image->name)
							{
								dump_image(image->name);
							}
						}
					}
					for (const auto* image_name : { "img_luts_t7_scurve", "identity_lut_win" })
					{
						dump_image(image_name);
					}
					for (const auto* set_name : { "exposure_compute", "create_lut2dv_compute", "create_lut2dv_compute_alt" })
					{
						const auto* entry = zonetool::t7::DB_FindXAssetEntry(ASSET_TYPE_COMPUTE_SHADER_SET, set_name, false);
						const auto* set = entry ? reinterpret_cast<const MaterialComputeShaderSet*>(entry->asset.header.data) : nullptr;
						if (!set || !set->shaders)
						{
							ZONETOOL_INFO("gfxworld: BO3 compute shader set %s is not loaded", set_name);
							continue;
						}
						for (auto i = 0u; i < set->count; i++)
						{
							const auto* shader = set->shaders[i];
							if (!shader)
							{
								continue;
							}
							for (auto m = 0; m < 8; m++)
							{
								const auto& def = shader->methods[m].prog.loadDef;
								if (!def.program || !def.programSize)
								{
									continue;
								}
								filesystem::file file(utils::string::va("shaders_bo3/%s_%s_%d.cso", set_name, shader->name ? shader->name : "unnamed", m));
								file.open("wb");
								file.write(def.program, def.programSize, 1);
								file.close();
								ZONETOOL_INFO("gfxworld: BO3 compute shader %s / %s method %d written (%u bytes)", set_name, shader->name ? shader->name : "?", m,
									def.programSize);
							}
						}
					}
					// BO3's exposure (per area, per lighting-state bank) and colour grading, for the level's vision
					for (auto v = 0u; v < asset->exposureVolumeCount; v++)
					{
						const auto& e = asset->exposureVolumes[v];
						ZONETOOL_INFO("gfxworld: BO3 exposure volume %u: priority %d, banks 0x%X, evmin %u %u %u %u, evmax %u %u %u %u, evcmp %u %u %u %u, "
							"exposure %g %g %g %g, adaptation %g %g %g %g", v, e.priority, e.bankMask, e.evmin[0], e.evmin[1], e.evmin[2],
							e.evmin[3], e.evmax[0], e.evmax[1], e.evmax[2], e.evmax[3], e.evcmp[0], e.evcmp[1], e.evcmp[2], e.evcmp[3], e.exposure[0],
							e.exposure[1], e.exposure[2], e.exposure[3], e.adaptation[0], e.adaptation[1], e.adaptation[2], e.adaptation[3]);
					}
					// ZT_SST_EXPORT=<file>: BO3's sun shadow data raw (volume 0, level 0: the SST config, the min/max tree
					// and its shadow tree elements when resident), for offline analysis
					if (const char* out = std::getenv("ZT_SST_EXPORT"); out && asset->sunVolumeCount)
					{
						const auto& sv = asset->sunVolumes[0];
						if (auto* sst_file = std::fopen(out, "wb"))
						{
							const auto& mm = sv.sstMinMaxs[0];
							std::fwrite(&sv.multiResSST[0].config, sizeof(sv.multiResSST[0].config), 1, sst_file);
							std::fwrite(&mm.mip0Width, sizeof(int), 1, sst_file);
							std::fwrite(&mm.mip0Height, sizeof(int), 1, sst_file);
							std::fwrite(&mm.sstMinMaxCount, sizeof(unsigned int), 1, sst_file);
							std::fwrite(&mm.treeMax, sizeof(float), 1, sst_file);
							const std::uint32_t has_max = mm.sstMaxCPU ? 1 : 0;
							std::fwrite(&has_max, sizeof(has_max), 1, sst_file);
							if (mm.sstMaxCPU)
							{
								std::fwrite(mm.sstMaxCPU, sizeof(unsigned short), mm.sstMinMaxCount, sst_file);
							}
							std::fclose(sst_file);
							ZONETOOL_INFO("gfxworld: BO3 SST exported to %s", out);
						}
					}
					ZONETOOL_INFO("gfxworld: main sun volume %u (map::main_sun_volume)", map::main_sun_volume(asset));
					for (auto v = 0u; v < asset->sunVolumeCount; v++)
					{
						const auto& sv = asset->sunVolumes[v];
						{
							std::string suns;
							for (auto s = 0; s < 4; s++)
							{
								const auto& st = sv.sun.settings[s];
								suns += utils::string::va(" [state %d%s: pitch %g yaw %g intensity %g colour %g %g %g shadow %d]", s,
									(sv.sun.lightStateMask & (1u << s)) ? "" : " off", st.pitch, st.yaw, st.intensity, st.color[0], st.color[1], st.color[2], st.shadow ? 1 : 0);
							}
							ZONETOOL_INFO("gfxworld: BO3 sun volume %u: shadow split distance %g, bias scale %g; planes %d..%d, export flags 0x%X, lighting state mask 0x%X, sun state mask 0x%X,%s",
								v, sv.sun.shadowSplitDistance, sv.sun.shadowBiasScale, sv.planeStart, sv.planeStart + sv.planeCount, sv.lightExportFlags, sv.lightingStateMask, sv.sun.lightStateMask, suns.data());
							for (auto b = 0; b < 4; b++)
							{
								const auto& sst = sv.multiResSST[b].config.uniformMip;
								const auto& mm = sv.sstMinMaxs[b];
								ZONETOOL_INFO("gfxworld: BO3 sun volume %u SST %d: pitch %g yaw %g, %g x %g tiles, %g inches a texel, span %g; min/max tree %d x %d, %u values, max %g",
									v, b, sst.pitch, sst.yaw, sst.dimensionInTiles[0], sst.dimensionInTiles[1], sst.inchesPerTexel, sst.spanInInches,
									mm.mip0Width, mm.mip0Height, mm.sstMinMaxCount, mm.treeMax);
							}
							if (const char* probe = std::getenv("ZT_SUN_PROBE"))
							{
								float p[3]{};
								if (std::sscanf(probe, "%f %f %f", &p[0], &p[1], &p[2]) == 3)
								{
									auto lo = FLT_MAX, hi = -FLT_MAX;
									for (auto k = 0; k < sv.planeCount; k++)
									{
										const auto& pl = asset->sunVolumePlanes[sv.planeStart + k];
										const auto d = pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] - pl[3];
										lo = std::min(lo, d);
										hi = std::max(hi, d);
									}
									{
										const auto& sst = sv.multiResSST[0].config.uniformMip;
										const auto& m = sst.wldToPinTransform;
										float q[4];
										for (auto r = 0; r < 4; r++)
										{
											q[r] = m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3];
										}
										float qc[4];
										for (auto c = 0; c < 4; c++)
										{
											qc[c] = m[0][c] * p[0] + m[1][c] * p[1] + m[2][c] * p[2] + m[3][c];
										}
										const auto& mm = sv.sstMinMaxs[0];
										ZONETOOL_INFO("gfxworld: sun probe point in SST 0: rows (%g %g %g %g), columns (%g %g %g %g); transform rows %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g",
											q[0], q[1], q[2], q[3], qc[0], qc[1], qc[2], qc[3], m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3],
											m[2][0], m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3]);
										if (mm.sstMaxCPU && sst.spanInInches > 0.0f)
										{
											// candidate mapping: uv = lateral / span + 0.5; a 7 x 7 neighbourhood of the max tree's first level
											const auto cu = qc[0] / sst.spanInInches + 0.5f, cv = qc[1] / sst.spanInInches + 0.5f;
											const auto tx = static_cast<int>(cu * mm.mip0Width), ty = static_cast<int>(cv * mm.mip0Height);
											std::string grid;
											for (auto y = ty - 3; y <= ty + 3; y++)
											{
												for (auto x = tx - 3; x <= tx + 3; x++)
												{
													const auto inside = x >= 0 && y >= 0 && x < mm.mip0Width && y < mm.mip0Height;
													grid += utils::string::va(" %6.0f", inside ? mm.sstMaxCPU[y * mm.mip0Width + x] / 65535.0f * mm.treeMax : -1.0f);
												}
												grid += " |";
											}
											ZONETOOL_INFO("gfxworld: sun probe SST 0 max tree around (%d, %d) (uv %g %g), point depth %g:%s", tx, ty, cu, cv, qc[2], grid.data());
										}
										if (false)
										{
											for (const auto* uv : { q, qc })
											{
												const auto x = static_cast<int>(uv[0] * mm.mip0Width), y = static_cast<int>(uv[1] * mm.mip0Height);
												if (x >= 0 && y >= 0 && x < mm.mip0Width && y < mm.mip0Height)
												{
													ZONETOOL_INFO("gfxworld: sun probe SST 0 max tree at (%d, %d): %u (of 65535, tree max %g)", x, y,
														mm.sstMaxCPU[y * mm.mip0Width + x], mm.treeMax);
												}
											}
										}
									}
									ZONETOOL_INFO("gfxworld: sun probe point vs sun volume %u planes: dot - w from %g to %g (all <= 0 or all >= 0 = inside)", v, lo, hi);
								}
							}
						}
						for (auto b = 0; b < 4; b++)
						{
							ZONETOOL_INFO("gfxworld: BO3 sun volume %u exposure grid %d: %u probes, base value %g, LUT %u, sky %u", v, b,
								sv.exposureGrid[b].nProbe, sv.exposureGrid[b].baseValue, sv.idxLUT, sv.idxSKY);
						}
					}
					for (auto v = 0u; v < asset->lutVolumeCount; v++)
					{
						const auto& l = asset->lutVolumes[v];
						ZONETOOL_INFO("gfxworld: BO3 LUT volume %u: LUT %u, control 0x%X, transition %g s", v, l.lutIndex, l.control, l.lutTransitionTime);
					}
					for (auto v = 0u; v < asset->litFogVolumeCount; v++)
					{
						const auto& vol = asset->litFogVolumes[v];
						ZONETOOL_INFO("gfxworld: BO3 lit fog volume %u: bounds (%g %g %g)-(%g %g %g), ambient %g %g %g, control 0x%X, priority %u, fog time %g",
							v, vol.mins[0], vol.mins[1], vol.mins[2], vol.maxs[0], vol.maxs[1], vol.maxs[2], vol.ambientColor[0], vol.ambientColor[1],
							vol.ambientColor[2], vol.control, vol.priority, vol.fogtime);
						for (auto b = 0; b < 4; b++)
						{
							const auto& k = vol.bank[b];
							ZONETOOL_INFO("gfxworld: BO3 lit fog volume %u bank %d: extinction colour %g %g %g, sun colour override %g %g %g %g, base/half dist %g %g, "
								"base/half height %g %g, probe bake density scale %g, distribution %g, albedo %g, density scale %g, probe contribution %g, "
								"sun intensity scale %g, max lit sun / omni-spot distance %g %g", v, b, k.extcolor[0], k.extcolor[1], k.extcolor[2],
								k.suncoloroverride[0], k.suncoloroverride[1], k.suncoloroverride[2], k.suncoloroverride[3], k.basedist, k.halfdist,
								k.baseheight, k.halfheight, k.probebakelitfogdensityscaler, k.distribution, k.albedo, k.densityScaler,
								k.probeContributionScaler, k.sunintensityscale, k.maxlitsunfogdistance, k.maxlitomnispotfogdistance);
						}
					}
					for (auto v = 0u; v < asset->worldFogVolumeCount; v++)
					{
						const auto& vol = asset->worldFogVolumes[v];
						ZONETOOL_INFO("gfxworld: BO3 fog volume %u: planes %d+%d, bounds (%g %g %g)-(%g %g %g), transition %g s, control 0x%X, state mask 0x%X",
							v, vol.planeStart, vol.planeCount, vol.mins[0], vol.mins[1], vol.mins[2], vol.maxs[0], vol.maxs[1], vol.maxs[2],
							vol.fogTransitionTime, vol.controlEx, vol.lightingStateMask);
						for (auto bank = 0; bank < 4; bank++)
						{
							const auto& b = vol.bank[bank];
							ZONETOOL_INFO("gfxworld: BO3 fog volume %u bank %d: colour %g %g %g opacity %g intensity %g, base/half dist %g %g, base/half height %g %g, "
								"sky half height offset %g, probe bake density %g, sun colour %g %g %g opacity %g intensity %g inner/outer %g %g pitch/yaw offset %g %g, "
								"atmosphere colour %g %g %g density %g, haze %g %g %g base/fade %g %g density %g spread %g, inscatter %g extinction %g sun %g pbr %g, "
								"sky size %g", v, bank, b._fogcolor[0], b._fogcolor[1], b._fogcolor[2], b.fogopacity, b.fogintensity, b.basedist, b.halfdist,
								b.baseheight, b.halfheight, b.skyhalfheightoffset, b.probebakeworldfogdensityscaler, b._sunfogcolor[0], b._sunfogcolor[1],
								b._sunfogcolor[2], b.sunfogopacity, b.sunfogintensity, b.sunfoginner, b.sunfogouter, b.sunPitchOffset, b.sunYawOffset,
								b._atmospherefogcolor[0], b._atmospherefogcolor[1], b._atmospherefogcolor[2], b.atmospherefogdensity, b._atmospherehazecolor[0],
								b._atmospherehazecolor[1], b._atmospherehazecolor[2], b.atmospherehazebasedist, b.atmospherehazefadedist, b.atmospherehazedensity,
								b.atmospherehazespread, b.atmosphereinscatterstrength, b.atmosphereextinctionstrength, b.atmospheresunstrength,
								b.atmospherepbramount, b.worldfogskysize);
						}
					}
				}

				{
					// the materials of the models the GfxWorld draws, whose decal layers take world decal keys
					std::vector<const Material*> props;
					std::unordered_set<const Material*> seen;
					for (const auto& group : static_model_clusters::current().world_clusters)
					{
						for (const auto i : group.members)
						{
							for (const auto& [mesh, mats] : xmodel::lod_meshes(asset->dpvs.smodelDrawInsts[i].model))
							{
								for (const auto* material : mats)
								{
									if (material && seen.insert(material).second)
									{
										props.push_back(material);
									}
								}
							}
						}
					}
					world_material::prepare(asset, props);
				}
				// which static models' sun shadows the shadow proxies draw (static_model_clusters.hpp), before the static model
				// count is used
				static_model_clusters::plan_shadow_proxies(asset, map::map_name(asset->name));

				// the world groups' surfaces (their members' LODs are known now), then the cells: a leaf a group, its surfaces'
				// sorted slots after the sky's
				const auto props = place_world_groups(asset, zone, allocator);
				{
					std::vector<extra_leaf> leaves;
					const auto first_slot = asset->dpvs.staticSurfaceCount + (sky ? 1u : 0u);
					for (auto g = 0u; g < props.group_begin.size(); g++)
					{
						const auto begin = props.group_begin[g];
						const auto end = g + 1 < props.group_begin.size() ? props.group_begin[g + 1] : static_cast<unsigned int>(props.surfaces.size());
						if (begin == end)
						{
							continue;
						}
						extra_leaf leaf{};
						leaf.begin = first_slot + begin;
						leaf.count = end - begin;
						for (auto k = 0; k < 3; k++)
						{
							leaf.mins[k] = FLT_MAX;
							leaf.maxs[k] = -FLT_MAX;
						}
						for (auto s = begin; s < end; s++)
						{
							for (auto k = 0; k < 3; k++)
							{
								leaf.mins[k] = std::min(leaf.mins[k], props.surfaces[s].mins[k]);
								leaf.maxs[k] = std::max(leaf.maxs[k], props.surfaces[s].maxs[k]);
							}
						}
						leaves.push_back(leaf);
					}
					convert_cells(asset, world, zone, sky.has_value(), leaves, allocator);
				}

				// vis-data word counts (stock formulas); surfaces are addressed by static index only
				const auto static_count = asset->dpvs.staticSurfaceCount + (sky ? 1u : 0u) + static_cast<unsigned int>(props.surfaces.size());
				world->dpvs.surfaceVisDataCount = (static_count + 31) >> 5;
				world->dpvs.smodelVisDataCount = (static_model_clusters::current().iw7_count + 31) >> 5;
				world->dpvs.primaryLightVisDataCount = (world->primaryLightCount + 31) >> 5;

				const auto plan = plan_surfaces(asset, sky ? &sky_info : nullptr, props);
				world->surfaceCount = static_cast<unsigned int>(plan.order.size());
				apply_uv_periods(asset, zone, plan);
				apply_vertex_alpha_white(zone, plan);
				convert_surfaces(asset, world, plan, props, sun_count, allocator);
				convert_static_models(asset, world, sun_count, allocator);
				{
					// a static model no node lists is never drawn: the root takes it
					auto* root = zone->aabbTrees[0].aabbTree;
					std::vector<char> listed(world->dpvs.smodelCount, 0);
					for (auto n = 0; n < zone->aabbTreeCounts[0].aabbTreeCount; n++)
					{
						const auto& node = root[n];
						for (auto m = 0; m < node.smodelIndexCount; m++)
						{
							listed[node.smodelIndexes[m]] = 1;
						}
					}
					std::vector<unsigned short> root_list(root->smodelIndexes, root->smodelIndexes + root->smodelIndexCount);
					auto unlisted = 0u;
					for (auto m = 0u; m < world->dpvs.smodelCount; m++)
					{
						if (!listed[m])
						{
							root_list.push_back(static_cast<unsigned short>(m));
							unlisted++;
						}
					}
					if (unlisted)
					{
						root->smodelIndexCount = static_cast<unsigned short>(root_list.size());
						root->smodelIndexes = allocator.allocate_array<unsigned short>(root_list.size());
						std::memcpy(root->smodelIndexes, root_list.data(), sizeof(unsigned short) * root_list.size());
					}
					auto grown = 0u;
					enclose_static_models(root, world, grown);
					ZONETOOL_INFO("gfxworld: %u static models no AABB tree node listed added to the root; %u nodes grown to enclose their "
						"static models (IW7 culls a node's list by its bounds; BO3's nodes list models reaching past them)", unlisted, grown);
				}
				convert_brush_models(asset, world, plan, allocator);

				if (sky)
				{
					world->skyCount = 1;
					world->skies = allocator.allocate_array<zonetool::iw7::GfxSky>(1);
					auto& entry = world->skies[0];
					entry.skySurfCount = 1;
					entry.skyStartSurfs = allocator.allocate_array<int>(1);
					entry.skyStartSurfs[0] = static_cast<int>(asset->dpvs.staticSurfaceCount); // the sky box's sorted slot
					entry.skyImage = allocator.allocate<zonetool::iw7::GfxImage>();
					entry.skyImage->name = allocator.duplicate_string(sky->image);
					entry.skySamplerState = sky->sampler_state;
				}

				// shadow geometry optimisation: one level per sun light
				world->dpvs.sunShadowOptCount = sun_count;
				world->dpvs.sunSurfVisDataCount = (world->dpvs.surfaceVisDataCount + 31) & ~31u;

				// stock world bounds are the sky box's grown by one unit
				{
					float mins[3], maxs[3];
					for (auto k = 0; k < 3; k++)
					{
						mins[k] = asset->mins[k] - (sky ? 1.0f : 0.0f);
						maxs[k] = asset->maxs[k] + (sky ? 1.0f : 0.0f);
					}
					set_bounds(world->bounds, mins, maxs);
				}
				world->checksum = asset->checksum;

				world->materialMemoryCount = asset->materialMemoryCount;
				world->materialMemory = allocator.allocate_array<zonetool::iw7::MaterialMemory>(asset->materialMemoryCount);
				for (auto i = 0; i < asset->materialMemoryCount; i++)
				{
					auto* stub = allocator.allocate<zonetool::iw7::Material>();
					stub->name = allocator.duplicate_string(world_material::get(asset->materialMemory[i].material).name);
					world->materialMemory[i].material = stub;
					world->materialMemory[i].memory = asset->materialMemory[i].memory;
				}

				world->materialLod0SizeThreshold = 0.5f;

				// dynamic entities
				for (auto i = 0; i < 2; i++)
				{
					world->dpvsDyn.dynEntClientCount[i] = i == 0 ? map::reserved_dynents : 0;
					world->dpvsDyn.dynEntClientWordCount[i] = (world->dpvsDyn.dynEntClientCount[i] + 31) >> 5;
				}

				// runtime buffer sizes (stock formulas)
				const auto static_lights = world->primaryLightCount - world->lastSunPrimaryLightIndex
					- world->movingScriptablePrimaryLightCount - 1;
				world->staticSpotOmniPrimaryLightCountAligned = (static_lights + 31) & ~31u;
				world->primaryLightMotionDetectBitsEntries = world->staticSpotOmniPrimaryLightCountAligned >> 4;
				world->entityMotionBitsEntries = 134;
				world->numPrimaryLightEntityShadowVisEntries = world->staticSpotOmniPrimaryLightCountAligned * 0x86;
				for (auto i = 0; i < 2; i++)
				{
					world->dynEntMotionBitsEntries[i] = ((world->dpvsDyn.dynEntClientCount[i] + 31) >> 5) * 2;
					world->numPrimaryLightDynEntShadowVisEntries[i] =
						(world->staticSpotOmniPrimaryLightCountAligned * world->dpvsDyn.dynEntClientCount[i]) >> 4;
				}

				// every light needs a (possibly empty) region, the dumper walks all of them
				world->lightRegion = allocator.allocate_array<zonetool::iw7::GfxLightRegion>(world->primaryLightCount);

				// the static surfaces that block light: the Umbra tome's occluders
				std::vector<std::uint8_t> occluders(world->dpvs.staticSurfaceCount);
				for (auto i = 0u; i < world->dpvs.staticSurfaceCount; i++)
				{
					const auto& material = *plan.materials[plan.order[i]];
					occluders[i] = plan.order[i] != plan.sky && material.cls == world_material::surface_class::opaque && !material.alpha_test;
				}
				// the lightmap's sun visibility: blocked by what IW7's sun shadow draws (flag 1: the material casts and BO3's
				// surface does), as the baked visibility takes over from the real-time shadow past the camera's cascades;
				// alpha tested ones (bars, grates) let light through their gaps, which a solid triangle would not
				std::vector<std::uint8_t> sun_occluders(world->dpvs.staticSurfaceCount);
				for (auto i = 0u; i < world->dpvs.staticSurfaceCount; i++)
				{
					// shadow-only surfaces too (BO3's caulk_shadow, which a level places to close light leaks, is never drawn
					// but IW7's sun shadow draws it, flag 1)
					const auto& material = *plan.materials[plan.order[i]];
					const auto blocks = occluders[i] || (plan.order[i] != plan.sky && material.cls == world_material::surface_class::shadow_only);
					// a world group's surfaces block as their static models (sun_blockers places those, alpha masks included)
					sun_occluders[i] = blocks && (world->dpvs.surfaces[i].flags & 1) && plan.order[i] < plan.prop_begin;
				}
				{
					// what is left out of the sun's blockers, by material (a left-out wall or ceiling lets the sun in)
					std::map<std::string, std::uint32_t> left_out;
					for (auto i = 0u; i < world->dpvs.staticSurfaceCount; i++)
					{
						if (sun_occluders[i] || plan.order[i] == plan.sky)
						{
							continue;
						}
						const auto& material = *plan.materials[plan.order[i]];
						const auto* why = !occluders[i] ? (material.alpha_test ? "alpha tested" : "not opaque") : "casts no shadow (flag 1 clear)";
						left_out[utils::string::va("%s: %s", why, material.name.data())] += world->dpvs.surfaces[i].tris.triCount;
					}
					for (const auto& [key, triangles] : left_out)
					{
						ZONETOOL_INFO("lightmap: sun blockers leave out world %s (%u triangles)", key.data(), triangles);
					}
				}
				// what shadows the sun for both baked visibilities, the lightmap's and the light grid's (before the lightmap
				// rebuilds the vertices)
				const world_lightmap::sun_blockers sun_blockers(asset, world, zone, sun_occluders);
				// lightmap charts split vertices, so this goes before anything reads them
				{
					// each world group surface's placed parts, by IW7 surface (charted on their own)
					std::unordered_map<unsigned int, std::vector<unsigned int>> pieces;
					for (auto k = 0u; k < props.surfaces.size(); k++)
					{
						pieces.emplace(plan.remap[plan.prop_begin + k], props.surfaces[k].pieces);
					}
					world_lightmap::bake(asset, world, zone, sun_blockers, pieces, allocator);
				}
				reflection_probes::convert(asset, world, allocator);
				{
					std::vector<char> sun_only(world->surfaceCount, 0);
					for (auto i = 0u; i < world->surfaceCount; i++)
					{
						sun_only[i] = plan.order[i] >= plan.prop_begin && plan.order[i] != plan.sky;
					}
					world_lights::set_sun_only_casters(std::move(sun_only));
				}
				world_lights::build(world, allocator);
				world_lightgrid::build(asset, world, zone, occluders, sun_blockers, allocator);
				generate_umbra_tome(world, zone, plan, allocator);

				// Every stock map has one heightfield (0x140DE7F50 binds its image and lookup matrix for the sun shadow pass);
				// Spaceland's and mp_riot's are a 1x1 R16_UNORM over the map. This one sits at the bottom of the map's bounds, so it
				// shades nothing and past the cascades the sun is the baked lightmaps' and probes'.
				if (world->modelCount > 0)
				{
					const auto& b = world->models[0].bounds;
					float mid[3], half[3];
					for (auto k = 0; k < 3; k++)
					{
						mid[k] = b.midPoint[k];
						half[k] = std::max(b.halfSize[k], 1.0f) + 1024.0f;
					}
					world->heightfieldCount = 1;
					world->heightfields = allocator.allocate_array<zonetool::iw7::GfxHeightfield>(1);
					auto& field = world->heightfields[0];
					std::memcpy(field.bounds.midPoint, mid, sizeof(mid));
					std::memcpy(field.bounds.halfSize, half, sizeof(half));
					std::memset(field.lookupMatrix, 0, sizeof(field.lookupMatrix));
					field.lookupMatrix[0][0] = 0.5f / half[0];
					field.lookupMatrix[1][1] = -0.5f / half[1];
					field.lookupMatrix[2][2] = 0.5f / half[2];
					field.lookupMatrix[3][0] = 0.5f - mid[0] * 0.5f / half[0];
					field.lookupMatrix[3][1] = 0.5f + mid[1] * 0.5f / half[1];
					field.lookupMatrix[3][2] = 0.5f - mid[2] * 0.5f / half[2];
					field.lookupMatrix[3][3] = 1.0f;

					const auto name = map::map_name(asset->name) + "_heightmap0";
					static std::uint8_t pixels[4] = { 0, 0, 0, 0 };
					zonetool::iw7::GfxImage image{};
					image.imageFormat = DXGI_FORMAT_R16_UNORM;
					image.flags = 3;
					image.mapType = zonetool::iw7::MAPTYPE_2D;
					image.semantic = zonetool::iw7::TS_FUNCTION;
					image.category = zonetool::iw7::IMG_CATEGORY_AUTO_GENERATED;
					image.dataLen1 = sizeof(pixels);
					image.dataLen2 = sizeof(pixels);
					image.width = 1;
					image.height = 1;
					image.depth = 1;
					image.numElements = 1;
					image.levelCount = 1;
					image.pixelData = pixels;
					image.name = name.data();
					zonetool::iw7::gfx_image::dump(&image);
					field.image = allocator.allocate<zonetool::iw7::GfxImage>();
					field.image->name = allocator.duplicate_string(name);
				}

				ZONETOOL_INFO("gfxworld \"%s\": %u surfaces (%u static: opaque [%u, %u) decal [%u, %u) trans [%u, %u) "
					"emissive [%u, %u)), %u vertices, %u indices", world->name, world->surfaceCount,
					world->dpvs.staticSurfaceCount, plan.range_begin[0], plan.range_end[0], plan.range_begin[1],
					plan.range_end[1], plan.range_begin[2], plan.range_end[2], plan.range_begin[3], plan.range_end[3],
					zone->vertexCount, world->draw.indexCount);

				return world;
			}

			void set_umbra_smallest_hole(const float units)
			{
				umbra_smallest_hole = units;
			}

			void dump(GfxWorld* asset)
			{
				auto* world = convert(asset, persistent_allocator);
				zonetool::iw7::gfx_world::dump(world);
				zonetool::iw7::gfx_world_tr::dump(world->draw.transientZones[0]);
			}
		}
	}
}
