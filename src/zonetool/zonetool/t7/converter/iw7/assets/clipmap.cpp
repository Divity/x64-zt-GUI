#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "clipmap.hpp"
#include "xmodel_collision.hpp"
#include "xmodel.hpp"

#include "zonetool/t7/converter/iw7/convex.hpp"
#include "zonetool/t7/converter/iw7/map_common.hpp"
#include "zonetool/t7/converter/iw7/surface_parms.hpp"
#include "zonetool/t7/converter/iw7/zombies_ents.hpp"

#include "zonetool/iw7/assets/clipmap.hpp"
#include "zonetool/iw7/assets/mapents.hpp"
#include "zonetool/iw7/assets/physics_asset.hpp"
#include "zonetool/iw7/common/havok_builder.hpp"
#include "zonetool/utils/gsc.hpp"

#include <utils/string.hpp>

#include <array>
#include <unordered_map>

// Sources for the non-obvious conversions below (BO3 = BlackOps3 dedi, IW7 = iw7_ship):
// * surface parms, contents and physics materials: surface_parms.cpp.
// * brushes: CM_TraceThroughBrush (0x1404EE980) takes axial_*[0][axis] for the min face (normal -axis) and
//   [1][axis] for the max face; sides are {plane, cflags, sflags}; a brush's contents is the OR of its
//   sides'. Brush models' brushes, trisoup and cmodel bounds are entity-local.
// * trisoup: BO3 winds collision triangles clockwise seen from outside, stock IW7 world meshes
//   counter-clockwise.
// * triggers: BO3 trigger hulls are entity-local AABBs cut by slabs {dir, midPoint, halfSize} like IW7's.
//   Every BO3 trigger is contents 0x20000001, every stock IW7 one 0x28000001 (or 0x28004000). Stock CP
//   info_volumes are trigger models ("?N"); IW7's info_volume spawn (0x140400320) gives the entity a
//   physics body with contents 0, so BO3's brush-model info_volumes become trigger hulls.
// * entity physics: stock IW7 binds every shaped trigger and script_brushmodel with key 51961, a dummy
//   PhysicsAsset and physicsShapeOverrideIdx. Stock cmodel 0 keeps override 0 and navObstacleIdx 0.

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace clipmap
		{
			namespace
			{
				namespace builder = zonetool::iw7::havok::builder;

				using vec3 = std::array<float, 3>;

				constexpr auto havok_scale = 1.0f / 32.0f;

				using surface_parms::user_data_brush;
				using surface_parms::bo3_caulk;
				using surface_parms::bo3_nodraw;
				using surface_parms::surface_flags;
				using surface_parms::contents;
				using surface_parms::material_crc;

				constexpr auto trigger_physics_asset = "triggermodeldummydefault";
				constexpr auto brushmodel_physics_asset = "scriptbrushmodeldummydefault";
				constexpr auto physics_asset_key = "51961";

				constexpr std::uint32_t iw7_trigger_contents = 0x28000001;
				constexpr std::uint32_t bo3_trigger_contents = 0x20000001;
				constexpr std::uint32_t ents_trigger_contents = 0xC7FFBFFF;
				constexpr std::uint64_t ents_trigger_user_data = user_data_brush | 0x40080; // nocastshadow | nodraw

				// ---- source walks -------------------------------------------------------------------

				// marks the brushes under a cLeafBrushNode tree (a negative leafBrushCount keeps the brushes
				// straddling the split in the next node)
				void mark_leaf_brushes(const ClipInfo& info, const std::int64_t root, std::vector<bool>& out)
				{
					std::vector<bool> visited(info.leafbrushNodesCount, false);
					std::vector<std::int64_t> stack{root};
					while (!stack.empty())
					{
						const auto index = stack.back();
						stack.pop_back();
						if (index < 0 || index >= static_cast<std::int64_t>(info.leafbrushNodesCount) || visited[index])
						{
							continue;
						}
						visited[index] = true;

						const auto& node = info.leafbrushNodes[index];
						if (node.leafBrushCount > 0)
						{
							for (auto i = 0; i < node.leafBrushCount; i++)
							{
								const auto brush = node.data.leaf.brushes[i];
								if (brush < info.numBrushes)
								{
									out[brush] = true;
								}
							}
							continue;
						}
						if (node.leafBrushCount < 0)
						{
							stack.push_back(index + 1);
						}
						stack.push_back(index + node.data.children.childOffset[0]);
						stack.push_back(index + node.data.children.childOffset[1]);
					}
				}

				void mark_tree_partitions(const clipMap_t* clip, const int root, std::vector<bool>& out)
				{
					std::vector<int> stack{root};
					while (!stack.empty())
					{
						const auto index = stack.back();
						stack.pop_back();
						if (index < 0 || index >= clip->aabbTreeCount)
						{
							continue;
						}
						const auto& tree = clip->aabbTrees[index];
						if (tree.childCount == 0)
						{
							if (tree.u.partitionIndex >= 0 && tree.u.partitionIndex < clip->partitionCount)
							{
								out[tree.u.partitionIndex] = true;
							}
							continue;
						}
						for (auto c = 0; c < tree.childCount; c++)
						{
							stack.push_back(tree.u.firstChildIndex + c);
						}
					}
				}

				// ---- brushes ------------------------------------------------------------------------

				struct solid
				{
					convex::hull hull;
					std::vector<std::uint32_t> face_flags; // BO3 surface flags of each hull face
				};

				bool brush_solid(const cbrush_t& brush, solid& out)
				{
					std::vector<convex::plane> planes;
					std::vector<std::uint32_t> flags;
					const auto add = [&](const float nx, const float ny, const float nz, const float dist, const std::uint32_t sflags)
					{
						for (const auto& p : planes)
						{
							if (std::fabs(p[0] - nx) < 1e-4f && std::fabs(p[1] - ny) < 1e-4f && std::fabs(p[2] - nz) < 1e-4f
								&& std::fabs(p[3] - dist) < 0.01f)
							{
								return;
							}
						}
						planes.push_back({nx, ny, nz, dist});
						flags.push_back(sflags);
					};

					for (auto side = 0; side < 2; side++)
					{
						for (auto axis = 0; axis < 3; axis++)
						{
							float n[3] = {0.0f, 0.0f, 0.0f};
							n[axis] = side ? 1.0f : -1.0f;
							add(n[0], n[1], n[2], side ? brush.maxs[axis] : -brush.mins[axis], brush.axial_sflags[side][axis]);
						}
					}
					for (auto s = 0u; s < brush.numsides; s++)
					{
						const auto* plane = brush.sides[s].plane;
						if (plane)
						{
							add(plane->normal[0], plane->normal[1], plane->normal[2], plane->dist, brush.sides[s].sflags);
						}
					}

					if (!convex::from_planes(planes, out.hull))
					{
						return false;
					}

					out.face_flags.resize(out.hull.faces.size());
					for (auto f = 0u; f < out.hull.faces.size(); f++)
					{
						const auto it = std::find(planes.begin(), planes.end(), out.hull.faces[f].p);
						out.face_flags[f] = flags[std::distance(planes.begin(), it)];
					}
					return true;
				}

				double face_area(const convex::hull& hull, const convex::face& face)
				{
					const auto& o = hull.vertices[face.indices[0]];
					double c[3] = {0.0, 0.0, 0.0};
					for (auto k = 1u; k + 1 < face.indices.size(); k++)
					{
						const auto& a = hull.vertices[face.indices[k]];
						const auto& b = hull.vertices[face.indices[k + 1]];
						const double e1[3] = {a[0] - o[0], a[1] - o[1], a[2] - o[2]};
						const double e2[3] = {b[0] - o[0], b[1] - o[1], b[2] - o[2]};
						c[0] += e1[1] * e2[2] - e1[2] * e2[1];
						c[1] += e1[2] * e2[0] - e1[0] * e2[2];
						c[2] += e1[0] * e2[1] - e1[1] * e2[0];
					}
					return 0.5 * std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
				}

				// A shape carries one surface: the one covering the most visible area (caulk and nodraw
				// faces only count when a solid has nothing else).
				struct surface_vote
				{
					std::map<std::uint32_t, double> visible;
					std::map<std::uint32_t, double> hidden;

					void add(const std::uint32_t bo3_flags, const double area)
					{
						((bo3_flags & (bo3_caulk | bo3_nodraw)) ? hidden : visible)[bo3_flags] += area;
					}

					void add(const solid& s)
					{
						for (auto f = 0u; f < s.hull.faces.size(); f++)
						{
							this->add(s.face_flags[f], face_area(s.hull, s.hull.faces[f]));
						}
					}

					std::uint32_t winner() const
					{
						const auto& pool = visible.empty() ? hidden : visible;
						std::uint32_t best = 0;
						auto best_area = -1.0;
						for (const auto& [flags, area] : pool)
						{
							if (area > best_area)
							{
								best = flags;
								best_area = area;
							}
						}
						return best;
					}
				};

				builder::polytope to_polytope(const convex::hull& hull)
				{
					builder::polytope out{};
					out.verts = hull.vertices;
					for (const auto& face : hull.faces)
					{
						builder::polytope_face f{};
						std::memcpy(f.plane, face.p.data(), sizeof(f.plane));
						for (const auto index : face.indices)
						{
							f.indices.push_back(static_cast<std::uint8_t>(index));
						}
						out.faces.push_back(std::move(f));
					}
					return out;
				}

				// a zero-thickness triangle as a two-faced polytope, the form of Havok's own triangle shape
				builder::polytope triangle_polytope(const vec3& a, const vec3& b, const vec3& c)
				{
					builder::polytope out{};
					out.verts = {a, b, c};
					const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
					const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
					double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
					const auto length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
					for (auto& v : n)
					{
						v /= length;
					}
					const auto d = n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
					out.faces.push_back({{static_cast<float>(n[0]), static_cast<float>(n[1]), static_cast<float>(n[2]),
						static_cast<float>(d)}, {0, 1, 2}});
					out.faces.push_back({{static_cast<float>(-n[0]), static_cast<float>(-n[1]), static_cast<float>(-n[2]),
						static_cast<float>(-d)}, {0, 2, 1}});
					return out;
				}

				// ---- spatial order ------------------------------------------------------------------

				std::uint64_t spread_bits(std::uint64_t v)
				{
					v &= 0x1FFFFF;
					v = (v | v << 32) & 0x1F00000000FFFFull;
					v = (v | v << 16) & 0x1F0000FF0000FFull;
					v = (v | v << 8) & 0x100F00F00F00F00Full;
					v = (v | v << 4) & 0x10C30C30C30C30C3ull;
					v = (v | v << 2) & 0x1249249249249249ull;
					return v;
				}

				// Havok sections take primitives in input order, so neighbours have to be neighbours in the
				// input for compact sections: Morton order of their centres.
				template <typename T, typename Centre>
				void sort_spatially(std::vector<T>& items, const Centre& centre)
				{
					if (items.size() < 2)
					{
						return;
					}

					std::vector<vec3> centres(items.size());
					vec3 lo{FLT_MAX, FLT_MAX, FLT_MAX};
					vec3 hi{-FLT_MAX, -FLT_MAX, -FLT_MAX};
					for (auto i = 0u; i < items.size(); i++)
					{
						centres[i] = centre(items[i]);
						for (auto k = 0; k < 3; k++)
						{
							lo[k] = std::min(lo[k], centres[i][k]);
							hi[k] = std::max(hi[k], centres[i][k]);
						}
					}

					std::vector<std::pair<std::uint64_t, std::uint32_t>> keys(items.size());
					for (auto i = 0u; i < items.size(); i++)
					{
						std::uint64_t code = 0;
						for (auto k = 0; k < 3; k++)
						{
							const auto extent = hi[k] - lo[k];
							const auto q = extent > 0.0f ? static_cast<std::uint64_t>((centres[i][k] - lo[k]) / extent * 2097151.0f) : 0;
							code |= spread_bits(q) << k;
						}
						keys[i] = {code, i};
					}
					std::sort(keys.begin(), keys.end());

					std::vector<T> sorted;
					sorted.reserve(items.size());
					for (const auto& key : keys)
					{
						sorted.push_back(std::move(items[key.second]));
					}
					items = std::move(sorted);
				}

				// ---- trigger hulls ------------------------------------------------------------------

				bool trigger_hull_solid(const zonetool::iw7::TriggerHull& hull, const zonetool::iw7::TriggerSlab* slabs,
					convex::hull& out)
				{
					std::vector<convex::plane> planes;
					for (auto axis = 0; axis < 3; axis++)
					{
						convex::plane lo{0.0f, 0.0f, 0.0f, -(hull.bounds.midPoint[axis] - hull.bounds.halfSize[axis])};
						convex::plane hi{0.0f, 0.0f, 0.0f, hull.bounds.midPoint[axis] + hull.bounds.halfSize[axis]};
						lo[axis] = -1.0f;
						hi[axis] = 1.0f;
						planes.push_back(lo);
						planes.push_back(hi);
					}
					for (auto s = 0; s < hull.slabCount; s++)
					{
						const auto& slab = slabs[hull.firstSlab + s];
						planes.push_back({slab.dir[0], slab.dir[1], slab.dir[2], slab.midPoint + slab.halfSize});
						planes.push_back({-slab.dir[0], -slab.dir[1], -slab.dir[2], -(slab.midPoint - slab.halfSize)});
					}
					return convex::from_planes(planes, out);
				}

				// A brush as a trigger hull: its AABB, cut by a slab per non-axial side (a side and its
				// opposite share one slab; a lone side's slab opens onto the box, as BO3's compiled ones do).
				void brush_trigger_hull(const cbrush_t& brush, std::vector<zonetool::iw7::TriggerHull>& hulls,
					std::vector<zonetool::iw7::TriggerSlab>& slabs)
				{
					zonetool::iw7::TriggerHull hull{};
					for (auto k = 0; k < 3; k++)
					{
						hull.bounds.midPoint[k] = (brush.mins[k] + brush.maxs[k]) * 0.5f;
						hull.bounds.halfSize[k] = (brush.maxs[k] - brush.mins[k]) * 0.5f;
					}
					hull.contents = iw7_trigger_contents;
					hull.firstSlab = static_cast<unsigned short>(slabs.size());

					std::vector<bool> used(brush.numsides, false);
					for (auto s = 0u; s < brush.numsides; s++)
					{
						const auto* plane = brush.sides[s].plane;
						if (used[s] || !plane)
						{
							continue;
						}
						const auto* n = plane->normal;
						if (std::fabs(n[0]) > 0.9999f || std::fabs(n[1]) > 0.9999f || std::fabs(n[2]) > 0.9999f)
						{
							continue; // axial: the AABB already is this side
						}
						used[s] = true;

						auto low = 0.0f;
						auto matched = false;
						for (auto o = s + 1; o < brush.numsides; o++)
						{
							const auto* other = brush.sides[o].plane;
							if (!used[o] && other && std::fabs(other->normal[0] + n[0]) < 1e-4f
								&& std::fabs(other->normal[1] + n[1]) < 1e-4f && std::fabs(other->normal[2] + n[2]) < 1e-4f)
							{
								used[o] = true;
								low = -other->dist;
								matched = true;
								break;
							}
						}
						if (!matched)
						{
							low = FLT_MAX;
							for (auto corner = 0; corner < 8; corner++)
							{
								const float p[3] = {(corner & 1) ? brush.maxs[0] : brush.mins[0], (corner & 2) ? brush.maxs[1] : brush.mins[1],
									(corner & 4) ? brush.maxs[2] : brush.mins[2]};
								low = std::min(low, p[0] * n[0] + p[1] * n[1] + p[2] * n[2]);
							}
						}

						zonetool::iw7::TriggerSlab slab{};
						std::memcpy(slab.dir, n, sizeof(slab.dir));
						slab.midPoint = (plane->dist + low) * 0.5f;
						slab.halfSize = (plane->dist - low) * 0.5f;
						slabs.push_back(slab);
					}
					hull.slabCount = static_cast<unsigned short>(slabs.size() - hull.firstSlab);
					hulls.push_back(hull);
				}

				// ---- entity string ------------------------------------------------------------------

				struct entity_edit
				{
					std::string model;         // replaces the "model" value when set
					std::string physics_asset; // adds key 51961 when set
				};

				// BO3 writes one "key" "value" pair per line; entities are rewritten line by line so every
				// untouched byte stays as it was.
				std::string edit_entities(const std::string& source,
					const std::function<entity_edit(const std::string& classname, const std::string& model)>& edit)
				{
					std::string out;
					out.reserve(source.size() + 0x10000);

					std::size_t cursor = 0;
					while (cursor < source.size())
					{
						const auto open = source.find('{', cursor);
						if (open == std::string::npos)
						{
							out.append(source, cursor, std::string::npos);
							break;
						}
						const auto close = source.find('}', open);
						if (close == std::string::npos)
						{
							out.append(source, cursor, std::string::npos);
							break;
						}
						out.append(source, cursor, open - cursor);

						std::vector<std::string> lines;
						std::string classname, model;
						std::size_t model_line = std::string::npos;
						const auto body = source.substr(open + 1, close - open - 1);
						std::size_t start = 0;
						while (start <= body.size())
						{
							auto end = body.find('\n', start);
							if (end == std::string::npos)
							{
								end = body.size();
							}
							const auto line = body.substr(start, end - start);
							lines.push_back(line);

							const auto key_open = line.find('"');
							const auto key_close = key_open == std::string::npos ? key_open : line.find('"', key_open + 1);
							const auto value_open = key_close == std::string::npos ? key_close : line.find('"', key_close + 1);
							const auto value_close = value_open == std::string::npos ? value_open : line.find('"', value_open + 1);
							if (value_close != std::string::npos)
							{
								const auto key = line.substr(key_open + 1, key_close - key_open - 1);
								const auto value = line.substr(value_open + 1, value_close - value_open - 1);
								if (key == "classname")
								{
									classname = value;
								}
								else if (key == "model")
								{
									model = value;
									model_line = lines.size() - 1;
								}
							}
							start = end + 1;
						}

						const auto change = edit(classname, model);
						if (!change.model.empty() && model_line != std::string::npos)
						{
							lines[model_line] = "\"model\" \"" + change.model + "\"";
						}

						out.push_back('{');
						for (auto i = 0u; i < lines.size(); i++)
						{
							// the body ends with the newline before '}'; the pair goes in front of it
							if (i + 1 == lines.size() && !change.physics_asset.empty())
							{
								out.append(physics_asset_key);
								out.append(" \"" + change.physics_asset + "\"\n");
							}
							out.append(lines[i]);
							if (i + 1 < lines.size())
							{
								out.push_back('\n');
							}
						}
						out.push_back('}');
						cursor = close + 1;
					}
					return out;
				}

				zonetool::iw7::PhysicsAsset* dummy_physics_asset(const std::string& name, const std::string& body_name,
					utils::memory::allocator& allocator)
				{
					builder::physics_asset_input input{};
					input.body_name = body_name;
					const auto blob = builder::build_physics_asset(input);
					if (blob.empty())
					{
						ZONETOOL_FATAL("clipmap: could not build the \"%s\" physics asset", name.data());
					}

					auto* asset = allocator.allocate<zonetool::iw7::PhysicsAsset>();
					asset->name = allocator.duplicate_string(name);
					asset->havokData = allocator.allocate_array<char>(blob.size());
					std::memcpy(asset->havokData, blob.data(), blob.size());
					asset->havokDataSize = static_cast<unsigned int>(blob.size());
					asset->numRigidBodies = 1;
					asset->numConstraints = 0;
					asset->numSFXEventAssets = 1;
					asset->sfxEventAssets = allocator.allocate_array<zonetool::iw7::PhysicsSFXEventAsset*>(1);
					asset->numVFXEventAssets = 1;
					asset->vfxEventAssets = allocator.allocate_array<zonetool::iw7::PhysicsVFXEventAsset*>(1);
					return asset;
				}

				char* copy_blob(const std::vector<std::uint8_t>& blob, utils::memory::allocator& allocator)
				{
					auto* memory = allocator.allocate_array<char>(blob.size());
					std::memcpy(memory, blob.data(), blob.size());
					return memory;
				}
			}

			void dump(clipMap_t* asset)
			{
				utils::memory::allocator allocator;
				const auto& info = asset->info;
				const auto* source_ents = asset->mapEnts;
				if (!source_ents)
				{
					ZONETOOL_FATAL("clipmap \"%s\" has no MapEnts", asset->name);
				}

				const auto name = map::bsp_name(asset->name);
				surface_parms::stats stats{};

				// ---- who owns what ----------------------------------------------------------------

				std::vector<bool> model_brushes(info.numBrushes, false);
				std::vector<bool> model_partitions(std::max(asset->partitionCount, 0), false);
				std::vector<std::vector<int>> brushes_of(asset->numSubModels);
				std::vector<std::vector<int>> partitions_of(asset->numSubModels);
				for (auto m = 1u; m < asset->numSubModels; m++)
				{
					const auto& leaf = asset->cmodels[m].leaf;
					std::vector<bool> brushes(info.numBrushes, false);
					mark_leaf_brushes(info, leaf.leafBrushNode, brushes);
					std::vector<bool> parts(model_partitions.size(), false);
					for (auto k = 0u; k < leaf.collAabbCount; k++)
					{
						mark_tree_partitions(asset, static_cast<int>(leaf.firstCollAabbIndex + k), parts);
					}
					for (auto b = 0; b < info.numBrushes; b++)
					{
						if (brushes[b])
						{
							if (model_brushes[b])
							{
								ZONETOOL_FATAL("clipmap: brush %d belongs to two brush models", b);
							}
							model_brushes[b] = true;
							brushes_of[m].push_back(b);
						}
					}
					for (auto p = 0u; p < parts.size(); p++)
					{
						if (parts[p])
						{
							model_partitions[p] = true;
							partitions_of[m].push_back(static_cast<int>(p));
						}
					}
				}

				std::vector<int> partition_material(model_partitions.size(), -1);
				for (auto i = 0; i < asset->aabbTreeCount; i++)
				{
					const auto& tree = asset->aabbTrees[i];
					if (tree.childCount == 0 && tree.u.partitionIndex >= 0 && tree.u.partitionIndex < asset->partitionCount)
					{
						partition_material[tree.u.partitionIndex] = tree.materialIndex;
					}
				}

				const auto triangle = [&](const int tri, vec3 (&out)[3])
				{
					// clockwise -> counter-clockwise
					const auto* indices = &asset->triIndices[tri * 3];
					const unsigned int order[3] = {indices[0], indices[2], indices[1]};
					for (auto c = 0; c < 3; c++)
					{
						std::memcpy(out[c].data(), asset->verts[order[c]], sizeof(float[3]));
					}
				};

				// ---- static model physics budget --------------------------------------------------

				// IW7 builds one Havok static compound shape from every placed static model's PhysicsAsset bodies and
				// one from their physics LOD shapes (StaticModels_CreateClipmapShapes 0x140574CF0, the compound
				// 0x14108CA90). A compound numbers its instances in 16 bits: int16 instance ids (0x14108C3A0) and u16
				// tree nodes, two per instance (0x140FB9C10), so one past 32767 instances overflows. Past the budget,
				// the instances with the fewest collision triangles leave the static model list and go into the world
				// shape as trisoup, placed in the world: every static model keeps its collision.
				constexpr auto static_model_physics_budget = 32000ull;
				std::unordered_map<const XModel*, xmodel_collision::physics_cost> model_cost;
				std::unordered_map<const XModel*, std::vector<xmodel_collision::triangle>> model_tris;
				std::vector<bool> baked_static_model(asset->numStaticModels, false);
				auto baked_count = 0u;
				{
					unsigned long long bodies = 0, lod_shapes = 0;
					for (auto i = 0u; i < asset->numStaticModels; i++)
					{
						auto* model = asset->staticModelList[i].xmodel;
						if (!model_cost.contains(model))
						{
							model_cost[model] = xmodel_collision::cost(model);
							model_tris[model] = xmodel_collision::triangles(model);
						}
						bodies += model_cost[model].bodies;
						lod_shapes += model_cost[model].lod_shapes;
					}
					if (bodies > static_model_physics_budget || lod_shapes > static_model_physics_budget)
					{
						std::vector<unsigned int> order(asset->numStaticModels);
						for (auto i = 0u; i < asset->numStaticModels; i++)
						{
							order[i] = i;
						}
						std::stable_sort(order.begin(), order.end(), [&](const unsigned int a, const unsigned int b)
						{
							return model_tris[asset->staticModelList[a].xmodel].size() > model_tris[asset->staticModelList[b].xmodel].size();
						});
						bodies = lod_shapes = 0;
						for (const auto i : order)
						{
							const auto& c = model_cost[asset->staticModelList[i].xmodel];
							if (bodies + c.bodies <= static_model_physics_budget && lod_shapes + c.lod_shapes <= static_model_physics_budget)
							{
								bodies += c.bodies;
								lod_shapes += c.lod_shapes;
								continue;
							}
							baked_static_model[i] = true;
							baked_count++;
						}
					}
				}

				// ---- world shape ------------------------------------------------------------------

				builder::mesh_input world{};
				auto skipped_empty_brushes = 0u, failed_brushes = 0u, face_fallback = 0u, skipped_empty_tris = 0u;
				auto unreferenced_partitions = 0u, unreferenced_tris = 0u;
				std::map<std::uint32_t, unsigned int> contents_hist;

				for (auto b = 0; b < info.numBrushes; b++)
				{
					if (model_brushes[b])
					{
						continue;
					}
					const auto& brush = info.brushes[b];
					const auto brush_contents = contents(brush.contents, stats);
					if (!brush_contents)
					{
						skipped_empty_brushes++;
						continue;
					}

					solid s{};
					if (!brush_solid(brush, s))
					{
						failed_brushes++;
						continue;
					}

					surface_vote vote{};
					vote.add(s);
					const auto flags = surface_flags(vote.winner(), stats);
					contents_hist[brush_contents]++;

					builder::convex cvx{};
					for (const auto& v : s.hull.vertices)
					{
						cvx.verts.push_back({v[0] * havok_scale, v[1] * havok_scale, v[2] * havok_scale});
					}
					cvx.surface_tag = 0;
					cvx.contents = static_cast<int>(brush_contents);
					cvx.material_crc = material_crc(flags);
					cvx.user_data = user_data_brush | flags;

					if (!builder::convex_rejection(cvx.verts))
					{
						world.convexes.push_back(std::move(cvx));
						continue;
					}

					// not a usable convex primitive: its faces go in as triangles
					face_fallback++;
					for (auto f = 0u; f < s.hull.faces.size(); f++)
					{
						const auto& face = s.hull.faces[f];
						const auto face_flags = surface_flags(s.face_flags[f], stats);
						for (auto k = 1u; k + 1 < face.indices.size(); k++)
						{
							builder::triangle tri{};
							const unsigned int corner[3] = {face.indices[0], face.indices[k], face.indices[k + 1]};
							for (auto c = 0; c < 3; c++)
							{
								for (auto a = 0; a < 3; a++)
								{
									tri.verts[c][a] = s.hull.vertices[corner[c]][a] * havok_scale;
								}
							}
							tri.surface_tag = 0;
							tri.contents = static_cast<int>(brush_contents);
							tri.material_crc = material_crc(face_flags);
							tri.user_data = user_data_brush | face_flags;
							world.triangles.push_back(tri);
						}
					}
				}

				auto world_tris = 0u;
				for (auto p = 0; p < asset->partitionCount; p++)
				{
					if (model_partitions[p])
					{
						continue;
					}
					const auto material = partition_material[p];
					if (material < 0 || static_cast<unsigned int>(material) >= info.numMaterials)
					{
						// no leaf of BO3's collision trees names it (a leaf gives its partition the material), so BO3's traces,
						// which reach partitions only through those leaves, never test it
						unreferenced_partitions++;
						unreferenced_tris += asset->partitions[p].triCount;
						continue;
					}
					const auto& clip_material = info.materials[material];
					const auto tri_contents = contents(clip_material.contentFlags, stats);
					const auto& partition = asset->partitions[p];
					if (!tri_contents)
					{
						skipped_empty_tris += partition.triCount;
						continue;
					}
					const auto flags = surface_flags(clip_material.surfaceFlags, stats);

					for (auto t = 0; t < partition.triCount; t++)
					{
						vec3 corners[3];
						triangle(partition.firstTri + t, corners);

						builder::triangle tri{};
						for (auto c = 0; c < 3; c++)
						{
							for (auto a = 0; a < 3; a++)
							{
								tri.verts[c][a] = corners[c][a] * havok_scale;
							}
						}
						tri.surface_tag = 0;
						tri.contents = static_cast<int>(tri_contents);
						tri.material_crc = material_crc(flags);
						tri.user_data = flags;
						world.triangles.push_back(tri);
						world_tris++;
					}
				}

				// the static models past the budget: the IW engines take a world point into model space as the row
				// vector (p - origin) times M, M's rows the invScaledAxis (MatrixTransformVector), so a model point q
				// lands at origin + q inverse(M); checked against every baked instance's absmin / absmax
				auto baked_tris = 0u, baked_posed = 0u, baked_outside = 0u;
				for (auto i = 0u; i < asset->numStaticModels; i++)
				{
					if (!baked_static_model[i])
					{
						continue;
					}
					const auto& sm = asset->staticModelList[i];
					double m[3][3];
					for (auto r = 0; r < 3; r++)
					{
						for (auto c = 0; c < 3; c++)
						{
							m[r][c] = sm.invScaledAxis[r][c];
						}
					}
					const auto det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
						+ m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
					if (std::fabs(det) < 1e-12)
					{
						ZONETOOL_FATAL("clipmap: static model %u (%s) has a singular invScaledAxis", i, sm.xmodel->name);
					}
					double inv[3][3];
					inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
					inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
					inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
					inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
					inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
					inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
					inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
					inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
					inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
					const auto mirrored = det < 0.0; // a mirrored placement turns the winding around
					if (sm.numBoneMtxs > 0)
					{
						baked_posed++;
					}
					auto outside = false;
					for (const auto& t : model_tris[sm.xmodel])
					{
						builder::triangle tri{};
						for (auto c = 0; c < 3; c++)
						{
							const auto& p = t.verts[mirrored && c ? 3 - c : c];
							for (auto k = 0; k < 3; k++)
							{
								const auto w = sm.origin[k] + p[0] * inv[0][k] + p[1] * inv[1][k] + p[2] * inv[2][k];
								outside |= w < sm.absmin[k] - 1.0 || w > sm.absmax[k] + 1.0;
								tri.verts[c][k] = static_cast<float>(w) * havok_scale;
							}
						}
						tri.surface_tag = 0;
						tri.contents = static_cast<int>(t.contents);
						tri.material_crc = material_crc(t.flags);
						tri.user_data = t.flags;
						world.triangles.push_back(tri);
						baked_tris++;
					}
					baked_outside += outside ? 1 : 0;
				}
				if (baked_count)
				{
					ZONETOOL_INFO("clipmap: %u of %u static models are past IW7's static model physics budget (%llu) and went into "
						"the world shape as %u trisoup triangles (%u with posed bones placed at the base pose)", baked_count,
						asset->numStaticModels, static_model_physics_budget, baked_tris, baked_posed);
					if (baked_outside * 100 > baked_count)
					{
						ZONETOOL_FATAL("clipmap: %u of the %u baked static models landed outside their absmin / absmax: the "
							"placement convention is wrong", baked_outside, baked_count);
					}
					if (baked_outside)
					{
						ZONETOOL_WARNING("clipmap: %u baked static models have collision more than 1 unit outside their "
							"absmin / absmax", baked_outside);
					}
				}

				sort_spatially(world.triangles, [](const builder::triangle& t)
				{
					return vec3{(t.verts[0][0] + t.verts[1][0] + t.verts[2][0]) / 3.0f, (t.verts[0][1] + t.verts[1][1] + t.verts[2][1]) / 3.0f,
						(t.verts[0][2] + t.verts[1][2] + t.verts[2][2]) / 3.0f};
				});
				sort_spatially(world.convexes, [](const builder::convex& c)
				{
					vec3 lo{FLT_MAX, FLT_MAX, FLT_MAX}, hi{-FLT_MAX, -FLT_MAX, -FLT_MAX};
					for (const auto& v : c.verts)
					{
						for (auto k = 0; k < 3; k++)
						{
							lo[k] = std::min(lo[k], v[k]);
							hi[k] = std::max(hi[k], v[k]);
						}
					}
					return vec3{(lo[0] + hi[0]) * 0.5f, (lo[1] + hi[1]) * 0.5f, (lo[2] + hi[2]) * 0.5f};
				});

				// ZT_COLLISION_EXPORT=<file>: what IW7 will collide with, in world units, for building the navmesh from it:
				// the world's triangles and brushes and every placed static model's collision
				// triangles, each with its IW7 contents; brush models (doors, debris) are left out, as script blocks them.
				// "T7CS" u32 1 | u32 triangles, each 9 f32 + u32 contents | u32 convexes, each u32 contents, u32 n, n x 3 f32
				if (const char* export_path = std::getenv("ZT_COLLISION_EXPORT"))
				{
					if (auto* f = std::fopen(export_path, "wb"))
					{
						const auto u32 = [&](const std::uint32_t v) { std::fwrite(&v, 4, 1, f); };
						const auto f32 = [&](const float v) { std::fwrite(&v, 4, 1, f); };
						std::vector<std::pair<std::array<float, 9>, std::uint32_t>> tris;
						for (const auto& t : world.triangles)
						{
							std::array<float, 9> v{};
							for (auto c = 0; c < 3; c++)
							{
								for (auto k = 0; k < 3; k++)
								{
									v[c * 3 + k] = t.verts[c][k] / havok_scale;
								}
							}
							tris.emplace_back(v, static_cast<std::uint32_t>(t.contents));
						}
						// the static models still in the static model list (baked ones are in world.triangles), placed as
						// the baked ones are
						for (auto i = 0u; i < asset->numStaticModels; i++)
						{
							if (baked_static_model[i])
							{
								continue;
							}
							const auto& sm = asset->staticModelList[i];
							double m[3][3];
							for (auto r = 0; r < 3; r++)
							{
								for (auto c = 0; c < 3; c++)
								{
									m[r][c] = sm.invScaledAxis[r][c];
								}
							}
							const auto det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
								+ m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
							if (std::fabs(det) < 1e-12)
							{
								continue;
							}
							double inv[3][3];
							inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
							inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
							inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
							inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
							inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
							inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
							inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
							inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
							inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
							const auto mirrored = det < 0.0; // as the baked ones: a mirrored placement turns the winding around
							for (const auto& t : model_tris[sm.xmodel])
							{
								std::array<float, 9> v{};
								for (auto c = 0; c < 3; c++)
								{
									const auto& p = t.verts[mirrored && c ? 3 - c : c];
									for (auto k = 0; k < 3; k++)
									{
										v[c * 3 + k] = static_cast<float>(sm.origin[k] + p[0] * inv[0][k] + p[1] * inv[1][k] + p[2] * inv[2][k]);
									}
								}
								tris.emplace_back(v, static_cast<std::uint32_t>(t.contents));
							}
						}
						std::fwrite("T7CS", 1, 4, f);
						u32(1);
						u32(static_cast<std::uint32_t>(tris.size()));
						for (const auto& [v, c] : tris)
						{
							for (const auto x : v)
							{
								f32(x);
							}
							u32(c);
						}
						u32(static_cast<std::uint32_t>(world.convexes.size()));
						for (const auto& c : world.convexes)
						{
							u32(static_cast<std::uint32_t>(c.contents));
							u32(static_cast<std::uint32_t>(c.verts.size()));
							for (const auto& v : c.verts)
							{
								for (auto k = 0; k < 3; k++)
								{
									f32(v[k] / havok_scale);
								}
							}
						}
						std::fclose(f);
						ZONETOOL_INFO("clipmap: collision export %s: %zu triangles, %zu convexes", export_path, tris.size(), world.convexes.size());
					}
				}

				ZONETOOL_INFO("clipmap: world shape from %zu brushes as convex primitives (%u as faces), %u trisoup triangles; "
					"%u non-colliding brushes and %u non-colliding triangles left out", world.convexes.size(), face_fallback,
					world_tris, skipped_empty_brushes, skipped_empty_tris);
				if (unreferenced_partitions)
				{
					ZONETOOL_INFO("clipmap: %u partitions (%u triangles) no collision tree leaf names left out (BO3 never traces them)",
						unreferenced_partitions, unreferenced_tris);
				}
				if (failed_brushes)
				{
					ZONETOOL_WARNING("clipmap: %u brushes have no closed hull and are not in the world shape", failed_brushes);
				}
				if (face_fallback)
				{
					ZONETOOL_WARNING("clipmap: %u brushes are not usable convex primitives and went in as their faces; the "
						"player movement cast does not collide with loose faces", face_fallback);
				}

				std::vector<builder::shape_tag> world_tags;
				const auto world_blob = builder::build_world_shape(world, &world_tags);
				if (world_blob.empty())
				{
					ZONETOOL_FATAL("clipmap: the world shape could not be built");
				}
				decltype(world.triangles){}.swap(world.triangles);
				decltype(world.convexes){}.swap(world.convexes);

				// ---- triggers ---------------------------------------------------------------------

				const auto& source_triggers = source_ents->trigger;
				std::vector<zonetool::iw7::TriggerModel> trigger_models;
				std::vector<zonetool::iw7::TriggerHull> trigger_hulls;
				std::vector<zonetool::iw7::TriggerSlab> trigger_slabs;
				for (auto t = 0u; t < source_triggers.count; t++)
				{
					const auto& src = source_triggers.models[t];
					if (src.contents != bo3_trigger_contents)
					{
						ZONETOOL_FATAL("clipmap: trigger %u has contents 0x%08X, not the 0x%08X every BO3 trigger measured has",
							t, src.contents, bo3_trigger_contents);
					}
					zonetool::iw7::TriggerModel model{};
					model.contents = iw7_trigger_contents;
					model.hullCount = src.hullCount;
					model.firstHull = src.firstHull;
					model.physicsShapeOverrideIdx = 0xFFFF;
					trigger_models.push_back(model);
				}
				for (auto h = 0u; h < source_triggers.hullCount; h++)
				{
					const auto& src = source_triggers.hulls[h];
					if (src.contents != bo3_trigger_contents)
					{
						ZONETOOL_FATAL("clipmap: trigger hull %u has contents 0x%08X", h, src.contents);
					}
					zonetool::iw7::TriggerHull hull{};
					std::memcpy(&hull.bounds, &src.bounds, sizeof(hull.bounds));
					hull.contents = iw7_trigger_contents;
					hull.slabCount = src.slabCount;
					hull.firstSlab = src.firstSlab;
					trigger_hulls.push_back(hull);
				}
				for (auto s = 0u; s < source_triggers.slabCount; s++)
				{
					zonetool::iw7::TriggerSlab slab{};
					std::memcpy(&slab, &source_triggers.slabs[s], sizeof(slab));
					trigger_slabs.push_back(slab);
				}

				// ---- entities: which brush models are what ----------------------------------------

				std::string source_string(source_ents->entityString, source_ents->numEntityChars);
				while (!source_string.empty() && source_string.back() == '\0')
				{
					source_string.pop_back();
				}
				std::vector<std::string> model_classname(asset->numSubModels);
				edit_entities(source_string, [&](const std::string& classname, const std::string& model) -> entity_edit
				{
					if (model.size() > 1 && model[0] == '*')
					{
						const auto index = std::strtoul(model.data() + 1, nullptr, 10);
						if (index < model_classname.size())
						{
							model_classname[index] = classname;
						}
					}
					return {};
				});

				// info_volume brush models -> trigger models
				std::vector<int> volume_trigger(asset->numSubModels, -1);
				for (auto m = 1u; m < asset->numSubModels; m++)
				{
					if (model_classname[m] != "info_volume")
					{
						continue;
					}
					if (!partitions_of[m].empty())
					{
						ZONETOOL_FATAL("clipmap: info_volume brush model %u has trisoup, which a trigger hull cannot hold", m);
					}
					zonetool::iw7::TriggerModel model{};
					model.contents = iw7_trigger_contents;
					model.firstHull = static_cast<unsigned short>(trigger_hulls.size());
					for (const auto b : brushes_of[m])
					{
						brush_trigger_hull(info.brushes[b], trigger_hulls, trigger_slabs);
					}
					model.hullCount = static_cast<unsigned short>(trigger_hulls.size() - model.firstHull);
					model.physicsShapeOverrideIdx = 0xFFFF;
					volume_trigger[m] = static_cast<int>(trigger_models.size());
					trigger_models.push_back(model);
				}

				// ---- entity shapes ----------------------------------------------------------------

				builder::ents_input ents{};
				ents.world_tags = world_tags;
				std::vector<unsigned short> cmodel_shape(asset->numSubModels, 0xFFFF);
				std::vector<unsigned short> trigger_shape(trigger_models.size(), 0xFFFF);
				auto flat_triangles = 0u;

				for (auto m = 1u; m < asset->numSubModels; m++)
				{
					if (model_classname[m] != "script_brushmodel")
					{
						continue;
					}

					builder::ents_shape shape{};
					std::uint32_t shape_contents = 0;
					surface_vote vote{};
					for (const auto b : brushes_of[m])
					{
						const auto& brush = info.brushes[b];
						const auto brush_contents = contents(brush.contents, stats);
						if (!brush_contents)
						{
							continue;
						}
						solid s{};
						if (!brush_solid(brush, s))
						{
							ZONETOOL_FATAL("clipmap: brush %d of brush model %u has no closed hull", b, m);
						}
						vote.add(s);
						shape_contents |= brush_contents;
						shape.convexes.push_back(to_polytope(s.hull));
					}
					for (const auto p : partitions_of[m])
					{
						const auto& clip_material = info.materials[partition_material[p]];
						const auto tri_contents = contents(clip_material.contentFlags, stats);
						if (!tri_contents)
						{
							continue;
						}
						shape_contents |= tri_contents;
						const auto& partition = asset->partitions[p];
						for (auto t = 0; t < partition.triCount; t++)
						{
							vec3 corners[3];
							triangle(partition.firstTri + t, corners);
							convex::hull flat{};
							flat.vertices = {corners[0], corners[1], corners[2]};
							vote.add(clip_material.surfaceFlags, face_area(flat, {{}, {0, 1, 2}}));
							shape.convexes.push_back(triangle_polytope(corners[0], corners[1], corners[2]));
							flat_triangles++;
						}
					}
					if (shape.convexes.empty())
					{
						continue; // nothing solid: stock leaves such brush models without a shape too
					}

					const auto flags = surface_flags(vote.winner(), stats);
					shape.contents = static_cast<int>(shape_contents);
					shape.entity_contents = shape_contents;
					shape.material_crc = material_crc(flags);
					shape.user_data = user_data_brush | flags;
					shape.name = name + ":brushmodel " + std::to_string(m);
					cmodel_shape[m] = static_cast<unsigned short>(ents.shapes.size());
					ents.shapes.push_back(std::move(shape));
				}

				for (auto t = 0u; t < trigger_models.size(); t++)
				{
					builder::ents_shape shape{};
					const auto& model = trigger_models[t];
					for (auto h = 0; h < model.hullCount; h++)
					{
						convex::hull hull{};
						if (!trigger_hull_solid(trigger_hulls[model.firstHull + h], trigger_slabs.data(), hull))
						{
							ZONETOOL_FATAL("clipmap: hull %d of trigger %u has no closed shape", model.firstHull + h, t);
						}
						shape.convexes.push_back(to_polytope(hull));
					}
					shape.contents = static_cast<int>(ents_trigger_contents);
					shape.entity_contents = ents_trigger_contents;
					shape.user_data = ents_trigger_user_data;
					shape.name = name + ":trigger " + std::to_string(t);
					trigger_shape[t] = static_cast<unsigned short>(ents.shapes.size());
					ents.shapes.push_back(std::move(shape));
				}

				builder::ents_tag_merge merge{};
				const auto ents_blob = builder::build_ents_shape_list(ents, &merge);
				if (ents_blob.empty())
				{
					ZONETOOL_FATAL("clipmap: the entity shapes could not be built");
				}

				auto* trigger_physics = dummy_physics_asset(trigger_physics_asset, "triggermodeldummy", allocator);
				auto* brushmodel_physics = dummy_physics_asset(brushmodel_physics_asset, "scriptbrushmodeldummy", allocator);

				// ---- MapEnts ----------------------------------------------------------------------

				auto* mapents = allocator.allocate<zonetool::iw7::MapEnts>();
				mapents->name = allocator.duplicate_string(name);

				auto shaped_brushmodels = 0u, shaped_triggers = 0u, volumes = 0u;
				auto entity_string = edit_entities(source_string,
					[&](const std::string& classname, const std::string& model) -> entity_edit
				{
					if (model.size() < 2)
					{
						return {};
					}
					const auto index = std::strtoul(model.data() + 1, nullptr, 10);
					if (model[0] == '?' && index < trigger_shape.size())
					{
						shaped_triggers++;
						return {{}, trigger_physics_asset};
					}
					if (model[0] == '*' && index < asset->numSubModels)
					{
						if (volume_trigger[index] >= 0)
						{
							volumes++;
							shaped_triggers++;
							return {"?" + std::to_string(volume_trigger[index]), trigger_physics_asset};
						}
						if (cmodel_shape[index] != 0xFFFF)
						{
							shaped_brushmodels++;
							return {{}, brushmodel_physics_asset};
						}
					}
					return {};
				});
				auto entities = map_entities::parse(entity_string);
				// BO3's render volumes (fog, lit fog, exposure, sun, LUT) and the worldspawn, as BO3 wrote them, for the
				// level's vision: maps/cp/<map>_bo3_volumes.ents
				{
					std::vector<map_entities::entity> render_volumes;
					for (auto i = 0u; i < entities.size(); i++)
					{
						const auto* classname = entities[i].get("classname");
						if (classname && ((i == 0 && *classname == "worldspawn") || classname->starts_with("volume_")))
						{
							render_volumes.push_back(entities[i]);
						}
					}
					const auto volumes_text = map_entities::write(render_volumes);
					filesystem::file file(name.substr(0, name.rfind('.')) + "_bo3_volumes.ents");
					file.open("wb");
					file.write(volumes_text.data(), volumes_text.size(), 1);
					file.close();
				}
				map_entities::keep_spawnable(entities);
				if (map::zombies_map(asset->name))
				{
					zombies_ents::convert(entities);
				}
				// A key IW7's script has no token for is dropped at load, a warning a line (iw7-mod mapents.cpp: the key,
				// lowercased, through gsc_ctx->token_id; the same table as gsc-tool's here). Such keys (BO3's lighting,
				// Umbra, prefab and zbarrier keys, and script fields IW7 never named) leave the entity string; their values
				// go to maps/cp/<map>_bo3_keys.csv (classname, origin, targetname, key, value) for the level script.
				{
					std::string table = "classname,origin,targetname,key,value\n";
					std::map<std::string, unsigned int> moved;
					for (auto& e : entities)
					{
						const auto field = [&](const char* key)
						{
							const auto* v = e.get(key);
							return v ? *v : std::string{};
						};
						const auto where = field("classname") + "," + field("origin") + "," + field("targetname") + ",";
						std::erase_if(e.keys, [&](const map_entities::key_value& kv)
						{
							if (!kv.quoted || gsc::iw7::gsc_ctx->token_id(utils::string::to_lower(kv.key)) != 0)
							{
								return false;
							}
							table += where + kv.key + "," + kv.value + "\n";
							moved[kv.key]++;
							return true;
						});
					}
					if (!moved.empty())
					{
						// maps/cp/<map>.d3dbsp -> maps/cp/<map>_bo3_keys.csv
						const auto table_path = name.substr(0, name.rfind('.')) + "_bo3_keys.csv";
						filesystem::file file(table_path);
						file.open("wb");
						file.write(table.data(), table.size(), 1);
						file.close();
						std::string summary;
						for (const auto& [key, count] : moved)
						{
							summary += utils::string::va(" %s %u", key.data(), count);
						}
						ZONETOOL_INFO("map_ents: keys IW7 has no script token for moved to %s:%s", table_path.data(), summary.data());
					}
				}
				entity_string = map_entities::write(entities);
				mapents->entityString = allocator.duplicate_string(entity_string);
				mapents->numEntityChars = static_cast<int>(entity_string.size()) + 1;

				mapents->trigger.count = static_cast<unsigned int>(trigger_models.size());
				mapents->trigger.models = allocator.allocate_array<zonetool::iw7::TriggerModel>(trigger_models.size());
				for (auto t = 0u; t < trigger_models.size(); t++)
				{
					mapents->trigger.models[t] = trigger_models[t];
					mapents->trigger.models[t].physicsAsset = trigger_physics;
					mapents->trigger.models[t].physicsShapeOverrideIdx = trigger_shape[t];
				}
				mapents->trigger.hullCount = static_cast<unsigned int>(trigger_hulls.size());
				mapents->trigger.hulls = allocator.allocate_array<zonetool::iw7::TriggerHull>(trigger_hulls.size());
				std::memcpy(mapents->trigger.hulls, trigger_hulls.data(), trigger_hulls.size() * sizeof(zonetool::iw7::TriggerHull));
				mapents->trigger.slabCount = static_cast<unsigned int>(trigger_slabs.size());
				mapents->trigger.slabs = allocator.allocate_array<zonetool::iw7::TriggerSlab>(trigger_slabs.size());
				std::memcpy(mapents->trigger.slabs, trigger_slabs.data(), trigger_slabs.size() * sizeof(zonetool::iw7::TriggerSlab));

				mapents->havokEntsShapeData = copy_blob(ents_blob, allocator);
				mapents->havokEntsShapeDataSize = static_cast<unsigned int>(ents_blob.size());

				// the clip planes: stock IW7 planes carry no signbits (byte 17 is 0)
				auto* planes = allocator.allocate_array<zonetool::iw7::cplane_s>(info.planeCount);
				for (auto i = 0; i < info.planeCount; i++)
				{
					std::memcpy(planes[i].normal, info.planes[i].normal, sizeof(planes[i].normal));
					planes[i].dist = info.planes[i].dist;
					planes[i].type = info.planes[i].type;
				}

				auto* model_info = allocator.allocate<zonetool::iw7::ClipInfo>();
				model_info->planeCount = info.planeCount;
				model_info->planes = planes;

				mapents->numSubModels = asset->numSubModels;
				mapents->cmodels = allocator.allocate_array<zonetool::iw7::cmodel_t>(asset->numSubModels);
				for (auto m = 0u; m < asset->numSubModels; m++)
				{
					const auto& src = asset->cmodels[m];
					auto& dst = mapents->cmodels[m];
					for (auto k = 0; k < 3; k++)
					{
						dst.bounds.midPoint[k] = (src.mins[k] + src.maxs[k]) * 0.5f;
						dst.bounds.halfSize[k] = (src.maxs[k] - src.mins[k]) * 0.5f;
					}
					dst.radius = src.radius;
					dst.info = model_info;
					dst.physicsShapeOverrideIdx = m == 0 ? 0 : cmodel_shape[m];
					dst.physicsAsset = cmodel_shape[m] != 0xFFFF && m != 0 ? brushmodel_physics : nullptr;
					dst.navObstacleIdx = m == 0 ? 0 : 0xFFFF;
					dst.edgeFirstIndex = 0;
				}

				// runtime slots: 64 dynents for script-spawned scriptables
				// and 500 scriptable instances (BO3's 500 dynents are empty runtime slots as well)
				constexpr unsigned short reserved_dynents = 64;
				constexpr unsigned int runtime_scriptables = 500;
				mapents->dynEntCount[0] = reserved_dynents;
				mapents->dynEntCount[1] = 0;
				mapents->dynEntCountTotal = reserved_dynents;
				mapents->dynEntDefList[0] = allocator.allocate_array<zonetool::iw7::DynEntityDef>(reserved_dynents);
				mapents->dynEntPoseList[0][0] = allocator.allocate_array<zonetool::iw7::DynEntityPose>(reserved_dynents);
				mapents->dynEntPoseList[1][0] = allocator.allocate_array<zonetool::iw7::DynEntityPose>(reserved_dynents);
				mapents->dynEntClientList[0][0] = allocator.allocate_array<zonetool::iw7::DynEntityClient>(reserved_dynents);
				mapents->dynEntClientList[1][0] = allocator.allocate_array<zonetool::iw7::DynEntityClient>(reserved_dynents);
				for (auto i = 0; i < reserved_dynents; i++)
				{
					auto& dyn = mapents->dynEntDefList[0][i];
					dyn.type = zonetool::iw7::DYNENT_TYPE_SCRIPTABLEINST;
					dyn.instanceIndex = static_cast<short>(runtime_scriptables);
					dyn.unk4 = static_cast<short>(i);
					dyn.spawnActive = true;
					dyn.unk5 = 4;
					dyn.unk6 = 0x666;
					dyn.spawnEnabled = true;
				}
				mapents->dynEntGlobalIdList[0] = allocator.allocate_array<zonetool::iw7::DynEntityGlobalId>(reserved_dynents);
				mapents->dynEntGlobalIdList[1] = allocator.allocate_array<zonetool::iw7::DynEntityGlobalId>(reserved_dynents);
				for (auto i = 0u; i < reserved_dynents; i++)
				{
					mapents->dynEntGlobalIdList[0][i] = {0, i};
					mapents->dynEntGlobalIdList[1][i] = {1, i};
				}
				std::fill_n(&mapents->dynEntPhysicsSetupHead[0][0], 4, static_cast<unsigned short>(0xFFFF));
				std::fill_n(&mapents->dynEntPhysicsSetupTail[0][0], 4, static_cast<unsigned short>(0xFFFF));

				auto& scriptables = mapents->scriptableMapEnts;
				scriptables.totalInstanceCount = runtime_scriptables;
				scriptables.runtimeInstanceCount = runtime_scriptables;
				scriptables.reservedInstanceCount = runtime_scriptables;
				scriptables.instances = allocator.allocate_array<zonetool::iw7::ScriptableInstance>(runtime_scriptables);
				scriptables.runtimeData.partRuntimeCount = static_cast<int>(runtime_scriptables);
				scriptables.runtimeData.partRuntime = allocator.allocate_array<zonetool::iw7::ScriptablePartRuntime>(runtime_scriptables);
				scriptables.runtimeData.partRuntimeLocalClientCount = static_cast<int>(runtime_scriptables);
				scriptables.runtimeData.partRuntimeLocalClient[0] = allocator.allocate_array<zonetool::iw7::ScriptablePartRuntime>(runtime_scriptables);
				scriptables.runtimeData.partRuntimeLocalClient[1] = allocator.allocate_array<zonetool::iw7::ScriptablePartRuntime>(runtime_scriptables);
				for (auto i = 0; i < 2; i++)
				{
					scriptables.reservedDynents[i].numReservedDynents = reserved_dynents;
					scriptables.reservedDynents[i].reservedDynents = allocator.allocate_array<zonetool::iw7::ScriptableReservedDynent>(reserved_dynents);
				}

				// ---- clipMap ----------------------------------------------------------------------

				auto* clip = allocator.allocate<zonetool::iw7::clipMap_t>();
				clip->name = allocator.duplicate_string(name);
				clip->info.planeCount = info.planeCount;
				clip->info.planes = planes;
				clip->pInfo = &clip->info;

				// the static models within the physics budget (the rest are in the world shape)
				std::unordered_map<const XModel*, zonetool::iw7::XModel*> model_stubs;
				const auto kept_static_models = asset->numStaticModels - baked_count;
				clip->numStaticModels = kept_static_models;
				clip->staticModelList = allocator.allocate_array<zonetool::iw7::cStaticModel_s>(kept_static_models);
				clip->staticModelCollisionModelList.numModels = kept_static_models;
				clip->staticModelCollisionModelList.staticModelIndex = allocator.allocate_array<int>(kept_static_models);
				for (auto i = 0u, k = 0u; i < asset->numStaticModels; i++)
				{
					if (baked_static_model[i])
					{
						continue;
					}
					const auto& src = asset->staticModelList[i];
					auto& dst = clip->staticModelList[k];
					auto& stub = model_stubs[src.xmodel];
					if (!stub)
					{
						stub = allocator.allocate<zonetool::iw7::XModel>();
						stub->name = allocator.duplicate_string(xmodel::iw7_name(src.xmodel->name));
					}
					dst.xmodel = stub;
					std::memcpy(dst.origin, src.origin, sizeof(dst.origin));
					std::memcpy(dst.invScaledAxis, src.invScaledAxis, sizeof(dst.invScaledAxis));
					clip->staticModelCollisionModelList.staticModelIndex[k] = static_cast<int>(k);
					k++;
				}

				clip->mapEnts = mapents;

				// BO3 has no stages; stock stage 0 is at the origin, has no trigger and uses sun light 1
				clip->stageCount = 1;
				clip->stages = allocator.allocate<zonetool::iw7::Stage>();
				clip->stages->name = "stage 0";
				clip->stages->triggerIndex = 1024;
				clip->stages->sunPrimaryLightIndex = 1;
				clip->stages->entityUID = 0x3A83126F;

				for (auto k = 0; k < 3; k++)
				{
					clip->broadphaseMin[k] = -131072.0f;
					clip->broadphaseMax[k] = 131072.0f;
				}

				clip->havokWorldShapeData = copy_blob(world_blob, allocator);
				clip->havokWorldShapeDataSize = static_cast<unsigned int>(world_blob.size());
				clip->checksum = asset->checksum;

				// ---- report -----------------------------------------------------------------------

				surface_parms::report(stats, "clipmap");

				ZONETOOL_INFO("clipmap \"%s\": %zu world tags; %u static models (%zu models); %u brush models (%u with a shape, "
					"%u flat trisoup triangles), %u info_volumes as trigger models, %zu trigger models (%zu hulls, %zu slabs); "
					"entity tags %zu (%zu from the world table, %zu new)", name.data(), world_tags.size(), clip->numStaticModels,
					model_stubs.size(), asset->numSubModels - 1, shaped_brushmodels, flat_triangles, volumes, trigger_models.size(),
					trigger_hulls.size(), trigger_slabs.size(), merge.total, merge.prefix, merge.appended);
				for (const auto& [value, count] : contents_hist)
				{
					if (count)
					{
						ZONETOOL_INFO("clipmap: %6u world brushes with contents 0x%08X", count, value);
					}
				}

				zonetool::iw7::physics_asset::dump(trigger_physics);
				zonetool::iw7::physics_asset::dump(brushmodel_physics);
				zonetool::iw7::map_ents::dump(mapents);
				zonetool::iw7::clip_map::dump(clip);
			}
		}
	}
}
