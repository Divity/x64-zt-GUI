#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "xmodel_collision.hpp"

#include "zonetool/t7/converter/iw7/convex.hpp"
#include "zonetool/t7/converter/iw7/model_offset.hpp"
#include "zonetool/t7/converter/iw7/surface_parms.hpp"

#include "zonetool/iw7/common/havok_builder.hpp"

#include <utils/string.hpp>

#include <array>

// How the two games split model collision (BO3 = BlackOps3 client/dedi, IW7 = iw7_ship):
// * BO3: collSurfs are the collision LOD mesh cut per bone: the bullet mesh debug draw (client 0x1422DDF10)
//   pairs every rigid vertex list of the collLod mesh with the next collSurf and draws it with that collSurf's
//   bone. Their triangles are in model space at the base pose. collmaps are the physics simulation's shapes
//   (brushes and cylinders, model space, bone -1).
// * IW7: a static model collides only when its XModel has a PhysicsAsset (0x574CF0 skips it otherwise). The
//   PhysicsAsset's bodies go into the physics worlds; stock prop bodies filter as 0x3180, which holds no solid
//   bit. physicsLODData goes into the detail worlds 1 and 4 (0x550400): one compressed mesh per bone, named by
//   the bone, in bone space, placed at the bone's baseMat (0x57C8A0, 0x574CF0), its tags {surface flags,
//   contents} (decoded by 0x59B480). Stock collSurfs carry the pairs the tags do, with model space bounds.
// So collSurfs become the physics LOD and IW7's collSurfs; collmaps (with a physics preset) become a dynamic
// PhysicsAsset like stock props; anything else that collides gets a static mesh PhysicsAsset (a stock form as
// well) so its detail shapes register.

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace xmodel_collision
		{
			namespace
			{
				namespace builder = zonetool::iw7::havok::builder;

				using vec3 = std::array<float, 3>;

				constexpr auto havok_scale = 1.0f / 32.0f;

				// stock prop bodies filter as 0x3180
				constexpr std::uint32_t prop_body_contents = 0x3180;

				// BO3 PhysPreset mass is IW7's / 1000 (models both games ship weigh 0.001 in BO3, 1.0 in IW7)
				constexpr auto preset_mass_scale = 1000.0f;

				// BO3 collmap geom types: 2 carries a brush; 3 is a cylinder along orientation[0], half height
				// halfLengths[0], radius halfLengths[1] (not a capsule: it spans its model's extent exactly)
				constexpr int phys_geom_brush = 2;
				constexpr int phys_geom_cylinder = 3;
				constexpr int cylinder_sides = 16;

				// A collision triangle is stored as its plane and two edge planes: corner c solves
				// dot(plane, p) = plane.w, dot(svec, p) - svec.w = s, dot(tvec, p) - tvec.w = t for
				// (s, t) = (0, 0), (1, 0), (0, 1). That order is clockwise around the (outward) plane normal; IW7
				// meshes are counter-clockwise seen from outside, so the corners come out as (0, 0), (0, 1), (1, 0).
				bool coll_triangle(const XModelCollTri& tri, vec3 (&out)[3])
				{
					const double r[3][3] =
					{
						{tri.plane[0], tri.plane[1], tri.plane[2]},
						{tri.svec[0], tri.svec[1], tri.svec[2]},
						{tri.tvec[0], tri.tvec[1], tri.tvec[2]},
					};
					const auto cross = [](const double* a, const double* b, double (&o)[3])
					{
						o[0] = a[1] * b[2] - a[2] * b[1];
						o[1] = a[2] * b[0] - a[0] * b[2];
						o[2] = a[0] * b[1] - a[1] * b[0];
					};
					double c12[3], c20[3], c01[3];
					cross(r[1], r[2], c12);
					cross(r[2], r[0], c20);
					cross(r[0], r[1], c01);
					const auto det = r[0][0] * c12[0] + r[0][1] * c12[1] + r[0][2] * c12[2];
					if (std::fabs(det) < 1e-12)
					{
						return false;
					}

					static constexpr double st[3][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}};
					for (auto c = 0; c < 3; c++)
					{
						const double rhs[3] = {tri.plane[3], st[c][0] + tri.svec[3], st[c][1] + tri.tvec[3]};
						for (auto k = 0; k < 3; k++)
						{
							out[c][k] = static_cast<float>((rhs[0] * c12[k] + rhs[1] * c20[k] + rhs[2] * c01[k]) / det);
						}
					}
					return true;
				}

				double triangle_area(const vec3 (&c)[3])
				{
					const double e1[3] = {c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]};
					const double e2[3] = {c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]};
					const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
					return 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
				}

				void add_plane(std::vector<convex::plane>& planes, const float nx, const float ny, const float nz, const float dist)
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
				}

				// BO3's collmap brushes are tool-made hulls whose planes meet almost at a point in places, where the
				// clipper's per-face welding leaves unclosed hulls; closed_from_planes always closes them.
				bool closed_hull(const std::vector<convex::plane>& planes, convex::hull& out)
				{
					return convex::closed_from_planes(planes, out);
				}

				// a collmap brush: its box cut by its sides (the box faces repeat among the sides)
				bool brush_hull(const BrushWrapper& brush, convex::hull& out)
				{
					std::vector<convex::plane> planes;
					for (auto axis = 0; axis < 3; axis++)
					{
						float n[3] = {0.0f, 0.0f, 0.0f};
						n[axis] = -1.0f;
						add_plane(planes, n[0], n[1], n[2], -brush.mins[axis]);
						n[axis] = 1.0f;
						add_plane(planes, n[0], n[1], n[2], brush.maxs[axis]);
					}
					for (auto s = 0u; s < brush.numsides; s++)
					{
						const auto* plane = brush.sides[s].plane;
						if (plane)
						{
							add_plane(planes, plane->normal[0], plane->normal[1], plane->normal[2], plane->dist);
						}
					}
					return closed_hull(planes, out);
				}

				bool cylinder_hull(const PhysGeomInfo& geom, convex::hull& out)
				{
					const auto* axis = geom.orientation[0];
					const auto* u = geom.orientation[1];
					const auto* v = geom.orientation[2];
					const auto* centre = geom.offset;
					const auto half_height = geom.halfLengths[0];
					const auto radius = geom.halfLengths[1];
					const auto along = axis[0] * centre[0] + axis[1] * centre[1] + axis[2] * centre[2];

					std::vector<convex::plane> planes;
					add_plane(planes, axis[0], axis[1], axis[2], along + half_height);
					add_plane(planes, -axis[0], -axis[1], -axis[2], -along + half_height);
					for (auto i = 0; i < cylinder_sides; i++)
					{
						const auto angle = 6.283185307179586 * i / cylinder_sides;
						const auto c = static_cast<float>(std::cos(angle));
						const auto s = static_cast<float>(std::sin(angle));
						const float n[3] = {c * u[0] + s * v[0], c * u[1] + s * v[1], c * u[2] + s * v[2]};
						add_plane(planes, n[0], n[1], n[2], n[0] * centre[0] + n[1] * centre[1] + n[2] * centre[2] + radius);
					}
					return closed_hull(planes, out);
				}

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

				zonetool::iw7::PhysicsAsset* physics_asset(const char* name, const std::vector<std::uint8_t>& blob,
					utils::memory::allocator& allocator)
				{
					// the fields every stock prop asset carries: one body, no constraints, one empty sfx and vfx
					// event slot
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
			}

			bone_frame frame_of(const zonetool::iw7::DObjAnimMat& mat)
			{
				bone_frame out{};
				const auto length = std::sqrt(static_cast<double>(mat.quat[0]) * mat.quat[0] + static_cast<double>(mat.quat[1]) * mat.quat[1]
					+ static_cast<double>(mat.quat[2]) * mat.quat[2] + static_cast<double>(mat.quat[3]) * mat.quat[3]);
				for (auto k = 0; k < 4; k++)
				{
					out.q[k] = length > 0.0 ? mat.quat[k] / length : (k == 3 ? 1.0 : 0.0);
				}
				for (auto k = 0; k < 3; k++)
				{
					out.t[k] = mat.trans[k];
				}
				return out;
			}

			// conj(q) * (p - t) * q
			std::array<float, 3> to_bone(const bone_frame& bone, const std::array<float, 3>& p)
			{
				const double v[3] = {p[0] - bone.t[0], p[1] - bone.t[1], p[2] - bone.t[2]};
				const double u[3] = {bone.q[0], bone.q[1], bone.q[2]};
				const auto w = bone.q[3];
				const double uv[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
				const double uuv[3] = {u[1] * uv[2] - u[2] * uv[1], u[2] * uv[0] - u[0] * uv[2], u[0] * uv[1] - u[1] * uv[0]};
				std::array<float, 3> out{};
				for (auto k = 0; k < 3; k++)
				{
					out[k] = static_cast<float>(v[k] - 2.0 * w * uv[k] + 2.0 * uuv[k]);
				}
				return out;
			}

			void convert(XModel* asset, zonetool::iw7::XModel* model, const xmodel_mesh::bone_map& bones, utils::memory::allocator& allocator)
			{
				const auto bone_count = asset->numBones + asset->numCosmeticBones;
				const auto has_collision = asset->numCollSurfs > 0 || (asset->numCollmaps && asset->collmaps);

				const auto& offset = model_offset::get(asset->name);
				if (has_collision && (offset.valid || !offset.bones.empty()))
				{
					ZONETOOL_FATAL("xmodel \"%s\": the model offset/bone renames of t7_to_iw7_model_offsets.json are not "
						"applied to collision", asset->name);
				}

				surface_parms::stats stats{};
				model->contents = static_cast<int>(surface_parms::contents(asset->contents, stats));

				model->numCollSurfs = static_cast<unsigned short>(asset->numCollSurfs);
				model->collSurfs = allocator.allocate_array<zonetool::iw7::XModelCollSurf_s>(asset->numCollSurfs);

				// per IW7 bone (a merged BO3 bone's collision rides its parent)
				std::vector<builder::mesh_input> bone_meshes(model->numBones);
				builder::mesh_input model_mesh{};
				std::map<std::uint32_t, double> surface_area;
				auto degenerate = 0u;

				for (auto i = 0; i < asset->numCollSurfs; i++)
				{
					auto& src = asset->collSurfs[i];
					auto& dst = model->collSurfs[i];
					compute(&dst.bounds, src.mins, src.maxs);
					const auto contents = surface_parms::contents(src.contents, stats);
					const auto flags = surface_parms::surface_flags(src.surfFlags, stats);
					dst.contents = static_cast<int>(contents);
					dst.surfFlags = static_cast<int>(flags);
					if (src.boneIdx < 0 || src.boneIdx >= bone_count)
					{
						if (!contents)
						{
							dst.boneIdx = src.boneIdx;
							continue; // collides with nothing in IW7
						}
						ZONETOOL_FATAL("xmodel \"%s\": collision surface %d is on bone %d of %d", asset->name, i, src.boneIdx,
							bone_count);
					}
					const auto iw7_bone = bones(static_cast<std::uint16_t>(src.boneIdx));
					dst.boneIdx = iw7_bone;
					if (!contents)
					{
						continue; // collides with nothing in IW7
					}

					const auto bone = frame_of(model->baseMat[iw7_bone]);
					for (auto t = 0; t < src.numCollTris; t++)
					{
						vec3 corners[3];
						if (!coll_triangle(src.collTris[t], corners) || triangle_area(corners) < 1e-6)
						{
							degenerate++;
							continue;
						}
						surface_area[flags] += triangle_area(corners);

						builder::triangle tri{};
						tri.surface_tag = 0;
						tri.contents = static_cast<int>(contents);
						tri.material_crc = surface_parms::material_crc(flags);
						tri.user_data = flags;

						for (auto c = 0; c < 3; c++)
						{
							for (auto k = 0; k < 3; k++)
							{
								tri.verts[c][k] = corners[c][k] * havok_scale;
							}
						}
						model_mesh.triangles.push_back(tri);

						for (auto c = 0; c < 3; c++)
						{
							const auto local = to_bone(bone, corners[c]);
							for (auto k = 0; k < 3; k++)
							{
								tri.verts[c][k] = local[k] * havok_scale;
							}
						}
						bone_meshes[iw7_bone].triangles.push_back(tri);
					}
				}

				if (degenerate)
				{
					ZONETOOL_WARNING("xmodel \"%s\": %u degenerate collision triangles left out", asset->name, degenerate);
				}

				// the physics material of the whole model: the surface covering the most area
				std::uint32_t main_flags = 0;
				auto main_area = -1.0;
				for (const auto& [flags, area] : surface_area)
				{
					if (area > main_area)
					{
						main_flags = flags;
						main_area = area;
					}
				}
				const auto material = surface_parms::material_crc(main_flags);
				const std::string root_bone = bone_count ? SL_ConvertToString(asset->boneNames[0]) : "tag_origin";

				// ---- physics LOD: one shape per bone that has any, in bone order ----------------------

				std::vector<builder::lod_shape> shapes;
				std::vector<zonetool::iw7::scr_string_t> shape_names;
				for (auto b = 0; b < model->numBones; b++)
				{
					if (bone_meshes[b].triangles.empty())
					{
						continue;
					}
					const auto bo3_bone = bones.empty() ? b : bones.to_bo3[b];
					shapes.push_back({SL_ConvertToString(asset->boneNames[bo3_bone]), std::move(bone_meshes[b])});
					shape_names.push_back(static_cast<zonetool::iw7::scr_string_t>(asset->boneNames[bo3_bone]));
				}

				if (!shapes.empty())
				{
					const auto lod = builder::build_model_physics_lod(shapes);
					if (lod.empty())
					{
						ZONETOOL_FATAL("xmodel \"%s\": the physics LOD could not be built", asset->name);
					}
					model->physicsLODData = allocator.allocate_array<char>(lod.size());
					std::memcpy(model->physicsLODData, lod.data(), lod.size());
					model->physicsLODDataSize = static_cast<unsigned int>(lod.size());
					model->physicsLODDataNameCount = static_cast<unsigned int>(shape_names.size());
					model->physicsLODDataNames = allocator.allocate_array<zonetool::iw7::scr_string_t>(shape_names.size());
					std::memcpy(model->physicsLODDataNames, shape_names.data(), shape_names.size() * sizeof(zonetool::iw7::scr_string_t));
				}

				// ---- PhysicsAsset -----------------------------------------------------------------------

				std::vector<std::uint8_t> blob;
				if (asset->physPreset && asset->numCollmaps && asset->collmaps)
				{
					builder::dynamic_physics_asset_input input{};
					for (auto c = 0; c < asset->numCollmaps; c++)
					{
						const auto& collmap = asset->collmaps[c];
						if (collmap.boneIndex >= 0)
						{
							ZONETOOL_FATAL("xmodel \"%s\": collmap %d is on bone %d; only model space collmaps (bone -1) "
								"were measured", asset->name, c, collmap.boneIndex);
						}
						if (!collmap.geomList)
						{
							continue;
						}
						for (auto g = 0u; g < collmap.geomList->count; g++)
						{
							const auto& geom = collmap.geomList->geoms[g];
							convex::hull hull{};
							auto built = false;
							if (geom.type == phys_geom_brush && geom.brush)
							{
								built = brush_hull(*geom.brush, hull);
							}
							else if (geom.type == phys_geom_cylinder && !geom.brush)
							{
								built = cylinder_hull(geom, hull);
							}
							else
							{
								ZONETOOL_FATAL("xmodel \"%s\": collmap %d geom %u has type %d%s, which was not measured",
									asset->name, c, g, geom.type, geom.brush ? " with a brush" : "");
							}
							if (!built)
							{
								ZONETOOL_WARNING("xmodel \"%s\": collmap %d geom %u has no closed hull and was left out",
									asset->name, c, g);
								continue;
							}
							input.convexes.push_back(to_polytope(hull));
						}
					}

					if (!input.convexes.empty())
					{
						if (!(asset->physPreset->mass > 0.0f))
						{
							ZONETOOL_FATAL("xmodel \"%s\": physics preset \"%s\" has mass %g", asset->name,
								asset->physPreset->name, asset->physPreset->mass);
						}
						input.body_name = root_bone;
						input.mass = asset->physPreset->mass * preset_mass_scale;
						input.material_crc = material;
						input.body_contents = prop_body_contents;
						input.scale = havok_scale;
						blob = builder::build_dynamic_physics_asset(input);
						if (blob.empty())
						{
							ZONETOOL_FATAL("xmodel \"%s\": the collmap physics asset could not be built", asset->name);
						}
					}
				}

				if (blob.empty() && !model_mesh.triangles.empty())
				{
					builder::physics_asset_input input{};
					input.body_name = root_bone;
					input.material_crc = material;
					input.body_contents = prop_body_contents;
					blob = builder::build_model_physics_asset(model_mesh, input);
					if (blob.empty())
					{
						ZONETOOL_FATAL("xmodel \"%s\": the mesh physics asset could not be built", asset->name);
					}
				}

				model->physicsAsset = blob.empty() ? nullptr : physics_asset(asset->name, blob, allocator);

				surface_parms::report(stats, asset->name);
			}

			std::vector<triangle> triangles(XModel* asset)
			{
				const auto bone_count = asset->numBones + asset->numCosmeticBones;
				surface_parms::stats stats{};
				std::vector<triangle> out;
				for (auto i = 0; i < asset->numCollSurfs; i++)
				{
					const auto& src = asset->collSurfs[i];
					const auto contents = surface_parms::contents(src.contents, stats);
					if (!contents || src.boneIdx < 0 || src.boneIdx >= bone_count)
					{
						continue;
					}
					const auto flags = surface_parms::surface_flags(src.surfFlags, stats);
					for (auto t = 0; t < src.numCollTris; t++)
					{
						vec3 corners[3];
						if (!coll_triangle(src.collTris[t], corners) || triangle_area(corners) < 1e-6)
						{
							continue;
						}
						triangle tri{};
						for (auto c = 0; c < 3; c++)
						{
							tri.verts[c] = corners[c];
						}
						tri.contents = contents;
						tri.flags = flags;
						out.push_back(tri);
					}
				}
				return out;
			}

			physics_cost cost(XModel* asset)
			{
				const auto bone_count = asset->numBones + asset->numCosmeticBones;
				surface_parms::stats stats{};
				std::set<int> bones;
				for (auto i = 0; i < asset->numCollSurfs; i++)
				{
					const auto& src = asset->collSurfs[i];
					if (!surface_parms::contents(src.contents, stats) || src.boneIdx < 0 || src.boneIdx >= bone_count)
					{
						continue;
					}
					for (auto t = 0; t < src.numCollTris; t++)
					{
						vec3 corners[3];
						if (coll_triangle(src.collTris[t], corners) && triangle_area(corners) >= 1e-6)
						{
							bones.insert(src.boneIdx);
							break;
						}
					}
				}
				auto collmap_geoms = false;
				for (auto c = 0; asset->physPreset && asset->collmaps && c < asset->numCollmaps; c++)
				{
					collmap_geoms |= asset->collmaps[c].geomList && asset->collmaps[c].geomList->count > 0;
				}
				return { (!bones.empty() || collmap_geoms) ? 1u : 0u, static_cast<unsigned int>(bones.size()) };
			}
		}
	}
}
