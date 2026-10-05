#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "static_model_clusters.hpp"
#include "zonetool/t7/converter/iw7/map_common.hpp"

#include "comworld.hpp"
#include "material.hpp"
#include "probe_lighting.hpp"
#include "world_material.hpp"
#include "world_material_bake.hpp"
#include "world_techset_donors.hpp"
#include "world_lightmap.hpp"
#include "xmodel.hpp"
#include "xmodel_mesh.hpp"
#include "../parallel.hpp"

#include "zonetool/iw7/assets/xmodel.hpp"
#include "zonetool/iw7/assets/xsurface.hpp"

#include "game/shared.hpp"

#include <utils/string.hpp>

namespace zonetool::t7
{
	namespace converter::iw7::static_model_clusters
	{
		namespace
		{
			// What the members of one cluster share: a cell of the grid (world units), their probe lighting's luminance
			// (over six directions) within a step of log2 x steps_per_stop (0: any), and optionally the set of local lights
			// reaching them. The finest grouping whose fullest view (static_model_clusters record_bytes) stays within record_budget is taken:
			// BO3 maps may show most of themselves at once, so culling cannot keep a view within
			// IW7's 16384 addressable record bytes. A member's brightness is kept either way (its vertices' self
			// visibility, dump_all), what a coarser grouping gives up is its light's colour and direction.
			struct grouping
			{
				float cell;
				float steps_per_stop;
				bool by_lights;
				// members larger than max_member_radius join too (the last steps: their records alone overflow a view)
				bool large;
			};
			constexpr grouping groupings[] = {
				{ 1024.0f, 2.0f, true, false }, { 1024.0f, 1.0f, true, false }, { 1024.0f, 0.5f, true, false },
				{ 1024.0f, 0.0f, true, false }, { 2048.0f, 0.0f, true, false }, { 2048.0f, 0.0f, false, false },
				{ 4096.0f, 0.0f, false, false }, { 65536.0f, 0.0f, false, false }, { 256.0f, 0.0f, false, true }, { 512.0f, 0.0f, false, true },
				{ 1024.0f, 0.0f, false, true }, { 2048.0f, 0.0f, false, true },
				{ 4096.0f, 0.0f, false, true }, { 65536.0f, 0.0f, false, true },
			};
			// IW7's 16384 record bytes less 4 KB (waterpark at an estimated 15336 still lost models looking into the park:
			// the estimate undercounts); it counts each unmerged model's instances split over
			// twice its LODs (0x140DCE680: a record per LOD x flag x draw type bucket of a model's run)
			constexpr std::size_t record_budget = 12288;
			// a view's opaque static model surface list: 8192 bytes in IW7 (0x140DCFF30's table 0x14153C768, camera region 0), 16
			// times that in iw7-mod's (renderer.cpp smodel_list_scale, 32); past it IW7 stops drawing static models (warning 24);
			// 0x140DCF3D0 writes 2 bytes a surface and a 10 byte header where the material changes or 16 surfaces share one.
			// Three quarters of iw7-mod's
			constexpr std::size_t surface_budget = 8192 * 32 * 3 / 4;
			// the surfaces waiting for their material's turn (0x140DCE680: those not of a model's first material until the
			// drawing passes their material), 256 a list in IW7 (0x14153C768), 32 times that in iw7-mod's; past it they are not
			// drawn (warning 25). Three quarters of iw7-mod's
			constexpr std::size_t delayed_budget = 256 * 32 * 3 / 4;
			// the merged and shadow proxy geometry a zone may hold (bytes, uncompressed, proxies before welding): LZ4 keeps about
			// 0.74 of it; with the rest of a large map (about 460 MB compressed) its fastfile stays near 1.8 GB, under the 2 GiB
			// IW7 can read (Water Park with cluster far LODs at 1.95 GB of it: a 2.016 GB fastfile)
			constexpr std::size_t geometry_budget = 1940ull * 1000ull * 1000ull;
			// a world group's extent (static_model_plan world_clusters): an AABB tree leaf, culled by its bounds and per surface
			constexpr float world_cell = 1024.0f;
			// world surfaces have no LOD or cull distance, so the world groups' triangles are drawn wherever Umbra sees them:
			// every member starts at its last LOD and the largest go back to their first while the total stays within this
			// (Water Park at every member's chosen LOD, 29.6M triangles: 2-3 fps)
			constexpr std::size_t world_triangle_budget = 3000000;
			// the largest placed radius (world units) a world group member has: past it a model keeps its LODs as a static model
			constexpr float world_max_radius = 1e9f;
			// the most triangles a world group member has at its last LOD
			constexpr std::size_t world_member_triangles = 400;
			// models reaching further than this from their origin (world units) stay single: buildings gain nothing
			constexpr float max_member_radius = 256.0f;
			// the smallest self visibility (7 bits, 1/127): a member darker than this against its cluster's lighting
			constexpr float min_visibility = 1.0f / 127.0f;
			// IW7 XSurface counts are 16-bit; a LOD of more than 16 surfaces breaks the static model draw
			constexpr std::size_t max_surfaces = 16;
			constexpr std::size_t max_vertices = 0xFFFF;

			// BO3 static model flags
			constexpr unsigned short t7_smodel_no_shadow = 0x1;

			// shadow proxies (static_model_clusters.hpp): members are taken in the order of their 2048 unit cell along a
			// Morton curve, so a proxy's surfaces stay within one area
			constexpr float proxy_cell = 2048.0f;
			// the opaque caster material of every proxy: stock mo_shadowcaster (camera region 11, casts), as BO3's
			// global_invisible converts
			constexpr auto proxy_material = "mo/t7_shadow_proxy";
			constexpr auto proxy_techset = "mo_shadowcaster";
			constexpr unsigned char iw7_region_shadow_only = 11;

			plan current_plan;
			const GfxWorld* planned_world = nullptr;
			std::unordered_set<const XModel*> models_left_single;

			// how a model surface's material casts the sun's shadow, as BO3 draws it (world_material::model_blocks_sun):
			// not at all, opaque (any proxy material draws it), or alpha tested (through its own texture)
			enum class caster_kind : char
			{
				none,
				opaque,
				alpha_tested,
			};
			std::unordered_map<const Material*, caster_kind> caster_kinds;

			std::string material_file_name(std::string name)
			{
				std::ranges::replace(name, '*', '_'); // as the material writers name their files
				return name;
			}

			// A converted material written again as `copy` in camera region 11: the camera's lists skip it (0x140DCE680),
			// its shadow technique draws it. Its techset state and constant buffer files are copied along. False when the
			// material was not written.
			bool write_shadow_copy(const std::string& material, const std::string& copy)
			{
				const auto root = filesystem::get_dump_path();
				const auto read = [&](const std::string& path, std::string& out)
				{
					std::ifstream in(root + path, std::ios::binary);
					if (!in)
					{
						return false;
					}
					out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
					return true;
				};
				const auto write = [](const std::string& path, const std::string& data)
				{
					filesystem::file file(path);
					file.open("wb");
					if (!file.get_fp())
					{
						ZONETOOL_FATAL("shadow proxies: could not write %s", path.data());
					}
					file.write(data.data(), data.size(), 1);
					file.close();
				};

				const auto from = material_file_name(material), to = material_file_name(copy);
				std::string text;
				if (!read("materials\\" + from + ".json", text))
				{
					return false;
				}
				auto j = ordered_json::parse(text);
				const auto techset = j["techniqueSet->name"].get<std::string>();
				j["cameraRegion"] = iw7_region_shadow_only;
				write("materials\\" + to + ".json", j.dump(4));
				for (const auto* file : { "state\\%s\\%s.stateinfo", "state\\%s\\%s.statebits", "state\\%s\\%s.statebitsmap",
					"constantbuffer\\%s\\%s.cbi", "constantbuffer\\%s\\%s.cbt" })
				{
					std::string data;
					if (read("techsets\\" + std::string(utils::string::va(file, techset.data(), from.data())), data))
					{
						write("techsets\\" + std::string(utils::string::va(file, techset.data(), to.data())), data);
					}
				}
				return true;
			}

			struct model_info
			{
				bool eligible = false;
				unsigned int lods = 0;
				// per LOD: the mesh, its surfaces' materials and vertex / triangle counts
				std::vector<std::pair<XModelMesh*, std::vector<const Material*>>> meshes;
				std::vector<std::vector<unsigned int>> vertices;
				std::vector<std::vector<unsigned int>> triangles;
			};

			model_info inspect(const XModel* model)
			{
				model_info info{};
				if (!model || !model->name || model->name[0] == '*')
				{
					return info;
				}
				info.lods = xmodel::lod_count(model);
				info.meshes = xmodel::lod_meshes(model);
				if (!info.lods || info.meshes.size() != info.lods)
				{
					return info;
				}
				for (const auto& [mesh, materials] : info.meshes)
				{
					if (mesh->numSurfs > max_surfaces || !xmodel_mesh::rigid_with_data(mesh))
					{
						return info;
					}
					std::vector<unsigned int> v, t;
					for (auto s = 0; s < mesh->numSurfs; s++)
					{
						v.push_back(mesh->surfs[s].vertCount);
						t.push_back(mesh->surfs[s].triCount);
					}
					info.vertices.push_back(std::move(v));
					info.triangles.push_back(std::move(t));
				}
				info.eligible = true;
				return info;
			}

			float determinant(const float (*axis)[3])
			{
				return axis[0][0] * (axis[1][1] * axis[2][2] - axis[1][2] * axis[2][1])
					- axis[0][1] * (axis[1][0] * axis[2][2] - axis[1][2] * axis[2][0])
					+ axis[0][2] * (axis[1][0] * axis[2][1] - axis[1][1] * axis[2][0]);
			}

			// what one cluster LOD holds: its surfaces' materials and vertex / triangle totals; a material takes another surface
			// when its last one would pass 65535 vertices or triangles (place_surface), at most 16 surfaces in all
			struct lod_budget
			{
				std::vector<const Material*> materials;
				std::vector<std::size_t> vertices;
				std::vector<std::size_t> triangles;

				// the surface a mesh surface of `material` with v vertices and t triangles goes to (a new one at the end)
				std::size_t place(const Material* material, const std::size_t v, const std::size_t t) const
				{
					for (auto at = this->materials.size(); at-- > 0;)
					{
						if (this->materials[at] == material)
						{
							if (this->vertices[at] + v <= max_vertices && this->triangles[at] + t <= max_vertices)
							{
								return at;
							}
							break;
						}
					}
					return this->materials.size();
				}

				void put(const Material* material, const std::size_t v, const std::size_t t)
				{
					const auto at = this->place(material, v, t);
					if (at == this->materials.size())
					{
						this->materials.push_back(material);
						this->vertices.push_back(0);
						this->triangles.push_back(0);
					}
					this->vertices[at] += v;
					this->triangles[at] += t;
				}

				bool fits(const model_info& info, const unsigned int lod) const
				{
					auto after = *this;
					after.add(info, lod);
					return after.materials.size() <= max_surfaces;
				}

				void add(const model_info& info, const unsigned int lod)
				{
					const auto& [mesh, mats] = info.meshes[lod];
					for (auto s = 0u; s < mats.size(); s++)
					{
						this->put(mats[s], info.vertices[lod][s], info.triangles[lod][s]);
					}
				}
			};

			// what one shadow proxy holds: its opaque surfaces (vertex and triangle totals), filled in order as dump_all
			// appends them, and one surface per alpha tested material
			// the LOD a shadow proxy copies: the model's last that has triangles (a shadow needs no near-LOD detail, and every
			// proxy is drawn again in each sun cascade: Water Park's third-LOD proxies held 15.3M triangles)
			unsigned int proxy_lod(const model_info& info)
			{
				for (auto lod = info.lods; lod > 1; lod--)
				{
					std::size_t triangles = 0;
					for (const auto t : info.triangles[lod - 1])
					{
						triangles += t;
					}
					if (triangles)
					{
						return lod - 1;
					}
				}
				return 0u;
			}

			// the bytes a LOD's surfaces take in the zone (vertices with their light probe simplex records, and triangles)
			std::size_t lod_bytes(const model_info& info, const unsigned int lod)
			{
				std::size_t b = 0;
				for (auto s = 0u; s < info.vertices[lod].size(); s++)
				{
					b += info.vertices[lod][s] * (sizeof(zonetool::iw7::GfxPackedVertex) + sizeof(zonetool::iw7::SHProbeSimplexData1))
						+ info.triangles[lod][s] * sizeof(zonetool::iw7::Face);
				}
				return b;
			}

			struct proxy_budget
			{
				std::vector<std::array<std::size_t, 2>> opaque;
				std::vector<const Material*> alpha;
				std::vector<std::array<std::size_t, 2>> alpha_size;

				// adds a member's proxy_lod caster surfaces of one kind; false (nothing added) past IW7's surface or vertex limits
				bool add(const model_info& info, const caster_kind only)
				{
					const auto lod = proxy_lod(info);
					auto after = *this;
					const auto& mats = info.meshes[lod].second;
					for (auto s = 0u; s < mats.size(); s++)
					{
						const auto v = info.vertices[lod][s], t = info.triangles[lod][s];
						auto kind = mats[s] ? caster_kinds.at(mats[s]) : caster_kind::none;
						kind = kind == only ? kind : caster_kind::none;
						if (kind == caster_kind::opaque)
						{
							if (after.opaque.empty() || after.opaque.back()[0] + v > max_vertices || after.opaque.back()[1] + t > max_vertices)
							{
								after.opaque.push_back({ 0, 0 });
							}
							after.opaque.back()[0] += v;
							after.opaque.back()[1] += t;
						}
						else if (kind == caster_kind::alpha_tested)
						{
							auto at = std::ranges::find(after.alpha, mats[s]) - after.alpha.begin();
							if (at == static_cast<std::ptrdiff_t>(after.alpha.size()))
							{
								after.alpha.push_back(mats[s]);
								after.alpha_size.push_back({ 0, 0 });
							}
							after.alpha_size[at][0] += v;
							after.alpha_size[at][1] += t;
							if (after.alpha_size[at][0] > max_vertices || after.alpha_size[at][1] > max_vertices)
							{
								return false;
							}
						}
					}
					if (after.opaque.size() + after.alpha.size() > max_surfaces)
					{
						return false;
					}
					*this = std::move(after);
					return true;
				}
			};

			std::unordered_map<const XModel*, model_info> model_infos;

			// a cluster whose shadow is all its shadow proxies' (no cast of its own, so a LOD switch redraws no cached sun shadow,
			// 0x140DD0040) also draws its members at their last LODs from where every member reaches its last
			bool has_far_lod(const cluster& group)
			{
				if (group.posed || group.too_many_surfaces || current_plan.proxied.empty())
				{
					return false;
				}
				auto any = false;
				for (const auto i : group.members)
				{
					if (!current_plan.proxied[i])
					{
						return false;
					}
					const auto& info = model_infos[planned_world->dpvs.smodelDrawInsts[i].model];
					any = any || (info.eligible && info.lods >= 2);
				}
				return any;
			}


			using matrix34 = std::array<float, 12>;

			// a DObjAnimMat (quaternion, translation) as a row-major 3x4 matrix
			matrix34 base_matrix(const DObjAnimMat& m)
			{
				auto q = std::array<float, 4>{ m.quat[0], m.quat[1], m.quat[2], m.quat[3] };
				const auto len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
				for (auto& c : q)
				{
					c = len > 0.0f ? c / len : 0.0f;
				}
				const auto x = q[0], y = q[1], z = q[2], w = q[3];
				return { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), m.trans[0],
					2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), m.trans[1],
					2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), m.trans[2] };
			}

			matrix34 inverse_rigid(const matrix34& m)
			{
				matrix34 out{};
				for (auto r = 0; r < 3; r++)
				{
					for (auto c = 0; c < 3; c++)
					{
						out[r * 4 + c] = m[c * 4 + r];
					}
					out[r * 4 + 3] = -(out[r * 4 + 0] * m[3] + out[r * 4 + 1] * m[7] + out[r * 4 + 2] * m[11]);
				}
				return out;
			}

			matrix34 multiply(const matrix34& a, const matrix34& b)
			{
				matrix34 out{};
				for (auto r = 0; r < 3; r++)
				{
					for (auto c = 0; c < 4; c++)
					{
						out[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c] + a[r * 4 + 2] * b[2 * 4 + c]
							+ (c == 3 ? a[r * 4 + 3] : 0.0f);
					}
				}
				return out;
			}

			// The skinning matrices of a posed static model: BO3's GfxStaticModelBone m[3] as three rows of four (translation
			// in .w), each the bone's model space skinning matrix (its pose x the bind pose's inverse): bones BO3 leaves
			// unposed hold the identity whatever their bind pose, which only a skinning matrix gives; extra is 0.
			// `vertex_error` / `bone_error`: how far the posed first LOD's vertices / the posed bones, placed, lie from BO3's
			// world bounds of the instance. Empty when a vertex names a bone past the pose.
			std::vector<matrix34> posed_skin(const GfxStaticModelDrawInst& src, float& vertex_error, float& bone_error)
			{
				const auto* model = src.model;
				const auto bones = src.numPosedBones;
				vertex_error = bone_error = FLT_MAX;
				if (!bones || !src.posedBones || !model->baseMat || bones > static_cast<unsigned int>(model->numBones + model->numCosmeticBones))
				{
					return {};
				}
				std::vector<matrix34> skin(bones);
				for (auto b = 0u; b < bones; b++)
				{
					const auto* f = &src.posedBones[b].m[0][0];
					skin[b] = { f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11] };
				}
				const auto meshes = xmodel::lod_meshes(model);
				if (meshes.empty())
				{
					return {};
				}
				const auto place = [&](const float* p, float* mins, float* maxs)
				{
					for (auto c = 0; c < 3; c++)
					{
						const auto w = src.placement.origin[c] + src.placement.scale * (p[0] * src.placement.axis[0][c]
							+ p[1] * src.placement.axis[1][c] + p[2] * src.placement.axis[2][c]);
						mins[c] = std::min(mins[c], w);
						maxs[c] = std::max(maxs[c], w);
					}
				};
				const auto off = [&](const float* mins, const float* maxs)
				{
					auto e = 0.0f;
					for (auto c = 0; c < 3; c++)
					{
						e = std::max({ e, std::fabs(mins[c] - src.mins[c]), std::fabs(maxs[c] - src.maxs[c]) });
					}
					return e;
				};
				float mins[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, maxs[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
				auto valid = true;
				xmodel_mesh::for_each_vertex(meshes.front().first, [&](const float* xyz, const xmodel_mesh::vertex_bones& v)
				{
					float p[3] = { 0, 0, 0 };
					for (auto k = 0u; k < v.count; k++)
					{
						if (v.bone[k] >= bones)
						{
							valid = false;
							return;
						}
						const auto& m = skin[v.bone[k]];
						for (auto r = 0; r < 3; r++)
						{
							p[r] += v.weight[k] * (m[r * 4] * xyz[0] + m[r * 4 + 1] * xyz[1] + m[r * 4 + 2] * xyz[2] + m[r * 4 + 3]);
						}
					}
					place(p, mins, maxs);
				});
				if (!valid)
				{
					return {};
				}
				vertex_error = off(mins, maxs);
				// the posed bones: skinning x bind position
				float bmins[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, bmaxs[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
				for (auto b = 0u; b < bones; b++)
				{
					const auto bind = base_matrix(model->baseMat[b]);
					const auto posed = multiply(skin[b], bind);
					const float p[3] = { posed[3], posed[7], posed[11] };
					place(p, bmins, bmaxs);
				}
				bone_error = off(bmins, bmaxs);
				return skin;
			}
		}

		void make_plan(const GfxWorld* asset, const std::string& map_name)
		{
			clear();
			planned_world = asset;
			const auto count = asset->dpvs.smodelCount;
			auto& p = current_plan;
			p.cluster_of.assign(count, -1);
			p.iw7_index.assign(count, 0);

			// ---- which static models may merge ------------------------------------------------------------------------
			std::vector<char> eligible(count, 0);
			std::vector<char> large_member(count, 0);
			std::vector<std::pair<unsigned int, std::vector<matrix34>>> posed_models;
			auto posed = 0u, posed_unmatched = 0u, scripted = 0u, large = 0u, mirrored = 0u, other = 0u;
			std::vector<float> posed_vertex_error, posed_bone_error;
			for (auto i = 0u; i < count; i++)
			{
				const auto& src = asset->dpvs.smodelDrawInsts[i];
				auto [it, added] = model_infos.try_emplace(src.model);
				if (added)
				{
					it->second = inspect(src.model);
				}
				if (src.numPosedBones || src.posedBones)
				{
					posed++;
					float vertex_error = 0.0f, bone_error = 0.0f;
					auto skin = posed_skin(src, vertex_error, bone_error);
					if (skin.empty() || src.targetname || src.hidden || src.sanim)
					{
						posed_unmatched++;
						ZONETOOL_WARNING("static model %u (%s, %u posed bones): %s; drawn in its bind pose", i, src.model->name, src.numPosedBones,
							skin.empty() ? "its vertices name bones past its pose" : "a script names it");
					}
					else
					{
						posed_vertex_error.push_back(vertex_error);
						posed_bone_error.push_back(bone_error);
						posed_models.emplace_back(i, std::move(skin));
					}
				}
				else if (src.targetname || src.hidden || src.sanim)
				{
					scripted++;
				}
				else if (!it->second.eligible)
				{
					other++;
				}
				else if (src.model->radius * src.placement.scale > max_member_radius)
				{
					large++;
					large_member[i] = determinant(src.placement.axis) > 0.0f;
				}
				else if (determinant(src.placement.axis) <= 0.0f)
				{
					mirrored++;
				}
				else
				{
					eligible[i] = 1;
				}
			}

			const auto world_radius = std::getenv("ZT_PLAN_WORLD_RADIUS") ? static_cast<float>(std::atof(std::getenv("ZT_PLAN_WORLD_RADIUS")))
				: world_max_radius;
			const auto world_max_triangles = std::getenv("ZT_PLAN_WORLD_TRIS") ? static_cast<std::size_t>(std::atoll(std::getenv("ZT_PLAN_WORLD_TRIS")))
				: world_member_triangles;
			// ---- which mergeable ones the GfxWorld could draw: every material of every LOD on a world surface ------------
			// (the large and mirrored ones too: a world surface has no size limit, and a mirrored one's triangles are turned)
			std::vector<char> world_ok(count, 0);
			{
				std::unordered_map<const Material*, bool> drawable;
				for (auto i = 0u; i < count; i++)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					const auto rigid = !src.numPosedBones && !src.posedBones && !src.targetname && !src.hidden && !src.sanim
						&& model_infos[src.model].eligible;
					// the cheap ones (few triangles at their last LOD): the many materials; the rest keep their LODs and cull distances
					// as static models
					if (!rigid || src.model->radius * src.placement.scale > world_radius)
					{
						continue;
					}
					std::size_t last_triangles = 0;
					{
						const auto& info = model_infos[src.model];
						for (const auto n : info.triangles[info.lods - 1])
						{
							last_triangles += n;
						}
					}
					if (last_triangles > world_max_triangles)
					{
						continue;
					}
					auto ok = true;
					for (const auto& [mesh, mats] : model_infos[asset->dpvs.smodelDrawInsts[i].model].meshes)
					{
						for (const auto* material : mats)
						{
							auto [it, added] = drawable.try_emplace(material, false);
							if (added)
							{
								it->second = world_material::world_drawable(material);
							}
							ok = ok && it->second;
						}
					}
					world_ok[i] = ok;
				}
			}

			if (std::getenv("ZT_ATLAS_REPORT"))
			{
				std::vector<const Material*> materials;
				std::unordered_map<const Material*, world_material::surface_usage> used;
				std::unordered_set<const XModel*> seen_models;
				for (auto i = 0u; i < count; i++)
				{
					const auto* model = asset->dpvs.smodelDrawInsts[i].model;
					if (!model || !seen_models.insert(model).second)
					{
						continue;
					}
					for (const auto& [mesh, mats] : xmodel::lod_meshes(model))
					{
						const auto usage = xmodel_mesh::surface_usage(mesh);
						for (auto s = 0u; s < mats.size() && s < usage.size(); s++)
						{
							if (!mats[s])
							{
								continue;
							}
							auto [it, added] = used.try_emplace(mats[s]);
							if (added)
							{
								materials.push_back(mats[s]);
							}
							it->second.add(usage[s]);
						}
					}
				}
				for (const auto& [why, names] : world_material::atlas_report(materials, used))
				{
					std::string some;
					for (auto k = 0u; k < names.size() && k < 12; k++)
					{
						some += " " + names[k];
					}
					ZONETOOL_INFO("atlas report: %s: %zu materials;%s", why.data(), names.size(), some.data());
				}
				std::fflush(stdout);
				std::exit(0);
			}

			// ---- the sun's share at each mergeable one's centre (its own triangles not counted): IW7 lights a static model
			// past the sun shadow cascades with one sun visibility (lprobe vertex shaders pass the probe's coefficient 27
			// through), so a merged model of sunlit and shaded members would draw them all one way
			std::vector<float> member_sun(count, 1.0f);
			{
				const auto* com = comworld::converted();
				// the sun is the converted ComWorld's light 1 (comworld.cpp), the GfxWorld's lastSunPrimaryLightIndex
				const auto sun_lit = com && com->primaryLightCount > 1
					&& com->primaryLights[1].color[0] + com->primaryLights[1].color[1] + com->primaryLights[1].color[2] > 0.0f;
				if (sun_lit)
				{
					const world_lightmap::sun_blockers blockers(asset);
					parallel_for(count, [&](const std::uint32_t i, std::uint32_t)
					{
						if (eligible[i] || large_member[i] || world_ok[i])
						{
							member_sun[i] = world_lightmap::sun_fraction(blockers, com->primaryLights[1].dir, asset->dpvs.smodelDrawInsts[i].center, &i);
						}
					});
				}
				auto lit = 0u, merged = 0u;
				for (auto i = 0u; i < count; i++)
				{
					if (eligible[i] || large_member[i])
					{
						merged++;
						lit += member_sun[i] >= 0.5f;
					}
				}
				ZONETOOL_INFO("static model clusters: the sun reaches %u of %u mergeable static models (grouped apart from the shaded)", lit, merged);
				if (sun_lit && std::getenv("ZT_CLUSTER_PLAN_ONLY"))
				{
					// what shades them: per blocker (its model, or world surfaces) how many of the shaded it stops first
					const world_lightmap::sun_blockers blockers(asset);
					std::map<std::string, unsigned int> by_blocker;
					const auto* dir = com->primaryLights[1].dir;
					const auto len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
					for (auto i = 0u; i < count; i++)
					{
						if (!(eligible[i] || large_member[i]) || member_sun[i] >= 0.5f)
						{
							continue;
						}
						const auto* c = asset->dpvs.smodelDrawInsts[i].center;
						const double from[3] = { c[0] + dir[0] / len, c[1] + dir[1] / len, c[2] + dir[2] / len };
						const double to[3] = { from[0] + dir[0] / len * 131072.0, from[1] + dir[1] / len * 131072.0, from[2] + dir[2] / len * 131072.0 };
						const auto [kind, owner] = blockers.blocker(from, to, &i);
						std::string key = kind == 0 ? "nothing on the centre ray (a side ray)" : owner == ~0u ? "world surfaces"
								: (owner & 0x80000000u) ? utils::string::va("world surface material %s", asset->dpvs.surfaces[owner & 0x7FFFFFFFu].material
									&& asset->dpvs.surfaces[owner & 0x7FFFFFFFu].material->name ? asset->dpvs.surfaces[owner & 0x7FFFFFFFu].material->name : "?")
							: utils::string::va("%s (static model %u)", asset->dpvs.smodelDrawInsts[owner].model->name, owner);
						if (kind == 2)
						{
							key += " [alpha tested]";
						}
						by_blocker[key]++;
					}
					std::vector<std::pair<unsigned int, std::string>> sorted;
					for (const auto& [key, n] : by_blocker)
					{
						sorted.emplace_back(n, key);
					}
					std::ranges::sort(sorted, std::greater<>());
					for (auto k = 0u; k < sorted.size() && k < 40; k++)
					{
						ZONETOOL_INFO("static model clusters plan: sun blocked first by %s: %u", sorted[k].second.data(), sorted[k].first);
					}
					ZONETOOL_INFO("static model clusters plan: sun direction %g %g %g", dir[0], dir[1], dir[2]);
				}
			}

			// ---- the probe lighting at each one's centre -------------------------------------------------------------
			std::vector<float> luminance(count, 0.0f);
			{
				probe_lighting::evaluator lighting(asset, map::lighting_state());
				std::vector<std::vector<unsigned int>> by_volume(lighting.volume_count());
				for (auto i = 0u; i < count; i++)
				{
					if (eligible[i] || large_member[i])
					{
						by_volume[lighting.volume_at(asset->dpvs.smodelDrawInsts[i].center)].push_back(i);
					}
				}
				static constexpr float directions[6][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
				for (auto v = 0u; v < by_volume.size(); v++)
				{
					if (by_volume[v].empty())
					{
						continue;
					}
					lighting.load(v);
					parallel_for(static_cast<std::uint32_t>(by_volume[v].size()), [&](const std::uint32_t k, std::uint32_t)
					{
						const auto i = by_volume[v][k];
						float out[6][3];
						lighting.diffuse(v, asset->dpvs.smodelDrawInsts[i].center, directions, 6, out);
						auto sum = 0.0f;
						for (const auto& c : out)
						{
							sum += (0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]) / 6.0f;
						}
						luminance[i] = std::max(sum, 1e-6f);
					});
					lighting.unload(v);
				}
			}

			// ---- group by cell, shadow and lighting; fill clusters within IW7's surface and vertex limits ----------
			// and by the local lights reaching each member (sphere against its bounds): a merged model's light list is then
			// its members', which world surfaces share; merging members lit by different lights can push the light list
			// entries past IW7's 16-bit list offsets (world_lights.cpp)
			const auto* com = comworld::converted();
			const auto light_key_of = [&](const unsigned int i)
			{
				std::size_t h = 0;
				if (!com)
				{
					return h;
				}
				const auto& src = asset->dpvs.smodelDrawInsts[i];
				for (auto l = 0u; l < com->primaryLightCount; l++)
				{
					const auto& light = com->primaryLights[l];
					if ((light.type != zonetool::iw7::GFX_LIGHT_TYPE_SPOT && light.type != zonetool::iw7::GFX_LIGHT_TYPE_OMNI)
						|| !(light.radius > 0.0f))
					{
						continue;
					}
					auto d2 = 0.0f;
					for (auto k = 0; k < 3; k++)
					{
						const auto v = std::max({ src.mins[k] - light.origin[k], 0.0f, light.origin[k] - src.maxs[k] });
						d2 += v * v;
					}
					if (d2 <= light.radius * light.radius)
					{
						h = h * 1000003u + l + 1;
					}
				}
				return h;
			};
			std::vector<std::size_t> light_key(count, 0);
			parallel_for(count, [&](const std::uint32_t i, std::uint32_t)
			{
				if (eligible[i] || large_member[i])
				{
					light_key[i] = light_key_of(i);
				}
			});

			// the clusters of one grouping: members of a group fill clusters in order within IW7's surface and vertex limits
			const auto fill = [&](const grouping& g, std::vector<std::vector<unsigned int>>& filled, std::vector<bool>& filled_shadow,
				std::vector<bool>& filled_world, const bool world_mode)
			{
				std::map<std::tuple<bool, int, int, int, bool, int, std::size_t>, std::vector<unsigned int>> groups;
				for (auto i = 0u; i < count; i++)
				{
					if (!eligible[i] && !(g.large && large_member[i]) && !(world_mode && world_ok[i]))
					{
						continue;
					}
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					const auto step = g.steps_per_stop > 0.0f ? static_cast<int>(std::floor(std::log2(luminance[i]) * g.steps_per_stop)) : 0;
					// a world member: one group a world_cell, whatever its lighting (the lightmap lights each surface)
					if (world_mode && world_ok[i])
					{
						groups[std::make_tuple(true, static_cast<int>(std::floor(src.center[0] / world_cell)) * 2 + (member_sun[i] >= 0.5f ? 1 : 0),
							static_cast<int>(std::floor(src.center[1] / world_cell)), static_cast<int>(std::floor(src.center[2] / world_cell)),
							!(src.flags & t7_smodel_no_shadow), 0, std::size_t{ 0 })].push_back(i);
						continue;
					}
					const auto key = std::make_tuple(false, static_cast<int>(std::floor(src.center[0] / g.cell)) * 2 + (member_sun[i] >= 0.5f ? 1 : 0),
						static_cast<int>(std::floor(src.center[1] / g.cell)), static_cast<int>(std::floor(src.center[2] / g.cell)),
						!(src.flags & t7_smodel_no_shadow), step, g.by_lights ? light_key[i] : 0);
					groups[key].push_back(i);
				}
				for (auto& [key, members] : groups)
				{
					if (std::get<0>(key))
					{
						filled.push_back(members);
						filled_shadow.push_back(std::get<4>(key));
						filled_world.push_back(true);
						continue;
					}
					// members fill clusters by material set: a cluster of few materials draws few surfaces (a view's static
					// model surface list holds 8192 bytes, 0x140DCFF30), and IW7's 16 surfaces a LOD close a cluster of mixed
					// materials long before its vertices do
					{
						std::ranges::stable_sort(members, [&](const unsigned int a, const unsigned int b)
						{
							const auto& ma = model_infos[asset->dpvs.smodelDrawInsts[a].model].meshes;
							const auto& mb = model_infos[asset->dpvs.smodelDrawInsts[b].model].meshes;
							std::vector<const Material*> sa, sb;
							if (!ma.empty())
							{
								sa = ma.front().second;
							}
							if (!mb.empty())
							{
								sb = mb.front().second;
							}
							std::ranges::sort(sa);
							std::ranges::sort(sb);
							return sa < sb;
						});
					}
					std::vector<lod_budget> budget;
					std::vector<unsigned int> current;
					const auto close = [&]()
					{
						// a world group of one member is still drawn by the world (it is no static model)
						if (current.size() > 1 || (std::get<0>(key) && !current.empty()))
						{
							filled.push_back(current);
							filled_shadow.push_back(std::get<4>(key));
							filled_world.push_back(std::get<0>(key));
						}
						current.clear();
						budget.clear();
					};
					for (const auto i : members)
					{
						const auto& info = model_infos[asset->dpvs.smodelDrawInsts[i].model];
						const auto lods = std::max<std::size_t>(budget.size(), info.lods);
						const auto fits = [&]()
						{
							for (auto k = 0u; k < lods; k++)
							{
								const auto member_lod = std::min(k, info.lods - 1);
								const auto& at = k < budget.size() ? budget[k] : (budget.empty() ? lod_budget{} : budget.back());
								if (!at.fits(info, member_lod))
								{
									return false;
								}
							}
							return true;
						};
						if (!current.empty() && !fits())
						{
							close();
						}
						// a new cluster LOD starts as a copy of the last one (its members stay at their last LOD there)
						while (budget.size() < info.lods)
						{
							budget.push_back(budget.empty() ? lod_budget{} : budget.back());
						}
						for (auto k = 0u; k < budget.size(); k++)
						{
							budget[k].add(info, std::min(k, info.lods - 1));
						}
						current.push_back(i);
					}
					close();
				}
			};

			const auto cull_scale = std::getenv("ZT_PLAN_CULL_SCALE") ? static_cast<float>(std::atof(std::getenv("ZT_PLAN_CULL_SCALE"))) : 1.0f;
			// every model material's place in IW7's material order (world_material::model_draw_order), as a rank
			std::unordered_map<const Material*, unsigned int> rank_of;
			std::vector<char> region_0{ 1 }; // by rank: drawn in camera region 0, whose lists are the 8 KB ones
			{
				std::vector<std::pair<std::tuple<unsigned char, unsigned char, std::string, std::string>, const Material*>> all;
				std::unordered_set<const Material*> seen;
				for (auto i = 0u; i < count; i++)
				{
					for (const auto& [mesh, mats] : xmodel::lod_meshes(asset->dpvs.smodelDrawInsts[i].model))
					{
						for (const auto* material : mats)
						{
							if (seen.insert(material).second)
							{
								all.emplace_back(world_material::model_draw_order(material), material);
							}
						}
					}
				}
				std::ranges::sort(all);
				// an atlas's members write one material (world_material::atlas_groups): one rank, the first member's
				std::unordered_map<const Material*, world_material::surface_usage> used;
				std::unordered_set<const XModel*> seen_models;
				for (auto i = 0u; i < count; i++)
				{
					const auto* model = asset->dpvs.smodelDrawInsts[i].model;
					if (!model || !seen_models.insert(model).second)
					{
						continue;
					}
					for (const auto& [mesh, mats] : xmodel::lod_meshes(model))
					{
						const auto usage = xmodel_mesh::surface_usage(mesh);
						for (auto s = 0u; s < mats.size() && s < usage.size(); s++)
						{
							if (mats[s])
							{
								used[mats[s]].add(usage[s]);
							}
						}
					}
				}
				std::vector<const Material*> ordered;
				for (const auto& [key, material] : all)
				{
					ordered.push_back(material);
				}
				const auto groups = world_material::atlas_groups(ordered, used);
				std::unordered_map<unsigned int, unsigned int> group_rank;
				auto next_rank = 1u; // 0: the far LOD pages, first in IW7's order
				for (const auto& [key, material] : all)
				{
					const auto group = groups.find(material);
					if (group != groups.end())
					{
						const auto [it, added] = group_rank.try_emplace(group->second, next_rank);
						rank_of[material] = it->second;
						if (!added)
						{
							continue;
						}
					}
					else
					{
						rank_of[material] = next_rank;
					}
					region_0.push_back(std::get<0>(key) == 0);
					next_rank++;
				}
				ZONETOOL_INFO("static model clusters: %zu model materials, %zu in %zu atlases: %u materials as IW7 lists them", all.size(),
					groups.size(), group_rank.size(), next_rank - 1);
			}

			// the record bytes of the fullest view (0x140DCE4C0: (2 x instances + 13) & ~3 a record,
			// a record per model and LOD, at most 128 instances): a merged or posed model is one record of one instance; an
			// unmerged model's instances can sit at every LOD it has at once
			// far_radius: a cluster further than it draws its far LOD, one surface of a far page material (0: none)
			const std::vector<unsigned int> far_page{ 0u };
			const auto record_bytes = [&](const std::vector<std::vector<unsigned int>>& filled, const float far_radius,
				const std::vector<char>& in_world)
			{
				std::vector<char> merged = in_world;
				for (const auto& [i, skin] : posed_models)
				{
					merged[i] = 1;
				}
				for (const auto& members : filled)
				{
					for (const auto i : members)
					{
						merged[i] = 1;
					}
				}
				std::vector<unsigned int> singles;
				for (auto i = 0u; i < count; i++)
				{
					if (!merged[i])
					{
						singles.push_back(i);
					}
				}
				// a merged or posed model: one record of one instance (12 bytes) where its members' bounds are in view; its
				// surfaces' material ranks in order (dump_all sorts them)
				std::vector<std::array<float, 6>> boxes;
				std::vector<std::vector<unsigned int>> box_surfaces;
				const auto surfaces_of = [&](const std::vector<unsigned int>& members)
				{
					std::set<unsigned int> distinct;
					for (const auto i : members)
					{
						const auto& info = model_infos[asset->dpvs.smodelDrawInsts[i].model];
						if (!info.meshes.empty())
						{
							for (const auto* material : info.meshes.front().second)
							{
								if (region_0[rank_of.at(material)])
								{
									distinct.insert(rank_of.at(material));
								}
							}
						}
					}
					return std::vector<unsigned int>(distinct.begin(), distinct.end());
				};
				const auto box_of = [&](const std::vector<unsigned int>& members)
				{
					std::array<float, 6> b{ FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
					for (const auto i : members)
					{
						const auto& src = asset->dpvs.smodelDrawInsts[i];
						for (auto k = 0; k < 3; k++)
						{
							b[k] = std::min(b[k], src.mins[k]);
							b[k + 3] = std::max(b[k + 3], src.maxs[k]);
						}
					}
					return b;
				};
				// drawn until its farthest member's cull distance from its origin (dump_all), the members' centroid
				std::vector<std::array<float, 4>> box_cull;
				const auto cull_of = [&](const std::vector<unsigned int>& members)
				{
					std::array<float, 4> c{};
					for (const auto i : members)
					{
						for (auto k = 0; k < 3; k++)
						{
							c[k] += asset->dpvs.smodelDrawInsts[i].center[k] / static_cast<float>(members.size());
						}
					}
					for (const auto i : members)
					{
						const auto& src = asset->dpvs.smodelDrawInsts[i];
						const float d[3] = { src.center[0] - c[0], src.center[1] - c[1], src.center[2] - c[2] };
						const auto lods = src.model ? xmodel::lod_count(src.model) : 0u;
						const auto reach = lods ? xmodel::lod_dist(src.model, lods - 1) * src.placement.scale * cull_scale : FLT_MAX;
						c[3] = std::max(c[3], reach + std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
					}
					return c;
				};
				for (const auto& members : filled)
				{
					boxes.push_back(box_of(members));
					box_surfaces.push_back(surfaces_of(members));
					box_cull.push_back(cull_of(members));
				}
				for (const auto& [i, skin] : posed_models)
				{
					boxes.push_back(box_of({ i }));
					box_surfaces.push_back(surfaces_of({ i }));
					box_cull.push_back(cull_of({ i }));
				}
				// a view draws an unmerged instance only within its model's cull distance (the last LOD's): the worst of
				// the sample camera spots (every view_stride-th static model's centre) counts
				const auto view_stride = std::max<std::size_t>(1u, count / 1024u);
				std::unordered_map<const XModel*, std::pair<unsigned int, float>> lods_cull;
				std::unordered_map<const XModel*, std::vector<unsigned int>> model_surfaces;
				for (const auto i : singles)
				{
					const auto* model = asset->dpvs.smodelDrawInsts[i].model;
					if (!lods_cull.contains(model))
					{
						const auto lods = model ? xmodel::lod_count(model) : 0u;
						lods_cull[model] = { std::max(1u, lods), lods ? xmodel::lod_dist(model, lods - 1) * cull_scale : FLT_MAX };
						const auto meshes = xmodel::lod_meshes(model);
						auto& ranks = model_surfaces[model];
						if (!meshes.empty())
						{
							for (const auto* material : meshes.front().second)
							{
								if (region_0[rank_of.at(material)])
								{
									ranks.push_back(rank_of.at(material));
								}
							}
						}
					}
				}
				// a view only builds records for what is in its frustum: eight headings a spot, each 100 degrees wide (IW7's
				// 80 at 16:9, widened), a model counting when any of its bounds reaches into the heading's wedge
				constexpr auto headings = 8;
				constexpr auto half_fov = 50.0f * 3.14159265f / 180.0f;
				std::size_t worst = 0, worst_surfaces = 0, worst_delayed = 0;
				std::string worst_view;
				std::vector<std::unordered_map<const XModel*, unsigned int>> seen(headings);
				std::vector<std::size_t> merged_seen(headings);
				std::vector<std::vector<const std::vector<unsigned int>*>> merged_surfaces(headings);
				const auto in_wedge = [&](const float* eye, const float* mins, const float* maxs, const int h)
				{
					const auto ox = (mins[0] + maxs[0]) * 0.5f - eye[0], oy = (mins[1] + maxs[1]) * 0.5f - eye[1];
					const auto hx = (maxs[0] - mins[0]) * 0.5f, hy = (maxs[1] - mins[1]) * 0.5f, hz = (maxs[2] - mins[2]) * 0.5f;
					const auto radius = std::sqrt(hx * hx + hy * hy + hz * hz);
					const auto flat = std::sqrt(ox * ox + oy * oy);
					if (flat <= radius)
					{
						return true;
					}
					auto diff = std::fabs(std::atan2(oy, ox) - h * 2.0f * 3.14159265f / headings);
					diff = std::min(diff, 2.0f * 3.14159265f - diff);
					return diff <= half_fov + std::asin(radius / flat);
				};
				for (auto v = 0u; v < count; v += static_cast<unsigned int>(view_stride))
				{
					const auto& eye = asset->dpvs.smodelDrawInsts[v].center;
					for (auto& s : seen)
					{
						s.clear();
					}
					std::fill(merged_seen.begin(), merged_seen.end(), 0u);
					for (auto& m : merged_surfaces)
					{
						m.clear();
					}
					for (auto b = 0u; b < boxes.size(); b++)
					{
						const auto& c = box_cull[b];
						const auto ox = c[0] - eye[0], oy = c[1] - eye[1], oz = c[2] - eye[2];
						if (ox * ox + oy * oy + oz * oz >= c[3] * c[3])
						{
							continue;
						}
						for (auto h = 0; h < headings; h++)
						{
							if (in_wedge(eye, boxes[b].data(), boxes[b].data() + 3, h))
							{
								merged_seen[h]++;
								const auto beyond = far_radius > 0.0f && ox * ox + oy * oy + oz * oz >= far_radius * far_radius;
								merged_surfaces[h].push_back(beyond ? &far_page : &box_surfaces[b]);
							}
						}
					}
					for (const auto i : singles)
					{
						const auto& src = asset->dpvs.smodelDrawInsts[i];
						const auto cull = lods_cull[src.model].second;
						const auto ex = src.center[0] - eye[0], ey = src.center[1] - eye[1], ez = src.center[2] - eye[2];
						const auto dist2 = ex * ex + ey * ey + ez * ez;
						if (cull > 0.0f && dist2 >= cull * cull)
						{
							continue;
						}
						const auto hx = (src.maxs[0] - src.mins[0]) * 0.5f, hy = (src.maxs[1] - src.mins[1]) * 0.5f,
							hz = (src.maxs[2] - src.mins[2]) * 0.5f;
						const auto radius = std::sqrt(hx * hx + hy * hy + hz * hz);
						const auto flat = std::sqrt(ex * ex + ey * ey);
						const auto yaw = std::atan2(ey, ex);
						const auto spread = flat <= radius ? 3.14159265f : std::asin(radius / flat);
						for (auto h = 0; h < headings; h++)
						{
							auto diff = std::fabs(yaw - h * 2.0f * 3.14159265f / headings);
							diff = std::min(diff, 2.0f * 3.14159265f - diff);
							if (diff <= half_fov + spread)
							{
								seen[h][src.model]++;
							}
						}
					}
					for (auto h = 0; h < headings; h++)
					{
						const auto& s = seen[h];
						std::size_t bytes = merged_seen[h] * 12u;
						// each record's surfaces, the models in their first material's order (make_plan orders them so)
						std::vector<std::pair<const std::vector<unsigned int>*, unsigned int>> drawn;
						for (const auto* ranks : merged_surfaces[h])
						{
							drawn.emplace_back(ranks, 1u);
						}
						for (const auto& [model, n] : s)
						{
							const auto lods = lods_cull[model].first;
							const auto records = std::max(std::min(n, 2u * lods), (n + 127u) / 128u);
							const auto base = n / records, extra = n % records;
							bytes += extra * ((2u * (base + 1u) + 13u) & ~3u) + (records - extra) * ((2u * base + 13u) & ~3u);
							drawn.emplace_back(&model_surfaces[model], records);
						}
						std::ranges::stable_sort(drawn, [](const auto& a, const auto& b)
						{
							const auto ka = a.first->empty() ? 0u : a.first->front(), kb = b.first->empty() ? 0u : b.first->front();
							return ka < kb;
						});
						// a surface of the model's first material goes straight to the list, the others wait until a model's first
						// material reaches theirs; the list takes 2 bytes a surface and a header a material run of up to 16
						std::priority_queue<unsigned int, std::vector<unsigned int>, std::greater<>> waiting;
						std::size_t entries = 0, runs = 0, peak = 0;
						std::unordered_map<unsigned int, std::size_t> per_material;
						for (const auto& [ranks, records] : drawn)
						{
							if (ranks->empty())
							{
								continue;
							}
							const auto first = ranks->front();
							while (!waiting.empty() && waiting.top() <= first)
							{
								waiting.pop();
							}
							for (auto r = 0u; r < records; r++)
							{
								for (const auto rank : *ranks)
								{
									entries++;
									per_material[rank]++;
									if (rank != first)
									{
										waiting.push(rank);
									}
								}
							}
							peak = std::max(peak, waiting.size());
						}
						for (const auto& [rank, n] : per_material)
						{
							runs += (n + 15u) / 16u;
						}
						worst = std::max(worst, bytes);
						if (entries * 2u + runs * 10u > worst_surfaces)
						{
							std::size_t merged_entries = 0;
							for (const auto* ranks : merged_surfaces[h])
							{
								merged_entries += ranks->size();
							}
							worst_view = utils::string::va("worst view at static model %u heading %d: %zu surfaces (%zu of merged models, %zu of %zu "
								"unmerged models' records), %zu materials", v, h, entries, merged_entries, entries - merged_entries, s.size(),
								per_material.size());
						}
						worst_surfaces = std::max(worst_surfaces, entries * 2u + runs * 10u);
						worst_delayed = std::max(worst_delayed, peak);
					}
				}
				if (std::getenv("ZT_CLUSTER_PLAN_ONLY"))
				{
					ZONETOOL_INFO("static model clusters plan: %s", worst_view.data());
				}
				return std::make_tuple(worst, worst_surfaces, worst_delayed);
			};

			std::vector<std::vector<unsigned int>> filled, world_filled;
			std::vector<bool> filled_shadow, world_shadow;
			const grouping* chosen = nullptr;
			auto chosen_world = false;
			std::size_t bytes = 0, surface_bytes = 0, delayed = 0;
			std::string tried;
			const auto any_world = !std::getenv("ZT_PLAN_NO_WORLD") && std::ranges::any_of(world_ok, [](const char ok) { return ok != 0; });
			for (const auto world_mode : { false, true })
			{
				if (chosen || (world_mode && !any_world))
				{
					break;
				}
				for (const auto& g : groupings)
				{
					std::vector<std::vector<unsigned int>> all;
					std::vector<bool> all_shadow, all_world;
					fill(g, all, all_shadow, all_world, world_mode);
					filled.clear();
					filled_shadow.clear();
					world_filled.clear();
					world_shadow.clear();
					std::vector<char> in_world(count, 0);
					for (auto c = 0u; c < all.size(); c++)
					{
						if (all_world[c])
						{
							for (const auto i : all[c])
							{
								in_world[i] = 1;
							}
							world_filled.push_back(std::move(all[c]));
							world_shadow.push_back(all_shadow[c]);
						}
						else
						{
							filled.push_back(std::move(all[c]));
							filled_shadow.push_back(all_shadow[c]);
						}
					}
					std::tie(bytes, surface_bytes, delayed) = record_bytes(filled, 0.0f, in_world);
					tried += utils::string::va(" [%s%.0f units, %g steps a stop, %s: %zu merged, %zu world groups, %zu record bytes, %zu surface "
						"list bytes, %zu delayed]", world_mode ? "world, " : "", g.cell, g.steps_per_stop, g.by_lights ? "by light set" : "any light set",
						filled.size(), world_filled.size(), bytes, surface_bytes, delayed);
					if (bytes <= record_budget && surface_bytes <= surface_budget && delayed <= delayed_budget)
					{
						chosen = &g;
						chosen_world = world_mode;
						break;
					}
				}
			}
			if (std::getenv("ZT_CLUSTER_PLAN_ONLY") && chosen_world == false)
			{
				// why the static models the world does not draw stay static models: per reason, models and their materials
				std::map<std::string, std::pair<unsigned int, std::set<std::string>>> reasons;
				for (auto i = 0u; i < count; i++)
				{
					if (world_ok[i])
					{
						continue;
					}
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					std::string why;
					if (src.numPosedBones || src.posedBones)
					{
						why = "posed";
					}
					else if (src.targetname || src.hidden || src.sanim)
					{
						why = "scripted";
					}
					else if (!model_infos[src.model].eligible)
					{
						why = "skinned / no vertex data / over 16 surfaces";
					}
					auto& r = reasons[why.empty() ? "materials" : why];
					r.first++;
					if (why.empty())
					{
						for (const auto& [mesh, mats] : model_infos[src.model].meshes)
						{
							for (const auto* material : mats)
							{
								if (material && !world_material::world_drawable(material))
								{
									r.second.insert(utils::string::va("%s (%s, region %u)", material->name,
										world_material::bake::bo3_template(material).data(), static_cast<unsigned int>(material->cameraRegion)));
								}
							}
						}
					}
				}
				for (const auto& [why, r] : reasons)
				{
					std::string names;
					for (const auto& n : r.second)
					{
						names += " " + n;
					}
					ZONETOOL_INFO("static model clusters plan: not world drawable, %s: %u static models;%s", why.data(), r.first, names.data());
				}
			}
			if (std::getenv("ZT_CLUSTER_PLAN_ONLY"))
			{
				std::size_t world_triangles = 0, world_members = 0;
				for (const auto& members : world_filled)
				{
					for (const auto i : members)
					{
						const auto& info = model_infos[asset->dpvs.smodelDrawInsts[i].model];
						for (const auto t : info.triangles[info.lods - 1])
						{
							world_triangles += t;
						}
						world_members++;
					}
				}
				ZONETOOL_INFO("static model clusters plan: world groups %zu members, %zu triangles at their last LODs", world_members,
					world_triangles);
				ZONETOOL_INFO("static model clusters plan: records and surfaces %s, %zu of %zu record bytes, %zu of %zu surface list bytes, %zu of %zu delayed; tried%s",
					chosen ? "PASS" : "FAIL", bytes, record_budget, surface_bytes, surface_budget, delayed, delayed_budget, tried.data());
				if (!chosen)
				{
					std::fflush(stdout);
					std::exit(3);
				}
			}
			if (!posed_bone_error.empty())
			{
				const auto median = [](std::vector<float> v)
				{
					std::ranges::sort(v);
					return v[v.size() / 2];
				};
				ZONETOOL_INFO("static model clusters: %zu posed static models against BO3's bounds: posed bones off by median %g max %g "
					"units, posed vertices by median %g max %g", posed_bone_error.size(), median(posed_bone_error),
					*std::ranges::max_element(posed_bone_error), median(posed_vertex_error), *std::ranges::max_element(posed_vertex_error));
			}
			if (!chosen)
			{
				ZONETOOL_FATAL("static model clusters: no grouping keeps the fullest view within %zu record bytes and the surface list budget (IW7 "
					"addresses 16384):%s", record_budget, tried.data());
			}
			std::vector<std::vector<matrix34>> filled_skin(filled.size());

			for (auto& [i, skin] : posed_models)
			{
				filled.push_back({ i });
				filled_shadow.push_back(!(asset->dpvs.smodelDrawInsts[i].flags & t7_smodel_no_shadow));
				filled_skin.push_back(std::move(skin));
			}

			// ---- the world groups: no IW7 static model, the GfxWorld draws them ------------------------------------------
			p.world_of.assign(count, -1);
			p.world_clusters.resize(world_filled.size());
			for (auto w = 0u; w < world_filled.size(); w++)
			{
				auto& group = p.world_clusters[w];
				group.name = utils::string::va("t7_smw_%s_%u", map_name.data(), w);
				group.members = std::move(world_filled[w]);
				group.casts_shadow = world_shadow[w];
				for (auto k = 0; k < 3; k++)
				{
					group.mins[k] = FLT_MAX;
					group.maxs[k] = -FLT_MAX;
				}
				for (const auto i : group.members)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					p.world_of[i] = static_cast<int>(w);
					p.iw7_index[i] = ~0u;
					for (auto k = 0; k < 3; k++)
					{
						group.origin[k] += src.center[k] / static_cast<float>(group.members.size());
						group.mins[k] = std::min(group.mins[k], src.mins[k]);
						group.maxs[k] = std::max(group.maxs[k], src.maxs[k]);
					}
					group.member_bounds.push_back({ src.mins[0], src.mins[1], src.mins[2], src.maxs[0], src.maxs[1], src.maxs[2] });
				}
			}
			if (chosen_world)
			{
				auto members = 0u;
				for (const auto& group : p.world_clusters)
				{
					members += static_cast<unsigned int>(group.members.size());
				}
				ZONETOOL_INFO("static model clusters: no grouping keeps IW7's static model lists within their limits as static models alone; "
					"%u static models drawn by the GfxWorld as %zu groups of world surfaces", members, p.world_clusters.size());
			}

			// ---- the IW7 static model list: the unmerged ones in BO3's order, then the clusters ----------------------
			for (auto c = 0u; c < filled.size(); c++)
			{
				for (const auto i : filled[c])
				{
					p.cluster_of[i] = static_cast<int>(c);
				}
			}
			auto next = 0u;
			for (auto i = 0u; i < count; i++)
			{
				if (p.cluster_of[i] < 0 && p.world_of[i] < 0)
				{
					p.iw7_index[i] = next++;
					models_left_single.insert(asset->dpvs.smodelDrawInsts[i].model);
				}
			}
			auto merged = 0u, clamped = 0u;
			p.clusters.resize(filled.size());
			for (auto c = 0u; c < filled.size(); c++)
			{
				auto& group = p.clusters[c];
				group.name = utils::string::va("t7_smc_%s_%u", map_name.data(), c);
				group.members = filled[c];
				group.casts_shadow = filled_shadow[c];
				group.posed = !filled_skin[c].empty();
				group.skin = std::move(filled_skin[c]);
				if (group.posed)
				{
					group.name = utils::string::va("t7_posed_%s_%u", map_name.data(), c);
					// one surface per material a LOD: more than 16 cannot be a static model
					for (const auto& [mesh, mats] : model_infos[asset->dpvs.smodelDrawInsts[group.members.front()].model].meshes)
					{
						std::unordered_set<const Material*> distinct(mats.begin(), mats.end());
						group.too_many_surfaces = group.too_many_surfaces || distinct.size() > max_surfaces;
					}
				}
				for (auto k = 0; k < 3; k++)
				{
					group.mins[k] = FLT_MAX;
					group.maxs[k] = -FLT_MAX;
				}
				// lit as its brightest member (the light grid samples at the lighting origin); every other member keeps its
				// own brightness as its vertices' self visibility, which scales the probe lighting (lprobe _sv_ techniques)
				auto brightest = group.members.front();
				for (const auto i : group.members)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					for (auto k = 0; k < 3; k++)
					{
						group.origin[k] += src.center[k] / static_cast<float>(group.members.size());
						group.mins[k] = std::min(group.mins[k], src.mins[k]);
						group.maxs[k] = std::max(group.maxs[k], src.maxs[k]);
					}
					group.member_bounds.push_back({ src.mins[0], src.mins[1], src.mins[2], src.maxs[0], src.maxs[1], src.maxs[2] });
					p.iw7_index[i] = next + c;
					if (luminance[i] > luminance[brightest])
					{
						brightest = i;
					}
				}
				std::memcpy(group.lighting_origin, asset->dpvs.smodelDrawInsts[brightest].center, sizeof(group.lighting_origin));
				if (!group.posed)
				{
					group.sun = 0.0f;
					for (const auto i : group.members)
					{
						group.sun += member_sun[i] / static_cast<float>(group.members.size());
					}
				}
				for (const auto i : group.members)
				{
					const auto v = group.posed ? 1.0f : luminance[i] / luminance[brightest];
					clamped += v < min_visibility;
					group.visibility.push_back(std::clamp(v, min_visibility, 1.0f));
				}
				merged += static_cast<unsigned int>(group.members.size());
			}
			p.iw7_count = next + static_cast<unsigned int>(filled.size());
			if (p.iw7_count > 0xFFFF)
			{
				ZONETOOL_FATAL("static model clusters: %u IW7 static models do not fit IW7's 16-bit static model indices", p.iw7_count);
			}

			// IW7 draws static models in index order and merges their surfaces by material sort index; a surface out of that
			// order waits in a 256 entry list (0x140DCE680, warning 25), so the static models go in their first material's order
			{
				using order_key = std::tuple<unsigned char, unsigned char, std::string, std::string>;
				std::unordered_map<const Material*, order_key> order_of;
				const auto order = [&](const Material* material) -> const order_key&
				{
					auto found = order_of.find(material);
					if (found == order_of.end())
					{
						found = order_of.emplace(material, world_material::model_draw_order(material)).first;
					}
					return found->second;
				};
				struct entry
				{
					order_key key;
					const XModel* model = nullptr;
					unsigned int index = 0;
				};
				std::vector<entry> entries(p.iw7_count);
				for (auto i = 0u; i < count; i++)
				{
					if (p.cluster_of[i] >= 0 || p.world_of[i] >= 0)
					{
						continue;
					}
					auto& e = entries[p.iw7_index[i]];
					e.model = asset->dpvs.smodelDrawInsts[i].model;
					e.index = p.iw7_index[i];
					const auto meshes = xmodel::lod_meshes(e.model);
					if (!meshes.empty() && !meshes.front().second.empty())
					{
						e.key = order(meshes.front().second.front());
					}
				}
				for (auto c = 0u; c < filled.size(); c++)
				{
					auto& e = entries[next + c];
					e.index = next + c;
					auto first = true;
					for (const auto i : filled[c])
					{
						const auto& meshes = model_infos[asset->dpvs.smodelDrawInsts[i].model].meshes;
						for (const auto* material : meshes.empty() ? std::vector<const Material*>{} : meshes.front().second)
						{
							if (material && (first || order(material) < e.key))
							{
								e.key = order(material);
								first = false;
							}
						}
					}
				}
				std::ranges::stable_sort(entries, [](const entry& a, const entry& b)
				{
					return std::tie(a.key, a.model, a.index) < std::tie(b.key, b.model, b.index);
				});
				std::vector<unsigned int> moved_to(p.iw7_count);
				for (auto k = 0u; k < entries.size(); k++)
				{
					moved_to[entries[k].index] = k;
				}
				for (auto& index : p.iw7_index)
				{
					if (index != ~0u)
					{
						index = moved_to[index];
					}
				}
			}
			ZONETOOL_INFO("static model clusters: %u of %u static models merged into %zu models (%.0f units cells, lighting within %g "
				"stops, %s; %zu record bytes in the fullest view, of %zu; tried%s), %zu posed ones with their pose baked in (%u "
				"of %u posed kept their bind pose); single: %u (%u scripted, %u larger than %.0f units, %u mirrored, %u of models that "
				"cannot merge (skinned, over %zu surfaces, no vertex data, brush models), the rest alone in their group); %u IW7 static "
				"models; %u members more than 7 stops darker than their cluster's lighting", merged - static_cast<unsigned int>(posed_models.size()),
				count, filled.size() - posed_models.size(), chosen->cell, chosen->steps_per_stop > 0.0f ? 1.0f / chosen->steps_per_stop : 0.0f,
				chosen->by_lights ? "by light set" : "any light set", bytes, record_budget, tried.data(), posed_models.size(), posed_unmatched,
				posed, next, scripted, large, max_member_radius, mirrored, other, max_surfaces, p.iw7_count, clamped);
		}

		void plan_shadow_proxies(const GfxWorld* asset, const std::string& map_name)
		{
			auto& p = current_plan;
			const auto count = asset->dpvs.smodelCount;
			p.proxies.clear();
			p.proxied.assign(count, 0);
			caster_kinds.clear();

			// the static models whose shadow a proxy draws: rigid ones a script never touches (hidden, moved or named) that
			// cast (a mirrored one's triangles are turned back, dump_all); the rest cast their own
			std::vector<unsigned int> candidates;
			std::unordered_set<const Material*> materials;
			auto own_shadow = 0u, own_posed = 0u, own_scripted = 0u, own_ineligible = 0u;
			for (auto i = 0u; i < count; i++)
			{
				const auto& src = asset->dpvs.smodelDrawInsts[i];
				if (src.flags & t7_smodel_no_shadow)
				{
					continue;
				}
				const auto& info = model_infos[src.model];
				if (p.world_of[i] >= 0)
				{
					continue; // its world surfaces cast
				}
				const auto posed = src.numPosedBones || src.posedBones;
				const auto scripted = src.targetname || src.hidden || src.sanim;
				if (posed || scripted || !info.eligible)
				{
					own_shadow++;
					own_posed += posed ? 1 : 0;
					own_scripted += !posed && scripted ? 1 : 0;
					own_ineligible += !posed && !scripted ? 1 : 0;
					continue;
				}
				candidates.push_back(i);
				for (const auto& [mesh, mats] : info.meshes)
				{
					for (const auto* material : mats)
					{
						if (material)
						{
							materials.insert(material);
						}
					}
				}
			}

			// each material's caster kind: alpha tested as the lightmap's sun blockers take it (a one-channel mask its gbuffer
			// program discards by, or a BO3 alpha test template without one); a proxy draws an alpha tested one with a copy of
			// its own material, which is right whatever it is, the shared opaque one only where nothing is discarded
			for (const auto* material : materials)
			{
				auto kind = caster_kind::none;
				if (world_material::model_blocks_sun(material))
				{
					const auto name = world_material::bake::bo3_template(material);
					const auto alpha = world_material::bake::alpha_mask_of(material).has_value() || name.find("alpha") != std::string::npos
						|| name.find("atest") != std::string::npos || name.find("foliage") != std::string::npos;
					kind = alpha ? caster_kind::alpha_tested : caster_kind::opaque;
				}
				caster_kinds.emplace(material, kind);
			}

			// along a Morton curve over the cells, then in BO3's order
			const auto morton = [](const float* center)
			{
				std::uint64_t code = 0;
				for (auto k = 0; k < 3; k++)
				{
					const auto cell = static_cast<std::uint64_t>(std::clamp(std::floor(center[k] / proxy_cell) + 1024.0f, 0.0f, 2047.0f));
					for (auto bit = 0; bit < 11; bit++)
					{
						code |= ((cell >> bit) & 1) << (bit * 3 + k);
					}
				}
				return code;
			};
			// opaque proxies along the curve; alpha tested ones by material first (one surface a material wherever it is), then
			// the same
			const auto first_alpha = [&](const unsigned int i)
			{
				std::string name;
				const auto& info = model_infos[asset->dpvs.smodelDrawInsts[i].model];
				for (const auto* m : info.meshes[proxy_lod(info)].second)
				{
					if (m && caster_kinds.at(m) == caster_kind::alpha_tested && (name.empty() || name > m->name))
					{
						name = m->name;
					}
				}
				return name;
			};
			std::vector<std::vector<unsigned int>> filled;
			std::vector<bool> filled_alpha;
			std::size_t opaque_surfaces = 0, alpha_surfaces = 0, opaque_triangles = 0, alpha_triangles = 0;
			for (const auto only : { caster_kind::opaque, caster_kind::alpha_tested })
			{
				std::vector<std::tuple<std::string, std::uint64_t, unsigned int>> order;
				for (const auto i : candidates)
				{
					order.emplace_back(only == caster_kind::alpha_tested ? first_alpha(i) : std::string(),
						morton(asset->dpvs.smodelDrawInsts[i].center), i);
				}
				std::ranges::sort(order);

				std::vector<unsigned int> current_members;
				proxy_budget budget;
				const auto close = [&]()
				{
					if (!current_members.empty())
					{
						filled.push_back(std::move(current_members));
						filled_alpha.push_back(only == caster_kind::alpha_tested);
						opaque_surfaces += budget.opaque.size();
						alpha_surfaces += budget.alpha.size();
					}
					current_members.clear();
					budget = {};
				};
				for (const auto& [name, code, i] : order)
				{
					const auto& info = model_infos[asset->dpvs.smodelDrawInsts[i].model];
					p.proxied[i] = 1;
					const auto lod = proxy_lod(info);
					const auto& mats = info.meshes[lod].second;
					auto casts = false;
					for (auto s = 0u; s < mats.size(); s++)
					{
						const auto kind = mats[s] ? caster_kinds.at(mats[s]) : caster_kind::none;
						if (kind == only)
						{
							casts = true;
							(kind == caster_kind::opaque ? opaque_triangles : alpha_triangles) += info.triangles[lod][s];
						}
					}
					if (!casts)
					{
						continue; // nothing of it casts this way
					}
					if (!budget.add(info, only))
					{
						close();
						if (!budget.add(info, only))
						{
							ZONETOOL_FATAL("shadow proxies: static model %u (%s) does not fit a proxy alone", i, asset->dpvs.smodelDrawInsts[i].model->name);
						}
					}
					current_members.push_back(i);
				}
				close();
			}

			p.proxy_begin = p.iw7_count;
			p.proxies.resize(filled.size());
			for (auto k = 0u; k < filled.size(); k++)
			{
				auto& proxy = p.proxies[k];
				proxy.name = utils::string::va("t7_smp_%s_%u", map_name.data(), k);
				proxy.members = std::move(filled[k]);
				proxy.alpha = filled_alpha[k];
				for (auto c = 0; c < 3; c++)
				{
					proxy.mins[c] = FLT_MAX;
					proxy.maxs[c] = -FLT_MAX;
				}
				for (const auto i : proxy.members)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					for (auto c = 0; c < 3; c++)
					{
						proxy.origin[c] += src.center[c] / static_cast<float>(proxy.members.size());
						proxy.mins[c] = std::min(proxy.mins[c], src.mins[c]);
						proxy.maxs[c] = std::max(proxy.maxs[c], src.maxs[c]);
					}
				}
				// cast as far as its farthest member draws (its own last LOD's distance, x its scale, plus how far it sits
				// from the origin), as a cluster
				for (const auto i : proxy.members)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					const float d[3] = { src.center[0] - proxy.origin[0], src.center[1] - proxy.origin[1], src.center[2] - proxy.origin[2] };
					const auto offset = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
					proxy.dist = std::max(proxy.dist, xmodel::lod_dist(src.model, model_infos[src.model].lods - 1) * src.placement.scale + offset);
				}
			}
			p.iw7_count += static_cast<unsigned int>(p.proxies.size());
			if (p.iw7_count > 0xFFFF)
			{
				ZONETOOL_FATAL("shadow proxies: %u IW7 static models do not fit IW7's 16-bit static model indices", p.iw7_count);
			}

			// the entries the models casting their own shadow add: IW7 lists one per surface of each run of one model in its
			// static model order (the unmerged ones keep BO3's), a bound with every one of them in a cascade
			std::size_t own_surfaces = 0;
			const XModel* last = nullptr;
			for (auto i = 0u; i < count; i++)
			{
				const auto& src = asset->dpvs.smodelDrawInsts[i];
				if ((src.flags & t7_smodel_no_shadow) || p.proxied[i] || p.world_of[i] >= 0)
				{
					continue;
				}
				if (src.model != last || p.cluster_of[i] >= 0)
				{
					const auto meshes = xmodel::lod_meshes(src.model);
					own_surfaces += meshes.empty() || !meshes.front().first ? 0u : meshes.front().first->numSurfs;
				}
				last = p.cluster_of[i] >= 0 ? nullptr : src.model;
			}
			ZONETOOL_INFO("shadow proxies: %zu static models' shadows drawn by %zu proxies (%zu opaque surfaces of %zu triangles, %zu alpha "
				"tested surfaces of %zu triangles); %u cast their own (%u posed, %u scripted, %u skinned or without vertex data: at most "
				"%zu entries); at most %zu sun shadow entries with every caster in a cascade (a cascade takes those its column reaches); "
				"%zu bytes of camera records (12 a proxy)", candidates.size(),
				p.proxies.size(), opaque_surfaces, opaque_triangles, alpha_surfaces, alpha_triangles, own_shadow, own_posed, own_scripted,
				own_ineligible, own_surfaces, opaque_surfaces + alpha_surfaces + own_surfaces, p.proxies.size() * 12);

			// ---- the LOD each cluster member copies: IW7 reads a fastfile through signed 32-bit offsets (0x140B7F6D0), so the
			// merged and proxy geometry stays within geometry_budget. Every member starts at its second LOD; models go back to
			// their first while it fits, those losing the most of their first LOD a byte first
			std::size_t proxy_bytes = 0;
			for (const auto& proxy : p.proxies)
			{
				const auto want = proxy.alpha ? caster_kind::alpha_tested : caster_kind::opaque;
				for (const auto i : proxy.members)
				{
					const auto& info = model_infos[asset->dpvs.smodelDrawInsts[i].model];
					const auto lod = proxy_lod(info);
					const auto& mats = info.meshes[lod].second;
					for (auto s = 0u; s < mats.size(); s++)
					{
						if (mats[s] && caster_kinds.at(mats[s]) == want)
						{
							proxy_bytes += info.vertices[lod][s] * sizeof(zonetool::iw7::GfxPackedVertex) + info.triangles[lod][s] * sizeof(zonetool::iw7::Face);
						}
					}
				}
			}
			p.member_lod.assign(count, 0);
			struct upgrade
			{
				const XModel* model;
				std::size_t extra;
				double priority;
			};
			std::unordered_map<const XModel*, std::pair<unsigned int, std::size_t>> extra_of;
			std::size_t cluster_bytes = 0, full_bytes = 0;
			// a world group's vertices are GfxWorldVertex (44 bytes, not 32), and its lightmap charts split some: 7/4 the bytes
			const auto weighed = [](const std::size_t b, const bool world)
			{
				return world ? b * 7u / 4u : b;
			};
			for (const auto* groups : { &p.clusters, &p.world_clusters })
			for (const auto& group : *groups)
			{
				const auto world = groups == &p.world_clusters;
				for (const auto i : group.members)
				{
					const auto* model = asset->dpvs.smodelDrawInsts[i].model;
					const auto& info = model_infos[model];
					std::size_t b0 = 0;
					if (info.eligible)
					{
						b0 = lod_bytes(info, 0);
					}
					else if (const auto meshes = xmodel::lod_meshes(model); !meshes.empty() && meshes.front().first)
					{
						const auto* mesh = meshes.front().first;
						for (auto k = 0; k < mesh->numSurfs; k++)
						{
							b0 += mesh->surfs[k].vertCount * sizeof(zonetool::iw7::GfxPackedVertex) + mesh->surfs[k].triCount * sizeof(zonetool::iw7::Face);
						}
					}
					b0 = weighed(b0, world);
					full_bytes += b0;
					if (group.posed || !info.eligible || info.lods < 2)
					{
						cluster_bytes += b0;
						continue;
					}
					p.member_lod[i] = 1;
					const auto b1 = weighed(lod_bytes(info, 1), world);
					cluster_bytes += b1;
					auto& [n, extra] = extra_of[model];
					n++;
					extra += b0 - std::min(b0, b1);
					// at its first LOD a two LOD member of a cluster with a far LOD is also in the far LOD at its second
					if (!world && info.lods == 2 && has_far_lod(group))
					{
						extra += b1;
					}
				}
			}
			// the far LODs (has_far_lod): every member's last LOD once more
			std::size_t far_bytes = 0;
			for (const auto& group : p.clusters)
			{
				if (!has_far_lod(group))
				{
					continue;
				}
				for (const auto i : group.members)
				{
					// (a member at its last LOD already draws the same in both; a two LOD one upgraded to its first is counted by
					// its upgrade)
					const auto& info = model_infos[asset->dpvs.smodelDrawInsts[i].model];
					if (info.eligible && info.lods >= 3)
					{
						far_bytes += lod_bytes(info, info.lods - 1);
					}
				}
			}
			cluster_bytes += far_bytes;
			// the single static models' light probe simplex records (light_probes): every LOD's vertices of each rigid model
			{
				std::unordered_set<const XModel*> singles;
				std::size_t simplex_bytes = 0;
				for (auto i = 0u; i < asset->dpvs.smodelCount; i++)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					if (p.cluster_of[i] >= 0 || src.numPosedBones || src.posedBones || !src.model || !singles.insert(src.model).second)
					{
						continue;
					}
					for (const auto& [mesh, mats] : xmodel::lod_meshes(src.model))
					{
						for (auto s = 0; mesh && s < mesh->numSurfs; s++)
						{
							simplex_bytes += mesh->surfs[s].vertCount * sizeof(zonetool::iw7::SHProbeSimplexData1);
						}
					}
				}
				proxy_bytes += simplex_bytes;
				ZONETOOL_INFO("static model clusters: %zu bytes of single static models' light probe simplex records", simplex_bytes);
			}
			if (cluster_bytes + proxy_bytes > geometry_budget)
			{
				ZONETOOL_FATAL("static model clusters: %zu bytes of merged geometry at the second LOD and %zu of shadow proxies, past the %zu "
					"the zone allows", cluster_bytes, proxy_bytes, geometry_budget);
			}
			std::vector<upgrade> upgrades;
			for (const auto& [model, t] : extra_of)
			{
				const auto& info = model_infos[model];
				const auto b0 = static_cast<double>(lod_bytes(info, 0));
				const auto lost = b0 > 0.0 ? 1.0 - static_cast<double>(lod_bytes(info, 1)) / b0 : 0.0;
				upgrades.push_back({ model, t.second, t.second ? lost * t.first / static_cast<double>(t.second) : 1e30 });
			}
			std::ranges::sort(upgrades, [](const upgrade& a, const upgrade& b) { return a.priority > b.priority; });
			std::unordered_set<const XModel*> full;
			for (const auto& u : upgrades)
			{
				if (cluster_bytes + proxy_bytes + u.extra <= geometry_budget)
				{
					cluster_bytes += u.extra;
					full.insert(u.model);
				}
			}
			auto members = 0u, members_full = 0u;
			for (const auto* groups : { &p.clusters, &p.world_clusters })
			for (const auto& group : *groups)
			{
				for (const auto i : group.members)
				{
					members++;
					if (full.contains(asset->dpvs.smodelDrawInsts[i].model))
					{
						p.member_lod[i] = 0;
					}
					members_full += p.member_lod[i] == 0;
				}
			}
			ZONETOOL_INFO("static model clusters: %u of %u members at their first LOD (%zu of %zu models with a second), geometry %zu bytes "
				"merged (%zu at every first LOD) + %zu of shadow proxies, of %zu", members_full, members, full.size(), upgrades.size(),
				cluster_bytes, full_bytes, proxy_bytes, geometry_budget);

			// ---- the world groups' members: their last LOD, the largest (placed radius) back to their first within the budget
			{
				const auto triangles_of = [&](const unsigned int i, const unsigned int lod)
				{
					const auto& info = model_infos[asset->dpvs.smodelDrawInsts[i].model];
					std::size_t t = 0;
					for (const auto n : info.triangles[std::min(lod, info.lods - 1)])
					{
						t += n;
					}
					return t;
				};
				std::vector<unsigned int> world_members;
				std::size_t total = 0, at_first = 0;
				for (const auto& group : p.world_clusters)
				{
					for (const auto i : group.members)
					{
						const auto& info = model_infos[asset->dpvs.smodelDrawInsts[i].model];
						p.member_lod[i] = static_cast<unsigned char>(info.lods - 1);
						total += triangles_of(i, info.lods - 1);
						world_members.push_back(i);
						at_first += triangles_of(i, 0);
					}
				}
				std::ranges::stable_sort(world_members, [&](const unsigned int a, const unsigned int b)
				{
					const auto& sa = asset->dpvs.smodelDrawInsts[a];
					const auto& sb = asset->dpvs.smodelDrawInsts[b];
					return sa.model->radius * sa.placement.scale > sb.model->radius * sb.placement.scale;
				});
				auto upgraded = 0u;
				for (const auto i : world_members)
				{
					const auto lod = p.member_lod[i];
					const auto extra = triangles_of(i, 0) - std::min(triangles_of(i, 0), triangles_of(i, lod));
					if (lod && total + extra <= world_triangle_budget)
					{
						total += extra;
						p.member_lod[i] = 0;
						upgraded++;
					}
				}
				if (!world_members.empty())
				{
					ZONETOOL_INFO("static model clusters: world groups %zu members, %zu triangles (%u largest at their first LOD, the rest "
						"at their last; %zu at every first LOD), budget %zu", world_members.size(), total, upgraded, at_first,
						world_triangle_budget);
				}
				if (total > world_triangle_budget && !std::getenv("ZT_CLUSTER_PLAN_ONLY"))
				{
					ZONETOOL_FATAL("static model clusters: the world groups' %zu triangles at their last LODs pass the %zu budget", total,
						world_triangle_budget);
				}
			}
			if (std::getenv("ZT_CLUSTER_PLAN_ONLY"))
			{
				ZONETOOL_INFO("static model clusters plan: PASS");
				std::fflush(stdout);
				std::exit(0);
			}
		}

		const plan& current()
		{
			return current_plan;
		}

		void umbra_geometry(const unsigned int iw7_index, std::vector<std::array<float, 6>>& member_bounds, std::vector<float>& vertices,
			std::vector<std::uint32_t>& indices)
		{
			member_bounds.clear();
			vertices.clear();
			indices.clear();
			const auto* asset = planned_world;
			if (!asset || iw7_index >= current_plan.proxy_begin)
			{
				return;
			}
			// the BO3 static models an IW7 index draws, found once
			static const GfxWorld* members_of_world = nullptr;
			static std::vector<std::vector<unsigned int>> members_of;
			if (members_of_world != asset)
			{
				members_of_world = asset;
				members_of.assign(current_plan.proxy_begin, {});
				for (auto i = 0u; i < asset->dpvs.smodelCount; i++)
				{
					const auto index = current_plan.iw7_index[i];
					if (index < members_of.size())
					{
						members_of[index].push_back(i);
					}
				}
			}
			std::unordered_map<const Material*, bool> opaque;
			const auto is_opaque = [&](const Material* material)
			{
				auto [it, added] = opaque.try_emplace(material, false);
				if (added && material && world_material::model_blocks_sun(material) && !world_material::bake::alpha_mask_of(material).has_value())
				{
					const auto name = world_material::bake::bo3_template(material);
					it->second = name.find("alpha") == std::string::npos && name.find("atest") == std::string::npos
						&& name.find("foliage") == std::string::npos;
				}
				return it->second;
			};
			for (const auto i : members_of[iw7_index])
			{
				const auto& src = asset->dpvs.smodelDrawInsts[i];
				member_bounds.push_back({ src.mins[0], src.mins[1], src.mins[2], src.maxs[0], src.maxs[1], src.maxs[2] });
				const auto posed = src.numPosedBones || src.posedBones;
				if (posed || src.targetname || src.hidden || src.sanim)
				{
					continue; // moved, hidden or posed by script or bones: not where its placement puts it
				}
				const auto meshes = xmodel::lod_meshes(src.model);
				if (meshes.empty())
				{
					continue;
				}
				const auto lod = current_plan.cluster_of[i] >= 0 ? std::min<std::size_t>(current_plan.member_lod[i], meshes.size() - 1) : 0;
				const auto& [mesh, mats] = meshes[lod];
				for (auto s = 0u; s < mats.size(); s++)
				{
					if (!is_opaque(mats[s]) || s >= static_cast<unsigned int>(mesh->numSurfs))
					{
						continue;
					}
					std::vector<zonetool::iw7::GfxWorldVertex> verts;
					std::vector<unsigned short> faces;
					if (!xmodel_mesh::append_placed_world(mesh, s, 0, mesh->surfs[s].triCount, src.placement.origin, src.placement.axis,
						src.placement.scale, verts, faces, 0xFFFF))
					{
						continue;
					}
					const auto first = static_cast<std::uint32_t>(vertices.size() / 3);
					for (const auto& v : verts)
					{
						vertices.insert(vertices.end(), v.xyz, v.xyz + 3);
					}
					for (const auto f : faces)
					{
						indices.push_back(first + f);
					}
				}
			}
		}

		namespace
		{
			constexpr unsigned int max_light_probes = 64;
			constexpr float probe_surface_offset = 4.0f;

			std::vector<std::vector<std::array<float, 6>>> cluster_probes; // by IW7 static model index
			std::vector<float> cluster_probe_normal_scale;
			const GfxWorld* probes_world = nullptr;

			void unpack_normal(const std::uint32_t packed, float n[3])
			{
				for (auto c = 0; c < 3; c++)
				{
					n[c] = static_cast<float>((packed >> (10 * c)) & 0x3FF) / 1023.0f * 2.0f - 1.0f;
				}
				const auto length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
				for (auto c = 0; c < 3 && length > 0.0f; c++)
				{
					n[c] /= length;
				}
			}

			// a vertex's simplex record as Spaceland's (XSurface flags 0x300, 8 bytes): four probe indices, four weights
			struct SHProbeSimplexWeights
			{
				unsigned char index[4];
				unsigned char weight[4];
			};
			static_assert(sizeof(SHProbeSimplexWeights) == sizeof(zonetool::iw7::SHProbeSimplexData1));

			float probe_distance_sq(const std::array<float, 6>& probe, const float p[3], const float n[3], const float normal_scale)
			{
				auto d = 0.0f;
				for (auto c = 0; c < 3; c++)
				{
					const auto dp = probe[c] - p[c];
					const auto dn = (probe[3 + c] - n[c]) * normal_scale;
					d += dp * dp + dn * dn;
				}
				return d;
			}

			// a vertex's four nearest probes (position, and normal scaled to the cluster's size), weighted by inverse squared
			// distance, the weights in bytes summing to 255
			SHProbeSimplexWeights probe_weights(const std::vector<std::array<float, 6>>& probes, const float normal_scale, const float p[3],
				const float n[3])
			{
				std::array<std::pair<float, unsigned int>, 4> best;
				best.fill({ FLT_MAX, 0u });
				for (auto k = 0u; k < probes.size(); k++)
				{
					const auto d = probe_distance_sq(probes[k], p, n, normal_scale);
					if (d < best[3].first)
					{
						best[3] = { d, k };
						std::ranges::sort(best);
					}
				}
				float w[4]{};
				auto total = 0.0f;
				const auto floor_sq = normal_scale * normal_scale * 0.0025f + 1.0f;
				for (auto k = 0; k < 4; k++)
				{
					w[k] = best[k].first == FLT_MAX ? 0.0f : 1.0f / (best[k].first + floor_sq);
					total += w[k];
				}
				SHProbeSimplexWeights out{};
				auto sum = 0;
				for (auto k = 0; k < 4; k++)
				{
					out.index[k] = static_cast<unsigned char>(best[k].first == FLT_MAX ? best[0].second : best[k].second);
					out.weight[k] = static_cast<unsigned char>(std::floor(w[k] / total * 255.0f + 0.5f));
					sum += out.weight[k];
				}
				out.weight[0] = static_cast<unsigned char>(std::clamp(out.weight[0] + 255 - sum, 0, 255));
				return out;
			}

			// probes over a set of surface samples (position, normal): k-means over both (the normal scaled to an eighth of
			// their diagonal, so a wall's two faces and a roof's top and underside light apart), each probe at the sample
			// nearest its centre, lifted off that surface along its normal; returns the normal scale
			float layout_probes(std::vector<std::array<float, 6>>& samples, const unsigned int max_count, std::vector<std::array<float, 6>>& probes)
			{
				probes.clear();
				if (samples.empty())
				{
					return 0.0f;
				}
				constexpr std::size_t max_samples = 16384;
				if (samples.size() > max_samples)
				{
					std::vector<std::array<float, 6>> kept;
					const auto step = static_cast<double>(samples.size()) / max_samples;
					for (auto k = 0u; k < max_samples; k++)
					{
						kept.push_back(samples[static_cast<std::size_t>(k * step)]);
					}
					samples = std::move(kept);
				}
				float lo[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, hi[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
				for (const auto& s : samples)
				{
					for (auto k = 0; k < 3; k++)
					{
						lo[k] = std::min(lo[k], s[k]);
						hi[k] = std::max(hi[k], s[k]);
					}
				}
				const auto diagonal = std::sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2]));
				const auto normal_scale = std::max(32.0f, diagonal / 8.0f);
				const auto count = static_cast<unsigned int>(std::min<std::size_t>(max_count, samples.size()));

				// farthest point seeds, then Lloyd iterations
				std::vector<std::array<float, 6>> centres{ samples.front() };
				std::vector<float> nearest(samples.size(), FLT_MAX);
				while (centres.size() < count)
				{
					auto farthest = 0u;
					for (auto s = 0u; s < samples.size(); s++)
					{
						nearest[s] = std::min(nearest[s], probe_distance_sq(centres.back(), samples[s].data(), &samples[s][3], normal_scale));
						if (nearest[s] > nearest[farthest])
						{
							farthest = s;
						}
					}
					if (nearest[farthest] <= 0.0f)
					{
						break;
					}
					centres.push_back(samples[farthest]);
				}
				std::vector<unsigned int> owner(samples.size(), 0u);
				for (auto iteration = 0; iteration < 10; iteration++)
				{
					for (auto s = 0u; s < samples.size(); s++)
					{
						auto best = FLT_MAX;
						for (auto k = 0u; k < centres.size(); k++)
						{
							const auto d = probe_distance_sq(centres[k], samples[s].data(), &samples[s][3], normal_scale);
							if (d < best)
							{
								best = d;
								owner[s] = k;
							}
						}
					}
					std::vector<std::array<double, 6>> sums(centres.size(), std::array<double, 6>{});
					std::vector<unsigned int> counts(centres.size(), 0u);
					for (auto s = 0u; s < samples.size(); s++)
					{
						for (auto k = 0; k < 6; k++)
						{
							sums[owner[s]][k] += samples[s][k];
						}
						counts[owner[s]]++;
					}
					for (auto k = 0u; k < centres.size(); k++)
					{
						if (counts[k])
						{
							for (auto j = 0; j < 6; j++)
							{
								centres[k][j] = static_cast<float>(sums[k][j] / counts[k]);
							}
						}
					}
				}
				for (auto k = 0u; k < centres.size(); k++)
				{
					auto best = FLT_MAX;
					const std::array<float, 6>* at = nullptr;
					for (auto s = 0u; s < samples.size(); s++)
					{
						if (owner[s] != k)
						{
							continue;
						}
						const auto d = probe_distance_sq(centres[k], samples[s].data(), &samples[s][3], normal_scale);
						if (d < best)
						{
							best = d;
							at = &samples[s];
						}
					}
					if (!at)
					{
						continue;
					}
					auto probe = *at;
					for (auto j = 0; j < 3; j++)
					{
						probe[j] += probe[3 + j] * probe_surface_offset;
					}
					probes.push_back(probe);
				}
				return normal_scale;
			}

			// a mesh's vertices placed (position, normal); the identity placement gives model space
			void append_samples(XModelMesh* mesh, const std::size_t surfaces, const float* origin, const float (*axis)[3], const float scale,
				std::vector<std::array<float, 6>>& samples)
			{
				for (auto s = 0u; s < surfaces && s < static_cast<unsigned int>(mesh->numSurfs); s++)
				{
					std::vector<zonetool::iw7::GfxWorldVertex> verts;
					std::vector<unsigned short> faces;
					if (!xmodel_mesh::append_placed_world(mesh, s, 0, mesh->surfs[s].triCount, origin, axis, scale, verts, faces, 0xFFFF))
					{
						continue;
					}
					for (const auto& v : verts)
					{
						std::array<float, 6> sample{ v.xyz[0], v.xyz[1], v.xyz[2] };
						unpack_normal(v.normal.packed, &sample[3]);
						samples.push_back(sample);
					}
				}
			}

			// a single static model's probes in its model's space (every placement shares its vertices' simplex records, as
			// Spaceland's shared models do): one per 96 units of its diagonal, 4 to 64
			struct model_probe_layout
			{
				std::vector<std::array<float, 6>> probes;
				float normal_scale = 0.0f;
			};
			std::unordered_map<const XModel*, model_probe_layout> single_layouts;
			std::unordered_map<const XModelMesh*, const model_probe_layout*> mesh_layouts;

			void plan_light_probes()
			{
				const auto* asset = planned_world;
				probes_world = asset;
				cluster_probes.assign(current_plan.proxy_begin, {});
				cluster_probe_normal_scale.assign(current_plan.proxy_begin, 0.0f);
				single_layouts.clear();
				mesh_layouts.clear();
				parallel_for(static_cast<std::uint32_t>(current_plan.clusters.size()), [&](const std::uint32_t c, std::uint32_t)
				{
					const auto& group = current_plan.clusters[c];
					if (group.posed || group.too_many_surfaces || group.members.empty())
					{
						return;
					}
					const auto index = current_plan.iw7_index[group.members.front()];
					if (index >= cluster_probes.size())
					{
						return;
					}
					std::vector<std::array<float, 6>> samples;
					for (const auto i : group.members)
					{
						const auto& src = asset->dpvs.smodelDrawInsts[i];
						const auto meshes = xmodel::lod_meshes(src.model);
						if (meshes.empty())
						{
							continue;
						}
						const auto& [mesh, mats] = meshes[std::min<std::size_t>(current_plan.member_lod[i], meshes.size() - 1)];
						append_samples(mesh, mats.size(), src.placement.origin, src.placement.axis, src.placement.scale, samples);
					}
					std::vector<std::array<float, 6>> probes;
					cluster_probe_normal_scale[index] = layout_probes(samples, max_light_probes, probes);
					cluster_probes[index] = std::move(probes);
				});
				auto clusters = 0u, cluster_probe_count = 0u;
				for (const auto& p : cluster_probes)
				{
					clusters += !p.empty();
					cluster_probe_count += static_cast<unsigned int>(p.size());
				}

				// the single static models: unposed (a skinned one drawn at its bind pose)
				std::vector<const XModel*> models;
				for (auto i = 0u; i < asset->dpvs.smodelCount; i++)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					if (current_plan.cluster_of[i] >= 0 || current_plan.iw7_index[i] >= current_plan.proxy_begin || src.numPosedBones
						|| src.posedBones || !src.model || single_layouts.contains(src.model))
					{
						continue;
					}
					const auto meshes = xmodel::lod_meshes(src.model);
					if (meshes.empty() || std::ranges::any_of(meshes, [](const auto& m) { return !m.first; }))
					{
						continue;
					}
					single_layouts.emplace(src.model, model_probe_layout{});
					models.push_back(src.model);
				}
				parallel_for(static_cast<std::uint32_t>(models.size()), [&](const std::uint32_t k, std::uint32_t)
				{
					const auto meshes = xmodel::lod_meshes(models[k]);
					std::vector<std::array<float, 6>> samples;
					xmodel_mesh::model_space_vertices(meshes.front().first, samples);
					float lo[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, hi[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
					for (const auto& s : samples)
					{
						for (auto c = 0; c < 3; c++)
						{
							lo[c] = std::min(lo[c], s[c]);
							hi[c] = std::max(hi[c], s[c]);
						}
					}
					const auto diagonal = samples.empty() ? 0.0f
						: std::sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2]));
					const auto count = std::clamp(static_cast<unsigned int>(std::ceil(diagonal / 96.0f)), 4u, max_light_probes);
					auto& layout = single_layouts.at(models[k]);
					layout.normal_scale = layout_probes(samples, count, layout.probes);
				});
				for (const auto* model : models)
				{
					const auto& layout = single_layouts.at(model);
					if (layout.probes.empty())
					{
						continue;
					}
					for (const auto& [mesh, mats] : xmodel::lod_meshes(model))
					{
						mesh_layouts.emplace(mesh, &layout);
					}
				}
				auto singles = 0u, single_probe_count = 0u;
				for (auto i = 0u; i < asset->dpvs.smodelCount; i++)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					const auto index = current_plan.iw7_index[i];
					const auto found = current_plan.cluster_of[i] < 0 && index < cluster_probes.size() ? single_layouts.find(src.model) : single_layouts.end();
					if (found == single_layouts.end() || found->second.probes.empty())
					{
						continue;
					}
					auto& out = cluster_probes[index];
					out.clear();
					for (const auto& local : found->second.probes)
					{
						std::array<float, 6> world{};
						for (auto c = 0; c < 3; c++)
						{
							world[c] = src.placement.origin[c] + src.placement.scale * (local[0] * src.placement.axis[0][c]
								+ local[1] * src.placement.axis[1][c] + local[2] * src.placement.axis[2][c]);
							world[3 + c] = local[3] * src.placement.axis[0][c] + local[4] * src.placement.axis[1][c] + local[5] * src.placement.axis[2][c];
						}
						const auto length = std::sqrt(world[3] * world[3] + world[4] * world[4] + world[5] * world[5]);
						for (auto c = 3; c < 6 && length > 0.0f; c++)
						{
							world[c] /= length;
						}
						out.push_back(world);
					}
					singles++;
					single_probe_count += static_cast<unsigned int>(out.size());
				}
				ZONETOOL_INFO("static model clusters: %u merged models lit by %u light probes (up to %u each), %u single static models by %u "
					"(%zu models); a vertex blends four", clusters, cluster_probe_count, max_light_probes, singles, single_probe_count, models.size());
			}
		}

		bool write_probe_simplex(const XModelMesh* mesh, const float p[3], const float n[3], unsigned char out[8])
		{
			if (!planned_world)
			{
				return false;
			}
			if (probes_world != planned_world)
			{
				plan_light_probes();
			}
			const auto found = mesh_layouts.find(mesh);
			if (found == mesh_layouts.end())
			{
				return false;
			}
			const auto weights = probe_weights(found->second->probes, found->second->normal_scale, p, n);
			std::memcpy(out, &weights, sizeof(weights));
			return true;
		}

		const std::vector<std::array<float, 6>>& light_probes(const unsigned int iw7_index)
		{
			static const std::vector<std::array<float, 6>> none;
			if (!planned_world)
			{
				return none;
			}
			if (probes_world != planned_world)
			{
				plan_light_probes();
			}
			return iw7_index < cluster_probes.size() ? cluster_probes[iw7_index] : none;
		}

		std::vector<unsigned short> remap_indices(const unsigned short* indices, const unsigned int count)
		{
			std::vector<unsigned short> out;
			out.reserve(count);
			std::unordered_set<unsigned int> seen;
			for (auto i = 0u; i < count; i++)
			{
				const auto mapped = current_plan.iw7_index.at(indices[i]);
				if (mapped != ~0u && seen.insert(mapped).second)
				{
					out.push_back(static_cast<unsigned short>(mapped));
				}
			}
			return out;
		}

		void dump_all()
		{
			if (!planned_world)
			{
				return;
			}
			const auto* asset = planned_world;
			for (const auto& group : current_plan.clusters)
			{
				utils::memory::allocator allocator;

				// One LOD, every member at its first: IW7 picks a static model's LOD by the camera's distance (0x140D87820)
				// for the view and for the cached sun shadow, which redraws a caster's shadow tiles whenever its LOD changes
				// (0x140DD0040). A merged model switching LOD would switch its whole cell, and its shadow, at once (BO3
				// switches each member on its own). Drawn until the last member's
				// cull distance (its own last LOD's, world units: x its scale, plus how far it sits from the origin).
				// A cluster with a far LOD (has_far_lod) switches to every member's last LOD where the last member to reach its own last LOD
				// does (its second last LOD's distance, x its scale, plus how far it sits from the origin): no member draws coarser than
				// BO3 would draw it, and the distant clusters draw their cheapest LODs
				std::vector<float> dist(1, 0.0f);
				auto far_from = 0.0f;
				std::unordered_set<const Material*> far_materials;
				for (const auto i : group.members)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					const auto& info = model_infos[src.model];
					const float d[3] = { src.center[0] - group.origin[0], src.center[1] - group.origin[1], src.center[2] - group.origin[2] };
					const auto offset = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
					dist[0] = std::max(dist[0], xmodel::lod_dist(src.model, info.lods - 1) * src.placement.scale + offset);
					if (info.lods >= 2 && current_plan.member_lod[i] < info.lods - 1)
					{
						far_from = std::max(far_from, xmodel::lod_dist(src.model, info.lods - 2) * src.placement.scale + offset);
					}
					if (!info.meshes.empty())
					{
						far_materials.insert(info.meshes[info.lods - 1].second.begin(), info.meshes[info.lods - 1].second.end());
					}
				}
				const auto far_lod = has_far_lod(group) && far_from > 0.0f && far_from < dist[0] && far_materials.size() <= max_surfaces;
				if (far_lod)
				{
					dist = { far_from, dist[0] };
				}
				const auto lods = static_cast<unsigned int>(dist.size());

				auto* model = allocator.allocate<zonetool::iw7::XModel>();
				model->name = allocator.duplicate_string(group.name);
				model->numLods = static_cast<unsigned char>(lods);
				model->numBones = 1;
				model->numRootBones = 1;
				model->scale = 1.0f;
				const auto* first = asset->dpvs.smodelDrawInsts[group.members.front()].model;
				model->boneNames = allocator.allocate_array<zonetool::iw7::scr_string_t>(1);
				model->boneNames[0] = static_cast<zonetool::iw7::scr_string_t>(first->boneNames[0]);
				model->partClassification = allocator.allocate_array<unsigned char>(1);
				model->baseMat = allocator.allocate_array<zonetool::iw7::DObjAnimMat>(1);
				model->baseMat[0].quat[3] = 1.0f;
				model->baseMat[0].transWeight = 2.0f;

				const auto iw7_index = current_plan.iw7_index[group.members.front()];
				const auto& probes = light_probes(iw7_index);
				const auto normal_scale = iw7_index < cluster_probe_normal_scale.size() ? cluster_probe_normal_scale[iw7_index] : 0.0f;

				std::vector<zonetool::iw7::Material*> handles;
				std::vector<float> himip;
				float mins[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, maxs[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
				for (auto k = 0u; k < lods; k++)
				{
					std::vector<const Material*> materials;
					std::vector<std::vector<zonetool::iw7::GfxPackedVertex>> verts;
					std::vector<std::vector<zonetool::iw7::Face>> faces;
					std::vector<float> surface_himip;
					for (auto m = 0u; m < group.members.size(); m++)
					{
						const auto i = group.members[m];
						const auto& src = asset->dpvs.smodelDrawInsts[i];
						const auto& info = model_infos[src.model];
						const auto member_lod = group.posed ? std::min(k, info.lods - 1)
							: k == 1 && far_lod ? info.lods - 1 : static_cast<unsigned int>(current_plan.member_lod[i]);
						const auto& [mesh, mats] = info.meshes[member_lod];
						const auto radii = xmodel::himip_inv_sq_radii(src.model);
						for (auto s = 0u; s < mats.size(); s++)
						{
							// the material's last surface while it has room, else a new one (lod_budget::place)
							auto at = static_cast<std::ptrdiff_t>(materials.size());
							for (auto q = materials.size(); q-- > 0;)
							{
								if (materials[q] == mats[s] || (materials[q] && mats[s] && material::get_converted_name(const_cast<Material*>(materials[q]))
									== material::get_converted_name(const_cast<Material*>(mats[s]))))
								{
									// a posed model is a cluster of its own (its info has no counts: it is not mergeable)
									if (group.posed || (verts[q].size() + info.vertices[member_lod][s] <= max_vertices
										&& faces[q].size() + info.triangles[member_lod][s] <= max_vertices))
									{
										at = static_cast<std::ptrdiff_t>(q);
									}
									break;
								}
							}
							if (at == static_cast<std::ptrdiff_t>(materials.size()))
							{
								materials.push_back(mats[s]);
								verts.emplace_back();
								faces.emplace_back();
								surface_himip.push_back(0.0f);
							}
							xmodel_mesh::surface_transform transform{};
							if (mats[s])
							{
								const auto& mat_info = world_material::get_model(mats[s]);
								transform = { { mat_info.uv_origin[0], mat_info.uv_origin[1] }, { mat_info.uv_span[0], mat_info.uv_span[1] },
									mat_info.vertex_alpha };
							}
							const auto appended = group.posed
								? xmodel_mesh::append_posed(mesh, s, transform, group.skin, src.placement.origin, src.placement.axis,
									src.placement.scale, group.origin, verts[at], faces[at])
								: xmodel_mesh::append_placed(mesh, s, transform, src.placement.origin, src.placement.axis, src.placement.scale,
									group.origin, verts[at], faces[at], group.visibility[m]);
							if (!appended)
							{
								ZONETOOL_FATAL("static model cluster %s: surface %u of %s did not fit", group.name.data(), s, mesh->name);
							}
							if (member_lod < radii.size() && s < radii[member_lod].size())
							{
								surface_himip[at] = std::max(surface_himip[at], radii[member_lod][s]);
							}
						}
					}

					// in IW7's material order, the first surface first (static models are drawn in their first material's order)
					{
						std::vector<std::size_t> by_order(materials.size());
						std::iota(by_order.begin(), by_order.end(), 0u);
						std::vector<std::tuple<unsigned char, unsigned char, std::string, std::string>> keys;
						for (const auto* material : materials)
						{
							keys.push_back(world_material::model_draw_order(material));
						}
						std::ranges::stable_sort(by_order, [&](const std::size_t a, const std::size_t b)
						{
							return keys[a] < keys[b];
						});
						std::vector<const Material*> m2;
						std::vector<std::vector<zonetool::iw7::GfxPackedVertex>> v2;
						std::vector<std::vector<zonetool::iw7::Face>> f2;
						std::vector<float> h2;
						for (const auto at : by_order)
						{
							m2.push_back(materials[at]);
							v2.push_back(std::move(verts[at]));
							f2.push_back(std::move(faces[at]));
							h2.push_back(surface_himip[at]);
						}
						materials = std::move(m2);
						verts = std::move(v2);
						faces = std::move(f2);
						surface_himip = std::move(h2);
					}

					auto* surfs = allocator.allocate<zonetool::iw7::XModelSurfs>();
					surfs->name = allocator.duplicate_string(utils::string::va("%s_lod%u", group.name.data(), k));
					surfs->numsurfs = static_cast<unsigned short>(materials.size());
					surfs->surfs = allocator.allocate_array<zonetool::iw7::XSurface>(materials.size());
					surfs->partBits[0] = static_cast<int>(0x80000000u);
					for (auto s = 0u; s < materials.size(); s++)
					{
						auto& surf = surfs->surfs[s];
						surf.flags = zonetool::iw7::SURF_FLAG_SELF_VISIBILITY;
						surf.vertCount = static_cast<unsigned short>(verts[s].size());
						surf.triCount = static_cast<unsigned short>(faces[s].size());
						surf.verts0.packedVerts0 = allocator.allocate_array<zonetool::iw7::GfxPackedVertex>(verts[s].size());
						std::memcpy(surf.verts0.packedVerts0, verts[s].data(), verts[s].size() * sizeof(zonetool::iw7::GfxPackedVertex));
						surf.triIndices = allocator.allocate_array<zonetool::iw7::Face>(faces[s].size());
						std::memcpy(surf.triIndices, faces[s].data(), faces[s].size() * sizeof(zonetool::iw7::Face));
						surf.rigidVertListCount = 1;
						surf.rigidVertLists = allocator.allocate_array<zonetool::iw7::XRigidVertList>(1);
						surf.rigidVertLists[0].boneOffsetIndex = 0;
						surf.rigidVertLists[0].vertCount = surf.vertCount;
						surf.rigidVertLists[0].triOffset = 0;
						surf.rigidVertLists[0].triCount = surf.triCount;
						surf.partBits[0] = static_cast<int>(0x80000000u);
						if (!probes.empty())
						{
							surf.flags |= zonetool::iw7::SURF_FLAG_SECONDUV | zonetool::iw7::SURF_FLAG_MAYHEM_CUSTOM_CHANNELS;
							auto* simplex = allocator.allocate_array<zonetool::iw7::SHProbeSimplexData1>(verts[s].size());
							for (auto v = 0u; v < verts[s].size(); v++)
							{
								const auto& vertex = verts[s][v];
								const float p[3] = { vertex.xyz[0] + group.origin[0], vertex.xyz[1] + group.origin[1], vertex.xyz[2] + group.origin[2] };
								float n[3];
								unpack_normal(vertex.normal.packed, n);
								const auto weights = probe_weights(probes, normal_scale, p, n);
								std::memcpy(&simplex[v], &weights, sizeof(weights));
							}
							surf.shProbeSimplexVertData.data1 = simplex;
						}
						if (k == 0)
						{
							for (const auto& v : verts[s])
							{
								for (auto c = 0; c < 3; c++)
								{
									mins[c] = std::min(mins[c], v.xyz[c]);
									maxs[c] = std::max(maxs[c], v.xyz[c]);
								}
							}
						}
						auto* handle = allocator.allocate<zonetool::iw7::Material>();
						handle->name = allocator.duplicate_string(material::get_converted_name(const_cast<Material*>(materials[s])));
						handles.push_back(handle);
						himip.push_back(surface_himip[s]);
					}
					zonetool::iw7::xsurface::dump(surfs);

					auto& lod = model->lodInfo[k];
					lod.dist = dist[k];
					lod.numsurfs = surfs->numsurfs;
					lod.surfIndex = static_cast<unsigned short>(handles.size() - materials.size());
					lod.modelSurfs = allocator.allocate<zonetool::iw7::XModelSurfs>();
					lod.modelSurfs->name = surfs->name;
					lod.partBits[0] = static_cast<int>(0x80000000u);
				}
				for (auto k = lods; k < 6; k++)
				{
					model->lodInfo[k].dist = 1000000.0f;
				}

				model->numsurfs = static_cast<unsigned char>(handles.size());
				model->materialHandles = allocator.allocate_array<zonetool::iw7::Material*>(handles.size());
				std::memcpy(model->materialHandles, handles.data(), handles.size() * sizeof(zonetool::iw7::Material*));
				model->invHighMipRadius = allocator.allocate_array<unsigned short>(himip.size());
				for (auto s = 0u; s < himip.size(); s++)
				{
					model->invHighMipRadius[s] = xmodel::inv_high_mip_radius(himip[s]);
				}

				auto radius_sq = 0.0f;
				for (auto c = 0; c < 3; c++)
				{
					const auto r = std::max(std::fabs(mins[c]), std::fabs(maxs[c]));
					radius_sq += r * r;
				}
				model->boneInfo = allocator.allocate_array<zonetool::iw7::XBoneInfo>(1);
				compute(&model->boneInfo[0].bounds, mins, maxs);
				model->boneInfo[0].radiusSquared = radius_sq;
				model->radius = std::sqrt(radius_sq);
				compute(&model->bounds, mins, maxs);

				model->collLod = 0xFF;
				model->flags = 0x40; // stock value
				model->hasLods = lods ? 1 : 0;
				model->shadowCutoffLod = 6;
				model->characterCollBoundsType = 1;
				model->unknownIndex = 0xFF;
				model->unknownIndex2 = 0xFF;
				zonetool::iw7::xmodel::dump(model);
			}
			ZONETOOL_INFO("static model clusters: %zu models written", current_plan.clusters.size());

			// ---- the shadow proxies -------------------------------------------------------------------------------------
			if (current_plan.proxies.empty())
			{
				return;
			}
			{
				const auto* donor = world_techset_donors::find(proxy_techset);
				if (!donor)
				{
					ZONETOOL_FATAL("shadow proxies: no donor for techset %s", proxy_techset);
				}
				std::vector<std::pair<std::uint32_t, std::string>> images;
				for (auto t = 0u; t < donor->texture_count; t++)
				{
					images.emplace_back(donor->textures[t].type_hash, donor->textures[t].stock_image);
				}
				world_material::bake::write_stock_material(proxy_material, proxy_techset, images);
			}
			// the alpha tested materials as shadow-only copies (their converted material missing: the opaque one)
			std::unordered_map<const Material*, std::string> alpha_copies;
			auto copies_missing = 0u;
			for (const auto& [material, kind] : caster_kinds)
			{
				if (kind != caster_kind::alpha_tested)
				{
					continue;
				}
				const auto name = material::get_converted_name(const_cast<Material*>(material));
				const auto copy = name + "_shadow";
				if (write_shadow_copy(name, copy))
				{
					alpha_copies.emplace(material, copy);
				}
				else
				{
					alpha_copies.emplace(material, proxy_material);
					copies_missing++;
					ZONETOOL_WARNING("shadow proxies: material %s was not written; its shadow is drawn opaque", name.data());
				}
			}

			std::size_t vertices_before = 0, vertices_after = 0;
			for (const auto& proxy : current_plan.proxies)
			{
				utils::memory::allocator allocator;

				// the surfaces as the plan filled them (proxy_budget): opaque ones in order, one per alpha tested material
				std::vector<std::string> materials;
				std::vector<bool> opaque;
				std::vector<std::vector<zonetool::iw7::GfxPackedVertex>> verts;
				std::vector<std::vector<zonetool::iw7::Face>> faces;
				std::unordered_map<const Material*, std::size_t> alpha_surface;
				auto opaque_surface = ~std::size_t{ 0 };
				const auto add_surface = [&](const std::string& material, const bool is_opaque)
				{
					materials.push_back(material);
					opaque.push_back(is_opaque);
					verts.emplace_back();
					faces.emplace_back();
					return materials.size() - 1;
				};
				for (const auto i : proxy.members)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					const auto& info = model_infos[src.model];
					const auto lod = proxy_lod(info);
					const auto& [mesh, mats] = info.meshes[lod];
					for (auto s = 0u; s < mats.size(); s++)
					{
						const auto kind = mats[s] ? caster_kinds.at(mats[s]) : caster_kind::none;
						if (kind == caster_kind::none || (kind == caster_kind::alpha_tested) != proxy.alpha)
						{
							continue;
						}
						std::size_t at;
						xmodel_mesh::surface_transform transform{};
						if (kind == caster_kind::opaque)
						{
							if (opaque_surface == ~std::size_t{ 0 } || verts[opaque_surface].size() + info.vertices[lod][s] > max_vertices
								|| faces[opaque_surface].size() + info.triangles[lod][s] > max_vertices)
							{
								opaque_surface = add_surface(proxy_material, true);
							}
							at = opaque_surface;
						}
						else
						{
							const auto found = alpha_surface.find(mats[s]);
							at = found != alpha_surface.end() ? found->second : alpha_surface.emplace(mats[s], add_surface(alpha_copies.at(mats[s]), false)).first->second;
							// its texture coordinates as its material's bake moved them
							const auto& mat_info = world_material::get_model(mats[s]);
							transform = { { mat_info.uv_origin[0], mat_info.uv_origin[1] }, { mat_info.uv_span[0], mat_info.uv_span[1] },
								mat_info.vertex_alpha };
						}
						if (!xmodel_mesh::append_placed(mesh, s, transform, src.placement.origin, src.placement.axis, src.placement.scale,
							proxy.origin, verts[at], faces[at]))
						{
							ZONETOOL_FATAL("shadow proxy %s: surface %u of %s did not fit", proxy.name.data(), s, mesh->name);
						}
						if (determinant(src.placement.axis) < 0.0f)
						{
							// mirrored: its placed triangles face inward, turned back as BO3 draws them
							for (auto f = faces[at].size() - info.triangles[lod][s]; f < faces[at].size(); f++)
							{
								std::swap(faces[at][f].v2, faces[at][f].v3);
							}
						}
					}
				}
				if (materials.size() > max_surfaces)
				{
					ZONETOOL_FATAL("shadow proxy %s: %zu surfaces, IW7 draws 16 a static model LOD", proxy.name.data(), materials.size());
				}

				// an opaque surface is drawn by its positions alone: one vertex per position
				for (auto s = 0u; s < materials.size(); s++)
				{
					vertices_before += verts[s].size();
					if (opaque[s])
					{
						std::map<std::array<float, 3>, std::uint16_t> index_of;
						std::vector<zonetool::iw7::GfxPackedVertex> welded;
						std::vector<std::uint16_t> remap(verts[s].size());
						for (auto v = 0u; v < verts[s].size(); v++)
						{
							const std::array<float, 3> key = { verts[s][v].xyz[0], verts[s][v].xyz[1], verts[s][v].xyz[2] };
							const auto [it, added] = index_of.try_emplace(key, static_cast<std::uint16_t>(welded.size()));
							if (added)
							{
								welded.push_back(verts[s][v]);
							}
							remap[v] = it->second;
						}
						for (auto& f : faces[s])
						{
							f.v1 = remap[f.v1];
							f.v2 = remap[f.v2];
							f.v3 = remap[f.v3];
						}
						verts[s] = std::move(welded);
					}
					vertices_after += verts[s].size();
				}

				auto* model = allocator.allocate<zonetool::iw7::XModel>();
				model->name = allocator.duplicate_string(proxy.name);
				model->numLods = 1;
				model->numBones = 1;
				model->numRootBones = 1;
				model->scale = 1.0f;
				const auto* first = asset->dpvs.smodelDrawInsts[proxy.members.front()].model;
				model->boneNames = allocator.allocate_array<zonetool::iw7::scr_string_t>(1);
				model->boneNames[0] = static_cast<zonetool::iw7::scr_string_t>(first->boneNames[0]);
				model->partClassification = allocator.allocate_array<unsigned char>(1);
				model->baseMat = allocator.allocate_array<zonetool::iw7::DObjAnimMat>(1);
				model->baseMat[0].quat[3] = 1.0f;
				model->baseMat[0].transWeight = 2.0f;

				float mins[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, maxs[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
				auto* surfs = allocator.allocate<zonetool::iw7::XModelSurfs>();
				surfs->name = allocator.duplicate_string(utils::string::va("%s_lod0", proxy.name.data()));
				surfs->numsurfs = static_cast<unsigned short>(materials.size());
				surfs->surfs = allocator.allocate_array<zonetool::iw7::XSurface>(materials.size());
				surfs->partBits[0] = static_cast<int>(0x80000000u);
				model->materialHandles = allocator.allocate_array<zonetool::iw7::Material*>(materials.size());
				model->invHighMipRadius = allocator.allocate_array<unsigned short>(materials.size());
				for (auto s = 0u; s < materials.size(); s++)
				{
					auto& surf = surfs->surfs[s];
					surf.flags = zonetool::iw7::SURF_FLAG_SELF_VISIBILITY;
					surf.vertCount = static_cast<unsigned short>(verts[s].size());
					surf.triCount = static_cast<unsigned short>(faces[s].size());
					surf.verts0.packedVerts0 = allocator.allocate_array<zonetool::iw7::GfxPackedVertex>(verts[s].size());
					std::memcpy(surf.verts0.packedVerts0, verts[s].data(), verts[s].size() * sizeof(zonetool::iw7::GfxPackedVertex));
					surf.triIndices = allocator.allocate_array<zonetool::iw7::Face>(faces[s].size());
					std::memcpy(surf.triIndices, faces[s].data(), faces[s].size() * sizeof(zonetool::iw7::Face));
					surf.rigidVertListCount = 1;
					surf.rigidVertLists = allocator.allocate_array<zonetool::iw7::XRigidVertList>(1);
					surf.rigidVertLists[0].boneOffsetIndex = 0;
					surf.rigidVertLists[0].vertCount = surf.vertCount;
					surf.rigidVertLists[0].triOffset = 0;
					surf.rigidVertLists[0].triCount = surf.triCount;
					surf.partBits[0] = static_cast<int>(0x80000000u);
					for (const auto& v : verts[s])
					{
						for (auto c = 0; c < 3; c++)
						{
							mins[c] = std::min(mins[c], v.xyz[c]);
							maxs[c] = std::max(maxs[c], v.xyz[c]);
						}
					}
					model->materialHandles[s] = allocator.allocate<zonetool::iw7::Material>();
					model->materialHandles[s]->name = allocator.duplicate_string(materials[s]);
					model->invHighMipRadius[s] = xmodel::inv_high_mip_radius(0.0f);
				}
				zonetool::iw7::xsurface::dump(surfs);

				auto& lod = model->lodInfo[0];
				lod.dist = proxy.dist;
				lod.numsurfs = surfs->numsurfs;
				lod.surfIndex = 0;
				lod.modelSurfs = allocator.allocate<zonetool::iw7::XModelSurfs>();
				lod.modelSurfs->name = surfs->name;
				lod.partBits[0] = static_cast<int>(0x80000000u);
				for (auto k = 1; k < 6; k++)
				{
					model->lodInfo[k].dist = 1000000.0f;
				}
				model->numsurfs = static_cast<unsigned char>(materials.size());

				auto radius_sq = 0.0f;
				for (auto c = 0; c < 3; c++)
				{
					const auto r = std::max(std::fabs(mins[c]), std::fabs(maxs[c]));
					radius_sq += r * r;
				}
				model->boneInfo = allocator.allocate_array<zonetool::iw7::XBoneInfo>(1);
				compute(&model->boneInfo[0].bounds, mins, maxs);
				model->boneInfo[0].radiusSquared = radius_sq;
				model->radius = std::sqrt(radius_sq);
				compute(&model->bounds, mins, maxs);

				model->collLod = 0xFF;
				model->flags = 0x40; // stock value
				model->hasLods = 1;
				model->shadowCutoffLod = 6;
				model->characterCollBoundsType = 1;
				model->unknownIndex = 0xFF;
				model->unknownIndex2 = 0xFF;
				zonetool::iw7::xmodel::dump(model);
			}
			ZONETOOL_INFO("shadow proxies: %zu models written (%zu alpha tested materials copied shadow-only, %u of them missing; "
				"opaque vertices welded: %zu vertices of %zu)", current_plan.proxies.size(), alpha_copies.size() - copies_missing,
				copies_missing, vertices_after, vertices_before);
		}

		bool only_clustered(const XModel* model)
		{
			return planned_world && !models_left_single.contains(model);
		}

		std::vector<std::string> model_names()
		{
			std::vector<std::string> out;
			for (const auto& group : current_plan.clusters)
			{
				out.push_back(group.name);
			}
			for (const auto& proxy : current_plan.proxies)
			{
				out.push_back(proxy.name);
			}
			return out;
		}

		void clear()
		{
			current_plan = {};
			planned_world = nullptr;
			models_left_single.clear();
			model_infos.clear();
			caster_kinds.clear();
		}
	}
}
