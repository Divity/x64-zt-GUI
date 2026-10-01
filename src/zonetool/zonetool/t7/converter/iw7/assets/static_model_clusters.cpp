#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "static_model_clusters.hpp"

#include "comworld.hpp"
#include "material.hpp"
#include "probe_lighting.hpp"
#include "world_material.hpp"
#include "world_material_bake.hpp"
#include "world_techset_donors.hpp"
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
			// reaching them. The finest grouping whose view of every static model stays within record_budget is taken:
			// BO3 maps may show most of themselves at once, so culling cannot keep a view within
			// IW7's 16384 addressable record bytes. A member's brightness is kept either way (its vertices' self
			// visibility, dump_all), what a coarser grouping gives up is its light's colour and direction.
			struct grouping
			{
				float cell;
				float steps_per_stop;
				bool by_lights;
			};
			constexpr grouping groupings[] = {
				{ 1024.0f, 2.0f, true }, { 1024.0f, 1.0f, true }, { 1024.0f, 0.5f, true }, { 1024.0f, 0.0f, true },
				{ 2048.0f, 0.0f, true }, { 2048.0f, 0.0f, false }, { 4096.0f, 0.0f, false }, { 65536.0f, 0.0f, false },
			};
			// a quarter below IW7's 16384 for the records the estimate cannot see (one model split by draw type)
			constexpr std::size_t record_budget = 12288;
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

			// what one cluster LOD holds: per material, its vertex and triangle totals
			struct lod_budget
			{
				std::vector<const Material*> materials;
				std::vector<std::size_t> vertices;
				std::vector<std::size_t> triangles;

				bool fits(const model_info& info, const unsigned int lod) const
				{
					auto materials_after = this->materials;
					auto vertices_after = this->vertices;
					auto triangles_after = this->triangles;
					const auto& [mesh, mats] = info.meshes[lod];
					for (auto s = 0u; s < mats.size(); s++)
					{
						auto at = std::ranges::find(materials_after, mats[s]) - materials_after.begin();
						if (at == static_cast<std::ptrdiff_t>(materials_after.size()))
						{
							materials_after.push_back(mats[s]);
							vertices_after.push_back(0);
							triangles_after.push_back(0);
						}
						vertices_after[at] += info.vertices[lod][s];
						triangles_after[at] += info.triangles[lod][s];
						if (vertices_after[at] > max_vertices || triangles_after[at] > max_vertices)
						{
							return false;
						}
					}
					return materials_after.size() <= max_surfaces;
				}

				void add(const model_info& info, const unsigned int lod)
				{
					const auto& [mesh, mats] = info.meshes[lod];
					for (auto s = 0u; s < mats.size(); s++)
					{
						auto at = std::ranges::find(this->materials, mats[s]) - this->materials.begin();
						if (at == static_cast<std::ptrdiff_t>(this->materials.size()))
						{
							this->materials.push_back(mats[s]);
							this->vertices.push_back(0);
							this->triangles.push_back(0);
						}
						this->vertices[at] += info.vertices[lod][s];
						this->triangles[at] += info.triangles[lod][s];
					}
				}
			};

			// what one shadow proxy holds: its opaque surfaces (vertex and triangle totals), filled in order as dump_all
			// appends them, and one surface per alpha tested material
			struct proxy_budget
			{
				std::vector<std::array<std::size_t, 2>> opaque;
				std::vector<const Material*> alpha;
				std::vector<std::array<std::size_t, 2>> alpha_size;

				// adds a member's first LOD caster surfaces of one kind; false (nothing added) past IW7's surface or vertex limits
				bool add(const model_info& info, const caster_kind only)
				{
					auto after = *this;
					const auto& mats = info.meshes[0].second;
					for (auto s = 0u; s < mats.size(); s++)
					{
						const auto v = info.vertices[0][s], t = info.triangles[0][s];
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

			// ---- the probe lighting at each one's centre -------------------------------------------------------------
			std::vector<float> luminance(count, 0.0f);
			{
				probe_lighting::evaluator lighting(asset, 0);
				std::vector<std::vector<unsigned int>> by_volume(lighting.volume_count());
				for (auto i = 0u; i < count; i++)
				{
					if (eligible[i])
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
				if (eligible[i])
				{
					light_key[i] = light_key_of(i);
				}
			});

			// the clusters of one grouping: members of a group fill clusters in order within IW7's surface and vertex limits
			const auto fill = [&](const grouping& g, std::vector<std::vector<unsigned int>>& filled, std::vector<bool>& filled_shadow)
			{
				std::map<std::tuple<int, int, int, bool, int, std::size_t>, std::vector<unsigned int>> groups;
				for (auto i = 0u; i < count; i++)
				{
					if (!eligible[i])
					{
						continue;
					}
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					const auto step = g.steps_per_stop > 0.0f ? static_cast<int>(std::floor(std::log2(luminance[i]) * g.steps_per_stop)) : 0;
					const auto key = std::make_tuple(static_cast<int>(std::floor(src.center[0] / g.cell)),
						static_cast<int>(std::floor(src.center[1] / g.cell)), static_cast<int>(std::floor(src.center[2] / g.cell)),
						!(src.flags & t7_smodel_no_shadow), step, g.by_lights ? light_key[i] : 0);
					groups[key].push_back(i);
				}
				for (auto& [key, members] : groups)
				{
					std::vector<lod_budget> budget;
					std::vector<unsigned int> current;
					const auto close = [&]()
					{
						if (current.size() > 1)
						{
							filled.push_back(current);
							filled_shadow.push_back(std::get<3>(key));
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

			// the record bytes of one view that sees every static model (0x140DCE4C0: (2 x instances + 13) & ~3 a record,
			// a record per model and LOD, at most 128 instances): a merged or posed model is one record of one instance; an
			// unmerged model's instances can sit at every LOD it has at once
			const auto record_bytes = [&](const std::vector<std::vector<unsigned int>>& filled)
			{
				std::vector<char> merged(count, 0);
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
				std::unordered_map<const XModel*, unsigned int> singles;
				for (auto i = 0u; i < count; i++)
				{
					if (!merged[i])
					{
						singles[asset->dpvs.smodelDrawInsts[i].model]++;
					}
				}
				std::size_t bytes = (filled.size() + posed_models.size()) * 12u;
				for (const auto& [model, n] : singles)
				{
					const auto lods = std::max(1u, model ? xmodel::lod_count(model) : 1u);
					const auto records = std::max(std::min(n, lods), (n + 127u) / 128u);
					const auto base = n / records, extra = n % records;
					bytes += extra * ((2u * (base + 1u) + 13u) & ~3u) + (records - extra) * ((2u * base + 13u) & ~3u);
				}
				return bytes;
			};

			std::vector<std::vector<unsigned int>> filled;
			std::vector<bool> filled_shadow;
			const grouping* chosen = nullptr;
			std::size_t bytes = 0;
			std::string tried;
			for (const auto& g : groupings)
			{
				filled.clear();
				filled_shadow.clear();
				fill(g, filled, filled_shadow);
				bytes = record_bytes(filled);
				tried += utils::string::va(" [%.0f units, %g steps a stop, %s: %zu merged, %zu bytes]", g.cell, g.steps_per_stop,
					g.by_lights ? "by light set" : "any light set", filled.size(), bytes);
				if (bytes <= record_budget)
				{
					chosen = &g;
					break;
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
				ZONETOOL_FATAL("static model clusters: no grouping keeps a view of every static model within %zu record bytes (IW7 "
					"addresses 16384):%s", record_budget, tried.data());
			}
			std::vector<std::vector<matrix34>> filled_skin(filled.size());

			for (auto& [i, skin] : posed_models)
			{
				filled.push_back({ i });
				filled_shadow.push_back(!(asset->dpvs.smodelDrawInsts[i].flags & t7_smodel_no_shadow));
				filled_skin.push_back(std::move(skin));
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
				if (p.cluster_of[i] < 0)
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
					p.iw7_index[i] = next + c;
					if (luminance[i] > luminance[brightest])
					{
						brightest = i;
					}
				}
				std::memcpy(group.lighting_origin, asset->dpvs.smodelDrawInsts[brightest].center, sizeof(group.lighting_origin));
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
			ZONETOOL_INFO("static model clusters: %u of %u static models merged into %zu models (%.0f units cells, lighting within %g "
				"stops, %s; %zu record bytes with every static model in view, of %zu; tried%s), %zu posed ones with their pose baked in (%u "
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
				for (const auto* material : info.meshes[0].second)
				{
					if (material)
					{
						materials.insert(material);
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
				for (const auto* m : model_infos[asset->dpvs.smodelDrawInsts[i].model].meshes[0].second)
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
					const auto& mats = info.meshes[0].second;
					auto casts = false;
					for (auto s = 0u; s < mats.size(); s++)
					{
						const auto kind = mats[s] ? caster_kinds.at(mats[s]) : caster_kind::none;
						if (kind == only)
						{
							casts = true;
							(kind == caster_kind::opaque ? opaque_triangles : alpha_triangles) += info.triangles[0][s];
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
				if ((src.flags & t7_smodel_no_shadow) || p.proxied[i])
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
		}

		const plan& current()
		{
			return current_plan;
		}

		std::vector<unsigned short> remap_indices(const unsigned short* indices, const unsigned int count)
		{
			std::vector<unsigned short> out;
			out.reserve(count);
			std::unordered_set<unsigned int> seen;
			for (auto i = 0u; i < count; i++)
			{
				const auto mapped = current_plan.iw7_index.at(indices[i]);
				if (seen.insert(mapped).second)
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
				const auto lods = 1u;
				std::vector<float> dist(lods, 0.0f);
				for (const auto i : group.members)
				{
					const auto& src = asset->dpvs.smodelDrawInsts[i];
					const auto& info = model_infos[src.model];
					const float d[3] = { src.center[0] - group.origin[0], src.center[1] - group.origin[1], src.center[2] - group.origin[2] };
					const auto offset = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
					dist[0] = std::max(dist[0], xmodel::lod_dist(src.model, info.lods - 1) * src.placement.scale + offset);
				}

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
						const auto member_lod = std::min(k, info.lods - 1);
						const auto& [mesh, mats] = info.meshes[member_lod];
						const auto radii = xmodel::himip_inv_sq_radii(src.model);
						for (auto s = 0u; s < mats.size(); s++)
						{
							auto at = std::ranges::find(materials, mats[s]) - materials.begin();
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
					const auto& [mesh, mats] = info.meshes[0];
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
							if (opaque_surface == ~std::size_t{ 0 } || verts[opaque_surface].size() + info.vertices[0][s] > max_vertices
								|| faces[opaque_surface].size() + info.triangles[0][s] > max_vertices)
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
							for (auto f = faces[at].size() - info.triangles[0][s]; f < faces[at].size(); f++)
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
