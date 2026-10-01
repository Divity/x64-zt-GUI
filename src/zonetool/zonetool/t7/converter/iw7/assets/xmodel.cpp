#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "xmodel.hpp"
#include "xmodel_collision.hpp"
#include "xmodel_mesh.hpp"
#include "material.hpp"
#include "world_material.hpp"

#include "zonetool/iw7/assets/physics_asset.hpp"
#include "zonetool/iw7/assets/xmodel.hpp"
#include "zonetool/t7/converter/iw7/model_offset.hpp"

#include <utils/string.hpp>

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace xmodel
		{
			namespace
			{
				// BO3 picks a LOD by projected triangle size (0x1422E0300): LOD i holds while
				// lodAreaScale * areaScale * averageTriArea[i] * P / T > D^2, P = (w/2)(h/2) / (tan(fovX/2) tan(fovY/2)),
				// T = 4 px^2; culled once radius * sqrt(P) / D < max(lodCullScale * cullOutRadius, 1).
				// IW7 LOD distances are normalized to an 80 degree horizontal 16:9 view (0xE3DBF0); evaluated at 1920x1080.
				constexpr double lod_ref_height = 1080.0;
				constexpr double lod_ref_tan_x = 0.83909963117728; // tan(40 degrees)
				constexpr double lod_ref_tan_y = lod_ref_tan_x * 9.0 / 16.0;
				constexpr double lod_tri_size = 4.0;

				float lod_distance(const XModel* asset, const unsigned int source_lod, const bool last)
				{
					const auto width = lod_ref_height * 16.0 / 9.0;
					const auto pixels = (width * 0.5) * (lod_ref_height * 0.5) / (lod_ref_tan_x * lod_ref_tan_y);
					const auto cull = asset->radius * std::sqrt(pixels)
						/ std::max(static_cast<double>(asset->lodCullScale[0]) * asset->cullOutRadius, 1.0);
					if (last)
					{
						return static_cast<float>(cull);
					}
					const auto area = static_cast<double>(asset->lodAreaScale[0]) * asset->areaScale * asset->averageTriArea[source_lod];
					return static_cast<float>(std::min(std::sqrt(std::max(area, 0.0) * pixels / lod_tri_size) + asset->radius, cull));
				}

				// BO3's meshes run from the lowest detail up; static placements never draw the LODs above lodCap[1]
				unsigned int top_lod(const XModel* asset)
				{
					return std::min<unsigned int>(asset->numLods, asset->lodCap[1] + 1u);
				}

				// IW7 redraws a model's cached sun shadow whenever its LOD changes (0xDD0040, 0xDD03A0), so closely spaced
				// BO3 LODs cause shadow flicker. A LOD closer than lod_min_ratio times the kept one before it is merged into
				// that one (stock IW7 lod1 / lod0 10th percentile); the last one, the cull distance, stays.
				constexpr double lod_min_ratio = 1.6;

				struct kept_lod
				{
					unsigned int source; // BO3 mesh index
					float dist; // IW7 distance: where the next kept LOD takes over (the last: culled)
				};

				// the LODs the converted model keeps, most detailed first; at most 6 (IW7's), BO3's lowest past them dropped
				std::vector<kept_lod> kept_lods(const XModel* asset)
				{
					const auto top = top_lod(asset);
					const auto count = std::min(top, 6u);
					std::vector<kept_lod> all;
					for (auto i = 0u; i < count; i++)
					{
						const auto source = top - 1 - i;
						all.push_back({ source, lod_distance(asset, source, i + 1 == count) });
					}
					std::vector<kept_lod> out;
					for (auto i = 0u; i < all.size(); i++)
					{
						const auto last = i + 1 == all.size();
						if (out.empty())
						{
							out.push_back(all[i]);
						}
						else if (!last && all[i].dist < out.back().dist * lod_min_ratio)
						{
							// too close to the one before: that one covers this range too
							continue;
						}
						else
						{
							// the kept one before holds until this LOD's range starts (the merged ones' end)
							out.back().dist = all[i - 1].dist;
							out.push_back(all[i]);
						}
						if (last)
						{
							out.back().dist = all[i].dist;
						}
					}
					// the cull distance too close to the kept one before: that one is drawn to the cull instead
					if (out.size() > 1 && out.back().dist < out[out.size() - 2].dist * lod_min_ratio)
					{
						const auto cull = out.back().dist;
						out.pop_back();
						out.back().dist = cull;
					}
					return out;
				}

				// the meshes of the LODs the converted model keeps
				template <typename F>
				void for_each_kept_mesh(const XModel* asset, F&& fn)
				{
					for (const auto& lod : kept_lods(asset))
					{
						if (auto* mesh = asset->meshes[lod.source])
						{
							fn(mesh);
						}
					}
				}

				// BO3 stores a regular bone's parent as an offset back (bone - parent) and a cosmetic bone's as the parent's
				// index
				std::vector<int> bo3_parents(const XModel* asset)
				{
					const int total = asset->numBones + asset->numCosmeticBones;
					std::vector<int> parent(total, -1);
					for (int i = asset->numRootBones; i < total; i++)
					{
						const int value = asset->parentList[i - asset->numRootBones];
						parent[i] = i < asset->numBones ? i - value : value;
					}
					return parent;
				}

				// BO3's regular and cosmetic bones become IW7 regular bones in BO3's order (no stock IW7 model uses client
				// bones). IW7 counts bones in a byte, so past 255 leaf cosmetic bones are merged into their parents, least
				// vertex weight first, until 255 are left.
				xmodel_mesh::bone_map map_bones(XModel* asset)
				{
					const int total = asset->numBones + asset->numCosmeticBones;
					if (total <= 255)
					{
						return {};
					}
					const auto parent = bo3_parents(asset);

					// the weight each bone carries over the meshes the converted model keeps
					std::vector<double> weight(total, 0.0);
					for_each_kept_mesh(asset, [&](XModelMesh* mesh)
					{
						xmodel_mesh::for_each_vertex(mesh, [&](const float*, const xmodel_mesh::vertex_bones& v)
						{
							for (auto k = 0u; k < v.count; k++)
							{
								if (v.bone[k] < total)
								{
									weight[v.bone[k]] += v.weight[k];
								}
							}
						});
					});

					std::vector<int> children(total, 0);
					for (auto i = 0; i < total; i++)
					{
						if (parent[i] >= 0)
						{
							children[parent[i]]++;
						}
					}
					std::vector<int> into(total, -1); // a merged bone -> the bone it was merged into
					auto left = total;
					auto unused = 0;
					auto moved = 0.0;
					std::string names;
					while (left > 255)
					{
						auto pick = -1;
						for (int i = asset->numBones; i < total; i++)
						{
							if (into[i] < 0 && !children[i] && parent[i] >= 0 && (pick < 0 || weight[i] < weight[pick]))
							{
								pick = i;
							}
						}
						if (pick < 0)
						{
							ZONETOOL_FATAL("model %s: %d bones with its cosmetic ones and no leaf cosmetic bone left to merge", asset->name, left);
						}
						into[pick] = parent[pick];
						weight[parent[pick]] += weight[pick];
						children[parent[pick]]--;
						unused += weight[pick] > 0.0 ? 0 : 1;
						moved += weight[pick];
						names += (names.empty() ? "" : ", ") + std::string(SL_ConvertToString(asset->boneNames[pick]));
						left--;
					}

					xmodel_mesh::bone_map map;
					map.to_iw7.resize(total);
					for (auto i = 0; i < total; i++)
					{
						if (into[i] < 0)
						{
							map.to_iw7[i] = static_cast<std::uint16_t>(map.to_bo3.size());
							map.to_bo3.push_back(static_cast<std::uint16_t>(i));
						}
					}
					for (auto i = 0; i < total; i++)
					{
						auto target = i;
						while (into[target] >= 0)
						{
							target = into[target];
						}
						map.to_iw7[i] = map.to_iw7[target];
					}
					ZONETOOL_INFO("model %s: %d bones with its cosmetic ones, more than IW7's 255: %d leaf cosmetic bones merged into their "
						"parents (%d that no vertex uses; the rest carried %.1f vertices' worth of weight): %s", asset->name, total,
						total - left, unused, moved, names.data());
					return map;
				}

				// the bounds of what an IW7 bone carries that BO3's boneInfo does not hold, in the bone's space: the vertices whose
				// heaviest bone was a cosmetic one (BO3's boneInfo holds regular bones only) or became another in the merge
				struct bone_extent
				{
					float min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
					float max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
					float radius_squared = 0.0f;

					bool empty() const
					{
						return this->min[0] > this->max[0];
					}
				};

				std::vector<bone_extent> cosmetic_extents(XModel* asset, const xmodel_mesh::bone_map& bones, const int bone_count)
				{
					std::vector<bone_extent> out(bone_count);
					std::vector<xmodel_collision::bone_frame> frames(bone_count);
					for (auto i = 0; i < bone_count; i++)
					{
						const auto bo3 = bones.empty() ? i : bones.to_bo3[i];
						frames[i] = xmodel_collision::frame_of(reinterpret_cast<const zonetool::iw7::DObjAnimMat&>(asset->baseMat[bo3]));
					}
					for_each_kept_mesh(asset, [&](XModelMesh* mesh)
					{
						xmodel_mesh::for_each_vertex(mesh, [&](const float* xyz, const xmodel_mesh::vertex_bones& v)
						{
							const auto primary = v.bone[0];
							if (primary >= asset->numBones + asset->numCosmeticBones)
							{
								return;
							}
							// the heaviest bone once the vertex's bones are IW7's
							std::array<std::uint16_t, 4> bone{};
							std::array<float, 4> weight{};
							auto count = 0u;
							for (auto k = 0u; k < v.count; k++)
							{
								const auto b = bones(v.bone[k]);
								auto at = 0u;
								while (at < count && bone[at] != b)
								{
									at++;
								}
								if (at == count)
								{
									bone[count] = b;
									weight[count++] = 0.0f;
								}
								weight[at] += v.weight[k];
							}
							auto heaviest = 0u;
							for (auto k = 1u; k < count; k++)
							{
								heaviest = weight[k] > weight[heaviest] ? k : heaviest;
							}
							const auto target = bone[heaviest];
							if (primary < asset->numBones && bones(primary) == target)
							{
								return; // BO3's own bounds of a regular bone hold it
							}
							const auto local = xmodel_collision::to_bone(frames[target], { xyz[0], xyz[1], xyz[2] });
							auto& e = out[target];
							for (auto k = 0; k < 3; k++)
							{
								e.min[k] = std::min(e.min[k], local[k]);
								e.max[k] = std::max(e.max[k], local[k]);
							}
							e.radius_squared = std::max(e.radius_squared, local[0] * local[0] + local[1] * local[1] + local[2] * local[2]);
						});
					});
					return out;
				}

				std::unordered_set<std::string> dumped_models;
			}

			std::string iw7_name(const char* bo3_name)
			{
				if (!bo3_name)
				{
					return {};
				}
				return bo3_name[0] == '*' ? std::string("t7_brushmodel_") + (bo3_name + 1) : std::string(bo3_name);
			}

			bool dumped(const char* name)
			{
				return name && dumped_models.contains(name);
			}

			void clear()
			{
				dumped_models.clear();
			}

			zonetool::iw7::XModel* convert(XModel* asset, const xmodel_mesh::bone_map& bones, utils::memory::allocator& allocator)
			{
				const auto new_asset = allocator.allocate<zonetool::iw7::XModel>();

				// every BO3 bone, regular and cosmetic, as an IW7 regular bone (map_bones)
				const int bone_count = bones.empty() ? asset->numBones + asset->numCosmeticBones : static_cast<int>(bones.to_bo3.size());
				if (bone_count > 255)
				{
					ZONETOOL_FATAL("model %s has %d bones with its cosmetic ones; IW7 counts its bones in a byte", asset->name, bone_count);
				}
				const auto bo3_bone = [&](const int bone)
				{
					return bones.empty() ? bone : static_cast<int>(bones.to_bo3[bone]);
				};

				const auto kept = kept_lods(asset);
				new_asset->numLods = static_cast<unsigned char>(kept.size());
				auto source_lod_index = [&kept](const unsigned int iw7_lod)
				{
					return kept[iw7_lod].source;
				};

				REINTERPRET_CAST_SAFE(name);
				if (asset->name && asset->name[0] == '*')
				{
					new_asset->name = allocator.duplicate_string(iw7_name(asset->name));
				}

				new_asset->numBones = static_cast<unsigned char>(bone_count);
				COPY_VALUE(numRootBones);
				new_asset->numsurfs = 0;
				new_asset->numReactiveMotionParts = 0;
				new_asset->numClientBones = 0;
				// stock value; BO3's areaScale is a LOD term (used in the distances below)
				new_asset->scale = 1.0f;
				memset(new_asset->noScalePartBits, 0, sizeof(float[8]));
				REINTERPRET_CAST_SAFE(boneNames);
				REINTERPRET_CAST_SAFE(parentList);
				if (asset->numCosmeticBones)
				{
					// the bones IW7 keeps, every parent as an offset back (bo3_parents)
					const auto parent = bo3_parents(asset);
					new_asset->boneNames = allocator.allocate_array<zonetool::iw7::scr_string_t>(bone_count);
					new_asset->parentList = allocator.allocate_array<unsigned char>(bone_count - asset->numRootBones);
					for (auto i = 0; i < bone_count; i++)
					{
						const auto b = bo3_bone(i);
						new_asset->boneNames[i] = static_cast<zonetool::iw7::scr_string_t>(asset->boneNames[b]);
						if (i >= asset->numRootBones)
						{
							const auto back = i - static_cast<int>(bones(static_cast<std::uint16_t>(parent[b])));
							if (back < 1 || back > 255)
							{
								ZONETOOL_FATAL("model %s: bone %d's parent is %d bones back", asset->name, i, back);
							}
							new_asset->parentList[i - asset->numRootBones] = static_cast<unsigned char>(back);
						}
					}
				}
				new_asset->tagAngles = allocator.allocate_array<zonetool::iw7::XModelAngle>(bone_count - asset->numRootBones);
				new_asset->tagPositions = allocator.allocate_array<zonetool::iw7::XModelTagPos>(bone_count - asset->numRootBones);
				for (auto i = 0; i < bone_count - asset->numRootBones; i++)
				{
					const auto b = bo3_bone(i + asset->numRootBones) - asset->numRootBones;
					new_asset->tagAngles[i].x = QuatInt16::ToInt16(half_float::half_to_float(asset->tagAngles[b].x));
					new_asset->tagAngles[i].y = QuatInt16::ToInt16(half_float::half_to_float(asset->tagAngles[b].y));
					new_asset->tagAngles[i].z = QuatInt16::ToInt16(half_float::half_to_float(asset->tagAngles[b].z));
					new_asset->tagAngles[i].base = QuatInt16::ToInt16(half_float::half_to_float(asset->tagAngles[b].w));
					new_asset->tagPositions[i].x = asset->tagPositions[b].x;
					new_asset->tagPositions[i].y = asset->tagPositions[b].y;
					new_asset->tagPositions[i].z = asset->tagPositions[b].z;
				}
				REINTERPRET_CAST_SAFE(partClassification);
				if (asset->numCosmeticBones && asset->partClassification)
				{
					// (BO3's has its regular bones' entries; the cosmetic ones get 0, no hit location)
					new_asset->partClassification = allocator.allocate_array<unsigned char>(bone_count);
					for (auto i = 0; i < bone_count; i++)
					{
						const auto b = bo3_bone(i);
						new_asset->partClassification[i] = b < asset->numBones ? asset->partClassification[b] : 0;
					}
				}
				REINTERPRET_CAST_SAFE(baseMat);

				const auto& offset = model_offset::get(asset->name ? asset->name : "");
				if (offset.valid || !bones.empty())
				{
					new_asset->baseMat = allocator.allocate_array<zonetool::iw7::DObjAnimMat>(bone_count);
					for (auto i = 0; i < bone_count; i++)
					{
						memcpy(&new_asset->baseMat[i], &asset->baseMat[bo3_bone(i)], sizeof(zonetool::iw7::DObjAnimMat));
						if (offset.valid)
						{
							model_offset::apply_quat(offset, new_asset->baseMat[i].quat);
							model_offset::apply_point(offset, new_asset->baseMat[i].trans);
						}
					}
				}
				new_asset->reactiveMotionParts = nullptr;
				new_asset->reactiveMotionTweaks = nullptr;

				for (auto i = 0; i < 6; i++)
				{
					new_asset->lodInfo[i].dist = 1000000.0f;
				}

				for (auto i = 0; i < new_asset->numLods; i++)
				{
					const auto source_index = source_lod_index(i);

					new_asset->lodInfo[i].dist = kept[i].dist;

					new_asset->lodInfo[i].numsurfs = asset->meshes[source_index]->numSurfs;
					new_asset->lodInfo[i].surfIndex = 0;
					new_asset->lodInfo[i].modelSurfs = allocator.allocate<zonetool::iw7::XModelSurfs>();
					new_asset->lodInfo[i].modelSurfs->name = allocator.duplicate_string(iw7_name(asset->meshes[source_index]->name));
					// the bones the LOD's vertices use, as its mesh's (xmodel_mesh part_bits)
					const auto bits = xmodel_mesh::part_bits(asset->meshes[source_index], bones);
					memcpy(new_asset->lodInfo[i].partBits, bits.data(), sizeof(new_asset->lodInfo[i].partBits));
				}

				std::vector<Material*> materials;
				std::vector<vec_t> himipInvSqRadiis;
				unsigned short surfIndex = 0;

				for (auto i = 0; i < new_asset->numLods; i++)
				{
					const auto source_index = source_lod_index(i);
					auto mesh_material = asset->meshMaterials[source_index];

					for (auto j = 0; j < mesh_material.numMaterials; j++)
					{
						auto* material = mesh_material.materials[j];
						materials.push_back(material);
						himipInvSqRadiis.push_back(mesh_material.himipInvSqRadii[j]);
					}

					new_asset->lodInfo[i].surfIndex = surfIndex;
					surfIndex += mesh_material.numMaterials;
				}

				new_asset->numsurfs = static_cast<unsigned char>(materials.size());
				new_asset->materialHandles = allocator.allocate_array<zonetool::iw7::Material*>(materials.size());
				for (auto i = 0; i < materials.size(); i++)
				{
					const auto converted = material::get_converted_name(materials[i]);
					const auto stub = allocator.allocate<zonetool::iw7::Material>();
					stub->name = allocator.duplicate_string(converted);
					new_asset->materialHandles[i] = stub;
				}

				new_asset->collLod = 0xFFui8;
				if (asset->collLod)
				{
					bool collision_lod_retained = false;
					for (char i = 0; i < new_asset->numLods; i++)
					{
						if (*asset->collLod == asset->meshes[source_lod_index(i)])
						{
							new_asset->collLod = i;
							collision_lod_retained = true;
							break;
						}
					}

					if (!collision_lod_retained && new_asset->numLods)
					{
						new_asset->collLod = new_asset->numLods - 1;
					}
				}

				new_asset->flags = 0x40; // stock value

				xmodel_collision::convert(asset, new_asset, bones, allocator);

				// a regular bone's bounds are BO3's, grown over what the merge gave it; a cosmetic bone's are its vertices'
				// (cosmetic_extents)
				std::vector<bone_extent> extents;
				if (asset->numCosmeticBones)
				{
					extents = cosmetic_extents(asset, bones, bone_count);
				}
				new_asset->boneInfo = allocator.allocate_array<zonetool::iw7::XBoneInfo>(bone_count);
				for (auto i = 0; i < bone_count; i++)
				{
					const auto b = bo3_bone(i);
					bone_extent e{};
					if (b < asset->numBones)
					{
						for (auto k = 0; k < 3; k++)
						{
							e.min[k] = asset->boneInfo[b].bounds[0][k];
							e.max[k] = asset->boneInfo[b].bounds[1][k];
						}
						e.radius_squared = asset->boneInfo[b].radiusSquared;
					}
					if (!extents.empty() && !extents[i].empty())
					{
						for (auto k = 0; k < 3; k++)
						{
							e.min[k] = std::min(e.min[k], extents[i].min[k]);
							e.max[k] = std::max(e.max[k], extents[i].max[k]);
						}
						e.radius_squared = std::max(e.radius_squared, extents[i].radius_squared);
					}
					if (e.empty())
					{
						e = {};
						for (auto k = 0; k < 3; k++)
						{
							e.min[k] = e.max[k] = 0.0f;
						}
					}
					compute(&new_asset->boneInfo[i].bounds, e.min, e.max);
					new_asset->boneInfo[i].radiusSquared = e.radius_squared;
				}

				COPY_VALUE(radius);
				compute(&new_asset->bounds, asset->mins, asset->maxs);

				new_asset->invHighMipRadius = allocator.allocate_array<unsigned short>(new_asset->numsurfs);
				for (unsigned char i = 0; i < new_asset->numsurfs; i++)
				{
					new_asset->invHighMipRadius[i] = inv_high_mip_radius(himipInvSqRadiis[i]);
				}

				new_asset->memUsage = 0;

				new_asset->hasLods = asset->numLods ? 1 : 0;
				new_asset->shadowCutoffLod = 6;
				new_asset->characterCollBoundsType = 1; // CharCollBoundsType_Human

				new_asset->unknownIndex = 0xFF;
				new_asset->unknownIndex2 = 0xFF;

				return new_asset;
			}

			std::vector<const Material*> materials(const XModel* asset)
			{
				std::vector<const Material*> out;
				for (const auto& kept : kept_lods(asset))
				{
					const auto& lod = asset->meshMaterials[kept.source];
					for (auto m = 0u; m < lod.numMaterials; m++)
					{
						if (lod.materials[m])
						{
							out.push_back(lod.materials[m]);
						}
					}
				}
				return out;
			}

			unsigned int lod_count(const XModel* asset)
			{
				return static_cast<unsigned int>(kept_lods(asset).size());
			}

			float lod_dist(const XModel* asset, const unsigned int lod)
			{
				return kept_lods(asset)[lod].dist;
			}

			std::vector<std::vector<float>> himip_inv_sq_radii(const XModel* asset)
			{
				std::vector<std::vector<float>> out;
				for (const auto& kept : kept_lods(asset))
				{
					const auto& lod = asset->meshMaterials[kept.source];
					out.emplace_back(lod.himipInvSqRadii, lod.himipInvSqRadii + lod.numMaterials);
				}
				return out;
			}

			std::vector<std::pair<XModelMesh*, std::vector<const Material*>>> lod_meshes(const XModel* asset)
			{
				std::vector<std::pair<XModelMesh*, std::vector<const Material*>>> out;
				for (const auto& kept : kept_lods(asset))
				{
					auto* mesh = asset->meshes[kept.source];
					const auto& lod = asset->meshMaterials[kept.source];
					if (!mesh || lod.numMaterials != mesh->numSurfs)
					{
						continue; // dump() stops on a count mismatch
					}
					out.emplace_back(mesh, std::vector<const Material*>(lod.materials, lod.materials + lod.numMaterials));
				}
				return out;
			}

			void dump(XModel* asset)
			{
				utils::memory::allocator allocator;
				const auto bones = map_bones(asset);
				const auto converted_asset = convert(asset, bones, allocator);

				// the meshes of the LODs the model keeps first, each surface's texture coordinates transformed for its
				// material's baked textures (and its vertex rgb white where the material takes only its vertex alpha): a mesh
				// another model drew with other transforms is written as a copy, which this model's LOD then names
				const auto kept = kept_lods(asset);
				for (auto i = 0u; i < converted_asset->numLods; i++)
				{
					const auto source = kept[i].source;
					auto* mesh = asset->meshes[source];
					if (!mesh)
					{
						continue;
					}
					const auto& lod_materials = asset->meshMaterials[source];
					if (lod_materials.numMaterials != mesh->numSurfs)
					{
						ZONETOOL_FATAL("model %s: LOD %u has %u materials for %u surfaces", asset->name, source,
							lod_materials.numMaterials, mesh->numSurfs);
					}
					xmodel_mesh::surface_transforms transforms(mesh->numSurfs);
					for (auto s = 0u; s < mesh->numSurfs; s++)
					{
						if (lod_materials.materials[s])
						{
							const auto& info = world_material::get_model(lod_materials.materials[s]);
							transforms[s] = { { info.uv_origin[0], info.uv_origin[1] }, { info.uv_span[0], info.uv_span[1] }, info.vertex_alpha };
						}
					}
					converted_asset->lodInfo[i].modelSurfs->name = allocator.duplicate_string(xmodel_mesh::dump(mesh, transforms, bones));
				}

				const auto& offset = model_offset::get(asset->name ? asset->name : "");
				if (!offset.bones.empty())
				{
					zonetool::iw7::xmodel::set_bone_name_remap(&offset.bones);
				}

				zonetool::iw7::xmodel::dump(converted_asset);
				if (converted_asset->physicsAsset)
				{
					zonetool::iw7::physics_asset::dump(converted_asset->physicsAsset);
				}

				zonetool::iw7::xmodel::set_bone_name_remap(nullptr);
				if (asset->name)
				{
					dumped_models.insert(asset->name);
				}
			}
		}
	}
}
