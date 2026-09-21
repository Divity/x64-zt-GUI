#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "xmodel.hpp"
#include "material.hpp"

#include "zonetool/iw7/assets/xmodel.hpp"
#include "zonetool/t7/converter/iw7/model_offset.hpp"

#include <utils/string.hpp>

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace xmodel
		{
			zonetool::iw7::XModel* convert(XModel* asset, utils::memory::allocator& allocator)
			{
				const auto new_asset = allocator.allocate<zonetool::iw7::XModel>();

				assert(asset->numBones < 256);

				if (asset->numCosmeticBones)
				{
					ZONETOOL_WARNING("model %s has cosmetic bones, this is untested and might not work!", asset->name);
					assert(asset->numBones + asset->numCosmeticBones < 256);
				}

				// T7 stores XModel meshes from the coarsest LOD to the most detailed,
				// while IW7 expects lodInfo[0] to be the most detailed. Keep the six
				// highest-detail T7 meshes and reverse them into IW7 order. Clamping
				// before choosing the source index used to discard the best LODs on
				// eight-LOD models and made the coarsest mesh render up close.
				const auto source_lod_count = asset->numLods;
				new_asset->numLods = static_cast<unsigned char>(std::min<unsigned int>(source_lod_count, 6));
				auto source_lod_index = [source_lod_count](const unsigned int iw7_lod)
				{
					return static_cast<unsigned int>(source_lod_count) - 1 - iw7_lod;
				};

				REINTERPRET_CAST_SAFE(name);

				COPY_VALUE(numBones);
				COPY_VALUE(numRootBones);
				new_asset->numsurfs = 0;
				new_asset->numReactiveMotionParts = 0;
				new_asset->numClientBones = asset->numCosmeticBones;
				new_asset->scale = asset->areaScale;
				memset(new_asset->noScalePartBits, 0, sizeof(float[8]));
				REINTERPRET_CAST_SAFE(boneNames);
				REINTERPRET_CAST_SAFE(parentList);
				REINTERPRET_CAST_SAFE(tagAngles);
				new_asset->tagAngles = allocator.allocate_array<zonetool::iw7::XModelAngle>(asset->numBones + asset->numCosmeticBones - asset->numRootBones);
				for (auto i = 0; i < asset->numBones + asset->numCosmeticBones - asset->numRootBones; i++)
				{
					new_asset->tagAngles[i].x = QuatInt16::ToInt16(half_float::half_to_float(asset->tagAngles[i].x));
					new_asset->tagAngles[i].y = QuatInt16::ToInt16(half_float::half_to_float(asset->tagAngles[i].y));
					new_asset->tagAngles[i].z = QuatInt16::ToInt16(half_float::half_to_float(asset->tagAngles[i].z));
					new_asset->tagAngles[i].base = QuatInt16::ToInt16(half_float::half_to_float(asset->tagAngles[i].w));
				}
				new_asset->tagPositions = allocator.allocate_array<zonetool::iw7::XModelTagPos>(asset->numBones + asset->numCosmeticBones - asset->numRootBones);
				for (auto i = 0; i < asset->numBones + asset->numCosmeticBones - asset->numRootBones; i++)
				{
					new_asset->tagPositions[i].x = asset->tagPositions[i].x;
					new_asset->tagPositions[i].y = asset->tagPositions[i].y;
					new_asset->tagPositions[i].z = asset->tagPositions[i].z;
				}
				REINTERPRET_CAST_SAFE(partClassification);
				REINTERPRET_CAST_SAFE(baseMat);

				// baseMat is world-space per bone. If the mesh was re-authored around
				// j_gun then the bones have to follow it, or tags like tag_flash stay
				// behind and the model reads as split.
				const auto& offset = model_offset::get(asset->name ? asset->name : "");
				if (offset.valid)
				{
					const auto bone_count = asset->numBones + asset->numCosmeticBones;
					new_asset->baseMat = allocator.allocate_array<zonetool::iw7::DObjAnimMat>(bone_count);
					for (auto i = 0; i < bone_count; i++)
					{
						memcpy(&new_asset->baseMat[i], &asset->baseMat[i], sizeof(zonetool::iw7::DObjAnimMat));
						model_offset::apply_quat(offset, new_asset->baseMat[i].quat);
						model_offset::apply_point(offset, new_asset->baseMat[i].trans);
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

					// i made up this function, not sure how its calculated in bo3
					auto calc_lod_dist = [&]()
					{
						float constantFactor = 1000000.0f;
						// Preserve the converter's existing near-to-far distance schedule.
						// The geometry order is reversed independently below.
						return std::round(sqrtf(constantFactor / asset->averageTriArea[i]));
					};

					new_asset->lodInfo[i].dist = calc_lod_dist();

					new_asset->lodInfo[i].numsurfs = asset->meshes[source_index]->numSurfs;
					new_asset->lodInfo[i].surfIndex = 0;
					new_asset->lodInfo[i].modelSurfs = allocator.allocate<zonetool::iw7::XModelSurfs>();
					new_asset->lodInfo[i].modelSurfs->name = allocator.duplicate_string(asset->meshes[source_index]->name);
					memcpy(&new_asset->lodInfo[i].partBits, &asset->meshes[source_index]->partBits, sizeof(float[8]));
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
					// the material converter writes these under a flattened name, the model
					// has to ask for the same one or the zone falls back to the default
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

					// If an eight-LOD source used one of the two discarded coarse
					// meshes for collision, use the coarsest retained mesh instead.
					if (!collision_lod_retained && new_asset->numLods)
					{
						new_asset->collLod = new_asset->numLods - 1;
					}
				}

				new_asset->flags = 0; //asset->flags; // not used in games

				new_asset->numCollSurfs = static_cast<short>(asset->numCollSurfs);
				new_asset->collSurfs = allocator.allocate_array<zonetool::iw7::XModelCollSurf_s>(new_asset->numCollSurfs);
				for (auto i = 0; i < new_asset->numCollSurfs; i++)
				{
					compute(&new_asset->collSurfs[i].bounds, asset->collSurfs[i].mins, asset->collSurfs[i].maxs);
					new_asset->collSurfs[i].boneIdx = asset->collSurfs[i].boneIdx;
					new_asset->collSurfs[i].contents = asset->collSurfs[i].contents; // convert...
					new_asset->collSurfs[i].surfFlags = asset->collSurfs[i].surfFlags; // convert...
				}

				new_asset->contents = asset->contents; // convert...

				new_asset->boneInfo = allocator.allocate_array<zonetool::iw7::XBoneInfo>(asset->numBones + asset->numCosmeticBones);
				for (auto i = 0; i < asset->numBones + asset->numCosmeticBones; i++)
				{
					compute(&new_asset->boneInfo[i].bounds, asset->boneInfo[i].bounds[0], asset->boneInfo[i].bounds[1]);
					new_asset->boneInfo[i].radiusSquared = asset->boneInfo[i].radiusSquared;
				}

				COPY_VALUE(radius);
				compute(&new_asset->bounds, asset->mins, asset->maxs);

				new_asset->invHighMipRadius = allocator.allocate_array<unsigned short>(new_asset->numsurfs);
				for (unsigned char i = 0; i < new_asset->numsurfs; i++)
				{
					auto val = sqrt(himipInvSqRadiis[i]);
					new_asset->invHighMipRadius[i] = half_float::float_to_half(val);
				}

				new_asset->memUsage = 0;

				// t7 collmaps are not havok packfiles, so there is nothing to put in a
				// PhysicsAsset. naming one anyway makes the zone carry an asset whose
				// havokData is null, and the havok loader returns null for that and is
				// dereferenced on the way back out
				new_asset->physicsAsset = nullptr;

				new_asset->hasLods = asset->numLods ? 1 : 0;
				new_asset->shadowCutoffLod = 6;
				new_asset->characterCollBoundsType = 1; // CharCollBoundsType_Human

				new_asset->unknownIndex = 0xFF;
				new_asset->unknownIndex2 = 0xFF;

				return new_asset;
			}

			void dump(XModel* asset)
			{
				utils::memory::allocator allocator;
				const auto converted_asset = convert(asset, allocator);

				// bone names are script strings resolved inside the IW7 dumper, so a
				// rename has to be applied there rather than on the converted asset
				const auto& offset = model_offset::get(asset->name ? asset->name : "");
				if (!offset.bones.empty())
				{
					zonetool::iw7::xmodel::set_bone_name_remap(&offset.bones);
				}

				zonetool::iw7::xmodel::dump(converted_asset);

				zonetool::iw7::xmodel::set_bone_name_remap(nullptr);
			}
		}
	}
}
