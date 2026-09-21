#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "fxeffectdef.hpp"

#include "zonetool/iw7/assets/fxeffectdef.hpp"

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace fxeffectdef
		{
			namespace
			{
				// referenced assets keep non null pointers into memory that is no longer
				// mapped, so nothing here is dereferenced without checking it first
				bool is_readable(const void* ptr, std::size_t size)
				{
					if (!ptr)
					{
						return false;
					}

					MEMORY_BASIC_INFORMATION mbi{};
					if (!VirtualQuery(ptr, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT)
					{
						return false;
					}

					constexpr auto readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
						PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;

					if ((mbi.Protect & readable) == 0 || (mbi.Protect & PAGE_GUARD) != 0)
					{
						return false;
					}

					const auto* end = static_cast<const char*>(mbi.BaseAddress) + mbi.RegionSize;
					return static_cast<const char*>(ptr) + size <= end;
				}

				const char* safe_name(const char* name)
				{
					if (!is_readable(name, 1))
					{
						return nullptr;
					}

					MEMORY_BASIC_INFORMATION mbi{};
					VirtualQuery(name, &mbi, sizeof(mbi));

					const auto* end = static_cast<const char*>(mbi.BaseAddress) + mbi.RegionSize;
					for (const auto* it = name; it < end && (it - name) < 256; it++)
					{
						const auto c = static_cast<unsigned char>(*it);
						if (!c)
						{
							return it > name ? name : nullptr;
						}

						if (c < 0x20 || c > 0x7E)
						{
							return nullptr;
						}
					}

					return nullptr;
				}

				// every asset in both games keeps its name in the first field
				const char* asset_name(const void* asset)
				{
					if (!is_readable(asset, sizeof(void*)))
					{
						return nullptr;
					}

					return safe_name(*static_cast<const char* const*>(asset));
				}


				// t7 element types line up with iw7 for everything the zombie effects
				// use, these are the ones the iw7 dumper branches on
				enum elem_type
				{
					ELEM_TYPE_MODEL = 9,
					ELEM_TYPE_SOUND = 12,
					ELEM_TYPE_DECAL = 13,
					ELEM_TYPE_RUNNER = 14,
				};

				// materials are dumped under a flattened name, the effect has to ask for
				// them the same way the material converter writes them
				const char* convert_material_name(const char* name, utils::memory::allocator& allocator)
				{
					std::string clean = name;

					const auto pos = clean.find_last_of("/\\");
					const auto effect = clean.starts_with("ei/") || clean.starts_with("ec/") ||
						clean.starts_with("el/") || clean.starts_with("vd/") ||
						clean.starts_with("vdd/") || clean.starts_with("gfx_");

					if (pos != std::string::npos)
					{
						clean = clean.substr(pos + 1);
					}

					for (auto i = 0u; i < clean.size(); i++)
					{
						switch (clean[i])
						{
						case '*':
						case '|':
						case ':':
						case '?':
						case '<':
						case '>':
						case '\"':
							clean[i] = '_';
							break;
						}
					}

					return allocator.duplicate_string((effect ? "el/"s : "mo/"s) + clean);
				}

				template <typename T>
				T* alias_asset(const void* asset, const char* name, utils::memory::allocator& allocator)
				{
					if (!asset || !name)
					{
						return nullptr;
					}

					// every asset in both games keeps its name first, so the effect only
					// needs a stand in carrying the converted name
					const auto stub = allocator.allocate<T>();
					stub->name = name;

					return stub;
				}

				void convert_visuals(FxElemDef* def, FxElemVisuals* visuals,
					zonetool::iw7::FxElemVisuals* new_visuals, utils::memory::allocator& allocator)
				{
					switch (def->elemType)
					{
					case ELEM_TYPE_MODEL:
					{
						new_visuals->model = alias_asset<zonetool::iw7::XModel>(
							visuals->model, asset_name(visuals->model), allocator);
						break;
					}
					case ELEM_TYPE_RUNNER:
					{
						new_visuals->effectDef.handle = alias_asset<zonetool::iw7::FxEffectDef>(
							visuals->effectDef.handle, asset_name(visuals->effectDef.handle), allocator);
						break;
					}
					case ELEM_TYPE_SOUND:
						new_visuals->soundName = safe_name(visuals->soundName.sound);
						break;
					default:
					{
						const auto* material = asset_name(visuals->material);
						new_visuals->material = alias_asset<zonetool::iw7::Material>(visuals->material,
							material ? convert_material_name(material, allocator) : nullptr, allocator);
						break;
					}
					}
				}

				zonetool::iw7::FxElemDef convert_elem(FxElemDef* def, utils::memory::allocator& allocator)
				{
					zonetool::iw7::FxElemDef new_def{};

					new_def.flags = def->flags;
					new_def.flags2 = def->extraFlags;

					std::memcpy(&new_def.spawn, &def->spawn, sizeof(new_def.spawn));
					std::memcpy(&new_def.spawnRange, &def->spawnRange, sizeof(new_def.spawnRange));
					std::memcpy(&new_def.fadeInRange, &def->fadeInRange, sizeof(new_def.fadeInRange));
					std::memcpy(&new_def.fadeOutRange, &def->fadeOutRange, sizeof(new_def.fadeOutRange));
					std::memcpy(&new_def.spawnDelayMsec, &def->spawnDelayMsec, sizeof(new_def.spawnDelayMsec));
					std::memcpy(&new_def.lifeSpanMsec, &def->lifeSpanMsec, sizeof(new_def.lifeSpanMsec));
					std::memcpy(new_def.spawnOrigin, def->spawnOrigin, sizeof(new_def.spawnOrigin));
					std::memcpy(&new_def.spawnOffsetRadius, &def->spawnOffsetRadius, sizeof(new_def.spawnOffsetRadius));
					std::memcpy(&new_def.spawnOffsetHeight, &def->spawnOffsetHeight, sizeof(new_def.spawnOffsetHeight));
					std::memcpy(new_def.spawnAngles, def->spawnAngles, sizeof(new_def.spawnAngles));
					std::memcpy(new_def.angularVelocity, def->angularVelocity, sizeof(new_def.angularVelocity));
					std::memcpy(&new_def.initialRotation, &def->initialRotation, sizeof(new_def.initialRotation));
					std::memcpy(&new_def.gravity, &def->gravity, sizeof(new_def.gravity));
					std::memcpy(&new_def.reflectionFactor, &def->reflectionFactor, sizeof(new_def.reflectionFactor));
					std::memcpy(&new_def.emitDist, &def->emitDist, sizeof(new_def.emitDist));
					std::memcpy(&new_def.emitDistVariance, &def->emitDistVariance, sizeof(new_def.emitDistVariance));

					new_def.spawnFrustumCullRadius = def->spawnFrustumCullRadius;

					new_def.atlas.behavior = def->atlas.behavior;
					new_def.atlas.index = def->atlas.index;
					new_def.atlas.fps = def->atlas.fps;
					new_def.atlas.loopCount = def->atlas.loopCount;
					new_def.atlas.colIndexBits = def->atlas.colIndexBits;
					new_def.atlas.rowIndexBits = def->atlas.rowIndexBits;
					new_def.atlas.entryCount = def->atlas.indexRange;

					new_def.elemType = static_cast<zonetool::iw7::FxElemType>(def->elemType);
					new_def.visualCount = def->visualCount;
					new_def.velIntervalCount = def->velIntervalCount;
					new_def.visStateIntervalCount = def->visStateIntervalCount;

					// the sample structs are only forward declared upstream so iw7's are used,
					// which means the source has to be checked for the size iw7 will read back
					// out of it, and an element that cannot supply them gets a blank one
					const auto vel_count = def->velIntervalCount + 1;
					if (is_readable(def->velSamples, sizeof(zonetool::iw7::FxElemVelStateSample) * vel_count))
					{
						new_def.velSamples = reinterpret_cast<zonetool::iw7::FxElemVelStateSample*>(def->velSamples);
					}
					else
					{
						new_def.velIntervalCount = 0;
						new_def.velSamples = allocator.allocate_array<zonetool::iw7::FxElemVelStateSample>(1);
					}

					const auto vis_count = def->visStateIntervalCount + 1;
					if (is_readable(def->visSamples, sizeof(zonetool::iw7::FxElemVisStateSample) * vis_count))
					{
						new_def.visSamples = reinterpret_cast<zonetool::iw7::FxElemVisStateSample*>(def->visSamples);
					}
					else
					{
						new_def.visStateIntervalCount = 0;
						new_def.visSamples = allocator.allocate_array<zonetool::iw7::FxElemVisStateSample>(1);
					}

					// t7 stores the collision box as its corners, iw7 as a centre and a
					// half size
					for (auto i = 0; i < 3; i++)
					{
						new_def.collBounds.midPoint[i] = (def->collMins[i] + def->collMaxs[i]) * 0.5f;
						new_def.collBounds.halfSize[i] = (def->collMaxs[i] - def->collMins[i]) * 0.5f;
					}

					new_def.effectOnImpact.handle = alias_asset<zonetool::iw7::FxEffectDef>(
						def->effectOnImpact.handle, asset_name(def->effectOnImpact.handle), allocator);
					new_def.effectOnDeath.handle = alias_asset<zonetool::iw7::FxEffectDef>(
						def->effectOnDeath.handle, asset_name(def->effectOnDeath.handle), allocator);
					new_def.effectEmitted.handle = alias_asset<zonetool::iw7::FxEffectDef>(
						def->effectEmitted.handle, asset_name(def->effectEmitted.handle), allocator);

					new_def.sortOrder = def->displacement;
					new_def.lightingFrac = def->lightingFrac;
					new_def.litUnlitBlendFactor = 1.0f;

					if (def->elemType == ELEM_TYPE_DECAL)
					{
						if (def->visuals.markArray)
						{
							const auto marks = allocator.allocate_array<zonetool::iw7::FxElemMarkVisuals>(
								def->visualCount);

							for (auto i = 0; i < def->visualCount; i++)
							{
								for (auto m = 0; m < 2; m++)
								{
									auto* material = def->visuals.markArray[i].materials[m];
									const auto* mat_name = asset_name(material);
									marks[i].materials[m] = alias_asset<zonetool::iw7::Material>(material,
										mat_name ? convert_material_name(mat_name, allocator) : nullptr,
										allocator);
								}
							}

							new_def.visuals.markArray = marks;
						}
					}
					else if (def->visualCount > 1 && is_readable(def->visuals.array,
						sizeof(FxElemVisuals) * def->visualCount))
					{
						const auto array = allocator.allocate_array<zonetool::iw7::FxElemVisuals>(
							def->visualCount);

						for (auto i = 0; i < def->visualCount; i++)
						{
							convert_visuals(def, &def->visuals.array[i], &array[i], allocator);
						}

						new_def.visuals.array = array;
					}
					else
					{
						// the iw7 dumper only writes the visuals array when the union pointer is
						// set, but its reader always expects one when visualCount is above 1, so
						// falling back to a single instance has to drop the count with it
						new_def.visualCount = 1;
						convert_visuals(def, &def->visuals.instance, &new_def.visuals.instance, allocator);
					}

					// the extended data is laid out differently per element type and is
					// not converted, dropping it leaves the element without its trail or
					// light rather than writing something wrong
					new_def.extended.unknownDef = nullptr;

					return new_def;
				}
			}

			zonetool::iw7::FxEffectDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator)
			{
				const auto new_asset = allocator.allocate<zonetool::iw7::FxEffectDef>();

				new_asset->name = safe_name(asset->name);

				// samples belong to the source asset and are written straight out
				// of it, so they have to be present before they are handed over

				COPY_VALUE(flags);
				COPY_VALUE(totalSize);
				COPY_VALUE(msecLoopingLife);
				COPY_VALUE(occlusionQueryDepthBias);
				COPY_VALUE(occlusionQueryFadeIn);
				COPY_VALUE(occlusionQueryFadeOut);

				std::memcpy(&new_asset->occlusionQueryScaleRange, &asset->occlusionQueryScaleRange,
					sizeof(new_asset->occlusionQueryScaleRange));

				new_asset->elemDefCountLooping = asset->elemDefCountLooping;
				new_asset->elemDefCountOneShot = asset->elemDefCountOneShot;
				new_asset->elemDefCountEmission = asset->elemDefCountEmission;

				// iw7 keeps a radius where t7 keeps the box it came from
				new_asset->elemMaxRadius = std::max({ asset->boundingBoxDim[0],
					asset->boundingBoxDim[1], asset->boundingBoxDim[2] }) * 0.5f;

				const auto count = new_asset->elemDefCountLooping + new_asset->elemDefCountOneShot +
					new_asset->elemDefCountEmission;

				// the iw7 dumper walks the counts, not the array, so dropping the elements
				// without clearing them walks a null pointer
				if (count <= 0 || !is_readable(asset->elemDefs, sizeof(FxElemDef)))
				{
					new_asset->elemDefs = nullptr;
					new_asset->elemDefCountLooping = 0;
					new_asset->elemDefCountOneShot = 0;
					new_asset->elemDefCountEmission = 0;
					return new_asset;
				}

				// a big element array can straddle more than one committed region, so each
				// one is checked on its own and whatever is present gets converted
				new_asset->elemDefs = allocator.allocate_array<zonetool::iw7::FxElemDef>(count);

				auto converted = 0;
				for (auto i = 0; i < count; i++)
				{
					if (!is_readable(&asset->elemDefs[i], sizeof(FxElemDef)))
					{
						break;
					}

					new_asset->elemDefs[converted++] = convert_elem(&asset->elemDefs[i], allocator);
				}

				if (converted < count)
				{
					ZONETOOL_WARNING("fx \"%s\" only had %i of its %i elements loaded",
						new_asset->name, converted, count);

					// the three counts are consecutive runs of the same array, trim from the end
					auto left = converted;
					new_asset->elemDefCountLooping = std::min(new_asset->elemDefCountLooping, left);
					left -= new_asset->elemDefCountLooping;
					new_asset->elemDefCountOneShot = std::min(new_asset->elemDefCountOneShot, left);
					left -= new_asset->elemDefCountOneShot;
					new_asset->elemDefCountEmission = std::min(new_asset->elemDefCountEmission, left);
				}

				return new_asset;
			}

			void dump(FxEffectDef* asset)
			{
				if (!is_readable(asset, sizeof(FxEffectDef)) || !safe_name(asset->name))
				{
					return;
				}

				utils::memory::allocator allocator;
				const auto converted_asset = convert(asset, allocator);
				zonetool::iw7::fx_effect_def::dump(converted_asset);
			}
		}
	}
}
