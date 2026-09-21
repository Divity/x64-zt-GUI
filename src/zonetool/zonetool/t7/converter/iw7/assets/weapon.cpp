#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "zonetool/t7/boiii_weapondef.hpp"
#include "weapon.hpp"
#include "weapon_template.hpp"

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace weapon
		{
			namespace
			{
			// confirmed against the definitions of several weapons, the computed
			// WeaponDef offsets from the supplied 0x1900 BOIII layout do not apply to
			// this retail executable. Keep these measured live offsets separate from
			// boiii::WeaponDef and snapshot the whole definition for layout research.
			namespace def
			{
				constexpr std::size_t projectileModel = 0x1010;
				constexpr std::size_t projExplosionEffect = 0x1028;
				constexpr std::size_t projTrailEffect = 0x11B8;
			}

			// t7 names every view anim after the action it plays, which is enough to
				// place it in the iw7 slot table, longest suffix first so that a name
				// like vm_x_ads_fire is not taken for a plain fire
				struct anim_slot
				{
					const char* suffix;
					const char* slot;
				};

				const anim_slot anim_slots[] =
				{
					{ "_ads_base_down", "WEAP_ANIM_ADS_DOWN" },
					{ "_ads_base_up", "WEAP_ANIM_ADS_UP" },
					{ "_pullout_quick", "WEAP_ANIM_QUICK_RAISE" },
					{ "_putaway_quick", "WEAP_ANIM_QUICK_DROP" },
					{ "_reload_empty", "WEAP_ANIM_RELOAD_EMPTY" },
					{ "_empty_raise", "WEAP_ANIM_EMPTY_RAISE" },
					{ "_first_raise", "WEAP_ANIM_FIRST_RAISE" },
					{ "_empty_drop", "WEAP_ANIM_EMPTY_DROP" },
					{ "_empty_idle", "WEAP_ANIM_EMPTY_IDLE" },
					{ "_sprint_loop", "WEAP_ANIM_SPRINT_LOOP" },
					{ "_melee_miss", "WEAP_ANIM_MELEE_MISS" },
					{ "_sprint_in", "WEAP_ANIM_SPRINT_IN" },
					{ "_jump_land", "WEAP_ANIM_ADDITIVE_JUMP_LAND" },
					{ "_crawl_in", "WEAP_ANIM_ADDITIVE_CRAWL_IN" },
					{ "_sprint_out", "WEAP_ANIM_SPRINT_OUT" },
					{ "_ads_fire", "WEAP_ANIM_ADS_FIRE" },
					{ "_fire_ads", "WEAP_ANIM_ADS_FIRE" },
					{ "_lastshot", "WEAP_ANIM_LASTSHOT" },
					{ "_gunbutt_swipe", "WEAP_ANIM_MELEE_SWIPE" },
					{ "_pullout", "WEAP_ANIM_RAISE" },
					{ "_putaway", "WEAP_ANIM_DROP" },
					{ "_reload", "WEAP_ANIM_RELOAD" },
					{ "_crawl_f", "WEAP_ANIM_ADDITIVE_CRAWL_LOOP" },
					{ "_crawl_l", "WEAP_ANIM_ADDITIVE_CRAWL_LOOP_LEFT" },
					{ "_crawl_r", "WEAP_ANIM_ADDITIVE_CRAWL_LOOP_RIGHT" },
					{ "_walk_f", "WEAP_ANIM_ADDITIVE_WALK" },
					{ "_melee", "WEAP_ANIM_MELEE_SWIPE" },
					{ "_jump", "WEAP_ANIM_ADDITIVE_JUMP" },
					{ "_idle", "WEAP_ANIM_IDLE" },
					{ "_fire", "WEAP_ANIM_FIRE" },
				};

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

				// the layout these offsets came from is computed, so every field is
				// checked before it is touched rather than trusting the struct size
				template <typename T>
				T read(const void* base, std::size_t off)
				{
					T value{};
					if (is_readable(static_cast<const char*>(base) + off, sizeof(T)))
					{
						std::memcpy(&value, static_cast<const char*>(base) + off, sizeof(T));
					}

					return value;
				}

				const char* read_string(const void* base, std::size_t off)
				{
					return safe_name(read<const char*>(base, off));
				}

				// every t7 asset keeps its name in the first field, so one path covers
				// the model and effect handles alike
				const char* read_asset_name(const void* base, std::size_t off)
				{
					const auto* handle = read<const void*>(base, off);
					if (!is_readable(handle, sizeof(void*)))
					{
						return nullptr;
					}

					return safe_name(*static_cast<const char* const*>(handle));
				}

				// the computed WeaponDef offsets do not survive either, so the models are
				// found by name: t7 suffixes them _view and _world
				void find_models(const void* weap, const char** view, const char** world)
				{
					for (std::size_t off = 0; off < 0x1800 && (!*view || !*world); off += sizeof(void*))
					{
						const auto* value = read<const void*>(weap, off);
						if (!is_readable(value, sizeof(void*)))
						{
							continue;
						}

						const auto* name = safe_name(read<const char*>(value, 0));
						if (!name)
						{
							continue;
						}

						const std::string model = name;

						// effect paths end in _world too, models never contain a slash
						if (model.find('/') != std::string::npos)
						{
							continue;
						}

						// upgraded variants suffix the model _view_upg and _world_upg
						if (!*view && model.find("_view") != std::string::npos)
						{
							*view = name;
						}
						else if (!*world && model.find("_world") != std::string::npos)
						{
							*world = name;
						}
					}
				}

				void set_scalar(ordered_json& data, const char* key, int value, int low, int high)
				{
					if (value >= low && value <= high)
					{
						data[key] = value;
					}
				}

				void set_string(ordered_json& data, const char* key, const char* value)
				{
					if (value && *value)
					{
						data[key] = value;
					}
				}

				void set_effect(ordered_json& data, const char* key, const char* value)
				{
					if (!value || !*value)
					{
						return;
					}

					ordered_json effect;
					// FX_COMBINED_FX, a converted t7 effect is an FxEffectDef and not a vfx
					effect["type"] = 0;
					effect["fx"] = value;
					data[key] = effect;
				}

				const char* match_slot(const std::string& anim)
				{
					for (const auto& entry : anim_slots)
					{
						if (anim.size() > std::strlen(entry.suffix) && anim.ends_with(entry.suffix))
						{
							return entry.slot;
						}
					}

					return nullptr;
				}

				// the oracle (a working iw7 weapon mod built by the zonetool author,
				// dump/raygun) leaves no slot a weapon needs empty: it reuses one anim
				// across the related slots rather than nulling them
				struct anim_fallback
				{
					const char* slot;
					const char* from;
				};

				const anim_fallback anim_fallbacks[] =
				{
					{ "WEAP_ANIM_RELOAD", "WEAP_ANIM_RELOAD_EMPTY" },
					{ "WEAP_ANIM_RELOAD_EMPTY", "WEAP_ANIM_RELOAD" },
					{ "WEAP_ANIM_FAST_RELOAD", "WEAP_ANIM_RELOAD" },
					{ "WEAP_ANIM_FAST_RELOAD_EMPTY", "WEAP_ANIM_RELOAD_EMPTY" },
					{ "WEAP_ANIM_EMPTY_IDLE", "WEAP_ANIM_IDLE" },
					{ "WEAP_ANIM_LASTSHOT", "WEAP_ANIM_FIRE" },
					{ "WEAP_ANIM_ADS_FIRE", "WEAP_ANIM_FIRE" },
					{ "WEAP_ANIM_ADS_LASTSHOT", "WEAP_ANIM_ADS_FIRE" },
					{ "WEAP_ANIM_FIRST_RAISE", "WEAP_ANIM_RAISE" },
					{ "WEAP_ANIM_QUICK_RAISE", "WEAP_ANIM_RAISE" },
					{ "WEAP_ANIM_EMPTY_RAISE", "WEAP_ANIM_RAISE" },
					{ "WEAP_ANIM_QUICK_DROP", "WEAP_ANIM_DROP" },
					{ "WEAP_ANIM_EMPTY_DROP", "WEAP_ANIM_DROP" },
				};

				// t7 weapons carry no iw7 additive or melee anims, so the oracle points
				// these at the stock viewmodel anims iw7 already ships
				struct anim_default
				{
					const char* anim;
					const char* slot;
				};

				const anim_default anim_defaults[] =
				{
					{ "viewmodel_jump_additive", "WEAP_ANIM_ADDITIVE_JUMP" },
					{ "vm_default_boost_jump", "WEAP_ANIM_ADDITIVE_JUMP_BOOST" },
					{ "viewmodel_jump_land_additive", "WEAP_ANIM_ADDITIVE_JUMP_LAND" },
					{ "viewmodel_honeybadger_walk", "WEAP_ANIM_ADDITIVE_WALK" },
					{ "viewmodel_sc2010_prone_crawl_in", "WEAP_ANIM_ADDITIVE_CRAWL_IN" },
					{ "viewmodel_sc2010_prone_crawl_f", "WEAP_ANIM_ADDITIVE_CRAWL_LOOP" },
					{ "viewmodel_sc2010_prone_crawl_l", "WEAP_ANIM_ADDITIVE_CRAWL_LOOP_LEFT" },
					{ "viewmodel_sc2010_prone_crawl_r", "WEAP_ANIM_ADDITIVE_CRAWL_LOOP_RIGHT" },
					{ "viewmodel_sc2010_prone_drop", "WEAP_ANIM_ADDITIVE_PRONE_DROP" },
					{ "vm_m200_ads_down_settle", "WEAP_ANIM_ADDITIVE_SETTLE_ADS" },
					{ "vm_default_knife_slice", "WEAP_ANIM_MELEE_SWIPE" },
					{ "vm_default_knife_slice", "WEAP_ANIM_MELEE_HIT" },
					{ "vm_default_knife_slice", "WEAP_ANIM_MELEE_FATAL" },
					{ "vm_default_knife_slice", "WEAP_ANIM_MELEE_MISS" },
					{ "vm_default_knife_slice", "WEAP_ANIM_MELEE_VICTIM_CROUCHING_HIT" },
					{ "vm_default_knife_slice", "WEAP_ANIM_MELEE_VICTIM_CROUCHING_FATAL" },
					{ "vm_default_knife_slice", "WEAP_ANIM_MELEE_VICTIM_CROUCHING_MISS" },
				};

				bool slot_empty(ordered_json& anims, const char* slot)
				{
					return anims.contains(slot) && anims[slot].get<std::string>().empty();
				}

				bool slot_filled(ordered_json& anims, const char* slot)
				{
					return anims.contains(slot) && !anims[slot].get<std::string>().empty();
				}

				void complete_anims(ordered_json& anims)
				{
					for (const auto& entry : anim_fallbacks)
					{
						if (slot_empty(anims, entry.slot) && slot_filled(anims, entry.from))
						{
							anims[entry.slot] = anims[entry.from];
						}
					}

					for (const auto& entry : anim_defaults)
					{
						if (slot_empty(anims, entry.slot))
						{
							anims[entry.slot] = entry.anim;
						}
					}
				}

				// the only attachment the oracle keeps from stock is doubletap. the
				// template's own pack-a-punch attachment and the packages its override
				// rows name belong to the template weapon, not to this one
				void clear_template_attachments(ordered_json& data, ordered_json& weap_def)
				{
					for (const auto* key : { "attachments", "attachments2", "attachments3",
						"attachments4", "attachments5", "attachments6" })
					{
						if (!data.contains(key) || !data[key].is_array())
						{
							continue;
						}

						ordered_json kept = ordered_json::array();
						for (const auto& entry : data[key])
						{
							if (entry.is_string() && entry.get<std::string>() == "doubletap")
							{
								kept.push_back(entry);
							}
						}

						data[key] = kept.empty() ? ordered_json(nullptr) : kept;
					}

					for (const auto* key : { "animOverrides", "sfxOverrides", "vfxOverrides" })
					{
						if (!weap_def.contains(key) || !weap_def[key].is_array())
						{
							continue;
						}

						for (auto& row : weap_def[key])
						{
							for (const auto* field : { "attachment1", "attachment2",
								"overrides", "overridesAlt" })
							{
								if (row.contains(field))
								{
									row[field] = "";
								}
							}
						}
					}
				}

				bool write_json(const std::string& path, const ordered_json& data)
				{
					filesystem::file file(path);
					file.open("wb");

					if (!file.get_fp())
					{
						return false;
					}

					file.write(data.dump(4));
					file.close();

					return true;
				}

				bool write_raw(const std::string& path, const void* data, const std::size_t size)
				{
					if (!is_readable(data, size))
					{
						return false;
					}

					filesystem::file file(path);
					file.open("wb");
					if (!file.get_fp())
					{
						return false;
					}

					file.write(data, size, 1);
					file.close();
					return true;
				}

				ordered_json vec3_json(const float* value)
				{
					return ordered_json::array({ value[0], value[1], value[2] });
				}

				void write_bo3_snapshot(const boiii::WeaponVariantDef* variant_asset,
					const void* weap, const std::string& name)
				{
					ordered_json audit;
					audit["asset"] = name;
					audit["layout"] = {
						{ "variantSize", sizeof(boiii::WeaponVariantDef) },
						{ "providedWeaponDefSize", sizeof(boiii::WeaponDef) },
						{ "providedLayoutMatchesRetailDefinition", false },
						{ "reason", "retail projectile handles are measured at 0x1010/0x1028/0x11B8, while the provided BOIII layout places them at 0x1250/0x1268/0x13C8" },
					};

					auto& variant = audit["variant"];
					variant["szInternalName"] = safe_name(variant_asset->szInternalName) ? variant_asset->szInternalName : "";
					variant["szModeIndependentName"] = safe_name(variant_asset->szModeIndependentName) ? variant_asset->szModeIndependentName : "";
					variant["sessionMode"] = variant_asset->sessionMode;
					variant["iVariantCount"] = variant_asset->iVariantCount;
					variant["szDisplayName"] = safe_name(variant_asset->szDisplayName) ? variant_asset->szDisplayName : "";
					variant["szAltWeaponName"] = safe_name(variant_asset->szAltWeaponName) ? variant_asset->szAltWeaponName : "";
					variant["szAttachmentUnique"] = safe_name(variant_asset->szAttachmentUnique) ? variant_asset->szAttachmentUnique : "";
					variant["stowedModelOffsets"] = vec3_json(variant_asset->stowedModelOffsets);
					variant["stowedModelRotations"] = vec3_json(variant_asset->stowedModelRotations);
					variant["altWeaponIndex"] = variant_asset->altWeaponIndex;
					variant["iAttachments"] = variant_asset->iAttachments;
					variant["bIgnoreAttachments"] = variant_asset->bIgnoreAttachments;
					variant["iClipSize"] = variant_asset->iClipSize;
					variant["iReloadTime"] = variant_asset->iReloadTime;
					variant["iReloadEmptyTime"] = variant_asset->iReloadEmptyTime;
					variant["iReloadQuickTime"] = variant_asset->iReloadQuickTime;
					variant["iReloadQuickEmptyTime"] = variant_asset->iReloadQuickEmptyTime;
					variant["iReloadSpecialComboTime"] = variant_asset->iReloadSpecialComboTime;
					variant["iReloadSpecialComboEmptyTime"] = variant_asset->iReloadSpecialComboEmptyTime;
					variant["iReloadSpecialComboQuickTime"] = variant_asset->iReloadSpecialComboQuickTime;
					variant["iReloadSpecialComboQuickEmptyTime"] = variant_asset->iReloadSpecialComboQuickEmptyTime;
					variant["iAdsTransInTime"] = variant_asset->iAdsTransInTime;
					variant["iAdsTransOutTime"] = variant_asset->iAdsTransOutTime;
					variant["iAltRaiseTime"] = variant_asset->iAltRaiseTime;
					variant["iAdsAltRaiseTime"] = variant_asset->iAdsAltRaiseTime;
					variant["meleeAssassinationStateTimeTransInTime"] = variant_asset->meleeAssassinationStateTimeTransInTime;
					variant["meleeAssassinationStateTimeTransOutTime"] = variant_asset->meleeAssassinationStateTimeTransOutTime;
					variant["szAmmoDisplayName"] = safe_name(variant_asset->szAmmoDisplayName) ? variant_asset->szAmmoDisplayName : "";
					variant["szAmmoName"] = safe_name(variant_asset->szAmmoName) ? variant_asset->szAmmoName : "";
					variant["iAmmoIndex"] = variant_asset->iAmmoIndex;
					variant["szClipName"] = safe_name(variant_asset->szClipName) ? variant_asset->szClipName : "";
					variant["iClipIndex"] = variant_asset->iClipIndex;
					variant["fAimAssistRangeAds"] = variant_asset->fAimAssistRangeAds;
					variant["fAdsSwayHorizScale"] = variant_asset->fAdsSwayHorizScale;
					variant["fAdsSwayVertScale"] = variant_asset->fAdsSwayVertScale;
					variant["fkickAlignedInputScalar"] = variant_asset->fkickAlignedInputScalar;
					variant["fkickOpposedInputScalar"] = variant_asset->fkickOpposedInputScalar;
					variant["fAdsViewKickCenterSpeed"] = variant_asset->fAdsViewKickCenterSpeed;
					variant["fHipViewKickCenterSpeed"] = variant_asset->fHipViewKickCenterSpeed;
					variant["fAdsFlinchScalar"] = variant_asset->fAdsFlinchScalar;
					variant["fAdsFiringFlinchScalar"] = variant_asset->fAdsFiringFlinchScalar;
					variant["fAdsTurnRateScalar"] = variant_asset->fAdsTurnRateScalar;
					variant["fAdsWallRunBobScalar"] = variant_asset->fAdsWallRunBobScalar;
					variant["fAdsAdditiveFallScalar"] = variant_asset->fAdsAdditiveFallScalar;
					variant["fAdsAdditiveJumpScalar"] = variant_asset->fAdsAdditiveJumpScalar;
					variant["fAdsAdditiveJumpLandScalar"] = variant_asset->fAdsAdditiveJumpLandScalar;
					variant["fAdsZoom1_focalLength"] = variant_asset->fAdsZoom1_focalLength;
					variant["fAdsZoom1_fStop"] = variant_asset->fAdsZoom1_fStop;
					variant["fAdsZoom2_focalLength"] = variant_asset->fAdsZoom2_focalLength;
					variant["fAdsZoom2_fStop"] = variant_asset->fAdsZoom2_fStop;
					variant["fAdsZoom3_focalLength"] = variant_asset->fAdsZoom3_focalLength;
					variant["fAdsZoom3_fStop"] = variant_asset->fAdsZoom3_fStop;
					variant["fAdsZoomFov1"] = variant_asset->fAdsZoomFov1;
					variant["fAdsZoomFov2"] = variant_asset->fAdsZoomFov2;
					variant["fAdsZoomFov3"] = variant_asset->fAdsZoomFov3;
					variant["fAdsZoomInFrac"] = variant_asset->fAdsZoomInFrac;
					variant["fAdsZoomOutFrac"] = variant_asset->fAdsZoomOutFrac;
					variant["fOverlayAlphaScale"] = variant_asset->fOverlayAlphaScale;
					variant["fOOPosAnimLength"] = ordered_json::array({
						variant_asset->fOOPosAnimLength[0], variant_asset->fOOPosAnimLength[1],
						variant_asset->fOOPosAnimLength[2], variant_asset->fOOPosAnimLength[3]
					});
					variant["bSilenced"] = variant_asset->bSilenced;
					variant["bDualMag"] = variant_asset->bDualMag;
					variant["bInfraRed"] = variant_asset->bInfraRed;
					variant["bTVGuided"] = variant_asset->bTVGuided;
					variant["perks"] = ordered_json::array({
						variant_asset->perks[0], variant_asset->perks[1],
						variant_asset->perks[2], variant_asset->perks[3]
					});
					variant["bAntiQuickScope"] = variant_asset->bAntiQuickScope;
					variant["dpadIconRatio"] = variant_asset->dpadIconRatio;
					variant["noAmmoOnDpadIcon"] = variant_asset->noAmmoOnDpadIcon;
					variant["ikLeftHandIdlePos"] = vec3_json(variant_asset->ikLeftHandIdlePos);
					variant["ikLeftHandOffset"] = vec3_json(variant_asset->ikLeftHandOffset);
					variant["ikLeftHandRotation"] = vec3_json(variant_asset->ikLeftHandRotation);
					variant["bUsingLeftHandProneIK"] = variant_asset->bUsingLeftHandProneIK;
					variant["ikLeftHandProneOffset"] = vec3_json(variant_asset->ikLeftHandProneOffset);
					variant["ikLeftHandProneRotation"] = vec3_json(variant_asset->ikLeftHandProneRotation);
					variant["ikLeftHandUiViewerOffset"] = vec3_json(variant_asset->ikLeftHandUiViewerOffset);
					variant["ikLeftHandUiViewerRotation"] = vec3_json(variant_asset->ikLeftHandUiViewerRotation);

					for (const auto& [key, offset] : {
						std::pair{ "overlayMaterial", offsetof(boiii::WeaponVariantDef, overlayMaterial) },
						std::pair{ "overlayMaterialLowRes", offsetof(boiii::WeaponVariantDef, overlayMaterialLowRes) },
						std::pair{ "dpadIcon", offsetof(boiii::WeaponVariantDef, dpadIcon) },
					})
					{
						const auto* value = read_asset_name(variant_asset, offset);
						variant[key] = value ? value : "";
					}

					auto& definition = audit["definition"];
					definition["measuredHandles"] = {
						{ "projectileModel", { { "offset", def::projectileModel }, { "name", read_asset_name(weap, def::projectileModel) ? read_asset_name(weap, def::projectileModel) : "" } } },
						{ "projExplosionEffect", { { "offset", def::projExplosionEffect }, { "name", read_asset_name(weap, def::projExplosionEffect) ? read_asset_name(weap, def::projExplosionEffect) : "" } } },
						{ "projTrailEffect", { { "offset", def::projTrailEffect }, { "name", read_asset_name(weap, def::projTrailEffect) ? read_asset_name(weap, def::projTrailEffect) : "" } } },
					};
					definition["pointerCandidates"] = ordered_json::array();
					for (std::size_t offset = 0; offset < sizeof(boiii::WeaponDef); offset += sizeof(void*))
					{
						const auto* pointer = read<const void*>(weap, offset);
						const auto* direct = safe_name(static_cast<const char*>(pointer));
						const char* asset_name = nullptr;
						if (!direct && is_readable(pointer, sizeof(void*)))
						{
							asset_name = safe_name(*static_cast<const char* const*>(pointer));
						}

						if (direct || asset_name)
						{
							definition["pointerCandidates"].push_back({
								{ "offset", offset },
								{ direct ? "string" : "asset", direct ? direct : asset_name },
							});
						}
					}

					const auto prefix = "weapons\\"s + name;
					write_json(prefix + ".bo3.json", audit);
					write_raw(prefix + ".bo3variant.bin", variant_asset, sizeof(*variant_asset));
					write_raw(prefix + ".bo3def.bin", weap, sizeof(boiii::WeaponDef));
				}

				int write_anim_package(const boiii::WeaponVariantDef* variant_asset, const std::string& name)
				{
					auto package = ordered_json::parse(weapon_template::anim_package());
					auto& anims = package["anims"];

					for (auto& entry : anims.items())
					{
						entry.value() = "";
					}

					const auto* table = variant_asset->szXAnims;
					auto mapped = 0;

					for (auto i = 0; table && i < 256; i++)
					{
						if (!is_readable(table + i, sizeof(void*)))
						{
							break;
						}

						const auto* anim = safe_name(table[i]);
						if (!anim)
						{
							continue;
						}

						const auto* slot = match_slot(anim);
						if (slot && anims.contains(slot) && anims[slot].get<std::string>().empty())
						{
							anims[slot] = anim;
							mapped++;
						}
					}

					complete_anims(anims);

					auto& timers = package["timers"];
					timers["iReloadTime"] = variant_asset->iReloadTime;
					timers["iReloadEmptyTime"] = variant_asset->iReloadEmptyTime;
					timers["iFastReloadTime"] = variant_asset->iReloadQuickTime;
					timers["iFastReloadEmptyTime"] = variant_asset->iReloadQuickEmptyTime;
					timers["iAltRaiseTime"] = variant_asset->iAltRaiseTime;
					timers["iAltRaiseADSTime"] = variant_asset->iAdsAltRaiseTime;

					write_json("animpkg\\"s + name + "_package.json"s, package);
					return mapped;
				}
			}

			void dump(void* asset, const char* asset_name)
			{
				if (!is_readable(asset, sizeof(boiii::WeaponVariantDef)) || !asset_name)
				{
					return;
				}

				const auto* variant_asset = static_cast<const boiii::WeaponVariantDef*>(asset);
				const auto* internal_name = safe_name(variant_asset->szInternalName);
				if (!internal_name || std::strcmp(internal_name, asset_name) != 0)
				{
					ZONETOOL_ERROR("weapon \"%s\" does not match its internal name \"%s\"",
						asset_name, internal_name ? internal_name : "");
					return;
				}

				// the definition leaves its own internal name empty, the variant is what
				// carries the name, so only its readability can be checked
				const auto* weap = static_cast<const void*>(variant_asset->weapDef);
				if (!is_readable(weap, sizeof(void*)))
				{
					ZONETOOL_ERROR("weapon \"%s\" has no definition", asset_name);
					return;
				}

				const std::string name = asset_name;

				auto data = ordered_json::parse(weapon_template::weapon());
				auto& weap_def = data["weapDef"];

				data["szInternalName"] = name;
				set_string(data, "szDisplayName", safe_name(variant_asset->szDisplayName));

				const char* view_model = nullptr;
				const char* world_model = nullptr;
				find_models(weap, &view_model, &world_model);

				set_string(weap_def, "gunXModel", view_model);
				set_string(weap_def, "defaultViewModel", view_model);
				set_string(weap_def, "worldModel", world_model);
				set_string(weap_def, "defaultWorldModel", world_model);

				weap_def["szXAnims"] = name + "_package";

				const auto* projectile = read_asset_name(weap, def::projectileModel);
				if (projectile && !std::strchr(projectile, '/'))
				{
					set_string(weap_def, "projectileModel", projectile);
				}

				set_effect(weap_def, "projExplosionEffect", read_asset_name(weap, def::projExplosionEffect));
				set_effect(weap_def, "projTrailEffect", read_asset_name(weap, def::projTrailEffect));

				data["iClipSize"] = variant_asset->iClipSize;
				data["iAdsTransInTime"] = variant_asset->iAdsTransInTime;
				data["iAdsTransOutTime"] = variant_asset->iAdsTransOutTime;
				data["fAdsZoomFov"] = variant_asset->fAdsZoomFov1;
				data["fAdsViewKickCenterSpeed"] = variant_asset->fAdsViewKickCenterSpeed;
				data["fHipViewKickCenterSpeed"] = variant_asset->fHipViewKickCenterSpeed;
				set_string(data, "szAltWeaponName", safe_name(variant_asset->szAltWeaponName));

				// the template is a real iw7 weapon, so anything of its own it still
				// names has to go: the oracle ships no complex models, no camo model
				// or material swaps, and only the stock attachments. left in place
				// these point at facemelter assets this zone never carries
				for (const auto* key : { "complexGunXModel", "complexGunXModelLeftHand",
					"complexGunXModelRightHand", "complexWorldModel",
					"complexWorldModelLeftHand", "complexWorldModelRightHand" })
				{
					weap_def[key] = nullptr;
				}

				for (const auto* key : { "camoViewModelMaterialOverride",
					"camoWorldModelMaterialOverride" })
				{
					for (auto& entry : weap_def[key])
					{
						entry = "";
					}
				}

				for (const auto* key : { "reticleOnePiece", "reticleTwoPiece", "tracerType",
					"hudIcon", "pickupIcon", "ammoCounterIcon" })
				{
					if (weap_def.contains(key))
					{
						weap_def[key] = "";
					}
				}

				clear_template_attachments(data, weap_def);

				// the oracle ships the packages its weapon names instead of pointing at
				// another weapon's. left as the template's these are referenced, so they
				// resolve to the default whenever this zone loads before the one that
				// owns them
				weap_def["vfxPackage"] = "t7_converted_vfxpackage";
				weap_def["sfxPackage"] = "t7_converted_sfxpackage";

				write_bo3_snapshot(variant_asset, weap, name);
				const auto mapped = write_anim_package(variant_asset, name);

				if (!write_json("weapons\\"s + name + ".json"s, data))
				{
					ZONETOOL_ERROR("could not open weapon file for \"%s\"", asset_name);
					return;
				}

				ZONETOOL_INFO("weapon \"%s\": view \"%s\" world \"%s\", clip %i, %i anims",
					asset_name, view_model ? view_model : "none", world_model ? world_model : "none",
					variant_asset->iClipSize, mapped);
			}

			void dump_camo(void* asset, const char* asset_name)
			{
				if (!is_readable(asset, sizeof(boiii::WeaponCamo)) || !asset_name)
				{
					return;
				}

				const auto* camo = static_cast<const boiii::WeaponCamo*>(asset);
				const auto* internal_name = safe_name(camo->name);
				if (!internal_name || std::strcmp(internal_name, asset_name) != 0)
				{
					ZONETOOL_ERROR("weapon camo \"%s\" does not match its internal name \"%s\"",
						asset_name, internal_name ? internal_name : "");
					return;
				}

				if (camo->numCamoMaterials > 64 || (camo->numCamoMaterials &&
					!is_readable(camo->camoMaterials,
						camo->numCamoMaterials * sizeof(boiii::WeaponCamoMaterialSet))))
				{
					ZONETOOL_ERROR("weapon camo \"%s\" has an invalid material-set table", asset_name);
					return;
				}

				ordered_json data;
				data["name"] = asset_name;
				data["numCamoMaterials"] = camo->numCamoMaterials;
				data["materialSets"] = ordered_json::array();

				for (std::uint32_t set_index = 0; set_index < camo->numCamoMaterials; set_index++)
				{
					const auto& set = camo->camoMaterials[set_index];
					if (set.numMaterials > 256 || (set.numMaterials &&
						!is_readable(set.materials, set.numMaterials * sizeof(boiii::CamoMaterial))))
					{
						ZONETOOL_ERROR("weapon camo \"%s\" set %u has an invalid material table",
							asset_name, set_index);
						return;
					}

					ordered_json set_json;
					set_json["index"] = set_index;
					set_json["numMaterials"] = set.numMaterials;
					set_json["materials"] = ordered_json::array();

					for (std::uint32_t material_index = 0; material_index < set.numMaterials; material_index++)
					{
						const auto& material = set.materials[material_index];
						if (material.numBaseMaterials > 64 || (material.numBaseMaterials &&
							!is_readable(material.baseMaterials,
								material.numBaseMaterials * sizeof(boiii::CamoBaseMaterial))))
						{
							ZONETOOL_ERROR("weapon camo \"%s\" set %u material %u has an invalid base-material table",
								asset_name, set_index, material_index);
							return;
						}

						ordered_json material_json;
						material_json["index"] = material_index;
						material_json["numBaseMaterials"] = material.numBaseMaterials;
						material_json["activeChannels"] = material.activeChannels;
						material_json["baseMaterials"] = ordered_json::array();
						for (std::uint16_t base_index = 0; base_index < material.numBaseMaterials; base_index++)
						{
							const auto& base = material.baseMaterials[base_index];
							const auto* base_name = read_asset_name(&base, offsetof(boiii::CamoBaseMaterial, material));
							const auto* mask_name = read_asset_name(&base, offsetof(boiii::CamoBaseMaterial, mask));
							material_json["baseMaterials"].push_back({
								{ "index", base_index },
								{ "material", base_name ? base_name : "" },
								{ "mask", mask_name ? mask_name : "" },
							});
						}

						material_json["channels"] = ordered_json::array();
						for (std::uint32_t channel_index = 0; channel_index < 4; channel_index++)
						{
							const auto& channel = material.camoMaterialChannels[channel_index];
							const auto* channel_material = read_asset_name(&channel,
								offsetof(boiii::CamoMaterialChannel, camoMaterial));
							const auto* detail_map = read_asset_name(&channel,
								offsetof(boiii::CamoMaterialChannel, detailMap));
							material_json["channels"].push_back({
								{ "index", channel_index },
								{ "active", (material.activeChannels & (1u << channel_index)) != 0 },
								{ "replaceFlags", channel.replaceFlags },
								{ "camoMaterial", channel_material ? channel_material : "" },
								{ "translation", ordered_json::array({ channel.translationX, channel.translationY }) },
								{ "scale", ordered_json::array({ channel.scaleX, channel.scaleY }) },
								{ "rotation", channel.rotation },
								{ "normalBlend", channel.normalBlend },
								{ "glossBlend", channel.glossBlend },
								{ "albedoTint", ordered_json::array({
									channel.albedoTint.rgba.r, channel.albedoTint.rgba.g,
									channel.albedoTint.rgba.b, channel.albedoTint.rgba.a }) },
								{ "detailMap", detail_map ? detail_map : "" },
								{ "detailHeight", channel.detailHeight },
								{ "detailScale", ordered_json::array({ channel.detailScale[0], channel.detailScale[1] }) },
							});
						}
						set_json["materials"].push_back(std::move(material_json));
					}
					data["materialSets"].push_back(std::move(set_json));
				}

				const auto prefix = "weaponcamo\\"s + asset_name;
				write_json(prefix + ".bo3.json", data);
				write_raw(prefix + ".bo3.bin", camo, sizeof(*camo));
				ZONETOOL_INFO("weapon camo \"%s\": %u material sets", asset_name,
					camo->numCamoMaterials);
			}
		}
	}
}
