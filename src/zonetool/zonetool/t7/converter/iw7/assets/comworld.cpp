#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "comworld.hpp"

#include "zonetool/t7/converter/iw7/map_common.hpp"

#include "zonetool/iw7/assets/comworld.hpp"
#include "material_texture.hpp"
#include "gfximage.hpp"
#include <DirectXTex.h>
#include "zonetool/t7/converter/iw7/map_entities.hpp"
#include <utils/string.hpp>

// How each field is derived. BO3 references are to BlackOps3.exe (client dump), IW7 ones to
// iw7_ship.exe and its stock data. Nothing below is a tuned constant.
//
// * visibility: BO3 draws a primary light only when (lightStateMask & (1 << lightingState))
//   and !exploderDisabled and |color|^2 >= 0.001 (light cull 0x141CBF070, state bit from the
//   frame setup 0x141CB9590). A mask of 0 is therefore never drawn at runtime; those lights
//   only exist for the offline probe bake. The map starts in lighting state 0, which the
//   worldspawn names "power_off" (state_alias_1; the aliases are 1-based).
// * color and falloff: BO3 uploads _color * exploderFade (0x141C437C0) and attenuates with
//   min(1, (dAttenuation / d)^2) (deferred_lighting shader, spot and omni), so its irradiance is
//   color * min(1, dAttenuation^2 / d^2). IW7 divides the colour by max(bulbRadius, 1)^2 when it
//   uploads a light and its shaders multiply by the sphere form factor min(1, (bulbRadius / d)^2),
//   so its irradiance is color / d^2 outside the bulb (stock: bulbRadius 1, colours in the
//   thousands). bulbRadius = max(dAttenuation, 1) with color = BO3 color * dAttenuation^2 gives
//   BO3's curve, near field included (dAttenuation < 1 only differs inside one unit). Both games
//   shade diffuse as albedo * sum(color * attenuation * N.L) with no 1/pi (BO3 deferred_lighting,
//   IW7 lmap_* world pixel shader); the absolute scale is map::bo3_light_scale (map_common.hpp).
// * range window: BO3 fades out over the last (1 - far_edge) of (cut_off - cut_on); IW7 fades
//   out over the last distanceFalloff of radius (stock lights all use 0.2).
// * cone: BO3's spot footprint is a superellipse in the projection of half angle
//   acos(finalCosHalfFov), fully lit inside semi-axes (x0, y0) and dark outside (x1, y1)
//   (spot branch of the shader, packed by 0x141CC2050). IW7 cones are circular, so each
//   edge becomes the circle of equal area.
// * direction: BO3 wldDir points back towards the light (the shadow projection puts
//   origin - wldDir in front of the light) and so does IW7's dir (stock spots, and
//   the world shader tests dir against the pixel-to-light vector).
// * orientation: IW7 keeps an up vector; BO3's spot orientation is the second column of its
//   shadow projection. BO3 omni projections carry no rotation, so omni lights keep +Y.
// * sun: BO3 keeps one sun per lighting state in each sun volume and uploads color * intensity
//   along AngleVectors(pitch, yaw), the way its light travels; IW7's sun dir is the negation (toward the sun).

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace comworld
		{
			namespace
			{
				utils::memory::allocator persistent_allocator;
				zonetool::iw7::ComWorld* last_converted = nullptr;
				const ComWorld* last_source = nullptr;
				std::vector<unsigned int> light_remap;

				struct flicker_light
				{
					unsigned int index;
					int slice;
					float scroll[2];
					float offset[2];
					float intensity;
					float origin[3];
					bool spot;
				};
				std::vector<flicker_light> flicker_lights;

				// BO3 script lights switched by an exploder (ZT_LIGHT_EXPLODERS_ON): IW7 scriptable lights with a light entity
				// named after the exploder
				struct script_light
				{
					unsigned int index;
					float origin[3];
					bool spot;
					std::string exploder;
				};
				std::vector<script_light> script_lights;

				constexpr auto pi = 3.14159265358979f;

				// The lighting state the converted map is frozen in (see the header comment).
				const auto start_lighting_state = map::lighting_state();

				// BO3's own threshold for a light that emits nothing (light cull 0x141CBF070).
				constexpr auto min_color_length_sq = 0.001f;

				// BO3 switches from the separable (rectangular) spot profile to the superellipse
				// one at this roundness (shader: lt cb9[+10].w, 0.000488).
				constexpr auto min_superellipse_roundness = 0.00048828125f;

				// Every stock IW7 spot and omni light carries these values;
				// BO3 has no counterpart for them.
				constexpr auto stock_shadow_softness = 0.55f;
				constexpr auto stock_shadow_bias = 0.4f;
				constexpr auto stock_shadow_area = 0.0018f;
				constexpr auto stock_rotation_limit = 1.0f;
				constexpr auto stock_light_def = "light_point_linear";

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

				bool is_drawn_at_start(const GfxConfig_Light& light)
				{
					if (!(light.lightStateMask & (1u << start_lighting_state)))
					{
						return false;
					}

					if (light.exploderDisabled)
					{
						return false;
					}

					const auto len_sq = light._color[0] * light._color[0] + light._color[1] * light._color[1]
						+ light._color[2] * light._color[2];
					return len_sq >= min_color_length_sq;
				}

				unsigned int unique_entity_id(const std::uint32_t guid, const unsigned int index,
					std::unordered_set<unsigned int>& used)
				{
					// every stock IW7 light has its own entityId; BO3's guid is the light's own
					// identifier, but prefab copies share it, so collisions are re-hashed
					auto id = guid;
					auto salt = index + 1;
					while (!id || used.contains(id))
					{
						id = (id ^ (salt * 0x9E3779B1u)) * 0x85EBCA6Bu;
						salt++;
					}
					used.insert(id);
					return id;
				}

				// Area of the superellipse |x/a|^p + |y/b|^p = 1.
				double superellipse_area(const double a, const double b, const double p)
				{
					if (!std::isfinite(p))
					{
						return 4.0 * a * b;
					}

					const auto g1 = std::tgamma(1.0 + 1.0 / p);
					return 4.0 * a * b * g1 * g1 / std::tgamma(1.0 + 2.0 / p);
				}

				// Circular edge (as a fraction of the projection edge) with the area of BO3's
				// superellipse edge. roundness < 2^-11 selects BO3's separable profile, whose
				// footprint is the rectangle |x| <= a, |y| <= b.
				double equal_area_radius(const double a, const double b, const float roundness)
				{
					const auto p = roundness < min_superellipse_roundness
						? std::numeric_limits<double>::infinity()
						: 2.0 / static_cast<double>(roundness);
					return std::sqrt(superellipse_area(a, b, p) / static_cast<double>(pi));
				}

				struct spot_edges
				{
					double inner_x, outer_x, inner_y, outer_y;
				};

				// BO3 normalises the superellipse by the larger of each pair before upload
				// (0x141C432D0): x uses (se.x, se.y), y uses (se.z, se.w).
				spot_edges normalized_edges(const GfxConfig_Light& light)
				{
					const auto* se = light.superellipse;
					const auto x_scale = std::max(se[0], se[1]);
					const auto y_scale = std::max(se[2], se[3]);

					spot_edges edges{};
					edges.inner_x = x_scale > 0.0f ? se[0] / x_scale : 0.0;
					edges.outer_x = x_scale > 0.0f ? se[1] / x_scale : 1.0;
					edges.inner_y = y_scale > 0.0f ? se[2] / y_scale : 0.0;
					edges.outer_y = y_scale > 0.0f ? se[3] / y_scale : 1.0;
					return edges;
				}

				float cone_cos(const double fraction, const double tan_edge)
				{
					return static_cast<float>(std::cos(std::atan(fraction * tan_edge)));
				}

				// The distance at which the converted cone should cover BO3's footprint. BO3
				// moves an ortho spot's apex back by orthoDist, IW7 cannot, so the footprint is
				// matched where the light ends (cut_off from the light).
				double cone_tan(const GfxConfig_Light& light)
				{
					// finalCosHalfFov is only computed for spots (0x141C40F00)
					const auto stored = light.type == LIGHT_TYPE_SPOT ? light.finalCosHalfFov : light.cosHalfFov;
					const auto final_cos = std::clamp(static_cast<double>(stored), 1e-4, 1.0);
					const auto tan_final = std::sqrt(std::max(0.0, 1.0 - final_cos * final_cos)) / final_cos;
					if (light.orthoDist <= 0.0f || light.cut_off <= 0.0f)
					{
						return tan_final;
					}

					return tan_final * (static_cast<double>(light.orthoDist) + light.cut_off) / light.cut_off;
				}

				// BO3 fades out over (1 - far_edge) * (cut_off - cut_on), far_edge clamped to
				// [0, 0.999] (0x141C437C0); IW7 fades out over distanceFalloff * radius.
				float distance_falloff(const GfxConfig_Light& light)
				{
					if (light.cut_off <= 0.0f)
					{
						return 1.0f;
					}

					const auto far_edge = std::clamp(light.far_edge, 0.0f, 0.999f);
					const auto range = std::max(light.cut_off - light.cut_on, 0.001f);
					return std::clamp((1.0f - far_edge) * range / light.cut_off, 0.001f, 1.0f);
				}

				// Second column of BO3's row-vector shadow projection: the light's up axis.
				bool projection_up(const GfxConfig_Light& light, float* up)
				{
					for (auto c = 0; c < 3; c++)
					{
						up[c] = light.nonSunShadowTransform[c][1];
					}

					const auto len = std::sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
					if (!(len > 1e-6f))
					{
						return false;
					}

					normalize(up);
					return true;
				}

				struct conversion_stats
				{
					unsigned int cone_approximated = 0;
					unsigned int ortho = 0;
					unsigned int fade_in_dropped = 0;
					unsigned int cookie_or_ies = 0;
					unsigned int spec_scale = 0;
					unsigned int up_fallback = 0;
				};

				void convert_local_light(const GfxConfig_Light& src, zonetool::iw7::ComPrimaryLight& dst,
					const unsigned int index, std::unordered_set<unsigned int>& used_ids, conversion_stats& stats)
				{
					const auto is_spot = src.type == LIGHT_TYPE_SPOT;
					dst.type = is_spot ? zonetool::iw7::GFX_LIGHT_TYPE_SPOT : zonetool::iw7::GFX_LIGHT_TYPE_OMNI;

					dst.canUseShadowMap = src.shadowQuality != SHADOW_QUALITY_NONE
						&& src.shadowQuality != SHADOW_QUALITY_INVALID;
					dst.needsDynamicShadows = dst.canUseShadowMap && src.shadowUpdate != SHADOW_UPDATE_NEVER;
					dst.isVolumetric = src.volumetric;
					dst.exponent = 0;
					dst.transientZoneList = 0;
					dst.entityId = unique_entity_id(src.guid, index, used_ids);
					dst.uvIntensity = 0.0f;
					dst.irIntensity = 0.0f;

					// both games read the light colour as fp32 (BO3 cb9 dwords, IW7 structured
					// buffer t24), so it is copied without range limits; the dAttenuation^2 moves
					// BO3's falloff numerator into IW7's colour (header comment)
					const auto d_att_sq = src.dAttenuation * src.dAttenuation;
					for (auto c = 0; c < 3; c++)
					{
						dst.color[c] = src._color[c] * src.exploderFade * d_att_sq * map::bo3_light_scale;
						dst.origin[c] = src.origin[c];
						dst.dir[c] = src.wldDir[c];
					}

					dst.bulbRadius = std::max(src.dAttenuation, 1.0f);
					dst.bulbLength[0] = src.bulbDirAndLength[0] * src.bulbDirAndLength[3];
					dst.bulbLength[1] = src.bulbDirAndLength[1] * src.bulbDirAndLength[3];
					dst.bulbLength[2] = src.bulbDirAndLength[2] * src.bulbDirAndLength[3];

					dst.radius = src.cut_off;
					dst.distanceFalloff = distance_falloff(src);
					dst.fadeOffsetRt[0] = 0.0f;
					dst.fadeOffsetRt[1] = 0.0f;

					const auto edges = normalized_edges(src);
					const auto tan_edge = cone_tan(src);
					const auto inner = equal_area_radius(edges.inner_x, edges.inner_y, src.roundness);
					const auto outer = equal_area_radius(edges.outer_x, edges.outer_y, src.roundness);
					dst.cosHalfFovOuter = cone_cos(outer, tan_edge);
					dst.cosHalfFovInner = std::max(cone_cos(inner, tan_edge), dst.cosHalfFovOuter);

					if (is_spot)
					{
						normalize(dst.dir);
						stats.cone_approximated += (src.roundness < 0.999f
							|| edges.inner_x != edges.inner_y || edges.outer_x != edges.outer_y);
						stats.ortho += src.orthoDist > 0.0f;

						if (!projection_up(src, dst.up))
						{
							dst.up[0] = 0.0f;
							dst.up[1] = 1.0f;
							dst.up[2] = 0.0f;
							stats.up_fallback++;
						}
					}
					else
					{
						// IW7 only reads dir for spots; BO3 omni lights carry a zero vector, stock
						// IW7 omni lights never do and most use +Z
						dst.dir[0] = 0.0f;
						dst.dir[1] = 0.0f;
						dst.dir[2] = 1.0f;
						dst.up[0] = 0.0f;
						dst.up[1] = 1.0f;
						dst.up[2] = 0.0f;
					}

					stats.fade_in_dropped += src.cut_on > 0.0f;
					stats.cookie_or_ies += src.lightDefName[0] != '\0';
					stats.spec_scale += src.specScale != 1.0f;

					dst.shadowSoftness = stock_shadow_softness;
					dst.shadowBias = stock_shadow_bias;
					dst.shadowArea = stock_shadow_area;
					dst.rotationLimit = stock_rotation_limit;
					dst.translationLimit = 0.0f;
					dst.defName = stock_light_def;
				}

				// IW7 keeps the sun as primary light 1. BO3 keeps it per lighting state in each
				// sun volume.
				const GfxGlobalLightSettings* find_sun(const GfxWorld* world)
				{
					if (!world || !world->sunVolumes)
					{
						return nullptr;
					}

					const auto main = map::main_sun_volume(world);
					if (world->sunVolumes[main].sun.lightStateMask & (1u << start_lighting_state))
					{
						return &world->sunVolumes[main].sun.settings[start_lighting_state];
					}
					for (auto i = 0u; i < world->sunVolumeCount; i++)
					{
						const auto& volume = world->sunVolumes[i];
						if (volume.sun.lightStateMask & (1u << start_lighting_state))
						{
							return &volume.sun.settings[start_lighting_state];
						}
					}

					return nullptr;
				}

				void convert_sun(const GfxGlobalLightSettings* sun, zonetool::iw7::ComPrimaryLight& dst)
				{
					std::memset(&dst, 0, sizeof(dst));
					dst.type = zonetool::iw7::GFX_LIGHT_TYPE_DIR;
					if (!sun)
					{
						return;
					}

					// IW7's sun dir points at the sun; AngleVectors(pitch, yaw) of BO3's settings is the way the light
					// travels, so the IW7 dir is its negation
					const auto pitch = sun->pitch * (pi / 180.0f);
					const auto yaw = sun->yaw * (pi / 180.0f);
					dst.dir[0] = -std::cos(pitch) * std::cos(yaw);
					dst.dir[1] = -std::cos(pitch) * std::sin(yaw);
					dst.dir[2] = std::sin(pitch);
					normalize(dst.dir);

					// the uploaded sun colour is color * intensity (0x141C9E840)
					for (auto c = 0; c < 3; c++)
					{
						dst.color[c] = sun->color[c] * sun->intensity * map::bo3_light_scale;
					}
				}
			}

			zonetool::iw7::ComWorld* convert(ComWorld* asset, const GfxWorld* world, utils::memory::allocator& allocator)
			{
				auto* new_asset = allocator.allocate<zonetool::iw7::ComWorld>();
				new_asset->name = allocator.duplicate_string(map::bsp_name(asset->name));
				new_asset->isInUse = 1;
				new_asset->useForwardPlus = 1;
				new_asset->bakeQuality = 3;

				light_remap.assign(asset->primaryLightCount, 0);

				// ZT_LIGHT_TRACE: every BO3 light as stored
				if (std::getenv("ZT_LIGHT_TRACE"))
				{
					for (auto i = 0u; i < asset->primaryLightCount; i++)
					{
						const auto& c = asset->primaryLights[i].config;
						ZONETOOL_INFO("light trace %u: type %d guid %08x origin %.2f %.2f %.2f colour %.4f %.4f %.4f dAtt %.3f cut %.2f %.2f "
							"edge %.3f %.3f state %x exploder %d fade %.3f def \"%s\" cookie angle %.3f rot %.3f scale %.3f %.3f scroll %.3f %.3f "
							"offset %.3f %.3f cookieIndex %d cull %.1f %.1f %.1f - %.1f %.1f %.1f cullRadius %.1f wldDir %.3f %.3f %.3f cos %.4f final %.4f ortho %.2f vol %d volCookies %d volSamples %u volScale %.3f", i, c.type, c.guid, c.origin[0], c.origin[1], c.origin[2], c._color[0], c._color[1],
							c._color[2], c.dAttenuation, c.cut_on, c.cut_off, c.near_edge, c.far_edge, c.lightStateMask, c.exploderDisabled,
							c.exploderFade, c.lightDefName, c.cookieAngle, c.cookieRotation, c.cookieScale[0], c.cookieScale[1], c.cookieScroll[0],
							c.cookieScroll[1], c.cookieOffset[0], c.cookieOffset[1], asset->primaryLights[i].cookieIndex, c.wldCullMin[0], c.wldCullMin[1], c.wldCullMin[2], c.wldCullMax[0], c.wldCullMax[1], c.wldCullMax[2], c.cullRadius, c.wldDir[0], c.wldDir[1], c.wldDir[2], c.cosHalfFov, c.finalCosHalfFov, c.orthoDist, c.volumetric, c.volumetricCookies, c.volumetricSampleCount, c.volumetricIntensityScale);
					}
					for (auto i = 0u; i < asset->lightExploderCount; i++)
					{
						const auto& e = asset->lightExploders[i];
						for (auto t = 0; t < e.triggerCount; t++)
						{
							const auto& d = e.triggerData[t];
							ZONETOOL_INFO("light trace exploder %08x: light %u on %d off %d in %d out %d reversed %d", e.nameHash,
								d.primaryLightIndex, d.delayOn, d.delayOff, d.fadeIn, d.fadeOut, d.reversed);
						}
					}
					for (auto i = 0u; i < asset->probeExploderCount; i++)
					{
						const auto& e = asset->probeExploders[i];
						for (auto t = 0; t < e.triggerCount; t++)
						{
							const auto& d = e.triggerData[t];
							ZONETOOL_INFO("light trace probe exploder %08x: probe %d volume %d on %d off %d in %d out %d reversed %d", e.nameHash,
								d.probeID, d.volumeID, d.delayOn, d.delayOff, d.fadeIn, d.fadeOut, d.reversed);
						}
					}
				}

				// lights a level script's exploder switches on for good (ZT_LIGHT_EXPLODERS_ON, comma separated names): BO3 keeps
				// them out of every lighting state (mask 0) and draws them once the exploder plays; nameHash is the djb2 hash of the name
				std::unordered_map<unsigned int, std::string> exploder_lit;
				if (const auto* names = std::getenv("ZT_LIGHT_EXPLODERS_ON"))
				{
					for (const auto& name : utils::string::split(names, ','))
					{
						std::uint32_t hash = 5381;
						for (const auto c : name)
						{
							hash = hash * 33 + static_cast<unsigned char>(c);
						}
						auto found = 0u;
						for (auto i = 0u; i < asset->lightExploderCount; i++)
						{
							const auto& e = asset->lightExploders[i];
							if (static_cast<std::uint32_t>(e.nameHash) != hash)
							{
								continue;
							}
							for (auto t = 0; t < e.triggerCount; t++)
							{
								if (!e.triggerData[t].reversed)
								{
									exploder_lit.emplace(e.triggerData[t].primaryLightIndex, name);
									found++;
								}
							}
						}
						ZONETOOL_INFO("comworld: exploder \"%s\" (%08x) lights %u lights from the start", name.data(), hash, found);
					}
				}

				std::vector<unsigned int> kept;
				kept.reserve(asset->primaryLightCount);
				auto never_drawn = 0u, other_states = 0u, exploder_off = 0u, dark = 0u, other_types = 0u;
				for (auto i = 0u; i < asset->primaryLightCount; i++)
				{
					const auto& config = asset->primaryLights[i].config;
					if (config.type != LIGHT_TYPE_SPOT && config.type != LIGHT_TYPE_OMNI)
					{
						other_types++;
						continue;
					}

					if (!is_drawn_at_start(config) && !exploder_lit.contains(i))
					{
						if (!config.lightStateMask)
						{
							never_drawn++;
						}
						else if (!(config.lightStateMask & (1u << start_lighting_state)))
						{
							other_states++;
						}
						else if (config.exploderDisabled)
						{
							exploder_off++;
						}
						else
						{
							dark++;
						}
						continue;
					}

					kept.push_back(i);
				}

				// BO3 flickers a light by scrolling its cookie_flicker* light def over time (light cookie: uv = frac(row . (x, y,
				// 1)), the row's constant frac(|scroll| t) sign(scroll) + offset + 0.5, 0x141C3F750; deferred_lighting samples slice
				// cookieIndex - 1 at level 0 and multiplies the light by it). IW7 cookies do not move; its scriptable primary
				// lights do: a light entity with "pl#" at or past firstScriptablePrimaryLight (0x1404003E0), whose intensity a
				// level script sets. These go last, as in stock maps
				const auto flickers = [&](const unsigned int i)
				{
					return std::string_view(asset->primaryLights[i].config.lightDefName).starts_with("cookie_flicker");
				};
				// order: static lights, exploder script lights, flickering lights; the last two are IW7's scriptable range
				const auto rank = [&](const unsigned int i) { return flickers(i) ? 2 : exploder_lit.contains(i) ? 1 : 0; };
				std::ranges::stable_sort(kept, [&](const unsigned int a, const unsigned int b) { return rank(a) < rank(b); });
				const auto first_script = static_cast<unsigned int>(std::ranges::count_if(kept, [&](const unsigned int i) { return rank(i) == 0; }));
				const auto first_flicker = static_cast<unsigned int>(std::ranges::count_if(kept, [&](const unsigned int i) { return rank(i) < 2; }));

				// 0 = none, 1 = sun, then the local lights
				new_asset->primaryLightCount = static_cast<unsigned int>(kept.size()) + 2;
				new_asset->primaryLights = allocator.allocate_array<zonetool::iw7::ComPrimaryLight>(new_asset->primaryLightCount);

				const auto* sun = find_sun(world);
				convert_sun(sun, new_asset->primaryLights[1]);

				conversion_stats stats{};
				std::unordered_set<unsigned int> used_ids;
				for (auto k = 0u; k < kept.size(); k++)
				{
					const auto t7_index = kept[k];
					light_remap[t7_index] = k + 2;
					convert_local_light(asset->primaryLights[t7_index].config, new_asset->primaryLights[k + 2], t7_index,
						used_ids, stats);
				}

				// IW7 holds 64 spot shadow maps (0x140E1C620: past its cache, a shadowed light waits for one of the 8
				// shadow updates a frame, and is not drawn until it gets one). BO3 has no such budget, so wherever more than
				// 24 shadowed lights share a neighbourhood, the least important keep their light without a shadow (64, the
				// cache's size, left Water Park 317 shadowed lights: their shadow maps redrawn as zombies move cost frames).
				{
					constexpr auto shadow_slots = 24u;
					constexpr auto neighbourhood = 2048.0f; // a sightline's reach
					std::vector<zonetool::iw7::ComPrimaryLight*> shadowed;
					for (auto k = 0u; k < kept.size(); k++)
					{
						auto& light = new_asset->primaryLights[k + 2];
						if (light.canUseShadowMap)
						{
							shadowed.push_back(&light);
						}
					}
					const auto importance = [](const zonetool::iw7::ComPrimaryLight* light)
					{
						return std::max({ light->color[0], light->color[1], light->color[2] }) * light->radius * light->radius;
					};
					std::ranges::sort(shadowed, [&](const auto* a, const auto* b) { return importance(a) > importance(b); });
					std::vector<zonetool::iw7::ComPrimaryLight*> kept_shadows;
					auto dropped = 0u;
					for (auto* light : shadowed)
					{
						auto sharing = 0u;
						for (const auto* other : kept_shadows)
						{
							const auto ox = light->origin[0] - other->origin[0], oy = light->origin[1] - other->origin[1],
								oz = light->origin[2] - other->origin[2];
							const auto reach = neighbourhood + light->radius + other->radius;
							sharing += ox * ox + oy * oy + oz * oz < reach * reach;
						}
						if (sharing + 1 > shadow_slots)
						{
							light->canUseShadowMap = 0;
							light->needsDynamicShadows = 0;
							dropped++;
							continue;
						}
						kept_shadows.push_back(light);
					}
					ZONETOOL_INFO("comworld: %zu shadowed lights, %u of the least important left unshadowed where more than %u share a %g-unit neighbourhood",
						shadowed.size(), dropped, shadow_slots, neighbourhood);
				}

				new_asset->primaryLightEnvCount = new_asset->primaryLightCount;
				new_asset->primaryLightEnvs = allocator.allocate_array<zonetool::iw7::ComPrimaryLightEnv>(new_asset->primaryLightEnvCount);
				for (auto i = 1u; i < new_asset->primaryLightEnvCount; i++)
				{
					new_asset->primaryLightEnvs[i].numIndices = 1;
					new_asset->primaryLightEnvs[i].primaryLightIndices[0] = static_cast<unsigned short>(i);
				}

				new_asset->scriptablePrimaryLightCount = static_cast<unsigned int>(kept.size()) - first_script;
				new_asset->firstScriptablePrimaryLight = first_script + 2;

				script_lights.clear();
				for (auto k = first_script; k < first_flicker; k++)
				{
					const auto& dst = new_asset->primaryLights[k + 2];
					script_light l{};
					l.index = k + 2;
					std::memcpy(l.origin, dst.origin, sizeof(l.origin));
					l.spot = dst.type == zonetool::iw7::GFX_LIGHT_TYPE_SPOT;
					l.exploder = exploder_lit.at(kept[k]);
					script_lights.push_back(l);
				}
				if (!script_lights.empty())
				{
					ZONETOOL_INFO("comworld: %zu exploder script lights scriptable from primary light %u", script_lights.size(),
						new_asset->firstScriptablePrimaryLight);
				}

				// the flickering lights for the level script, and their entities
				flicker_lights.clear();
				for (auto k = first_flicker; k < kept.size(); k++)
				{
					const auto& src = asset->primaryLights[kept[k]];
					const auto& dst = new_asset->primaryLights[k + 2];
					flicker_light f{};
					f.index = k + 2;
					f.slice = src.cookieIndex - 1;
					f.scroll[0] = src.config.cookieScroll[0];
					f.scroll[1] = src.config.cookieScroll[1];
					f.offset[0] = src.config.cookieOffset[0];
					f.offset[1] = src.config.cookieOffset[1];
					// the light entity's intensity: 0x1404003E0 sets it to the colour's luma / 493.38 (Rec. 709 weights)
					f.intensity = (0.2126f * dst.color[0] + 0.7152f * dst.color[1] + 0.0722f * dst.color[2]) * 0.0020268299f;
					std::memcpy(f.origin, dst.origin, sizeof(f.origin));
					f.spot = dst.type == zonetool::iw7::GFX_LIGHT_TYPE_SPOT;
					flicker_lights.push_back(f);
				}
				if (!flicker_lights.empty())
				{
					ZONETOOL_INFO("comworld: %zu flickering lights (cookie_flicker*) scriptable from primary light %u",
						flicker_lights.size(), new_asset->firstScriptablePrimaryLight);
				}

				// stock IW7 build metadata, as the IW5 converter writes it
				new_asset->changeListInfo.changeListNumber = 1232774;
				new_asset->changeListInfo.time = 1480386331;
				new_asset->changeListInfo.userName = "manyomi";

				new_asset->numUmbraGates = 0;
				new_asset->umbraGateNames = nullptr;
				for (auto i = 0; i < 4; i++)
				{
					new_asset->umbraGateInitialStates[i] = -1;
				}

				ZONETOOL_INFO("comworld \"%s\": %u of %u lights drawn in lighting state %u (+ null + sun); not drawn: "
					"%u bake-only (state mask 0), %u other states, %u held by an exploder, %u dark, %u other types",
					new_asset->name, static_cast<unsigned int>(kept.size()), asset->primaryLightCount, start_lighting_state,
					never_drawn, other_states, exploder_off, dark, other_types);
				ZONETOOL_INFO("comworld \"%s\": sun %s (%.3f %.3f %.3f); circular cones for %u shaped spots, %u ortho spots "
					"matched at cut_off, %u fade-ins (cut_on) not representable, %u light defs pending, %u per-light spec "
					"scales not representable, %u spots without a projection up axis",
					new_asset->name, sun ? "from sun volume" : "MISSING", new_asset->primaryLights[1].color[0],
					new_asset->primaryLights[1].color[1], new_asset->primaryLights[1].color[2], stats.cone_approximated,
					stats.ortho, stats.fade_in_dropped, stats.cookie_or_ies, stats.spec_scale, stats.up_fallback);

				return new_asset;
			}

			void export_flicker(const GfxWorld* world)
			{
				// the light entities: "pl#" (key ID 1) ties each to its scriptable primary light (0x1404003E0)
				std::vector<map_entities::entity> ents;
				for (const auto& l : script_lights)
				{
					map_entities::entity e{};
					e.set("classname", l.spot ? "light_spot" : "light_omni");
					e.set("origin", utils::string::va("%g %g %g", l.origin[0], l.origin[1], l.origin[2]));
					e.set("targetname", l.exploder);
					e.keys.push_back({ "1", std::to_string(l.index), false });
					ents.push_back(std::move(e));
				}
				if (flicker_lights.empty())
				{
					map_entities::set_light_entities(std::move(ents));
					return;
				}

				for (const auto& f : flicker_lights)
				{
					map_entities::entity e{};
					e.set("classname", f.spot ? "light_spot" : "light_omni");
					e.set("origin", utils::string::va("%g %g %g", f.origin[0], f.origin[1], f.origin[2]));
					e.set("targetname", "t7_light_flicker");
					e.keys.push_back({ "1", std::to_string(f.index), false });
					ents.push_back(std::move(e));
				}
				map_entities::set_light_entities(std::move(ents));

				// the cookie slices they scroll, level 0 as linear float RGBA ("slice_<n>.rgba": width, height (uint32), texels)
				// and the lights ("t7_light_flicker.json"), in the dump's maps folder
				const auto dir = filesystem::get_dump_path() + "maps/t7_light_cookies/";
				std::filesystem::create_directories(dir);
				std::set<int> slices;
				std::string json = "{\n\t\"lights\": [\n";
				for (auto i = 0u; i < flicker_lights.size(); i++)
				{
					const auto& f = flicker_lights[i];
					slices.insert(f.slice);
					json += utils::string::va("\t\t{ \"index\": %u, \"slice\": %d, \"scroll\": [%.9g, %.9g], \"offset\": [%.9g, %.9g], "
						"\"intensity\": %.9g, \"origin\": [%.9g, %.9g, %.9g] }%s\n", f.index, f.slice, f.scroll[0], f.scroll[1], f.offset[0],
						f.offset[1], f.intensity, f.origin[0], f.origin[1], f.origin[2], i + 1 < flicker_lights.size() ? "," : "");
				}
				json += "\t]\n}\n";
				std::ofstream(dir + "t7_light_flicker.json", std::ios::binary) << json;

				const auto* image = world ? world->draw.cookieArray : nullptr;
				if (!image || !image->pixels)
				{
					ZONETOOL_FATAL("comworld: %zu flickering lights, but the world's cookie array has no pixels", flicker_lights.size());
				}
				// mip-major, as the loader keeps arrays: every slice's level 0, then every slice's level 1, ...
				DirectX::ScratchImage scratch;
				if (FAILED(scratch.Initialize2D(image->format, image->width, image->height, image->depth, image->levelCount))
					|| scratch.GetPixelsSize() != image->totalSize)
				{
					ZONETOOL_FATAL("light cookies \"%s\": %u bytes do not match %ux%u x %u slices, %u levels, format %d", image->name,
						image->totalSize, image->width, image->height, image->depth, image->levelCount, image->format);
				}
				std::size_t offset = 0;
				for (auto l = 0u; l < image->levelCount; l++)
				{
					for (auto s = 0u; s < image->depth; s++)
					{
						const auto* level = scratch.GetImage(l, s, 0);
						std::memcpy(level->pixels, image->pixels + offset, level->slicePitch);
						offset += level->slicePitch;
					}
				}
				for (const auto slice : slices)
				{
					if (slice < 0 || slice >= image->depth)
					{
						ZONETOOL_FATAL("light cookies \"%s\": slice %d of %u", image->name, slice, image->depth);
					}
					DirectX::ScratchImage rgba;
					const auto* src = scratch.GetImage(0, slice, 0);
					const auto hr = DirectX::IsCompressed(image->format)
						? DirectX::Decompress(*src, DXGI_FORMAT_R32G32B32A32_FLOAT, rgba)
						: DirectX::Convert(*src, DXGI_FORMAT_R32G32B32A32_FLOAT, DirectX::TEX_FILTER_DEFAULT, 0.5f, rgba);
					if (FAILED(hr))
					{
						ZONETOOL_FATAL("light cookies \"%s\": slice %d does not decode", image->name, slice);
					}
					const auto* img = rgba.GetImage(0, 0, 0);
					std::ofstream out(dir + "slice_" + std::to_string(slice) + ".rgba", std::ios::binary);
					const auto w = static_cast<std::uint32_t>(img->width), h = static_cast<std::uint32_t>(img->height);
					out.write(reinterpret_cast<const char*>(&w), 4);
					out.write(reinterpret_cast<const char*>(&h), 4);
					for (auto y = 0u; y < h; y++)
					{
						out.write(reinterpret_cast<const char*>(img->pixels + y * img->rowPitch), static_cast<std::streamsize>(w) * 16);
					}
				}
				ZONETOOL_INFO("comworld: %zu flickering lights and %zu cookie slices written to %s", flicker_lights.size(), slices.size(),
					dir.data());
			}

			void dump(ComWorld* asset, const GfxWorld* world)
			{
				auto* converted_asset = convert(asset, world, persistent_allocator);
				zonetool::iw7::com_world::dump(converted_asset);

				last_converted = converted_asset;
				last_source = asset;
			}

			const zonetool::iw7::ComWorld* converted()
			{
				return last_converted;
			}

			const ComWorld* source()
			{
				return last_source;
			}

			unsigned int remap_light(const unsigned int t7_index)
			{
				return t7_index < light_remap.size() ? light_remap[t7_index] : 0;
			}
		}
	}
}
