#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "zonetool/t7/converter/iw7/memory_probe.hpp"
#include "particle_system.hpp"
#include "material.hpp"

#include "zonetool/t7/functions.hpp"
#include "zonetool/t7/converter/iw7/map_common.hpp"
#include "zonetool/iw7/assets/particle_system.hpp"

// BO3 effects (FxEffectDef) as IW7 particle systems (ParticleSystemDef).
//
// BO3:
// * visual state is sampled at N + 1 points over the particle's life (N = visStateIntervalCount), each a base and
//   amplitude half (0x50); colour lerps between the halves, other fields are base + random x amplitude. Half layout:
//   colour +0x00, rotationDelta +0x04, rotationTotal +0x08, size +0x0C / +0x10, scale +0x14, HDR value or light
//   intensity +0x18, light radius +0x1C, spot cone in degrees +0x20.
// * velocity samples (0x60: local then world) are N x lerp(k, k + 1) x 1000 units a second (0x140225730); gravity
//   adds gravity x 0.8 units a second every millisecond.
// * tails and lines are quads along the velocity (0x140201820 / 0x140201950): a tail sits size[1] behind the
//   particle, a line size[1] ahead.
// * sprite rgb is linear; light colours go through the sRGB table. A light's visual is a template (+8:
//   GfxConfig_Light) whose colour, radius and cone the draw (0x140201EA0) overrides per particle.
//
// IW7:
// * sprite shaders square texture x vertex colour, so colour curves carry the square root.
// * state flags: 0x400 lit, 0x400000 colour graph, 0x1000000 emissive graph, 0x2000000 intensity graph,
//   0x100 / 0x200 local / world velocity graph, 0x100000000 sprite.
// * a tail (0xD095B0) is drawn only once a velocity component exceeds 0.001 units a second; m_tailLeading 0 puts
//   the quad ahead of the particle, 1 behind it.
// * INIT_ATLAS (0xD0F950): playRate, startFrame, loopCount (-1: forever), then two bytes: +20 random start frame,
//   +21 play over life.
// * an omni light's colour is rgb x size.y x 493.38132 with radius size.x (R_AddDynamicOmniLight), falling off as
//   1 / d^2; BO3's falls off as min(1, dAtt^2 / d^2), so size.y = intensity x dAtt^2 / 493.38132.

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace particlesystem
		{
			namespace
			{
				namespace i7 = zonetool::iw7;

				struct t7_vis_state
				{
					unsigned char color[4];
					float rotation_delta;
					float rotation_total;
					float size[2];
					float scale;
					float intensity; // sprites: the particle's HDR value; lights: intensity
					float radius; // lights
					float cone; // spot lights: full cone angle in degrees
					float unk_24;
				};
				static_assert(sizeof(t7_vis_state) == 0x28);

				struct t7_vis_sample
				{
					t7_vis_state base;
					t7_vis_state amplitude;
				};
				static_assert(sizeof(t7_vis_sample) == 0x50);

				struct t7_vel_frame
				{
					float velocity_base[3];
					float velocity_amplitude[3];
					float total_delta_base[3];
					float total_delta_amplitude[3];
				};
				static_assert(sizeof(t7_vel_frame) == 0x30);

				struct t7_vel_sample
				{
					t7_vel_frame local;
					t7_vel_frame world;
				};
				static_assert(sizeof(t7_vel_sample) == 0x60);

				struct t7_trail_def
				{
					int scroll_time_msec;
					int repeat_dist;
					int unk_08;
					float fade_in_dist;
					float fade_out_dist;
					int vert_count;
					void* verts;
					int ind_count;
					int pad_24;
					void* inds;
				};

				enum t7_elem_flags : unsigned int
				{
					T7_FX_ELEM_SPAWN_RELATIVE_TO_EFFECT = 0x2,
					T7_FX_ELEM_SPAWN_OFFSET_SPHERE = 0x10,
					T7_FX_ELEM_SPAWN_OFFSET_CYLINDER = 0x20,
					T7_FX_ELEM_SPAWN_OFFSET_MASK = 0x30,
					T7_FX_ELEM_RUN_RELATIVE_TO_WORLD = 0x0,
					T7_FX_ELEM_RUN_RELATIVE_TO_SPAWN = 0x40,
					T7_FX_ELEM_RUN_RELATIVE_TO_EFFECT = 0x80,
					T7_FX_ELEM_RUN_RELATIVE_TO_OFFSET = 0xC0,
					T7_FX_ELEM_RUN_RELATIVE_TO_CAMERA = 0x100,
					T7_FX_ELEM_RUN_MASK = 0x1C0,
					T7_FX_ELEM_DRAW_PAST_FOG = 0x400,
					T7_FX_ELEM_HAS_VELOCITY_GRAPH_LOCAL = 0x1000000,
					T7_FX_ELEM_HAS_VELOCITY_GRAPH_WORLD = 0x2000000,
					T7_FX_ELEM_HAS_GRAVITY = 0x4000000,
					T7_FX_ELEM_NONUNIFORM_SCALE = 0x10000000,
				};

				enum t7_elem_type : unsigned char
				{
					T7_ELEM_TYPE_SPRITE_BILLBOARD = 0,
					T7_ELEM_TYPE_SPRITE_ORIENTED = 1,
					T7_ELEM_TYPE_SPRITE_ROTATED = 2,
					T7_ELEM_TYPE_TAIL = 3,
					T7_ELEM_TYPE_LINE = 4,
					T7_ELEM_TYPE_TRAIL = 5,
					T7_ELEM_TYPE_CLOUD = 6,
					T7_ELEM_TYPE_MODEL = 7,
					T7_ELEM_TYPE_OMNI_LIGHT = 8,
					T7_ELEM_TYPE_UNKNOWN_9 = 9,
					T7_ELEM_TYPE_SPOT_LIGHT = 10,
					T7_ELEM_TYPE_SOUND = 11,
					T7_ELEM_TYPE_LENS_FLARE = 12,
					T7_ELEM_TYPE_DECAL = 13,
					T7_ELEM_TYPE_RUNNER = 14,
				};

				// BO3's atlas behaviour bits (FxElemAtlas.behavior, evaluated by 0x140207A40): the start frame (0 index,
				// 1 random, 2 the element's sequence, 3 index within a range of indexRange frames), play over the
				// particle's life, loop only loopCount times, blend between frames, frames reversed, atlas used at all
				constexpr unsigned char atlas_start_mask = 0x3;
				constexpr unsigned char atlas_start_range = 0x3;
				constexpr unsigned char atlas_play_over_life = 0x4;
				constexpr unsigned char atlas_loop_only_n_times = 0x8;
				constexpr unsigned char atlas_reverse = 0x20;
				constexpr unsigned char atlas_enabled = 0x80;

				// IW7 state flags (see the header comment)
				constexpr unsigned __int64 state_lit = 0x400;
				constexpr unsigned __int64 state_emissive_graph = 0x1000000;

				// R_AddDynamicOmniLight: an effect light's colour is rgb x size.y x this
				constexpr float omni_light_unit = 493.38132f;

				bool is_readable(const void* ptr, std::size_t size)
				{
					return probe::readable(ptr, size);
				}

				const char* safe_name(const char* name)
				{
					return probe::terminated(name);
				}

				const char* asset_name(const void* asset)
				{
					if (!is_readable(asset, sizeof(void*)))
					{
						return nullptr;
					}
					return safe_name(*static_cast<const char* const*>(asset));
				}

				template <typename T>
				T* alias_asset(const std::string& name, utils::memory::allocator& allocator)
				{
					const auto stub = allocator.allocate<T>();
					stub->name = allocator.duplicate_string(name);
					return stub;
				}

				bool is_sprite_type(const unsigned char type)
				{
					return type <= T7_ELEM_TYPE_CLOUD;
				}

				bool is_light_type(const unsigned char type)
				{
					return type == T7_ELEM_TYPE_OMNI_LIGHT || type == T7_ELEM_TYPE_SPOT_LIGHT;
				}

				float srgb_decode(const float c)
				{
					return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
				}

				// An element converts whole or, for a spot light aimed away from the effect's axis, as two parts: IW7 aims a
				// particle spot light along its effect's axis (0x140D08130), so stock effects use a runner turned by
				// INIT_ROTATION_3D whose child effect holds the light.
				enum class part
				{
					whole,
					spot_runner,
					spot_light,
				};

				struct convert_context
				{
					utils::memory::allocator* allocator = nullptr;
					const references* refs = nullptr;
					const FxEffectDef* effect = nullptr;
					const FxElemDef* elem = nullptr;
					int elem_index = 0;
					unsigned int system_flags = 0;
					unsigned __int64 state_flags = 0;
					unsigned int emitter_flags = 0;
					int test_module_index = 0;
					// a sprite element's material and how it shades (set by the material module)
					bool has_material = false;
					effect_material::converted material{};
					bool frame_blend = false;
					part element_part = part::whole;
					std::string light_child; // spot_runner: the child effect holding the light
				};

				// ---- samples and curves ---------------------------------------------------------------------

				struct vis_samples
				{
					const t7_vis_sample* samples = nullptr;
					int count = 0; // N + 1
					int intervals() const
					{
						return std::max(1, this->count - 1);
					}
				};

				vis_samples get_vis_samples(const FxElemDef* elem)
				{
					const auto count = elem->visStateIntervalCount + 1;
					const auto* samples = reinterpret_cast<const t7_vis_sample*>(elem->visSamples);
					if (!is_readable(samples, sizeof(t7_vis_sample) * count))
					{
						return {};
					}
					return { samples, count };
				}

				struct vel_samples
				{
					const t7_vel_sample* samples = nullptr;
					int count = 0;
					int intervals() const
					{
						return std::max(1, this->count - 1);
					}
				};

				vel_samples get_vel_samples(const FxElemDef* elem)
				{
					const auto count = elem->velIntervalCount + 1;
					const auto* samples = reinterpret_cast<const t7_vel_sample*>(elem->velSamples);
					if (!is_readable(samples, sizeof(t7_vel_sample) * count))
					{
						return {};
					}
					return { samples, count };
				}

				float largest_magnitude(const std::vector<float>& values)
				{
					auto scale = 0.0f;
					for (const auto v : values)
					{
						scale = std::max(scale, std::fabs(v));
					}
					return scale;
				}

				// An IW7 curve through `values` spread evenly over the particle's life (a single value is held for its
				// whole life): points times `scale` (0: the largest magnitude, 1 when every value is 0)
				i7::ParticleCurveDef make_curve(convert_context& ctx, const std::vector<float>& values, float scale = 0.0f)
				{
					i7::ParticleCurveDef curve{};
					const auto count = std::max<std::size_t>(2, values.size());
					if (scale == 0.0f)
					{
						scale = largest_magnitude(values);
					}
					if (scale == 0.0f)
					{
						scale = 1.0f;
					}
					curve.scale = scale;
					curve.numControlPoints = static_cast<int>(count);
					curve.controlPoints = ctx.allocator->allocate_array<i7::ParticleCurveControlPointDef>(count);
					for (auto i = 0u; i < count; i++)
					{
						auto& p = curve.controlPoints[i];
						p.time = static_cast<float>(i) / static_cast<float>(count - 1);
						p.value = (values.size() == 1 ? values[0] : values[i]) / scale;
						p.invTimeDelta = i ? 1.0f / (p.time - curve.controlPoints[i - 1].time) : 0.0f;
					}
					return curve;
				}

				// min and max curves over the same points with the same scale (IW7 randomises between the two); `scale` as
				// make_curve's, shared by both
				void make_curve_pair(convert_context& ctx, const std::vector<float>& min, const std::vector<float>& max,
					i7::ParticleCurveDef& out_min, i7::ParticleCurveDef& out_max, unsigned int& module_flags, float scale = 0.0f)
				{
					if (scale == 0.0f)
					{
						scale = std::max(largest_magnitude(min), largest_magnitude(max));
					}
					out_min = make_curve(ctx, min, scale);
					out_max = make_curve(ctx, max, scale);
					if (min != max)
					{
						module_flags |= i7::PARTICLE_MODULE_FLAG_RANDOMIZE_BETWEEN_CURVES;
					}
				}

				i7::ParticleModuleDef new_module(const i7::ParticleModuleType type)
				{
					i7::ParticleModuleDef module{};
					module.moduleType = type;
					module.moduleData.moduleBase.type = type;
					module.moduleData.moduleBase.m_flags = 0;
					return module;
				}

				// ---- update modules ------------------------------------------------------------------------

				// colour: sprites square it (square roots), distortion multiplies the scene by it (as is), lights take it
				// through BO3's sRGB table
				void generate_color_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto vis = get_vis_samples(ctx.elem);
					if (!vis.samples)
					{
						return;
					}
					const auto light = is_light_type(ctx.elem->elemType);
					const auto distortion = ctx.has_material && ctx.material.shade == effect_material::shading::distortion;
					const auto sprite = is_sprite_type(ctx.elem->elemType) && !distortion;
					const auto channel = [&](const unsigned char byte, const int c)
					{
						const auto v = byte / 255.0f;
						if (c == 3)
						{
							return v;
						}
						return light ? srgb_decode(v) : (sprite ? std::sqrt(v) : v);
					};

					auto module = new_module(i7::PARTICLE_MODULE_COLOR_GRAPH);
					auto& data = module.moduleData.colorGraph;
					for (auto c = 0; c < 4; c++)
					{
						std::vector<float> min, max;
						for (auto i = 0; i < vis.count; i++)
						{
							min.push_back(channel(vis.samples[i].base.color[c], c));
							max.push_back(channel(vis.samples[i].amplitude.color[c], c));
						}
						// stock colour curves are all scaled by 1
						make_curve_pair(ctx, min, max, data.m_curves[c], data.m_curves[c + 4], data.m_flags, 1.0f);
					}
					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_COLOR;
					modules.push_back(module);
				}

				// The per-particle emissive scale an em / emm material reads. BO3 emits colour x vertex rgb x mask x
				// alpha^2 x hdrScale x HDR value; IW7's emm emits mask x (colour x vertex rgb)^2 x this (so alpha^2 x HDR),
				// its em alpha x emissive map^2 x this (so alpha x grey x HDR).
				void generate_emissive_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					if (!ctx.has_material || (ctx.material.shade != effect_material::shading::emissive_colour &&
						ctx.material.shade != effect_material::shading::emissive_mask))
					{
						return;
					}
					const auto vis = get_vis_samples(ctx.elem);
					if (!vis.samples)
					{
						return;
					}
					// BO3 applies hdrScale only on its compute sprite path (0x140205AA0: a compute visual, type 5 or below,
					// neither flag 0x1000 nor 0x20000000); the vertex path (0x140206A80) uses the sample alone.
					const auto compute_path = (static_cast<unsigned int>(ctx.elem->flags) & 0x20001000u) == 0 &&
						ctx.elem->elemType <= T7_ELEM_TYPE_TRAIL && ctx.elem->computeVisuals.instance.anonymous != nullptr;
					const auto hdr_scale = compute_path ? ctx.elem->hdrScale : 1.0f;
					const auto emissive = [&](const t7_vis_state& colour_end, const float hdr)
					{
						const auto a = colour_end.color[3] / 255.0f;
						const auto particle_hdr = ctx.material.old_hdr_scale ? 1.0f : hdr * hdr_scale;
						if (ctx.material.shade == effect_material::shading::emissive_mask)
						{
							return a * a * particle_hdr;
						}
						return a * (colour_end.color[0] / 255.0f) * particle_hdr;
					};
					std::vector<float> min, max;
					for (auto i = 0; i < vis.count; i++)
					{
						const auto& s = vis.samples[i];
						min.push_back(emissive(s.base, s.base.intensity));
						max.push_back(emissive(s.amplitude, s.base.intensity + s.amplitude.intensity));
					}
					auto module = new_module(i7::PARTICLE_MODULE_EMISSIVE_GRAPH);
					auto& data = module.moduleData.emissiveGraph;
					data.firstCurve = false;
					make_curve_pair(ctx, min, max, data.m_curves[0], data.m_curves[1], data.m_flags);
					ctx.state_flags |= state_emissive_graph;
					modules.push_back(module);
				}

				// Sizes are absolute (INIT_ATTRIBUTES keeps stock's 10). Sprites: width size[0], height size[1] (size[0]
				// without non-uniform scale); models: scale on every axis; decals: size[0]; lights: radius and intensity.
				void generate_size_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto vis = get_vis_samples(ctx.elem);
					if (!vis.samples)
					{
						return;
					}
					const auto* elem = ctx.elem;
					const auto nonuniform = (elem->flags & T7_FX_ELEM_NONUNIFORM_SCALE) != 0;

					// the template's attenuation distance for lights (light_template)
					auto light_scale = map::bo3_light_scale;
					if (is_light_type(elem->elemType))
					{
						const auto* visual = elem->visualCount == 1 ? elem->visuals.instance.anonymous : nullptr;
						const auto* config = visual ? reinterpret_cast<const GfxConfig_Light*>(static_cast<const char*>(visual) + 8) : nullptr;
						if (config && is_readable(config, sizeof(GfxConfig_Light)))
						{
							light_scale = config->dAttenuation * config->dAttenuation / omni_light_unit * map::bo3_light_scale;
						}
					}

					enum class channel { none, size0, size1, scale, radius, intensity };
					channel axes[3] = { channel::none, channel::none, channel::none };
					switch (elem->elemType)
					{
					case T7_ELEM_TYPE_SPRITE_BILLBOARD:
					case T7_ELEM_TYPE_SPRITE_ORIENTED:
					case T7_ELEM_TYPE_SPRITE_ROTATED:
					case T7_ELEM_TYPE_TAIL:
					case T7_ELEM_TYPE_LINE:
					case T7_ELEM_TYPE_TRAIL:
					case T7_ELEM_TYPE_CLOUD:
						axes[0] = channel::size0;
						axes[1] = nonuniform ? channel::size1 : channel::size0;
						break;
					case T7_ELEM_TYPE_DECAL:
						axes[0] = channel::size0;
						axes[1] = channel::size0;
						break;
					case T7_ELEM_TYPE_OMNI_LIGHT:
					case T7_ELEM_TYPE_SPOT_LIGHT:
						axes[0] = channel::radius;
						axes[1] = channel::intensity;
						break;
					case T7_ELEM_TYPE_MODEL:
						axes[0] = axes[1] = axes[2] = channel::scale;
						break;
					default:
						return;
					}
					const auto value = [&](const channel which, const t7_vis_state& s)
					{
						switch (which)
						{
						case channel::size0: return s.size[0];
						case channel::size1: return s.size[1];
						case channel::scale: return s.scale;
						case channel::radius: return s.radius;
						case channel::intensity: return s.intensity * light_scale;
						default: return 0.0f;
						}
					};

					auto module = new_module(i7::PARTICLE_MODULE_SIZE_GRAPH);
					auto& data = module.moduleData.sizeGraph;
					data.firstCurve = false;
					for (auto axis = 0; axis < 3; axis++)
					{
						std::vector<float> min, max;
						for (auto i = 0; i < vis.count && axes[axis] != channel::none; i++)
						{
							const auto base = value(axes[axis], vis.samples[i].base);
							min.push_back(base);
							max.push_back(base + value(axes[axis], vis.samples[i].amplitude));
						}
						if (min.empty())
						{
							// an unused axis: stock writes scale 0 over value 1
							for (auto* curve : { &data.m_curves[axis], &data.m_curves[axis + 3] })
							{
								*curve = make_curve(ctx, { 1.0f });
								curve->scale = 0.0f;
							}
							continue;
						}
						make_curve_pair(ctx, min, max, data.m_curves[axis], data.m_curves[axis + 3], data.m_flags);
					}
					modules.push_back(module);
				}

				// BO3's rotation delta is per interval and millisecond: N x delta x 1000 radians a second
				void generate_rotation_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto vis = get_vis_samples(ctx.elem);
					if (!vis.samples)
					{
						return;
					}
					const auto rate = static_cast<float>(vis.intervals()) * 1000.0f;
					std::vector<float> min, max;
					auto any = false;
					for (auto i = 0; i < vis.count; i++)
					{
						const auto base = vis.samples[i].base.rotation_delta;
						const auto ampl = vis.samples[i].amplitude.rotation_delta;
						min.push_back(base * rate);
						max.push_back((base + ampl) * rate);
						any |= base != 0.0f || ampl != 0.0f;
					}
					if (!any)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_ROTATION_GRAPH);
					auto& data = module.moduleData.rotationGraph;
					data.m_useRotationRate = true;
					make_curve_pair(ctx, min, max, data.m_curves[0], data.m_curves[1], data.m_flags);
					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_ROTATION_1D_CURVE;
					modules.push_back(module);
				}

				// BO3 applies local and world velocity together: one IW7 graph for each space that moves
				void generate_velocity_modules(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto vel = get_vel_samples(ctx.elem);
					if (!vel.samples)
					{
						return;
					}
					const auto rate = static_cast<float>(vel.intervals()) * 1000.0f;
					for (const auto world : { false, true })
					{
						const auto flag = world ? T7_FX_ELEM_HAS_VELOCITY_GRAPH_WORLD : T7_FX_ELEM_HAS_VELOCITY_GRAPH_LOCAL;
						if ((ctx.elem->flags & flag) == 0)
						{
							continue;
						}
						std::vector<float> min[3], max[3];
						auto moves = false;
						for (auto i = 0; i < vel.count; i++)
						{
							const auto& frame = world ? vel.samples[i].world : vel.samples[i].local;
							for (auto axis = 0; axis < 3; axis++)
							{
								min[axis].push_back(frame.velocity_base[axis] * rate);
								max[axis].push_back((frame.velocity_base[axis] + frame.velocity_amplitude[axis]) * rate);
								moves |= frame.velocity_base[axis] != 0.0f || frame.velocity_amplitude[axis] != 0.0f;
							}
						}
						if (!moves)
						{
							continue;
						}
						auto module = new_module(i7::PARTICLE_MODULE_VELOCITY_GRAPH);
						auto& data = module.moduleData.velocityGraph;
						data.m_flags |= world ? i7::PARTICLE_MODULE_FLAG_USE_WORLD_SPACE : 0;
						for (auto axis = 0; axis < 3; axis++)
						{
							make_curve_pair(ctx, min[axis], max[axis], data.m_curves[axis], data.m_curves[axis + 3], data.m_flags);
						}
						ctx.state_flags |= world ? i7::PARTICLE_STATE_DEF_FLAG_HAS_VELOCITY_CURVE_WORLD1 : i7::PARTICLE_STATE_DEF_FLAG_HAS_VELOCITY_CURVE_LOCAL1;
						modules.push_back(module);
					}
				}

				void generate_gravity_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					if (elem->gravity.base == 0.0f && elem->gravity.amplitude == 0.0f)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_GRAVITY);
					auto& data = module.moduleData.gravity;
					data.m_gravityPercentage.min = elem->gravity.base;
					data.m_gravityPercentage.max = elem->gravity.base + elem->gravity.amplitude;
					modules.push_back(module);
				}

				// IW7 draws a tail only once a velocity component exceeds 0.001 units a second (0x140D095B0); BO3 draws a line
				// along whatever velocity it has (0x140201950). Slow velocity graphs are scaled up to 0.002, and a line only
				// gravity moves gets a constant world velocity of 0.002 units a second along gravity.
				void generate_tail_orientation_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					if (elem->elemType != T7_ELEM_TYPE_TAIL && elem->elemType != T7_ELEM_TYPE_LINE)
					{
						return;
					}
					// graphs orient it; ones that never reach the threshold are scaled up to it, keeping BO3's direction
					auto peak = 0.0f;
					auto graphs = false;
					for (const auto& m : modules)
					{
						if (m.moduleType != i7::PARTICLE_MODULE_VELOCITY_GRAPH)
						{
							continue;
						}
						graphs = true;
						for (const auto& c : m.moduleData.velocityGraph.m_curves)
						{
							for (auto k = 0; k < c.numControlPoints; k++)
							{
								peak = std::max(peak, std::fabs(c.controlPoints[k].value * c.scale));
							}
						}
					}
					if (graphs)
					{
						if (peak > 0.0f && peak < 0.002f)
						{
							for (auto& m : modules)
							{
								if (m.moduleType == i7::PARTICLE_MODULE_VELOCITY_GRAPH)
								{
									for (auto& c : m.moduleData.velocityGraph.m_curves)
									{
										c.scale *= 0.002f / peak;
									}
								}
							}
						}
						return;
					}
					// the direction of the range's middle
					const auto g_mid = elem->gravity.base + elem->gravity.amplitude * 0.5f;
					if (g_mid == 0.0f)
					{
						return;
					}
					constexpr auto speed = 0.002f;
					const auto up = g_mid < 0.0f ? speed : -speed; // BO3 gravity pulls along -z
					auto module = new_module(i7::PARTICLE_MODULE_VELOCITY_GRAPH);
					auto& data = module.moduleData.velocityGraph;
					data.m_flags |= i7::PARTICLE_MODULE_FLAG_USE_WORLD_SPACE;
					for (auto axis = 0; axis < 3; axis++)
					{
						const std::vector<float> value = { axis == 2 ? up : 0.0f };
						make_curve_pair(ctx, value, value, data.m_curves[axis], data.m_curves[axis + 3], data.m_flags);
					}
					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_VELOCITY_CURVE_WORLD1;
					modules.push_back(module);
				}

				// ---- init modules --------------------------------------------------------------------------

				void generate_init_spawn_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					// stock's spawn curve (1, 1, 0 at 0, 0.75, 1)
					auto module = new_module(i7::PARTICLE_MODULE_INIT_SPAWN);
					auto& data = module.moduleData.initSpawn;
					data.m_curves[0] = make_curve(ctx, { 1.0f, 1.0f, 0.0f });
					data.m_curves[0].controlPoints[1].time = 0.75f;
					data.m_curves[0].controlPoints[1].invTimeDelta = 1.0f / 0.75f;
					data.m_curves[0].controlPoints[2].invTimeDelta = 1.0f / 0.25f;
					modules.push_back(module);
				}

				void generate_init_attributes_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					auto module = new_module(i7::PARTICLE_MODULE_INIT_ATTRIBUTES);
					auto& data = module.moduleData.initAttributes;
					for (auto i = 0; i < 3; i++)
					{
						data.m_sizeMin.v[i] = 10.0f;
						data.m_sizeMax.v[i] = 10.0f;
					}
					for (auto i = 0; i < 4; i++)
					{
						data.m_colorMin.v[i] = 1.0f;
						data.m_colorMax.v[i] = 1.0f;
					}
					modules.push_back(module);
				}

				void generate_init_relative_velocity_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					auto module = new_module(i7::PARTICLE_MODULE_INIT_RELATIVE_VELOCITY);
					auto& data = module.moduleData.initRelativeVelocity;
					switch (ctx.elem->flags & T7_FX_ELEM_RUN_MASK)
					{
					case T7_FX_ELEM_RUN_RELATIVE_TO_SPAWN:
					case T7_FX_ELEM_RUN_RELATIVE_TO_EFFECT:
						data.m_velocityType = i7::PARTICLE_RELATIVE_VELOCITY_TYPE_LOCAL;
						break;
					case T7_FX_ELEM_RUN_RELATIVE_TO_OFFSET:
						data.m_velocityType = i7::PARTICLE_RELATIVE_VELOCITY_TYPE_RELATIVE_TO_EFFECT_ORIGIN;
						break;
					default:
						data.m_velocityType = i7::PARTICLE_RELATIVE_VELOCITY_TYPE_WORLD;
						break;
					}
					// a trail, and both parts of an aimed spot light (the light follows its runner particle, as stock's)
					if (ctx.elem->elemType == T7_ELEM_TYPE_TRAIL || ctx.element_part != part::whole)
					{
						data.m_velocityType = i7::PARTICLE_RELATIVE_VELOCITY_TYPE_LOCAL_WITH_BOLT_INFO;
						data.m_useBoltInfo = true;
					}
					modules.push_back(module);
				}

				void generate_init_rotation_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto& r = ctx.elem->initialRotation;
					if (r.base == 0.0f && r.amplitude == 0.0f)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_ROTATION);
					auto& data = module.moduleData.initRotation;
					data.m_rotationAngle.min = r.base;
					data.m_rotationAngle.max = r.base + r.amplitude;
					// stock clouds never carry the flag; a BO3 cloud drawn as a billboard (its material is not a cloud one) does
					if (!ctx.has_material || ctx.material.shade != effect_material::shading::cloud)
					{
						ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_ROTATION_1D_INIT;
					}
					modules.push_back(module);
				}

				// spawn angles (radians) and angular velocity (radians a millisecond)
				void generate_init_rotation3d_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					auto any = false;
					for (auto i = 0; i < 3; i++)
					{
						any |= elem->spawnAngles[i].base != 0.0f || elem->spawnAngles[i].amplitude != 0.0f ||
							elem->angularVelocity[i].base != 0.0f || elem->angularVelocity[i].amplitude != 0.0f;
					}
					if (!any)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_ROTATION_3D);
					auto& data = module.moduleData.initRotation3D;
					for (auto i = 0; i < 3; i++)
					{
						data.m_rotationAngleMin.v[i] = elem->spawnAngles[i].base;
						data.m_rotationAngleMax.v[i] = elem->spawnAngles[i].base + elem->spawnAngles[i].amplitude;
						data.m_rotationRateMin.v[i] = elem->angularVelocity[i].base * 1000.0f;
						data.m_rotationRateMax.v[i] = (elem->angularVelocity[i].base + elem->angularVelocity[i].amplitude) * 1000.0f;
					}
					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_ROTATION_3D_INIT;
					modules.push_back(module);
				}

				// BO3's atlas as IW7's INIT_ATLAS (sub_D0F950 reads the two bytes after loopCount); frame blending is the
				// material's
				void generate_init_atlas_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					if (!ctx.has_material || ctx.material.shade == effect_material::shading::cloud)
					{
						return;
					}
					const auto& atlas = ctx.elem->atlas;
					if ((atlas.behavior & atlas_enabled) == 0)
					{
						return;
					}
					const auto frames = 1 << (atlas.colIndexBits + atlas.rowIndexBits);
					const auto start = atlas.behavior & atlas_start_mask;
					auto module = new_module(i7::PARTICLE_MODULE_INIT_ATLAS);
					auto& data = module.moduleData.initAtlas;
					auto* extra = reinterpret_cast<std::uint8_t*>(&data) + sizeof(i7::ParticleModuleInitAtlas);
					if (ctx.material.atlas_slots)
					{
						// the material's atlas is the element's frame range in play order (effect_material.cpp): played from its
						// first frame forwards, a loop as long as BO3's
						data.m_playRate = static_cast<int>(std::lround(atlas.fps * static_cast<double>(ctx.material.atlas_slots) /
							ctx.material.atlas_range));
						data.m_startFrame = 0;
						data.m_loopCount = (atlas.behavior & atlas_loop_only_n_times) ? atlas.loopCount : -1;
						extra[0] = 0;
						extra[1] = (atlas.behavior & atlas_play_over_life) ? 1 : 0;
						modules.push_back(module);
						return;
					}
					if (start == atlas_start_range)
					{
						ZONETOOL_WARNING("effect \"%s\" element %d: its atlas plays frames %d to %d only; IW7 plays on through the others",
							ctx.effect->name, ctx.elem_index, atlas.index, atlas.index + atlas.indexRange - 1);
					}
					// BO3 reverses by mirroring the frame (count - 1 - frame), IW7 by a negative rate from the start frame
					const auto reverse = (atlas.behavior & atlas_reverse) != 0;
					data.m_playRate = reverse ? -static_cast<int>(atlas.fps) : atlas.fps;
					data.m_startFrame = reverse ? frames - 1 - atlas.index : atlas.index;
					data.m_loopCount = (atlas.behavior & atlas_loop_only_n_times) ? atlas.loopCount : -1;
					// random, or the element's sequence (IW7 has no sequence start): a random frame
					extra[0] = start == 1 || start == 2 ? 1 : 0;
					extra[1] = (atlas.behavior & atlas_play_over_life) ? 1 : 0;
					modules.push_back(module);
				}

				// the IW7 materials of a sprite element's visuals; false when none converts
				bool generate_init_material_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					std::vector<const Material*> materials;
					if (elem->visualCount == 1)
					{
						materials.push_back(elem->visuals.instance.material);
					}
					else if (elem->visualCount > 1 && is_readable(elem->visuals.array, sizeof(FxElemVisuals) * elem->visualCount))
					{
						for (auto i = 0; i < elem->visualCount; i++)
						{
							materials.push_back(elem->visuals.array[i].material);
						}
					}

					std::vector<std::string> names;
					for (const auto* material : materials)
					{
						if (!is_readable(material, sizeof(Material)))
						{
							continue;
						}
						const auto converted = ctx.refs->material(material, elem);
						if (converted.name.empty())
						{
							continue;
						}
						if (!ctx.has_material)
						{
							ctx.material = converted;
							ctx.has_material = true;
						}
						else if (converted.shade != ctx.material.shade)
						{
							ZONETOOL_WARNING("effect \"%s\" element %d: its visuals shade differently (%s); IW7 draws them all as the first one",
								ctx.effect->name, ctx.elem_index, converted.name.data());
						}
						names.push_back(converted.name);
					}
					if (names.empty())
					{
						return false;
					}

					auto module = new_module(i7::PARTICLE_MODULE_INIT_MATERIAL);
					auto& data = module.moduleData.initMaterial;
					data.m_linkedAssetList.numAssets = static_cast<int>(names.size());
					data.m_linkedAssetList.assetList = ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(names.size());
					for (auto i = 0u; i < names.size(); i++)
					{
						data.m_linkedAssetList.assetList[i].material = alias_asset<i7::Material>(names[i], *ctx.allocator);
					}
					modules.push_back(module);

					if (ctx.material.shade == effect_material::shading::cloud)
					{
						ctx.system_flags |= i7::PARTICLE_SYSTEM_DEF_FLAG_HAS_NON_SPRITES;
					}
					else
					{
						ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_IS_SPRITE;
						ctx.system_flags |= i7::PARTICLE_SYSTEM_DEF_FLAG_HAS_SPRITES;
					}
					if (ctx.material.shade == effect_material::shading::lit || ctx.material.shade == effect_material::shading::emissive_colour ||
						ctx.material.shade == effect_material::shading::emissive_mask)
					{
						ctx.state_flags |= state_lit;
					}
					return true;
				}

				// BO3's oriented sprites face along the effect's axis; rotated sprites turn by their random spawn angles on
				// top (INIT_ROTATION_3D)
				void generate_init_oriented_sprite_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					if (ctx.elem->elemType != T7_ELEM_TYPE_SPRITE_ORIENTED && ctx.elem->elemType != T7_ELEM_TYPE_SPRITE_ROTATED)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_ORIENTED_SPRITE);
					auto& data = module.moduleData.initOrientedSprite;
					data.m_orientationQuat.v[0] = 0.5f;
					data.m_orientationQuat.v[1] = 0.5f;
					data.m_orientationQuat.v[2] = 0.5f;
					data.m_orientationQuat.v[3] = 0.5f;
					if ((ctx.elem->flags & T7_FX_ELEM_RUN_MASK) == T7_FX_ELEM_RUN_RELATIVE_TO_SPAWN)
					{
						data.m_orientationQuat.v[1] *= -1.0f;
						data.m_orientationQuat.v[2] *= -1.0f;
					}
					modules.push_back(module);
				}

				// BO3's tail sits behind its particle, its line ahead (see the header comment)
				void generate_init_tail_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					if (ctx.elem->elemType != T7_ELEM_TYPE_TAIL && ctx.elem->elemType != T7_ELEM_TYPE_LINE)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_TAIL);
					auto& data = module.moduleData.initTail;
					data.m_tailLeading = ctx.elem->elemType == T7_ELEM_TYPE_TAIL;
					modules.push_back(module);
				}

				void generate_init_cloud_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					if (!ctx.has_material || ctx.material.shade != effect_material::shading::cloud)
					{
						return;
					}
					const auto vis = get_vis_samples(ctx.elem);
					auto module = new_module(i7::PARTICLE_MODULE_INIT_CLOUD);
					auto& data = module.moduleData.initCloud;
					std::vector<float> min, max;
					for (auto i = 0; i < vis.count; i++)
					{
						min.push_back(vis.samples[i].base.scale);
						max.push_back(vis.samples[i].base.scale + vis.samples[i].amplitude.scale);
					}
					auto any = false;
					for (const auto v : max)
					{
						any |= v != 0.0f;
					}
					if (!any)
					{
						min = max = { 1.0f };
					}
					make_curve_pair(ctx, min, max, data.curves[0], data.curves[1], data.m_flags);
					modules.push_back(module);
				}

				void generate_init_light_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					if (!is_light_type(elem->elemType))
					{
						return;
					}
					const auto* visual = elem->visualCount == 1 ? elem->visuals.instance.anonymous : nullptr;
					const auto* config = visual ? reinterpret_cast<const GfxConfig_Light*>(static_cast<const char*>(visual) + 8) : nullptr;
					if (config && !is_readable(config, sizeof(GfxConfig_Light)))
					{
						config = nullptr;
					}
					// the template's type decides (BO3 draws type 4 as omni, others with the particle's cone)
					const auto spot = config && config->type != 4;

					ctx.system_flags |= i7::PARTICLE_SYSTEM_DEF_FLAG_HAS_LIGHTS | i7::PARTICLE_SYSTEM_DEF_FLAG_HAS_NON_SPRITES;
					ctx.emitter_flags |= i7::PARTICLE_EMITTER_DEF_FLAG_HAS_LIGHTS;

					i7::ParticleLinkedAssetListDef list{};
					list.numAssets = 1;
					list.assetList = ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(1);
					list.assetList[0].lightDef = alias_asset<i7::GfxLightDef>(light_def, *ctx.allocator);

					if (!spot)
					{
						auto module = new_module(i7::PARTICLE_MODULE_INIT_LIGHT_OMNI);
						auto& data = module.moduleData.initLightOmni;
						data.m_flags |= i7::PARTICLE_MODULE_FLAG_HAS_LIGHT_DEFS;
						data.m_linkedAssetList = list;
						data.m_tonemappingScaleFactor = 0.0f;
						modules.push_back(module);
						return;
					}

					const auto vis = get_vis_samples(elem);
					const auto cone = vis.samples ? vis.samples[0].base.cone : 90.0f;
					auto module = new_module(i7::PARTICLE_MODULE_INIT_LIGHT_SPOT);
					auto& data = module.moduleData.initLightSpot;
					data.m_flags |= i7::PARTICLE_MODULE_FLAG_HAS_LIGHT_DEFS;
					data.m_linkedAssetList = list;
					data.m_fovOuter = cone * 0.5f * 0.017453292f;
					data.m_fovInner = data.m_fovOuter * std::clamp(config->near_edge, 0.0f, 1.0f);
					data.m_bulbRadius = 1.0f;
					data.m_bulbLength = 0.0f;
					data.m_brightness = 1.0f;
					data.m_toneMappingScaleFactor = 0.0f;
					data.m_disableShadowMap = true;
					data.m_disableDynamicShadows = true;
					modules.push_back(module);
				}

				void generate_init_model_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					if (elem->elemType != T7_ELEM_TYPE_MODEL || !elem->visualCount)
					{
						return;
					}
					std::vector<std::string> names;
					for (auto i = 0; i < elem->visualCount; i++)
					{
						const auto* model = elem->visualCount == 1 ? elem->visuals.instance.model : elem->visuals.array[i].model;
						if (const auto* name = asset_name(model))
						{
							names.push_back(name);
						}
					}
					if (names.empty())
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_MODEL);
					auto& data = module.moduleData.initModel;
					data.m_linkedAssetList.numAssets = static_cast<int>(names.size());
					data.m_linkedAssetList.assetList = ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(names.size());
					for (auto i = 0u; i < names.size(); i++)
					{
						data.m_linkedAssetList.assetList[i].model = alias_asset<i7::XModel>(names[i], *ctx.allocator);
					}
					ctx.system_flags |= i7::PARTICLE_SYSTEM_DEF_FLAG_HAS_NON_SPRITES;
					modules.push_back(module);
				}

				// a runner spawns its visual effects as children (an aimed spot light's runner: the effect holding its light)
				void generate_init_runner_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					std::vector<std::string> names;
					if (ctx.element_part == part::spot_runner)
					{
						names.push_back(ctx.light_child);
					}
					else if (elem->elemType == T7_ELEM_TYPE_RUNNER)
					{
						for (auto i = 0; i < elem->visualCount; i++)
						{
							const auto& visual = elem->visualCount == 1 ? elem->visuals.instance : elem->visuals.array[i];
							const auto* name = asset_name(visual.effectDef.handle);
							if (name && *name)
							{
								names.push_back(ctx.refs->effect(name));
							}
						}
					}
					if (names.empty())
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_RUNNER);
					auto& data = module.moduleData.initRunner;
					data.m_linkedAssetList.numAssets = static_cast<int>(names.size());
					data.m_linkedAssetList.assetList = ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(names.size());
					for (auto i = 0u; i < names.size(); i++)
					{
						data.m_linkedAssetList.assetList[i].particleSystem = alias_asset<i7::ParticleSystemDef>(names[i], *ctx.allocator);
					}
					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_CHILD_EFFECTS;
					modules.push_back(module);
				}

				// spawnOrigin: a box of offsets per axis (base + random x amplitude), in world axes unless the element spawns
				// relative to the effect
				void generate_init_spawn_shape_box_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					auto any = false;
					for (auto i = 0; i < 3; i++)
					{
						any |= elem->spawnOrigin[i].base != 0.0f || elem->spawnOrigin[i].amplitude != 0.0f;
					}
					if (!any)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_SPAWN_SHAPE_BOX);
					auto& data = module.moduleData.initSpawnShapeBox;
					if ((elem->flags & T7_FX_ELEM_SPAWN_RELATIVE_TO_EFFECT) == 0 && elem->elemType != T7_ELEM_TYPE_TRAIL)
					{
						data.m_flags |= i7::PARTICLE_MODULE_FLAG_USE_WORLD_SPACE;
					}
					data.m_axisFlags = i7::PARTICLE_MODULE_AXES_FLAG_ALL;
					for (auto i = 0; i < 3; i++)
					{
						data.m_dimensionsMin.v[i] = elem->spawnOrigin[i].base;
						data.m_dimensionsMax.v[i] = elem->spawnOrigin[i].base + elem->spawnOrigin[i].amplitude;
					}
					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_SPAWN_SHAPE;
					modules.push_back(module);
				}

				void generate_init_spawn_shape_sphere_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					if ((elem->flags & T7_FX_ELEM_SPAWN_OFFSET_MASK) != T7_FX_ELEM_SPAWN_OFFSET_SPHERE)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_SPAWN_SHAPE_SPHERE);
					auto& data = module.moduleData.initSpawnShapeSphere;
					data.m_axisFlags = i7::PARTICLE_MODULE_AXES_FLAG_ALL;
					data.m_radius.min = elem->spawnOffsetRadius.base;
					data.m_radius.max = elem->spawnOffsetRadius.base + elem->spawnOffsetRadius.amplitude;
					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_SPAWN_SHAPE;
					modules.push_back(module);
				}

				// IW7's axis quaternion, half height from the height range
				void generate_init_spawn_shape_cylinder_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* elem = ctx.elem;
					if ((elem->flags & T7_FX_ELEM_SPAWN_OFFSET_MASK) != T7_FX_ELEM_SPAWN_OFFSET_CYLINDER)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_SPAWN_SHAPE_CYLINDER);
					auto& data = module.moduleData.initSpawnShapeCylinder;
					data.m_axisFlags = i7::PARTICLE_MODULE_AXES_FLAG_ALL;
					data.m_hasRotation = true;
					data.m_directionQuat.v[1] = 0.7071067690849304f;
					data.m_directionQuat.v[3] = 0.7071067690849304f;
					data.m_radius.min = elem->spawnOffsetRadius.base;
					data.m_radius.max = elem->spawnOffsetRadius.base + elem->spawnOffsetRadius.amplitude;
					data.m_halfHeight = elem->spawnOffsetHeight.amplitude * 0.5f;
					data.unk.v[0] = elem->spawnOffsetHeight.base + data.m_halfHeight;
					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_SPAWN_SHAPE;
					modules.push_back(module);
				}

				void generate_init_geo_trail_module(convert_context& ctx, std::vector<i7::ParticleModuleDef>& modules)
				{
					if (ctx.elem->elemType != T7_ELEM_TYPE_TRAIL)
					{
						return;
					}
					auto module = new_module(i7::PARTICLE_MODULE_INIT_GEO_TRAIL);
					auto& data = module.moduleData.initGeoTrail;
					data.m_numPointsMax = 16;
					data.m_numSheets = 2;
					data.m_splitDistance = 8.0f;
					data.m_tileDistance = 8.0f;
					const auto* trail = static_cast<const t7_trail_def*>(ctx.elem->extended.trailDef);
					if (is_readable(trail, sizeof(t7_trail_def)))
					{
						data.m_splitDistance = static_cast<float>(trail->repeat_dist);
						data.m_tileDistance = static_cast<float>(trail->repeat_dist);
						data.m_scrollTime = trail->scroll_time_msec / 1000.0f;
						data.m_fadeInDistance = trail->fade_in_dist;
						data.m_fadeOutDistance = trail->fade_out_dist;
					}
					modules.push_back(module);
				}

				// ---- test modules: the effects particles spawn -----------------------------------------------

				void generate_test_module(convert_context& ctx, const FxEffectDefRef ref, const i7::ParticleModuleType type,
					const bool kill, std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* name = asset_name(ref.handle);
					if (!name || !*name)
					{
						return;
					}
					auto module = new_module(type);
					auto& data = module.moduleData.testDeath;
					data.m_moduleIndex = static_cast<unsigned short>(ctx.test_module_index++);
					data.m_eventHandlerData.m_kill = kill;
					data.m_eventHandlerData.m_linkedAssetList.numAssets = 1;
					data.m_eventHandlerData.m_linkedAssetList.assetList = ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(1);
					data.m_eventHandlerData.m_linkedAssetList.assetList[0].particleSystem =
						alias_asset<i7::ParticleSystemDef>(ctx.refs->effect(name), *ctx.allocator);
					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_CHILD_EFFECTS;
					if (type == i7::PARTICLE_MODULE_TEST_IMPACT)
					{
						ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HANDLE_ON_IMPACT;
					}
					modules.push_back(module);
				}

				void store_group(convert_context& ctx, i7::ParticleModuleGroupDef* group, const std::vector<i7::ParticleModuleDef>& modules)
				{
					group->numModules = static_cast<int>(modules.size());
					group->disabled = false;
					group->moduleDefs = nullptr;
					if (!modules.empty())
					{
						group->moduleDefs = ctx.allocator->allocate_array<i7::ParticleModuleDef>(modules.size());
						std::memcpy(group->moduleDefs, modules.data(), modules.size() * sizeof(i7::ParticleModuleDef));
					}
				}

				// ---- emitters --------------------------------------------------------------------------------

				unsigned int convert_elem_type(const FxElemDef* elem, const convert_context& ctx)
				{
					switch (elem->elemType)
					{
					case T7_ELEM_TYPE_SPRITE_BILLBOARD: return i7::PARTICLE_ELEMENT_TYPE_BILLBOARD_SPRITE;
					case T7_ELEM_TYPE_SPRITE_ORIENTED:
					case T7_ELEM_TYPE_SPRITE_ROTATED: return i7::PARTICLE_ELEMENT_TYPE_ORIENTED_SPRITE;
					case T7_ELEM_TYPE_TAIL:
					case T7_ELEM_TYPE_LINE: return i7::PARTICLE_ELEMENT_TYPE_TAIL;
					case T7_ELEM_TYPE_TRAIL: return i7::PARTICLE_ELEMENT_TYPE_GEO_TRAIL;
					case T7_ELEM_TYPE_CLOUD:
						// IW7 draws a cloud with particle_cloud techsets only (a quad material draws other effects' records)
						return ctx.has_material && ctx.material.shade == effect_material::shading::cloud
							? i7::PARTICLE_ELEMENT_TYPE_CLOUD : i7::PARTICLE_ELEMENT_TYPE_BILLBOARD_SPRITE;
					case T7_ELEM_TYPE_MODEL: return i7::PARTICLE_ELEMENT_TYPE_MODEL;
					case T7_ELEM_TYPE_OMNI_LIGHT:
					case T7_ELEM_TYPE_SPOT_LIGHT:
					{
						const auto* visual = elem->visualCount == 1 ? elem->visuals.instance.anonymous : nullptr;
						const auto* config = visual ? reinterpret_cast<const GfxConfig_Light*>(static_cast<const char*>(visual) + 8) : nullptr;
						const auto spot = config && is_readable(config, sizeof(GfxConfig_Light)) && config->type != 4;
						return spot ? i7::PARTICLE_ELEMENT_TYPE_LIGHT_SPOT : i7::PARTICLE_ELEMENT_TYPE_LIGHT_OMNI;
					}
					case T7_ELEM_TYPE_DECAL: return i7::PARTICLE_ELEMENT_TYPE_DECAL;
					case T7_ELEM_TYPE_RUNNER: return i7::PARTICLE_ELEMENT_TYPE_RUNNER;
					default: return i7::PARTICLE_ELEMENT_TYPE_BILLBOARD_SPRITE;
					}
				}

				// false: the element is left out (said why)
				bool convert_elem(convert_context& ctx, i7::ParticleEmitterDef* emitter, const bool looping)
				{
					const auto* elem = ctx.elem;
					ctx.emitter_flags = 0;
					ctx.state_flags = 0;
					ctx.has_material = false;
					ctx.material = {};
					const auto whole = ctx.element_part == part::whole;
					const auto runner_part = ctx.element_part == part::spot_runner;
					const auto light_part = ctx.element_part == part::spot_light;
					const auto no_visual = whole && is_sprite_type(elem->elemType) && !elem->visualCount;

					// the init group first: the material module decides how the rest converts
					std::vector<i7::ParticleModuleDef> init_modules;
					generate_init_spawn_module(ctx, init_modules);
					generate_init_attributes_module(ctx, init_modules);
					if (no_visual)
					{
						// BO3 draws nothing for it; it still spawns what its particles spawn: then a billboard without a
						// material (as stock's material-less sound billboards) carries them
						const auto spawns = asset_name(elem->effectOnDeath.handle) || asset_name(elem->effectOnImpact.handle) ||
							asset_name(elem->effectEmitted.handle);
						if (!spawns)
						{
							ZONETOOL_INFO("effect \"%s\": element %d has no visuals and spawns nothing; left out (BO3 draws nothing for it)",
								ctx.effect->name, ctx.elem_index);
							return false;
						}
						ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_IS_SPRITE;
						ctx.system_flags |= i7::PARTICLE_SYSTEM_DEF_FLAG_HAS_SPRITES;
					}
					else if (whole && is_sprite_type(elem->elemType) && !generate_init_material_module(ctx, init_modules))
					{
						ZONETOOL_WARNING("effect \"%s\": element %d (type %u) has no material that converts; left out", ctx.effect->name,
							ctx.elem_index, elem->elemType);
						return false;
					}
					if (whole && !no_visual)
					{
						generate_init_cloud_module(ctx, init_modules);
						generate_init_tail_module(ctx, init_modules);
						generate_init_geo_trail_module(ctx, init_modules);
					}
					if (!runner_part)
					{
						generate_init_light_module(ctx, init_modules);
					}
					if (whole)
					{
						generate_init_model_module(ctx, init_modules);
					}
					if (!light_part)
					{
						generate_init_runner_module(ctx, init_modules);
					}
					if (whole && !no_visual)
					{
						generate_init_oriented_sprite_module(ctx, init_modules);
						generate_init_atlas_module(ctx, init_modules);
					}
					generate_init_relative_velocity_module(ctx, init_modules);
					if (whole)
					{
						generate_init_rotation_module(ctx, init_modules);
					}
					if (!light_part)
					{
						generate_init_rotation3d_module(ctx, init_modules);
						generate_init_spawn_shape_box_module(ctx, init_modules);
						generate_init_spawn_shape_cylinder_module(ctx, init_modules);
						generate_init_spawn_shape_sphere_module(ctx, init_modules);
					}

					std::vector<i7::ParticleModuleDef> update_modules;
					if (!runner_part)
					{
						generate_color_module(ctx, update_modules);
						generate_emissive_module(ctx, update_modules);
						generate_size_module(ctx, update_modules);
					}
					if (whole)
					{
						generate_rotation_module(ctx, update_modules);
					}
					if (!light_part)
					{
						generate_velocity_modules(ctx, update_modules);
						generate_gravity_module(ctx, update_modules);
					}
					if (whole && !no_visual)
					{
						generate_tail_orientation_module(ctx, update_modules);
					}

					ctx.test_module_index = 0;
					std::vector<i7::ParticleModuleDef> test_modules;
					if (!light_part)
					{
						generate_test_module(ctx, elem->effectOnDeath, i7::PARTICLE_MODULE_TEST_DEATH, false, test_modules);
						generate_test_module(ctx, elem->effectOnImpact, i7::PARTICLE_MODULE_TEST_IMPACT, false, test_modules);
						generate_test_module(ctx, elem->effectEmitted, i7::PARTICLE_MODULE_TEST_BIRTH, false, test_modules);
					}

					*emitter = {};
					emitter->particleSpawnRate = { 5.0f, 5.0f };
					emitter->particleBurstCount = { 1, 1 };
					emitter->particleLife.min = elem->lifeSpanMsec.base / 1000.0f;
					emitter->particleLife.max = (elem->lifeSpanMsec.base + elem->lifeSpanMsec.amplitude) / 1000.0f;
					emitter->particleDelay.min = elem->spawnDelayMsec.base / 1000.0f;
					emitter->particleDelay.max = (elem->spawnDelayMsec.base + elem->spawnDelayMsec.amplitude) / 1000.0f;

					if (light_part)
					{
						// one light for its runner particle's life, as stock's child (burst of 1, no delay)
						emitter->particleSpawnRate = { 1.0f, 1.0f };
						emitter->particleDelay = { 0.0f, 0.0f };
						emitter->particleCountMax = 1;
						ctx.emitter_flags |= i7::PARTICLE_EMITTER_DEF_FLAG_USE_BURST_MODE;
					}
					else if (looping)
					{
						// spawnCount particles every intervalMsec, count times (0x7FFFFFFF: for ever)
						const auto interval = std::max(1, elem->spawn.looping.intervalMsec);
						const auto per_second = 1000.0f / static_cast<float>(interval);
						const auto spawn_min = std::max(0, elem->spawn.looping.spawnCount.base);
						const auto spawn_max = std::max(spawn_min, elem->spawn.looping.spawnCount.base + elem->spawn.looping.spawnCount.amplitude);
						emitter->particleSpawnRate.min = spawn_min * per_second;
						emitter->particleSpawnRate.max = spawn_max * per_second;
						auto alive = static_cast<int>(std::ceil(emitter->particleSpawnRate.max * std::max(0.0f, emitter->particleLife.max))) + 1;
						if (elem->spawn.looping.count != 0x7FFFFFFF)
						{
							const auto count = std::max(1, elem->spawn.looping.count);
							emitter->emitterLife.min = emitter->emitterLife.max = count * interval / 1000.0f;
							alive = std::min(alive, count * std::max(1, spawn_max));
						}
						emitter->particleCountMax = static_cast<unsigned int>(std::max(1, alive));
					}
					else
					{
						emitter->particleBurstCount.min = elem->spawn.oneShot.count.base;
						emitter->particleBurstCount.max = elem->spawn.oneShot.count.base + elem->spawn.oneShot.count.amplitude;
						emitter->particleCountMax = static_cast<unsigned int>(std::max(1, emitter->particleBurstCount.max));
						ctx.emitter_flags |= i7::PARTICLE_EMITTER_DEF_FLAG_USE_BURST_MODE;
					}

					if (elem->fadeInRange.amplitude != 0.0f || elem->fadeOutRange.amplitude != 0.0f)
					{
						ZONETOOL_INFO("effect \"%s\": element %d fades by camera distance: in [%g, %g], out [%g, %g] (not converted)",
							ctx.effect->name, ctx.elem_index, elem->fadeInRange.base, elem->fadeInRange.base + elem->fadeInRange.amplitude,
							elem->fadeOutRange.base, elem->fadeOutRange.base + elem->fadeOutRange.amplitude);
					}
					// BO3 spawns only while the camera is within [base, base + amplitude] (no range when the amplitude is 0);
					// its fade in / out ranges are left out (IW7's fadeOutMaxDistance is not mapped). An aimed light's
					// runner does the spawning.
					if (elem->spawnRange.amplitude != 0.0f && !light_part)
					{
						const auto range_min = std::max(0.0f, elem->spawnRange.base);
						const auto range_max = elem->spawnRange.base + elem->spawnRange.amplitude;
						emitter->spawnRangeSq.min = range_min * range_min;
						emitter->spawnRangeSq.max = range_max * range_max;
					}
					emitter->spawnFrustumCullRadius = light_part ? 0.0f : elem->spawnFrustumCullRadius;
					emitter->unk2 = 100.0f;
					if (elem->flags & T7_FX_ELEM_DRAW_PAST_FOG)
					{
						ctx.emitter_flags |= i7::PARTICLE_EMITTER_DEF_FLAG_DRAW_PAST_FOG;
					}

					emitter->numStates = 1;
					emitter->stateDefs = ctx.allocator->allocate<i7::ParticleStateDef>();
					auto* state = emitter->stateDefs;
					state->elementType = runner_part ? i7::PARTICLE_ELEMENT_TYPE_RUNNER
						: light_part ? i7::PARTICLE_ELEMENT_TYPE_LIGHT_SPOT
						: no_visual ? i7::PARTICLE_ELEMENT_TYPE_BILLBOARD_SPRITE : convert_elem_type(elem, ctx);
					state->moduleGroupDefs = ctx.allocator->allocate_array<i7::ParticleModuleGroupDef>(i7::PARTICLE_MODULE_GROUP_COUNT);
					store_group(ctx, &state->moduleGroupDefs[i7::PARTICLE_MODULE_GROUP_INIT], init_modules);
					store_group(ctx, &state->moduleGroupDefs[i7::PARTICLE_MODULE_GROUP_UPDATE], update_modules);
					store_group(ctx, &state->moduleGroupDefs[i7::PARTICLE_MODULE_GROUP_TEST], test_modules);

					emitter->flags = ctx.emitter_flags;
					state->flags = ctx.state_flags;
					return true;
				}

				std::string identity(const std::string& name)
				{
					return name;
				}

				effect_material::converted legacy_material(const Material* material, const FxElemDef*)
				{
					effect_material::converted out{};
					const auto* entry = asset_name(material) ? DB_FindXAssetEntry(ASSET_TYPE_MATERIAL, asset_name(material), false) : nullptr;
					const auto* full = entry && is_readable(entry->asset.header.material, sizeof(Material)) ? entry->asset.header.material : material;
					out.name = material::get_converted_name(const_cast<Material*>(full));
					return out;
				}
			}

			const references& standalone_references()
			{
				static const references refs{ identity, legacy_material };
				return refs;
			}

			namespace
			{
				void finish_system(i7::ParticleSystemDef* out, const unsigned int system_flags)
				{
					out->flags = system_flags | i7::PARTICLE_SYSTEM_DEF_FLAG_KILL_STOPPED_INFINITE_EFFECTS;
					out->version = 15;
					out->occlusionOverrideEmitterIndex = -1;
					out->phaseOptions = i7::PARTICLE_PHASE_OPTION_PHASE_NEVER;
					// stock ambient effects cull neither their drawing nor their update by the frustum (-1, -1)
					out->drawFrustumCullRadius = -1.0f;
					out->updateFrustumCullRadius = -1.0f;
					out->sunDistance = 100000.0f;
					out->editorRotation.v[3] = 1.0f;
				}

				// a spot light BO3 turns away from the effect's axis (its particle axis, 0x140207ED0, from the spawn angles and
				// angular velocity; the light shines along it, 0x140201EA0)
				bool aimed_spot(const FxElemDef* elem, const convert_context& ctx)
				{
					if (convert_elem_type(elem, ctx) != i7::PARTICLE_ELEMENT_TYPE_LIGHT_SPOT)
					{
						return false;
					}
					for (auto i = 0; i < 3; i++)
					{
						if (elem->spawnAngles[i].base != 0.0f || elem->spawnAngles[i].amplitude != 0.0f ||
							elem->angularVelocity[i].base != 0.0f || elem->angularVelocity[i].amplitude != 0.0f)
						{
							return true;
						}
					}
					return false;
				}
			}

			zonetool::iw7::ParticleSystemDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator, const references& refs,
				std::vector<zonetool::iw7::ParticleSystemDef*>* children)
			{
				auto* out = allocator.allocate<i7::ParticleSystemDef>();
				out->name = allocator.duplicate_string(refs.effect(asset->name));

				convert_context ctx{};
				ctx.allocator = &allocator;
				ctx.refs = &refs;
				ctx.effect = asset;

				const auto count = asset->elemDefCountLooping + asset->elemDefCountOneShot + asset->elemDefCountEmission;
				out->emitterDefs = allocator.allocate_array<i7::ParticleEmitterDef>(std::max(1, count));
				auto converted = 0;
				for (auto i = 0; i < count && is_readable(asset->elemDefs, sizeof(FxElemDef) * count); i++)
				{
					auto* elem = &asset->elemDefs[i];
					ctx.elem = elem;
					ctx.elem_index = i;
					ctx.element_part = part::whole;
					// ZT_FX_TRACE=<effect name>: logs each element's type, visuals and first visual state
					if (const auto* trace = std::getenv("ZT_FX_TRACE"); trace && std::strstr(asset->name, trace))
					{
						const auto vis = get_vis_samples(elem);
						const auto* c = vis.samples ? vis.samples[0].base.color : nullptr;
						ZONETOOL_INFO("trace \"%s\" elem %d: type %u, visuals %u, flags 0x%X, colour %u %u %u %u, intensity %g, radius %g",
							asset->name, i, elem->elemType, elem->visualCount, elem->flags, c ? c[0] : 0, c ? c[1] : 0, c ? c[2] : 0,
							c ? c[3] : 0, vis.samples ? vis.samples[0].base.intensity : 0.0f, vis.samples ? vis.samples[0].base.radius : 0.0f);
					}
					// sounds, lens flares and the types without a visual have no conversion yet
					if (elem->elemType > T7_ELEM_TYPE_RUNNER || elem->elemType == T7_ELEM_TYPE_SOUND ||
						elem->elemType == T7_ELEM_TYPE_LENS_FLARE || elem->elemType == T7_ELEM_TYPE_UNKNOWN_9)
					{
						ZONETOOL_WARNING("effect \"%s\": element %d of type %u has no conversion; left out", asset->name, i, elem->elemType);
						continue;
					}
					if (children && aimed_spot(elem, ctx))
					{
						// the light alone in an effect of its own, which the element's runner spawns, turned
						auto* child = allocator.allocate<i7::ParticleSystemDef>();
						child->name = allocator.duplicate_string(std::string(out->name) + "_l" + std::to_string(i));
						child->emitterDefs = allocator.allocate_array<i7::ParticleEmitterDef>(1);
						auto light = ctx;
						light.system_flags = 0;
						light.element_part = part::spot_light;
						if (!convert_elem(light, &child->emitterDefs[0], false))
						{
							continue;
						}
						child->numEmitters = 1;
						finish_system(child, light.system_flags);
						children->push_back(child);
						ctx.element_part = part::spot_runner;
						ctx.light_child = child->name;
					}
					if (convert_elem(ctx, &out->emitterDefs[converted], i < asset->elemDefCountLooping))
					{
						converted++;
					}
				}
				out->numEmitters = converted;
				finish_system(out, ctx.system_flags);
				return out;
			}

			zonetool::iw7::ParticleSystemDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator, const references& refs)
			{
				return convert(asset, allocator, refs, nullptr);
			}

			zonetool::iw7::ParticleSystemDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator)
			{
				return convert(asset, allocator, standalone_references());
			}

			void dump(FxEffectDef* asset, const references& refs)
			{
				if (!is_readable(asset, sizeof(FxEffectDef)) || !safe_name(asset->name))
				{
					return;
				}
				utils::memory::allocator allocator;
				std::vector<zonetool::iw7::ParticleSystemDef*> children;
				zonetool::iw7::particle_system::dump(convert(asset, allocator, refs, &children));
				for (auto* child : children)
				{
					zonetool::iw7::particle_system::dump(child);
				}
			}

			void dump(FxEffectDef* asset)
			{
				dump(asset, standalone_references());
			}
		}
	}
}
