#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "particle_system.hpp"
#include "material.hpp"

#include "zonetool/t7/functions.hpp"
#include "zonetool/iw7/assets/particle_system.hpp"

// t7 fx are element based, iw7 vfx are a module graph.  iw7 weapons only consume
// vfx - every shipped vfx package slot uses FX_COMBINED_VFX and none use the legacy
// FxEffectDef path - so a converted .fxe cannot be referenced by a weapon and has to
// become a ParticleSystemDef instead.
//
// the element -> module mapping below is ported from the proven IW5 -> IW7 converter
// in Joelrau/zonetool (src/IW5/Converter/IW7/Assets/ParticleSystem.cpp, based on
// xoxor4d's curve research), with the source side changed from IW5's FxElemDef to
// t7's.  each t7 FxElemDef becomes one ParticleEmitterDef carrying a single state,
// and the element's fields are spread across that state's INIT/UPDATE/TEST module
// groups.  the built struct is handed to iw7's own writer rather than emitting
// .iw7VFX here.

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace particlesystem
		{
			namespace
			{
				namespace i7 = zonetool::iw7;

				// ---- t7 sample layouts -------------------------------------------------
				// t7 grew the shared cod fx sample structs, so iw7's cannot be reused:
				// the vis sample stride is 0x50, not iw7's 0x48, and the color is packed
				// bytes rather than floats.  measured from dumped core_mod element memory
				// (ring element size ramps 0->50 in both size floats, rotationTotal
				// accumulates rotationDelta, colors ramp 255,132,45 style) - field names
				// past size[] follow the t6 lineage and are pending confirmation against
				// BlackOps3.exe.

				struct t7_vis_state
				{
					unsigned char color[4];
					float rotationDelta;
					float rotationTotal;
					float size[2];
					float scale;
					// Measured from BO3's HDR-era sample layout. IW7 represents this
					// channel with PARTICLE_MODULE_INTENSITY_GRAPH; dropping it made
					// emissive T7 effects valid but effectively invisible.
					float emission;
					float unk[3];
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

				// t7 trail extended def (elemType 5), 0x30 bytes, measured from
				// BlackOps3.exe's trail loader/renderer.  only scrollTimeMsec and
				// repeatDist line up with iw7's FxTrailDef; the two floats iw7 calls
				// invSplit* are actually alpha fade lengths in t7, and 0x08 is unknown -
				// the converter only reads the first two fields so the rest stay labelled
				// as measured rather than force-fit to iw7's names
				struct t7_trail_def
				{
					int scrollTimeMsec;   // 0x00 time modulus
					int repeatDist;       // 0x04 distance divisor
					int unk_08;           // 0x08 open
					float fadeInDist;     // 0x0C alpha fade-in length
					float fadeOutDist;    // 0x10 alpha fade-out length
					int vertCount;        // 0x14
					void* verts;          // 0x18
					int indCount;         // 0x20
					int pad_24;           // 0x24
					void* inds;           // 0x28
				};

				// ---- t7 element flags --------------------------------------------------
				// low bits shared with the whole cod lineage; verified against the
				// thundergun set: sphere offset rides the spark cluster elements
				// (spawnOffsetRadius 1), cylinder rides the dust swirls, the velocity
				// graph bit rides every element carrying non zero velocity samples and
				// the gravity bit rides the one element with a non zero gravity range.
				enum t7_elem_flags : unsigned int
				{
					T7_FX_ELEM_SPAWN_RELATIVE_TO_EFFECT = 0x2,
					T7_FX_ELEM_SPAWN_FRUSTUM_CULL = 0x4,
					T7_FX_ELEM_RUNNER_USES_RAND_ROT = 0x8,
					T7_FX_ELEM_SPAWN_OFFSET_NONE = 0x0,
					T7_FX_ELEM_SPAWN_OFFSET_SPHERE = 0x10,
					T7_FX_ELEM_SPAWN_OFFSET_CYLINDER = 0x20,
					T7_FX_ELEM_SPAWN_OFFSET_MASK = 0x30,
					T7_FX_ELEM_RUN_RELATIVE_TO_WORLD = 0x0,
					T7_FX_ELEM_RUN_RELATIVE_TO_SPAWN = 0x40,
					T7_FX_ELEM_RUN_RELATIVE_TO_EFFECT = 0x80,
					T7_FX_ELEM_RUN_RELATIVE_TO_OFFSET = 0xC0,
					T7_FX_ELEM_RUN_RELATIVE_TO_CAMERA = 0x100,
					T7_FX_ELEM_RUN_MASK = 0x1C0,
					T7_FX_ELEM_DIE_ON_TOUCH = 0x200,
					T7_FX_ELEM_DRAW_PAST_FOG = 0x400,
					T7_FX_ELEM_DRAW_WITH_VIEWMODEL = 0x800,
					T7_FX_ELEM_BLOCK_SIGHT = 0x1000,
					T7_FX_ELEM_USE_COLLISION = 0x200000,
					T7_FX_ELEM_HAS_VELOCITY_GRAPH_LOCAL = 0x1000000,
					T7_FX_ELEM_HAS_VELOCITY_GRAPH_WORLD = 0x2000000,
					T7_FX_ELEM_HAS_GRAVITY = 0x4000000,
					T7_FX_ELEM_USE_MODEL_PHYSICS = 0x8000000,
					T7_FX_ELEM_NONUNIFORM_SCALE = 0x10000000,
				};

				// t7 uses the treyarch FxElemType enum, not the iw lineage one iw7
				// inherited.  0..6 are the t5/t6 material-sprite block (it inserts
				// SPRITE_ROTATED and LINE that iw removed, so trail is 5 not 3), but from
				// 9 up t7 diverges from t6 too: it inserts one type at 9 and a lens flare
				// at 12, shifting decal to 13 and runner to 14.  measured from
				// BlackOps3.exe's single-visual zone loader (sub_1413FBDF0) which
				// identifies each type by the asset kind it loads, cross-checked against
				// the extended-def switch {5,8,10,12} mapping 1:1 onto boiii-free's
				// FxElemExtendedDefPtr {trailDef, dynamicLightDef, spotLightDef,
				// lensFlareDef}.  byte 8 = omni light matches the thundergun's
				// gfx_dlight_gen_omni element.
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
					T7_ELEM_TYPE_UNKNOWN_9 = 9,   // t7 insertion, no visual/extended, identity unknown
					T7_ELEM_TYPE_SPOT_LIGHT = 10,
					T7_ELEM_TYPE_SOUND = 11,
					T7_ELEM_TYPE_LENS_FLARE = 12, // t7 insertion (FxLensFlareVisualDef asset)
					T7_ELEM_TYPE_DECAL = 13,
					T7_ELEM_TYPE_RUNNER = 14,
					T7_ELEM_TYPE_UNKNOWN_15 = 15, // no visual, routed from FX_SpawnElem
					T7_ELEM_TYPE_UNKNOWN_16 = 16, // no visual
				};

				bool is_sprite_type(unsigned char type)
				{
					switch (type)
					{
					case T7_ELEM_TYPE_SPRITE_BILLBOARD:
					case T7_ELEM_TYPE_SPRITE_ORIENTED:
					case T7_ELEM_TYPE_SPRITE_ROTATED:
					case T7_ELEM_TYPE_TAIL:
					case T7_ELEM_TYPE_LINE:
					case T7_ELEM_TYPE_TRAIL:
					case T7_ELEM_TYPE_CLOUD:
						return true;
					default:
						return false;
					}
				}

				// the element types this converter can build a real iw7 element for:
				// the sprite block, plus model/decal/runner which carry their own visuals.
				// Omni light is converted through IW7's stock light_fx_default profile;
				// the T7 element curves retain its color/intensity/size. Spot light and
				// sound still need asset-specific translation, lens flare needs an
				// FxLensFlareVisualDef converter, and 9/15/16 remain unidentified. Those
				// unsupported forms are dropped rather than mapped to dangling assets.
				bool is_convertible_type(unsigned char type)
				{
					if (is_sprite_type(type))
					{
						return true;
					}

					switch (type)
					{
					case T7_ELEM_TYPE_MODEL:
					case T7_ELEM_TYPE_OMNI_LIGHT:
					case T7_ELEM_TYPE_DECAL:
					case T7_ELEM_TYPE_RUNNER:
						return true;
					default:
						return false;
					}
				}

				// ---- pointer safety ----------------------------------------------------
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
					if (!(mbi.Protect & readable) || (mbi.Protect & PAGE_GUARD))
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
						if (!*it)
						{
							return it == name ? nullptr : name;
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

				template <typename T>
				T* alias_asset(const char* name, utils::memory::allocator& allocator)
				{
					if (!name)
					{
						return nullptr;
					}

					const auto stub = allocator.allocate<T>();
					stub->name = name;
					return stub;
				}

				// a converted material must be referenced by the exact name the material
				// converter writes it under (mo/ vs el/ depends on the material's techset,
				// not its name) or the reference dangles and the game draws a null material
				// - crashing in the technique lookup.  reuse the material converter's own
				// naming rather than the old name-only heuristic that guessed el/ for
				// everything gfx_*
				i7::Material* material_alias(Material* material, utils::memory::allocator& allocator)
				{
					if (!is_readable(material, sizeof(void*)))
					{
						return nullptr;
					}

					// the material pointer stored on an fx element is a name-only stub: its
					// techniqueSet is null, so the techset-based el/mo naming would fall back
					// to the wrong guess.  resolve the full material from the asset pool by
					// name so get_converted_name sees the real techset
					Material* full = material;
					const auto* stub_name = asset_name(material);
					if (stub_name)
					{
						const auto* entry = zonetool::t7::DB_FindXAssetEntry(
							zonetool::t7::ASSET_TYPE_MATERIAL, stub_name, false);
						if (entry && is_readable(entry->asset.header.material, sizeof(void*)))
						{
							full = entry->asset.header.material;
						}
					}

					const auto name = zonetool::t7::converter::iw7::material::get_converted_name(full);
					if (name.empty())
					{
						return nullptr;
					}

					return alias_asset<i7::Material>(allocator.duplicate_string(name), allocator);
				}

				// ---- element type mapping ---------------------------------------------

				unsigned int convert_elem_type(const FxElemDef* elem)
				{
					switch (elem->elemType)
					{
					case T7_ELEM_TYPE_SPRITE_BILLBOARD: return i7::PARTICLE_ELEMENT_TYPE_BILLBOARD_SPRITE;
					case T7_ELEM_TYPE_SPRITE_ORIENTED:  return i7::PARTICLE_ELEMENT_TYPE_ORIENTED_SPRITE;
					case T7_ELEM_TYPE_SPRITE_ROTATED:   return i7::PARTICLE_ELEMENT_TYPE_ORIENTED_SPRITE;
					case T7_ELEM_TYPE_TAIL:             return i7::PARTICLE_ELEMENT_TYPE_TAIL;
					case T7_ELEM_TYPE_LINE:
						// T7 line elements with an authored velocity graph are stretched streaks,
						// not camera-facing quads. The Servant muzzle uses exactly this form.
						// Lines without velocity (the Thundergun impact ring) would collapse as
						// tails, so keep only that form on the billboard fallback.
						return (elem->flags & (T7_FX_ELEM_HAS_VELOCITY_GRAPH_LOCAL |
							T7_FX_ELEM_HAS_VELOCITY_GRAPH_WORLD))
							? i7::PARTICLE_ELEMENT_TYPE_TAIL
							: i7::PARTICLE_ELEMENT_TYPE_BILLBOARD_SPRITE;
					case T7_ELEM_TYPE_TRAIL:            return i7::PARTICLE_ELEMENT_TYPE_GEO_TRAIL;
					case T7_ELEM_TYPE_CLOUD:            return i7::PARTICLE_ELEMENT_TYPE_CLOUD;
					case T7_ELEM_TYPE_MODEL:            return i7::PARTICLE_ELEMENT_TYPE_MODEL;
					case T7_ELEM_TYPE_OMNI_LIGHT:       return i7::PARTICLE_ELEMENT_TYPE_LIGHT_OMNI;
					case T7_ELEM_TYPE_SPOT_LIGHT:       return i7::PARTICLE_ELEMENT_TYPE_LIGHT_SPOT;
					case T7_ELEM_TYPE_DECAL:            return i7::PARTICLE_ELEMENT_TYPE_DECAL;
					case T7_ELEM_TYPE_RUNNER:           return i7::PARTICLE_ELEMENT_TYPE_RUNNER;

					// iw7 has no rotated sprite or line element; both draw a flat sprite of
					// their material.  the thundergun's line elements carry no velocity, so
					// a billboard renders them properly where an iw7 tail would collapse to
					// a point
					default:                            return i7::PARTICLE_ELEMENT_TYPE_BILLBOARD_SPRITE;
					}
				}

				// ---- curve helpers (xoxor4d's research, via the reference converter) ---

				enum class sample_value_type
				{
					base,
					amplitude,
				};

				struct min_max_curve_sample
				{
					sample_value_type min_type{};
					int min_index{};
					float min_comp{ FLT_MAX };

					sample_value_type max_type{};
					int max_index{};
					float max_comp{ -FLT_MAX };

					float get_abs_max() const
					{
						auto max = this->max_comp;
						const auto abs = std::abs(this->min_comp);
						if (abs > max)
						{
							max = abs;
						}
						return max;
					}
				};

				void get_min_max_for_sample(min_max_curve_sample& sample, float comp_base, float comp_ampl, int index)
				{
					if (comp_base < sample.min_comp)
					{
						sample.min_index = index;
						sample.min_type = sample_value_type::base;
						sample.min_comp = comp_base;
					}

					if (comp_base > sample.max_comp)
					{
						sample.max_index = index;
						sample.max_type = sample_value_type::base;
						sample.max_comp = comp_base;
					}

					if (comp_ampl < sample.min_comp)
					{
						sample.min_index = index;
						sample.min_type = sample_value_type::amplitude;
						sample.min_comp = comp_ampl;
					}

					if (comp_ampl > sample.max_comp)
					{
						sample.max_index = index;
						sample.max_type = sample_value_type::amplitude;
						sample.max_comp = comp_ampl;
					}
				}

				float get_velocity_scale(const min_max_curve_sample& mm_sample, float scalar)
				{
					const auto min = mm_sample.min_comp;
					const auto max = mm_sample.max_comp;

					if (min != 0.0f || max != 0.0f)
					{
						auto abs = std::abs(min);
						if (max > abs)
						{
							abs = max;
						}

						return abs / scalar * 2.0f;
					}

					return 0.0f;
				}

				enum class vel_space
				{
					local,
					world,
				};

				float get_velocity_value(vel_space space, int dir, sample_value_type kind,
					const t7_vel_sample* samples, int index)
				{
					const auto& frame = (space == vel_space::local) ? samples[index].local : samples[index].world;
					return (kind == sample_value_type::base) ? frame.velocity_base[dir] : frame.velocity_amplitude[dir];
				}

				void get_largest_velocity_sample_value(vel_space space, int dir,
					const t7_vel_sample* vel_samples, int index, min_max_curve_sample& mm_sample)
				{
					const auto vel_base = get_velocity_value(space, dir, sample_value_type::base, vel_samples, index);
					const auto vel_amp = get_velocity_value(space, dir, sample_value_type::amplitude, vel_samples, index);

					get_min_max_for_sample(mm_sample, vel_base, vel_amp, index);
				}

				void calculate_velocity_scales(float* local_scales, float* world_scales,
					const t7_vel_sample* vel_samples, int vel_samples_count, float sample_scalar)
				{
					min_max_curve_sample local_mm[3]{};
					min_max_curve_sample world_mm[3]{};

					for (auto s = 0; s < vel_samples_count; s++)
					{
						for (auto dir = 0; dir < 3; dir++)
						{
							get_largest_velocity_sample_value(vel_space::local, dir, vel_samples, s, local_mm[dir]);
							get_largest_velocity_sample_value(vel_space::world, dir, vel_samples, s, world_mm[dir]);
						}
					}

					for (auto dir = 0; dir < 3; dir++)
					{
						local_scales[dir] = get_velocity_scale(local_mm[dir], sample_scalar);
						world_scales[dir] = get_velocity_scale(world_mm[dir], sample_scalar);
					}
				}

				void calculate_inv_time_delta(i7::ParticleCurveDef* curves, unsigned int curves_count)
				{
					for (auto i = 0u; i < curves_count; i++)
					{
						float prev_time = 0.0f;

						for (auto j = 1; j < curves[i].numControlPoints; j++)
						{
							curves[i].controlPoints[j].invTimeDelta = 1.0f / (curves[i].controlPoints[j].time - prev_time);
							prev_time = curves[i].controlPoints[j].time;
						}
					}
				}

				void fixup_randomization_flags(const i7::ParticleCurveDef& curve_base,
					const i7::ParticleCurveDef& curve_ampl, unsigned int* flags)
				{
					if (curve_base.numControlPoints != curve_ampl.numControlPoints ||
						curve_base.scale != curve_ampl.scale)
					{
						return;
					}

					for (auto i = 0; i < curve_base.numControlPoints; i++)
					{
						if (curve_base.controlPoints[i].value != curve_ampl.controlPoints[i].value)
						{
							*flags |= i7::PARTICLE_MODULE_FLAG_RANDOMIZE_BETWEEN_CURVES;
						}
					}
				}

				void set_default_size_values(i7::ParticleCurveDef& curve)
				{
					curve.scale = 0.0f;

					curve.controlPoints[0].time = 0.0f;
					curve.controlPoints[0].value = 1.0f;

					curve.controlPoints[1].time = 1.0f;
					curve.controlPoints[1].value = 1.0f;
				}

				void set_default_velocity_values(i7::ParticleCurveDef& curve)
				{
					curve.scale = 1.0f;

					curve.controlPoints[0].time = 0.0f;
					curve.controlPoints[0].value = 0.0f;

					curve.controlPoints[1].time = 1.0f;
					curve.controlPoints[1].value = 0.0f;
				}

				// ---- conversion context ------------------------------------------------
				// the reference keeps these as file scope globals; a context keeps the
				// same accumulate-then-apply behavior without leaking between assets

				struct convert_context
				{
					utils::memory::allocator* allocator;
					unsigned int system_flags;
					unsigned __int64 state_flags;
					unsigned int emitter_flags;
					int test_module_index;
					int elem_index;
					const char* fx_name;
				};

				// BO3's shared particle materials arrive as name-only references, so the
				// material converter cannot recover their image pointers and emits white/
				// black placeholders. The Servant uses the same visual roles as IW7's
				// stock F&F black hole; bind those roles to the stock, always-loaded EQ
				// materials whose classifier, atlas, z-feather and HDR state are complete.
				struct particle_material_remap
				{
					const char* source;
					const char* target;
				};

				constexpr particle_material_remap idgun_material_remaps[] =
				{
					{ "gfx_debris_trash_multiple_em", "eq/vfx_debris_brick_atlas" },
					{ "gfx_distort_ring_hvy", "eq/vfx_dist_blast_wave_loop_sxy_30" },
					{ "gfx_distort_ring_ripple", "eq/vfx_dist_blast_wave_loop_sxy_30" },
					{ "gfx_dust_gen_lit", "eq/vfx_vol_dust_rotorwash_anim_loop" },
					{ "gfx_fog_slow_md_anim_lit", "eq/vfx_vol_smk_wispy_swirl_lg" },
					{ "gfx_fog_slow_sm_anim_lit", "eq/vfx_vol_smk_wispy_swirl" },
					{ "gfx_smk_puff_light_varied", "eq/vfx_vol_smk_wispy_swirl" },
					{ "gfx_shockwave_elec_anim_em_i2048", "eq/vfx_energy_shockwave_04_en" },
					{ "gfx_spark_blink_anim_em", "eq/vfx_pyro_spark_single" },
				};

				const char* idgun_invalid_material_fallback(const convert_context& ctx)
				{
					const std::string effect_name = ctx.fx_name ? ctx.fx_name : "";
					if (effect_name.find("fx_idgun_vortex_explo") != std::string::npos)
					{
						if (ctx.elem_index == 1)
						{
							return "el/gfx_water_splash_em";
						}
						if (ctx.elem_index == 7 || ctx.elem_index == 12)
						{
							return "el/gfx_fire_flame_base_2_anim_em_i1024";
						}
					}
					else if (effect_name.find("fx_idgun_vortex") != std::string::npos &&
						(ctx.elem_index == 3 || ctx.elem_index == 4))
					{
						return "eq/vfx_debris_brick_atlas";
					}
					else if (effect_name.find("fx_idgun_projectile") != std::string::npos &&
						ctx.elem_index == 4)
					{
						return "el/gfx_fire_flame_base_2_anim_em_i1024";
					}
					else if (effect_name.find("fx_idgun_muz_") != std::string::npos)
					{
						// All four Servant muzzle effects use a three-visual gel-splat
						// emitter.  The streamed middle material can collapse to the
						// owning effect name; the preserved working conversion proves
						// that visual is the spread material below.
						return "el/gfx_gel_splat_spread_em";
					}
					else if (effect_name.find("fx_idgun_hole_") != std::string::npos)
					{
						return "el/gfx_debris_clump_em";
					}
					else if (effect_name.find("fx_idgun_ug_hole_") != std::string::npos)
					{
						return "el/gfx_debris_clump_em";
					}

					return nullptr;
				}

				float converted_effect_size_scale(const convert_context& ctx)
				{
					const std::string effect_name = ctx.fx_name ? ctx.fx_name : "";
					const bool idgun_world_effect = effect_name.find("fx_idgun_") != std::string::npos &&
						effect_name.find("fx_idgun_muz_") == std::string::npos &&
						effect_name.find("fx_idgun_projectile") == std::string::npos;

					// The Servant portal graphs are authored much larger than IW7's
					// world-space presentation. Scale the graph itself; init-attribute
					// defaults do not scale graph-driven emitters at runtime.
					return idgun_world_effect ? 0.2675f : 1.0f;
				}

				bool element_uses_material(FxElemDef* elem, const char* expected)
				{
					if (!is_sprite_type(elem->elemType)) return false;
					const auto matches = [expected](Material* material)
					{
						const auto* name = asset_name(material);
						if (!name) return false;
						std::string basename = name;
						const auto separator = basename.find_last_of("/\\");
						if (separator != std::string::npos) basename = basename.substr(separator + 1);
						return basename == expected;
					};

					if (elem->visualCount > 1 && is_readable(elem->visuals.array,
						sizeof(FxElemVisuals) * elem->visualCount))
					{
						for (auto i = 0; i < elem->visualCount; i++)
						{
							if (matches(elem->visuals.array[i].material)) return true;
						}
						return false;
					}

					return elem->visualCount == 1 && matches(elem->visuals.instance.material);
				}

				bool invalid_particle_material_visual(const convert_context& ctx, Material* material)
				{
					const auto* source_name_ptr = asset_name(material);
					if (!source_name_ptr)
					{
						return true;
					}

					std::string source_name = source_name_ptr;
					const auto source_separator = source_name.find_last_of("/\\");
					if (source_separator != std::string::npos)
					{
						source_name = source_name.substr(source_separator + 1);
					}

					std::string effect_name = ctx.fx_name ? ctx.fx_name : "";
					const auto effect_separator = effect_name.find_last_of("/\\");
					if (effect_separator != std::string::npos)
					{
						effect_name = effect_name.substr(effect_separator + 1);
					}

					// Some streamed T7 sprite visual arrays contain a non-material slot
					// pointing back at the owning FxEffectDef (or a generated new########
					// placeholder). Treating that pointer as Material* serializes names such
					// as mo/fx_bow_* and makes IW7 request assets that cannot exist.
					return source_name == effect_name || source_name.starts_with("new");
				}

				bool element_has_resolvable_material_visual(const convert_context& ctx, FxElemDef* elem)
				{
					if (!is_sprite_type(elem->elemType) || !elem->visualCount)
					{
						return false;
					}

					const auto resolvable = [&ctx](Material* material)
					{
						if (!invalid_particle_material_visual(ctx, material))
						{
							return true;
						}

						// The Servant has measured per-element material fallbacks for damaged
						// streamed references. Preserve those instead of dropping the emitter.
						return idgun_invalid_material_fallback(ctx) != nullptr;
					};

					if (elem->visualCount > 1)
					{
						if (!is_readable(elem->visuals.array, sizeof(FxElemVisuals) * elem->visualCount))
						{
							return false;
						}

						for (auto i = 0; i < elem->visualCount; i++)
						{
							if (resolvable(elem->visuals.array[i].material))
							{
								return true;
							}
						}

						return false;
					}

					return resolvable(elem->visuals.instance.material);
				}

				i7::Material* particle_material_alias(const convert_context& ctx, Material* material,
					utils::memory::allocator& allocator)
				{
					const std::string effect_name = ctx.fx_name ? ctx.fx_name : "";
					const auto* source_name_ptr = asset_name(material);
					if (source_name_ptr)
					{
						std::string source_name = source_name_ptr;
						const auto separator = source_name.find_last_of("/\\");
						if (separator != std::string::npos)
						{
							source_name = source_name.substr(separator + 1);
						}

						if (effect_name.find("fx_idgun") != std::string::npos)
						{
							for (const auto& remap : idgun_material_remaps)
							{
								if (source_name == remap.source)
								{
									return alias_asset<i7::Material>(allocator.duplicate_string(remap.target), allocator);
								}
							}
						}

						std::string effect_base = effect_name;
						const auto effect_separator = effect_base.find_last_of("/\\");
						if (effect_separator != std::string::npos)
						{
							effect_base = effect_base.substr(effect_separator + 1);
						}

						// During a streamed T7 zone load, a damaged name-only material
						// reference can resolve to the owning effect name (or a generated
						// "new########" placeholder). Use the already measured per-element
						// role fallback instead of serializing that non-material name.
						if (source_name == effect_base || source_name.starts_with("new"))
						{
							if (const auto* fallback = idgun_invalid_material_fallback(ctx))
							{
								return alias_asset<i7::Material>(allocator.duplicate_string(fallback), allocator);
							}

							ZONETOOL_WARNING("vfx \"%s\" element %i: dropping invalid material visual \"%s\"",
								ctx.fx_name, ctx.elem_index, source_name.c_str());
							return nullptr;
						}
					}

					return material_alias(material, allocator);
				}

				const t7_vis_sample* get_vis_samples(FxElemDef* elem)
				{
					const auto count = elem->visStateIntervalCount + 1;
					const auto* samples = reinterpret_cast<const t7_vis_sample*>(elem->visSamples);
					return is_readable(samples, sizeof(t7_vis_sample) * count) ? samples : nullptr;
				}

				const t7_vel_sample* get_vel_samples(FxElemDef* elem)
				{
					const auto count = elem->velIntervalCount + 1;
					const auto* samples = reinterpret_cast<const t7_vel_sample*>(elem->velSamples);
					return is_readable(samples, sizeof(t7_vel_sample) * count) ? samples : nullptr;
				}

				// ---- UPDATE modules ----------------------------------------------------

				void generate_color_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* vis = get_vis_samples(elem);
					if (!vis)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_COLOR_GRAPH;
					auto& module_data = module.moduleData.colorGraph;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.firstCurve = false;
					module_data.m_modulateColorByAlpha = false;

					const auto sample_count = elem->visStateIntervalCount + 1;
					const auto sample_size = 1.0f / std::max(1, sample_count - 1);

					for (auto i = 0; i < 8; i++)
					{
						module_data.m_curves[i].numControlPoints = sample_count;
						module_data.m_curves[i].controlPoints =
							ctx.allocator->allocate_array<i7::ParticleCurveControlPointDef>(sample_count);
						module_data.m_curves[i].scale = 1.0f;
					}

					// T7's source sample bytes are RGBA, but IW7's particle color graph
					// consumes its RGB curves in B,G,R order. Writing source R to curve 0
					// made every blue/cyan bow effect render red in game. Keep alpha in
					// curve 3 and swap the red/blue curve destinations. T7's amplitude
					// half is a delta, while IW7's second curve is the randomized maximum,
					// so add base + amplitude rather than treating it as an absolute.
					for (auto i = 0; i < sample_count; i++)
					{
						const auto& base = vis[i].base.color;
						const auto& ampl = vis[i].amplitude.color;

						module_data.m_curves[0].controlPoints[i].value = base[2] / 255.0f;
						module_data.m_curves[1].controlPoints[i].value = base[1] / 255.0f;
						module_data.m_curves[2].controlPoints[i].value = base[0] / 255.0f;
						module_data.m_curves[3].controlPoints[i].value = base[3] / 255.0f;

						module_data.m_curves[4].controlPoints[i].value =
							std::min(255, static_cast<int>(base[2]) + static_cast<int>(ampl[2])) / 255.0f;
						module_data.m_curves[5].controlPoints[i].value =
							std::min(255, static_cast<int>(base[1]) + static_cast<int>(ampl[1])) / 255.0f;
						module_data.m_curves[6].controlPoints[i].value =
							std::min(255, static_cast<int>(base[0]) + static_cast<int>(ampl[0])) / 255.0f;
						module_data.m_curves[7].controlPoints[i].value =
							std::min(255, static_cast<int>(base[3]) + static_cast<int>(ampl[3])) / 255.0f;

						for (auto j = 0; j < 8; j++)
						{
							module_data.m_curves[j].controlPoints[i].time = sample_size * i;
						}
					}

					calculate_inv_time_delta(module_data.m_curves, 8);

					fixup_randomization_flags(module_data.m_curves[0], module_data.m_curves[4], &module_data.m_flags);
					fixup_randomization_flags(module_data.m_curves[1], module_data.m_curves[5], &module_data.m_flags);
					fixup_randomization_flags(module_data.m_curves[2], module_data.m_curves[6], &module_data.m_flags);
					fixup_randomization_flags(module_data.m_curves[3], module_data.m_curves[7], &module_data.m_flags);

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_COLOR;

					modules.push_back(module);
				}

				void generate_intensity_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* vis = get_vis_samples(elem);
					if (!vis)
					{
						return;
					}

					const auto sample_count = elem->visStateIntervalCount + 1;
					float intensity_scale = 0.0f;
					for (auto i = 0; i < sample_count; i++)
					{
						const auto value_min = std::max(0.0f, vis[i].base.emission);
						const auto value_max = std::max(0.0f, vis[i].base.emission + vis[i].amplitude.emission);
						intensity_scale = std::max(intensity_scale, std::max(value_min, value_max));
					}

					if (intensity_scale <= 0.0f || !std::isfinite(intensity_scale))
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INTENSITY_GRAPH;
					auto& module_data = module.moduleData.intensityGraph;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;
					module_data.firstCurve = false;

					const auto sample_step = 1.0f / std::max(1, sample_count - 1);
					for (auto curve = 0; curve < 2; curve++)
					{
						module_data.m_curves[curve].numControlPoints = sample_count;
						module_data.m_curves[curve].controlPoints =
							ctx.allocator->allocate_array<i7::ParticleCurveControlPointDef>(sample_count);
						module_data.m_curves[curve].scale = intensity_scale;
					}

					for (auto i = 0; i < sample_count; i++)
					{
						const auto value_min = std::max(0.0f, vis[i].base.emission);
						const auto value_max = std::max(0.0f, vis[i].base.emission + vis[i].amplitude.emission);

						module_data.m_curves[0].controlPoints[i].time = sample_step * i;
						module_data.m_curves[0].controlPoints[i].value = value_min / intensity_scale;
						module_data.m_curves[1].controlPoints[i].time = sample_step * i;
						module_data.m_curves[1].controlPoints[i].value = value_max / intensity_scale;
					}

					calculate_inv_time_delta(module_data.m_curves, 2);
					fixup_randomization_flags(module_data.m_curves[0], module_data.m_curves[1],
						&module_data.m_flags);

					const std::string effect_name = ctx.fx_name ? ctx.fx_name : "";
					if (effect_name.find("fx_idgun_") != std::string::npos &&
						element_uses_material(elem, "gfx_shockwave_elec_anim_em_i2048"))
					{
						// The BO3 HDR purple shockwave reads at half intensity after the
						// IW7 curve translation. Restore it to a 1.0 presentation scale.
						for (auto& curve : module_data.m_curves) curve.scale *= 2.0f;
					}

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_INTENSITY_CURVE;
					modules.push_back(module);
				}

				void generate_size_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* vis = get_vis_samples(elem);
					if (!vis)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_SIZE_GRAPH;
					auto& module_data = module.moduleData.sizeGraph;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.firstCurve = false;

					const auto sample_count = elem->visStateIntervalCount + 1;
					const auto sample_size = 1.0f / std::max(1, sample_count - 1);

					// curve scale is the largest key (pos or neg) doubled, key values are
					// key / scale - width and height each pair up as base + amplitude
					float width_scale = 0.0f;
					float height_scale = 0.0f;
					float scale_scale = 0.0f;

					{
						min_max_curve_sample width{};
						min_max_curve_sample height{};
						min_max_curve_sample scale{};

						for (auto s = 0; s < sample_count; s++)
						{
							get_min_max_for_sample(width, vis[s].base.size[0], vis[s].amplitude.size[0], s);
							get_min_max_for_sample(height, vis[s].base.size[1], vis[s].amplitude.size[1], s);
							get_min_max_for_sample(scale, vis[s].base.scale, vis[s].amplitude.scale, s);
						}

						width_scale = width.get_abs_max() * 2.0f;
						height_scale = height.get_abs_max() * 2.0f;
						scale_scale = scale.get_abs_max() * 2.0f;
					}

					if (!width_scale && !height_scale && !scale_scale)
					{
						return;
					}

					// used curves pack to the front, unused ones follow with defaults;
					// the second curve of each pair sits three slots later
					int width_index0 = -1;
					int height_index0 = -1;
					int scale_index0 = -1;

					{
						int index = 0;

						if (width_scale) width_index0 = index++;
						if (height_scale) height_index0 = index++;
						if (scale_scale) scale_index0 = index++;

						if (!width_scale) width_index0 = index++;
						if (!height_scale) height_index0 = index++;
						if (!scale_scale) scale_index0 = index++;
					}

					const auto width_index1 = width_index0 + 3;
					const auto height_index1 = height_index0 + 3;
					const auto scale_index1 = scale_index0 + 3;

					const auto alloc_pair = [&](int index0, int index1, float scale)
					{
						const auto count = scale ? sample_count : 2;

						module_data.m_curves[index0].numControlPoints = count;
						module_data.m_curves[index0].controlPoints =
							ctx.allocator->allocate_array<i7::ParticleCurveControlPointDef>(count);

						module_data.m_curves[index1].numControlPoints = count;
						module_data.m_curves[index1].controlPoints =
							ctx.allocator->allocate_array<i7::ParticleCurveControlPointDef>(count);

						if (scale)
						{
							module_data.m_curves[index0].scale = scale;
							module_data.m_curves[index1].scale = scale;
						}
						else
						{
							set_default_size_values(module_data.m_curves[index0]);
							set_default_size_values(module_data.m_curves[index1]);
						}
					};

					alloc_pair(width_index0, width_index1, width_scale);
					alloc_pair(height_index0, height_index1, height_scale);
					alloc_pair(scale_index0, scale_index1, scale_scale);

					for (auto i = 0; i < sample_count; i++)
					{
						if (width_scale)
						{
							module_data.m_curves[width_index0].controlPoints[i].time = sample_size * i;
							module_data.m_curves[width_index0].controlPoints[i].value = vis[i].base.size[0] / width_scale;

							module_data.m_curves[width_index1].controlPoints[i].time = sample_size * i;
							module_data.m_curves[width_index1].controlPoints[i].value = vis[i].amplitude.size[0] / width_scale +
								module_data.m_curves[width_index0].controlPoints[i].value;
						}

						if (height_scale)
						{
							module_data.m_curves[height_index0].controlPoints[i].time = sample_size * i;
							module_data.m_curves[height_index0].controlPoints[i].value = vis[i].base.size[1] / height_scale;

							module_data.m_curves[height_index1].controlPoints[i].time = sample_size * i;
							module_data.m_curves[height_index1].controlPoints[i].value = vis[i].amplitude.size[1] / height_scale +
								module_data.m_curves[height_index0].controlPoints[i].value;
						}

						if (scale_scale)
						{
							module_data.m_curves[scale_index0].controlPoints[i].time = sample_size * i;
							module_data.m_curves[scale_index0].controlPoints[i].value = vis[i].base.scale / scale_scale;

							module_data.m_curves[scale_index1].controlPoints[i].time = sample_size * i;
							module_data.m_curves[scale_index1].controlPoints[i].value = vis[i].amplitude.scale / scale_scale +
								module_data.m_curves[scale_index0].controlPoints[i].value;
						}
					}

					calculate_inv_time_delta(module_data.m_curves, 6);

					const auto effect_size_scale = converted_effect_size_scale(ctx);
					if (effect_size_scale != 1.0f)
					{
						for (auto& curve : module_data.m_curves)
						{
							curve.scale *= effect_size_scale;
						}
					}

					if (width_scale)
					{
						fixup_randomization_flags(module_data.m_curves[width_index0], module_data.m_curves[width_index1], &module_data.m_flags);
					}
					if (height_scale)
					{
						fixup_randomization_flags(module_data.m_curves[height_index0], module_data.m_curves[height_index1], &module_data.m_flags);
					}
					if (scale_scale)
					{
						fixup_randomization_flags(module_data.m_curves[scale_index0], module_data.m_curves[scale_index1], &module_data.m_flags);
					}

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_SIZE_CURVE;

					modules.push_back(module);
				}

				void generate_rotation_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* vis = get_vis_samples(elem);
					if (!vis)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_ROTATION_GRAPH;
					auto& module_data = module.moduleData.rotationGraph;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_useRotationRate = true;

					const auto sample_count = elem->visStateIntervalCount + 1;

					// rotation deltas are per sample interval; the curve carries them as a
					// rate, so the scale folds in the interval count and the msec unit
					min_max_curve_sample rotation{};
					for (auto s = 0; s < sample_count; s++)
					{
						get_min_max_for_sample(rotation, vis[s].base.rotationDelta, vis[s].amplitude.rotationDelta, s);
					}

					const auto rotation_scale = rotation.get_abs_max() * (sample_count - 1) * 1000.0f * 2.0f;
					if (!rotation_scale)
					{
						return;
					}

					const auto sample_size = 1.0f / std::max(1, sample_count - 1);
					const auto sample_scalar = (sample_count - 1) * 1000.0f;

					for (auto i = 0; i < 2; i++)
					{
						module_data.m_curves[i].numControlPoints = sample_count;
						module_data.m_curves[i].controlPoints =
							ctx.allocator->allocate_array<i7::ParticleCurveControlPointDef>(sample_count);
						module_data.m_curves[i].scale = rotation_scale;
					}

					for (auto i = 0; i < sample_count; i++)
					{
						const auto base_vel = vis[i].base.rotationDelta * sample_scalar / rotation_scale;
						const auto ampl_vel = vis[i].amplitude.rotationDelta * sample_scalar / rotation_scale;

						module_data.m_curves[0].controlPoints[i].value = base_vel;
						module_data.m_curves[0].controlPoints[i].time = sample_size * i;

						module_data.m_curves[1].controlPoints[i].value = base_vel + ampl_vel;
						module_data.m_curves[1].controlPoints[i].time = sample_size * i;
					}

					calculate_inv_time_delta(module_data.m_curves, 2);

					fixup_randomization_flags(module_data.m_curves[0], module_data.m_curves[1], &module_data.m_flags);

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_ROTATION_1D_CURVE;

					modules.push_back(module);
				}

				void generate_velocity_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* vel = get_vel_samples(elem);
					if (!vel)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_VELOCITY_GRAPH;
					auto& module_data = module.moduleData.velocityGraph;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					const auto sample_count = elem->velIntervalCount + 1;
					const auto sample_size = 1.0f / std::max(1, sample_count - 1);
					const auto sample_scalar = 1.0f / (std::max(1, sample_count - 1) * 1000.0f);

					float local_scales[3]{};
					float world_scales[3]{};
					calculate_velocity_scales(local_scales, world_scales, vel, sample_count, sample_scalar);

					if (!local_scales[0] && !local_scales[1] && !local_scales[2] &&
						!world_scales[0] && !world_scales[1] && !world_scales[2])
					{
						return;
					}

					const bool local = (elem->flags & T7_FX_ELEM_HAS_VELOCITY_GRAPH_LOCAL) != 0;
					const bool world = (elem->flags & T7_FX_ELEM_HAS_VELOCITY_GRAPH_WORLD) != 0;

					if (local && world)
					{
						ZONETOOL_WARNING("vfx \"%s\": element type %u has simultaneous local/world velocity; leaving its velocity graph out",
							ctx.fx_name, elem->elemType);
						return;
					}

					if (!local && !world)
					{
						// samples exist but neither graph flag is set; nothing would read them
						return;
					}

					module_data.m_flags |= world ? i7::PARTICLE_MODULE_FLAG_USE_WORLD_SPACE : 0;

					const auto space = local ? vel_space::local : vel_space::world;
					const auto* scales = local ? local_scales : world_scales;

					const auto alloc_curve = [&](int index, float scale)
					{
						const auto count = scale ? sample_count : 2;
						module_data.m_curves[index].numControlPoints = count;
						module_data.m_curves[index].controlPoints =
							ctx.allocator->allocate_array<i7::ParticleCurveControlPointDef>(count);

						if (scale)
						{
							module_data.m_curves[index].scale = scale;
						}
						else
						{
							set_default_velocity_values(module_data.m_curves[index]);
						}
					};

					for (auto dir = 0; dir < 3; dir++)
					{
						alloc_curve(dir, scales[dir]);
						alloc_curve(dir + 3, scales[dir]);
					}

					for (auto i = 0; i < sample_count; i++)
					{
						for (auto dir = 0; dir < 3; dir++)
						{
							if (!scales[dir])
							{
								continue;
							}

							const auto base_vel = get_velocity_value(space, dir, sample_value_type::base, vel, i) /
								sample_scalar / scales[dir];
							const auto ampl_vel = get_velocity_value(space, dir, sample_value_type::amplitude, vel, i) /
								sample_scalar / scales[dir];

							module_data.m_curves[dir].controlPoints[i].value = base_vel;
							module_data.m_curves[dir].controlPoints[i].time = sample_size * i;

							module_data.m_curves[dir + 3].controlPoints[i].value = base_vel + ampl_vel;
							module_data.m_curves[dir + 3].controlPoints[i].time = sample_size * i;
						}
					}

					calculate_inv_time_delta(module_data.m_curves, 6);

					fixup_randomization_flags(module_data.m_curves[0], module_data.m_curves[3], &module_data.m_flags);
					fixup_randomization_flags(module_data.m_curves[1], module_data.m_curves[4], &module_data.m_flags);
					fixup_randomization_flags(module_data.m_curves[2], module_data.m_curves[5], &module_data.m_flags);

					ctx.state_flags |= local ? i7::PARTICLE_STATE_DEF_FLAG_HAS_VELOCITY_CURVE_LOCAL1 : 0;
					ctx.state_flags |= world ? i7::PARTICLE_STATE_DEF_FLAG_HAS_VELOCITY_CURVE_WORLD1 : 0;

					modules.push_back(module);
				}

				void generate_gravity_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (elem->gravity.base == 0.0f && elem->gravity.amplitude == 0.0f)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_GRAVITY;
					auto& module_data = module.moduleData.gravity;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_gravityPercentage.min = elem->gravity.base;
					module_data.m_gravityPercentage.max = elem->gravity.base + elem->gravity.amplitude;

					modules.push_back(module);
				}

				// ---- INIT modules ------------------------------------------------------

				void generate_init_spawn_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_SPAWN;
					auto& module_data = module.moduleData.initSpawn;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_curves[0].scale = 1.0f;
					module_data.m_curves[0].numControlPoints = 3;
					module_data.m_curves[0].controlPoints =
						ctx.allocator->allocate_array<i7::ParticleCurveControlPointDef>(3);

					module_data.m_curves[0].controlPoints[0].value = 1.0f;
					module_data.m_curves[0].controlPoints[1].value = 1.0f;
					module_data.m_curves[0].controlPoints[2].value = 0.0f;

					module_data.m_curves[0].controlPoints[0].time = 0.0f;
					module_data.m_curves[0].controlPoints[1].time = 0.75f;
					module_data.m_curves[0].controlPoints[2].time = 1.0f;

					calculate_inv_time_delta(module_data.m_curves, 1);

					modules.push_back(module);
				}

				void generate_init_attributes_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_ATTRIBUTES;
					auto& module_data = module.moduleData.initAttributes;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_useNonUniformInterpolationForColor = false;
					module_data.m_useNonUniformInterpolationForSize = (elem->flags & T7_FX_ELEM_NONUNIFORM_SCALE) != 0;

					// Preserve the upstream/native baseline for other converted effects.
					// Servant assets use a unit base, while their actual world-size
					// correction is applied directly to size-curve scales above.
					const std::string effect_name = ctx.fx_name ? ctx.fx_name : "";
					const bool idgun_world_effect = effect_name.find("fx_idgun_") != std::string::npos &&
						effect_name.find("fx_idgun_muz_") == std::string::npos &&
						effect_name.find("fx_idgun_projectile") == std::string::npos;
					const auto base_size = idgun_world_effect ? 1.0f : 10.0f;
					for (auto i = 0; i < 3; i++)
					{
						module_data.m_sizeMin.v[i] = base_size;
						module_data.m_sizeMax.v[i] = base_size;
					}
					module_data.m_sizeMin.v[3] = 0.0f;
					module_data.m_sizeMax.v[3] = 0.0f;

					for (auto i = 0; i < 4; i++)
					{
						module_data.m_colorMin.v[i] = 1.0f;
						module_data.m_colorMax.v[i] = 1.0f;
						module_data.m_velocityMin.v[i] = 0.0f;
						module_data.m_velocityMax.v[i] = 0.0f;
					}

					modules.push_back(module);
				}

				void generate_init_relative_velocity_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_RELATIVE_VELOCITY;
					auto& module_data = module.moduleData.initRelativeVelocity;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_useBoltInfo = false;

					switch (elem->flags & T7_FX_ELEM_RUN_MASK)
					{
					case T7_FX_ELEM_RUN_RELATIVE_TO_WORLD:
					case T7_FX_ELEM_RUN_RELATIVE_TO_CAMERA:
						module_data.m_velocityType = i7::PARTICLE_RELATIVE_VELOCITY_TYPE_WORLD;
						break;
					case T7_FX_ELEM_RUN_RELATIVE_TO_SPAWN:
					case T7_FX_ELEM_RUN_RELATIVE_TO_EFFECT:
						module_data.m_velocityType = i7::PARTICLE_RELATIVE_VELOCITY_TYPE_LOCAL;
						break;
					case T7_FX_ELEM_RUN_RELATIVE_TO_OFFSET:
						module_data.m_velocityType = i7::PARTICLE_RELATIVE_VELOCITY_TYPE_RELATIVE_TO_EFFECT_ORIGIN;
						break;
					}

					if (elem->elemType == T7_ELEM_TYPE_TRAIL)
					{
						module_data.m_velocityType = i7::PARTICLE_RELATIVE_VELOCITY_TYPE_LOCAL_WITH_BOLT_INFO;
						module_data.m_useBoltInfo = true;
					}

					modules.push_back(module);
				}

				void generate_init_rotation_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (elem->initialRotation.base == 0.0f && elem->initialRotation.amplitude == 0.0f)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_ROTATION;
					auto& module_data = module.moduleData.initRotation;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_rotationAngle.min = elem->initialRotation.base;
					module_data.m_rotationAngle.max = elem->initialRotation.base + elem->initialRotation.amplitude;

					module_data.m_rotationRate.min = 0.0f;
					module_data.m_rotationRate.max = 0.0f;

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_ROTATION_1D_INIT;

					modules.push_back(module);
				}

				void generate_init_rotation3d_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					bool has_rotation = false;
					for (auto i = 0; i < 3; i++)
					{
						has_rotation |= elem->spawnAngles[i].base != 0.0f ||
							elem->spawnAngles[i].amplitude != 0.0f ||
							elem->angularVelocity[i].base != 0.0f ||
							elem->angularVelocity[i].amplitude != 0.0f;
					}

					if (!has_rotation)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_ROTATION_3D;
					auto& module_data = module.moduleData.initRotation3D;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					for (auto i = 0; i < 3; i++)
					{
						module_data.m_rotationRateMin.v[i] = elem->angularVelocity[i].base;
						module_data.m_rotationRateMax.v[i] = elem->angularVelocity[i].base + elem->angularVelocity[i].amplitude;

						module_data.m_rotationAngleMin.v[i] = elem->spawnAngles[i].base;
						module_data.m_rotationAngleMax.v[i] = elem->spawnAngles[i].base + elem->spawnAngles[i].amplitude;
					}
					module_data.m_rotationRateMin.v[3] = 0.0f;
					module_data.m_rotationRateMax.v[3] = 0.0f;
					module_data.m_rotationAngleMin.v[3] = 0.0f;
					module_data.m_rotationAngleMax.v[3] = 0.0f;

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_ROTATION_3D_INIT;

					modules.push_back(module);
				}

				void generate_init_atlas_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (!is_sprite_type(elem->elemType))
					{
						return;
					}

					// a static single-frame sprite needs no atlas module - the shipped iw7
					// vfx only carry one when the effect actually animates (e.g. medusa at
					// 24fps) or draws from a multi-frame grid.  emitting one for a plain
					// sprite would also feed a bogus start frame
					if (elem->atlas.fps == 0 && elem->atlas.indexRange <= 1)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_ATLAS;
					auto& module_data = module.moduleData.initAtlas;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					// iw7's m_startFrame is the first cell to show, not a count - the stock
					// vfx start at 0 and take the grid size from the material.  fps and
					// loopCount carry straight across
					module_data.m_playRate = elem->atlas.fps;
					module_data.m_startFrame = 0;
					module_data.m_loopCount = elem->atlas.loopCount;

					modules.push_back(module);
				}

				void generate_init_material_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (!is_sprite_type(elem->elemType))
					{
						ctx.system_flags |= i7::PARTICLE_SYSTEM_DEF_FLAG_HAS_NON_SPRITES;
						return;
					}

					if (elem->elemType == T7_ELEM_TYPE_TRAIL)
					{
						ZONETOOL_WARNING("vfx \"%s\": geo trail material may need a manual eq->ev fixup", ctx.fx_name);
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_MATERIAL;
					auto& module_data = module.moduleData.initMaterial;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					const auto visual_count = elem->visualCount;
					if (!visual_count)
					{
						return;
					}

					if (visual_count > 1)
					{
						if (!is_readable(elem->visuals.array, sizeof(FxElemVisuals) * visual_count))
						{
							return;
						}

						std::vector<i7::Material*> materials{};
						materials.reserve(visual_count);
						for (auto i = 0; i < visual_count; i++)
						{
							if (auto* material = particle_material_alias(ctx,
								elem->visuals.array[i].material, *ctx.allocator))
							{
								materials.push_back(material);
							}
						}

						if (materials.empty())
						{
							return;
						}

						module_data.m_linkedAssetList.numAssets = static_cast<unsigned int>(materials.size());
						module_data.m_linkedAssetList.assetList =
							ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(materials.size());

						for (auto i = 0u; i < materials.size(); i++)
						{
							module_data.m_linkedAssetList.assetList[i].material = materials[i];
						}
					}
					else
					{
						auto* material = particle_material_alias(ctx,
							elem->visuals.instance.material, *ctx.allocator);
						if (!material)
						{
							return;
						}

						module_data.m_linkedAssetList.numAssets = 1;
						module_data.m_linkedAssetList.assetList =
							ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(1);
						module_data.m_linkedAssetList.assetList[0].material = material;
					}

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_IS_SPRITE;
					ctx.system_flags |= i7::PARTICLE_SYSTEM_DEF_FLAG_HAS_SPRITES;

					modules.push_back(module);
				}

				void generate_init_oriented_sprite_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (elem->elemType != T7_ELEM_TYPE_SPRITE_ORIENTED &&
						elem->elemType != T7_ELEM_TYPE_SPRITE_ROTATED)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_ORIENTED_SPRITE;
					auto& module_data = module.moduleData.initOrientedSprite;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					if (elem->elemType == T7_ELEM_TYPE_SPRITE_ROTATED)
					{
						// Rotated sprites already carry their authored 3D angle in
						// spawnAngles. Native IW7 ground-oriented sprites use identity here
						// and apply orientation through the 3D rotation state.
						module_data.m_orientationQuat.v[3] = 1.0f;
					}
					else
					{
						module_data.m_orientationQuat.v[0] = 0.5f;
						module_data.m_orientationQuat.v[1] = 0.5f;
						module_data.m_orientationQuat.v[2] = 0.5f;
						module_data.m_orientationQuat.v[3] = 0.5f;

						if ((elem->flags & T7_FX_ELEM_RUN_MASK) == T7_FX_ELEM_RUN_RELATIVE_TO_SPAWN)
						{
							module_data.m_orientationQuat.v[1] *= -1.0f;
							module_data.m_orientationQuat.v[2] *= -1.0f;
						}
					}

					modules.push_back(module);
				}

				void generate_init_omni_light_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (elem->elemType != T7_ELEM_TYPE_OMNI_LIGHT)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_LIGHT_OMNI;
					auto& module_data = module.moduleData.initLightOmni;
					module_data.type = module.moduleType;
					module_data.m_flags = i7::PARTICLE_MODULE_FLAG_HAS_LIGHT_DEFS;

					// T7 type 8 points at a 0x240-byte LightDescription asset, not an
					// IW7 GfxLightDef. The element's existing color/intensity/size curves
					// carry its authored appearance; use IW7's all-map FX light profile as
					// the renderable falloff definition. This matches the native IW7 black
					// hole's omni-light module contract exactly.
					module_data.m_linkedAssetList.numAssets = 1;
					module_data.m_linkedAssetList.assetList =
						ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(1);
					module_data.m_linkedAssetList.assetList[0].lightDef =
						alias_asset<i7::GfxLightDef>(ctx.allocator->duplicate_string("light_fx_default"),
							*ctx.allocator);
					module_data.m_tonemappingScaleFactor = 1.0f;
					module_data.m_intensityIR = 0.0f;
					module_data.m_disableVolumetric = false;
					module_data.m_exponent = 0;

					ctx.system_flags |= i7::PARTICLE_SYSTEM_DEF_FLAG_HAS_LIGHTS;
					ctx.emitter_flags |= i7::PARTICLE_EMITTER_DEF_FLAG_HAS_LIGHTS;
					modules.push_back(module);
				}

				void generate_init_tail_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto line_with_velocity = elem->elemType == T7_ELEM_TYPE_LINE &&
						(elem->flags & (T7_FX_ELEM_HAS_VELOCITY_GRAPH_LOCAL |
							T7_FX_ELEM_HAS_VELOCITY_GRAPH_WORLD));
					if (elem->elemType != T7_ELEM_TYPE_TAIL && !line_with_velocity)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_TAIL;
					auto& module_data = module.moduleData.initTail;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_averagePastVelocities = 0;
					module_data.m_maxParentSpeed = 0;
					// BO3's render handlers place a type-3 tail backward along velocity
					// and a type-4 line forward along it. IW7 expresses that distinction
					// with m_tailLeading. Native IW7 trailing tails store false.
					module_data.m_tailLeading = elem->elemType == T7_ELEM_TYPE_LINE;
					module_data.m_scaleWithVelocity = false;
					module_data.m_rotateAroundPivot = false;

					modules.push_back(module);
				}

				void generate_init_geo_trail_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (elem->elemType != T7_ELEM_TYPE_TRAIL)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_GEO_TRAIL;
					auto& module_data = module.moduleData.initGeoTrail;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					// reference defaults; the trail def fills the distances when present
					module_data.m_numPointsMax = 16;
					module_data.m_splitAngle = 0.0f;
					module_data.m_centerOffset = 0.0f;
					module_data.m_numSheets = 2;
					module_data.m_fadeInDistance = 0.0f;
					module_data.m_fadeOutDistance = 0.0f;
					module_data.m_tileOffset.min = 0.0f;
					module_data.m_tileOffset.max = 0.0f;
					module_data.m_useLocalVelocity = false;
					module_data.m_useVerticalTexture = false;
					module_data.m_cameraFacing = false;
					module_data.m_fixLeadingEdge = false;
					module_data.m_clampUVs = false;

					module_data.m_splitDistance = 8.0f;
					module_data.m_tileDistance = 8.0f;
					module_data.m_scrollTime = 0.0f;

					const auto* trail = static_cast<const t7_trail_def*>(
						static_cast<const void*>(elem->extended.trailDef));
					if (is_readable(trail, sizeof(t7_trail_def)))
					{
						module_data.m_splitDistance = static_cast<float>(trail->repeatDist);
						module_data.m_tileDistance = static_cast<float>(trail->repeatDist);
						module_data.m_scrollTime = trail->scrollTimeMsec / 1000.0f;
					}

					modules.push_back(module);
				}

				void generate_init_model_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (elem->elemType != T7_ELEM_TYPE_MODEL)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_MODEL;
					auto& module_data = module.moduleData.initModel;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_usePhysics = (elem->flags & T7_FX_ELEM_USE_MODEL_PHYSICS) != 0;
					module_data.m_motionBlurHQ = false;

					const auto visual_count = elem->visualCount;
					if (!visual_count)
					{
						return;
					}

					module_data.m_linkedAssetList.numAssets = visual_count;
					module_data.m_linkedAssetList.assetList =
						ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(visual_count);

					if (visual_count > 1)
					{
						if (!is_readable(elem->visuals.array, sizeof(FxElemVisuals) * visual_count))
						{
							return;
						}

						for (auto i = 0; i < visual_count; i++)
						{
							module_data.m_linkedAssetList.assetList[i].model =
								alias_asset<i7::XModel>(asset_name(elem->visuals.array[i].model), *ctx.allocator);
						}
					}
					else
					{
						module_data.m_linkedAssetList.assetList[0].model =
							alias_asset<i7::XModel>(asset_name(elem->visuals.instance.model), *ctx.allocator);
					}

					modules.push_back(module);
				}

				void generate_init_runner_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (elem->elemType != T7_ELEM_TYPE_RUNNER)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_RUNNER;
					auto& module_data = module.moduleData.initRunner;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					const auto visual_count = elem->visualCount;
					if (!visual_count)
					{
						return;
					}

					module_data.m_linkedAssetList.numAssets = visual_count;
					module_data.m_linkedAssetList.assetList =
						ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(visual_count);

					if (visual_count > 1)
					{
						if (!is_readable(elem->visuals.array, sizeof(FxElemVisuals) * visual_count))
						{
							return;
						}

						for (auto i = 0; i < visual_count; i++)
						{
							module_data.m_linkedAssetList.assetList[i].particleSystem =
								alias_asset<i7::ParticleSystemDef>(asset_name(elem->visuals.array[i].effectDef.handle),
									*ctx.allocator);
						}
					}
					else
					{
						module_data.m_linkedAssetList.assetList[0].particleSystem =
							alias_asset<i7::ParticleSystemDef>(asset_name(elem->visuals.instance.effectDef.handle),
								*ctx.allocator);
					}

					modules.push_back(module);
				}

				void generate_init_decal_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if (elem->elemType != T7_ELEM_TYPE_DECAL)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_DECAL;
					auto& module_data = module.moduleData.initDecal;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_fadeInTime = static_cast<unsigned short>(elem->fadeInRange.base);
					module_data.m_fadeOutTime = static_cast<unsigned short>(elem->fadeOutRange.base);
					module_data.m_stoppableFadeOutTime = 0;
					module_data.m_lerpWaitTime = 1280;
					module_data.m_lerpColor.v[0] = 1.0f;
					module_data.m_lerpColor.v[1] = 1.0f;
					module_data.m_lerpColor.v[2] = 1.0f;
					module_data.m_lerpColor.v[3] = 1.0f;

					const auto visual_count = elem->visualCount;
					if (!visual_count || !is_readable(elem->visuals.markArray,
						sizeof(FxElemMarkVisuals) * visual_count))
					{
						return;
					}

					module_data.m_linkedAssetList.numAssets = visual_count;
					module_data.m_linkedAssetList.assetList =
						ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(visual_count);

					for (auto i = 0; i < visual_count; i++)
					{
						auto& dest = module_data.m_linkedAssetList.assetList[i].decal;
						dest.materials[0] = material_alias(elem->visuals.markArray[i].materials[0], *ctx.allocator);
						dest.materials[1] = material_alias(elem->visuals.markArray[i].materials[1], *ctx.allocator);
						dest.materials[2] = material_alias(elem->visuals.markArray[i].materials[1], *ctx.allocator);
					}

					modules.push_back(module);
				}

				void generate_init_spawn_shape_box_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					bool has_origin = false;
					for (auto i = 0; i < 3; i++)
					{
						has_origin |= elem->spawnOrigin[i].base != 0.0f ||
							elem->spawnOrigin[i].amplitude != 0.0f;
					}

					if (!has_origin)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_SPAWN_SHAPE_BOX;
					auto& module_data = module.moduleData.initSpawnShapeBox;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					if ((elem->flags & T7_FX_ELEM_RUN_MASK) == T7_FX_ELEM_RUN_RELATIVE_TO_WORLD &&
						elem->elemType != T7_ELEM_TYPE_TRAIL)
					{
						module_data.m_flags |= i7::PARTICLE_MODULE_FLAG_USE_WORLD_SPACE;
					}

					module_data.m_axisFlags = i7::PARTICLE_MODULE_AXES_FLAG_ALL;
					module_data.m_spawnFlags = 0;
					module_data.m_normalAxis = 0;
					module_data.m_spawnType = 0;
					module_data.m_volumeCubeRoot = 0.0f;

					for (auto i = 0; i < 3; i++)
					{
						module_data.m_dimensionsMin.v[i] = elem->spawnOrigin[i].base;
						module_data.m_dimensionsMax.v[i] = elem->spawnOrigin[i].base + elem->spawnOrigin[i].amplitude;
					}
					module_data.m_dimensionsMin.v[3] = 0.0f;
					module_data.m_dimensionsMax.v[3] = 0.0f;

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_SPAWN_SHAPE;

					modules.push_back(module);
				}

				void generate_init_spawn_shape_sphere_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if ((elem->flags & T7_FX_ELEM_SPAWN_OFFSET_MASK) != T7_FX_ELEM_SPAWN_OFFSET_SPHERE)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_SPAWN_SHAPE_SPHERE;
					auto& module_data = module.moduleData.initSpawnShapeSphere;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_axisFlags = i7::PARTICLE_MODULE_AXES_FLAG_ALL;
					module_data.m_spawnFlags = 0;
					module_data.m_normalAxis = 0;
					module_data.m_spawnType = 0;
					module_data.m_volumeCubeRoot = 0.0f;

					module_data.m_radius.min = elem->spawnOffsetRadius.base;
					module_data.m_radius.max = elem->spawnOffsetRadius.base + elem->spawnOffsetRadius.amplitude;

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_SPAWN_SHAPE;

					modules.push_back(module);
				}

				void generate_init_spawn_shape_cylinder_module(convert_context& ctx, FxElemDef* elem,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					if ((elem->flags & T7_FX_ELEM_SPAWN_OFFSET_MASK) != T7_FX_ELEM_SPAWN_OFFSET_CYLINDER)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = i7::PARTICLE_MODULE_INIT_SPAWN_SHAPE_CYLINDER;
					auto& module_data = module.moduleData.initSpawnShapeCylinder;
					module_data.type = module.moduleType;
					module_data.m_flags = 0;

					module_data.m_axisFlags = i7::PARTICLE_MODULE_AXES_FLAG_ALL;
					module_data.m_spawnFlags = 0;
					module_data.m_normalAxis = 0;
					module_data.m_spawnType = 0;
					module_data.m_volumeCubeRoot = 0.0f;

					module_data.m_hasRotation = true;
					module_data.m_rotateCalculatedOffset = false;

					module_data.m_directionQuat.v[0] = 0.0f;
					module_data.m_directionQuat.v[1] = 0.7071067690849304f;
					module_data.m_directionQuat.v[2] = 0.0f;
					module_data.m_directionQuat.v[3] = 0.7071067690849304f;

					module_data.m_halfHeight = (elem->spawnOffsetHeight.base +
						elem->spawnOffsetHeight.amplitude * 0.5f) * 0.5f;

					module_data.m_radius.min = elem->spawnOffsetRadius.base;
					module_data.m_radius.max = elem->spawnOffsetRadius.base + elem->spawnOffsetRadius.amplitude;

					ctx.state_flags |= i7::PARTICLE_STATE_DEF_FLAG_HAS_SPAWN_SHAPE;

					modules.push_back(module);
				}

				// ---- TEST modules ------------------------------------------------------

				void generate_test_module(convert_context& ctx, FxEffectDefRef ref,
					i7::ParticleModuleType type, bool kill, unsigned __int64 state_flag,
					std::vector<i7::ParticleModuleDef>& modules)
				{
					const auto* name = asset_name(ref.handle);
					if (!name)
					{
						return;
					}

					// BO3 owns runner/death children through the parent FX lifecycle. In
					// IW7, the converted Servant vortex death event detaches its looping
					// hole_xl child as the parent is deleted, leaving an immortal black-
					// smoke system. The GSC already plays the authored collapse effect,
					// so omit only this orphaning child event.
					const std::string effect_name = ctx.fx_name ? ctx.fx_name : "";
					const std::string child_name = name;
					if (type == i7::PARTICLE_MODULE_TEST_DEATH &&
						effect_name.find("fx_idgun_vortex") != std::string::npos &&
						effect_name.find("fx_idgun_vortex_explo") == std::string::npos &&
						child_name.find("fx_idgun_") != std::string::npos &&
						child_name.find("hole_xl") != std::string::npos)
					{
						return;
					}

					i7::ParticleModuleDef module{};
					module.moduleType = type;
					auto& module_data = module.moduleData.testDeath;
					module_data.type = type;
					module_data.m_flags = 0;

					module_data.m_moduleIndex = static_cast<unsigned short>(ctx.test_module_index++);

					module_data.m_eventHandlerData.m_kill = kill;

					module_data.m_eventHandlerData.m_linkedAssetList.numAssets = 1;
					module_data.m_eventHandlerData.m_linkedAssetList.assetList =
						ctx.allocator->allocate_array<i7::ParticleLinkedAssetDef>(1);
					module_data.m_eventHandlerData.m_linkedAssetList.assetList[0].particleSystem =
						alias_asset<i7::ParticleSystemDef>(ctx.allocator->duplicate_string(name), *ctx.allocator);

					ctx.state_flags |= state_flag;

					modules.push_back(module);
				}

				// ---- emitter -----------------------------------------------------------

				void store_group(convert_context& ctx, i7::ParticleModuleGroupDef* group,
					const std::vector<i7::ParticleModuleDef>& modules)
				{
					group->numModules = static_cast<int>(modules.size());
					group->disabled = false;
					group->moduleDefs = nullptr;

					if (!modules.empty())
					{
						group->moduleDefs = ctx.allocator->allocate_array<i7::ParticleModuleDef>(modules.size());
						std::memcpy(group->moduleDefs, modules.data(),
							modules.size() * sizeof(i7::ParticleModuleDef));
					}
				}

				void convert_elem(convert_context& ctx, i7::ParticleEmitterDef* emitter,
					FxElemDef* elem, bool looping)
				{
					ctx.emitter_flags = 0;
					ctx.state_flags = 0;

					emitter->flags = 0;

					emitter->particleSpawnRate.min = 5.0f;
					emitter->particleSpawnRate.max = 5.0f;

					emitter->particleBurstCount.min = 1;
					emitter->particleBurstCount.max = 1;

					emitter->emitterLife.min = 0.0f;
					emitter->emitterLife.max = 0.0f;

					emitter->emitterDelay.min = 0.0f;
					emitter->emitterDelay.max = 0.0f;

					if (looping)
					{
						// BO3's loop scheduler reads FxSpawnDefLooping::spawnCount at
						// element +0x10/+0x14 and spawns base..base+amplitude particles on
						// every interval. The old IW5-derived path ignored this T7 extension
						// and therefore emitted only one particle per interval.
						const auto spawn_count_min = std::max(0, elem->spawn.looping.spawnCount.base);
						const auto spawn_count_max = std::max(spawn_count_min,
							elem->spawn.looping.spawnCount.base + elem->spawn.looping.spawnCount.amplitude);

						if (elem->spawn.looping.count == 0x7FFFFFFF)
						{
							// T7 stores milliseconds BETWEEN spawns; IW7 stores spawns per
							// second. The old code copied 100 ms as 100/sec instead of 10/sec.
							const auto interval = std::max(1, elem->spawn.looping.intervalMsec);
							const auto intervals_per_second = 1000.0f / static_cast<float>(interval);
							emitter->particleSpawnRate.min = spawn_count_min * intervals_per_second;
							emitter->particleSpawnRate.max = spawn_count_max * intervals_per_second;

							// A zero emitter lifetime already means it loops until stopped.
							// INFINITE_PARTICLE_LIFE incorrectly made every emitted sprite
							// immortal, accumulating the vortex into an opaque cloud.
							emitter->particleCountMax = 1; // finalized after particleLife
						}
						else
						{
							const auto interval = elem->spawn.looping.intervalMsec;
							const auto interval_count = elem->spawn.looping.count;

							const auto emitter_life = (interval_count * interval) / 1000.0f;
							const auto intervals_per_second = interval > 0
								? 1000.0f / static_cast<float>(interval) : 0.0f;

							emitter->particleSpawnRate.min = spawn_count_min * intervals_per_second;
							emitter->particleSpawnRate.max = spawn_count_max * intervals_per_second;

							emitter->emitterLife.min = emitter_life;
							emitter->emitterLife.max = emitter_life;

							emitter->particleCountMax = std::max(1,
								interval_count * spawn_count_max);
						}
					}
					else
					{
						emitter->particleBurstCount.min = elem->spawn.oneShot.count.base;
						emitter->particleBurstCount.max = elem->spawn.oneShot.count.base +
							elem->spawn.oneShot.count.amplitude;
						emitter->particleCountMax = std::max(1, emitter->particleBurstCount.max);

						ctx.emitter_flags |= i7::PARTICLE_EMITTER_DEF_FLAG_USE_BURST_MODE;
					}

					emitter->particleLife.min = elem->lifeSpanMsec.base / 1000.0f;
					emitter->particleLife.max = elem->lifeSpanMsec.base / 1000.0f +
						elem->lifeSpanMsec.amplitude / 1000.0f;

					if (looping && elem->spawn.looping.count == 0x7FFFFFFF)
					{
						// IW7 caps concurrently alive particles per emitter. One was too
						// small for any continuous ring/smoke trail; derive the exact
						// capacity required by source rate * maximum particle lifetime.
						const auto concurrent = static_cast<unsigned int>(std::ceil(
							emitter->particleSpawnRate.max * std::max(0.0f, emitter->particleLife.max)));
						emitter->particleCountMax = std::max(1u, concurrent);
					}

					emitter->particleDelay.min = elem->spawnDelayMsec.base / 1000.0f;
					emitter->particleDelay.max = elem->spawnDelayMsec.base / 1000.0f +
						elem->spawnDelayMsec.amplitude / 1000.0f;

					emitter->spawnRangeSq.min = elem->spawnRange.base;
					emitter->spawnRangeSq.max = elem->spawnRange.base + elem->spawnRange.amplitude;
					emitter->spawnRangeSq.min *= emitter->spawnRangeSq.min;
					emitter->spawnRangeSq.max *= emitter->spawnRangeSq.max;

					emitter->spawnFrustumCullRadius = elem->spawnFrustumCullRadius;

					// t7 dropped the per element random seed the iw lineage kept
					emitter->randomSeed = 0;

					emitter->particleSpawnShapeRange.min = 0.0f;
					emitter->particleSpawnShapeRange.max = 0.0f;

					emitter->groupIDs[0] = 0;
					emitter->groupIDs[1] = 0;
					emitter->groupIDs[2] = 0;
					emitter->groupIDs[3] = 0;

					emitter->unk1 = 0.0f;
					emitter->unk2 = 100.0f;

					ctx.emitter_flags |= (elem->flags & T7_FX_ELEM_DRAW_PAST_FOG) != 0
						? i7::PARTICLE_EMITTER_DEF_FLAG_DRAW_PAST_FOG : 0;

					emitter->numStates = 1;
					emitter->stateDefs = ctx.allocator->allocate<i7::ParticleStateDef>();

					auto* state = emitter->stateDefs;

					state->elementType = convert_elem_type(elem);
					state->flags = 0;

					ctx.state_flags |= (elem->flags & T7_FX_ELEM_USE_MODEL_PHYSICS) != 0
						? i7::PARTICLE_STATE_DEF_FLAG_USE_PHYSICS : 0;
					ctx.state_flags |= (elem->flags & T7_FX_ELEM_USE_COLLISION) != 0
						? i7::PARTICLE_STATE_DEF_FLAG_REQUIRES_WORLD_COLLISION : 0;
					ctx.state_flags |= (elem->flags & T7_FX_ELEM_DRAW_WITH_VIEWMODEL) != 0
						? i7::PARTICLE_STATE_DEF_FLAG_DRAW_WITH_VIEW_MODEL : 0;
					ctx.state_flags |= (elem->flags & T7_FX_ELEM_BLOCK_SIGHT) != 0
						? i7::PARTICLE_STATE_DEF_FLAG_BLOCKS_SIGHT : 0;

					state->moduleGroupDefs =
						ctx.allocator->allocate_array<i7::ParticleModuleGroupDef>(i7::PARTICLE_MODULE_GROUP_COUNT);

					{
						std::vector<i7::ParticleModuleDef> init_modules{};
						generate_init_spawn_module(ctx, elem, init_modules);
						generate_init_attributes_module(ctx, elem, init_modules);
						generate_init_omni_light_module(ctx, elem, init_modules);
						generate_init_tail_module(ctx, elem, init_modules);
						generate_init_geo_trail_module(ctx, elem, init_modules);
						generate_init_model_module(ctx, elem, init_modules);
						generate_init_runner_module(ctx, elem, init_modules);
						generate_init_decal_module(ctx, elem, init_modules);
						generate_init_oriented_sprite_module(ctx, elem, init_modules);
						generate_init_material_module(ctx, elem, init_modules);
						generate_init_atlas_module(ctx, elem, init_modules);
						generate_init_relative_velocity_module(ctx, elem, init_modules);
						generate_init_rotation_module(ctx, elem, init_modules);
						generate_init_rotation3d_module(ctx, elem, init_modules);
						generate_init_spawn_shape_box_module(ctx, elem, init_modules);
						generate_init_spawn_shape_sphere_module(ctx, elem, init_modules);
						generate_init_spawn_shape_cylinder_module(ctx, elem, init_modules);

						store_group(ctx, &state->moduleGroupDefs[i7::PARTICLE_MODULE_GROUP_INIT], init_modules);
					}

					{
						std::vector<i7::ParticleModuleDef> update_modules{};
						generate_color_module(ctx, elem, update_modules);
						generate_intensity_module(ctx, elem, update_modules);
						generate_size_module(ctx, elem, update_modules);
						generate_rotation_module(ctx, elem, update_modules);
						generate_velocity_module(ctx, elem, update_modules);
						generate_gravity_module(ctx, elem, update_modules);

						store_group(ctx, &state->moduleGroupDefs[i7::PARTICLE_MODULE_GROUP_UPDATE], update_modules);
					}

					{
						ctx.test_module_index = 0;

						std::vector<i7::ParticleModuleDef> test_modules{};
						generate_test_module(ctx, elem->effectOnDeath, i7::PARTICLE_MODULE_TEST_DEATH,
							false, 0, test_modules);
						generate_test_module(ctx, elem->effectOnImpact, i7::PARTICLE_MODULE_TEST_IMPACT,
							true, i7::PARTICLE_STATE_DEF_FLAG_HANDLE_ON_IMPACT, test_modules);
						generate_test_module(ctx, elem->effectEmitted, i7::PARTICLE_MODULE_TEST_BIRTH,
							false, i7::PARTICLE_STATE_DEF_FLAG_HAS_CHILD_EFFECTS, test_modules);

						store_group(ctx, &state->moduleGroupDefs[i7::PARTICLE_MODULE_GROUP_TEST], test_modules);
					}

					emitter->flags |= ctx.emitter_flags;
					state->flags |= ctx.state_flags;
				}
			}

			zonetool::iw7::ParticleSystemDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator)
			{
				const auto iw7_asset = allocator.allocate<i7::ParticleSystemDef>();

				iw7_asset->name = asset->name;

				convert_context ctx{};
				ctx.allocator = &allocator;
				ctx.system_flags = 0;
				ctx.fx_name = asset->name;

				const auto count = asset->elemDefCountLooping + asset->elemDefCountOneShot +
					asset->elemDefCountEmission;

				iw7_asset->numEmitters = 0;

				if (count > 0 && is_readable(asset->elemDefs, sizeof(FxElemDef)))
				{
					iw7_asset->emitterDefs = allocator.allocate_array<i7::ParticleEmitterDef>(count);

					auto converted = 0;
					for (auto i = 0; i < count; i++)
					{
						if (!is_readable(&asset->elemDefs[i], sizeof(FxElemDef)))
						{
							break;
						}

						auto* elem = &asset->elemDefs[i];
						ctx.elem_index = i;

						// Only convert element types with an IW7 equivalent and a resolvable
						// visual. Omni lights use the stock IW7 FX light profile. Spot lights,
						// sound, lens flare and unidentified types remain explicit drops.
						// rather than referencing an asset the game cannot load
						if (!is_convertible_type(elem->elemType))
						{
							ZONETOOL_WARNING("vfx \"%s\": dropping element %i (unsupported type %u)",
								asset->name, i, elem->elemType);
							continue;
						}

						if (is_sprite_type(elem->elemType) &&
							!element_has_resolvable_material_visual(ctx, elem))
						{
							ZONETOOL_WARNING("vfx \"%s\": dropping element %i (no resolvable material visual)",
								asset->name, i);
							continue;
						}

						convert_elem(ctx, &iw7_asset->emitterDefs[converted], elem,
							i < asset->elemDefCountLooping);
						converted++;
					}

					if (converted < count)
					{
						ZONETOOL_WARNING("vfx \"%s\" converted %i of its %i elements",
							asset->name, converted, count);
					}

					iw7_asset->numEmitters = converted;
				}

				ctx.system_flags |= i7::PARTICLE_SYSTEM_DEF_FLAG_KILL_STOPPED_INFINITE_EFFECTS;

				iw7_asset->flags = ctx.system_flags;

				iw7_asset->version = 15;

				iw7_asset->occlusionOverrideEmitterIndex = -1;

				iw7_asset->phaseOptions = i7::PARTICLE_PHASE_OPTION_PHASE_NEVER;

				// the shipped iw7 viewmodel muzzle flashes (vfx_muz_ar_v/_w, the exact
				// effect class) use drawFrustumCullRadius 0 with updateFrustumCullRadius
				// -1; a negative draw radius risks culling the whole system, so match the
				// game's own value rather than the reference converter's blanket -1
				iw7_asset->drawFrustumCullRadius = 0.0f;
				iw7_asset->updateFrustumCullRadius = -1.0f;

				iw7_asset->sunDistance = 100000.0f;

				iw7_asset->preRollMSec = 0;

				iw7_asset->editorPosition.v[0] = 0.0f;
				iw7_asset->editorPosition.v[1] = 0.0f;
				iw7_asset->editorPosition.v[2] = 0.0f;
				iw7_asset->editorPosition.v[3] = 0.0f;

				iw7_asset->editorRotation.v[0] = 0.0f;
				iw7_asset->editorRotation.v[1] = 0.0f;
				iw7_asset->editorRotation.v[2] = 0.0f;
				iw7_asset->editorRotation.v[3] = 1.0f;

				return iw7_asset;
			}

			void dump(FxEffectDef* asset)
			{
				if (!is_readable(asset, sizeof(FxEffectDef)) || !safe_name(asset->name))
				{
					return;
				}

				utils::memory::allocator allocator;
				const auto converted_asset = convert(asset, allocator);
				zonetool::iw7::particle_system::dump(converted_asset);
			}
		}
	}
}
