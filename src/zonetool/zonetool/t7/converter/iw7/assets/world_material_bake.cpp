#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "world_material_bake.hpp"

#include "material_texture.hpp"
#include "world_techset_donors.hpp"
#include "../shader_eval.hpp"
#include "../parallel.hpp"
#include "../gpu_eval.hpp"
#include "../forward_surface.hpp"

#include "zonetool/iw7/assets/gfximage.hpp"
#include "zonetool/t7/functions.hpp"
#include "zonetool/t7/converter/iw7/map_common.hpp"

#include <DirectXTex.h>
#include <utils/string.hpp>
#include <utils/io.hpp>

#include <numeric>

// How a BO3 gbuffer material becomes an IW7 packed world material.
//
// BO3 side (all from the game's own code):
// * the world is drawn with draw method 0 of a technique; methods 2-4 carry the same pixel shader
//   and method 1 is the debug-override permutation (PerSceneConsts debugColorOverride...).
// * gbuffer layout, decoded as BO3's deferred lighting pass (deferred_lighting.hlsl) reads it:
//   RT0 rgb albedo; RT1 xy normal (Lambert azimuthal around one of four tetrahedral axes picked by
//   RT1.w * 3), RT1.z gloss g encoded g * 0.497556 + 0.001466 (+0.5 when the shading model flag is
//   set), RT2 x specular colour luma Y, RT2 y one chroma channel (R - B on pixels whose x and y
//   parity match, G - (R + B) / 2 on the others, stored * 0.5 + 0.5), RT2 z occlusion, RT2 w
//   shading model. Roughness is alpha^2 = 2 / (2^(17 g) + 2).
// * decals write premultiplied values with ONE / INV_SRC_ALPHA blending and per-target write masks
//   (the technique's GfxStateMap); a masked-out channel keeps the surface underneath.
// * vertex colour reaches the pixel shader linearised (c^2.2) and is baked into the vertices, not
//   the textures: the evaluation runs with vertex colour 1.
//
// IW7 side (stock world shader lmap_specenv_..._p0, stock packed images and materials):
// * _packed_cs BC7: rgb = sqrt(diffuse albedo) (the shader squares it, UNORM), a = reflectance:
//   F0 = min(a, 0.1) + m * albedo and diffuse = albedo * (1 - m) with m = saturate((a - 0.1) / 0.9).
// * _packed_ng / _packed_nog BC7: r = gloss (g' = 1 - sqrt(1 - r), alpha^2 = 2 / (2^(20 g') + 2),
//   so r = 1 - (1 - 17 g / 20)^2 reproduces BO3's roughness exactly), g / a = the normal in the
//   45-degree rotated octahedral square (g = (px + py) * 0.5 + 0.5, a = (px - py) * 0.5 + 0.5),
//   b = occlusion for the o0 techsets (255 = none; IW7 only uses it for specular occlusion).
// * _packed_a BC4 (decal / alpha-test coverage), _packed_r / _packed_ar BC1 (r = coverage, g =
//   reveal). Stock flags: cs 0x2300, ng 0x308, a / r 0x300 (the 0x40 streamed bit dropped).
// * detail normal maps (q0) BC7: g / a as the packed normal, r 255, b 0; the world shader adds the
//   detail's octahedral coordinates to the base's.
// * reveal (v0): alpha = saturate((reveal - e0) / (e1 - e0)) with e0 = 1 - a - p * sqrt(a),
//   e1 = 1 - a + p * sqrt(1 - a), a = vertex alpha, p = revealParams.x. BO3 uses exponent
//   alphaRevealRamp instead of the fixed 0.5; p is fitted so both agree at a = 0.5.

namespace zonetool::t7
{
	namespace converter::iw7::world_material::bake
	{
		namespace
		{
			constexpr std::uint32_t sit_cbuffer = 0;
			constexpr std::uint32_t sit_texture = 2;
			constexpr std::uint32_t sit_sampler = 3;
			constexpr std::uint32_t sit_structured = 5;

			constexpr std::uint32_t hash_color_map = 0xA0AB1041; // "colorMap"
			constexpr std::uint32_t hash_normal_map = 0x59D30D0F; // "normalMap"
			constexpr std::uint32_t hash_spec_occlusion_map = 0xA52C26B4; // "specOcclusionMap"
			constexpr std::uint32_t hash_detail_map = 0xEB529B4D; // "detailMap"
			constexpr std::uint32_t hash_emissive_map = 0x34614347; // "emissiveMap"

			constexpr unsigned char iw7_region_lit_decal = 2; // stock IW7 decals' camera region (world_material classify)

			std::uint32_t r_hash_string(const std::string& s)
			{
				std::uint32_t h = 0;
				for (const auto c : s)
				{
					h = (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) | 0x20u) ^ (33u * h);
				}
				return h;
			}

			std::string technique_name(const MaterialTechnique* technique)
			{
				if (!technique || !technique->name)
				{
					return {};
				}
				const std::string name = technique->name;
				const auto dot = name.find('.');
				return dot == std::string::npos ? name : name.substr(dot + 1);
			}

			std::string clean_name(const std::string& name)
			{
				auto out = name;
				for (auto& c : out)
				{
					if (c == '*')
					{
						c = '_';
					}
				}
				return out;
			}

			// thread time per stage, summed over the workers
			struct stage_clock
			{
				std::atomic<std::uint64_t> textures{ 0 }; // decoding BO3 textures on the CPU
				std::atomic<std::uint64_t> gpu{ 0 }; // evaluating on the GPU, uploads and readback included
				std::atomic<std::uint64_t> cpu{ 0 }; // evaluating on the interpreter
				std::atomic<std::uint64_t> check{ 0 }; // the interpreter's side of the GPU agreement checks
				std::atomic<std::uint64_t> pack{ 0 }; // gbuffer decode and IW7 packing, coverage, fills
				std::atomic<std::uint64_t> compress{ 0 };
				std::atomic<std::uint64_t> write{ 0 };
			};
			stage_clock stage_ns;

			class timed
			{
			public:
				explicit timed(std::atomic<std::uint64_t>& total)
					: total_(total), start_(std::chrono::steady_clock::now())
				{
				}

				~timed()
				{
					this->total_ += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
						std::chrono::steady_clock::now() - this->start_).count());
				}

			private:
				std::atomic<std::uint64_t>& total_;
				std::chrono::steady_clock::time_point start_;
			};

			// ---- the material's program and bindings ------------------------------------------------

			struct texture_binding
			{
				std::uint32_t slot;
				std::string name;
				const GfxImage* image; // null: `constant` everywhere
				std::array<float, 4> constant = { 0.0f, 0.0f, 0.0f, 0.0f };
				material_texture::address_mode u = material_texture::address_mode::wrap;
				material_texture::address_mode v = material_texture::address_mode::wrap;
			};

			struct input_binding
			{
				enum class kind
				{
					position,
					colour,
					uv,
					normal,
					tangent,
					bitangent,
					uv_offsets,
					front_face,
					view_offset, // OFFPOSITION: the camera-relative position (forward shaders)
					zero, // interpolants a forward shader reads that do not carry material data
				};

				kind what;
				std::uint32_t reg;
				std::uint32_t mask;
				std::uint32_t semantic_index = 0; // COLOR0 / COLOR1
			};

			struct material_program
			{
				// what the vertex shader passes as COLOR0 and COLOR1 (use_vertex_colour), 1 unless known
				std::array<std::array<float, 4>, 2> colour = { { { 1.0f, 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } } };
				std::shared_ptr<shader_eval::program> program;
				bool forward = false; // run with the engine's lighting resources unbound (they read zero)
				int technique_index = -1;
				const MaterialTechnique* technique = nullptr;
				int draw_method = 0; // of the technique, whose shaders the program comes from
				const MaterialPixelShader* pixel_shader = nullptr; // the technique's pixel shader the program comes from
				// the program when it is not that pixel shader as it is: the forward_surface cut of the lit technique's
				// debug-override permutation (it lives as long as the process), else null
				const std::vector<std::uint8_t>* bytecode = nullptr;
				std::vector<std::pair<std::uint32_t, std::vector<std::uint8_t>>> cbuffers; // slot -> data
				std::uint32_t globals_slot = 0;
				const shader_eval::cbuffer_desc* globals = nullptr;
				std::vector<texture_binding> textures;
				struct buffer_binding
				{
					std::uint32_t slot;
					std::vector<std::uint8_t> data;
					std::uint32_t stride;
				};
				std::vector<buffer_binding> buffers; // structured buffers: BO3's per-instance data (load_program)
				std::optional<std::uint32_t> resolved_normal_slot;
				std::vector<input_binding> inputs;
				std::int32_t target_reg[3] = { -1, -1, -1 };
				shader_eval::taint_result taint;
			};

			std::mutex program_mutex;
			std::unordered_map<const void*, std::shared_ptr<shader_eval::program>> programs;

			// the program of `data`, parsed once per key (a pixel shader, or a program made from one)
			std::shared_ptr<shader_eval::program> get_program(const void* key, const std::uint8_t* data, const std::size_t size)
			{
				std::lock_guard _(program_mutex);
				const auto found = programs.find(key);
				if (found != programs.end())
				{
					return found->second;
				}
				auto prog = shader_eval::program::parse(data, size);
				programs[key] = prog;
				return prog;
			}

			std::shared_ptr<shader_eval::program> get_program(const MaterialPixelShader* ps)
			{
				return get_program(ps, ps->prog.loadDef.program, ps->prog.loadDef.programSize);
			}

			const MaterialTextureDef* find_texture(const Material* material, const std::uint32_t hash)
			{
				for (auto i = 0; i < material->textureCount; i++)
				{
					if (material->textureTable[i].nameHash == hash)
					{
						return &material->textureTable[i];
					}
				}
				return nullptr;
			}

			const MaterialSamplerDef* find_sampler(const Material* material, const std::uint32_t hash)
			{
				for (auto i = 0; i < material->samplerCount; i++)
				{
					if (material->samplerTable[i].nameHash == hash)
					{
						return &material->samplerTable[i];
					}
				}
				return nullptr;
			}

			// GfxSamplerState address bits (inferred, see the note in address_modes): states with non-zero
			// bits 5-10 seen are 0x2A1 and 0x2A2; others (0x2, 0xA, 0x12, 0x13, 0x14, 0x15) have them clear.
			void address_modes(const GfxSamplerState state, material_texture::address_mode& u, material_texture::address_mode& v)
			{
				// the bit layout is inferred from which materials set these bits, not read from BO3's
				// sampler creation
				const auto mode = [](std::uint32_t bits)
				{
					switch (bits & 3)
					{
					case 1: return material_texture::address_mode::clamp;
					case 2: return material_texture::address_mode::mirror;
					default: return material_texture::address_mode::wrap;
					}
				};
				u = mode(state >> 5);
				v = mode(state >> 7);
			}

			// which sampler register each texture register is sampled with (first use)
			std::unordered_map<std::uint32_t, std::uint32_t> sampler_of_texture(const shader_eval::program& prog)
			{
				std::unordered_map<std::uint32_t, std::uint32_t> result;
				for (const auto& ins : prog.instructions())
				{
					if (ins.operands.size() >= 4 && ins.operands[2].kind == shader_eval::operand_kind::resource &&
						ins.operands[3].kind == shader_eval::operand_kind::sampler)
					{
						result.emplace(ins.operands[2].index[0], ins.operands[3].index[0]);
					}
				}
				return result;
			}

			std::string describe_material(const Material* material)
			{
				const auto* ts = material->techniqueSet;
				return utils::string::va("%s (%s)", material->name, ts && ts->name ? ts->name : "no techset");
			}

			struct program_options
			{
				// another technique set's shader, evaluated with this material's textures and its $Globals
				// filled by variable name from `constants_from` (a program of the material's own technique set)
				const MaterialTechniqueSet* techset = nullptr;
				const material_program* constants_from = nullptr;
				// a forward shader run without lighting: the engine's textures and buffers stay unbound (read
				// zero), the relative HDR exposure is 1, the camera sits over the surface along its normal
				bool forward = false;
				// the technique's debug-override permutation (draw method 1) cut to the surface it computes
				// (forward_surface), run like a forward shader with every override weight 0
				bool surface = false;
				// the technique's pixel shader cut at its alpha test, the coverage it compares in render target 0
				// (forward_surface::coverage)
				bool coverage = false;
				// only the constant buffers are filled (the material's constants are read, the program is not run): no
				// textures, buffers or inputs are bound
				bool constants_only = false;
			};

			bool has_technique(const MaterialTechniqueSet* ts, const char* technique)
			{
				for (auto i = 0; ts && i < 12; i++)
				{
					if (ts->techniques[i] && technique_name(ts->techniques[i]) == technique)
					{
						return true;
					}
				}
				return false;
			}

			// whether a technique adds its output: BO3 blend SRC_ALPHA (5) or ONE (2) onto ONE (2), ADD (1)
			bool adds(const MaterialTechnique* technique)
			{
				const auto* state = technique ? technique->stateMap : nullptr;
				if (!state)
				{
					return false;
				}
				const auto& b = state->blend[0];
				return b.rgbOp == 1 && b.rgbDst == 2 && (b.rgbSrc == 2 || b.rgbSrc == 5);
			}

			material_program load_program(const Material* material, const char* technique, const program_options& options = {})
			{
				material_program mp{};
				mp.forward = options.forward || options.surface;

				const auto* ts = options.techset ? options.techset : material->techniqueSet;
				if (!ts)
				{
					throw std::runtime_error("no technique set");
				}
				for (auto i = 0; i < 12; i++)
				{
					if (ts->techniques[i] && technique_name(ts->techniques[i]) == technique)
					{
						mp.technique_index = i;
						mp.technique = ts->techniques[i];
						break;
					}
				}
				if (!mp.technique)
				{
					throw std::runtime_error(utils::string::va("no \"%s\" technique", technique));
				}

				const auto method = options.surface ? 1 : 0;
				const auto* ps = mp.technique->drawMethods[method].pixelShader;
				if (!ps || !ps->prog.loadDef.program || !ps->prog.loadDef.programSize)
				{
					throw std::runtime_error(utils::string::va("draw method %d has no pixel shader program", method));
				}
				mp.pixel_shader = ps;
				mp.draw_method = method;
				if (options.coverage)
				{
					const auto& cut = forward_surface::coverage(ps->prog.loadDef.program, ps->prog.loadDef.programSize);
					if (cut.bytecode.empty())
					{
						throw std::runtime_error("alpha-test coverage: " + cut.reason);
					}
					mp.bytecode = &cut.bytecode;
					mp.program = get_program(cut.bytecode.data(), cut.bytecode.data(), cut.bytecode.size());
				}
				else if (options.surface)
				{
					const auto& surface = forward_surface::get(ps->prog.loadDef.program, ps->prog.loadDef.programSize);
					if (surface.bytecode.empty())
					{
						throw std::runtime_error("forward surface: " + surface.reason);
					}
					mp.bytecode = &surface.bytecode;
					mp.program = get_program(surface.bytecode.data(), surface.bytecode.data(), surface.bytecode.size());

					// the material's static $Globals data is laid out for the technique's draw method 0 shader
					const auto* method0 = mp.technique->drawMethods[0].pixelShader;
					if (!method0 || !method0->prog.loadDef.program)
					{
						throw std::runtime_error("the lit technique's draw method 0 has no pixel shader program");
					}
					const auto* ours = mp.program->find_cbuffer("$Globals");
					const auto* theirs = get_program(method0)->find_cbuffer("$Globals");
					for (const auto& v : ours ? ours->variables : std::vector<shader_eval::cbuffer_variable>{})
					{
						const auto same = theirs && std::ranges::any_of(theirs->variables, [&](const shader_eval::cbuffer_variable& w)
						{
							return w.name == v.name && w.offset == v.offset && w.size == v.size;
						});
						if (!same)
						{
							throw std::runtime_error(utils::string::va("the debug-override permutation places $Globals \"%s\" elsewhere", v.name.data()));
						}
					}
				}
				else
				{
					mp.program = get_program(ps);
				}
				const auto& prog = *mp.program;

				// constant buffers: $Globals from the material's static buffer, the rest zero
				const auto& cb_info = material->constantBufferInfo.constantBuffers[mp.technique_index];
				for (const auto& b : prog.bindings())
				{
					if (b.input_type != sit_cbuffer)
					{
						continue;
					}
					const auto* desc = prog.find_cbuffer(b.name);
					const auto size = desc ? desc->size : 0u;
					std::vector<std::uint8_t> data(size, 0);
					// (a surface program's debug override weights stay 0: every labelled value is the material's)
					if (mp.forward && b.name == "PerSceneConsts" && desc)
					{
						// relHDRExposure as BO3 renders every scene with it: RB_CallExecuteRenderCommands (dedi
						// 0x140338300) sets code constant 7 to (1, 1, 0, 0) and no other path writes it (the view model
						// transparent callback's (1, 1, 0, 1) differs only in .w, which no shader reads). .y scales the
						// lit output (BO3 forward shaders' last multiply: no pre-exposure); .x is the relativeHDR emissive
						// floor, emissive * max(luma(lit colour), .x), 1 here as the lighting is zero
						for (const auto& v : desc->variables)
						{
							if (v.name == "relHDRExposure" && v.offset + 8 <= data.size())
							{
								constexpr float value[2] = { 1.0f, 1.0f };
								std::memcpy(data.data() + v.offset, value, sizeof(value));
							}
						}
					}
					if (b.name == "$Globals" && options.constants_from)
					{
						// by variable name from the material's own program
						const auto& from = *options.constants_from;
						if (!from.globals || !desc)
						{
							throw std::runtime_error("no $Globals to take the constants from");
						}
						const std::vector<std::uint8_t>* source = nullptr;
						for (const auto& [slot, bytes] : from.cbuffers)
						{
							if (slot == from.globals_slot)
							{
								source = &bytes;
							}
						}
						for (const auto& v : desc->variables)
						{
							const shader_eval::cbuffer_variable* match = nullptr;
							for (const auto& w : from.globals->variables)
							{
								if (w.name == v.name)
								{
									match = &w;
								}
							}
							if (!match || !source)
							{
								throw std::runtime_error(utils::string::va("the material has no constant \"%s\" for %s", v.name.data(), ts->name));
							}
							const auto count = std::min(v.size, match->size);
							if (v.offset + count > data.size() || match->offset + count > source->size())
							{
								throw std::runtime_error(utils::string::va("constant \"%s\" is outside its buffer", v.name.data()));
							}
							std::memcpy(data.data() + v.offset, source->data() + match->offset, count);
						}
						mp.globals_slot = b.bind_point;
						mp.globals = desc;
					}
					else if (b.name == "$Globals")
					{
						const MaterialStaticConstantBuffer* buffer = nullptr;
						for (const auto* candidate : cb_info.psStatic)
						{
							if (candidate && candidate->deviceSlot == b.bind_point)
							{
								buffer = candidate;
							}
						}
						if (!buffer || !buffer->data.loadDef.initData)
						{
							throw std::runtime_error(utils::string::va("no static constant buffer for $Globals (slot %u)", b.bind_point));
						}
						const auto available = std::min<std::size_t>(buffer->data.loadDef.initDataSize, data.size());
						std::memcpy(data.data(), buffer->data.loadDef.initData, available);
						mp.globals_slot = b.bind_point;
						mp.globals = desc;
					}
					mp.cbuffers.emplace_back(b.bind_point, std::move(data));
				}
				if (options.constants_only)
				{
					return mp;
				}

				// textures by R_HashString of the RDEF name, as the material's texture table keys them
				const auto samplers = sampler_of_texture(prog);
				std::unordered_map<std::uint32_t, std::string> sampler_names;
				for (const auto& b : prog.bindings())
				{
					if (b.input_type == sit_sampler)
					{
						sampler_names[b.bind_point] = b.name;
					}
				}
				for (const auto& b : prog.bindings())
				{
					if (b.input_type != sit_texture)
					{
						continue;
					}
					if (b.name == "resolvedNormal")
					{
						mp.resolved_normal_slot = b.bind_point;
						continue;
					}
					const auto* def = find_texture(material, r_hash_string(b.name));
					if ((!def || !def->image) && mp.forward)
					{
						continue; // an engine resource (probes, lights, shadows): left unbound
					}
					if ((!def || !def->image) && b.name.starts_with("gWeather"))
					{
						// BO3's weather-grime layers, which the engine binds from the volume_weathergrime a draw is in: the
						// bake's LightingGlobals.weather (their direction, tint, tiling) is 0, as outside every weather
						// volume, and they read 0 (a texture without an image)
						mp.textures.push_back({ b.bind_point, b.name, nullptr });
						continue;
					}
					if (!def || !def->image)
					{
						throw std::runtime_error(utils::string::va("texture \"%s\" is not in the material's texture table", b.name.data()));
					}
					if (!readable(def->image, sizeof(GfxImage)))
					{
						// an image of a zone that is not on disk, so not loaded in game either: BO3 draws its default image
						// asset instead, $white (the default name table 0x1410C1BF0 of the dedicated server, image type 9),
						// an 8x8 BC7 image of white: white everywhere
						ZONETOOL_INFO("material %s: texture \"%s\" is an image of a zone that was not loaded: BO3's default image $white "
							"(white), as the game draws it", material->name, b.name.data());
						mp.textures.push_back({ b.bind_point, b.name, nullptr, { 1.0f, 1.0f, 1.0f, 1.0f } });
						continue;
					}
					texture_binding tb{ b.bind_point, b.name, def->image };
					const auto sampler = samplers.find(b.bind_point);
					if (sampler != samplers.end() && sampler_names.contains(sampler->second))
					{
						if (const auto* s = find_sampler(material, r_hash_string(sampler_names[sampler->second])))
						{
							address_modes(s->samplerState, tb.u, tb.v);
						}
					}
					mp.textures.emplace_back(std::move(tb));
				}

				// BO3's per-instance data (the RDEF structs: ModelInstanceData 24 bytes, whose shaderConstantSet indexes the
				// GpuShaderConstantSet of 384 bytes): one instance, constant set 0, every script and weapon constant 0 as
				// the engine has them until a script sets one, except a reveal template's reveal amount (scriptVector0.x):
				// 1, fully revealed, as the transparent reveal_script templates' vertex shaders pass it (COLOR1.x, bake 1).
				// A forward program reads what is not bound as 0, so there only a constant set that is not 0 is bound (a
				// bound buffer keeps the program off the GPU, whose evaluator binds none).
				const auto revealed = bo3_template(material).find("reveal_script") != std::string::npos;
				for (const auto& b : prog.bindings())
				{
					if (b.input_type != sit_structured)
					{
						continue;
					}
					if (b.name == "modelInstanceBuffer" && !mp.forward)
					{
						mp.buffers.push_back({ b.bind_point, std::vector<std::uint8_t>(24), 24 });
					}
					else if (b.name == "shaderConstantSetBuffer" && (!mp.forward || revealed))
					{
						std::vector<std::uint8_t> set(384);
						if (revealed)
						{
							const auto one = 1.0f;
							std::memcpy(set.data(), &one, sizeof(one));
						}
						mp.buffers.push_back({ b.bind_point, std::move(set), 384 });
					}
				}

				// inputs by semantic
				for (const auto& e : prog.inputs())
				{
					input_binding ib{};
					ib.reg = e.reg;
					ib.mask = e.mask;
					const auto& s = e.semantic;
					if (!_stricmp(s.data(), "SV_POSITION"))
					{
						ib.what = input_binding::kind::position;
					}
					else if (!_stricmp(s.data(), "COLOR"))
					{
						ib.what = input_binding::kind::colour;
						ib.semantic_index = e.semantic_index;
					}
					else if (!_stricmp(s.data(), "TEXCOORD") && e.semantic_index == 0)
					{
						ib.what = input_binding::kind::uv;
					}
					else if (!_stricmp(s.data(), "TEXCOORD") && e.semantic_index == 1)
					{
						ib.what = input_binding::kind::normal;
					}
					else if (!_stricmp(s.data(), "TEXCOORD") && e.semantic_index == 2)
					{
						ib.what = input_binding::kind::tangent;
					}
					else if (!_stricmp(s.data(), "TEXCOORD") && e.semantic_index == 3)
					{
						ib.what = input_binding::kind::bitangent;
					}
					else if (!_stricmp(s.data(), "UVOFFSETS"))
					{
						ib.what = input_binding::kind::uv_offsets;
					}
					else if (!_stricmp(s.data(), "SV_IsFrontFace"))
					{
						ib.what = input_binding::kind::front_face;
					}
					else if (!_stricmp(s.data(), "OFFPOSITION"))
					{
						// forward shaders, and the cloth gbuffer shaders' view-dependent sheen: evaluated head-on
						ib.what = input_binding::kind::view_offset;
					}
					else if (mp.forward && !_stricmp(s.data(), "TEXCOORD") && e.semantic_index >= 4)
					{
						ib.what = input_binding::kind::zero;
					}
					else if (!_stricmp(s.data(), "TEXCOORD") && e.semantic_index == 4 && prog.find_binding("modelInstanceBuffer", sit_structured))
					{
						// the model instance index into modelInstanceBuffer: its one instance (the buffers below)
						ib.what = input_binding::kind::zero;
					}
					else if (e.interpolation == 0)
					{
						// in the signature but never declared, so never read (lit_decal_raindrops / lit_paintshop
						// gbuffer shaders list the instance index TEXCOORD4 without a dcl_input for it)
						ib.what = input_binding::kind::zero;
					}
					else
					{
						throw std::runtime_error(utils::string::va("unhandled pixel shader input %s%u", s.data(), e.semantic_index));
					}
					mp.inputs.push_back(ib);
				}

				for (const auto& e : prog.outputs())
				{
					if (!_stricmp(e.semantic.data(), "SV_TARGET") && e.semantic_index < 3)
					{
						mp.target_reg[e.semantic_index] = static_cast<std::int32_t>(e.reg);
					}
				}

				mp.taint = shader_eval::analyse(prog);
				return mp;
			}

			const shader_eval::cbuffer_variable* find_variable(const material_program& mp, const char* name)
			{
				if (!mp.globals)
				{
					return nullptr;
				}
				for (const auto& v : mp.globals->variables)
				{
					if (v.name == name)
					{
						return &v;
					}
				}
				return nullptr;
			}

			float read_float(const material_program& mp, const shader_eval::cbuffer_variable* var, std::uint32_t component)
			{
				for (const auto& [slot, data] : mp.cbuffers)
				{
					if (slot == mp.globals_slot && var && var->offset + component * 4 + 4 <= data.size())
					{
						float value;
						std::memcpy(&value, data.data() + var->offset + component * 4, 4);
						return value;
					}
				}
				return 0.0f;
			}

			// a bool constant: BO3 stores true as the bits of 1.0f, the shaders test the bits against zero
			bool read_bool(const material_program& mp, const shader_eval::cbuffer_variable* var)
			{
				return std::bit_cast<std::uint32_t>(read_float(mp, var, 0)) != 0;
			}

			void write_float(material_program& mp, const shader_eval::cbuffer_variable* var, std::uint32_t component, float value)
			{
				for (auto& [slot, data] : mp.cbuffers)
				{
					if (slot == mp.globals_slot && var && var->offset + component * 4 + 4 <= data.size())
					{
						std::memcpy(data.data() + var->offset + component * 4, &value, 4);
					}
				}
			}

			const texture_binding* find_binding(const material_program& mp, const char* name)
			{
				for (const auto& t : mp.textures)
				{
					if (t.name == name)
					{
						return &t;
					}
				}
				return nullptr;
			}

			// What BO3's model vertex shader passes the pixel shader as COLOR0 / COLOR1 for a material whose surfaces have one
			// value of a vertex colour channel, as shader_eval::vertex_colour reads it from the technique's vertex shader
			// (lit_flag's gbuffer VS: COLOR0.rgb = rgb^2.2; lit_alphatest's: COLOR1.x = alpha); anything the vertex shader makes
			// another way (a _script template's script constant) stays 1, as do world materials and channels that vary. The
			// stream's bytes are taken in x64-zt's GfxStreamVertex order (Color[0] as x): the alpha is the fourth byte in RGBA
			// and BGRA alike, the others are not verified against BO3's input layout.
			// how the program's technique's vertex shader makes each component of COLOR0 / COLOR1, when it can be read
			std::optional<std::array<std::vector<shader_eval::colour_source>, 2>> colour_sources(const material_program& mp)
			{
				const auto* vs = mp.technique ? mp.technique->drawMethods[mp.draw_method].vertexShader : nullptr;
				if (!vs || !vs->prog.loadDef.program || !vs->prog.loadDef.programSize)
				{
					return std::nullopt;
				}
				try
				{
					return shader_eval::vertex_colour(*get_program(vs, vs->prog.loadDef.program, vs->prog.loadDef.programSize));
				}
				catch (const std::exception&)
				{
					return std::nullopt;
				}
			}

			void use_vertex_colour(material_program& mp, const info& inf)
			{
				if (!inf.model || !inf.used_colour || inf.used_colour->empty())
				{
					return;
				}
				const auto sources = colour_sources(mp);
				if (!sources)
				{
					return;
				}
				const auto& range = *inf.used_colour;
				for (auto index = 0u; index < 2; index++)
				{
					for (auto k = 0u; k < (*sources)[index].size() && k < 4; k++)
					{
						const auto& s = (*sources)[index][k];
						if (range.min[s.channel] != range.max[s.channel])
						{
							continue; // (colour_channels_read reports it)
						}
						if (s.what == shader_eval::colour_source::kind::channel)
						{
							mp.colour[index][k] = range.min[s.channel] / 255.0f;
						}
						else if (s.what == shader_eval::colour_source::kind::linearised)
						{
							mp.colour[index][k] = std::pow(range.min[s.channel] / 255.0f, 2.2f);
						}
					}
				}
			}

			// The vertex colour channels (bit 0-3: r, g, b, a) the program's reads of COLOR come from, through its vertex
			// shader (colour_sources); bit 4: a component it reads that the vertex shader makes another way (a script
			// constant) or that cannot be traced. `linearised`: the channels of those it reads raised to 2.2.
			std::uint32_t colour_channels_read(const material_program& mp, std::uint32_t* linearised = nullptr)
			{
				if (linearised)
				{
					*linearised = 0;
				}
				const auto sources = colour_sources(mp);
				std::uint32_t channels = 0;
				for (const auto& in : mp.inputs)
				{
					if (in.what != input_binding::kind::colour || in.reg >= 32)
					{
						continue;
					}
					auto slot = 0u;
					for (auto c = 0u; c < 4; c++)
					{
						if (!(in.mask & (1u << c)))
						{
							continue;
						}
						const auto k = slot++;
						const auto bit = in.reg * 4 + c;
						auto read = mp.taint.discard_inputs.test(bit);
						for (const auto& target : mp.taint.output_inputs)
						{
							for (const auto& component : target)
							{
								read = read || component.test(bit);
							}
						}
						if (!read)
						{
							continue;
						}
						const auto index = in.semantic_index == 0 ? 0u : 1u;
						const auto* s = sources && k < (*sources)[index].size() ? &(*sources)[index][k] : nullptr;
						if (s && (s->what == shader_eval::colour_source::kind::channel || s->what == shader_eval::colour_source::kind::linearised))
						{
							channels |= 1u << s->channel;
							if (linearised && s->what == shader_eval::colour_source::kind::linearised)
							{
								*linearised |= 1u << s->channel;
							}
						}
						else if (!s || s->what != shader_eval::colour_source::kind::one)
						{
							channels |= 0x10;
						}
					}
				}
				return channels;
			}

			// whether the program reads BO3's weather constants (LightingGlobals.weather: rain, wind, weather tile and
			// vectors), which the engine sets at run time from the level's weather; the bake runs with them at 0
			bool reads_weather(const material_program& mp)
			{
				if (!mp.program)
				{
					return false;
				}
				const auto& prog = *mp.program;
				const auto* lighting = prog.find_cbuffer("LightingGlobals");
				const auto* binding = prog.find_binding("LightingGlobals", sit_cbuffer);
				if (!lighting || !binding)
				{
					return false;
				}
				const auto weather = std::ranges::find_if(lighting->variables, [](const shader_eval::cbuffer_variable& v)
				{
					return v.name == "weather";
				});
				if (weather == lighting->variables.end() || !weather->size)
				{
					return false;
				}
				const auto first = weather->offset / 16;
				const auto last = (weather->offset + weather->size - 1) / 16;
				for (const auto& ins : prog.instructions())
				{
					for (const auto& op : ins.operands)
					{
						if (op.kind == shader_eval::operand_kind::cbuffer && op.index[0] == binding->bind_point
							&& (op.relative || (op.index[1] >= first && op.index[1] <= last)))
						{
							return true;
						}
					}
				}
				return false;
			}

			const texture_binding* binding_by_slot(const material_program& mp, std::uint32_t slot)
			{
				for (const auto& t : mp.textures)
				{
					if (t.slot == slot)
					{
						return &t;
					}
				}
				return nullptr;
			}

			// ---- evaluation -------------------------------------------------------------------------

			struct bound_textures
			{
				std::vector<std::unique_ptr<shader_eval::texture_source>> owned;
				std::vector<std::pair<std::uint32_t, const shader_eval::texture_source*>> slots;
			};

			// Binds every texture of the program. `overrides` replaces a named texture with a constant.
			bound_textures bind_textures(const material_program& mp,
				const std::unordered_map<std::string, std::array<float, 4>>& overrides)
			{
				bound_textures bound{};
				for (const auto& t : mp.textures)
				{
					if (!t.image)
					{
						// a texture without an image: its constant (gWeather* 0, BO3's $white in place of an unloaded image 1)
						auto c = std::make_unique<material_texture::constant_texture>(t.constant[0], t.constant[1], t.constant[2], t.constant[3],
							1, 1);
						bound.slots.emplace_back(t.slot, c.get());
						bound.owned.emplace_back(std::move(c));
						continue;
					}
					const auto o = overrides.find(t.name);
					if (o != overrides.end())
					{
						auto c = std::make_unique<material_texture::constant_texture>(o->second[0], o->second[1], o->second[2], o->second[3],
							t.image->width, t.image->height);
						bound.slots.emplace_back(t.slot, c.get());
						bound.owned.emplace_back(std::move(c));
						continue;
					}
					auto decoded = material_texture::decode(t.image);
					if (!decoded)
					{
						throw std::runtime_error(utils::string::va("texture \"%s\" (%s) has no pixels", t.name.data(), t.image->name));
					}
					auto tex = std::make_unique<material_texture::texture>(decoded, t.u, t.v);
					bound.slots.emplace_back(t.slot, tex.get());
					bound.owned.emplace_back(std::move(tex));
				}
				if (mp.resolved_normal_slot)
				{
					// what a decal reads back from the gbuffer under it: a surface that takes decals
					// (encoded gloss in (0, 0.5)) whose normal was encoded around basis 0
					auto c = std::make_unique<material_texture::constant_texture>(0.0f, 0.0f, 0.25f, 0.0f, 1, 1);
					bound.slots.emplace_back(*mp.resolved_normal_slot, c.get());
					bound.owned.emplace_back(std::move(c));
				}
				return bound;
			}

			// a level of the bake: texel (x, y) is base uv (origin_u + (x + 0.5) / width * span_u, likewise v)
			struct grid
			{
				std::uint32_t width;
				std::uint32_t height;
				float origin_u;
				float origin_v;
				float span_u;
				float span_v;
				std::uint32_t parity_shift; // added to the pixel x position (checkerboard channels)
			};

			struct texel_out
			{
				float target[3][4];
				bool discarded;
			};

			// texels [x, x + width) x [y, y + height) of a level, evaluated together; x and y are multiples of the
			// GPU tile size (so of the interpreter's 8x8 blocks, and of every power of two a shader takes the pixel
			// position modulo, which the GPU sees relative to the tile)
			struct tile
			{
				std::uint32_t x;
				std::uint32_t y;
				std::uint32_t width;
				std::uint32_t height;
			};

			std::vector<tile> tiles_of(const std::uint32_t width, const std::uint32_t height)
			{
				constexpr auto size = gpu_eval::context::max_tile;
				std::vector<tile> tiles;
				for (auto y = 0u; y < height; y += size)
				{
					for (auto x = 0u; x < width; x += size)
					{
						tiles.push_back({ x, y, std::min(size, width - x), std::min(size, height - y) });
					}
				}
				return tiles;
			}

			// a COLOR input by register component: its element's components in order take mp.colour's (COLOR0 rgb, COLOR1
			// its one value)
			std::array<float, 4> colour_input(const material_program& mp, const input_binding& in)
			{
				std::array<float, 4> values = { 1.0f, 1.0f, 1.0f, 1.0f };
				const auto& colour = mp.colour[in.semantic_index == 0 ? 0 : 1];
				auto slot = 0u;
				for (auto c = 0u; c < 4; c++)
				{
					if (in.mask & (1u << c))
					{
						values[c] = colour[slot++];
					}
				}
				return values;
			}

			// whether the program's discard reads a COLOR input component the bake feeds at another value than 1
			bool discard_reads_colour(const material_program& mp)
			{
				for (const auto& in : mp.inputs)
				{
					if (in.what != input_binding::kind::colour || in.reg >= 32)
					{
						continue;
					}
					const auto values = colour_input(mp, in);
					for (auto c = 0u; c < 4; c++)
					{
						if ((in.mask & (1u << c)) && values[c] != 1.0f && mp.taint.discard_inputs.test(in.reg * 4 + c))
						{
							return true;
						}
					}
				}
				return false;
			}

			void set_inputs(shader_eval::machine& m, const material_program& mp, const grid& g, const std::uint32_t block_x, const std::uint32_t block_y)
			{
				for (auto lane = 0u; lane < shader_eval::lane_count; lane++)
				{
					const auto quad = lane / 4;
					const auto x = block_x * 8 + (quad % 4) * 2 + (lane & 1u);
					const auto y = block_y * 8 + (quad / 4) * 2 + ((lane >> 1) & 1u);
					const auto u = g.origin_u + (static_cast<float>(x) + 0.5f) / static_cast<float>(g.width) * g.span_u;
					const auto v = g.origin_v + (static_cast<float>(y) + 0.5f) / static_cast<float>(g.height) * g.span_v;

					for (const auto& in : mp.inputs)
					{
						float values[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
						std::uint32_t bits[4] = { 0, 0, 0, 0 };
						auto use_bits = false;
						switch (in.what)
						{
						case input_binding::kind::position:
							values[0] = static_cast<float>(x + g.parity_shift) + 0.5f;
							values[1] = static_cast<float>(y) + 0.5f;
							values[2] = 0.5f;
							values[3] = 1.0f;
							break;
						case input_binding::kind::colour:
						{
							const auto colour = colour_input(mp, in);
							std::copy(colour.begin(), colour.end(), values);
							break;
						}
						case input_binding::kind::uv:
						{
							// the first two components the register carries: xy in the world shaders, yz in the
							// model ones (COLOR1 takes x)
							auto slot = 0u;
							for (auto c = 0u; c < 4 && slot < 2; c++)
							{
								if (in.mask & (1u << c))
								{
									values[c] = slot++ == 0 ? u : v;
								}
							}
							break;
						}
						case input_binding::kind::normal:
							values[2] = 1.0f;
							break;
						case input_binding::kind::tangent:
							values[0] = 1.0f;
							break;
						case input_binding::kind::bitangent:
							values[1] = 1.0f;
							break;
						case input_binding::kind::uv_offsets:
						case input_binding::kind::zero:
							break;
						case input_binding::kind::view_offset:
							// the camera 100 units over the surface along its normal: view-dependent terms
							// (emissive falloff, parallax layers) as seen head-on
							values[2] = -100.0f;
							break;
						case input_binding::kind::front_face:
							use_bits = true;
							bits[0] = bits[1] = bits[2] = bits[3] = 0xFFFFFFFFu;
							break;
						}
						for (auto c = 0u; c < 4; c++)
						{
							if (in.mask & (1u << c))
							{
								if (use_bits)
								{
									m.set_input(in.reg, c, lane, bits[c]);
								}
								else
								{
									m.set_input(in.reg, c, lane, values[c]);
								}
							}
						}
					}
				}
			}

			// the program over `t` on the interpreter, tile-local row-major
			std::vector<texel_out> evaluate(const material_program& mp, const bound_textures& bound, const grid& g, const tile& t)
			{
				std::vector<texel_out> out(static_cast<std::size_t>(t.width) * t.height);
				const auto blocks_x = (t.width + 7) / 8;
				const auto blocks_y = (t.height + 7) / 8;

				std::vector<std::unique_ptr<shader_eval::machine>> machines(std::thread::hardware_concurrency() + 1);

				parallel_for(blocks_x * blocks_y, [&](const std::uint32_t block, const std::uint32_t thread)
				{
					auto& m = machines[thread];
					if (!m)
					{
						m = std::make_unique<shader_eval::machine>(mp.program);
						m->set_unbound_resources_zero(mp.forward);
						for (const auto& [slot, data] : mp.cbuffers)
						{
							m->bind_cbuffer(slot, data.data(), data.size());
						}
						for (const auto& [slot, tex] : bound.slots)
						{
							m->bind_texture(slot, tex);
						}
						for (const auto& b : mp.buffers)
						{
							m->bind_structured(b.slot, b.data.data(), b.data.size(), b.stride);
						}
					}

					const auto block_x = t.x / 8 + block % blocks_x;
					const auto block_y = t.y / 8 + block / blocks_x;
					set_inputs(*m, mp, g, block_x, block_y);
					m->run();

					const auto discarded = m->discarded();
					for (auto lane = 0u; lane < shader_eval::lane_count; lane++)
					{
						const auto quad = lane / 4;
						const auto x = block_x * 8 + (quad % 4) * 2 + (lane & 1u);
						const auto y = block_y * 8 + (quad / 4) * 2 + ((lane >> 1) & 1u);
						if (x >= t.x + t.width || y >= t.y + t.height)
						{
							continue;
						}
						auto& o = out[static_cast<std::size_t>(y - t.y) * t.width + (x - t.x)];
						o.discarded = (discarded >> lane) & 1u;
						for (auto rt = 0; rt < 3; rt++)
						{
							for (auto c = 0u; c < 4; c++)
							{
								o.target[rt][c] = mp.target_reg[rt] >= 0 ? m->output(mp.target_reg[rt], c, lane) : 0.0f;
							}
						}
					}
				});

				return out;
			}

			// ---- BO3 gbuffer decode (deferred_lighting.hlsl) ----------------------------------------

			void decode_normal(const float enc[4], float n[3])
			{
				const auto basis = static_cast<std::uint32_t>(std::max(0.0f, enc[3] * 3.0f + 0.5f));
				const float px = (enc[0] * 2.0f - 1.0f) * 0.85f;
				const float py = (enc[1] * 2.0f - 1.0f) * 0.85f;
				const auto d = px * px + py * py;
				const auto s = std::sqrt(std::max(0.0f, 2.0f - d));
				const auto qx = s * px;
				const auto qy = s * py;

				const float axis_x = (basis & 2u) ? -1.0f : 1.0f;
				const float axis_y = (basis & 1u) ? -1.0f : 1.0f;
				const float axis_z = (((basis & 1u) != 0) != ((basis & 2u) != 0)) ? -1.0f : 1.0f;

				const auto k = (1.0f - d) * 0.577350f;
				const auto kx = qx * 0.408248f;
				const auto ky = qy * 0.707107f;
				n[0] = axis_x * k + axis_x * kx - axis_x * ky;
				n[1] = axis_y * k - 2.0f * axis_y * kx;
				n[2] = axis_z * k + axis_z * kx + axis_z * ky;
				const auto len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
				if (len > 0.0f)
				{
					n[0] /= len;
					n[1] /= len;
					n[2] /= len;
				}
			}

			struct surface
			{
				float albedo[3];
				float normal[3];
				float gloss; // BO3 g
				float f0[3];
				float occlusion;
				float coverage[3]; // alpha of RT0, RT1, RT2 (decals)
				float shading_model; // RT2.w of opaque surfaces: 1/3 standard, 2/3 backlit
				bool model_flag;
				bool discarded;
			};

			float unpremultiply(float value, float alpha)
			{
				return alpha > (1.0f / 1024.0f) ? value / alpha : value;
			}

			// Decodes the three render targets. Decals write premultiplied values (every target's w
			// is the coverage) and encode their normal around the basis of the gbuffer under them,
			// which is the constant resolved normal bound for the evaluation (basis 0); opaque
			// surfaces write RT1.w = basis / 3 and RT2.w = shading model. `other` is the evaluation
			// with the opposite checkerboard parity (nullptr when the specular colour is constant). Both are the
			// tile `tl`, whose texels keep their parity in the level.
			std::vector<surface> decode_gbuffer(const std::vector<texel_out>& same, const std::vector<texel_out>* other,
				const tile& tl, const bool decal)
			{
				std::vector<surface> out(same.size());
				parallel_for(tl.height, [&](const std::uint32_t ly, std::uint32_t)
				{
					const auto y = tl.y + ly;
					for (auto lx = 0u; lx < tl.width; lx++)
					{
						const auto x = tl.x + lx;
						const auto i = static_cast<std::size_t>(ly) * tl.width + lx;
						const auto& t = same[i];
						auto& s = out[i];
						s.discarded = t.discarded;

						const auto a0 = decal ? t.target[0][3] : 1.0f;
						const auto a1 = decal ? t.target[1][3] : 1.0f;
						const auto a2 = decal ? t.target[2][3] : 1.0f;
						s.coverage[0] = a0;
						s.coverage[1] = a1;
						s.coverage[2] = a2;
						s.shading_model = decal ? 1.0f / 3.0f : t.target[2][3];

						for (auto c = 0; c < 3; c++)
						{
							s.albedo[c] = unpremultiply(t.target[0][c], a0);
						}

						const float enc[4] = { unpremultiply(t.target[1][0], a1), unpremultiply(t.target[1][1], a1),
							unpremultiply(t.target[1][2], a1), decal ? 0.0f : t.target[1][3] };
						decode_normal(enc, s.normal);

						const auto z = enc[2];
						s.model_flag = z >= 0.5f;
						s.gloss = std::clamp((z - (s.model_flag ? 0.5f : 0.001466f)) * 2.009823f, 0.0f, 1.0f);

						// RT2: Y and one chroma; chroma R - B where x and y parity match
						const auto y_luma = unpremultiply(t.target[2][0], a2);
						const auto chroma_here = (unpremultiply(t.target[2][1], a2) - 0.5f) * 2.0f;
						float chroma_other = 0.0f;
						if (other)
						{
							const auto& o = (*other)[i];
							chroma_other = (unpremultiply(o.target[2][1], decal ? o.target[2][3] : 1.0f) - 0.5f) * 2.0f;
						}
						const auto same_parity = ((x & 1u) == (y & 1u));
						const auto rb = same_parity ? chroma_here : chroma_other; // R - B
						const auto g_minus_rb = same_parity ? chroma_other : chroma_here; // G - (R + B) / 2
						const auto sum_rb = 2.0f * y_luma - g_minus_rb; // R + B
						s.f0[1] = y_luma + g_minus_rb * 0.5f;
						s.f0[0] = (sum_rb + rb) * 0.5f;
						s.f0[2] = (sum_rb - rb) * 0.5f;
						s.occlusion = unpremultiply(t.target[2][2], a2);
					}
				});
				return out;
			}

			// The targets of a forward_surface program: the surface as BO3's forward shader computes it, without the
			// gbuffer's encoding (RT0 albedo + alpha, RT1 normal + gloss, RT2 F0 + occlusion). The alpha goes to
			// coverage[0]. The pack shader in gpu_eval.cpp does the same.
			std::vector<surface> decode_surface(const std::vector<texel_out>& same)
			{
				std::vector<surface> out(same.size());
				for (std::size_t i = 0; i < same.size(); i++)
				{
					const auto& t = same[i];
					auto& s = out[i];
					s.discarded = t.discarded;
					for (auto c = 0; c < 3; c++)
					{
						s.albedo[c] = t.target[0][c];
						s.normal[c] = t.target[1][c];
						s.f0[c] = t.target[2][c];
					}
					s.coverage[0] = t.target[0][3];
					s.coverage[1] = 1.0f;
					s.coverage[2] = 1.0f;
					const auto len = std::sqrt(s.normal[0] * s.normal[0] + s.normal[1] * s.normal[1] + s.normal[2] * s.normal[2]);
					if (len > 0.0f)
					{
						s.normal[0] /= len;
						s.normal[1] /= len;
						s.normal[2] /= len;
					}
					s.gloss = t.target[1][3];
					s.occlusion = t.target[2][3];
					s.shading_model = 1.0f / 3.0f;
					s.model_flag = false;
				}
				return out;
			}

			// ---- evaluation on the GPU --------------------------------------------------------------

			// pixel shaders whose GPU evaluation agreed with the interpreter (true) or cannot run there (false)
			std::mutex gpu_programs_mutex;
			std::unordered_map<const void*, bool> gpu_programs;

			// the largest difference, relative to the value, a GPU texel may show against the interpreter: the
			// hardware filters with 8-bit weights and decodes block-compressed texels with its own rounding (up to
			// 5/255 in a BC-compressed decal alpha); a wrong binding, format or constant differs far more
			constexpr float gpu_tolerance = 1.0f / 32.0f;

			// the program with what set_inputs feeds each input, for the generated vertex shader
			gpu_eval::program gpu_program_of(const Material* material, const material_program& mp,
				const std::unordered_map<std::string, std::array<float, 4>>& overrides)
			{
				gpu_eval::program gp{};
				if (mp.bytecode)
				{
					gp.bytecode = mp.bytecode->data();
					gp.bytecode_size = mp.bytecode->size();
				}
				else
				{
					gp.bytecode = mp.pixel_shader->prog.loadDef.program;
					gp.bytecode_size = mp.pixel_shader->prog.loadDef.programSize;
				}

				const auto& inputs = mp.program->inputs();
				if (inputs.size() != mp.inputs.size())
				{
					throw std::runtime_error("the program's inputs and their bindings differ");
				}
				for (auto i = 0u; i < inputs.size(); i++)
				{
					const auto& e = inputs[i];
					if (mp.inputs[i].reg != e.reg || mp.inputs[i].mask != e.mask)
					{
						throw std::runtime_error("the program's inputs and their bindings differ");
					}
					gpu_eval::input_element ge{};
					ge.semantic = e.semantic;
					ge.semantic_index = e.semantic_index;
					ge.system_value = e.system_value;
					ge.component_type = e.component_type;
					ge.reg = e.reg;
					ge.mask = e.mask;
					ge.interpolation = e.interpolation;
					const auto constant = [&ge](const float x, const float y, const float z, const float w)
					{
						ge.kind = gpu_eval::input_kind::constant;
						ge.value[0] = x;
						ge.value[1] = y;
						ge.value[2] = z;
						ge.value[3] = w;
					};
					switch (mp.inputs[i].what)
					{
					case input_binding::kind::position:
						ge.kind = gpu_eval::input_kind::position;
						break;
					case input_binding::kind::front_face:
						ge.kind = gpu_eval::input_kind::system;
						break;
					case input_binding::kind::uv:
						ge.kind = gpu_eval::input_kind::uv;
						break;
					case input_binding::kind::colour:
					{
						const auto colour = colour_input(mp, mp.inputs[i]);
						constant(colour[0], colour[1], colour[2], colour[3]);
						break;
					}
					case input_binding::kind::normal:
						constant(0.0f, 0.0f, 1.0f, 0.0f);
						break;
					case input_binding::kind::tangent:
						constant(1.0f, 0.0f, 0.0f, 0.0f);
						break;
					case input_binding::kind::bitangent:
						constant(0.0f, 1.0f, 0.0f, 0.0f);
						break;
					case input_binding::kind::view_offset:
						constant(0.0f, 0.0f, -100.0f, 0.0f);
						break;
					case input_binding::kind::uv_offsets:
					case input_binding::kind::zero:
						constant(0.0f, 0.0f, 0.0f, 0.0f);
						break;
					}
					gp.inputs.push_back(ge);
				}

				for (const auto& [slot, data] : mp.cbuffers)
				{
					gp.cbuffers.emplace_back(slot, &data);
				}
				for (const auto& t : mp.textures)
				{
					gpu_eval::texture gt{};
					gt.slot = t.slot;
					gt.image = t.image;
					const auto o = overrides.find(t.name);
					if (!t.image)
					{
						// a texture without an image: its constant, as bind_textures binds it
						gt.constant = true;
						std::memcpy(gt.value, t.constant.data(), sizeof(gt.value));
						gt.width = 1;
						gt.height = 1;
					}
					else if (o != overrides.end())
					{
						gt.constant = true;
						std::memcpy(gt.value, o->second.data(), sizeof(gt.value));
						gt.width = t.image->width;
						gt.height = t.image->height;
					}
					gp.textures.push_back(gt);
				}
				// samplers as the material sets them, by name
				for (const auto& b : mp.program->bindings())
				{
					if (b.input_type != sit_sampler)
					{
						continue;
					}
					gpu_eval::sampler sampler{};
					sampler.slot = b.bind_point;
					if (const auto* def = find_sampler(material, r_hash_string(b.name)))
					{
						address_modes(def->samplerState, sampler.u, sampler.v);
					}
					gp.samplers.push_back(sampler);
				}
				if (mp.resolved_normal_slot)
				{
					// bind_textures' resolved normal: a surface that takes decals, its normal encoded around basis 0
					gp.pixel_constant_slot = *mp.resolved_normal_slot;
					gp.pixel_constant[2] = 0.25f;
				}
				return gp;
			}

			// One program evaluated tile by tile: on the worker's GPU once the GPU and the interpreter agreed on the
			// program's first 8x8 texels, on the interpreter otherwise (and when the worker has no GPU).
			class evaluator
			{
			public:
				evaluator(worker& w, const Material* material, const material_program& mp,
					std::unordered_map<std::string, std::array<float, 4>> overrides)
					: worker_(w), mp_(mp), overrides_(std::move(overrides)),
					shader_(mp.bytecode ? static_cast<const void*>(mp.bytecode->data()) : mp.pixel_shader)
				{
					if (!w.gpu)
					{
						return;
					}
					if (!mp.buffers.empty())
					{
						// (the GPU evaluator binds no structured buffers)
						ZONETOOL_INFO("material %s: evaluated on the CPU (it reads BO3's per-instance buffers)", material->name);
						return;
					}
					{
						std::lock_guard _(gpu_programs_mutex);
						const auto found = gpu_programs.find(this->shader_);
						if (found != gpu_programs.end() && !found->second)
						{
							return;
						}
						this->checked_ = found != gpu_programs.end();
					}

					this->gpu_program_ = gpu_program_of(material, mp, this->overrides_);
					std::string why;
					auto shader_failed = false;
					if (w.gpu->bind(this->gpu_program_, why, shader_failed))
					{
						w.bound = this;
						this->gpu_ = true;
						return;
					}
					w.bound = nullptr;
					if (shader_failed)
					{
						std::lock_guard _(gpu_programs_mutex);
						if (gpu_programs.emplace(this->shader_, false).second)
						{
							ZONETOOL_INFO("material shader %s: evaluated on the CPU (%s)", this->name().data(), why.data());
						}
					}
					else
					{
						ZONETOOL_INFO("material %s: evaluated on the CPU (%s)", material->name, why.data());
					}
				}

				~evaluator()
				{
					if (this->worker_.bound == this)
					{
						this->worker_.bound = nullptr;
					}
				}

				evaluator(const evaluator&) = delete;
				evaluator& operator=(const evaluator&) = delete;

				// the program's targets over `t` (those in `targets`, the rest zero), tile-local row-major
				std::vector<texel_out> run(const grid& g, const tile& t, const std::uint32_t targets = 7)
				{
					if (this->gpu_)
					{
						auto out = this->run_gpu(g, t, targets);
						if (this->checked_ || this->agrees(g, t, out, t.width, targets))
						{
							return out;
						}
						this->gpu_ = false;
					}
					const auto& textures = this->cpu_textures();
					timed _(stage_ns.cpu);
					return evaluate(this->mp_, textures, g, t);
				}

				// Settles where the program runs before a level is drawn with draw(): an unchecked program's first tile
				// is drawn and its top-left 8x8 texels compared with the interpreter, as run() does. True: on the GPU.
				bool prepare(const grid& g, const tile& t)
				{
					if (!this->gpu_ || this->checked_)
					{
						return this->gpu_;
					}
					this->draw(g, t, false);
					const auto block_w = std::min(8u, t.width);
					const auto block_h = std::min(8u, t.height);
					std::vector<texel_out> block(static_cast<std::size_t>(block_w) * block_h);
					{
						timed _(stage_ns.gpu);
						for (auto rt = 0u; rt < 3; rt++)
						{
							if (this->mp_.target_reg[rt] < 0)
							{
								continue;
							}
							const auto values = this->worker_.gpu->read_target(rt, { 0, 0, block_w, block_h });
							for (std::size_t i = 0; i < block.size(); i++)
							{
								std::memcpy(block[i].target[rt], &values[i * 4], sizeof(block[i].target[rt]));
							}
						}
					}
					if (!this->agrees(g, t, block, block_w, 7))
					{
						this->gpu_ = false;
					}
					return this->gpu_;
				}

				// the program over `t` into the worker's GPU targets, and with `other_parity` its parity-shifted pass
				// (gpu_eval::context::draw); only after prepare() said the program runs on the GPU
				void draw(const grid& g, const tile& t, const bool other_parity)
				{
					timed _(stage_ns.gpu);
					this->bind();
					this->worker_.gpu->draw({ g.width, g.height, g.origin_u, g.origin_v, g.span_u, g.span_v }, { t.x, t.y, t.width, t.height },
						other_parity);
				}

			private:
				// another evaluator of the material (its forward emissive shader) may have run in between
				void bind()
				{
					if (this->worker_.bound == this)
					{
						return;
					}
					std::string why;
					auto shader_failed = false;
					if (!this->worker_.gpu->bind(this->gpu_program_, why, shader_failed))
					{
						throw std::runtime_error(utils::string::va("GPU evaluation: %s", why.data()));
					}
					this->worker_.bound = this;
				}

				std::string name() const
				{
					const auto* ps = this->mp_.pixel_shader;
					// (a cut of the shader: the surface of the debug-override permutation, draw method 1, or the alpha-test
					// coverage of draw method 0)
					return utils::string::va("%s (%s%s)", this->mp_.technique->name ? this->mp_.technique->name : "?",
						ps && ps->name ? ps->name : "?", !this->mp_.bytecode ? "" : (this->mp_.draw_method == 1 ? ", its surface" : ", its coverage"));
				}

				std::vector<texel_out> run_gpu(const grid& g, const tile& t, const std::uint32_t targets)
				{
					timed _(stage_ns.gpu);
					this->bind();

					std::uint32_t mask = 0;
					for (auto rt = 0u; rt < 3; rt++)
					{
						if ((targets & (1u << rt)) && this->mp_.target_reg[rt] >= 0)
						{
							mask |= 1u << rt;
						}
					}
					std::vector<float> planes[3];
					this->worker_.gpu->run({ g.width, g.height, g.origin_u, g.origin_v, g.span_u, g.span_v }, { t.x, t.y, t.width, t.height },
						g.parity_shift, mask, planes);

					std::vector<texel_out> out(static_cast<std::size_t>(t.width) * t.height);
					for (auto rt = 0u; rt < 3; rt++)
					{
						if (!(mask & (1u << rt)))
						{
							continue;
						}
						for (std::size_t i = 0; i < out.size(); i++)
						{
							std::memcpy(out[i].target[rt], &planes[rt][i * 4], sizeof(out[i].target[rt]));
						}
					}
					return out;
				}

				// the first tile of each program is checked once: its top-left 8x8 texels against the interpreter (`gpu`
				// holds the GPU's texels from the tile's corner, `stride` a row)
				bool agrees(const grid& g, const tile& t, const std::vector<texel_out>& gpu, const std::uint32_t stride, const std::uint32_t targets)
				{
					{
						std::lock_guard _(gpu_programs_mutex);
						const auto found = gpu_programs.find(this->shader_);
						if (found != gpu_programs.end())
						{
							this->checked_ = true;
							return found->second;
						}
					}

					const auto& textures = this->cpu_textures();
					const tile block{ t.x, t.y, std::min(8u, t.width), std::min(8u, t.height) };
					std::vector<texel_out> cpu;
					{
						timed _(stage_ns.check);
						cpu = evaluate(this->mp_, textures, g, block);
					}

					auto worst = 0.0f;
					std::string where;
					for (auto y = 0u; y < block.height; y++)
					{
						for (auto x = 0u; x < block.width; x++)
						{
							const auto& c = cpu[static_cast<std::size_t>(y) * block.width + x];
							const auto& o = gpu[static_cast<std::size_t>(y) * stride + x];
							for (auto rt = 0u; rt < 3; rt++)
							{
								if (!(targets & (1u << rt)) || this->mp_.target_reg[rt] < 0)
								{
									continue;
								}
								for (auto k = 0u; k < 4; k++)
								{
									const auto a = c.target[rt][k];
									const auto b = o.target[rt][k];
									if (a == b || (std::isnan(a) && std::isnan(b)))
									{
										continue;
									}
									auto diff = std::fabs(a - b) / std::max({ 1.0f, std::fabs(a), std::fabs(b) });
									if (std::isnan(diff))
									{
										diff = std::numeric_limits<float>::infinity();
									}
									if (diff > worst)
									{
										worst = diff;
										where = utils::string::va("texel %u,%u target %u.%c: interpreter %g, GPU %g", t.x + x, t.y + y, rt,
											"xyzw"[k], a, b);
									}
								}
							}
						}
					}

					const auto agree = worst <= gpu_tolerance;
					{
						std::lock_guard _(gpu_programs_mutex);
						gpu_programs.emplace(this->shader_, agree);
					}
					ZONETOOL_INFO("material shader %s: the GPU %s the interpreter (largest difference %.6f%s%s)", this->name().data(),
						agree ? "agrees with" : "differs from", worst, where.empty() ? "" : " at ", where.data());
					this->checked_ = true;
					return agree;
				}

				const bound_textures& cpu_textures()
				{
					if (!this->cpu_)
					{
						timed _(stage_ns.textures);
						this->cpu_ = std::make_unique<bound_textures>(bind_textures(this->mp_, this->overrides_));
					}
					return *this->cpu_;
				}

				worker& worker_;
				const material_program& mp_;
				std::unordered_map<std::string, std::array<float, 4>> overrides_;
				const void* shader_;
				gpu_eval::program gpu_program_{};
				bool gpu_ = false;
				bool checked_ = false;
				std::unique_ptr<bound_textures> cpu_;
			};

			// ---- IW7 packing ------------------------------------------------------------------------

			std::uint8_t to_byte(const float v)
			{
				return static_cast<std::uint8_t>(std::clamp(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f), 0L, 255L));
			}

			void octahedral(const float n[3], float& ga_g, float& ga_a)
			{
				auto nz = n[2];
				if (nz < 0.0f)
				{
					nz = 0.0f; // IW7 decodes the upper hemisphere only
				}
				const auto sum = std::fabs(n[0]) + std::fabs(n[1]) + nz;
				const auto px = sum > 0.0f ? n[0] / sum : 0.0f;
				const auto py = sum > 0.0f ? n[1] / sum : 0.0f;
				ga_g = (px + py) * 0.5f + 0.5f;
				ga_a = (px - py) * 0.5f + 0.5f;
			}

			float iw7_gloss(const float bo3_gloss)
			{
				const auto t = 1.0f - 17.0f * bo3_gloss / 20.0f;
				return 1.0f - t * t;
			}

			// the reflectance bytes the metal fit tries: 26 (0.1) .. 255
			constexpr auto metal_first_byte = 26;
			constexpr auto metal_candidates = 256 - metal_first_byte;

			// one candidate's quantised least-squares error and colour bytes (diffuse = C (1 - m), F0 = 0.1 + m C)
			float metal_error(const float albedo[3], const float f0[3], const int a_byte, std::uint8_t candidate[4])
			{
				const auto a = a_byte / 255.0f;
				const auto m = std::clamp((a - 0.1f) / 0.9f, 0.0f, 1.0f);
				float error = 0.0f;
				for (auto c = 0; c < 3; c++)
				{
					const auto denom = (1.0f - m) * (1.0f - m) + m * m;
					const auto colour = std::clamp(((1.0f - m) * albedo[c] + m * (f0[c] - 0.1f)) / denom, 0.0f, 1.0f);
					const auto q = to_byte(std::sqrt(colour));
					const auto cq = (q / 255.0f) * (q / 255.0f);
					const auto ed = cq * (1.0f - m) - albedo[c];
					const auto ef = 0.1f + m * cq - f0[c];
					error += ed * ed + ef * ef;
					candidate[c] = q;
				}
				candidate[3] = static_cast<std::uint8_t>(a_byte);
				return error;
			}

			// IW7 reflectance/metalness for a BO3 (diffuse albedo, F0) pair; returns the cs texel
			void pack_colour(const float albedo[3], const float f0[3], std::uint8_t out[4])
			{
				const auto grey = std::fabs(f0[0] - f0[1]) < 1e-3f && std::fabs(f0[1] - f0[2]) < 1e-3f;
				if (grey && f0[1] <= 0.1f)
				{
					for (auto c = 0; c < 3; c++)
					{
						out[c] = to_byte(std::sqrt(std::max(albedo[c], 0.0f)));
					}
					out[3] = to_byte(f0[1]);
					return;
				}

				// metal: diffuse = C (1 - m), F0 = 0.1 + m C, a = 0.1 + 0.9 m; least squares over the eight-bit reflectance
				// values above 0.1, the lowest byte winning a tie. A candidate's error is never below the unquantised
				// minimum over C of (C (1 - m) - A)^2 + (0.1 + m C - F)^2, (m A - (1 - m)(F - 0.1))^2 / ((1 - m)^2 + m^2)
				// per channel, so only the candidates whose minimum reaches the best error found need the quantised
				// fit; the result is the full search's (the margin covers float rounding in either sum).
				float bound[metal_candidates];
				auto first = 0;
				for (auto k = 0; k < metal_candidates; k++)
				{
					const auto a = (metal_first_byte + k) / 255.0f;
					const auto m = std::clamp((a - 0.1f) / 0.9f, 0.0f, 1.0f);
					const auto denom = (1.0f - m) * (1.0f - m) + m * m;
					auto b = 0.0f;
					for (auto c = 0; c < 3; c++)
					{
						const auto cross = m * albedo[c] - (1.0f - m) * (f0[c] - 0.1f);
						b += cross * cross / denom;
					}
					bound[k] = b;
					if (b < bound[first])
					{
						first = k;
					}
				}

				std::uint8_t best[4]{};
				auto best_error = metal_error(albedo, f0, metal_first_byte + first, best);
				auto best_k = first;
				for (auto k = 0; k < metal_candidates; k++)
				{
					if (k == first || bound[k] > best_error * 1.0001f + 1e-12f)
					{
						continue;
					}
					std::uint8_t candidate[4];
					const auto error = metal_error(albedo, f0, metal_first_byte + k, candidate);
					if (error < best_error || (error == best_error && k < best_k))
					{
						best_error = error;
						best_k = k;
						std::memcpy(best, candidate, 4);
					}
				}
				// also compare with the best dielectric
				{
					float error = 0.0f;
					std::uint8_t candidate[4]{};
					const auto f = std::min((f0[0] + f0[1] + f0[2]) / 3.0f, 0.1f);
					for (auto c = 0; c < 3; c++)
					{
						candidate[c] = to_byte(std::sqrt(std::max(albedo[c], 0.0f)));
						const auto ef = f - f0[c];
						error += ef * ef;
					}
					candidate[3] = to_byte(f);
					if (error <= best_error)
					{
						std::memcpy(best, candidate, 4);
					}
				}
				std::memcpy(out, best, 4);
			}

			// pull-push fill of texels a decal does not cover, so filtering at its edges does not
			// blend in undefined colours: each pass gives every uncovered texel next to a known one (the 8 wrapped
			// neighbours) the average of those known at the start of the pass, for up to 64 passes; what is left
			// gets `neutral`. A pass only visits the texels next to the previous pass's.
			void fill_uncovered(std::vector<std::uint8_t>& rgba, const std::vector<bool>& valid, std::uint32_t w, std::uint32_t h,
				const std::uint8_t neutral[4])
			{
				std::vector<std::uint8_t> known(valid.begin(), valid.end());
				const auto neighbours = [&](const std::size_t i, auto&& fn)
				{
					const auto x = static_cast<std::int32_t>(i % w);
					const auto y = static_cast<std::int32_t>(i / w);
					for (auto oy = -1; oy <= 1; oy++)
					{
						for (auto ox = -1; ox <= 1; ox++)
						{
							const auto nx = (x + ox + static_cast<std::int32_t>(w)) % static_cast<std::int32_t>(w);
							const auto ny = (y + oy + static_cast<std::int32_t>(h)) % static_cast<std::int32_t>(h);
							fn(static_cast<std::size_t>(ny) * w + nx);
						}
					}
				};

				// the uncovered texels next to a covered one
				std::vector<std::uint8_t> queued(known.size(), 0);
				std::vector<std::uint32_t> frontier;
				for (std::size_t i = 0; i < known.size(); i++)
				{
					if (known[i])
					{
						continue;
					}
					auto next_to_known = false;
					neighbours(i, [&](const std::size_t j)
					{
						next_to_known |= known[j] != 0;
					});
					if (next_to_known)
					{
						queued[i] = 1;
						frontier.push_back(static_cast<std::uint32_t>(i));
					}
				}

				for (auto pass = 0; pass < 64 && !frontier.empty(); pass++)
				{
					std::vector<std::array<std::uint8_t, 4>> values(frontier.size());
					parallel_for(static_cast<std::uint32_t>(frontier.size()), [&](const std::uint32_t f, std::uint32_t)
					{
						std::uint32_t sum[4]{}, n = 0;
						neighbours(frontier[f], [&](const std::size_t j)
						{
							if (known[j])
							{
								for (auto c = 0; c < 4; c++)
								{
									sum[c] += rgba[j * 4 + c];
								}
								n++;
							}
						});
						for (auto c = 0; c < 4; c++)
						{
							values[f][c] = static_cast<std::uint8_t>(sum[c] / n);
						}
					});

					for (auto f = 0u; f < frontier.size(); f++)
					{
						std::memcpy(&rgba[static_cast<std::size_t>(frontier[f]) * 4], values[f].data(), 4);
						known[frontier[f]] = 1;
					}
					std::vector<std::uint32_t> next;
					for (const auto i : frontier)
					{
						neighbours(i, [&](const std::size_t j)
						{
							if (!known[j] && !queued[j])
							{
								queued[j] = 1;
								next.push_back(static_cast<std::uint32_t>(j));
							}
						});
					}
					frontier.swap(next);
				}
				for (auto i = 0u; i < known.size(); i++)
				{
					if (!known[i])
					{
						std::memcpy(&rgba[i * 4], neutral, 4);
					}
				}
			}

			// ---- block compression and IW7 image records --------------------------------------------

			// one level on the CPU: strips of 64 rows compressed in parallel (DirectXTex's own parallel path needs OpenMP)
			std::vector<std::uint8_t> compress_level(const std::vector<std::uint8_t>& rgba, const std::uint32_t w, const std::uint32_t h,
				const DXGI_FORMAT format)
			{
				const auto strip_rows = 64u;
				const auto strips = (h + strip_rows - 1) / strip_rows;
				std::vector<std::vector<std::uint8_t>> parts(strips);

				parallel_for(strips, [&](const std::uint32_t s, std::uint32_t)
				{
					const auto y0 = s * strip_rows;
					const auto rows = std::min(strip_rows, h - y0);
					DirectX::Image src{};
					src.width = w;
					src.height = rows;
					src.format = DXGI_FORMAT_R8G8B8A8_UNORM;
					src.rowPitch = static_cast<std::size_t>(w) * 4;
					src.slicePitch = src.rowPitch * rows;
					src.pixels = const_cast<std::uint8_t*>(rgba.data()) + static_cast<std::size_t>(y0) * w * 4;

					DirectX::ScratchImage compressed;
					const auto hr = DirectX::Compress(src, format, DirectX::TEX_COMPRESS_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, compressed);
					if (FAILED(hr))
					{
						throw std::runtime_error(utils::string::va("block compression failed (0x%08X)", static_cast<unsigned int>(hr)));
					}
					parts[s].assign(compressed.GetPixels(), compressed.GetPixels() + compressed.GetPixelsSize());
				});

				std::vector<std::uint8_t> out;
				for (const auto& p : parts)
				{
					out.insert(out.end(), p.begin(), p.end());
				}
				return out;
			}

			struct image_levels
			{
				std::uint32_t width = 0;
				std::uint32_t height = 0;
				std::vector<std::vector<std::uint8_t>> rgba; // per level, 8-bit RGBA
			};

			// BC7 goes to DirectXTex's DirectCompute encoder shaders on the worker's GPU when it has one, the whole mip
			// chain at once
			std::vector<std::vector<std::uint8_t>> compress_levels(const image_levels& levels, const DXGI_FORMAT format,
				gpu_eval::context* gpu)
			{
				if (format == DXGI_FORMAT_BC7_UNORM && gpu && !levels.rgba.empty())
				{
					return gpu->compress_bc7(levels.width, levels.height, levels.rgba);
				}
				std::vector<std::vector<std::uint8_t>> out(levels.rgba.size());
				for (auto l = 0u; l < levels.rgba.size(); l++)
				{
					out[l] = compress_level(levels.rgba[l], std::max(1u, levels.width >> l), std::max(1u, levels.height >> l), format);
				}
				return out;
			}

			// Stock IW7 streams an image as up to four parts, each a whole mip chain from its own top level down: the
			// levels 0-3 of the full chain that are at least 128 texels (stock: 256 -> 128/256, 512 ->
			// 128..512, 1024 -> 128..1024, 2048 -> 256..2048, 4096 -> 512..4096), smallest part first; the zone keeps a
			// 1x1 placeholder with no pixels and flag 0x40. Part data goes to the zone's own pak: file index 431, which
			// IW7 opens while the zone loads (0xA7DB10 -> 0xCFDA30) and iw7-mod names "<loading zone>.pak"
			// (iw7-mod.exe 0x16008C1B8). A part's pixelSize is 26 bits, so the top part stays under 64 MB.
			constexpr std::uint32_t stream_min_size = 128;
			constexpr std::size_t stream_max_part = (1u << 26) - 1;

			void write_streamed_image(const std::string& name, const std::uint32_t width, const std::uint32_t height, const DXGI_FORMAT format,
				const zonetool::iw7::TextureSemantic semantic, const std::uint32_t flags,
				const std::vector<std::vector<std::uint8_t>>& compressed)
			{
				zonetool::iw7::GfxImage image{};
				image.imageFormat = format;
				image.flags = flags | zonetool::iw7::IMG_DISK_FLAG_STREAMED;
				image.mapType = zonetool::iw7::MAPTYPE_2D;
				image.semantic = semantic;
				image.category = zonetool::iw7::IMG_CATEGORY_LOAD_FROM_FILE;
				image.picmip.platform[0] = 0;
				image.picmip.platform[1] = 2;
				image.width = 1;
				image.height = 1;
				image.depth = 1;
				image.numElements = 1;
				image.levelCount = 1;
				image.streamed = 1;
				image.name = name.data();

				std::vector<unsigned int> part_levels;
				for (auto l = 3; l >= 0; l--)
				{
					// a part's top level is created as a texture of its own (Create2DTexture), which a block-compressed
					// format takes only in whole blocks: the size exactly halved l times, a multiple of 4
					const auto whole_blocks = width % (4u << l) == 0 && height % (4u << l) == 0;
					if (static_cast<std::size_t>(l) < compressed.size() && whole_blocks
						&& std::max(width >> l, height >> l) >= stream_min_size)
					{
						part_levels.push_back(static_cast<unsigned int>(l));
					}
				}

				// A part's pixelSize is its whole chain, but its pak data is only the levels the part before lacks: the
				// stream reader (0x140A7F370) reads pixelSize[k] - pixelSize[k - 1] bytes of part k and raises "Disc read
				// error [8.n]" when the part's pak span holds n bytes more than it read
				for (auto part = 0u; part < part_levels.size(); part++)
				{
					const auto first = part_levels[part];
					const auto end = part ? part_levels[part - 1] : static_cast<unsigned int>(compressed.size());
					std::size_t chain = 0;
					for (auto l = first; l < compressed.size(); l++)
					{
						chain += compressed[l].size();
					}
					std::string data;
					for (auto l = first; l < end; l++)
					{
						data.append(reinterpret_cast<const char*>(compressed[l].data()), compressed[l].size());
					}
					if (chain > stream_max_part)
					{
						throw std::runtime_error(utils::string::va("image %s: streamed part of %zu bytes", name.data(), chain));
					}
					auto& stream = image.streams[part];
					stream.width = static_cast<unsigned short>(std::max(1u, width >> first));
					stream.height = static_cast<unsigned short>(std::max(1u, height >> first));
					stream.levelCountAndSize.levelCount = static_cast<unsigned int>(compressed.size() - first);
					stream.levelCountAndSize.pixelSize = static_cast<unsigned int>(chain);
					image.dataLen2 = static_cast<unsigned int>(chain);

					filesystem::file file(utils::string::va("streamed_images\\%s_stream%u.pixels", name.data(), part));
					file.open("wb");
					if (!file.get_fp())
					{
						throw std::runtime_error(utils::string::va("could not write streamed part %u of %s", part, name.data()));
					}
					file.write(data.data(), data.size(), 1);
					file.close();
				}

				assetmanager::dumper write;
				if (!write.open("streamed_images\\" + name + ".iw7Image"))
				{
					throw std::runtime_error(utils::string::va("could not write the streamed image record of %s", name.data()));
				}
				write.dump_single(&image);
				write.dump_string(image.name);
				write.close();
			}

			// Images written in this dump and the material each is for: a name two materials would write is a naming
			// collision, except for images whose name is their content (detail normals), written once.
			std::mutex written_mutex;
			std::unordered_map<std::string, std::string> written_images;

			// false: the (shared) image is written already
			bool claim_image(const std::string& image, const std::string& material, const bool shared)
			{
				std::lock_guard _(written_mutex);
				const auto [at, added] = written_images.emplace(image, material);
				if (added)
				{
					return true;
				}
				if (!shared)
				{
					throw std::runtime_error(utils::string::va("image %s is also written for material %s: the names collide",
						image.data(), at->second.data()));
				}
				return false;
			}

			// writes an image's block-compressed levels (level l max(1, width >> l) x max(1, height >> l)): streamed parts
			// from stream_min_size up, a zone image below; returns their bytes
			std::size_t write_levels(const std::string& name, const std::uint32_t width, const std::uint32_t height, const DXGI_FORMAT format,
				const zonetool::iw7::TextureSemantic semantic, const std::uint32_t flags, const std::vector<std::vector<std::uint8_t>>& compressed)
			{
				std::size_t total = 0;
				for (const auto& level : compressed)
				{
					total += level.size();
				}

				timed _(stage_ns.write);
				if (std::max(width, height) >= stream_min_size)
				{
					write_streamed_image(name, width, height, format, semantic, flags, compressed);
					return total;
				}

				std::vector<std::uint8_t> pixels;
				for (const auto& level : compressed)
				{
					pixels.insert(pixels.end(), level.begin(), level.end());
				}

				zonetool::iw7::GfxImage image{};
				image.imageFormat = format;
				image.flags = flags | (compressed.size() <= 1 ? zonetool::iw7::IMG_DISK_FLAG_NOMIPMAPS : 0);
				image.mapType = zonetool::iw7::MAPTYPE_2D;
				image.semantic = semantic;
				image.category = zonetool::iw7::IMG_CATEGORY_LOAD_FROM_FILE;
				image.picmip.platform[0] = 0;
				image.picmip.platform[1] = 2;
				image.dataLen1 = static_cast<unsigned int>(pixels.size());
				image.dataLen2 = image.dataLen1;
				image.width = static_cast<unsigned short>(width);
				image.height = static_cast<unsigned short>(height);
				image.depth = 1;
				image.numElements = 1;
				image.levelCount = static_cast<unsigned char>(compressed.size());
				image.streamed = 0;
				image.pixelData = pixels.data();
				image.name = name.data();
				zonetool::iw7::gfx_image::dump(&image);
				return pixels.size();
			}

			std::size_t write_image(const std::string& name, const image_levels& levels, const DXGI_FORMAT format,
				const zonetool::iw7::TextureSemantic semantic, const std::uint32_t flags, gpu_eval::context* gpu)
			{
				std::vector<std::vector<std::uint8_t>> compressed;
				{
					timed _(stage_ns.compress);
					compressed = compress_levels(levels, format, gpu);
				}
				return write_levels(name, levels.width, levels.height, format, semantic, flags, compressed);
			}

			std::uint32_t mip_count(std::uint32_t w, std::uint32_t h)
			{
				auto count = 1u;
				while (w > 1 || h > 1)
				{
					w = std::max(1u, w / 2);
					h = std::max(1u, h / 2);
					count++;
				}
				return count;
			}

			// ---- model material atlases -------------------------------------------------------------

			// an atlas's tiles as their materials finish baking: each tile's BC7 levels, cs and ng
			struct atlas_pending
			{
				atlas_slot layout; // (column, row unused)
				std::uint32_t expected = 0;
				std::uint32_t received = 0;
				bool written = false;
				std::vector<std::vector<std::vector<std::uint8_t>>> cs; // by tile (row x columns + column), then level
				std::vector<std::vector<std::vector<std::uint8_t>>> ng;
			};

			std::mutex atlas_mutex;
			std::unordered_map<const Material*, atlas_slot> atlas_members;
			std::map<std::string, atlas_pending> atlases; // by cs name

			// the atlas's levels from its tiles' levels: block rows copied while every tile's level is whole BC7 blocks
			// at its place (the tile size divides by 2^level and the level by 4), the levels below that halved from the
			// level above (a box filter; 16 texels a tile and less)
			std::vector<std::vector<std::uint8_t>> compose_atlas(const atlas_pending& a, const std::vector<std::vector<std::vector<std::uint8_t>>>& tiles)
			{
				const auto& l = a.layout;
				const auto width = l.columns * l.tile_width;
				const auto height = l.rows * l.tile_height;
				const auto levels = mip_count(width, height);
				std::vector<std::vector<std::uint8_t>> out;
				std::vector<std::uint8_t> rgba; // the level above, once levels are halved
				for (auto level = 0u; level < levels; level++)
				{
					const auto atlas_w = std::max(1u, width >> level);
					const auto atlas_h = std::max(1u, height >> level);
					const auto tw = l.tile_width >> level;
					const auto th = l.tile_height >> level;
					const auto whole = rgba.empty() && (l.tile_width % (1u << level)) == 0 && (l.tile_height % (1u << level)) == 0
						&& tw >= 4 && th >= 4 && tw % 4 == 0 && th % 4 == 0;
					if (whole)
					{
						const auto blocks_across = atlas_w / 4;
						std::vector<std::uint8_t> blocks(static_cast<std::size_t>(blocks_across) * (atlas_h / 4) * 16, 0);
						for (auto t = 0u; t < tiles.size(); t++)
						{
							if (tiles[t].size() <= level)
							{
								continue; // a tile its material did not bake
							}
							const auto& src = tiles[t][level];
							const auto row_bytes = static_cast<std::size_t>(tw / 4) * 16;
							if (src.size() != row_bytes * (th / 4))
							{
								throw std::runtime_error(utils::string::va("atlas %s: tile %u level %u holds %zu bytes, not %zu", l.cs.data(), t,
									level, src.size(), row_bytes * (th / 4)));
							}
							const auto column = t % l.columns;
							const auto row = t / l.columns;
							for (auto by = 0u; by < th / 4; by++)
							{
								const auto at = (static_cast<std::size_t>(row * (th / 4) + by) * blocks_across + column * (tw / 4)) * 16;
								std::memcpy(&blocks[at], &src[by * row_bytes], row_bytes);
							}
						}
						out.emplace_back(std::move(blocks));
						continue;
					}

					if (rgba.empty())
					{
						// the level above, decoded
						const auto pw = std::max(1u, width >> (level - 1));
						const auto ph = std::max(1u, height >> (level - 1));
						DirectX::Image src{};
						src.width = pw;
						src.height = ph;
						src.format = DXGI_FORMAT_BC7_UNORM;
						src.rowPitch = static_cast<std::size_t>((pw + 3) / 4) * 16;
						src.slicePitch = src.rowPitch * ((ph + 3) / 4);
						src.pixels = out.back().data();
						DirectX::ScratchImage decoded;
						const auto hr = DirectX::Decompress(src, DXGI_FORMAT_R8G8B8A8_UNORM, decoded);
						if (FAILED(hr))
						{
							throw std::runtime_error(utils::string::va("atlas %s: decoding level %u failed (0x%08X)", l.cs.data(), level - 1,
								static_cast<unsigned int>(hr)));
						}
						rgba.assign(decoded.GetPixels(), decoded.GetPixels() + static_cast<std::size_t>(pw) * ph * 4);
					}
					const auto pw = std::max(1u, width >> (level - 1));
					const auto ph = std::max(1u, height >> (level - 1));
					std::vector<std::uint8_t> half(static_cast<std::size_t>(atlas_w) * atlas_h * 4);
					for (auto y = 0u; y < atlas_h; y++)
					{
						for (auto x = 0u; x < atlas_w; x++)
						{
							for (auto c = 0u; c < 4; c++)
							{
								auto sum = 0u;
								for (auto ky = 0u; ky < 2; ky++)
								{
									for (auto kx = 0u; kx < 2; kx++)
									{
										const auto sx = std::min(x * 2 + kx, pw - 1);
										const auto sy = std::min(y * 2 + ky, ph - 1);
										sum += rgba[(static_cast<std::size_t>(sy) * pw + sx) * 4 + c];
									}
								}
								half[(static_cast<std::size_t>(y) * atlas_w + x) * 4 + c] = static_cast<std::uint8_t>((sum + 2) / 4);
							}
						}
					}
					rgba = std::move(half);
					out.emplace_back(compress_level(rgba, atlas_w, atlas_h, DXGI_FORMAT_BC7_UNORM));
				}
				return out;
			}

			// returns the bytes written
			std::size_t write_atlas(const atlas_pending& a)
			{
				const auto& l = a.layout;
				const auto width = l.columns * l.tile_width;
				const auto height = l.rows * l.tile_height;
				claim_image(l.cs, l.cs, false);
				claim_image(l.ng, l.cs, false);
				return write_levels(l.cs, width, height, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_COLOR_SPECULAR_MAP, 0x2300, compose_atlas(a, a.cs))
					+ write_levels(l.ng, width, height, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_NORMAL_OCCLUSSION_GLOSS_MAP, 0x308, compose_atlas(a, a.ng));
			}

			// takes an atlas out of the table to write it (under atlas_mutex)
			atlas_pending take_atlas(atlas_pending& a)
			{
				atlas_pending done;
				done.layout = a.layout;
				done.cs = std::move(a.cs);
				done.ng = std::move(a.ng);
				a.cs.clear();
				a.ng.clear();
				a.written = true;
				return done;
			}

			// a material's tile, block-compressed; the atlas is written once it has every tile. Returns the bytes written
			std::size_t deposit_tile(const atlas_slot& slot, std::vector<std::vector<std::uint8_t>> cs_tile, std::vector<std::vector<std::uint8_t>> ng_tile)
			{
				atlas_pending done;
				{
					std::lock_guard _(atlas_mutex);
					auto& a = atlases.at(slot.cs);
					const auto t = slot.row * slot.columns + slot.column;
					a.cs[t] = std::move(cs_tile);
					a.ng[t] = std::move(ng_tile);
					if (++a.received < a.expected)
					{
						return 0;
					}
					done = take_atlas(a);
				}
				return write_atlas(done);
			}

			// ---- planning ---------------------------------------------------------------------------

			std::mutex plan_mutex;
			// by material and whether it is planned for models (their techset family and bake area differ)
			std::map<std::pair<const Material*, bool>, plan> plans;

			// largest denominator accepted when turning a texture's tiling rate into a fraction
			constexpr std::uint32_t max_denominator = 64;

			// texels (at the bake's density) around the area a model material's surfaces use, when only that area is baked
			constexpr double window_margin = 16.0;

			bool to_fraction(const float value, std::uint32_t& num, std::uint32_t& den)
			{
				for (auto d = 1u; d <= max_denominator; d++)
				{
					const auto n = std::lround(value * static_cast<float>(d));
					if (n > 0 && std::fabs(static_cast<float>(n) / static_cast<float>(d) - value) < 1e-4f * std::max(1.0f, value))
					{
						num = static_cast<std::uint32_t>(n);
						den = d;
						return true;
					}
				}
				return false;
			}

			// The gbuffer technique set a forward _emissive one adds its emissive layer to: the same name without
			// "_emissive" (wc/lit_emissive -> wc/lit: their $Globals share colorTint, baseNormalHeight and glossRange
			// at the same offsets), or for the animated variants also without their animation tokens
			// (wc/lit_emissive_scroll_2layer_plus -> wc/lit_plus). The first one loaded, not a placeholder and with a
			// gbuffer technique; load_program then requires every constant it reads to exist in the material by name.
			const MaterialTechniqueSet* sibling_techset(const Material* material)
			{
				const std::string name = material->techniqueSet->name;
				const auto hash_at = name.find('#');
				auto base = name.substr(0, hash_at);
				const auto suffix = hash_at == std::string::npos ? std::string{} : name.substr(hash_at);
				const auto erase_token = [](std::string& s, const char* token)
				{
					const auto at = s.find(token);
					if (at == std::string::npos)
					{
						return false;
					}
					s.erase(at, std::strlen(token));
					return true;
				};
				if (!erase_token(base, "_emissive"))
				{
					throw std::runtime_error("forward-lit material whose technique set has no _emissive sibling");
				}

				std::vector<std::string> candidates = { base };
				auto still = base;
				auto animated = false;
				for (const auto* token : { "_scroll", "_2layer", "_3layer", "_script" })
				{
					animated |= erase_token(still, token);
				}
				if (animated)
				{
					candidates.push_back(still);
				}

				for (const auto& candidate : candidates)
				{
					const auto sibling = candidate + suffix;
					const auto* entry = zonetool::t7::DB_FindXAssetEntry(ASSET_TYPE_TECHNIQUE_SET, sibling.data(), false);
					if (entry && !entry->placeholder && entry->asset.header.techniqueSet
						&& has_technique(entry->asset.header.techniqueSet, "gbuffer"))
					{
						return entry->asset.header.techniqueSet;
					}
				}
				throw std::runtime_error(utils::string::va("no loaded gbuffer technique set %s%s", candidates.back().data(), suffix.data()));
			}

			// the texture and channel that carries alpha-test coverage (what reaches the discard) or decal coverage (the
			// albedo target's alpha), the reveal map aside
			std::pair<const texture_binding*, std::uint32_t> coverage_source(const material_program& mp, const bool alpha_test, const bool reveal)
			{
				const auto& set = alpha_test ? mp.taint.discard_textures : mp.taint.output_textures[mp.target_reg[0] >= 0 ? mp.target_reg[0] : 0][3];
				const texture_binding* found = nullptr;
				std::uint32_t channel = 0;
				auto count = 0;
				for (auto bit = 0u; bit < set.size(); bit++)
				{
					if (!set.test(bit))
					{
						continue;
					}
					const auto* t = binding_by_slot(mp, bit / 4);
					if (!t || (reveal && t->name == "revealMap"))
					{
						continue;
					}
					if (!t->image)
					{
						throw std::runtime_error(utils::string::va("coverage depends on engine texture \"%s\"", t->name.data()));
					}
					found = t;
					channel = bit % 4;
					count++;
				}
				if (count > 1)
				{
					throw std::runtime_error("coverage depends on more than one texture channel");
				}
				return { found, channel };
			}

			// Whether a decal program takes its vertex alpha as IW7's vertex colour decal techsets (mco_) take COLOR0.a,
			// measured on the probe quad with every texture at a constant (0.5, alpha 1; the reveal map at r) and the alpha
			// fed where the vertex shader passes it raw, every component the decal writes (its blend write masks) against
			//  * without reveal, its value at alpha 1 times the alpha (IW7: coverage x COLOR0.a, premultiplied);
			//  * with reveal, its value fully revealed times BO3's reveal of r at alpha a, sat((r - e0) / (e1 - e0)) with
			//    a' = sat(0.998 a + 0.001), e0 = sat(1 - a' - s a'^k), e1 = sat(1 - a' + s (1 - a')^k), s = alphaRevealSoftEdge,
			//    k = sat(alphaRevealRamp): IW7's v0 reveal has k = 1/2 (write_material fits s to it).
			// A forward surface's program (surface): only its alpha (target 0 w) is scaled so, the rest stays as it is.
			// Empty when it does, else what differs.
			std::string vertex_alpha_mismatch(const material_program& mp, const bool reveal, const bool surface = false)
			{
				const auto sources = colour_sources(mp);
				const auto* state = mp.technique ? mp.technique->stateMap : nullptr;
				if (!sources || !state)
				{
					return "its vertex shader or its blend state cannot be read";
				}
				const auto evaluate = [&](const float alpha, const float reveal_value)
				{
					auto fed = mp;
					for (auto index = 0u; index < 2; index++)
					{
						for (auto k = 0u; k < (*sources)[index].size() && k < 4; k++)
						{
							const auto& s = (*sources)[index][k];
							if (s.what == shader_eval::colour_source::kind::channel && s.channel == 3)
							{
								fed.colour[index][k] = alpha;
							}
						}
					}
					std::unordered_map<std::string, std::array<float, 4>> constants;
					for (const auto& t : fed.textures)
					{
						constants[t.name] = { 0.5f, 0.5f, 0.5f, 1.0f };
					}
					if (reveal)
					{
						constants["revealMap"] = { reveal_value, reveal_value, reveal_value, reveal_value };
					}
					auto bound = bind_textures(fed, constants);
					shader_eval::machine m(fed.program);
					m.set_unbound_resources_zero(fed.forward);
					for (const auto& [slot, data] : fed.cbuffers)
					{
						m.bind_cbuffer(slot, data.data(), data.size());
					}
					for (const auto& [slot, tex] : bound.slots)
					{
						m.bind_texture(slot, tex);
					}
					for (const auto& b : fed.buffers)
					{
						m.bind_structured(b.slot, b.data.data(), b.data.size(), b.stride);
					}
					set_inputs(m, fed, grid{ 64, 64, 0.0f, 0.0f, 1.0f, 1.0f, 0 }, 0, 0);
					m.run();
					// the components the vertex alpha scales, and (a surface's) those it leaves
					std::pair<std::vector<float>, std::vector<float>> out;
					for (auto rt = 0; rt < 3; rt++)
					{
						const auto& b = state->primitive.mrtBlending ? state->blend[rt] : state->blend[0];
						for (auto c = 0u; c < 4 && fed.target_reg[rt] >= 0; c++)
						{
							if (!surface && !(b.writeMask & (1u << c)))
							{
								continue;
							}
							auto& into = !surface || (rt == 0 && c == 3) ? out.first : out.second;
							for (auto lane = 0u; lane < shader_eval::lane_count; lane++)
							{
								// a discarded texel draws nothing: 0 in every target
								into.push_back(((m.discarded() >> lane) & 1) ? 0.0f : m.output(static_cast<std::uint32_t>(fed.target_reg[rt]), c, lane));
							}
						}
					}
					return out;
				};
				const auto scaled = [](const std::vector<float>& got, const std::vector<float>& full, const float scale, const float tolerance)
				{
					for (std::size_t i = 0; i < got.size(); i++)
					{
						if (!(std::fabs(got[i] - full[i] * scale) <= tolerance * (1.0f + std::fabs(full[i]))))
						{
							return false;
						}
					}
					return true;
				};

				const auto full = evaluate(1.0f, 1.0f);
				const auto matches = [&](const std::pair<std::vector<float>, std::vector<float>>& got, const float scale, const float tolerance)
				{
					return scaled(got.first, full.first, scale, tolerance) && scaled(got.second, full.second, 1.0f, tolerance);
				};
				if (!reveal)
				{
					for (const auto a : { 0.5f, 0.25f })
					{
						if (!matches(evaluate(a, 1.0f), a, 1e-4f))
						{
							return utils::string::va("its output at vertex alpha %g is not %g times its output at 1", a, a);
						}
					}
					return {};
				}

				const auto soft = read_float(mp, find_variable(mp, "alphaRevealSoftEdge"), 0);
				const auto ramp = std::clamp(read_float(mp, find_variable(mp, "alphaRevealRamp"), 0), 0.0f, 1.0f);
				// BO3's reveal of r at vertex alpha a; none where its edge has no width
				const auto bo3_reveal = [&](const float r, const float a) -> std::optional<float>
				{
					const auto a1 = std::clamp(a * 0.998f + 0.001f, 0.0f, 1.0f);
					const auto e0 = std::clamp(1.0f - a1 - soft * std::pow(a1, ramp), 0.0f, 1.0f);
					const auto e1 = std::clamp(1.0f - a1 + soft * std::pow(1.0f - a1, ramp), 0.0f, 1.0f);
					if (!(e1 > e0))
					{
						return std::nullopt;
					}
					return std::clamp((r - e0) / (e1 - e0), 0.0f, 1.0f);
				};
				const auto at_full = bo3_reveal(1.0f, 1.0f);
				if (!at_full || *at_full <= 0.0f)
				{
					return "BO3's reveal at vertex alpha 1 is empty";
				}
				for (const auto a : { 0.2f, 0.5f, 0.8f })
				{
					for (const auto r : { 0.1f, 0.35f, 0.6f, 0.85f })
					{
						const auto revealed = bo3_reveal(r, a);
						if (!revealed)
						{
							continue;
						}
						// (the reveal divides by e1 - e0: a narrow edge magnifies the float error)
						const auto scale = *revealed / *at_full;
						if (!matches(evaluate(a, r), scale, 1e-3f))
						{
							return utils::string::va("its output at vertex alpha %g, reveal map %g is not %g (BO3's reveal) times its output "
								"at alpha 1", a, r, scale);
						}
					}
				}
				return {};
			}

			// Where `prog` samples texture `t`, from the probes of its sample instructions over the quad at the origin of a
			// 4096 x 4096 grid (lane 0 at base uv (0.5, 0.5) / 4096). Throws when it is not probed or sampled two ways.
			sample_map sample_map_of(const std::vector<std::pair<const material_program*, shader_eval::sample_probe>>& probes,
				const material_program* prog, const texture_binding& t)
			{
				constexpr auto texel = 1.0f / 4096.0f;
				constexpr auto lane0 = 0.5f / 4096.0f;
				std::optional<sample_map> out;
				for (const auto& [pr, probe] : probes)
				{
					if (pr != prog || probe.texture_slot != t.slot)
					{
						continue;
					}
					sample_map map{};
					map.m[0] = probe.dudx / texel;
					map.m[1] = probe.dudy / texel;
					map.m[3] = probe.dvdx / texel;
					map.m[4] = probe.dvdy / texel;
					map.m[2] = probe.u - (map.m[0] * lane0 + map.m[1] * lane0);
					map.m[5] = probe.v - (map.m[3] * lane0 + map.m[4] * lane0);
					map.offset[0] = probe.offset_u;
					map.offset[1] = probe.offset_v;
					if (out && (std::memcmp(out->m, map.m, sizeof(map.m)) || out->offset[0] != map.offset[0] || out->offset[1] != map.offset[1]))
					{
						throw std::runtime_error(utils::string::va("texture \"%s\" is sampled at two different places", t.name.data()));
					}
					out = map;
				}
				if (!out)
				{
					throw std::runtime_error(utils::string::va("texture \"%s\" is not sampled where the probe sees it", t.name.data()));
				}
				return *out;
			}

			plan make_plan(const Material* material, const info& inf)
			{
				plan p{};
				if (!inf.techset_loaded)
				{
					p.reason = "its technique set is in a zone that was not loaded";
					return p;
				}
				if (inf.cls == surface_class::shadow_only)
				{
					p.supported = true;
					p.techset = inf.model ? "mo_shadowcaster" : "w_shadowcaster";
					return p;
				}
				if (inf.cls != surface_class::opaque && inf.cls != surface_class::decal && inf.cls != surface_class::trans)
				{
					p.reason = "forward-lit emissive template not converted yet";
					return p;
				}

				const auto bo3 = bo3_template(material);
				if (bo3.starts_with("lit_water_sim_flow") || bo3.starts_with("water_shore_flow"))
				{
					// BO3's water, converted without a bake (write_water): stock IW7's refractive water with scrolled normals on
					// the world; on models, which IW7 has no such water for, its UV-animated lit water (stock
					// water_lake_geneva_close_tsunami). BO3 samples the water's normal map (and _color's decal map) at
					// uv x normalMapScale: the surfaces' texture coordinates are moved there (uv_span), a model's also by whole
					// periods of it to centre them on 0 (uv_origin), which keeps its half floats as fine as they go
					p.water = true;
					p.techset = inf.model ? "mco_l_sm_ua_replace_i0c0s0n0p0" : "w_l_sm_ndw_blend_i0c0s0n0a0tm0pa0_nop_sraf0_osn_trans";
					if (!world_techset_donors::find(p.techset, inf.camera_region))
					{
						p.reason = utils::string::va("no donor for techset %s in camera region %u", p.techset.data(), inf.camera_region);
						return p;
					}
					float tiling[2];
					try
					{
						const auto lit = load_program(material, "lit", { nullptr, nullptr, true });
						const auto* scale = find_variable(lit, "normalMapScale");
						for (auto axis = 0u; axis < 2; axis++)
						{
							tiling[axis] = scale ? read_float(lit, scale, axis) : 0.0f;
							if (!(tiling[axis] > 0.0f))
							{
								throw std::runtime_error("water without a positive normalMapScale");
							}
						}
					}
					catch (const std::exception& e)
					{
						p.reason = e.what();
						return p;
					}
					for (auto axis = 0u; axis < 2; axis++)
					{
						p.uv_span[axis] = 1.0f / tiling[axis];
						if (inf.model && inf.used_uv && !inf.used_uv->empty())
						{
							const auto centre = (inf.used_uv->min[axis] + inf.used_uv->max[axis]) * 0.5f;
							p.uv_origin[axis] = std::round(centre * tiling[axis]) / tiling[axis];
						}
					}
					p.supported = true;
					return p;
				}

				material_program mp{};
				material_program lit{};
				auto surface_alpha_test = false; // an opaque forward surface's lit shader discards (its cut surface program does not)
				try
				{
					if (inf.cls == surface_class::trans)
					{
						lit = load_program(material, "lit", { nullptr, nullptr, true });
						// BO3's additive emissive templates (decal_emissive): the lit technique adds its output and reads no
						// texture but emissive maps, so what it adds is its emission: IW7's unlit add
						const auto emission_only = !lit.textures.empty() && std::ranges::all_of(lit.textures, [](const texture_binding& t)
						{
							return t.name.starts_with("emissiveMap");
						});
						p.unlit_emissive = adds(lit.technique) && emission_only;
					}
					if (p.unlit_emissive)
					{
						mp = lit;
					}
					else if (inf.cls == surface_class::trans)
					{
						// the surface from the lit technique's debug-override permutation; the lit shader itself for the
						// emissive and the alpha it draws with
						mp = load_program(material, "lit", { nullptr, nullptr, false, true });
						p.surface = true;
						p.forward_emissive = std::ranges::any_of(lit.textures, [](const texture_binding& t)
						{
							return t.name.starts_with("emissiveMap");
						});

						// the values its debug-override permutation does not label
						const auto& surface = forward_surface::get(mp.pixel_shader->prog.loadDef.program, mp.pixel_shader->prog.loadDef.programSize);

						// a surface whose albedo or occlusion forward_surface holds constant: every material texture the lit
						// shader reads feeds a surface value (the emissive maps the emissive, from the lit shader itself),
						// so no colour or occlusion map is dropped
						constexpr auto constant_roles = (1u << static_cast<std::uint32_t>(forward_surface::role::albedo))
							| (1u << static_cast<std::uint32_t>(forward_surface::role::occlusion));
						if (surface.literal & constant_roles)
						{
							shader_eval::taint_result::texture_set feeds{};
							for (const auto& target : mp.taint.output_textures)
							{
								for (const auto& component : target)
								{
									feeds |= component;
								}
							}
							for (const auto& t : lit.textures)
							{
								const auto* own = find_binding(mp, t.name.data());
								auto fed = t.name.starts_with("emissiveMap");
								for (auto c = 0u; own && own->slot < 128 && c < 4; c++)
								{
									fed = fed || feeds.test(own->slot * 4 + c);
								}
								if (!fed)
								{
									throw std::runtime_error(utils::string::va("texture \"%s\" of the lit shader feeds no surface value", t.name.data()));
								}
							}
						}
						constexpr const char* role_names[] = { "albedo", "alpha", "normal", "specular", "gloss", "occlusion" };
						std::string blended;
						for (auto r = 0u; r < std::size(role_names); r++)
						{
							if (surface.from_decal_blend & (1u << r))
							{
								blended += (blended.empty() ? "" : ", ") + std::string(role_names[r]);
							}
						}
						if (!blended.empty())
						{
							p.surface_sources = blended + " read at its forward decal blend";
						}
						const auto add_source = [&](const std::string& text)
						{
							p.surface_sources += (p.surface_sources.empty() ? "" : "; ") + text;
						};
						if (surface.literal & (1u << static_cast<std::uint32_t>(forward_surface::role::albedo)))
						{
							add_source("albedo 0 (its decal blend has no albedo)");
						}
						if (surface.literal & (1u << static_cast<std::uint32_t>(forward_surface::role::occlusion)))
						{
							add_source("occlusion 1 (no debug-override step, no occlusion input)");
						}
						if (surface.anchored)
						{
							std::string anchored;
							for (auto r = 0u; r < std::size(role_names); r++)
							{
								if (surface.anchored & (1u << r))
								{
									anchored += (anchored.empty() ? "" : ", ") + std::string(role_names[r]);
								}
							}
							add_source(anchored + " read where its lighting takes them (drawn with IW7's standard lighting)");
						}
					}
					else if (inf.cls == surface_class::opaque && !has_technique(material->techniqueSet, "gbuffer")
						&& has_technique(material->techniqueSet, "lit"))
					{
						lit = load_program(material, "lit", { nullptr, nullptr, true });
						// emissiveMap, or the layers emissiveMap1-3 of the scroll_2layer / 3layer templates
						const auto emissive_map = std::ranges::any_of(lit.textures, [](const texture_binding& t)
						{
							return t.name.starts_with("emissiveMap");
						});
						if (!emissive_map && bo3_template(material) == "skin")
						{
							// BO3's skin is forward-lit (its subsurface scattering) with no gbuffer technique: the surface from
							// its lit technique's debug-override permutation, drawn opaque with IW7's skin (sss) techsets
							mp = load_program(material, "lit", { nullptr, nullptr, false, true });
							p.surface = true;
							p.opaque_surface = true;
							p.skin = true;
						}
						else if (!emissive_map && bo3_template(material) == "eye")
						{
							// BO3's eye is IW7's eye shader family: converted to it without a bake (write_eye)
							p.eye = true;
						}
						else if (!emissive_map && (bo3_template(material).starts_with("hair")
							|| bo3_template(material).find("anisotropic") != std::string::npos))
						{
							// BO3's opaque hair is forward-lit too (its two anisotropic lobes), as is its lit_anisotropic (satin
							// cloth): the surface from the debug-override permutation, drawn as a
							// standard opaque surface (IW7's anih hair takes a strand direction map BO3's has not). An alpha test
							// in the lit shader (hair_alphatest: colorMap.a x vertex alpha < 0.5, IW7's threshold) stays one, on
							// the surface's alpha.
							mp = load_program(material, "lit", { nullptr, nullptr, false, true });
							p.surface = true;
							p.opaque_surface = true;
							surface_alpha_test = lit.taint.discard_textures.any();
							ZONETOOL_INFO("model material %s: BO3's %s shading converted as a standard surface%s", describe_material(material).data(),
								bo3_template(material).data(), surface_alpha_test ? " with its alpha test" : "");
						}
						else if (!emissive_map)
						{
							throw std::runtime_error("forward-lit material without an emissive map");
						}
						else
						{
							const auto* sibling = sibling_techset(material);
							mp = load_program(material, "gbuffer", { sibling, &lit, false });
							p.forward_emissive = true;
							p.base_techset = sibling->name;
						}
					}
					else
					{
						mp = load_program(material, "gbuffer");
					}
				}
				catch (const std::exception& e)
				{
					p.reason = e.what();
					return p;
				}
				if (p.eye)
				{
					// stock IW7 eyes: camera region 1 (world_material classify_model), all 133 with this techset
					p.techset = "mo_l_sm_replace_i0c0s0_sss_eye_gtao";
					if (!inf.model || !world_techset_donors::find(p.techset, inf.camera_region))
					{
						p.reason = utils::string::va("no donor for techset %s in camera region %u", p.techset.data(), inf.camera_region);
						return p;
					}
					p.supported = true;
					return p;
				}

				use_vertex_colour(mp, inf);
				use_vertex_colour(lit, inf);
				if (inf.model && inf.used_colour)
				{
					// the vertex colour channels the baked program reads, with the values the material's surfaces have: one value
					// each is baked as its vertex shader passes it, a channel that varies per vertex at 1 (a model decal's alpha
					// is IW7's to apply: below)
					const auto read = colour_channels_read(mp);
					const auto& c = *inf.used_colour;
					std::string channels;
					auto varies = false;
					for (auto k = 0u; k < 4; k++)
					{
						if (read & (1u << k))
						{
							channels += c.min[k] == c.max[k] ? utils::string::va(" %c %u", "rgba"[k], c.min[k])
								: utils::string::va(" %c %u-%u", "rgba"[k], c.min[k], c.max[k]);
							varies |= c.min[k] != c.max[k];
						}
					}
					if (read)
					{
						ZONETOOL_INFO("model material %s: its shader reads vertex colour%s%s: %s", describe_material(material).data(), channels.data(),
							(read & 0x10) ? " and a COLOR its vertex shader makes another way (at 1)" : "",
							varies ? "varying per vertex, baked at 1" : "one value each, baked as its vertex shader passes them");
					}
				}

				p.decal = inf.cls == surface_class::decal;
				p.alpha_test = mp.taint.discard_textures.any() || surface_alpha_test;
				// (what is baked: the program, and the lit shader where its emission is taken)
				if (reads_weather(mp) || (p.forward_emissive && reads_weather(lit)))
				{
					ZONETOOL_INFO("%s material %s: its shader reads BO3's weather constants (rain, wind), which the bake has at 0",
						inf.model ? "model" : "world", describe_material(material).data());
				}

				// a surface whose occlusion is labelled in one branch only: skin's, whose other branch is BO3's screen-space
				// GTAO (enableGTAO), which samples no aoMap; IW7's skin techsets make the same choice (_gtao, no o0)
				const auto conditional_occlusion = p.surface && forward_surface::get(mp.pixel_shader->prog.loadDef.program,
					mp.pixel_shader->prog.loadDef.programSize).conditional_occlusion;
				if (conditional_occlusion && !p.skin)
				{
					p.reason = "forward surface whose occlusion is labelled in one branch only";
					return p;
				}
				if (p.skin)
				{
					const auto* gtao = find_variable(mp, "enableGTAO");
					p.gtao = gtao && read_bool(mp, gtao);
				}

				if (p.surface && !p.opaque_surface)
				{
					// BO3 numbers the blend factors as Direct3D 11 does (ONE 2, SRC_ALPHA 5, INV_SRC_ALPHA 6; op ADD 1), and so
					// do IW7's blend state bits: blend is SRC_ALPHA / INV_SRC_ALPHA, blendadd ONE / INV_SRC_ALPHA (stock
					// cnd_window_cracks_01 0x0F012165 against cnd_glass_cut_edge 0x0F012162)
					const auto* state = mp.technique->stateMap;
					if (!state)
					{
						p.reason = "lit technique without a state map";
						return p;
					}
					// (the factors decide, as for decals: BO3's transparent lit techniques read 5/1/6 with the struct's
					// "enabled" bit clear, so that bit is not what the struct names it)
					const auto& b = state->blend[0];
					if (b.rgbOp != 1 || b.rgbDst != 6 || (b.rgbSrc != 5 && b.rgbSrc != 2))
					{
						p.reason = utils::string::va("transparent blend %u/%u/%u is neither alpha nor premultiplied alpha", b.rgbSrc, b.rgbOp,
							b.rgbDst);
						return p;
					}
					p.premultiplied = b.rgbSrc == 2;
					if (p.alpha_test)
					{
						p.reason = "alpha-tested transparent material: no stock IW7 blend techset has alpha test";
						return p;
					}
					if (find_binding(lit, "revealMap"))
					{
						// a _script template's reveal is a script constant (lit_transparent_reveal_script's lit vertex shader
						// passes shaderConstantSet[0] as COLOR1.x, 0 hidden, 1 revealed), as is a reveal whose COLOR the
						// vertex shader makes from no vertex channel (lit_emissive_spectre): the bake's colour input 1 is its
						// fully revealed state, which IW7 script can only show or hide
						const auto script = bo3_template(material).find("_script") != std::string::npos;
						const auto made_otherwise = (colour_channels_read(lit) & 0x1F) == 0x10;
						if (!script && !made_otherwise)
						{
							// revealed by its vertex alpha, painted per vertex: IW7 reveals by the vertex alpha only in its decal
							// techsets (v0), so the surface is drawn as IW7's world reveal decal (reveal_decal), which takes BO3 to
							// reveal it as IW7 does and to read the vertex rgb linearised as every lit world template does (IW7's
							// wc_ techsets multiply the albedo by it squared, the vertex byte being c^1.1: gfxworld)
							if (inf.model && !p.premultiplied && !p.forward_emissive)
							{
								// a model has no IW7 blend techset with reveal (v0), and IW7's model reveal decals (mco_) are opaque
								// decals: drawn fully revealed, as the script-driven reveals below, not left out
								ZONETOOL_INFO("model material %s: BO3 reveals it by its vertex alpha, baked fully revealed (no stock IW7 model "
									"blend techset has reveal)", describe_material(material).data());
							}
							else
							{
								if (inf.model || p.premultiplied || p.forward_emissive)
								{
									p.reason = utils::string::va("transparent %s material with a reveal map: no stock IW7 %s techset has reveal (v0)",
										inf.model ? "model" : (p.forward_emissive ? "emissive" : "premultiplied"),
										inf.model ? "model blend" : (p.forward_emissive ? "emissive blend" : "blendadd"));
									return p;
								}
								if (!find_binding(mp, "revealMap") || !find_variable(mp, "alphaRevealSoftEdge") || !find_variable(mp, "alphaRevealRamp"))
								{
									p.reason = "transparent material with a reveal map its surface program does not reveal with (revealMap, "
										"alphaRevealSoftEdge, alphaRevealRamp)";
									return p;
								}
								std::uint32_t linearised = 0;
								if ((colour_channels_read(lit, &linearised) & 0x7) != 0x7 || (linearised & 0x7) != 0x7)
								{
									p.reason = "transparent material revealed by its vertex alpha whose shader does not take the vertex rgb "
										"linearised (IW7's reveal decals multiply the albedo by it)";
									return p;
								}
								const auto why = vertex_alpha_mismatch(mp, true, true);
								if (!why.empty())
								{
									p.reason = "transparent material revealed by its vertex alpha, not as IW7's reveal decals: " + why;
									return p;
								}
								p.reveal_decal = true;
								ZONETOOL_INFO("world material %s: revealed by its vertex alpha, drawn as IW7's world reveal decal (v0) after the "
									"world's decals", describe_material(material).data());
							}
						}
						else
						{
							ZONETOOL_INFO("%s material %s: BO3 reveals it by a script constant%s, baked fully revealed (no stock IW7 blend "
								"techset has reveal)", inf.model ? "model" : "world", describe_material(material).data(),
								script ? "" : " (its vertex shader makes the reveal amount from no vertex channel)");
						}
					}
				}

				if (p.decal)
				{
					const auto* state = mp.technique->stateMap;
					if (!state)
					{
						p.reason = "decal technique without a state map";
						return p;
					}
					const auto blend_of = [&](const int rt) -> const GfxStateMap::Blend&
					{
						return state->primitive.mrtBlending ? state->blend[rt] : state->blend[0];
					};
					const auto& b0 = blend_of(0);
					const auto& b1 = blend_of(1);
					const auto& b2 = blend_of(2);
					// premultiplied alpha: ONE (2) / ADD (1) / INVSRCALPHA (6)
					const auto premultiplied = [](const GfxStateMap::Blend& b)
					{
						return b.writeMask == 0 || (b.rgbSrc == 2 && b.rgbOp == 1 && b.rgbDst == 6);
					};
					// a multiply (ZERO (1) / ADD (1) / SRC_COLOR (3): the albedo under it times its RT0) or no albedo at all
					const auto multiplies = b0.rgbSrc == 1 && b0.rgbOp == 1 && b0.rgbDst == 3;
					if (multiplies || (b0.writeMask & 7) == 0)
					{
						p.multiply = true;
						p.identity = (b0.writeMask & 7) == 0;
						std::string dropped;
						const auto drop = [&](const bool written, const char* what)
						{
							if (written)
							{
								dropped += (dropped.empty() ? "" : ", ") + std::string(what);
							}
						};
						drop((b1.writeMask & 3) == 3, "normal");
						drop((b1.writeMask & 4) != 0, "gloss");
						drop((b2.writeMask & 3) == 3, "specular");
						drop((b2.writeMask & 4) != 0, "occlusion");
						ZONETOOL_WARNING("%s material %s: %s%s", inf.model ? "model" : "world", describe_material(material).data(),
							p.identity ? "a decal that does not write the albedo (normal-only): drawn as a multiply by 1, nothing"
								: "a multiply decal: its albedo multiply as IW7's unlit multiply of the lit result",
							dropped.empty() ? "" : ("; its " + dropped + " not drawn (IW7's forward decals cannot change the lighting "
								"inputs of the surface under them)").data());
						p.writes_normal = p.writes_gloss = p.writes_specular = p.writes_occlusion = false;
					}
					else
					{
						if (!premultiplied(b0) || !premultiplied(b1) || !premultiplied(b2))
						{
							p.reason = utils::string::va("decal blend is not premultiplied alpha (RT0 %u/%u/%u)", b0.rgbSrc, b0.rgbOp, b0.rgbDst);
							return p;
						}
						if ((b0.writeMask & 7) != 7)
						{
							p.reason = utils::string::va("decal writes part of the albedo (RT0 mask %u)", b0.writeMask);
							return p;
						}
						p.writes_normal = (b1.writeMask & 3) == 3;
						p.writes_gloss = (b1.writeMask & 4) != 0;
						p.writes_specular = (b2.writeMask & 3) == 3;
						p.writes_occlusion = (b2.writeMask & 4) != 0;
					}
				}
				if (p.multiply)
				{
					// stock IW7 multiply decals: wc_ / mco_unlit_multiply_lin_ndw, decal region (vertex rgb written white: BO3's
					// multiply reads only the vertex alpha)
					p.techset = inf.model ? "mco_unlit_multiply_lin_ndw" : "wc_unlit_multiply_lin_ndw";
					if (!world_techset_donors::find(p.techset, inf.camera_region))
					{
						p.reason = utils::string::va("no donor for techset %s in camera region %u", p.techset.data(), inf.camera_region);
						return p;
					}
					if (p.identity)
					{
						p.width = p.height = 4;
						p.supported = true;
						return p;
					}
				}
				p.reveal = p.reveal_decal
					|| (!p.surface && !p.multiply && find_binding(mp, "revealMap") && find_variable(mp, "alphaRevealSoftEdge") && find_variable(mp, "alphaRevealRamp"));
				// an alpha test on a reveal whose amount is known: a _script template's script constant (its vertex shader's
				// COLOR1.x, the bake's 1: fully revealed) or the vertex alpha of surfaces that have one (lit_alphatest_reveal:
				// COLOR1.x = v.w). No stock IW7 atest techset has reveal (v0), so the coverage that reveal gives is evaluated.
				auto reveal_evaluated = false;
				if (p.alpha_test && p.reveal && !p.decal)
				{
					const auto script = bo3_template(material).find("_script") != std::string::npos;
					const auto one_colour = inf.model && inf.used_colour && inf.used_colour->uniform();
					if (script || one_colour)
					{
						p.reveal = false;
						reveal_evaluated = true;
						ZONETOOL_INFO("%s material %s: alpha test on a reveal %s, its coverage baked (no stock IW7 atest techset has reveal)",
							inf.model ? "model" : "world", describe_material(material).data(),
							script ? "BO3 drives by a script constant, fully revealed" : "at its surfaces' one vertex alpha");
					}
				}
				// A model decal whose surfaces have a vertex alpha below 1: BO3 scales the decal's coverage by it or reveals the
				// decal by it, which no bake holds (a decal's coverage comes straight from its texture, and one texture cannot
				// follow an alpha that varies per vertex). IW7's vertex colour decal techsets (mco_) take it from the vertex at
				// run time, where BO3's program is measured to take it their way (vertex_alpha_mismatch).
				if (p.decal && !p.multiply && inf.model && inf.used_colour && !inf.used_colour->empty() && inf.used_colour->min[3] < 255)
				{
					std::uint32_t linearised = 0;
					const auto read = colour_channels_read(mp, &linearised);
					if (read & 0x8)
					{
						const auto why = read != 0x8 || linearised ? std::string("it reads more of the vertex colour than its raw alpha")
							: vertex_alpha_mismatch(mp, p.reveal);
						if (why.empty())
						{
							p.vertex_alpha = true;
							ZONETOOL_INFO("model material %s: its vertex alpha (%u-%u) %s at run time with IW7's vertex colour decal techset "
								"(mco_), its surfaces' vertex rgb written white", describe_material(material).data(), inf.used_colour->min[3],
								inf.used_colour->max[3], p.reveal ? "reveals it" : "scales its coverage");
						}
						else
						{
							ZONETOOL_WARNING("model material %s: its vertex alpha (%u-%u) is left out: IW7's vertex colour decal techsets "
								"take it another way (%s)", describe_material(material).data(), inf.used_colour->min[3], inf.used_colour->max[3],
								why.data());
						}
					}
				}
				// a detail normal map stays a separate IW7 texture (q0) where a stock techset takes one; alpha-tested, forward
				// emissive, transparent and model decal materials have none (a model decal with reveal does, but not one with
				// its vertex alpha), so theirs is baked in like the other layers
				p.detail = find_binding(mp, "detailMap") && find_variable(mp, "detailScale") && find_variable(mp, "detailScaleHeight")
					&& !p.alpha_test && !p.forward_emissive && !p.surface && !p.multiply && !(p.decal && inf.model && (!p.reveal || p.vertex_alpha));

				const auto rt2 = mp.target_reg[2];
				if (rt2 >= 0 && p.surface)
				{
					// the occlusion in RT2.w; F0 is read as it is (no parity pass). A skin with enableGTAO has none of its own.
					p.occlusion = mp.taint.output_textures[rt2][3].any() && !p.gtao;
				}
				else if (rt2 >= 0)
				{
					p.occlusion = mp.taint.output_textures[rt2][2].any() && p.writes_occlusion;
					p.coloured_specular = (mp.taint.output_textures[rt2][0].any() || mp.taint.output_textures[rt2][1].any()) && p.writes_specular;
				}
				if (p.vertex_alpha && p.occlusion)
				{
					// no stock mco_ decal has o0: the decal's occlusion (IW7 applies it to specular only) is left out
					p.occlusion = false;
					ZONETOOL_WARNING("model material %s: its occlusion is left out (no stock IW7 vertex colour decal techset has occlusion)",
						describe_material(material).data());
				}

				// probe every texture's tiling rate with one quad at the origin (detail maps stay
				// separate textures, so they do not set the bake size); a forward emissive material's
				// emissive layer comes from its lit shader, so that one is probed too
				auto probe_mp = mp;
				if (p.detail)
				{
					write_float(probe_mp, find_variable(mp, "detailScaleHeight"), 0, 0.0f);
				}
				std::vector<const material_program*> probed = { &probe_mp };
				if (p.forward_emissive)
				{
					probed.push_back(&lit);
				}
				std::vector<std::pair<const material_program*, shader_eval::sample_probe>> probes;
				for (const auto* prog : probed)
				{
					std::unordered_map<std::string, std::array<float, 4>> constants_everywhere;
					for (const auto& t : prog->textures)
					{
						constants_everywhere[t.name] = { 0.5f, 0.5f, 0.5f, 1.0f };
					}
					auto bound = bind_textures(*prog, constants_everywhere);
					shader_eval::machine m(prog->program);
					m.set_unbound_resources_zero(prog->forward);
					for (const auto& [slot, data] : prog->cbuffers)
					{
						m.bind_cbuffer(slot, data.data(), data.size());
					}
					for (const auto& [slot, tex] : bound.slots)
					{
						m.bind_texture(slot, tex);
					}
					for (const auto& b : prog->buffers)
					{
						m.bind_structured(b.slot, b.data.data(), b.data.size(), b.stride);
					}
					const grid probe_grid{ 4096, 4096, 0.0f, 0.0f, 1.0f, 1.0f, 0 };
					set_inputs(m, *prog, probe_grid, 0, 0);
					m.enable_probe(true);
					m.run();
					for (const auto& probe : m.probes())
					{
						probes.emplace_back(prog, probe);
					}
				}

				const auto texel = 1.0f / 4096.0f;
				std::uint32_t period[2] = { 1, 1 };
				double density[2] = { 0.0, 0.0 };
				std::string density_from[2]; // the texture that sets each axis's density
				std::string aperiodic; // why the composite repeats after no whole number of units, when it does not
				for (const auto& [prog, probe] : probes)
				{
					const auto* t = binding_by_slot(*prog, probe.texture_slot);
					if (!t || !t->image || (p.detail && t->name == "detailMap"))
					{
						continue; // (an engine texture without an image reads 0 everywhere)
					}
					const auto w = static_cast<double>(t->image->width);
					const auto h = static_cast<double>(t->image->height);
					// coordinate change per unit of base u (x) and base v (y)
					const double j[2][2] = { { probe.dudx / texel, probe.dudy / texel }, { probe.dvdx / texel, probe.dvdy / texel } };
					const auto axis_aligned = std::fabs(j[0][1]) < 1e-4 && std::fabs(j[1][0]) < 1e-4;
					const auto swapped = std::fabs(j[0][0]) < 1e-4 && std::fabs(j[1][1]) < 1e-4;
					if (!axis_aligned && !swapped)
					{
						// repeats along no base axis; a base axis crosses the texture's texels diagonally
						aperiodic = utils::string::va("texture \"%s\" is sampled rotated by a non-right angle", t->name.data());
						for (auto base_axis = 0; base_axis < 2; base_axis++)
						{
							const auto texels = std::hypot(j[0][base_axis] * w, j[1][base_axis] * h);
							if (texels > density[base_axis])
							{
								density[base_axis] = texels;
								density_from[base_axis] = utils::string::va("%s (%s) rotated, %.0f texels a unit", t->name.data(), t->image->name, texels);
							}
						}
						continue;
					}
					for (auto base_axis = 0; base_axis < 2; base_axis++)
					{
						// texture axis driven by this base axis
						const auto tex_axis = axis_aligned ? base_axis : 1 - base_axis;
						const auto rate = std::fabs(j[tex_axis][base_axis]);
						if (rate < 1e-6)
						{
							continue; // constant along this axis
						}
						std::uint32_t num, den;
						if (to_fraction(static_cast<float>(rate), num, den))
						{
							// the texture repeats every den / num base units; the composite repeats
							// after an integer number of units that is a multiple of den / num
							const auto repeat = den / std::gcd(num, den);
							period[base_axis] = std::lcm(period[base_axis], repeat);
						}
						else
						{
							aperiodic = utils::string::va("texture \"%s\" tiles %.6f times per unit", t->name.data(), rate);
						}
						const auto size = tex_axis == 0 ? w : h;
						if (rate * size > density[base_axis])
						{
							density[base_axis] = rate * size;
							density_from[base_axis] = utils::string::va("%s (%s, %.0f texels) %.4g times a unit", t->name.data(), t->image->name,
								size, rate);
						}
					}
				}
				p.size_from = utils::string::va("u: %s; v: %s", density_from[0].data(), density_from[1].data());

				if (density[0] <= 0.0 || density[1] <= 0.0)
				{
					auto sampled = false;
					for (const auto& [prog, probe] : probes)
					{
						const auto* t = binding_by_slot(*prog, probe.texture_slot);
						sampled = sampled || (t && t->image);
					}
					if (!sampled)
					{
						p.reason = "no texture varies across the surface";
						return p;
					}
					// every texture is sampled at one place along the axis (lit_paintshop at BO3's zero weapon parameters:
					// its paint layer's texture transform is 0): the material is uniform along it, one block holds it
					for (auto axis = 0; axis < 2; axis++)
					{
						if (density[axis] <= 0.0)
						{
							density[axis] = 4.0;
							density_from[axis] = "no texture varies along it";
						}
					}
					p.size_from = utils::string::va("u: %s; v: %s", density_from[0].data(), density_from[1].data());
				}
				// a composite without a period is only held by a bake of the area the surfaces use, which never wraps
				const auto windowable = inf.used_uv && !inf.used_uv->empty();
				if (!aperiodic.empty() && !windowable)
				{
					p.reason = aperiodic;
					return p;
				}

				// coverage and reveal come straight from their textures (the discard's threshold is IW7's: every BO3
				// alpha-test template discards where alpha x vertex alpha < 0.5), sampled where the shader samples them
				if ((p.decal && !p.multiply) || p.alpha_test)
				{
					// an alpha test on more than one texture channel, or on a vertex colour its surfaces have at another value
					// than 1: the coverage the pixel shader compares, from the shader cut at its discard
					std::pair<const texture_binding*, std::uint32_t> source{};
					std::string why_evaluated;
					try
					{
						if (reveal_evaluated)
						{
							why_evaluated = "its alpha test is on a reveal map, which no stock IW7 atest techset has";
						}
						else
						{
							source = coverage_source(mp, p.alpha_test, p.reveal);
							if (p.alpha_test && discard_reads_colour(mp))
							{
								why_evaluated = "its alpha test reads a vertex colour its surfaces have at another value than 1";
							}
						}
					}
					catch (const std::exception& e)
					{
						if (!p.alpha_test)
						{
							throw;
						}
						why_evaluated = e.what();
					}
					if (!why_evaluated.empty())
					{
						const auto& cut = forward_surface::coverage(mp.pixel_shader->prog.loadDef.program, mp.pixel_shader->prog.loadDef.programSize);
						if (p.decal || p.reveal || p.surface || mp.draw_method != 0 || mp.bytecode || cut.bytecode.empty())
						{
							throw std::runtime_error(why_evaluated + (cut.bytecode.empty() ? "; " + cut.reason : std::string{}));
						}
						p.coverage_program = true;
						source = {};
					}
					if (source.first)
					{
						p.coverage_map = sample_map_of(probes, &probe_mp, *source.first);
					}
					if (p.reveal)
					{
						p.reveal_map = sample_map_of(probes, &probe_mp, *find_binding(mp, "revealMap"));
					}
				}
				else if (p.reveal_decal)
				{
					// (its coverage is its surface program's alpha, evaluated)
					p.reveal_map = sample_map_of(probes, &probe_mp, *find_binding(mp, "revealMap"));
				}

				p.uv_period[0] = period[0];
				p.uv_period[1] = period[1];
				for (auto axis = 0; axis < 2; axis++)
				{
					p.uv_origin[axis] = 0.0f;
					p.uv_span[axis] = static_cast<float>(period[axis]);
					if (windowable)
					{
						// the area the surfaces use and a margin of texels at the bake's density, which keeps bilinear
						// filtering and the first levels' footprints from reaching across the edge (the image wraps)
						const auto margin = window_margin / density[axis];
						const auto lo = static_cast<double>(inf.used_uv->min[axis]) - margin;
						const auto hi = static_cast<double>(inf.used_uv->max[axis]) + margin;
						if (!aperiodic.empty() || hi - lo < static_cast<double>(period[axis]))
						{
							p.uv_origin[axis] = static_cast<float>(lo);
							p.uv_span[axis] = static_cast<float>(hi - lo);
							p.windowed = true;
						}
					}
				}
				p.width = static_cast<std::uint32_t>(std::ceil(density[0] * p.uv_span[0] - 1e-6));
				p.height = static_cast<std::uint32_t>(std::ceil(density[1] * p.uv_span[1] - 1e-6));
				// BC blocks, and from 128 texels whole blocks in the first four levels: a streamed part's top level must be
				// (write_streamed_image), else a part is lost and distant surfaces draw the 1 x 1 placeholder
				const auto blocks = [](const std::uint32_t size)
				{
					return size >= stream_min_size ? (size + 31) & ~31u : std::max(4u, (size + 3) & ~3u);
				};
				p.width = blocks(p.width);
				p.height = blocks(p.height);
				// a streamed part is a whole BC7 mip chain (1 byte a texel, 4/3 with the mips) with a 26-bit size, and
				// D3D11 textures stop at 16384, and IW7 stops a level whose streamed images add up to more than 17.5 GB
				// (0x1404B7100, "Memory Error: 1 384"): a bake keeps at most 2048 x 2048 texels. Halve it until
				// all hold
				while (static_cast<std::uint64_t>(p.width) * p.height * 4 / 3 > stream_max_part || p.width > 16384 || p.height > 16384 ||
						static_cast<std::uint64_t>(p.width) * p.height > 2048ull * 2048ull)
				{
					p.width = blocks(p.width / 2);
					p.height = blocks(p.height / 2);
					p.reduced++;
				}

				// IW7 techset: the world's wc_ ones, or the model mo_ ones of the same layout (stock IW7 has no mo_
				// decal with reveal (v0) or a detail map (q0))
				const std::string family = inf.model ? "mo" : "wc";
				if (p.multiply)
				{
					// (set with the decal's blend above)
				}
				else if (p.unlit_emissive)
				{
					// stock IW7's emissive overlays (light_*_on, decal region): an unlit add with an HDR colour (mkhdr)
					p.techset = family + "_unlit_add_lin_ct_ndw_mkhdr";
					if (std::ranges::any_of(mp.program->bindings(), [](const shader_eval::binding_desc& b) { return b.name == "shaderConstantSetBuffer"; }))
					{
						ZONETOOL_INFO("%s material %s: its emission is scaled by per-instance script constants (shaderConstantSet), baked "
							"with them at 0", inf.model ? "model" : "world", describe_material(material).data());
					}
				}
				else if (p.skin)
				{
					// stock IW7 skin: mo_ sss techsets, camera region 1 (all 613 stock skin materials); the _gtao one the stock
					// skin without an occlusion map uses
					if (!inf.model)
					{
						p.reason = "world material of BO3's skin template: IW7's skin (sss) techsets are model techsets";
						return p;
					}
					p.techset = p.occlusion ? "mo_l_sm_replace_i0c0s0o0n0p0_sss" : "mo_l_sm_replace_i0c0s0n0p0_sss_gtao";
				}
				else if (p.opaque_surface)
				{
					// another opaque forward surface (BO3's eye, hair): a standard opaque surface, alpha-tested where BO3's is
					p.techset = family + (p.alpha_test ? (p.occlusion ? "_l_sm_atest_i0c0s0o0n0pa0" : "_l_sm_atest_i0c0s0n0pa0")
						: (p.occlusion ? "_l_sm_replace_i0c0s0o0n0p0" : "_l_sm_replace_i0c0s0n0p0"));
				}
				else if (p.surface)
				{
					// stock IW7's lit transparent (camera region 3) techsets with packed textures and a packed alpha: the
					// world's have vertex colour (wc_) except the occlusion and emissive blends, which stock only has as w_
					// (no vertex colour input); the model emissive blends only come with occlusion (o0)
					const std::string mode = p.premultiplied ? "_l_sm_ndw_blendadd_" : "_l_sm_ndw_blend_";
					if (p.reveal_decal)
					{
						// stock IW7's world reveal decals (v0, decal region), as the world's gbuffer decals with reveal
						p.techset = p.occlusion ? "wc_l_sm_ndw_blend_i0c0s0o0n0v0pa0" : "wc_l_sm_ndw_blend_i0c0s0n0v0pa0";
					}
					else if (inf.model)
					{
						p.techset = "mo" + mode + (p.forward_emissive ? "i0c0s0o0n0e0pa0" : (p.occlusion ? "i0c0s0o0n0pa0" : "i0c0s0n0pa0"));
					}
					else if (p.forward_emissive)
					{
						p.techset = p.premultiplied ? "wc_l_sm_ndw_blendadd_i0c0s0n0e0pa0_nop" : "w_l_sm_ndw_blend_i0c0s0n0e0pa0";
					}
					else if (p.occlusion)
					{
						p.techset = p.premultiplied ? "wc_l_sm_ndw_blendadd_i0c0s0o0n0pa0" : "w_l_sm_ndw_blend_i0c0s0o0n0pa0";
					}
					else
					{
						p.techset = "wc" + mode + "i0c0s0n0pa0";
					}
				}
				else if (p.forward_emissive)
				{
					// stock IW7's emissive (e0) replace techsets: the world's is w_ (no vertex colour input) and has
					// no o0 variant,
					// whose occlusion IW7 only uses for specular occlusion
					if (p.alpha_test || p.reveal)
					{
						p.reason = utils::string::va("forward emissive material with %s: no stock IW7 e0 techset has it",
							p.alpha_test ? "alpha test" : "a reveal map");
						return p;
					}
					p.techset = inf.model ? (p.occlusion ? "mo_l_sm_replace_i0c0s0o0n0e0p0" : "mo_l_sm_replace_i0c0s0n0e0p0")
						: "w_l_sm_replace_i0c0s0n0e0p0";
				}
				else if (p.decal)
				{
					if (p.vertex_alpha)
					{
						// IW7's vertex colour decals (coverage x COLOR0.a; v0: the reveal edges from COLOR0.a), which stock only has
						// without occlusion and a detail normal;
						// mcopw_ ones also read per-vertex probe weights the converted models have not
						p.techset = p.reveal ? "mco_l_sm_ndw_blend_i0c0s0n0v0p0" : "mco_l_sm_ndw_blend_i0c0s0n0pa0";
					}
					else if (inf.model && p.reveal)
					{
						// stock IW7's one model decal with reveal (cp_rave mo/plastic_wood_siding_chipped_rvl_01) also takes a
						// detail normal (q0) and occlusion (o0): a decal without a detail map gets a flat one
						p.techset = "mo_l_sm_ndw_blend_i0c0s0o0n0v0q0pa0";
						p.flat_detail = !p.detail;
					}
					else if (p.reveal)
					{
						p.techset = p.detail ? "wc_l_sm_ndw_blend_i0c0s0n0v0q0p0" : (p.occlusion ? "wc_l_sm_ndw_blend_i0c0s0o0n0v0pa0" : "wc_l_sm_ndw_blend_i0c0s0n0v0pa0");
					}
					else
					{
						p.techset = family + (p.detail ? "_l_sm_ndw_blend_i0c0s0n0q0p0" : (p.occlusion ? "_l_sm_ndw_blend_i0c0s0o0n0pa0" : "_l_sm_ndw_blend_i0c0s0n0pa0"));
					}
				}
				else if (p.alpha_test)
				{
					if (p.reveal)
					{
						p.reason = utils::string::va("alpha-tested material with a reveal map: no stock %s_ atest techset has reveal (v0)", family.data());
						return p;
					}
					p.techset = family + (p.occlusion ? "_l_sm_atest_i0c0s0o0n0pa0" : "_l_sm_atest_i0c0s0n0pa0");
				}
				else
				{
					p.techset = family + (p.detail ? (p.occlusion ? "_l_sm_replace_i0c0s0o0n0q0p0" : "_l_sm_replace_i0c0s0n0q0p0")
						: (p.occlusion ? "_l_sm_replace_i0c0s0o0n0p0" : "_l_sm_replace_i0c0s0n0p0"));
				}

				// (a reveal decal is drawn in the decal region: world_material classify)
				const auto region = p.reveal_decal ? iw7_region_lit_decal : inf.camera_region;
				if (!world_techset_donors::find(p.techset, region))
				{
					p.reason = utils::string::va("no donor for techset %s in camera region %u", p.techset.data(), region);
					return p;
				}

				p.supported = true;
				return p;
			}

			// ---- writing ----------------------------------------------------------------------------

			void write_file(const std::string& path, const void* data, const std::size_t size)
			{
				filesystem::file file(path);
				file.open("wb");
				if (!file.get_fp())
				{
					throw std::runtime_error(utils::string::va("could not write %s", path.data()));
				}
				file.write(data, size, 1);
				file.close();
			}

			std::uint8_t iw7_state_flags(const world_techset_donors::techset_donor& donor, const MaterialTechnique* technique)
			{
				auto flags = static_cast<std::uint8_t>(donor.state_flags & ~0x3u);
				const auto* state = technique ? technique->stateMap : nullptr;
				if (!state)
				{
					return donor.state_flags;
				}
				// BO3 cullMode: 0 none, 1 back (the gbuffer state of every *_nocull template is 0)
				switch (state->primitive.cullMode)
				{
				case 0:
					break;
				case 1:
					flags |= 0x1;
					break;
				default:
					flags |= 0x2;
					break;
				}
				return flags;
			}

			struct iw7_texture_slot
			{
				std::uint32_t hash;
				std::string image;
			};

			void write_material_json(const std::string& name, const world_techset_donors::techset_donor& donor, const info& inf,
				std::uint8_t state_flags, const std::vector<std::pair<std::uint32_t, std::array<float, 4>>>& constants,
				const std::vector<iw7_texture_slot>& images, const bool atlas_frame_blend = false)
			{
				ordered_json j;
				j["techniqueSet->name"] = donor.techset;
				// IW7 0x80 = casts shadow (H1's MTL_GAMEFLAG layout; the IW5 port maps IW5's 0x40 to it and
				// no stock decal carries it); the rest is the techset's stock value
				auto game_flags = donor.game_flags;
				if (inf.cls == surface_class::opaque)
				{
					game_flags = static_cast<unsigned char>((game_flags & ~0x80u) | (inf.casts_shadow ? 0x80u : 0u));
				}
				j["gameFlags"] = game_flags;
				j["unkFlags"] = 0;
				j["sortKey"] = inf.sort_key;
				j["renderFlags"] = donor.render_flags;
				j["textureAtlasRowCount"] = donor.atlas_rows;
				j["textureAtlasColumnCount"] = donor.atlas_columns;
				j["textureAtlasFrameBlend"] = atlas_frame_blend ? 1 : 0;
				j["textureAtlasAsArray"] = 0;
				j["surfaceTypeBits"] = 0; // BO3 surface types are not mapped
				j["stateFlags"] = state_flags;
				j["cameraRegion"] = inf.camera_region;
				j["materialType"] = donor.material_type;
				j["assetFlags"] = 0;

				for (const auto& [hash, v] : constants)
				{
					auto known = false;
					for (auto i = 0u; i < donor.constant_count; i++)
					{
						known |= donor.constants[i].hash == hash;
					}
					if (!known)
					{
						throw std::runtime_error(utils::string::va("techset %s has no constant 0x%08X", donor.techset, hash));
					}
				}

				ordered_json table = ordered_json::array();
				for (auto i = 0u; i < donor.constant_count; i++)
				{
					const auto& c = donor.constants[i];
					std::array<float, 4> value = { c.literal[0], c.literal[1], c.literal[2], c.literal[3] };
					for (const auto& [hash, v] : constants)
					{
						if (hash == c.hash)
						{
							value = v;
						}
					}
					ordered_json entry;
					entry["name"] = c.name;
					entry["nameHash"] = c.hash;
					entry["literal"] = { value[0], value[1], value[2], value[3] };
					table.push_back(entry);
				}
				j["constantTable"] = table;

				ordered_json textures = ordered_json::array();
				for (auto i = 0u; i < donor.texture_count; i++)
				{
					const auto& t = donor.textures[i];
					std::string image = t.stock_image;
					auto found = false;
					for (const auto& slot : images)
					{
						if (slot.hash == t.type_hash)
						{
							image = slot.image;
							found = true;
						}
					}
					if (!found)
					{
						throw std::runtime_error(utils::string::va("techset %s slot 0x%08X has no converted image", donor.techset, t.type_hash));
					}
					ordered_json entry;
					entry["image"] = image;
					entry["semantic"] = t.semantic;
					entry["samplerState"] = t.sampler_state;
					entry["lastCharacter"] = t.last_character;
					entry["firstCharacter"] = t.first_character;
					entry["typeHash"] = t.type_hash;
					textures.push_back(entry);
				}
				j["textureTable"] = textures;

				const auto c_name = clean_name(name);
				const auto str = j.dump(4);
				write_file("materials\\" + c_name + ".json", str.data(), str.size());

				const auto state = "techsets\\state\\"s + donor.techset + "\\" + c_name;
				write_file(state + ".statebits", donor.statebits.data, donor.statebits.size);
				write_file(state + ".statebitsmap", donor.statebitsmap.data, donor.statebitsmap.size);
				if (donor.cbi.data)
				{
					const auto cb = "techsets\\constantbuffer\\"s + donor.techset + "\\" + c_name;
					write_file(cb + ".cbi", donor.cbi.data, donor.cbi.size);
					write_file(cb + ".cbt", donor.cbt.data, donor.cbt.size);
				}
			}

			std::string image_base_name(const std::string& material_name)
			{
				auto base = material_name;
				const auto slash = base.find_last_of('/');
				if (slash != std::string::npos)
				{
					base = base.substr(slash + 1);
				}
				return clean_name(base) + "_t7";
			}

			std::atomic<std::size_t> total_image_bytes{ 0 };
			float underlying_gloss = 0.0f;

			// BO3 fades an emissive with the view angle, pow(N.V, p) or with invertFalloff pow(1 - N.V, p) (p = 0: no falloff);
			// IW7's is view-independent. Sets p to 0 in `lit` and returns the scale that takes its output to the constant
			// radiance emitting the same flux: 2 * integral of f(mu) mu over [0, 1], 2 / (p + 2) or 2 / ((p + 1)(p + 2)).
			float remove_emissive_falloff(material_program& lit)
			{
				const auto* power_var = find_variable(lit, "emissiveFalloffPower");
				const auto power = power_var ? read_float(lit, power_var, 0) : 0.0f;
				if (power <= 0.0f)
				{
					return 1.0f;
				}
				const auto* invert_var = find_variable(lit, "invertFalloff");
				const auto inverted = invert_var && read_bool(lit, invert_var);
				write_float(lit, power_var, 0, 0.0f);
				return inverted ? 2.0f / ((power + 1.0f) * (power + 2.0f)) : 2.0f / (power + 2.0f);
			}

			// BO3's additive emissive (decal_emissive): its lit shader's output with every light, probe and fog term at 0 is
			// what it adds, the emission E (times its alpha with a SRC_ALPHA blend). IW7's unlit add mkhdr adds
			// -ln(1 - min(c^2, 0.99)) * colorTint.rgb * c.a of its colour map c, read UNORM (stock light_*_on overlays: BC7 /
			// BC1 UNORM): c = sqrt(1 - exp(-E / T)) with T = the largest E / -ln(0.01), which puts the brightest texel at the
			// clamp. A material that emits nothing at rest keeps a black 4x4 image.
			void write_unlit_emissive(const Material* material, const info& inf, worker& wk, const plan& p,
				const world_techset_donors::techset_donor& donor)
			{
				auto lit = load_program(material, "lit", { nullptr, nullptr, true });
				use_vertex_colour(lit, inf);
				const auto falloff_scale = remove_emissive_falloff(lit);
				const auto alpha_scaled = lit.technique->stateMap && lit.technique->stateMap->blend[0].rgbSrc == 5;

				// made after the constant change above: the GPU takes its copy of the constant buffers when it binds
				evaluator eval(wk, material, lit, {});
				const auto levels = mip_count(p.width, p.height);
				std::vector<std::vector<float>> emission(levels);
				auto peak = 0.0f;
				for (auto level = 0u; level < levels; level++)
				{
					const auto w = std::max(1u, p.width >> level);
					const auto h = std::max(1u, p.height >> level);
					const grid g{ w, h, p.uv_origin[0], p.uv_origin[1], p.uv_span[0], p.uv_span[1], 0 };
					auto& e = emission[level];
					e.resize(static_cast<std::size_t>(w) * h * 3);
					for (const auto& t : tiles_of(w, h))
					{
						const auto out = eval.run(g, t, 1);
						timed _(stage_ns.pack);
						for (auto ly = 0u; ly < t.height; ly++)
						{
							for (auto lx = 0u; lx < t.width; lx++)
							{
								const auto& o = out[static_cast<std::size_t>(ly) * t.width + lx];
								const auto i = static_cast<std::size_t>(t.y + ly) * w + t.x + lx;
								const auto a = o.discarded ? 0.0f : (alpha_scaled ? std::clamp(o.target[0][3], 0.0f, 1.0f) : 1.0f);
								for (auto c = 0u; c < 3; c++)
								{
									const auto v = o.target[0][c];
									if (!std::isfinite(v))
									{
										throw std::runtime_error("the lit shader evaluated to a non-finite emission");
									}
									e[i * 3 + c] = std::max(v, 0.0f) * a * falloff_scale;
									if (level == 0)
									{
										peak = std::max(peak, e[i * 3 + c]);
									}
								}
							}
						}
					}
				}

				constexpr auto clamp_log = 4.60517019f; // -ln(1 - 0.99)
				const auto scale = peak / clamp_log;
				image_levels image{ p.width, p.height };
				if (scale > 0.0f)
				{
					for (const auto& e : emission)
					{
						std::vector<std::uint8_t> rgba(e.size() / 3 * 4);
						for (std::size_t i = 0; i < e.size() / 3; i++)
						{
							for (auto c = 0u; c < 3; c++)
							{
								rgba[i * 4 + c] = to_byte(std::sqrt(1.0f - std::exp(-e[i * 3 + c] / scale)));
							}
							rgba[i * 4 + 3] = 255;
						}
						image.rgba.emplace_back(std::move(rgba));
					}
				}
				else
				{
					image = { 4, 4 };
					for (auto l = 0u; l < mip_count(4, 4); l++)
					{
						const auto size = std::max(1u, 4u >> l);
						std::vector<std::uint8_t> rgba(static_cast<std::size_t>(size) * size * 4, 0);
						for (auto i = 3u; i < rgba.size(); i += 4)
						{
							rgba[i] = 255;
						}
						image.rgba.emplace_back(std::move(rgba));
					}
				}

				const auto& name = inf.name;
				const auto e_name = image_base_name(name) + (inf.model ? "m" : "") + "_e";
				claim_image(e_name, name, false);
				total_image_bytes += write_image(e_name, image, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_COLOR_MAP, 0x300, wk.gpu.get());
				write_material_json(name, donor, inf, iw7_state_flags(donor, lit.technique), { { r_hash_string("colorTint"), { scale * map::bo3_light_scale, scale * map::bo3_light_scale, scale * map::bo3_light_scale, 1.0f } } },
					{ { hash_color_map, e_name } });
				ZONETOOL_INFO("%s material %s: additive emission as IW7's unlit add, peak %.3f (colorTint %.3f)%s", inf.model ? "model" : "world",
					name.data(), peak, scale, peak > 0.0f ? "" : ": emits nothing at rest");
			}

			// BO3's water as IW7's refractive water on the world (w_l_sm_ndw_blend_i0c0s0n0a0tm0pa0_nop_sraf0_osn_trans, stock
			// water_prisoner_murky: both draw the refracted scene under a lit colour of their own, add reflections and move a
			// normal map) and on models as IW7's UV-animated lit water (mco_l_sm_ua_replace_i0c0s0n0p0, stock
			// water_lake_geneva_close_tsunami: an opaque lit colour, reflections, textures scrolled by uvAnimParms a second).
			// What maps:
			// * the texture space: BO3 samples the normal map (and _color's decal map) at uv x normalMapScale, where the
			//   surfaces' texture coordinates are moved (plan uv_span), so IW7's textures tile once a unit;
			// * the colour: lit_water_sim_flow_normal's colorMap is a ramp BO3 reads at (N.V, 0), taken at its mean over N.V;
			//   _color's decalMap x decalTint, which BO3 moves along the flow as it does the normals (here at rest on the world,
			//   scrolled with them on a model); water_shore_flow's deep water colour; F0 0.02 (BO3's water Fresnel
			//   0.02 + 0.98 (1 - N.V)^5);
			// * the opacity of that colour over the refracted scene (world, IW7's _packed_at R): BO3 blends it by opacityScale.x
			//   x the colour's alpha x its depth ramp's (rColorRamp, read at log2(2 depth + 1) / 12) alpha, the last taken where
			//   the ramp ends (deep water); water_shore_flow is opaque (it reads the scene only for its reflections). A model's
			//   water is drawn opaque: IW7's model water has no refraction;
			// * the normal: BO3's n.xy = normal map x a + b (normalAdjustAndGloss.xy, shore_flow normalScale; b = -a / 2 in
			//   every BO3 water, a tilt IW7 cannot draw otherwise), times |flow| x normalsFlowScaleMin.y + .x with
			//   scaleNormalsWithFlow (at the flow map's mean |flow|), z 1: on the world IW7 scales the normal map's octahedral
			//   coordinates, ~ n.xy, by oceanScrollN.x = a / 2; on a model that normal is baked; gloss normalAdjustAndGloss.z
			//   (shore_flow reflectionGloss);
			// * the motion: BO3 moves the normal map along its flow map's directions, (1 - 2 f.x, 2 f.y - 1) per cell, blended,
			//   by flowParams.y a second in normal-map units; IW7 moves its textures one way: the flow map's mean direction, on
			//   the world as two layers at IW7's own stream water's ratios (tiling x 0.55, speed x 0.37: stock
			//   cp_town_stream_water's two oceanScrollN).
			// Not drawn: BO3's depth absorption of the refracted scene (rColorRamp rgb; IW7's lit pass leaves absorptionSc
			// unused), its sun-glint lobes, and foam (useFoam, along the water's intersections).
			void write_water(const Material* material, const info& inf, worker& wk, const world_techset_donors::techset_donor& donor)
			{
				constexpr std::uint32_t hash_scroll0 = 1506292936u; // stock water_prisoner_murky's two oceanScrollN
				constexpr std::uint32_t hash_scroll1 = 1506292937u;
				constexpr std::uint32_t hash_uv_anim = 1894494101u; // stock water_lake_geneva_close_tsunami's uvAnimParms

				const auto lit = load_program(material, "lit", { nullptr, nullptr, true });
				const auto shore = bo3_template(material).starts_with("water_shore_flow");
				const auto value = [&](const char* name, const std::uint32_t component) -> std::optional<float>
				{
					const auto* v = find_variable(lit, name);
					return v ? std::optional<float>(read_float(lit, v, component)) : std::nullopt;
				};
				const auto required = [&](const char* name, const std::uint32_t component)
				{
					const auto v = value(name, component);
					if (!v)
					{
						throw std::runtime_error(utils::string::va("water without %s", name));
					}
					return *v;
				};
				const auto decode = [&](const texture_binding* t)
				{
					std::shared_ptr<const material_texture::decoded> d;
					if (t && t->image)
					{
						timed _(stage_ns.textures);
						d = material_texture::decode(t->image);
					}
					return d;
				};
				// a ramp's value at u (its first row, as BO3 reads it at v 0)
				const auto ramp_at = [](const material_texture::texture& ramp, const float u)
				{
					std::array<float, 4> out;
					ramp.sample(u, 0.0f, 0.0f, 0, 0, out.data());
					return out;
				};

				// the colour: a texture (_color's decal map) times a tint, or a constant; and its alpha
				const texture_binding* colour_binding = nullptr;
				std::array<float, 4> tint = { 1.0f, 1.0f, 1.0f, 1.0f };
				const char* colour_source = "";
				if (shore)
				{
					tint = { required("waterColorDeep", 0), required("waterColorDeep", 1), required("waterColorDeep", 2), 1.0f };
					colour_source = "its deep water colour";
				}
				else if (const auto* ramp_binding = find_binding(lit, "colorMap"))
				{
					const auto ramp = decode(ramp_binding);
					if (!ramp)
					{
						throw std::runtime_error("the water's colour ramp has no pixels");
					}
					const material_texture::texture sampler(ramp, ramp_binding->u, ramp_binding->v);
					constexpr auto samples = 256;
					tint = { 0.0f, 0.0f, 0.0f, 0.0f };
					for (auto i = 0; i < samples; i++)
					{
						const auto texel = ramp_at(sampler, (static_cast<float>(i) + 0.5f) / samples);
						for (auto c = 0u; c < 4; c++)
						{
							tint[c] += texel[c] / samples;
						}
					}
					colour_source = "its colour ramp's mean over N.V";
				}
				else
				{
					colour_binding = find_binding(lit, "decalMap");
					for (auto c = 0u; c < 4; c++)
					{
						tint[c] = required("decalTint", c);
					}
					colour_source = "its decal map x decalTint";
				}
				const auto colour = decode(colour_binding);
				if (colour_binding && !colour)
				{
					throw std::runtime_error("the water's colour map has no pixels");
				}
				const auto normal = decode(find_binding(lit, "normalMap"));
				const auto flow = decode(find_binding(lit, "flowMap"));
				if (!normal || !flow)
				{
					throw std::runtime_error("water without a normal map or a flow map");
				}

				// the colour's weight in deep water: the depth ramp's alpha where it ends
				auto depth_weight = 1.0f;
				std::array<float, 3> ramp_alpha = { 1.0f, 1.0f, 1.0f }; // at the surface, halfway, the end (for the log)
				if (!shore)
				{
					const auto* depth_binding = find_binding(lit, "rColorRamp");
					const auto depth_ramp = decode(depth_binding);
					if (!depth_ramp)
					{
						throw std::runtime_error("water without a depth ramp (rColorRamp)");
					}
					const material_texture::texture sampler(depth_ramp, depth_binding->u, depth_binding->v);
					const auto width = static_cast<float>(depth_ramp->width);
					ramp_alpha = { ramp_at(sampler, 0.5f / width)[3], ramp_at(sampler, 0.5f)[3], ramp_at(sampler, (width - 0.5f) / width)[3] };
					depth_weight = ramp_alpha[2];
				}
				const auto opacity = shore ? 1.0f : std::clamp(required("opacityScale", 0), 0.0f, 1.0f) * depth_weight;

				const auto gloss = std::clamp(shore ? required("reflectionGloss", 0) : required("normalAdjustAndGloss", 2), 0.0f, 1.0f);
				const auto* scale_name = shore ? "normalScale" : "normalAdjustAndGloss";
				const auto scale = required(scale_name, 0);
				const auto offset = required(scale_name, 1);
				if (std::fabs(offset + scale * 0.5f) > 1e-4f * std::max(1.0f, std::fabs(scale)))
				{
					throw std::runtime_error(utils::string::va("water whose normal offset %g is not -%g / 2", offset, scale));
				}
				const auto speed = required("flowParams", 1);

				// the flow map's mean direction and mean length, as the shader decodes it
				double mean[2] = { 0.0, 0.0 };
				double mean_length = 0.0;
				{
					std::uint32_t w, h;
					flow->level_size(0, w, h);
					for (auto y = 0u; y < h; y++)
					{
						for (auto x = 0u; x < w; x++)
						{
							float t4[4];
							flow->fetch(0, x, y, t4);
							const auto fx = 1.0 - 2.0 * t4[0];
							const auto fy = 2.0 * t4[1] - 1.0;
							mean[0] += fx;
							mean[1] += fy;
							mean_length += std::sqrt(fx * fx + fy * fy);
						}
					}
					mean[0] /= static_cast<double>(w) * h;
					mean[1] /= static_cast<double>(w) * h;
					mean_length /= static_cast<double>(w) * h;
				}
				auto flow_scale = 1.0f;
				if (const auto* with_flow = find_variable(lit, "scaleNormalsWithFlow"); with_flow && read_bool(lit, with_flow))
				{
					flow_scale = static_cast<float>(mean_length) * required("normalsFlowScaleMin", 1) + required("normalsFlowScaleMin", 0);
				}
				const auto strength = scale * 0.5f * flow_scale;
				const std::array<float, 2> scroll = { static_cast<float>(mean[0]) * speed, static_cast<float>(mean[1]) * speed };

				const auto levels_of = [&](const std::shared_ptr<const material_texture::decoded>& d, const auto& texel)
				{
					image_levels image{ d ? d->width : 4u, d ? d->height : 4u };
					timed _(stage_ns.pack);
					for (auto l = 0u; l < (d ? d->level_count() : mip_count(4, 4)); l++)
					{
						std::uint32_t w = std::max(1u, 4u >> l), h = std::max(1u, 4u >> l);
						if (d)
						{
							d->level_size(l, w, h);
						}
						std::vector<std::uint8_t> rgba(static_cast<std::size_t>(w) * h * 4);
						for (auto y = 0u; y < h; y++)
						{
							for (auto x = 0u; x < w; x++)
							{
								float t4[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
								if (d)
								{
									d->fetch(l, x, y, t4);
								}
								texel(t4, &rgba[(static_cast<std::size_t>(y) * w + x) * 4]);
							}
						}
						image.rgba.emplace_back(std::move(rgba));
					}
					return image;
				};
				const auto cs_levels = levels_of(colour, [&](const float t4[4], std::uint8_t* out)
				{
					for (auto c = 0u; c < 3; c++)
					{
						out[c] = to_byte(std::sqrt(std::clamp(t4[c] * tint[c], 0.0f, 1.0f)));
					}
					out[3] = to_byte(0.02f);
				});
				// the world's normal map as BO3's normal map (IW7 applies the strength); a model's with BO3's strength
				const auto ng_levels = levels_of(normal, [&](const float t4[4], std::uint8_t* out)
				{
					float n[3] = { t4[0] * 2.0f - 1.0f, t4[1] * 2.0f - 1.0f, 0.0f };
					if (inf.model)
					{
						n[0] *= strength;
						n[1] *= strength;
						n[2] = 1.0f;
					}
					else
					{
						n[2] = std::sqrt(std::max(0.0f, 1.0f - n[0] * n[0] - n[1] * n[1]));
					}
					float g, a;
					octahedral(n, g, a);
					out[0] = to_byte(iw7_gloss(gloss));
					out[1] = to_byte(g);
					out[2] = 255;
					out[3] = to_byte(a);
				});

				const auto& name = inf.name;
				const auto base = image_base_name(name) + (inf.model ? "m" : "");
				const auto cs_name = base + "_packed_cs", ng_name = base + "_packed_ng";
				claim_image(cs_name, name, false);
				claim_image(ng_name, name, false);
				total_image_bytes += write_image(cs_name, cs_levels, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_COLOR_SPECULAR_MAP, 0x2300, wk.gpu.get());
				total_image_bytes += write_image(ng_name, ng_levels, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_NORMAL_OCCLUSSION_GLOSS_MAP, 0x308, wk.gpu.get());
				const std::string colour_text = colour ? colour_source
					: utils::string::va("%s (%.4f, %.4f, %.4f)", colour_source, tint[0], tint[1], tint[2]);
				// the colour's weight over the refracted scene (with a decal map, times its alpha)
				const auto weight = opacity * std::clamp(tint[3], 0.0f, 1.0f);
				const std::string flow_text = utils::string::va("flow (%.3f, %.3f) x %.3f a second, normal strength %.3f (a %.3f / 2 x flow scale %.3f)",
					mean[0], mean[1], speed, strength, scale, flow_scale);

				if (inf.model)
				{
					// how fine the moved texture coordinates stay in the mesh's half floats
					auto largest = 0.0f;
					if (inf.used_uv && !inf.used_uv->empty())
					{
						for (auto axis = 0; axis < 2; axis++)
						{
							largest = std::max({ largest, std::fabs((inf.used_uv->min[axis] - inf.uv_origin[axis]) / inf.uv_span[axis]),
								std::fabs((inf.used_uv->max[axis] - inf.uv_origin[axis]) / inf.uv_span[axis]) });
						}
					}
					const auto half_step = largest >= std::ldexp(1.0f, -14) ? std::ldexp(1.0f, static_cast<int>(std::floor(std::log2(largest))) - 10) : 0.0f;
					write_material_json(name, donor, inf, iw7_state_flags(donor, lit.technique),
						{ { hash_uv_anim, { scroll[0], scroll[1], 0.0f, 0.0f } }, { r_hash_string("colorTint"), { 1.0f, 1.0f, 1.0f, 1.0f } } },
						{ { hash_color_map, cs_name }, { hash_normal_map, ng_name } });
					ZONETOOL_INFO("model material %s: BO3's water as IW7's UV-animated lit water, opaque (BO3 blends its colour %.2f%s over "
						"the refracted scene; the depth ramp's alpha %.2f / %.2f / %.2f); colour %s, gloss %.2f, %s; texture coordinates x (%.3f, %.3f), "
						"up to %.2f normal-map periods (half-float step %.5f)", name.data(), weight, colour ? " x the decal map's alpha" : "",
						ramp_alpha[0], ramp_alpha[1], ramp_alpha[2],
						colour_text.data(), gloss, flow_text.data(), 1.0f / inf.uv_span[0], 1.0f / inf.uv_span[1], largest, half_step);
					return;
				}

				const auto at_levels = levels_of(colour, [&](const float t4[4], std::uint8_t* out)
				{
					out[0] = to_byte(opacity * std::clamp(t4[3] * tint[3], 0.0f, 1.0f));
					out[1] = 255;
					out[2] = 255; // thickness 1: IW7's refraction ray is subsurfacePa.x long
					out[3] = 255;
				});
				const auto at_name = base + "_packed_at";
				claim_image(at_name, name, false);
				total_image_bytes += write_image(at_name, at_levels, DXGI_FORMAT_BC1_UNORM, zonetool::iw7::TS_ALPHA_REVEAL_THICKNESS_MAP, 0x300, wk.gpu.get());
				const std::array<float, 4> scroll0 = { strength, 1.0f, scroll[0], scroll[1] };
				const std::array<float, 4> scroll1 = { strength, 0.55f, scroll[0] * 0.37f, scroll[1] * 0.37f };
				// at the donor's sort key, the world's sortKeyDistortion (24, as stock water_prisoner_murky): IW7 copies the
				// scene for the refraction before that key. At the general transparent key (26) it would read a
				// stale scene copy
				auto water_inf = inf;
				water_inf.sort_key = donor.sort_key;
				write_material_json(name, donor, water_inf, iw7_state_flags(donor, lit.technique),
					{ { hash_scroll0, scroll0 }, { hash_scroll1, scroll1 }, { r_hash_string("colorTint"), { 1.0f, 1.0f, 1.0f, 1.0f } } },
					{ { hash_color_map, cs_name }, { hash_normal_map, ng_name }, { hash_spec_occlusion_map, at_name } });
				ZONETOOL_INFO("world material %s: BO3's water as IW7's refractive water (opacity %.2f%s; the depth ramp's alpha %.2f / %.2f / %.2f); "
					"colour %s, gloss %.2f, %s; texture coordinates x (%.3f, %.3f)", name.data(), weight, colour ? " x the decal map's alpha" : "",
					ramp_alpha[0], ramp_alpha[1], ramp_alpha[2], colour_text.data(), gloss, flow_text.data(), 1.0f / inf.uv_span[0], 1.0f / inf.uv_span[1]);
			}

			// BO3's multiply decal as IW7's unlit multiply (plan multiply). IW7 multiplies what is under it by
			// lerp(1, (c x vertex rgb)^2, vertex alpha) of its colour map c (wc_unlit_multiply_lin_ndw's pixel shader; the
			// vertex rgb is written white), so c = sqrt of BO3's RT0, the albedo multiplier, evaluated at vertex alpha 1; a
			// decal that does not write RT0 multiplies by 1. IW7's multiply only darkens: RT0 above 1 is clamped.
			void write_multiply(const Material* material, const info& inf, worker& wk, const plan& p,
				const world_techset_donors::techset_donor& donor)
			{
				auto mp = load_program(material, "gbuffer");
				image_levels image{ p.width, p.height };
				const auto levels = mip_count(p.width, p.height);
				auto clamped = 0ull;
				auto darkest = 1.0f;
				if (p.identity)
				{
					for (auto l = 0u; l < levels; l++)
					{
						const auto w = std::max(1u, p.width >> l);
						const auto h = std::max(1u, p.height >> l);
						image.rgba.emplace_back(static_cast<std::size_t>(w) * h * 4, static_cast<std::uint8_t>(255));
					}
				}
				else
				{
					evaluator eval(wk, material, mp, {});
					for (auto l = 0u; l < levels; l++)
					{
						const auto w = std::max(1u, p.width >> l);
						const auto h = std::max(1u, p.height >> l);
						const grid g{ w, h, p.uv_origin[0], p.uv_origin[1], p.uv_span[0], p.uv_span[1], 0 };
						std::vector<std::uint8_t> rgba(static_cast<std::size_t>(w) * h * 4, 255);
						for (const auto& t : tiles_of(w, h))
						{
							const auto out = eval.run(g, t, 1);
							timed _(stage_ns.pack);
							for (auto ly = 0u; ly < t.height; ly++)
							{
								for (auto lx = 0u; lx < t.width; lx++)
								{
									const auto& o = out[static_cast<std::size_t>(ly) * t.width + lx];
									if (o.discarded)
									{
										continue; // nothing drawn: a multiply by 1
									}
									const auto i = (static_cast<std::size_t>(t.y + ly) * w + t.x + lx) * 4;
									for (auto c = 0u; c < 3; c++)
									{
										const auto v = o.target[0][c];
										if (!std::isfinite(v))
										{
											throw std::runtime_error("the decal evaluated to a non-finite multiplier");
										}
										clamped += v > 1.0f ? 1u : 0u;
										darkest = std::min(darkest, v);
										rgba[i + c] = to_byte(std::sqrt(std::clamp(v, 0.0f, 1.0f)));
									}
								}
							}
						}
						image.rgba.emplace_back(std::move(rgba));
					}
				}

				const auto& name = inf.name;
				const auto c_name = image_base_name(name) + (inf.model ? "m" : "") + "_mul";
				claim_image(c_name, name, false);
				total_image_bytes += write_image(c_name, image, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_COLOR_MAP, 0x300, wk.gpu.get());
				write_material_json(name, donor, inf, iw7_state_flags(donor, mp.technique), {}, { { hash_color_map, c_name } });
				ZONETOOL_INFO("%s material %s: %s%s", inf.model ? "model" : "world", name.data(),
					p.identity ? "a multiply by 1" : utils::string::va("an unlit multiply, darkest %.3f", darkest),
					clamped ? utils::string::va(" (%llu texel channels above 1 clamped)", clamped) : "");
			}

			// BO3's eye as IW7's (mo_l_sm_replace_i0c0s0_sss_eye_gtao, every stock head's eye): the same shader family (cornea
			// refraction 0.726957, the limbus sigmoid 1.01358 / 0.0067, the iris radius from an iris scale, BO3 mc/eye against
			// IW7's lit technique), IW7's with its own inner constants. eyeShaderParams0 = (eyeIrisScale, the limbus width BO3
			// makes of eyeLimbusScale x 0.05 + 0.02, eyeRedness, 0); eyeShaderParams1.x = the pupil radius BO3 takes from its
			// script constant (0 at rest: 0.001 x 0.22) in IW7's units (x 0.2), the rest stock. The colour map is BO3's, stored
			// as IW7's eye reads it (it squares the texel: sqrt of linear, UNORM); the sclera specular / gloss and irradiance
			// maps are the stock ones every IW7 eye uses (BO3's has neither; its detail normal is left out).
			void write_eye(const Material* material, const info& inf, worker& wk, const world_techset_donors::techset_donor& donor)
			{
				constexpr std::uint32_t hash_eye_params0 = 3001795517u; // stock head_zmb_dj_eyes' two eyeShaderParams
				constexpr std::uint32_t hash_eye_params1 = 3001795518u;
				constexpr std::uint32_t hash_eye_irradiance = 654214108u;
				constexpr std::uint32_t hash_eye_specular_gloss = 887934131u;

				const auto lit = load_program(material, "lit", { nullptr, nullptr, true });
				const auto* colour = find_binding(lit, "colorMap");
				const auto* iris = find_variable(lit, "eyeIrisScale");
				const auto* limbus = find_variable(lit, "eyeLimbusScale");
				const auto* redness = find_variable(lit, "eyeRedness");
				if (!colour || !colour->image || !iris || !limbus || !redness)
				{
					throw std::runtime_error("eye without a colour map or its iris constants");
				}
				std::shared_ptr<const material_texture::decoded> decoded;
				{
					timed _(stage_ns.textures);
					decoded = material_texture::decode(colour->image);
				}
				if (!decoded)
				{
					throw std::runtime_error("the eye's colour map has no pixels");
				}
				image_levels image{ decoded->width, decoded->height };
				{
					timed _(stage_ns.pack);
					for (auto l = 0u; l < decoded->level_count(); l++)
					{
						std::uint32_t w, h;
						decoded->level_size(l, w, h);
						std::vector<std::uint8_t> level(static_cast<std::size_t>(w) * h * 4);
						for (auto y = 0u; y < h; y++)
						{
							for (auto x = 0u; x < w; x++)
							{
								float t4[4];
								decoded->fetch(l, x, y, t4);
								const auto i = (static_cast<std::size_t>(y) * w + x) * 4;
								for (auto c = 0u; c < 3; c++)
								{
									level[i + c] = to_byte(std::sqrt(std::clamp(t4[c], 0.0f, 1.0f)));
								}
								level[i + 3] = 255;
							}
						}
						image.rgba.emplace_back(std::move(level));
					}
				}

				const auto& name = inf.name;
				const auto c_name = image_base_name(name) + "m_eye_c";
				claim_image(c_name, name, false);
				total_image_bytes += write_image(c_name, image, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_COLOR_MAP, 0x300, wk.gpu.get());

				const auto iris_scale = read_float(lit, iris, 0);
				const auto limbus_width = read_float(lit, limbus, 0) * 0.05f + 0.02f;
				const auto red = read_float(lit, redness, 0);
				const auto pupil = 0.001f * 0.22f / 0.2f;
				std::array<float, 4> params1 = { pupil, 0.05f, 0.3f, 0.95f };
				for (auto i = 0u; i < donor.constant_count; i++)
				{
					if (donor.constants[i].hash == hash_eye_params1)
					{
						params1 = { pupil, donor.constants[i].literal[1], donor.constants[i].literal[2], donor.constants[i].literal[3] };
					}
				}
				write_material_json(name, donor, inf, iw7_state_flags(donor, lit.technique),
					{ { hash_eye_params0, { iris_scale, limbus_width, red, 0.0f } }, { hash_eye_params1, params1 } },
					{ { hash_color_map, c_name }, { hash_normal_map, "$identitynormalmap" }, { hash_eye_irradiance, "eye_irradiance_m" },
						{ hash_eye_specular_gloss, "eye_physicallybased_sg" } });
				ZONETOOL_INFO("model material %s: BO3's eye as IW7's eye (iris scale %.3f, limbus width %.3f, redness %.3f)", name.data(),
					iris_scale, limbus_width, red);
			}

			// the IW7 cs and ng texels of a decoded BO3 gbuffer texel (the pack shader in gpu_eval.cpp does the same);
			// true when the texel is BO3's backlit shading model
			bool pack_texel(const plan& p, surface& s, std::uint8_t cs_texel[4], std::uint8_t ng_texel[4])
			{
				// channels a decal leaves to the surface under it
				if (!p.writes_normal)
				{
					s.normal[0] = 0.0f;
					s.normal[1] = 0.0f;
					s.normal[2] = 1.0f;
				}
				if (!p.writes_gloss)
				{
					s.gloss = underlying_gloss;
				}
				if (!p.writes_specular)
				{
					s.f0[0] = s.f0[1] = s.f0[2] = 0.04f;
				}
				if (!p.writes_occlusion)
				{
					s.occlusion = 1.0f;
				}
				// BO3's backlit shading model keeps its scatter colour where F0 would be and
				// lights with the dielectric 0.04
				auto backlit = false;
				if (!p.decal && std::fabs(s.shading_model - 2.0f / 3.0f) < 0.01f)
				{
					s.f0[0] = s.f0[1] = s.f0[2] = 0.04f;
					backlit = true;
				}

				pack_colour(s.albedo, s.f0, cs_texel);

				float g_ga, a_ga;
				octahedral(s.normal, g_ga, a_ga);
				ng_texel[0] = to_byte(iw7_gloss(s.gloss));
				ng_texel[1] = to_byte(g_ga);
				ng_texel[2] = p.occlusion ? to_byte(s.occlusion) : 255;
				ng_texel[3] = to_byte(a_ga);
				return backlit;
			}

			gpu_eval::pack_options pack_options_of(const plan& p)
			{
				gpu_eval::pack_options o{};
				o.raw = p.surface;
				o.decal = p.decal;
				o.writes_normal = p.writes_normal;
				o.writes_gloss = p.writes_gloss;
				o.writes_specular = p.writes_specular;
				o.writes_occlusion = p.writes_occlusion;
				o.occlusion = p.occlusion;
				o.coloured_specular = p.coloured_specular;
				o.underlying_gloss = underlying_gloss;
				return o;
			}

			// The first tile a worker's GPU packs is packed on the CPU too, from the same GPU targets: its top-left 8x8
			// texels. Bytes may differ by one where the GPU's float rounding (square roots, divisions) moves a value
			// across a byte boundary; anything more is a difference between the two packers.
			bool packing_agrees(evaluator& eval, gpu_eval::context& gpu, const plan& p, const grid& g, const tile& t, const std::string& material)
			{
				const auto block_w = std::min(8u, t.width);
				const auto block_h = std::min(8u, t.height);
				gpu.begin_level(t.x + t.width, t.y + t.height, p.decal || p.surface); // (the raw pack writes a surface's alpha there)
				eval.draw(g, t, p.coloured_specular);
				gpu.pack({ t.x, t.y, t.width, t.height }, pack_options_of(p));

				const auto read = [&](const std::uint32_t target)
				{
					std::vector<texel_out> block(static_cast<std::size_t>(block_w) * block_h);
					const auto values = gpu.read_target(target, { 0, 0, block_w, block_h });
					for (std::size_t i = 0; i < block.size(); i++)
					{
						std::memcpy(block[i].target[target == 3 ? 2 : target], &values[i * 4], sizeof(block[i].target[0]));
					}
					return block;
				};
				auto same = read(0);
				for (auto rt = 1u; rt < 3; rt++)
				{
					const auto more = read(rt);
					for (std::size_t i = 0; i < same.size(); i++)
					{
						std::memcpy(same[i].target[rt], more[i].target[rt], sizeof(same[i].target[rt]));
					}
				}
				std::vector<texel_out> other;
				if (p.coloured_specular)
				{
					other = read(3);
				}
				auto surf = p.surface ? decode_surface(same)
					: decode_gbuffer(same, p.coloured_specular ? &other : nullptr, { t.x, t.y, block_w, block_h }, p.decal);

				const auto gpu_cs = gpu.read_packed(0, { t.x, t.y, block_w, block_h });
				const auto gpu_ng = gpu.read_packed(1, { t.x, t.y, block_w, block_h });
				// a surface's alpha byte, which the pack shader writes to the coverage texture
				std::vector<std::uint8_t> gpu_alpha;
				if (p.surface_coverage())
				{
					gpu_alpha = gpu.read_covered();
				}
				auto largest = 0;
				auto differing = 0u;
				std::string where;
				for (std::size_t i = 0; i < surf.size(); i++)
				{
					std::uint8_t cpu_cs[4], cpu_ng[4];
					pack_texel(p, surf[i], cpu_cs, cpu_ng);
					auto texel_differs = false;
					if (p.surface_coverage())
					{
						const auto at = static_cast<std::size_t>(t.y + i / block_w) * (t.x + t.width) + t.x + i % block_w;
						const auto da = std::abs(static_cast<int>(to_byte(surf[i].coverage[0])) - gpu_alpha[at]);
						texel_differs |= da != 0;
						if (da > largest)
						{
							largest = da;
							where = utils::string::va("texel %zu,%zu: CPU alpha %u, GPU alpha %u", t.x + i % block_w, t.y + i / block_w,
								to_byte(surf[i].coverage[0]), gpu_alpha[at]);
						}
					}
					for (auto c = 0u; c < 4; c++)
					{
						const auto dc = std::abs(static_cast<int>(cpu_cs[c]) - gpu_cs[i * 4 + c]);
						const auto dn = std::abs(static_cast<int>(cpu_ng[c]) - gpu_ng[i * 4 + c]);
						texel_differs |= dc != 0 || dn != 0;
						if (std::max(dc, dn) > largest)
						{
							largest = std::max(dc, dn);
							where = utils::string::va("texel %zu,%zu: CPU cs %u %u %u %u ng %u %u %u %u, GPU cs %u %u %u %u ng %u %u %u %u",
								t.x + i % block_w, t.y + i / block_w, cpu_cs[0], cpu_cs[1], cpu_cs[2], cpu_cs[3], cpu_ng[0], cpu_ng[1], cpu_ng[2],
								cpu_ng[3], gpu_cs[i * 4], gpu_cs[i * 4 + 1], gpu_cs[i * 4 + 2], gpu_cs[i * 4 + 3], gpu_ng[i * 4], gpu_ng[i * 4 + 1],
								gpu_ng[i * 4 + 2], gpu_ng[i * 4 + 3]);
						}
					}
					differing += texel_differs ? 1 : 0;
				}
				const auto agree = largest <= 1;
				ZONETOOL_INFO("material bake: GPU packing %s the CPU on material %s (%u of %zu texels differ, largest byte difference %d%s%s)",
					agree ? "agrees with" : "differs from", material.data(), differing, surf.size(), largest, where.empty() ? "" : "; ",
					where.data());
				return agree;
			}
		}

		std::optional<alpha_mask> alpha_mask_of(const Material* material)
		{
			static std::mutex mutex;
			static std::unordered_map<const Material*, std::optional<alpha_mask>> cache;
			{
				std::lock_guard _(mutex);
				if (const auto found = cache.find(material); found != cache.end())
				{
					return found->second;
				}
			}
			std::optional<alpha_mask> result;
			try
			{
				// what reaches the gbuffer program's discard (coverage_source, as the material bake reads it)
				const auto mp = load_program(material, "gbuffer");
				if (mp.taint.discard_textures.any())
				{
					const auto [binding, channel] = coverage_source(mp, true, false);
					if (binding && binding->image)
					{
						// where the shader samples it (make_plan's probe: one quad at the origin of a 4096 grid)
						std::unordered_map<std::string, std::array<float, 4>> constants_everywhere;
						for (const auto& t : mp.textures)
						{
							constants_everywhere[t.name] = { 0.5f, 0.5f, 0.5f, 1.0f };
						}
						auto bound = bind_textures(mp, constants_everywhere);
						shader_eval::machine m(mp.program);
						m.set_unbound_resources_zero(mp.forward);
						for (const auto& [slot, data] : mp.cbuffers)
						{
							m.bind_cbuffer(slot, data.data(), data.size());
						}
						for (const auto& [slot, tex] : bound.slots)
						{
							m.bind_texture(slot, tex);
						}
						for (const auto& b : mp.buffers)
						{
							m.bind_structured(b.slot, b.data.data(), b.data.size(), b.stride);
						}
						const grid probe_grid{ 4096, 4096, 0.0f, 0.0f, 1.0f, 1.0f, 0 };
						set_inputs(m, mp, probe_grid, 0, 0);
						m.enable_probe(true);
						m.run();
						std::vector<std::pair<const material_program*, shader_eval::sample_probe>> probes;
						for (const auto& probe : m.probes())
						{
							probes.emplace_back(&mp, probe);
						}
						const auto map = sample_map_of(probes, &mp, *binding);
						if (auto image = material_texture::decode(binding->image))
						{
							result = alpha_mask{ std::move(image), channel, binding->u, binding->v, map };
						}
					}
				}
			}
			catch (const std::exception&)
			{
				// no gbuffer program (forward materials) or coverage from more than one texture: no mask
			}
			std::lock_guard _(mutex);
			return cache.emplace(material, std::move(result)).first->second;
		}

		std::string bo3_template(const Material* material)
		{
			if (!material || !readable(material->techniqueSet, sizeof(MaterialTechniqueSet)) || !material->techniqueSet->name)
			{
				return {};
			}
			std::string name = material->techniqueSet->name;
			if (const auto hash = name.find('#'); hash != std::string::npos)
			{
				name.resize(hash);
			}
			if (const auto slash = name.rfind('/'); slash != std::string::npos)
			{
				name.erase(0, slash + 1);
			}
			return name;
		}

		bool additive_decal(const Material* material)
		{
			if (!material || !readable(material->techniqueSet, sizeof(MaterialTechniqueSet)))
			{
				return false;
			}
			// renderFlags 2: drawn in the gbuffer, 8: decal (world_material.cpp's classification)
			const auto* ts = material->techniqueSet;
			if ((ts->renderFlags & 2) || !(ts->renderFlags & 8))
			{
				return false;
			}
			for (const auto* technique : ts->techniques)
			{
				if (technique && technique_name(technique) == "lit")
				{
					return adds(technique) && technique->stateMap->primitive.polygonOffset != 0;
				}
			}
			return false;
		}

		const plan& get_plan(const Material* material, const info& inf)
		{
			const std::pair key{ material, inf.model };
			{
				std::lock_guard _(plan_mutex);
				const auto found = plans.find(key);
				if (found != plans.end())
				{
					return found->second;
				}
			}
			// a shader the interpreter cannot run is a material this converter does not handle yet, not a failed dump
			plan p{};
			try
			{
				p = make_plan(material, inf);
			}
			catch (const std::exception& e)
			{
				p = {};
				p.reason = e.what();
			}
			std::lock_guard _(plan_mutex);
			return plans.emplace(key, std::move(p)).first->second;
		}

		void clear()
		{
			std::lock_guard _(plan_mutex);
			plans.clear();
			programs.clear();
			material_texture::clear_cache();
			total_image_bytes = 0;
			{
				std::lock_guard written(written_mutex);
				written_images.clear();
			}
			{
				std::lock_guard atlas(atlas_mutex);
				atlas_members.clear();
				atlases.clear();
			}
			std::lock_guard checked(gpu_programs_mutex);
			gpu_programs.clear();
		}

		bool atlas_tile(const Material* material, const info& inf, std::uint32_t& width, std::uint32_t& height)
		{
			if (!inf.model || inf.cls == surface_class::shadow_only || !inf.used_uv || inf.used_uv->empty())
			{
				return false;
			}
			const auto& p = get_plan(material, inf);
			// the plain gbuffer bake (cs + ng alone): no coverage, emissive, detail (its tiling follows the texture
			// coordinates) or other layout
			if (!p.supported || p.unlit_emissive || p.eye || p.multiply || p.water || p.surface || p.forward_emissive || p.decal
				|| p.alpha_test || p.reveal || p.reveal_decal || p.detail || p.width > 2048 || p.height > 2048)
			{
				return false;
			}
			// every surface inside the baked area (half a texel over at most: bilinear filtering's reach)
			for (auto axis = 0; axis < 2; axis++)
			{
				const auto texels = static_cast<double>(axis == 0 ? p.width : p.height);
				const auto slack = 0.5 * p.uv_span[axis] / texels;
				if (inf.used_uv->min[axis] < p.uv_origin[axis] - slack || inf.used_uv->max[axis] > p.uv_origin[axis] + p.uv_span[axis] + slack)
				{
					return false;
				}
			}
			// up to the next of 2^n and 1.5 x 2^n: few tile sizes, and a tile's first levels stay whole BC7 blocks
			const auto round_up = [](const std::uint32_t size)
			{
				auto step = 4u;
				while (step < size && step + step / 2 < size)
				{
					step *= 2;
				}
				return step >= size ? step : step + step / 2;
			};
			width = round_up(p.width);
			height = round_up(p.height);
			return true;
		}

		void set_atlas(const Material* material, const info& inf, const atlas_slot& slot)
		{
			{
				std::lock_guard _(plan_mutex);
				auto& p = plans.at({ material, inf.model });
				p.width = slot.tile_width;
				p.height = slot.tile_height;
			}
			std::lock_guard _(atlas_mutex);
			atlas_members[material] = slot;
			auto& a = atlases[slot.cs];
			if (!a.expected)
			{
				a.layout = slot;
				a.cs.resize(static_cast<std::size_t>(slot.columns) * slot.rows);
				a.ng.resize(a.cs.size());
			}
			a.expected++;
		}

		void flush_atlases()
		{
			std::vector<atlas_pending> left;
			{
				std::lock_guard _(atlas_mutex);
				for (auto& [name, a] : atlases)
				{
					if (!a.written && a.received)
					{
						ZONETOOL_WARNING("atlas %s: %u of %u materials baked into it (the rest were not converted)", name.data(), a.received,
							a.expected);
						left.emplace_back(take_atlas(a));
					}
				}
			}
			for (const auto& a : left)
			{
				total_image_bytes += write_atlas(a);
			}
		}

		void write_stock_material(const std::string& name, const std::string& techset,
			const std::vector<std::pair<std::uint32_t, std::string>>& images)
		{
			const auto* donor = world_techset_donors::find(techset);
			if (!donor)
			{
				throw std::runtime_error(utils::string::va("no stock donor material for techset %s", techset.data()));
			}
			info inf{};
			inf.name = name;
			// a shadow-only donor (camera region 11, mo_shadowcaster) keeps its casting flag, which an opaque one gets from inf
			inf.cls = donor->camera_region == 11 ? surface_class::shadow_only : surface_class::opaque;
			inf.sort_key = donor->sort_key;
			inf.camera_region = donor->camera_region;
			inf.casts_shadow = false;
			inf.lightmapped = false;
			std::vector<iw7_texture_slot> slots;
			for (const auto& [hash, image] : images)
			{
				slots.push_back({ hash, image });
			}
			write_material_json(name, *donor, inf, donor->state_flags, {}, slots);
		}

		effect_globals read_effect_globals(const Material* material)
		{
			effect_globals g{};
			const auto lit = load_program(material, "lit", { .forward = true, .constants_only = true });
			const auto value = [&](const char* name, const std::uint32_t component, const float fallback)
			{
				const auto* v = find_variable(lit, name);
				return v ? read_float(lit, v, component) : fallback;
			};
			g.hdr_scale = value("hdrScale", 0, g.hdr_scale);
			if (const auto* v = find_variable(lit, "useOldHDRScale"))
			{
				g.old_hdr_scale = read_bool(lit, v);
			}
			g.desaturation = value("desaturationAmount", 0, g.desaturation);
			for (auto c = 0u; c < 4; c++)
			{
				g.levels[c] = value("levelsControls", c, g.levels[c]);
			}
			for (auto c = 0u; c < 2; c++)
			{
				g.distortion_scale[c] = value("distortionScale", c, g.distortion_scale[c]);
			}
			return g;
		}

		namespace
		{
			float srgb_decode(const float c)
			{
				return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
			}

			std::uint8_t unit_byte(const float v)
			{
				return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
			}

			// The BO3 effect material's colour at one texel as its lit shaders compute it before lighting (ei/effect_lit_*,
			// dedicated server shader disassembly): the premultiplied sample desaturated, then, unless levelsControls is
			// (0, 1, 0, 1), unpremultiplied, levelled, decoded with the sRGB curve and premultiplied again. Returns the
			// unpremultiplied colour; alpha is the sample's.
			std::array<float, 3> effect_colour(const std::uint8_t* texel, const bool srgb, const effect_globals& g)
			{
				const auto a = texel[3] / 255.0f;
				std::array<float, 3> p{};
				for (auto c = 0; c < 3; c++)
				{
					const auto v = texel[c] / 255.0f;
					p[c] = srgb ? srgb_decode(v) : v;
				}
				const auto luma = p[0] * 0.2126f + p[1] * 0.7152f + p[2] * 0.0722f;
				for (auto& v : p)
				{
					v += g.desaturation * (luma - v);
				}
				const auto levelled = !(g.levels[0] == 0.0f && g.levels[1] == 1.0f && g.levels[2] == 0.0f && g.levels[3] == 1.0f);
				const auto alpha = std::max(a, 0.0001f);
				for (auto& v : p)
				{
					auto c = std::clamp(v / alpha, 0.0f, 1.0f);
					if (levelled)
					{
						c = std::clamp(c - g.levels[0], 0.0f, 1.0f) / std::max(g.levels[1] - g.levels[0], 0.0001f);
						c = std::min(c, 1.0f) * (g.levels[3] - g.levels[2]) + g.levels[2];
						c = srgb_decode(c);
					}
					v = c;
				}
				return p;
			}

			// one level of an effect image (8-bit RGBA) from the BO3 colour map's level `level`, the emissive mask sampled
			// where each of its texels lies
			std::vector<std::uint8_t> effect_level(const effect_texel texel, const material_texture::decoded& colour, const std::uint32_t level,
				const material_texture::texture* mask, const effect_globals& g)
			{
				std::uint32_t w, h;
				colour.level_size(level, w, h);
				const auto& source = colour.levels8[level];
				std::vector<std::uint8_t> out(static_cast<std::size_t>(w) * h * 4);

				float mask_lod = 0.0f;
				if (mask)
				{
					std::uint32_t mw, mh, ml;
					mask->dimensions(0, mw, mh, ml);
					mask_lod = std::max(0.0f, std::log2(static_cast<float>(mw) / static_cast<float>(w)));
				}
				const auto mask_at = [&](const std::uint32_t x, const std::uint32_t y)
				{
					if (!mask)
					{
						return 1.0f;
					}
					float m[4];
					mask->sample((x + 0.5f) / w, (y + 0.5f) / h, mask_lod, 0, 0, m);
					return std::clamp(m[0], 0.0f, 1.0f);
				};

				parallel_for(h, [&](const std::uint32_t y, std::uint32_t)
				{
					for (auto x = 0u; x < w; x++)
					{
						const auto i = static_cast<std::size_t>(y) * w + x;
						const auto* in = &source[i * 4];
						auto* o = &out[i * 4];
						switch (texel)
						{
						case effect_texel::colour:
						case effect_texel::emissive_colour:
						{
							const auto c = effect_colour(in, colour.srgb, g);
							const auto m = texel == effect_texel::emissive_colour ? mask_at(x, y) : 1.0f;
							for (auto k = 0; k < 3; k++)
							{
								o[k] = unit_byte(std::sqrt(c[k] * m));
							}
							o[3] = texel == effect_texel::colour ? in[3] : 255;
							break;
						}
						case effect_texel::emissive_mask:
						{
							const auto m = unit_byte(mask_at(x, y) * (in[3] / 255.0f));
							o[0] = o[1] = o[2] = m;
							o[3] = 255;
							break;
						}
						case effect_texel::distortion:
							for (auto k = 0; k < 3; k++)
							{
								o[k] = colour.srgb ? unit_byte(srgb_decode(in[k] / 255.0f)) : in[k];
							}
							o[3] = in[3];
							break;
						}
					}
				});

				// the colour of a texel BO3 draws fully transparent is undefined (premultiplied to 0); filtering at the
				// edge of what it draws would blend it in
				if (texel == effect_texel::colour || texel == effect_texel::emissive_colour)
				{
					std::vector<std::uint8_t> alpha(static_cast<std::size_t>(w) * h);
					std::vector<bool> covered(alpha.size());
					for (std::size_t i = 0; i < alpha.size(); i++)
					{
						alpha[i] = out[i * 4 + 3];
						covered[i] = source[i * 4 + 3] != 0;
					}
					constexpr std::uint8_t neutral[4] = { 0, 0, 0, 0 };
					fill_uncovered(out, covered, w, h, neutral);
					for (std::size_t i = 0; i < alpha.size(); i++)
					{
						out[i * 4 + 3] = alpha[i];
					}
				}
				return out;
			}

			// An alpha test on a mip chain loses what it cuts out as the levels get smaller: a hole's texels average with
			// the cloth around them and pass the 0.5 test again. Each level below the first has its alpha (channel 0)
			// scaled so the share of texels that pass matches the first level's, which keeps the holes' area.
			void preserve_coverage(image_levels& levels)
			{
				if (levels.rgba.size() < 2)
				{
					return;
				}
				const auto passing = [](const std::vector<std::uint8_t>& rgba, const double scale)
				{
					std::size_t pass = 0;
					const auto count = rgba.size() / 4;
					for (std::size_t i = 0; i < count; i++)
					{
						if (rgba[i * 4] * scale >= 127.5)
						{
							pass++;
						}
					}
					return static_cast<double>(pass) / static_cast<double>(std::max<std::size_t>(count, 1));
				};
				const auto target = passing(levels.rgba[0], 1.0);
				for (auto l = 1u; l < levels.rgba.size(); l++)
				{
					auto& rgba = levels.rgba[l];
					// the share passing grows with the scale: bisect for the scale that matches the first level's
					auto lo = 0.0, hi = 64.0;
					for (auto step = 0; step < 32; step++)
					{
						const auto mid = (lo + hi) * 0.5;
						(passing(rgba, mid) < target ? lo : hi) = mid;
					}
					const auto scale = hi;
					for (std::size_t i = 0; i < rgba.size(); i += 4)
					{
						rgba[i] = static_cast<std::uint8_t>(std::clamp(std::lround(rgba[i] * scale), 0l, 255l));
					}
				}
			}

			// a 2x2 box average of an 8-bit RGBA level (for the levels of the chain BO3's image does not carry)
			std::vector<std::uint8_t> halve(const std::vector<std::uint8_t>& rgba, const std::uint32_t w, const std::uint32_t h)
			{
				const auto hw = std::max(1u, w / 2), hh = std::max(1u, h / 2);
				std::vector<std::uint8_t> out(static_cast<std::size_t>(hw) * hh * 4);
				for (auto y = 0u; y < hh; y++)
				{
					for (auto x = 0u; x < hw; x++)
					{
						for (auto c = 0; c < 4; c++)
						{
							std::uint32_t sum = 0, n = 0;
							for (auto oy = 0u; oy < 2; oy++)
							{
								for (auto ox = 0u; ox < 2; ox++)
								{
									const auto sx = std::min(w - 1, x * 2 + ox), sy = std::min(h - 1, y * 2 + oy);
									sum += rgba[(static_cast<std::size_t>(sy) * w + sx) * 4 + c];
									n++;
								}
							}
							out[(static_cast<std::size_t>(y) * hw + x) * 4 + c] = static_cast<std::uint8_t>((sum + n / 2) / n);
						}
					}
				}
				return out;
			}
		}

		void write_effect_material(const effect_material_def& def, worker& wk)
		{
			const auto* donor = world_techset_donors::find(def.techset);
			if (!donor)
			{
				throw std::runtime_error(utils::string::va("no stock donor material for techset %s", def.techset.data()));
			}
			const auto* material = def.source;
			const auto globals = read_effect_globals(material);

			const auto image_of = [&](const std::uint32_t hash) -> const GfxImage*
			{
				const auto* t = find_texture(material, hash);
				return t && readable(t->image, sizeof(GfxImage)) ? t->image : nullptr;
			};
			const auto* colour_image = image_of(hash_color_map);
			if (!colour_image)
			{
				throw std::runtime_error(utils::string::va("effect material %s has no colour map", material->name));
			}
			const auto colour = material_texture::decode(colour_image);
			if (!colour || colour->is_float)
			{
				throw std::runtime_error(utils::string::va("effect material %s: colour map %s does not decode", material->name,
					colour_image->name));
			}
			std::shared_ptr<const material_texture::decoded> mask_image;
			if (const auto* e = image_of(hash_emissive_map))
			{
				mask_image = material_texture::decode(e);
			}
			std::optional<material_texture::texture> mask;
			if (mask_image && !mask_image->is_float)
			{
				mask.emplace(mask_image, material_texture::address_mode::wrap, material_texture::address_mode::wrap);
			}

			// a range of the BO3 atlas (def.frames, a power of two of them) as its own atlas, as square as it goes, columns
			// first; frame n sits at column n % columns, row n / columns, in BO3's atlases and IW7's
			const auto sub_atlas = !def.frames.empty();
			auto sub_columns = 1u, sub_rows = 1u;
			if (sub_atlas)
			{
				auto bits = 0u;
				while ((std::size_t{ 1 } << bits) < def.frames.size())
				{
					bits++;
				}
				sub_columns = 1u << ((bits + 1) / 2);
				sub_rows = 1u << (bits / 2);
				if (colour->width % def.source_columns || colour->height % def.source_rows)
				{
					throw std::runtime_error(utils::string::va("effect material %s: colour map %ux%u is not a grid of %ux%u frames",
						material->name, colour->width, colour->height, def.source_columns, def.source_rows));
				}
			}

			std::vector<iw7_texture_slot> slots;
			for (const auto& image : def.images)
			{
				slots.push_back({ image.slot, image.name });
				if (!claim_image(image.name, def.name, true))
				{
					continue; // written for another use of the same BO3 material
				}
				image_levels levels{};
				if (sub_atlas)
				{
					const auto whole = effect_level(image.texel, *colour, 0, mask ? &*mask : nullptr, globals);
					const auto frame_w = colour->width / def.source_columns, frame_h = colour->height / def.source_rows;
					levels.width = frame_w * sub_columns;
					levels.height = frame_h * sub_rows;
					std::vector<std::uint8_t> sub(static_cast<std::size_t>(levels.width) * levels.height * 4);
					for (auto n = 0u; n < def.frames.size(); n++)
					{
						const auto from_x = def.frames[n] % def.source_columns * frame_w, from_y = def.frames[n] / def.source_columns * frame_h;
						const auto to_x = n % sub_columns * frame_w, to_y = n / sub_columns * frame_h;
						for (auto y = 0u; y < frame_h; y++)
						{
							std::memcpy(&sub[(static_cast<std::size_t>(to_y + y) * levels.width + to_x) * 4],
								&whole[(static_cast<std::size_t>(from_y + y) * colour->width + from_x) * 4], static_cast<std::size_t>(frame_w) * 4);
						}
					}
					levels.rgba.push_back(std::move(sub));
					const auto full = mip_count(levels.width, levels.height);
					for (auto l = 1u; l < full; l++)
					{
						levels.rgba.push_back(halve(levels.rgba.back(), std::max(1u, levels.width >> (l - 1)),
							std::max(1u, levels.height >> (l - 1))));
					}
				}
				else
				{
					levels.width = colour->width;
					levels.height = colour->height;
					const auto full = mip_count(colour->width, colour->height);
					for (auto l = 0u; l < full; l++)
					{
						if (l < colour->level_count())
						{
							levels.rgba.push_back(effect_level(image.texel, *colour, l, mask ? &*mask : nullptr, globals));
						}
						else
						{
							levels.rgba.push_back(halve(levels.rgba.back(), std::max(1u, colour->width >> (l - 1)),
								std::max(1u, colour->height >> (l - 1))));
						}
					}
				}
				const auto flags = image.texel == effect_texel::distortion ? 0u : 0x300u;
				total_image_bytes += write_image(image.name, levels, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_COLOR_MAP, flags, wk.gpu.get());
			}

			// the atlas (the BO3 material's, or the range's): textureAtlas is (columns, rows, ...)
			auto atlas_donor = *donor;
			atlas_donor.atlas_rows = sub_atlas ? static_cast<unsigned char>(sub_rows) : std::max<unsigned char>(1, material->info.textureAtlasRowCount);
			atlas_donor.atlas_columns = sub_atlas ? static_cast<unsigned char>(sub_columns)
				: std::max<unsigned char>(1, material->info.textureAtlasColumnCount);
			// textureAtlas = (columns, rows, 1, 1) in every stock eq_ material (3605 of 3605); the donor's is its own atlas'
			auto constants = def.constants;
			constexpr std::uint32_t atlas_hash = 1128936273u; // textureAtlas
			for (auto i = 0u; i < donor->constant_count; i++)
			{
				if (donor->constants[i].hash == atlas_hash)
				{
					constants.push_back({ atlas_hash, { static_cast<float>(atlas_donor.atlas_columns), static_cast<float>(atlas_donor.atlas_rows),
						1.0f, 1.0f } });
				}
			}

			info inf{};
			inf.name = def.name;
			inf.cls = surface_class::trans;
			inf.sort_key = donor->sort_key;
			inf.camera_region = donor->camera_region;
			inf.casts_shadow = false;
			inf.lightmapped = false;
			const MaterialTechnique* lit = nullptr;
			for (const auto* technique : material->techniqueSet->techniques)
			{
				if (technique && technique_name(technique) == "lit")
				{
					lit = technique;
				}
			}
			// BO3's *_nocull effect templates draw both faces (oriented sprites seen from behind)
			write_material_json(def.name, atlas_donor, inf, iw7_state_flags(*donor, lit), constants, slots, def.frame_blend);
		}

		worker::worker()
			: gpu(gpu_eval::context::create())
		{
			if (!this->gpu)
			{
				ZONETOOL_INFO("material bake: no hardware Direct3D 11 device: materials are evaluated and compressed on the CPU");
			}
		}

		worker::~worker() = default;

		bool write_material(const Material* material, const info& inf, worker& wk)
		{
			const auto& p = get_plan(material, inf);
			if (!p.supported)
			{
				ZONETOOL_WARNING("%s material %s: not converted: %s", inf.model ? "model" : "world", describe_material(material).data(),
					p.reason.data());
				return false;
			}

			const auto* donor = world_techset_donors::find(p.techset, inf.camera_region);
			const std::string& name = inf.name;

			if (inf.cls == surface_class::shadow_only)
			{
				std::vector<iw7_texture_slot> images;
				for (auto i = 0u; i < donor->texture_count; i++)
				{
					images.push_back({ donor->textures[i].type_hash, donor->textures[i].stock_image });
				}
				write_material_json(name, *donor, inf, donor->state_flags, {}, images);
				return true;
			}

			// earlier materials' images leave the device once it holds more than its budget
			if (wk.gpu)
			{
				wk.gpu->trim();
				wk.bound = nullptr;
			}
			if (p.unlit_emissive)
			{
				write_unlit_emissive(material, inf, wk, p, *donor);
				return true;
			}
			if (p.eye)
			{
				write_eye(material, inf, wk, *donor);
				return true;
			}
			if (p.multiply)
			{
				write_multiply(material, inf, wk, p, *donor);
				return true;
			}
			if (p.water)
			{
				write_water(material, inf, wk, *donor);
				return true;
			}
			auto* const gpu = wk.gpu.get();

			material_program lit{};
			material_program mp{};
			if (p.surface)
			{
				mp = load_program(material, "lit", { nullptr, nullptr, false, true });
				if (p.forward_emissive)
				{
					lit = load_program(material, "lit", { nullptr, nullptr, true });
				}
			}
			else if (p.forward_emissive)
			{
				lit = load_program(material, "lit", { nullptr, nullptr, true });
				mp = load_program(material, "gbuffer", { sibling_techset(material), &lit, false });
			}
			else
			{
				mp = load_program(material, "gbuffer");
			}
			use_vertex_colour(mp, inf);
			use_vertex_colour(lit, inf);

			// the detail map becomes IW7's q0 detail texture: take it out of the base bake
			float detail_scale[2] = { 1.0f, 1.0f };
			float detail_height = 0.0f;
			if (p.detail)
			{
				const auto* scale = find_variable(mp, "detailScale");
				const auto* height = find_variable(mp, "detailScaleHeight");
				detail_scale[0] = read_float(mp, scale, 0);
				detail_scale[1] = read_float(mp, scale, 1);
				detail_height = read_float(mp, height, 0);
				write_float(mp, height, 0, 0.0f);
			}

			// coverage: reveal and alpha come from their own textures, so evaluate with them opaque
			std::unordered_map<std::string, std::array<float, 4>> overrides;
			if (p.reveal)
			{
				overrides["revealMap"] = { 1.0f, 1.0f, 1.0f, 1.0f };
			}

			// a model material and a world material can share a name (mc/x, wc/x)
			const auto base = image_base_name(name) + (inf.model ? "m" : "");
			const auto levels = mip_count(p.width, p.height);
			auto backlit = false;

			image_levels cs_image{ p.width, p.height }, ng_image{ p.width, p.height }, coverage{ p.width, p.height };
			// a transparent surface's coverage is its alpha, per texel as BO3 computes it: the surface program's (RT0.w), or
			// with an emissive the alpha the lit shader draws with (it raises the alpha where the emissive is bright)
			const auto needs_coverage = p.decal || p.alpha_test || p.surface_coverage();

			// forward emissive: the lit shader's output with every light, probe and fog term at zero is the
			// emissive radiance (rgb per texel, per level)
			std::vector<std::vector<float>> emissive_levels;
			const auto falloff_scale = p.forward_emissive ? remove_emissive_falloff(lit) : 1.0f;

			// the texture and channel that carries alpha-test coverage / decal coverage (a transparent surface's is evaluated)
			const auto coverage_channel = p.surface || p.coverage_program ? std::pair<const texture_binding*, std::uint32_t>{ nullptr, 0 }
				: coverage_source(mp, p.alpha_test, p.reveal);

			// an alpha test the texture alone does not give: the coverage its pixel shader compares
			material_program coverage_mp{};
			if (p.coverage_program)
			{
				coverage_mp = load_program(material, "gbuffer", { nullptr, nullptr, false, false, true });
				use_vertex_colour(coverage_mp, inf);
			}

			// made after every constant change above: the GPU takes its copy of the constant buffers when it binds
			evaluator eval(wk, material, mp, overrides);
			std::optional<evaluator> lit_eval;
			if (p.forward_emissive)
			{
				lit_eval.emplace(wk, material, lit, std::unordered_map<std::string, std::array<float, 4>>{});
			}
			std::optional<evaluator> coverage_eval;
			if (p.coverage_program)
			{
				coverage_eval.emplace(wk, material, coverage_mp, overrides);
			}

			// Once the gbuffer program runs on the worker's GPU, its targets are packed into the IW7 texels there and
			// block-compressed from there; only a decal's texels come back, for the fill of its uncovered texels.
			// Otherwise the targets are read back and packed on the CPU.
			const grid top{ p.width, p.height, p.uv_origin[0], p.uv_origin[1], p.uv_span[0], p.uv_span[1], 0 };
			const auto first_tile = tiles_of(p.width, p.height).front();
			auto& pack_checked = wk.pack_checked[p.surface ? 1 : 0];
			auto gpu_pack = gpu && eval.prepare(top, first_tile) && gpu->can_pack() && pack_checked >= 0;
			if (gpu_pack && !pack_checked)
			{
				timed _(stage_ns.gpu);
				pack_checked = packing_agrees(eval, *gpu, p, top, first_tile, name) ? 1 : -1;
				gpu_pack = pack_checked > 0;
			}
			const auto pack_opts = pack_options_of(p);
			std::vector<std::vector<std::uint8_t>> cs_blocks, ng_blocks; // BC7 levels packed and compressed on the GPU
			if (gpu_pack)
			{
				timed _(stage_ns.gpu);
				gpu->take_backlit(); // what an earlier material left
			}

			// forward emissive: lit shader texel li (tile-local) into the level's emissive radiance at texel i
			const auto store_emissive = [&](const std::vector<texel_out>& lit_out, const std::size_t li, const std::size_t i, std::vector<float>& emissive)
			{
				for (auto c = 0u; c < 3; c++)
				{
					const auto value = lit_out[li].target[0][c];
					if (!std::isfinite(value))
					{
						throw std::runtime_error("the lit shader evaluated to a non-finite emissive value");
					}
					emissive[i * 3 + c] = std::max(value, 0.0f) * falloff_scale;
				}
			};

			for (auto level = 0u; level < levels; level++)
			{
				const auto w = std::max(1u, p.width >> level);
				const auto h = std::max(1u, p.height >> level);
				const grid g{ w, h, p.uv_origin[0], p.uv_origin[1], p.uv_span[0], p.uv_span[1], 0 };
				auto g_other = g;
				g_other.parity_shift = 1;

				// the level's texels on the CPU (with GPU packing a decal's only, read back for its fill)
				std::vector<std::uint8_t> cs_level;
				std::vector<std::uint8_t> ng_level;
				if (!gpu_pack)
				{
					cs_level.resize(static_cast<std::size_t>(w) * h * 4);
					ng_level.resize(static_cast<std::size_t>(w) * h * 4);
				}
				std::vector<std::uint8_t> cov_level(needs_coverage ? static_cast<std::size_t>(w) * h * 4 : 0);
				std::vector<bool> covered(static_cast<std::size_t>(w) * h, true);
				// forward emissive: the lit shader's output with every light, probe and fog term at zero is the
				// emissive radiance (rgb per texel)
				std::vector<float> emissive(p.forward_emissive ? static_cast<std::size_t>(w) * h * 3 : 0);
				// a transparent surface's alpha per texel (a byte)
				std::vector<std::uint8_t> alpha_level(p.surface_coverage() ? static_cast<std::size_t>(w) * h : 0);
				std::atomic<bool> backlit_level{ false };

				if (gpu_pack)
				{
					{
						timed _(stage_ns.gpu);
						gpu->begin_level(w, h, p.decal || p.surface); // (the raw pack writes a surface's alpha there)
					}
					for (const auto& t : tiles_of(w, h))
					{
						eval.draw(g, t, p.coloured_specular);
						{
							timed _(stage_ns.gpu);
							gpu->pack({ t.x, t.y, t.width, t.height }, pack_opts);
						}
						if (lit_eval)
						{
							const auto lit_out = lit_eval->run(g, t, 1);
							timed _(stage_ns.pack);
							parallel_for(t.height, [&](const std::uint32_t ly, std::uint32_t)
							{
								for (auto lx = 0u; lx < t.width; lx++)
								{
									const auto li = static_cast<std::size_t>(ly) * t.width + lx;
									const auto i = static_cast<std::size_t>(t.y + ly) * w + t.x + lx;
									store_emissive(lit_out, li, i, emissive);
									if (p.surface_coverage())
									{
										alpha_level[i] = to_byte(lit_out[li].target[0][3]);
									}
								}
							});
						}
					}
					if (p.surface_coverage() && !lit_eval)
					{
						timed _(stage_ns.gpu);
						alpha_level = gpu->read_covered();
					}
					if (p.decal)
					{
						timed _(stage_ns.gpu);
						cs_level = gpu->read_packed(0, { 0, 0, w, h });
						ng_level = gpu->read_packed(1, { 0, 0, w, h });
						const auto cover = gpu->read_covered();
						for (std::size_t i = 0; i < cover.size(); i++)
						{
							covered[i] = cover[i] != 0;
						}
					}
					else
					{
						timed _(stage_ns.compress);
						cs_blocks.emplace_back(gpu->compress_packed(0));
						ng_blocks.emplace_back(gpu->compress_packed(1));
					}
				}
				else
				{
					for (const auto& t : tiles_of(w, h))
					{
						const auto same = eval.run(g, t);
						std::vector<texel_out> other;
						if (p.coloured_specular)
						{
							other = eval.run(g_other, t, 4); // only the other parity's chroma, in target 2
						}
						std::vector<texel_out> lit_out;
						if (lit_eval)
						{
							lit_out = lit_eval->run(g, t, 1);
						}

						timed _(stage_ns.pack);
						auto surf = p.surface ? decode_surface(same) : decode_gbuffer(same, p.coloured_specular ? &other : nullptr, t, p.decal);

						// rows in parallel: the metal fit in pack_colour is a search per texel
						parallel_for(t.height, [&](const std::uint32_t ly, std::uint32_t)
						{
							for (auto lx = 0u; lx < t.width; lx++)
							{
								const auto li = static_cast<std::size_t>(ly) * t.width + lx;
								const auto i = static_cast<std::size_t>(t.y + ly) * w + t.x + lx;
								if (pack_texel(p, surf[li], &cs_level[i * 4], &ng_level[i * 4]))
								{
									backlit_level = true;
								}
								if (!lit_out.empty())
								{
									store_emissive(lit_out, li, i, emissive);
								}
								if (p.surface_coverage())
								{
									alpha_level[i] = to_byte(lit_out.empty() ? surf[li].coverage[0] : lit_out[li].target[0][3]);
								}
							}
						});

						if (p.decal)
						{
							for (auto ly = 0u; ly < t.height; ly++)
							{
								for (auto lx = 0u; lx < t.width; lx++)
								{
									covered[static_cast<std::size_t>(t.y + ly) * w + t.x + lx] =
										surf[static_cast<std::size_t>(ly) * t.width + lx].coverage[0] > (1.0f / 1024.0f);
								}
							}
						}
					}
				}
				backlit |= backlit_level.load();

				// a texture where the shader samples it, at the level of detail the interpreter gives that sample
				const auto sample = [&](const material_texture::texture& tex, const sample_map& map, const float u, const float v, float out[4])
				{
					std::uint32_t tw, th, tl;
					tex.dimensions(0, tw, th, tl);
					const auto& m = map.m;
					const auto dudx = m[0] * g.span_u / static_cast<float>(w) * static_cast<float>(tw);
					const auto dvdx = m[3] * g.span_u / static_cast<float>(w) * static_cast<float>(th);
					const auto dudy = m[1] * g.span_v / static_cast<float>(h) * static_cast<float>(tw);
					const auto dvdy = m[4] * g.span_v / static_cast<float>(h) * static_cast<float>(th);
					const auto rho = std::max(std::sqrt(dudx * dudx + dvdx * dvdx), std::sqrt(dudy * dudy + dvdy * dvdy));
					const auto lod = rho > 0.0f ? std::log2(rho) : -1000.0f;
					tex.sample(m[0] * u + m[1] * v + m[2], m[3] * u + m[4] * v + m[5], std::max(lod, 0.0f), map.offset[0], map.offset[1], out);
				};

				if (p.surface_coverage())
				{
					// its evaluated alpha; a reveal decal's reveal map (the packed alpha's g)
					std::shared_ptr<const material_texture::decoded> reveal;
					const auto* rb = find_binding(mp, "revealMap");
					if (p.reveal)
					{
						timed _(stage_ns.textures);
						reveal = rb ? material_texture::decode(rb->image) : nullptr;
						if (!reveal)
						{
							throw std::runtime_error("the reveal map has no pixels");
						}
					}
					timed _(stage_ns.pack);
					parallel_for(h, [&](const std::uint32_t y, std::uint32_t)
					{
						std::unique_ptr<material_texture::texture> rev_tex;
						if (reveal)
						{
							rev_tex = std::make_unique<material_texture::texture>(reveal, rb->u, rb->v);
						}
						for (auto x = 0u; x < w; x++)
						{
							const auto i = static_cast<std::size_t>(y) * w + x;
							auto reveal_byte = static_cast<std::uint8_t>(255);
							if (rev_tex)
							{
								float t4[4];
								sample(*rev_tex, p.reveal_map, g.origin_u + (x + 0.5f) / w * g.span_u, g.origin_v + (y + 0.5f) / h * g.span_v, t4);
								reveal_byte = to_byte(t4[0]);
							}
							cov_level[i * 4 + 0] = alpha_level[i];
							cov_level[i * 4 + 1] = reveal_byte;
							cov_level[i * 4 + 2] = 255;
							cov_level[i * 4 + 3] = 255;
						}
					});
				}
				else if (coverage_eval)
				{
					// the coverage the pixel shader compares with 0.5, as IW7's alpha test does
					for (const auto& t : tiles_of(w, h))
					{
						const auto out = coverage_eval->run(g, t, 1);
						timed _(stage_ns.pack);
						for (auto ly = 0u; ly < t.height; ly++)
						{
							for (auto lx = 0u; lx < t.width; lx++)
							{
								const auto i = static_cast<std::size_t>(t.y + ly) * w + t.x + lx;
								cov_level[i * 4 + 0] = to_byte(out[static_cast<std::size_t>(ly) * t.width + lx].target[0][0]);
								cov_level[i * 4 + 1] = 255;
								cov_level[i * 4 + 2] = 255;
								cov_level[i * 4 + 3] = 255;
							}
						}
					}
				}
				else if (needs_coverage)
				{
					// coverage straight from its texture, sampled like the shader samples it
					const auto [tex, channel] = coverage_channel;
					std::shared_ptr<const material_texture::decoded> decoded;
					std::shared_ptr<const material_texture::decoded> reveal;
					{
						timed _(stage_ns.textures);
						if (tex)
						{
							decoded = material_texture::decode(tex->image);
						}
						if (p.reveal)
						{
							reveal = material_texture::decode(find_binding(mp, "revealMap")->image);
						}
					}
					timed _(stage_ns.pack);
					parallel_for(h, [&](const std::uint32_t y, std::uint32_t)
					{
						std::unique_ptr<material_texture::texture> cov_tex, rev_tex;
						if (decoded)
						{
							cov_tex = std::make_unique<material_texture::texture>(decoded, tex->u, tex->v);
						}
						if (reveal)
						{
							const auto* rb = find_binding(mp, "revealMap");
							rev_tex = std::make_unique<material_texture::texture>(reveal, rb->u, rb->v);
						}
						for (auto x = 0u; x < w; x++)
						{
							const auto u = g.origin_u + (x + 0.5f) / w * g.span_u;
							const auto v = g.origin_v + (y + 0.5f) / h * g.span_v;
							const auto i = static_cast<std::size_t>(y) * w + x;
							float alpha = 1.0f;
							if (cov_tex)
							{
								float t4[4];
								sample(*cov_tex, p.coverage_map, u, v, t4);
								alpha = t4[channel];
							}
							float reveal_value = 1.0f;
							if (rev_tex)
							{
								float t4[4];
								sample(*rev_tex, p.reveal_map, u, v, t4);
								reveal_value = t4[0];
							}
							cov_level[i * 4 + 0] = to_byte(alpha);
							cov_level[i * 4 + 1] = to_byte(reveal_value);
							cov_level[i * 4 + 2] = 255;
							cov_level[i * 4 + 3] = 255;
						}
					});
				}

				if (p.decal)
				{
					timed _(stage_ns.pack);
					const std::uint8_t neutral_cs[4] = { 128, 128, 128, 10 };
					const std::uint8_t neutral_ng[4] = { 128, 128, 255, 128 };
					fill_uncovered(cs_level, covered, w, h, neutral_cs);
					fill_uncovered(ng_level, covered, w, h, neutral_ng);
				}

				if (!gpu_pack || p.decal)
				{
					cs_image.rgba.emplace_back(std::move(cs_level));
					ng_image.rgba.emplace_back(std::move(ng_level));
				}
				if (needs_coverage)
				{
					coverage.rgba.emplace_back(std::move(cov_level));
				}
				if (p.forward_emissive)
				{
					emissive_levels.emplace_back(std::move(emissive));
				}
			}
			if (gpu_pack)
			{
				timed _(stage_ns.gpu);
				backlit |= gpu->take_backlit();
			}

			std::vector<iw7_texture_slot> images;
			std::vector<std::pair<std::uint32_t, std::array<float, 4>>> constants;

			std::optional<atlas_slot> slot;
			if (inf.model)
			{
				std::lock_guard _(atlas_mutex);
				const auto found = atlas_members.find(material);
				if (found != atlas_members.end())
				{
					slot = found->second;
				}
			}
			const auto cs_name = slot ? slot->cs : base + "_packed_cs";
			const auto ng_name = slot ? slot->ng : base + (p.occlusion ? "_packed_nog" : "_packed_ng");
			if (slot)
			{
				// (atlas_tile keeps decals out: the levels are packed as they are)
				if (gpu_pack)
				{
					total_image_bytes += deposit_tile(*slot, std::move(cs_blocks), std::move(ng_blocks));
				}
				else
				{
					std::vector<std::vector<std::uint8_t>> cs_levels, ng_levels;
					{
						timed _(stage_ns.compress);
						cs_levels = compress_levels(cs_image, DXGI_FORMAT_BC7_UNORM, gpu);
						ng_levels = compress_levels(ng_image, DXGI_FORMAT_BC7_UNORM, gpu);
					}
					total_image_bytes += deposit_tile(*slot, std::move(cs_levels), std::move(ng_levels));
				}
			}
			else
			{
				claim_image(cs_name, name, false);
				claim_image(ng_name, name, false);
				if (gpu_pack && !p.decal)
				{
					total_image_bytes += write_levels(cs_name, p.width, p.height, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_COLOR_SPECULAR_MAP, 0x2300, cs_blocks);
					total_image_bytes += write_levels(ng_name, p.width, p.height, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_NORMAL_OCCLUSSION_GLOSS_MAP, 0x308,
						ng_blocks);
				}
				else
				{
					total_image_bytes += write_image(cs_name, cs_image, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_COLOR_SPECULAR_MAP, 0x2300, gpu);
					total_image_bytes += write_image(ng_name, ng_image, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_NORMAL_OCCLUSSION_GLOSS_MAP, 0x308, gpu);
				}
			}
			images.push_back({ hash_color_map, cs_name });
			images.push_back({ hash_normal_map, ng_name });

			if (needs_coverage)
			{
				if (p.reveal)
				{
					const auto name_r = base + (coverage_channel.first || p.surface ? "_packed_ar" : "_packed_r");
					claim_image(name_r, name, false);
					total_image_bytes += write_image(name_r, coverage, DXGI_FORMAT_BC1_UNORM, zonetool::iw7::TS_ALPHA_REVEAL_THICKNESS_MAP, 0x300, gpu);
					images.push_back({ hash_spec_occlusion_map, name_r });

					// IW7's reveal edge uses a fixed square root where BO3 uses alphaRevealRamp;
					// match the ramp's width at half revealed
					const auto soft = read_float(mp, find_variable(mp, "alphaRevealSoftEdge"), 0);
					const auto ramp = std::clamp(read_float(mp, find_variable(mp, "alphaRevealRamp"), 0), 0.0f, 1.0f);
					const auto fitted = soft * std::pow(0.5f, ramp - 0.5f);
					constants.push_back({ r_hash_string("revealParams"), { fitted, 0.5f, 0.0f, 0.0f } });
				}
				else
				{
					const auto name_a = base + "_packed_a";
					claim_image(name_a, name, false);
					if (p.alpha_test)
					{
						preserve_coverage(coverage);
					}
					total_image_bytes += write_image(name_a, coverage, DXGI_FORMAT_BC4_UNORM, zonetool::iw7::TS_ALPHA_REVEAL_THICKNESS_MAP, 0x300, gpu);
					images.push_back({ hash_spec_occlusion_map, name_a });
				}
			}

			if (p.detail)
			{
				// BO3 detail normal: xy = t * 1.992188 - 1, scaled by detailScaleHeight, z = 1
				const auto* db = find_binding(mp, "detailMap");
				// the height scale is baked in, so it is part of the name; materials sharing both share the image
				const std::string detail_name = utils::string::va("%s_%08x_detail_n", image_base_name(db->image->name).data(),
					std::bit_cast<std::uint32_t>(detail_height));
				if (claim_image(detail_name, name, true))
				{
					std::shared_ptr<const material_texture::decoded> detail;
					{
						timed _(stage_ns.textures);
						detail = material_texture::decode(db->image);
					}
					if (!detail)
					{
						throw std::runtime_error("detail map has no pixels");
					}
					image_levels detail_levels{ detail->width, detail->height };
					{
						timed _(stage_ns.pack);
						for (auto l = 0u; l < detail->level_count(); l++)
						{
							std::uint32_t w, h;
							detail->level_size(l, w, h);
							std::vector<std::uint8_t> level(static_cast<std::size_t>(w) * h * 4);
							parallel_for(h, [&](const std::uint32_t y, std::uint32_t)
							{
								for (auto x = 0u; x < w; x++)
								{
									float t4[4];
									detail->fetch(l, x, y, t4);
									float n[3] = { (t4[0] * 1.992188f - 1.0f) * detail_height, (t4[1] * 1.992188f - 1.0f) * detail_height, 1.0f };
									const auto len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
									n[0] /= len;
									n[1] /= len;
									n[2] /= len;
									float g_ga, a_ga;
									octahedral(n, g_ga, a_ga);
									const auto i = static_cast<std::size_t>(y) * w + x;
									level[i * 4 + 0] = 255;
									level[i * 4 + 1] = to_byte(g_ga);
									level[i * 4 + 2] = 0;
									level[i * 4 + 3] = to_byte(a_ga);
								}
							});
							detail_levels.rgba.emplace_back(std::move(level));
						}
					}
					total_image_bytes += write_image(detail_name, detail_levels, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_NORMAL_MAP, 0, gpu);
				}
				images.push_back({ hash_detail_map, detail_name });
				constants.push_back({ r_hash_string("detailScale"), { detail_scale[0], detail_scale[1], 0.0f, 0.0f } });
			}
			else if (p.flat_detail)
			{
				// a flat detail normal, as the detail conversion above packs the normal (0, 0, 1): the octahedral centre in g / a
				const std::string detail_name = "t7_flat_detail_n";
				if (claim_image(detail_name, name, true))
				{
					image_levels flat{ 4, 4 };
					for (auto l = 0u; l < mip_count(4, 4); l++)
					{
						const auto size = std::max(1u, 4u >> l);
						std::vector<std::uint8_t> level(static_cast<std::size_t>(size) * size * 4);
						for (std::size_t i = 0; i < level.size(); i += 4)
						{
							level[i + 0] = 255;
							level[i + 1] = to_byte(0.5f);
							level[i + 2] = 0;
							level[i + 3] = to_byte(0.5f);
						}
						flat.rgba.emplace_back(std::move(level));
					}
					total_image_bytes += write_image(detail_name, flat, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_NORMAL_MAP, 0, gpu);
				}
				images.push_back({ hash_detail_map, detail_name });
				constants.push_back({ r_hash_string("detailScale"), { 1.0f, 1.0f, 0.0f, 0.0f } });
			}

			if (p.forward_emissive)
			{
				// IW7 e0: emissive = emissiveMap.rgb^2 * emissiveParams.x (UNORM read, the shader squares it), so the
				// image holds sqrt(E / S) with S the largest emissive value. A material that emits nothing at rest (a
				// script turns it on) keeps a black 4x4 image and S = 0.
				auto scale = 0.0f;
				for (const auto v : emissive_levels[0])
				{
					scale = std::max(scale, v);
				}
				image_levels e_image{ p.width, p.height };
				if (scale > 0.0f)
				{
					for (const auto& level : emissive_levels)
					{
						std::vector<std::uint8_t> rgba(level.size() / 3 * 4);
						for (auto i = 0u; i < level.size() / 3; i++)
						{
							for (auto c = 0u; c < 3; c++)
							{
								rgba[i * 4 + c] = to_byte(std::sqrt(level[i * 3 + c] / scale));
							}
							rgba[i * 4 + 3] = 255;
						}
						e_image.rgba.emplace_back(std::move(rgba));
					}
				}
				else
				{
					e_image = { 4, 4 };
					for (auto l = 0u; l < mip_count(4, 4); l++)
					{
						const auto size = std::max(1u, 4u >> l);
						std::vector<std::uint8_t> rgba(static_cast<std::size_t>(size) * size * 4, 0);
						for (auto i = 3u; i < rgba.size(); i += 4)
						{
							rgba[i] = 255;
						}
						e_image.rgba.emplace_back(std::move(rgba));
					}
				}
				const auto e_name = base + "_e";
				claim_image(e_name, name, false);
				total_image_bytes += write_image(e_name, e_image, DXGI_FORMAT_BC7_UNORM, zonetool::iw7::TS_COLOR_MAP, 0x300, gpu);
				images.push_back({ hash_emissive_map, e_name });
				constants.push_back({ r_hash_string("emissiveParams"), { scale * map::bo3_light_scale, 0.0f, 0.0f, 0.0f } });
				const std::string techset_name = material->techniqueSet->name;
				const auto animated = techset_name.find("_scroll") != std::string::npos || techset_name.find("_script") != std::string::npos;
				ZONETOOL_INFO("%s material %s: forward emissive over %s, peak emissive %.3f%s", inf.model ? "model" : "world", name.data(),
					p.surface ? "its own surface (transparent)" : p.base_techset.data(), scale,
					animated ? " (BO3 animates it; IW7 has no lit animated emissive: the layers as at time 0)" : "");
				if (!inf.model && p.occlusion)
				{
					ZONETOOL_INFO("world material %s: BO3 occlusion not kept (stock IW7 w_ e0 has no o0 variant)", name.data());
				}
			}

			constants.push_back({ r_hash_string("colorTint"), { 1.0f, 1.0f, 1.0f, 1.0f } });

			// the state BO3 drew the material with: its own lit technique for a forward material
			write_material_json(name, *donor, inf, iw7_state_flags(*donor, p.forward_emissive ? lit.technique : mp.technique),
				constants, images);

			if (backlit)
			{
				ZONETOOL_INFO("%s material %s: BO3 backlit scattering has no IW7 packed equivalent, converted as a standard surface",
					inf.model ? "model" : "world", name.data());
			}
			if (!p.surface_sources.empty())
			{
				ZONETOOL_INFO("%s material %s: forward surface %s", inf.model ? "model" : "world", name.data(), p.surface_sources.data());
			}
			if (p.reduced)
			{
				ZONETOOL_INFO("%s material %s: baked at %ux%u, 1/%u of the BO3 texel density (a streamed image part holds at most 64 MB)",
					inf.model ? "model" : "world", name.data(), p.width, p.height, 1u << p.reduced);
			}
			if (static_cast<std::uint64_t>(p.width) * p.height >= 4096ull * 4096ull)
			{
				ZONETOOL_INFO("%s material %s: baked %ux%u over %.4g x %.4g texture units, its size set by %s", inf.model ? "model" : "world",
					name.data(), p.width, p.height, p.uv_span[0], p.uv_span[1], p.size_from.data());
			}
			if (p.decal && (!p.writes_normal || !p.writes_gloss || !p.writes_specular))
			{
				ZONETOOL_INFO("%s material %s: BO3 decal keeps the underlying%s%s%s; IW7 lights it with a flat normal / gloss %.3f / F0 0.04",
					inf.model ? "model" : "world", name.data(), p.writes_normal ? "" : " normal", p.writes_gloss ? "" : " gloss", p.writes_specular ? "" : " specular",
					underlying_gloss);
			}
			return true;
		}

		void set_underlying_gloss(const float gloss)
		{
			underlying_gloss = gloss;
		}

		float estimate_underlying_gloss(const std::vector<const Material*>& materials)
		{
			std::vector<float> values;
			for (const auto* material : materials)
			{
				if (world_material::get(material).cls != surface_class::opaque)
				{
					continue;
				}
				try
				{
					const auto mp = load_program(material, "gbuffer");
					if (const auto* range = find_variable(mp, "glossRange"))
					{
						values.push_back(std::clamp(read_float(mp, range, 1) / 17.0f, 0.0f, 1.0f));
					}
				}
				catch (const std::exception&)
				{
				}
			}
			if (values.empty())
			{
				return 0.0f;
			}
			std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
			const auto median = values[values.size() / 2];
			ZONETOOL_INFO("world materials: median opaque gloss %.3f (glossRange.y %.2f) over %zu materials", median, median * 17.0f,
				values.size());
			return median;
		}

		std::size_t image_bytes()
		{
			return total_image_bytes;
		}

		std::string stage_report()
		{
			const auto seconds = [](std::atomic<std::uint64_t>& ns)
			{
				return static_cast<double>(ns.exchange(0)) / 1e9;
			};
			const auto textures = seconds(stage_ns.textures);
			const auto gpu = seconds(stage_ns.gpu);
			const auto cpu = seconds(stage_ns.cpu);
			const auto check = seconds(stage_ns.check);
			const auto pack = seconds(stage_ns.pack);
			const auto compress = seconds(stage_ns.compress);
			const auto write = seconds(stage_ns.write);
			return utils::string::va("thread time: GPU evaluation %.0f s, interpreter %.0f s (+%.0f s checking the GPU), CPU texture "
				"decode %.0f s, packing %.0f s, compression %.0f s, writing %.0f s", gpu, cpu, check, textures, pack, compress, write);
		}
	}
}
