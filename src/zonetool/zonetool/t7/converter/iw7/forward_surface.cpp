#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "forward_surface.hpp"
#include "shader_eval.hpp"
#include "gpu_eval.hpp"

#include <shader-tool/shader.hpp>
#include <utils/string.hpp>

namespace zonetool::t7
{
	namespace converter::iw7::forward_surface
	{
		namespace
		{
			using alys::shader::shader_object;
			namespace detail = alys::shader::detail;

			constexpr std::uint32_t chunk_rdef = 'FEDR';
			constexpr std::uint32_t chunk_osgn = 'NGSO';
			constexpr std::uint32_t chunk_shex = 'XEHS';

			// PerSceneConsts' debug override of each role, in role order
			constexpr const char* override_names[] = { "debugColorOverride", "debugAlphaOverride", "debugNormalOverride",
				"debugSpecularOverride", "debugGlossOverride", "debugOcclusionOverride" };
			static_assert(std::size(override_names) == static_cast<std::size_t>(role::count));
			constexpr const char* role_names[] = { "albedo", "alpha", "normal", "specular", "gloss", "occlusion" };
			static_assert(std::size(role_names) == static_cast<std::size_t>(role::count));

			constexpr std::uint32_t role_count = static_cast<std::uint32_t>(role::count);

			std::uint32_t read_u32(const std::uint8_t* data, const std::size_t size, const std::size_t at)
			{
				if (at + 4 > size)
				{
					throw std::runtime_error("read past the end of the shader container");
				}
				std::uint32_t value;
				std::memcpy(&value, data + at, 4);
				return value;
			}

			// the whole chunk (header included) of the first chunk with this FourCC
			std::optional<std::string> find_chunk(const std::uint8_t* data, const std::size_t size, const std::uint32_t fourcc)
			{
				const auto count = read_u32(data, size, 28);
				for (auto i = 0u; i < count; i++)
				{
					const auto offset = read_u32(data, size, 32 + i * 4);
					if (read_u32(data, size, offset) != fourcc)
					{
						continue;
					}
					const auto length = read_u32(data, size, offset + 4);
					if (static_cast<std::size_t>(offset) + 8 + length > size)
					{
						throw std::runtime_error("a shader chunk runs past its container");
					}
					return std::string(reinterpret_cast<const char*>(data) + offset, 8 + length);
				}
				return std::nullopt;
			}

			bool is_block_start(const std::uint32_t opcode)
			{
				return opcode == D3D10_SB_OPCODE_IF || opcode == D3D10_SB_OPCODE_LOOP || opcode == D3D10_SB_OPCODE_SWITCH;
			}

			bool is_block_end(const std::uint32_t opcode)
			{
				return opcode == D3D10_SB_OPCODE_ENDIF || opcode == D3D10_SB_OPCODE_ENDLOOP || opcode == D3D10_SB_OPCODE_ENDSWITCH;
			}

			// dcl_* and the immediate constant buffer, which precede a program's code
			bool is_declaration(const std::uint32_t opcode)
			{
				return opcode == D3D10_SB_OPCODE_CUSTOMDATA
					|| (opcode >= D3D10_SB_OPCODE_DCL_RESOURCE && opcode <= D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS)
					|| (opcode >= D3D11_SB_OPCODE_DCL_STREAM && opcode <= D3D11_SB_OPCODE_DCL_RESOURCE_STRUCTURED)
					|| opcode == D3D11_SB_OPCODE_DCL_GS_INSTANCE_COUNT;
			}

			// component indices a destination mask writes, x first
			std::vector<std::uint32_t> components_of(const detail::operand_t& dest)
			{
				std::vector<std::uint32_t> out;
				if (dest.components.type != D3D10_SB_OPERAND_4_COMPONENT
					|| dest.components.selection_mode != D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE)
				{
					return out;
				}
				const auto mask = dest.components.mask ? dest.components.mask : 0xFu;
				for (auto c = 0u; c < 4; c++)
				{
					if (mask & (1u << c))
					{
						out.push_back(c);
					}
				}
				return out;
			}

			// `cb[slot][reg].w` with no modifier, as a whole-register selection of w
			std::optional<std::uint32_t> weight_register(const detail::operand_t& op, const std::uint32_t slot)
			{
				if (op.type != D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER || op.dimension != 2 || !op.extensions.empty()
					|| op.indices[0].representation != D3D10_SB_OPERAND_INDEX_IMMEDIATE32
					|| op.indices[1].representation != D3D10_SB_OPERAND_INDEX_IMMEDIATE32 || op.indices[0].value.uint32 != slot
					|| op.components.type != D3D10_SB_OPERAND_4_COMPONENT)
				{
					return std::nullopt;
				}
				constexpr auto w = static_cast<std::uint8_t>(D3D10_SB_4_COMPONENT_W);
				const auto& c = op.components;
				const auto all_w = c.selection_mode == D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE ? c.names[0] == w
					: c.selection_mode == D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE && c.names[0] == w && c.names[1] == w && c.names[2] == w && c.names[3] == w;
				if (!all_w)
				{
					return std::nullopt;
				}
				return op.indices[1].value.uint32;
			}

			detail::instruction_t mov(const detail::operand_t& dest, const detail::operand_t& source)
			{
				auto ins = detail::create_instruction(D3D10_SB_OPCODE_MOV);
				ins.operands.push_back(dest);
				ins.operands.push_back(source);
				return ins;
			}

			std::string swizzle_of(const std::vector<std::uint32_t>& components)
			{
				std::string out;
				for (const auto c : components)
				{
					out.push_back("xyzw"[c]);
				}
				return out;
			}

			// a temp read as one replicated component (.xxxx, or a select-1 operand) without modifiers: register, component
			std::optional<std::pair<std::uint32_t, std::uint32_t>> scalar_temp(const detail::operand_t& op)
			{
				if (op.type != D3D10_SB_OPERAND_TYPE_TEMP || op.dimension != 1 || !op.extensions.empty()
					|| op.components.type != D3D10_SB_OPERAND_4_COMPONENT)
				{
					return std::nullopt;
				}
				const auto& c = op.components;
				const auto replicated = c.selection_mode == D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE
					|| (c.selection_mode == D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE && c.names[0] == c.names[1]
						&& c.names[0] == c.names[2] && c.names[0] == c.names[3]);
				if (!replicated)
				{
					return std::nullopt;
				}
				return std::make_pair(op.indices[0].value.uint32, static_cast<std::uint32_t>(c.names[0]));
			}

			// the value an immediate gives destination component `component`
			std::optional<float> literal_component(const detail::operand_t& op, const std::uint32_t component)
			{
				if (op.type != D3D10_SB_OPERAND_TYPE_IMMEDIATE32 || !op.extensions.empty())
				{
					return std::nullopt;
				}
				if (op.components.type == D3D10_SB_OPERAND_1_COMPONENT)
				{
					return op.immediate_values[0].float32;
				}
				if (op.components.type == D3D10_SB_OPERAND_4_COMPONENT && component < 4)
				{
					return op.immediate_values[component].float32;
				}
				return std::nullopt;
			}

			using scalar = std::pair<std::uint32_t, std::uint32_t>; // temp register, component

			// whether instruction `ins` writes component `s.second` of temp `s.first`
			bool writes(const detail::instruction_t& ins, const scalar& s)
			{
				if (ins.operands.empty())
				{
					return false;
				}
				const auto& dest = ins.operands[0];
				if (dest.type != D3D10_SB_OPERAND_TYPE_TEMP || dest.dimension != 1 || dest.indices[0].value.uint32 != s.first)
				{
					return false;
				}
				const auto comps = components_of(dest);
				return std::ranges::find(comps, s.second) != comps.end();
			}

			// x when the last write of `s` before instruction `at` is `add s, -x, l(1)` (either order), x a replicated
			// temp; `where` gets that write's index
			std::optional<scalar> one_minus_source(const std::vector<detail::instruction_t>& instructions, const std::size_t at,
				const scalar& s, std::size_t* where = nullptr)
			{
				for (auto i = at; i-- > 0;)
				{
					const auto& ins = instructions[i];
					if (!writes(ins, s))
					{
						continue;
					}
					if (ins.opcode.type != D3D10_SB_OPCODE_ADD || ins.operands.size() != 3 || components_of(ins.operands[0]).size() != 1)
					{
						return std::nullopt;
					}
					const auto negated = [](const detail::operand_t& op) -> std::optional<scalar>
					{
						if (op.type != D3D10_SB_OPERAND_TYPE_TEMP || op.extensions.size() != 1
							|| op.extensions[0].type != D3D10_SB_EXTENDED_OPERAND_MODIFIER
							|| op.extensions[0].modifier != D3D10_SB_OPERAND_MODIFIER_NEG)
						{
							return std::nullopt;
						}
						auto plain = op;
						plain.extensions.clear();
						return scalar_temp(plain);
					};
					const auto one = [&](const detail::operand_t& op)
					{
						const auto v = literal_component(op, s.second);
						return v && *v == 1.0f;
					};
					std::optional<scalar> x;
					if (one(ins.operands[2]))
					{
						x = negated(ins.operands[1]);
					}
					else if (one(ins.operands[1]))
					{
						x = negated(ins.operands[2]);
					}
					if (x && where)
					{
						*where = i;
					}
					return x;
				}
				return std::nullopt;
			}

			// whether the last write of `s` before instruction `at` is `add s, -x, l(1)` (either order)
			bool is_one_minus(const std::vector<detail::instruction_t>& instructions, const std::size_t at, const scalar& s)
			{
				return one_minus_source(instructions, at, s).has_value();
			}

			// the F0 of a template without a specular label: the constant of the one top-level forward decal blend
			// `mad s.xyz, W, l(c, c, c, 0), decal.xyz` whose W is `add W, -coverage, l(1)`, 0 < c < 1
			std::optional<float> constant_specular(const std::vector<detail::instruction_t>& instructions)
			{
				std::optional<float> found;
				auto count = 0;
				auto depth = 0;
				for (std::size_t i = 0; i < instructions.size(); i++)
				{
					const auto& ins = instructions[i];
					if (is_block_end(ins.opcode.type))
					{
						depth--;
					}
					if (depth == 0 && ins.opcode.type == D3D10_SB_OPCODE_MAD && ins.operands.size() == 4)
					{
						const auto& dest = ins.operands[0];
						const auto& decal = ins.operands[3];
						const auto weight = scalar_temp(ins.operands[1]);
						const auto c = literal_component(ins.operands[2], 0);
						const auto rgb = ins.operands[2].components.type == D3D10_SB_OPERAND_4_COMPONENT && c
							&& *literal_component(ins.operands[2], 1) == *c && *literal_component(ins.operands[2], 2) == *c
							&& *literal_component(ins.operands[2], 3) == 0.0f;
						if (dest.type == D3D10_SB_OPERAND_TYPE_TEMP && components_of(dest) == std::vector<std::uint32_t>{ 0, 1, 2 }
							&& weight && rgb && *c > 0.0f && *c < 1.0f && decal.type == D3D10_SB_OPERAND_TYPE_TEMP
							&& decal.extensions.empty() && is_one_minus(instructions, i, *weight))
						{
							found = *c;
							count++;
						}
					}
					if (is_block_start(ins.opcode.type))
					{
						depth++;
					}
				}
				return count == 1 ? found : std::nullopt;
			}

			// The F0 of a template whose debug-override step has no specular (hair: sat(0.04 + 0.02 albedo) x
			// specColorTint): the one top-level `mul f.xyz, x, cb[slot][reg].xyz` of its $Globals specColorTint, whose
			// destination holds it right after
			std::optional<std::size_t> specular_tint_anchor(const std::vector<detail::instruction_t>& instructions, const std::uint32_t slot,
				const std::uint32_t reg)
			{
				const auto is_tint = [&](const detail::operand_t& op)
				{
					const auto& c = op.components;
					return op.type == D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER && op.dimension == 2 && op.extensions.empty()
						&& op.indices[0].representation == D3D10_SB_OPERAND_INDEX_IMMEDIATE32
						&& op.indices[1].representation == D3D10_SB_OPERAND_INDEX_IMMEDIATE32 && op.indices[0].value.uint32 == slot
						&& op.indices[1].value.uint32 == reg && c.type == D3D10_SB_OPERAND_4_COMPONENT
						&& c.selection_mode == D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE && c.names[0] == 0 && c.names[1] == 1 && c.names[2] == 2;
				};
				std::optional<std::size_t> found;
				auto count = 0;
				auto depth = 0;
				for (std::size_t i = 0; i < instructions.size(); i++)
				{
					const auto& ins = instructions[i];
					if (is_block_end(ins.opcode.type))
					{
						depth--;
					}
					if (depth == 0 && ins.opcode.type == D3D10_SB_OPCODE_MUL && ins.operands.size() == 3
						&& !(ins.opcode.controls & (D3D10_SB_INSTRUCTION_SATURATE_MASK >> 11))
						&& ins.operands[0].type == D3D10_SB_OPERAND_TYPE_TEMP
						&& components_of(ins.operands[0]) == std::vector<std::uint32_t>{ 0, 1, 2 }
						&& (is_tint(ins.operands[1]) || is_tint(ins.operands[2])))
					{
						found = i;
						count++;
					}
					if (is_block_start(ins.opcode.type))
					{
						depth++;
					}
				}
				return count == 1 ? found : std::nullopt;
			}

			// The gloss of a template whose debug-override step has no gloss: the value its lighting raises 2 to, the one
			// top-level `mul t.c, g, l(17)` right before `exp u, t.c` (BO3's specular power 2^(17 g)) whose g is not last
			// written from a PerSceneConsts debug override (the hair's debug permutation also raises 2 to its override's
			// gloss, `mul g', x, cb.debugGlossOverride.w`); the mul's index and g
			std::optional<std::pair<std::size_t, detail::operand_t>> gloss_power_anchor(const std::vector<detail::instruction_t>& instructions,
				const std::uint32_t scene_slot, const std::uint32_t* override_registers, const std::size_t override_count)
			{
				const auto from_override = [&](const std::size_t at, const scalar& g)
				{
					for (auto k = at; k-- > 0;)
					{
						if (!writes(instructions[k], g))
						{
							continue;
						}
						for (std::size_t o = 1; o < instructions[k].operands.size(); o++)
						{
							const auto& op = instructions[k].operands[o];
							if (op.type == D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER && op.dimension == 2
								&& op.indices[0].representation == D3D10_SB_OPERAND_INDEX_IMMEDIATE32
								&& op.indices[1].representation == D3D10_SB_OPERAND_INDEX_IMMEDIATE32 && op.indices[0].value.uint32 == scene_slot
								&& std::find(override_registers, override_registers + override_count, op.indices[1].value.uint32)
									!= override_registers + override_count)
							{
								return true;
							}
						}
						return false;
					}
					return false;
				};
				std::optional<std::pair<std::size_t, detail::operand_t>> found;
				auto count = 0;
				auto depth = 0;
				for (std::size_t i = 0; i + 1 < instructions.size(); i++)
				{
					const auto& ins = instructions[i];
					if (is_block_end(ins.opcode.type))
					{
						depth--;
					}
					if (depth == 0 && ins.opcode.type == D3D10_SB_OPCODE_MUL && ins.operands.size() == 3
						&& !(ins.opcode.controls & (D3D10_SB_INSTRUCTION_SATURATE_MASK >> 11))
						&& ins.operands[0].type == D3D10_SB_OPERAND_TYPE_TEMP && components_of(ins.operands[0]).size() == 1)
					{
						const scalar t{ ins.operands[0].indices[0].value.uint32, components_of(ins.operands[0])[0] };
						const auto seventeen = [&](const detail::operand_t& op)
						{
							const auto v = literal_component(op, t.second);
							return v && *v == 17.0f;
						};
						const auto g = seventeen(ins.operands[2]) ? &ins.operands[1] : seventeen(ins.operands[1]) ? &ins.operands[2] : nullptr;
						const auto& next = instructions[i + 1];
						const auto exp_reads = next.opcode.type == D3D10_SB_OPCODE_EXP && next.operands.size() == 2 && scalar_temp(next.operands[1])
							&& *scalar_temp(next.operands[1]) == t;
						if (g && scalar_temp(*g) && exp_reads && !from_override(i, *scalar_temp(*g)))
						{
							found = std::make_pair(i, *g);
							count++;
						}
					}
					if (is_block_start(ins.opcode.type))
					{
						depth++;
					}
				}
				return count == 1 ? found : std::nullopt;
			}

			// the F0 of a template without a specular label and without forward decals: its lighting's specular sum with F0
			// folded into literals, `mul t.xyz, b, l(1 - c, 1 - c, 1 - c, 0)` `mad u.xyz, a, l(c, c, c, 0), t.xyz` at the top
			// level (the glass shaders' (1 - F0) B + F0 A with F0 a register; lit_flag_transparent: 0.96 / 0.04)
			std::optional<float> folded_specular(const std::vector<detail::instruction_t>& instructions)
			{
				const auto replicated = [](const detail::operand_t& op) -> std::optional<float>
				{
					const auto c = literal_component(op, 0);
					if (op.components.type != D3D10_SB_OPERAND_4_COMPONENT || !c || *literal_component(op, 1) != *c
						|| *literal_component(op, 2) != *c || *literal_component(op, 3) != 0.0f)
					{
						return std::nullopt;
					}
					return c;
				};
				std::optional<float> found;
				auto count = 0;
				auto depth = 0;
				for (std::size_t i = 0; i + 1 < instructions.size(); i++)
				{
					const auto& ins = instructions[i];
					if (is_block_end(ins.opcode.type))
					{
						depth--;
					}
					const auto& next = instructions[i + 1];
					if (depth == 0 && ins.opcode.type == D3D10_SB_OPCODE_MUL && ins.operands.size() == 3
						&& next.opcode.type == D3D10_SB_OPCODE_MAD && next.operands.size() == 4)
					{
						const auto& t = ins.operands[0];
						const auto one_minus = replicated(ins.operands[2]);
						const auto c = replicated(next.operands[2]);
						const auto& addend = next.operands[3];
						const auto sum = t.type == D3D10_SB_OPERAND_TYPE_TEMP && components_of(t) == std::vector<std::uint32_t>{ 0, 1, 2 }
							&& components_of(next.operands[0]) == std::vector<std::uint32_t>{ 0, 1, 2 } && addend.type == D3D10_SB_OPERAND_TYPE_TEMP
							&& addend.extensions.empty() && addend.indices[0].value.uint32 == t.indices[0].value.uint32;
						if (sum && one_minus && c && *c > 0.0f && *c < 1.0f && std::fabs(*one_minus + *c - 1.0f) < 1e-6f)
						{
							found = *c;
							count++;
						}
					}
					if (is_block_start(ins.opcode.type))
					{
						depth++;
					}
				}
				return count == 1 ? found : std::nullopt;
			}

			// whether instruction `at` + 1 is `dp3 t, x, x` over the three components `dest` writes (a normalisation)
			bool normalised_next(const std::vector<detail::instruction_t>& instructions, const std::size_t at, const detail::operand_t& dest)
			{
				if (at + 1 >= instructions.size())
				{
					return false;
				}
				const auto& ins = instructions[at + 1];
				if (ins.opcode.type != D3D10_SB_OPCODE_DP3 || ins.operands.size() != 3)
				{
					return false;
				}
				const auto comps = components_of(dest);
				const auto same = [&](const detail::operand_t& op)
				{
					if (op.type != D3D10_SB_OPERAND_TYPE_TEMP || op.dimension != 1 || !op.extensions.empty()
						|| op.indices[0].value.uint32 != dest.indices[0].value.uint32 || op.components.type != D3D10_SB_OPERAND_4_COMPONENT
						|| op.components.selection_mode != D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE)
					{
						return false;
					}
					for (auto k = 0u; k < 3; k++)
					{
						if (op.components.names[k] != comps[k])
						{
							return false;
						}
					}
					return true;
				};
				return comps.size() == 3 && same(ins.operands[1]) && same(ins.operands[2]);
			}

			// A value one of the template's forward decal blends reads (see find_decal_blends)
			struct decal_blend
			{
				bool found = false;
				std::size_t index = 0; // the blend
				detail::operand_t dest{}; // its destination: its mask orders the value's components
				detail::operand_t value{}; // the template's own value it reads
			};

			struct decal_blends
			{
				decal_blend albedo, alpha, normal, gloss;
			};

			// the blends at the top level that read W = 1 - C (C the coverage) from W's write at `w_at` until its next write,
			// and how many there are (the constant F0 blend counted); null when one is not of the decal blend's layout
			std::optional<decal_blends> blends_of(const std::vector<detail::instruction_t>& instructions, const std::vector<int>& depth_of,
				const scalar& w, const scalar& c, const std::size_t w_at, std::uint32_t& count)
			{
				decal_blends out{};
				count = 0;
				for (auto i = w_at + 1; i < instructions.size() && !writes(instructions[i], w); i++)
				{
					const auto& ins = instructions[i];
					if (depth_of[i] != 0 || ins.opcode.type != D3D10_SB_OPCODE_MAD || ins.operands.size() != 4)
					{
						continue;
					}
					auto m = 0u;
					for (auto k = 1u; k <= 2; k++)
					{
						if (scalar_temp(ins.operands[k]) == w)
						{
							m = m ? 3u : k; // (3: both multiplicands are W, which no blend of this layout is)
						}
					}
					if (!m)
					{
						continue;
					}
					const auto& dest = ins.operands[0];
					const auto comps = components_of(dest);
					if (m == 3 || dest.type != D3D10_SB_OPERAND_TYPE_TEMP || dest.dimension != 1 || !dest.extensions.empty())
					{
						return std::nullopt;
					}
					count++;
					const auto& value = ins.operands[3 - m];
					decal_blend* blend = nullptr;
					if (comps.size() == 1)
					{
						blend = scalar_temp(ins.operands[3]) == c ? &out.alpha : &out.gloss;
					}
					else if (comps.size() == 3)
					{
						if (value.type == D3D10_SB_OPERAND_TYPE_IMMEDIATE32)
						{
							continue; // the constant F0 (constant_specular)
						}
						blend = normalised_next(instructions, i, dest) ? &out.normal : &out.albedo;
					}
					else
					{
						return std::nullopt;
					}
					if (blend->found)
					{
						return std::nullopt; // two blends of one kind
					}
					*blend = { true, i, dest, value };
				}
				if (!out.alpha.found)
				{
					return std::nullopt;
				}
				return out;
			}

			// Every BO3 forward template blends the forward decals over its own surface values after its decal loop, with
			// the decals' coverage C and W = 1 - C (`mov_sat C, C` `add W, -C, l(1)`), at the top level: the alpha
			// `mad a', a, W, C`; the albedo, the normal (normalised right after) and the gloss `mad v', v, W, decal` (either
			// multiplicand order, the albedo possibly a constant: glass_colorswatch's colorTint); the constant F0
			// `mad s, W, l(c, c, c, 0), decal`, which is constant_specular's. The value each blend reads is the template's
			// own before the decals, the one its debug-override label holds where it has one (glass: the same
			// registers). An alpha "over" of that shape also ends the glass shaders (the Fresnel term over the alpha, alone
			// on its W): the decal blend is the one alpha blend whose W takes other blends too. Null when there is no
			// such blend, more than one, or a blend on its W is not of this layout.
			std::optional<decal_blends> find_decal_blends(const std::vector<detail::instruction_t>& instructions)
			{
				struct candidate
				{
					scalar w;
					scalar c;
					std::size_t w_at;
				};
				std::vector<candidate> alpha_blends;
				std::vector<int> depth_of(instructions.size());
				auto depth = 0;
				for (std::size_t i = 0; i < instructions.size(); i++)
				{
					const auto& ins = instructions[i];
					if (is_block_end(ins.opcode.type))
					{
						depth--;
					}
					depth_of[i] = depth;
					if (depth == 0 && ins.opcode.type == D3D10_SB_OPCODE_MAD && ins.operands.size() == 4
						&& ins.operands[0].type == D3D10_SB_OPERAND_TYPE_TEMP && components_of(ins.operands[0]).size() == 1)
					{
						if (const auto addend = scalar_temp(ins.operands[3]))
						{
							for (auto m = 1u; m <= 2; m++)
							{
								const auto w = scalar_temp(ins.operands[m]);
								std::size_t w_at = 0;
								const auto c = w ? one_minus_source(instructions, i, *w, &w_at) : std::nullopt;
								if (c && *c == *addend)
								{
									alpha_blends.push_back({ *w, *c, w_at });
								}
							}
						}
					}
					if (is_block_start(ins.opcode.type))
					{
						depth++;
					}
				}

				std::optional<decal_blends> found;
				auto matches = 0;
				for (const auto& a : alpha_blends)
				{
					std::uint32_t count = 0;
					const auto blends = depth_of[a.w_at] == 0 ? blends_of(instructions, depth_of, a.w, a.c, a.w_at, count) : std::nullopt;
					if (blends && count >= 2)
					{
						found = blends;
						matches++;
					}
				}
				return matches == 1 ? found : std::nullopt;
			}

			// a DXBC container's program to change, and its resource definitions (which the writer does not keep)
			struct opened_program
			{
				shader_object obj;
				std::string rdef;
			};

			// the writer must give back the program it read before anything is changed
			opened_program open(const std::uint8_t* data, const std::size_t size)
			{
				const std::string original(reinterpret_cast<const char*>(data), size);
				opened_program out{ shader_object::parse(original), {} };
				const auto rdef = find_chunk(data, size, chunk_rdef);
				const auto shex = find_chunk(data, size, chunk_shex);
				if (!rdef || !shex)
				{
					throw std::runtime_error("the program has no RDEF or SHEX chunk");
				}
				out.rdef = *rdef;
				auto copy = out.obj;
				copy.get_unknown_chunks()[chunk_rdef] = out.rdef;
				const auto again = copy.serialize();
				const auto again_shex = find_chunk(reinterpret_cast<const std::uint8_t*>(again.data()), again.size(), chunk_shex);
				if (!again_shex || *again_shex != *shex)
				{
					throw std::runtime_error("the shader writer does not reproduce the program");
				}
				return out;
			}

			// the changed program in a container with its resource definitions, whose checksum verifies as the runtime computes it
			std::string close(opened_program& program)
			{
				program.obj.get_unknown_chunks()[chunk_rdef] = program.rdef;
				auto bytes = program.obj.serialize();
				const auto* p = reinterpret_cast<const std::uint8_t*>(bytes.data());
				std::array<std::uint32_t, 4> stored;
				std::memcpy(stored.data(), p + 4, 16);
				if (gpu_eval::container_checksum(p, bytes.size()) != stored)
				{
					throw std::runtime_error("the cut program's container checksum does not verify");
				}
				return bytes;
			}

			struct label
			{
				bool found = false;
				std::size_t index = 0;
				detail::operand_t dest{};
			};

			// where the surface program takes a role's value from
			struct source
			{
				enum class kind
				{
					none,
					label, // the debug-override blend's destination, after it
					decal_blend, // the value the forward decal blend reads, before it
					literal,
				};

				kind how = kind::none;
				std::size_t index = 0;
				detail::operand_t dest{}; // label, decal_blend: the blend's destination (its mask orders the components)
				detail::operand_t value{}; // decal_blend: the operand the blend reads
				float literal[4]{};
			};

			surface_program make(const std::uint8_t* data, const std::size_t size)
			{
				surface_program result{};
				if (size < 32 || std::memcmp(data, "DXBC", 4) != 0)
				{
					throw std::runtime_error("not a DXBC container");
				}

				// where the debug overrides live: PerSceneConsts' slot and each override's register
				const auto parsed = shader_eval::program::parse(data, size);
				const auto* scene = parsed->find_cbuffer("PerSceneConsts");
				std::optional<std::uint32_t> slot;
				for (const auto& b : parsed->bindings())
				{
					if (b.input_type == 0 && b.name == "PerSceneConsts")
					{
						slot = b.bind_point;
					}
				}
				if (!scene || !slot)
				{
					throw std::runtime_error("the permutation has no PerSceneConsts buffer");
				}
				std::uint32_t override_register[role_count];
				for (auto r = 0u; r < role_count; r++)
				{
					const shader_eval::cbuffer_variable* found = nullptr;
					for (const auto& v : scene->variables)
					{
						if (v.name == override_names[r])
						{
							found = &v;
						}
					}
					if (!found || found->offset % 16 != 0)
					{
						throw std::runtime_error(utils::string::va("PerSceneConsts has no %s", override_names[r]));
					}
					override_register[r] = found->offset / 16;
				}

				auto opened = open(data, size);
				auto& obj = opened.obj;

				// the labelled values: `mad dst, cb.override.w, ..., ...` at the top level; the occlusion's may sit in a
				// branch of if/else blocks (skin: its aoMap path, the other being BO3's screen-space GTAO)
				constexpr auto occlusion = static_cast<std::uint32_t>(role::occlusion);
				label labels[role_count]{};
				auto conditional_occlusion = false;
				auto& instructions = obj.get_instructions();
				std::optional<std::size_t> temps_at, output_at, code_at;
				std::uint32_t declared_targets = 0; // bit per render target the permutation declares
				std::vector<std::uint32_t> blocks; // the enclosing blocks' opcodes
				for (std::size_t i = 0; i < instructions.size(); i++)
				{
					const auto& ins = instructions[i];
					const auto type = ins.opcode.type;
					if (is_block_end(type))
					{
						blocks.pop_back();
					}
					if (type == D3D10_SB_OPCODE_DCL_TEMPS)
					{
						temps_at = i;
					}
					if (type == D3D10_SB_OPCODE_DCL_OUTPUT || type == D3D10_SB_OPCODE_DCL_OUTPUT_SIV || type == D3D10_SB_OPCODE_DCL_OUTPUT_SGV)
					{
						const auto& o = ins.operands[0];
						const auto target = o.indices[0].value.uint32;
						if (type != D3D10_SB_OPCODE_DCL_OUTPUT || o.type != D3D10_SB_OPERAND_TYPE_OUTPUT || target > 2 || (declared_targets & (1u << target)))
						{
							throw std::runtime_error("the permutation writes more than render targets 0-2");
						}
						declared_targets |= 1u << target;
						output_at = output_at ? output_at : i;
					}
					if (!code_at && !is_declaration(type))
					{
						code_at = i;
					}
					if (type == D3D10_SB_OPCODE_MAD && ins.operands.size() == 4)
					{
						if (const auto reg = weight_register(ins.operands[1], *slot))
						{
							for (auto r = 0u; r < role_count; r++)
							{
								if (override_register[r] != *reg)
								{
									continue;
								}
								const auto in_ifs = std::ranges::all_of(blocks, [](const std::uint32_t b) { return b == D3D10_SB_OPCODE_IF; });
								if (!blocks.empty() && (r != occlusion || !in_ifs))
								{
									throw std::runtime_error(utils::string::va("the %s blend sits inside control flow", override_names[r]));
								}
								if (labels[r].found)
								{
									throw std::runtime_error(utils::string::va("%s blends two values", override_names[r]));
								}
								labels[r] = { true, i, ins.operands[0] };
								conditional_occlusion |= !blocks.empty();
							}
						}
					}
					if (is_block_start(type))
					{
						blocks.push_back(type);
					}
				}
				result.conditional_occlusion = conditional_occlusion;

				constexpr auto specular = static_cast<std::uint32_t>(role::specular);
				source sources[role_count]{};
				const auto set_literal = [&](const std::uint32_t r, const float x, const float y, const float z, const float w)
				{
					sources[r].how = source::kind::literal;
					sources[r].literal[0] = x;
					sources[r].literal[1] = y;
					sources[r].literal[2] = z;
					sources[r].literal[3] = w;
					result.literal |= 1u << r;
				};
				for (auto r = 0u; r < role_count; r++)
				{
					if (labels[r].found)
					{
						sources[r] = { source::kind::label, labels[r].index, labels[r].dest };
						result.labelled |= 1u << r;
					}
				}
				if (!labels[specular].found)
				{
					result.constant_specular = constant_specular(instructions);
					if (!result.constant_specular)
					{
						result.constant_specular = folded_specular(instructions);
					}
					if (result.constant_specular)
					{
						const auto c = *result.constant_specular;
						set_literal(specular, c, c, c, 0.0f);
					}
				}

				// the roles without a label where the template blends its forward decals over them
				if (std::ranges::any_of(sources, [](const source& s) { return s.how == source::kind::none; }))
				{
					if (const auto blends = find_decal_blends(instructions))
					{
						const std::pair<role, const decal_blend*> blended[] = { { role::albedo, &blends->albedo },
							{ role::alpha, &blends->alpha }, { role::normal, &blends->normal }, { role::gloss, &blends->gloss } };
						for (const auto& [r, b] : blended)
						{
							const auto index = static_cast<std::uint32_t>(r);
							if (sources[index].how == source::kind::none && b->found)
							{
								sources[index] = { source::kind::decal_blend, b->index, b->dest, b->value };
								result.from_decal_blend |= 1u << index;
							}
						}

						// no albedo blend: albedo 0, as the blend keeps albedo x (1 - coverage) of any other
						// (glass_nocolor samples no colour and lights its decals' albedo accumulation as its albedo)
						constexpr auto albedo = static_cast<std::uint32_t>(role::albedo);
						if (sources[albedo].how == source::kind::none && !blends->albedo.found)
						{
							set_literal(albedo, 0.0f, 0.0f, 0.0f, 0.0f);
						}

						// a permutation without a single debug-override blend has no debug-override step, the only place
						// the occlusion shows: glass_nocolor_tile / glass_colorswatch_tile, whose debug and draw
						// programs are glass_nocolor's / glass_colorswatch's instruction for instruction but for the uv
						// scale and that step (theirs labels the occlusion as the literal 1: no occlusion map).
						// An occlusion map would be a texture of the lit shader that feeds no surface value, which the
						// caller rejects.
						if (sources[occlusion].how == source::kind::none && result.labelled == 0)
						{
							set_literal(occlusion, 1.0f, 1.0f, 1.0f, 1.0f);
						}
					}
				}

				// a specular or gloss the debug-override step does not label and no forward decal blend reads (hair):
				// where its lighting takes them, the F0 its $Globals specColorTint tints (the destination, after) and the gloss
				// its specular power raises 2 to (the mul's source, before)
				constexpr auto gloss = static_cast<std::uint32_t>(role::gloss);
				if (sources[specular].how == source::kind::none)
				{
					const auto* globals = parsed->find_cbuffer("$Globals");
					std::optional<std::uint32_t> globals_slot;
					for (const auto& b : parsed->bindings())
					{
						if (b.input_type == 0 && b.name == "$Globals")
						{
							globals_slot = b.bind_point;
						}
					}
					const shader_eval::cbuffer_variable* tint = nullptr;
					for (const auto& v : globals ? globals->variables : std::vector<shader_eval::cbuffer_variable>{})
					{
						if (v.name == "specColorTint")
						{
							tint = &v;
						}
					}
					if (globals_slot && tint && tint->offset % 16 == 0)
					{
						if (const auto at = specular_tint_anchor(instructions, *globals_slot, tint->offset / 16))
						{
							sources[specular] = { source::kind::label, *at, instructions[*at].operands[0] };
							result.anchored |= 1u << specular;
						}
					}
				}
				if (sources[gloss].how == source::kind::none)
				{
					if (const auto at = gloss_power_anchor(instructions, *slot, override_register, role_count))
					{
						sources[gloss] = { source::kind::decal_blend, at->first, instructions[at->first].operands[0], at->second };
						result.anchored |= 1u << gloss;
					}
				}

				std::string missing;
				std::size_t last = 0;
				for (auto r = 0u; r < role_count; r++)
				{
					const auto& s = sources[r];
					if (s.how == source::kind::label || s.how == source::kind::decal_blend)
					{
						const auto want = r == static_cast<std::uint32_t>(role::alpha) || r == static_cast<std::uint32_t>(role::gloss)
							|| r == occlusion ? 1u : 3u;
						if (components_of(s.dest).size() != want || s.dest.dimension != 1 || !s.dest.extensions.empty())
						{
							throw std::runtime_error(utils::string::va("the %s blend writes %zu components", role_names[r],
								components_of(s.dest).size()));
						}
						last = std::max(last, s.index);
					}
					else if (s.how == source::kind::none)
					{
						missing += missing.empty() ? role_names[r] : std::string(", ") + role_names[r];
					}
				}
				if (!missing.empty())
				{
					result.reason = "the debug-override permutation gives no " + missing + " (no debug-override blend, no forward decal blend)";
					return result;
				}
				if (!temps_at || !output_at)
				{
					throw std::runtime_error("the permutation declares no temps or no render target");
				}
				for (std::size_t i = 0; i <= last; i++)
				{
					if (instructions[i].opcode.type == D3D10_SB_OPCODE_RET || instructions[i].opcode.type == D3D10_SB_OPCODE_RETC)
					{
						throw std::runtime_error("the permutation returns before its last surface value");
					}
				}

				// every label's value is copied into a temp of its own right after it (a later instruction may reuse its
				// register), a label that writes an output register writing that temp instead (outputs are not read); a
				// decal blend's value is copied into its role's temp right before the blend reads it
				const auto first_temp = instructions[*temps_at].operands[0].custom.u.value;
				if (!(declared_targets & 1u) || !code_at)
				{
					throw std::runtime_error("the permutation declares no render target 0 or has no code");
				}
				std::vector<detail::instruction_t> out;
				for (std::size_t i = 0; i <= last; i++)
				{
					auto ins = instructions[i];
					if (i == *temps_at)
					{
						ins.operands[0].custom.u.value = first_temp + role_count;
					}
					if (i == *code_at && conditional_occlusion)
					{
						// an occlusion labelled in one branch only is 1 where that branch does not run (skin's screen-space
						// GTAO path: world_material_bake picks IW7's GTAO techset without an occlusion map for it)
						auto to = labels[occlusion].dest;
						to.type = D3D10_SB_OPERAND_TYPE_TEMP;
						to.indices[0].value.uint32 = first_temp + occlusion;
						out.push_back(mov(to, detail::create_literal_operand(1.0f, 1.0f, 1.0f, 1.0f)));
					}
					for (auto r = 0u; r < role_count; r++)
					{
						const auto& s = sources[r];
						if (s.how == source::kind::decal_blend && s.index == i)
						{
							auto to = s.dest;
							to.indices[0].value.uint32 = first_temp + r;
							out.push_back(mov(to, s.value));
						}
					}
					auto labelled_role = role_count;
					for (auto r = 0u; r < role_count; r++)
					{
						if (sources[r].how == source::kind::label && sources[r].index == i)
						{
							labelled_role = r;
						}
					}
					if (labelled_role == role_count)
					{
						out.push_back(std::move(ins));
						if (i == *output_at)
						{
							// the render targets of 0-2 the permutation does not declare
							for (auto target = 1u; target < 3; target++)
							{
								if (!(declared_targets & (1u << target)))
								{
									out.push_back(instructions[i]);
									out.back().operands[0].indices[0].value.uint32 = target;
								}
							}
						}
						continue;
					}

					const auto& dest = sources[labelled_role].dest;
					const auto temp = first_temp + labelled_role;
					if (dest.type == D3D10_SB_OPERAND_TYPE_OUTPUT)
					{
						ins.operands[0].type = D3D10_SB_OPERAND_TYPE_TEMP;
						ins.operands[0].indices[0].value.uint32 = temp;
						out.push_back(std::move(ins));
					}
					else if (dest.type == D3D10_SB_OPERAND_TYPE_TEMP)
					{
						out.push_back(std::move(ins));
						auto to = dest;
						to.indices[0].value.uint32 = temp;
						out.push_back(mov(to, detail::create_operand(D3D10_SB_OPERAND_TYPE_TEMP, std::string("xyzw"), { dest.indices[0].value.uint32 })));
					}
					else
					{
						throw std::runtime_error(utils::string::va("the %s blend writes neither a temp nor an output", override_names[labelled_role]));
					}
				}

				// the values into the render targets, then the end
				const auto put = [&](const role r, const std::uint32_t target, const std::uint32_t mask)
				{
					const auto index = static_cast<std::uint32_t>(r);
					const auto& s = sources[index];
					const auto to = detail::create_operand(D3D10_SB_OPERAND_TYPE_OUTPUT, mask, { target });
					if (s.how == source::kind::literal)
					{
						out.push_back(mov(to, detail::create_literal_operand(s.literal[0], s.literal[1], s.literal[2], s.literal[3])));
						return;
					}
					out.push_back(mov(to, detail::create_operand(D3D10_SB_OPERAND_TYPE_TEMP, swizzle_of(components_of(s.dest)), { first_temp + index })));
				};
				put(role::albedo, 0, 0x7);
				put(role::alpha, 0, 0x8);
				put(role::normal, 1, 0x7);
				put(role::gloss, 1, 0x8);
				put(role::specular, 2, 0x7);
				put(role::occlusion, 2, 0x8);
				out.push_back(detail::create_instruction(D3D10_SB_OPCODE_RET));

				// the declared render target writes all four components
				for (auto& ins : out)
				{
					if (ins.opcode.type == D3D10_SB_OPCODE_DCL_OUTPUT)
					{
						ins.operands[0].components.mask = 0xF;
					}
				}

				auto& osgn = obj.get_signatures()[chunk_osgn];
				std::uint32_t signed_targets = 0;
				for (auto& element : osgn)
				{
					if (_stricmp(element.name.data(), "SV_TARGET") || element.register_ != element.semantic_index || element.register_ > 2
						|| !(declared_targets & (1u << element.register_)) || (signed_targets & (1u << element.register_)))
					{
						throw std::runtime_error("the permutation's output signature is not render targets 0-2 as declared");
					}
					signed_targets |= 1u << element.register_;
					element.mask = 0xF;
				}
				if (signed_targets != declared_targets)
				{
					throw std::runtime_error("the permutation's output signature is not render targets 0-2 as declared");
				}
				const auto template_element = osgn.front();
				for (auto target = 1u; target < 3; target++)
				{
					if (!(signed_targets & (1u << target)))
					{
						auto element = template_element;
						element.semantic_index = target;
						element.register_ = target;
						osgn.push_back(element);
					}
				}
				std::ranges::sort(osgn, [](const auto& a, const auto& b) { return a.register_ < b.register_; });

				instructions = std::move(out);
				const auto bytes = close(opened);
				const auto* p = reinterpret_cast<const std::uint8_t*>(bytes.data());
				const auto cut = shader_eval::program::parse(p, bytes.size());
				auto targets = 0u;
				for (const auto& e : cut->outputs())
				{
					if (!_stricmp(e.semantic.data(), "SV_TARGET") && e.semantic_index < 3 && e.reg == e.semantic_index && e.mask == 0xF)
					{
						targets |= 1u << e.semantic_index;
					}
				}
				if (targets != 7)
				{
					throw std::runtime_error("the cut program does not declare render targets 0-2");
				}

				result.bytecode.assign(p, p + bytes.size());
				result.cut_at = static_cast<std::uint32_t>(last + 1);
				return result;
			}

			// the alpha-test coverage cut (coverage())
			surface_program make_coverage(const std::uint8_t* data, const std::size_t size)
			{
				surface_program result{};
				if (size < 32 || std::memcmp(data, "DXBC", 4) != 0)
				{
					throw std::runtime_error("not a DXBC container");
				}
				auto opened = open(data, size);
				auto& instructions = opened.obj.get_instructions();

				// the one discard, at the top level
				std::vector<int> depth_of(instructions.size());
				std::optional<std::size_t> discard_at;
				auto discards = 0;
				auto depth = 0;
				for (std::size_t i = 0; i < instructions.size(); i++)
				{
					const auto type = instructions[i].opcode.type;
					if (is_block_end(type))
					{
						depth--;
					}
					depth_of[i] = depth;
					if (type == D3D10_SB_OPCODE_DISCARD)
					{
						discards++;
						discard_at = depth == 0 ? std::optional<std::size_t>(i) : discard_at;
					}
					if (is_block_start(type))
					{
						depth++;
					}
				}
				if (discards != 1 || !discard_at)
				{
					result.reason = "the program does not discard exactly once, at the top level";
					return result;
				}

				// `discard_nz c` of `lt c, v, l(0.5)`: v is the coverage
				const auto& discard = instructions[*discard_at];
				const auto condition = discard.operands.empty() ? std::nullopt : scalar_temp(discard.operands[0]);
				std::optional<std::size_t> lt_at;
				for (auto i = *discard_at; condition && i-- > 0;)
				{
					if (writes(instructions[i], *condition))
					{
						lt_at = i;
						break;
					}
				}
				const auto* lt = lt_at ? &instructions[*lt_at] : nullptr;
				const auto threshold = lt && lt->operands.size() == 3 ? literal_component(lt->operands[2], condition->second) : std::nullopt;
				const auto value = lt && lt->operands.size() == 3 ? scalar_temp(lt->operands[1]) : std::nullopt;
				if (!(discard.opcode.controls & 0x80) || !lt || lt->opcode.type != D3D10_SB_OPCODE_LT || depth_of[*lt_at] != 0 || !threshold
					|| *threshold != 0.5f || !value)
				{
					result.reason = "the discard is not `discard_nz c` of `lt c, coverage, l(0.5)`";
					return result;
				}

				// the program up to the compare, then the coverage into all of render target 0
				std::vector<detail::instruction_t> out(instructions.begin(), instructions.begin() + static_cast<std::ptrdiff_t>(*lt_at));
				out.push_back(mov(detail::create_operand(D3D10_SB_OPERAND_TYPE_OUTPUT, 0xFu, { 0u }),
					detail::create_operand(D3D10_SB_OPERAND_TYPE_TEMP, std::string(4, "xyzw"[value->second]), { value->first })));
				out.push_back(detail::create_instruction(D3D10_SB_OPCODE_RET));
				auto target0 = false;
				for (auto& ins : out)
				{
					if (ins.opcode.type == D3D10_SB_OPCODE_DCL_OUTPUT && ins.operands[0].type == D3D10_SB_OPERAND_TYPE_OUTPUT
						&& ins.operands[0].indices[0].value.uint32 == 0)
					{
						ins.operands[0].components.mask = 0xF;
						target0 = true;
					}
				}
				auto signed0 = false;
				for (auto& element : opened.obj.get_signatures()[chunk_osgn])
				{
					if (!_stricmp(element.name.data(), "SV_TARGET") && element.register_ == 0 && element.semantic_index == 0)
					{
						element.mask = 0xF;
						signed0 = true;
					}
				}
				if (!target0 || !signed0)
				{
					result.reason = "the program has no render target 0";
					return result;
				}

				instructions = std::move(out);
				const auto bytes = close(opened);
				const auto* p = reinterpret_cast<const std::uint8_t*>(bytes.data());
				const auto cut = shader_eval::program::parse(p, bytes.size());
				const auto written = std::ranges::any_of(cut->outputs(), [](const shader_eval::signature_desc& e)
				{
					return !_stricmp(e.semantic.data(), "SV_TARGET") && e.semantic_index == 0 && e.reg == 0 && e.mask == 0xF;
				});
				if (!written)
				{
					throw std::runtime_error("the cut program does not declare render target 0");
				}
				result.bytecode.assign(p, p + bytes.size());
				result.cut_at = static_cast<std::uint32_t>(*lt_at);
				return result;
			}

			std::mutex mutex;
			std::unordered_map<const std::uint8_t*, std::unique_ptr<surface_program>> programs;
			std::unordered_map<const std::uint8_t*, std::unique_ptr<surface_program>> coverage_programs;
		}

		const surface_program& get(const std::uint8_t* bytecode, const std::size_t size)
		{
			std::lock_guard _(mutex);
			auto& entry = programs[bytecode];
			if (!entry)
			{
				entry = std::make_unique<surface_program>();
				try
				{
					*entry = make(bytecode, size);
				}
				catch (const std::exception& e)
				{
					*entry = {};
					entry->reason = e.what();
				}
			}
			return *entry;
		}

		const surface_program& coverage(const std::uint8_t* bytecode, const std::size_t size)
		{
			std::lock_guard _(mutex);
			auto& entry = coverage_programs[bytecode];
			if (!entry)
			{
				entry = std::make_unique<surface_program>();
				try
				{
					*entry = make_coverage(bytecode, size);
				}
				catch (const std::exception& e)
				{
					*entry = {};
					entry->reason = e.what();
				}
			}
			return *entry;
		}
	}
}
