#include <std_include.hpp>
#include "shader_eval.hpp"

#include <shader-tool/shader.hpp>
#include <utils/string.hpp>

#include <bit>
#include <cfenv>

// Instruction semantics follow the D3D11 functional specification (section 22, "Shader
// Instruction Reference"): comparison results are 0xFFFFFFFF / 0, min and max return the non-NaN
// operand, saturate maps NaN to 0, ftoi/ftou clamp and send NaN to 0, shifts use the low 5 bits
// of the shift amount, and the neg modifier on an integer operand is a two's complement negate.

namespace zonetool::t7
{
	namespace converter::iw7::shader_eval
	{
		using alys::shader::shader_object;
		namespace detail = alys::shader::detail;

		namespace
		{
			constexpr std::uint32_t chunk_rdef = 'FEDR';
			constexpr std::uint32_t chunk_isgn = 'NGSI';
			constexpr std::uint32_t chunk_osgn = 'NGSO';
			constexpr std::uint32_t chunk_osg5 = '5GSO';

			std::uint32_t read_u32(const std::uint8_t* data, std::size_t size, std::size_t offset)
			{
				if (offset + 4 > size)
				{
					throw std::runtime_error("shader: read past the end of a chunk");
				}

				std::uint32_t value;
				std::memcpy(&value, data + offset, 4);
				return value;
			}

			std::string read_string(const std::uint8_t* data, std::size_t size, std::size_t offset)
			{
				std::string value;
				for (auto i = offset; i < size && data[i]; i++)
				{
					value.push_back(static_cast<char>(data[i]));
				}
				return value;
			}

			// RDEF: header {cbuffer count, cbuffer offset, binding count, binding offset, minor, major,
			// type, flags, creator}, then for shader model 5 an RD11 block giving the descriptor sizes
			// (header, cbuffer, binding, variable, type, member). Offsets are from the chunk data.
			void parse_rdef(const std::uint8_t* data, std::size_t size, std::vector<cbuffer_desc>& cbuffers,
				std::vector<binding_desc>& bindings)
			{
				const auto cb_count = read_u32(data, size, 0);
				const auto cb_offset = read_u32(data, size, 4);
				const auto rb_count = read_u32(data, size, 8);
				const auto rb_offset = read_u32(data, size, 12);
				const auto major = data[17];

				std::uint32_t cb_desc_size = 24;
				std::uint32_t rb_desc_size = 32;
				std::uint32_t var_desc_size = 24;
				if (major >= 5)
				{
					if (read_u32(data, size, 28) != '11DR')
					{
						throw std::runtime_error("shader: shader model 5 RDEF without an RD11 block");
					}
					cb_desc_size = read_u32(data, size, 36);
					rb_desc_size = read_u32(data, size, 40);
					var_desc_size = read_u32(data, size, 44);
				}

				for (auto i = 0u; i < rb_count; i++)
				{
					const auto at = rb_offset + i * rb_desc_size;
					binding_desc binding{};
					binding.name = read_string(data, size, read_u32(data, size, at));
					binding.input_type = read_u32(data, size, at + 4);
					binding.dimension = read_u32(data, size, at + 12);
					binding.bind_point = read_u32(data, size, at + 20);
					binding.bind_count = read_u32(data, size, at + 24);
					bindings.emplace_back(std::move(binding));
				}

				for (auto i = 0u; i < cb_count; i++)
				{
					const auto at = cb_offset + i * cb_desc_size;
					cbuffer_desc cb{};
					cb.name = read_string(data, size, read_u32(data, size, at));
					const auto var_count = read_u32(data, size, at + 4);
					const auto var_offset = read_u32(data, size, at + 8);
					cb.size = read_u32(data, size, at + 12);
					for (auto v = 0u; v < var_count; v++)
					{
						const auto vat = var_offset + v * var_desc_size;
						cbuffer_variable var{};
						var.name = read_string(data, size, read_u32(data, size, vat));
						var.offset = read_u32(data, size, vat + 4);
						var.size = read_u32(data, size, vat + 8);
						cb.variables.emplace_back(std::move(var));
					}
					cbuffers.emplace_back(std::move(cb));
				}
			}

			std::vector<signature_desc> convert_signature(const shader_object::signature& sig)
			{
				std::vector<signature_desc> out;
				for (const auto& e : sig)
				{
					out.push_back({ e.name, e.semantic_index, e.system_value_type, e.component_type, e.register_, e.mask });
				}
				return out;
			}

			operand_kind kind_of(const std::uint32_t type)
			{
				switch (type)
				{
				case D3D10_SB_OPERAND_TYPE_NULL: return operand_kind::null;
				case D3D10_SB_OPERAND_TYPE_TEMP: return operand_kind::temp;
				case D3D10_SB_OPERAND_TYPE_INDEXABLE_TEMP: return operand_kind::indexable_temp;
				case D3D10_SB_OPERAND_TYPE_INPUT: return operand_kind::input;
				case D3D10_SB_OPERAND_TYPE_OUTPUT: return operand_kind::output;
				case D3D10_SB_OPERAND_TYPE_IMMEDIATE32: return operand_kind::immediate;
				case D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER: return operand_kind::cbuffer;
				case D3D10_SB_OPERAND_TYPE_IMMEDIATE_CONSTANT_BUFFER: return operand_kind::icb;
				case D3D10_SB_OPERAND_TYPE_RESOURCE: return operand_kind::resource;
				case D3D10_SB_OPERAND_TYPE_SAMPLER: return operand_kind::sampler;
				}
				return operand_kind::other;
			}

			operand convert_operand(const detail::operand_t& src)
			{
				operand op{};
				op.kind = kind_of(src.type);

				switch (src.components.type)
				{
				case D3D10_SB_OPERAND_0_COMPONENT:
					op.component_count = 0;
					break;
				case D3D10_SB_OPERAND_1_COMPONENT:
					op.component_count = 1;
					op.mask = 1;
					op.swizzle[0] = op.swizzle[1] = op.swizzle[2] = op.swizzle[3] = 0;
					break;
				case D3D10_SB_OPERAND_4_COMPONENT:
					op.component_count = 4;
					switch (src.components.selection_mode)
					{
					case D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE:
						op.mask = src.components.mask ? src.components.mask : 0xF;
						break;
					case D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE:
						op.mask = 0xF;
						for (auto i = 0; i < 4; i++)
						{
							op.swizzle[i] = src.components.names[i];
						}
						break;
					case D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE:
						op.mask = 1u << src.components.names[0];
						for (auto i = 0; i < 4; i++)
						{
							op.swizzle[i] = src.components.names[0];
						}
						break;
					}
					break;
				default:
					throw std::runtime_error("shader: operand with an N-component selection");
				}

				for (const auto& ext : src.extensions)
				{
					if (ext.type == D3D10_SB_EXTENDED_OPERAND_MODIFIER)
					{
						op.modifier = ext.modifier;
					}
				}

				for (auto i = 0u; i < src.dimension && i < 2; i++)
				{
					const auto& index = src.indices[i];
					switch (index.representation)
					{
					case D3D10_SB_OPERAND_INDEX_IMMEDIATE32:
						op.index[i] = index.value.uint32;
						break;
					case D3D10_SB_OPERAND_INDEX_RELATIVE:
					case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE:
					{
						op.index[i] = index.representation == D3D10_SB_OPERAND_INDEX_RELATIVE ? 0u : index.value.uint32;
						const auto& rel = *index.extra_operand;
						op.relative = true;
						op.relative_kind = kind_of(rel.type);
						op.relative_register = rel.indices[0].value.uint32;
						op.relative_component = rel.components.selection_mode == D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE
							? rel.components.names[0] : rel.components.names[0];
						// the relative index belongs to the last dimension the operand uses
						if (i + 1 != src.dimension)
						{
							throw std::runtime_error("shader: relative addressing on an inner index");
						}
						break;
					}
					default:
						throw std::runtime_error("shader: 64-bit operand index");
					}
				}
				if (src.dimension > 2)
				{
					throw std::runtime_error("shader: operand with three index dimensions");
				}

				if (op.kind == operand_kind::immediate)
				{
					const auto count = op.component_count == 4 ? 4 : 1;
					for (auto i = 0; i < 4; i++)
					{
						op.immediate[i] = src.immediate_values[i < count ? i : 0].uint32;
					}
				}

				return op;
			}

			float as_float(const std::uint32_t bits) { return std::bit_cast<float>(bits); }
			std::uint32_t as_bits(const float value) { return std::bit_cast<std::uint32_t>(value); }

			float saturate(const float value)
			{
				if (!(value > 0.0f))
				{
					return 0.0f; // also NaN
				}
				return value < 1.0f ? value : 1.0f;
			}

			std::uint32_t ftou(const float value)
			{
				if (std::isnan(value) || value <= 0.0f)
				{
					return 0;
				}
				if (value >= 4294967296.0f)
				{
					return 0xFFFFFFFFu;
				}
				return static_cast<std::uint32_t>(value);
			}

			std::int32_t ftoi(const float value)
			{
				if (std::isnan(value))
				{
					return 0;
				}
				if (value >= 2147483648.0f)
				{
					return std::numeric_limits<std::int32_t>::max();
				}
				if (value <= -2147483648.0f)
				{
					return std::numeric_limits<std::int32_t>::min();
				}
				return static_cast<std::int32_t>(value);
			}

			float round_even(const float value)
			{
				const auto previous = std::fegetround();
				std::fesetround(FE_TONEAREST);
				const auto result = std::nearbyint(value);
				std::fesetround(previous);
				return result;
			}

			float half_to_float(const std::uint32_t half)
			{
				const auto sign = (half >> 15) & 1u;
				const auto exponent = (half >> 10) & 0x1Fu;
				const auto mantissa = half & 0x3FFu;
				float value;
				if (exponent == 0)
				{
					value = std::ldexp(static_cast<float>(mantissa), -24);
				}
				else if (exponent == 31)
				{
					value = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
				}
				else
				{
					value = std::ldexp(static_cast<float>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
				}
				return sign ? -value : value;
			}

			std::uint32_t float_to_half(const float value)
			{
				const auto bits = as_bits(value);
				const auto sign = (bits >> 16) & 0x8000u;
				const auto exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
				const auto mantissa = bits & 0x7FFFFFu;
				if (((bits >> 23) & 0xFFu) == 0xFFu)
				{
					return sign | 0x7C00u | (mantissa ? 0x200u : 0u);
				}
				if (exponent >= 31)
				{
					return sign | 0x7C00u;
				}
				if (exponent <= 0)
				{
					if (exponent < -10)
					{
						return sign;
					}
					const auto m = mantissa | 0x800000u;
					const auto shift = static_cast<std::uint32_t>(14 - exponent);
					auto half = m >> shift;
					const auto rest = m & ((1u << shift) - 1u);
					const auto halfway = 1u << (shift - 1);
					if (rest > halfway || (rest == halfway && (half & 1u)))
					{
						half++;
					}
					return sign | half;
				}
				auto half = sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
				const auto rest = mantissa & 0x1FFFu;
				if (rest > 0x1000u || (rest == 0x1000u && (half & 1u)))
				{
					half++;
				}
				return half;
			}

			bool is_integer_op(const std::uint16_t opcode)
			{
				switch (opcode)
				{
				case D3D10_SB_OPCODE_AND:
				case D3D10_SB_OPCODE_OR:
				case D3D10_SB_OPCODE_XOR:
				case D3D10_SB_OPCODE_NOT:
				case D3D10_SB_OPCODE_IADD:
				case D3D10_SB_OPCODE_IMUL:
				case D3D10_SB_OPCODE_IMAD:
				case D3D10_SB_OPCODE_IMAX:
				case D3D10_SB_OPCODE_IMIN:
				case D3D10_SB_OPCODE_INEG:
				case D3D10_SB_OPCODE_IEQ:
				case D3D10_SB_OPCODE_INE:
				case D3D10_SB_OPCODE_IGE:
				case D3D10_SB_OPCODE_ILT:
				case D3D10_SB_OPCODE_ISHL:
				case D3D10_SB_OPCODE_ISHR:
				case D3D10_SB_OPCODE_USHR:
				case D3D10_SB_OPCODE_UDIV:
				case D3D10_SB_OPCODE_UMUL:
				case D3D10_SB_OPCODE_UMAD:
				case D3D10_SB_OPCODE_UMAX:
				case D3D10_SB_OPCODE_UMIN:
				case D3D10_SB_OPCODE_ULT:
				case D3D10_SB_OPCODE_UGE:
				case D3D10_SB_OPCODE_ITOF:
				case D3D10_SB_OPCODE_UTOF:
				case D3D11_SB_OPCODE_UBFE:
				case D3D11_SB_OPCODE_IBFE:
				case D3D11_SB_OPCODE_BFI:
				case D3D11_SB_OPCODE_BFREV:
				case D3D11_SB_OPCODE_COUNTBITS:
				case D3D11_SB_OPCODE_FIRSTBIT_HI:
				case D3D11_SB_OPCODE_FIRSTBIT_LO:
				case D3D11_SB_OPCODE_FIRSTBIT_SHI:
				case D3D11_SB_OPCODE_F16TOF32:
					return true;
				}
				return false;
			}
		}

		std::shared_ptr<program> program::parse(const std::uint8_t* data, std::size_t size)
		{
			if (size < 32 || std::memcmp(data, "DXBC", 4) != 0)
			{
				throw std::runtime_error("shader: not a DXBC container");
			}

			auto result = std::make_shared<program>();

			auto obj = shader_object::parse(std::string(reinterpret_cast<const char*>(data), size));

			result->inputs_ = convert_signature(obj.get_signature(chunk_isgn));
			auto outputs = obj.get_signatures();
			if (outputs.contains(chunk_osgn))
			{
				result->outputs_ = convert_signature(outputs[chunk_osgn]);
			}
			else if (outputs.contains(chunk_osg5))
			{
				result->outputs_ = convert_signature(outputs[chunk_osg5]);
			}

			// RDEF (parsed here, see the header comment)
			const auto chunk_count = read_u32(data, size, 28);
			for (auto i = 0u; i < chunk_count; i++)
			{
				const auto offset = read_u32(data, size, 32 + i * 4);
				if (read_u32(data, size, offset) == chunk_rdef)
				{
					const auto length = read_u32(data, size, offset + 4);
					if (offset + 8 + length > size)
					{
						throw std::runtime_error("shader: RDEF chunk runs past the container");
					}
					parse_rdef(data + offset + 8, length, result->cbuffers_, result->bindings_);
				}
			}

			const auto version = obj.get_version();
			for (const auto& ins : obj.get_instructions())
			{
				alys::utils::string_writer text;
				detail::dump_instruction(text, ins, version);
				const auto line = text.get_buffer();

				switch (ins.opcode.type)
				{
				case D3D10_SB_OPCODE_DCL_TEMPS:
					result->temp_count_ = ins.operands[0].custom.u.value;
					continue;
				case D3D10_SB_OPCODE_DCL_INDEXABLE_TEMP:
				{
					const auto reg = ins.operands[0].custom.u.value;
					if (result->indexable_temps_.size() <= reg)
					{
						result->indexable_temps_.resize(reg + 1);
					}
					result->indexable_temps_[reg] = { reg, ins.operands[1].custom.u.value };
					continue;
				}
				case D3D10_SB_OPCODE_CUSTOMDATA:
					if (ins.operands[0].custom.u.value == D3D10_SB_CUSTOMDATA_DCL_IMMEDIATE_CONSTANT_BUFFER)
					{
						for (auto i = 2u; i < ins.operands.size(); i++)
						{
							result->icb_.push_back(ins.operands[i].custom.u.value);
						}
					}
					continue;
				case D3D10_SB_OPCODE_DCL_INPUT_PS:
				case D3D10_SB_OPCODE_DCL_INPUT_PS_SIV:
				case D3D10_SB_OPCODE_DCL_INPUT_PS_SGV:
				{
					// the interpolation mode (the opcode's control bits) of the input elements in the declared register
					const auto& reg = ins.operands[0];
					const auto mask = reg.components.selection_mode == D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE && reg.components.mask
						? static_cast<std::uint32_t>(reg.components.mask) : 0xFu;
					for (auto& e : result->inputs_)
					{
						if (e.reg == reg.indices[0].value.uint32 && (e.mask & mask))
						{
							e.interpolation = ins.opcode.controls & 0xFu;
						}
					}
					continue;
				}
				}

				if (ins.opcode.type >= D3D10_SB_OPCODE_DCL_RESOURCE && ins.opcode.type <= D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS)
				{
					continue;
				}
				if (ins.opcode.type >= D3D11_SB_OPCODE_DCL_STREAM && ins.opcode.type <= D3D11_SB_OPCODE_DCL_RESOURCE_STRUCTURED)
				{
					continue;
				}

				instruction out{};
				out.opcode = ins.opcode.type;
				out.saturate = (ins.opcode.controls & 0x4) != 0;
				out.test_nonzero = (ins.opcode.controls & 0x80) != 0;
				for (const auto& ext : ins.opcode.extensions)
				{
					if (ext.type == D3D10_SB_EXTENDED_OPCODE_SAMPLE_CONTROLS)
					{
						for (auto i = 0; i < 3; i++)
						{
							// 4-bit two's complement immediates
							const auto v = static_cast<std::int32_t>(ext.values[i] & 0xF);
							out.offsets[i] = static_cast<std::int8_t>(v >= 8 ? v - 16 : v);
						}
					}
				}
				for (const auto& op : ins.operands)
				{
					out.operands.emplace_back(convert_operand(op));
				}

				result->instructions_.emplace_back(std::move(out));
				result->text_.emplace_back(line);
			}

			return result;
		}

		const binding_desc* program::find_binding(const std::string& name, const std::uint32_t input_type) const
		{
			for (const auto& b : this->bindings_)
			{
				if (b.input_type == input_type && b.name == name)
				{
					return &b;
				}
			}
			return nullptr;
		}

		const cbuffer_desc* program::find_cbuffer(const std::string& name) const
		{
			for (const auto& cb : this->cbuffers_)
			{
				if (cb.name == name)
				{
					return &cb;
				}
			}
			return nullptr;
		}

		const signature_desc* program::find_input(const std::string& semantic, const std::uint32_t semantic_index) const
		{
			for (const auto& e : this->inputs_)
			{
				if (e.semantic_index == semantic_index && _stricmp(e.semantic.data(), semantic.data()) == 0)
				{
					return &e;
				}
			}
			return nullptr;
		}

		std::string program::describe(const std::size_t instruction_index) const
		{
			return instruction_index < this->text_.size() ? this->text_[instruction_index] : std::string{};
		}

		machine::machine(std::shared_ptr<program> prog)
			: program_(std::move(prog))
		{
			const auto& p = *this->program_;
			this->temps_.resize(p.temp_count());
			this->indexable_.resize(p.indexable_temps().size());
			for (auto i = 0u; i < p.indexable_temps().size(); i++)
			{
				this->indexable_[i].resize(p.indexable_temps()[i].second);
			}

			std::uint32_t max_input = 0, max_output = 0;
			for (const auto& e : p.inputs())
			{
				max_input = std::max(max_input, e.reg + 1);
			}
			for (const auto& e : p.outputs())
			{
				max_output = std::max(max_output, e.reg + 1);
			}
			this->inputs_.resize(max_input);
			this->outputs_.resize(max_output);
			for (auto& reg : this->inputs_)
			{
				for (auto& c : reg)
				{
					c.fill(0);
				}
			}
		}

		void machine::bind_cbuffer(const std::uint32_t slot, const void* data, const std::size_t size)
		{
			if (this->cbuffers_.size() <= slot)
			{
				this->cbuffers_.resize(slot + 1);
			}
			auto& cb = this->cbuffers_[slot];
			cb.assign((size + 3) / 4, 0);
			std::memcpy(cb.data(), data, size);
		}

		void machine::bind_texture(const std::uint32_t slot, const texture_source* source)
		{
			if (this->textures_.size() <= slot)
			{
				this->textures_.resize(slot + 1, nullptr);
			}
			this->textures_[slot] = source;
		}

		void machine::bind_structured(const std::uint32_t slot, const void* data, const std::size_t size, const std::uint32_t stride)
		{
			auto& buffer = this->buffers_[slot];
			buffer.data.assign(static_cast<const std::uint8_t*>(data), static_cast<const std::uint8_t*>(data) + size);
			buffer.stride = stride;
		}

		void machine::set_input(const std::uint32_t reg, const std::uint32_t component, const std::uint32_t lane, const std::uint32_t bits)
		{
			if (reg >= this->inputs_.size())
			{
				this->inputs_.resize(reg + 1);
			}
			this->inputs_[reg][component][lane] = bits;
		}

		void machine::set_input(const std::uint32_t reg, const std::uint32_t component, const std::uint32_t lane, const float value)
		{
			this->set_input(reg, component, lane, as_bits(value));
		}

		float machine::output(const std::uint32_t reg, const std::uint32_t component, const std::uint32_t lane) const
		{
			return as_float(this->outputs_.at(reg)[component][lane]);
		}

		std::uint32_t machine::output_bits(const std::uint32_t reg, const std::uint32_t component, const std::uint32_t lane) const
		{
			return this->outputs_.at(reg)[component][lane];
		}

		void machine::fail(const std::size_t index, const char* why) const
		{
			throw std::runtime_error(utils::string::va("shader: %s at instruction %zu (%s)", why, index,
				this->program_->describe(index).data()));
		}

		std::uint32_t machine::read_index(const operand& op, const std::uint32_t lane) const
		{
			if (!op.relative)
			{
				return op.index[1];
			}

			std::uint32_t base = 0;
			switch (op.relative_kind)
			{
			case operand_kind::temp:
				base = this->temps_.at(op.relative_register)[op.relative_component][lane];
				break;
			case operand_kind::input:
				base = this->inputs_.at(op.relative_register)[op.relative_component][lane];
				break;
			default:
				throw std::runtime_error("shader: unsupported relative address register");
			}
			return base + op.index[1];
		}

		void machine::fetch(const operand& op, reg4& out, const bool integer) const
		{
			switch (op.kind)
			{
			case operand_kind::temp:
			{
				const auto& reg = this->temps_.at(op.index[0]);
				for (auto c = 0; c < 4; c++)
				{
					out[c] = reg[op.swizzle[c]];
				}
				break;
			}
			case operand_kind::input:
			{
				const auto& reg = this->inputs_.at(op.index[0]);
				for (auto c = 0; c < 4; c++)
				{
					out[c] = reg[op.swizzle[c]];
				}
				break;
			}
			case operand_kind::output:
			{
				const auto& reg = this->outputs_.at(op.index[0]);
				for (auto c = 0; c < 4; c++)
				{
					out[c] = reg[op.swizzle[c]];
				}
				break;
			}
			case operand_kind::immediate:
				for (auto c = 0; c < 4; c++)
				{
					out[c].fill(op.immediate[op.component_count == 4 ? c : 0]);
				}
				break;
			case operand_kind::cbuffer:
			{
				const auto slot = op.index[0];
				static const std::vector<std::uint32_t> empty;
				const auto& cb = slot < this->cbuffers_.size() ? this->cbuffers_[slot] : empty;
				for (auto lane = 0u; lane < lane_count; lane++)
				{
					const auto element = this->read_index(op, lane);
					for (auto c = 0; c < 4; c++)
					{
						const auto at = element * 4 + op.swizzle[c];
						out[c][lane] = at < cb.size() ? cb[at] : 0;
					}
				}
				break;
			}
			case operand_kind::icb:
			{
				const auto& icb = this->program_->immediate_cbuffer();
				for (auto lane = 0u; lane < lane_count; lane++)
				{
					std::uint32_t element = op.index[0];
					if (op.relative)
					{
						element = op.index[0] + (op.relative_kind == operand_kind::temp
							? this->temps_.at(op.relative_register)[op.relative_component][lane]
							: this->inputs_.at(op.relative_register)[op.relative_component][lane]);
					}
					for (auto c = 0; c < 4; c++)
					{
						const auto at = element * 4 + op.swizzle[c];
						out[c][lane] = at < icb.size() ? icb[at] : 0;
					}
				}
				break;
			}
			case operand_kind::indexable_temp:
			{
				const auto& arr = this->indexable_.at(op.index[0]);
				for (auto lane = 0u; lane < lane_count; lane++)
				{
					const auto element = this->read_index(op, lane);
					for (auto c = 0; c < 4; c++)
					{
						out[c][lane] = element < arr.size() ? arr[element][op.swizzle[c]][lane] : 0;
					}
				}
				break;
			}
			default:
				throw std::runtime_error("shader: unsupported source operand");
			}

			if (op.modifier)
			{
				for (auto c = 0; c < 4; c++)
				{
					for (auto lane = 0u; lane < lane_count; lane++)
					{
						auto& v = out[c][lane];
						if (integer)
						{
							if (op.modifier == D3D10_SB_OPERAND_MODIFIER_NEG)
							{
								v = static_cast<std::uint32_t>(-static_cast<std::int32_t>(v));
							}
							else
							{
								throw std::runtime_error("shader: abs modifier on an integer operand");
							}
						}
						else
						{
							if (op.modifier & D3D10_SB_OPERAND_MODIFIER_ABS)
							{
								v &= 0x7FFFFFFFu;
							}
							if (op.modifier & D3D10_SB_OPERAND_MODIFIER_NEG)
							{
								v ^= 0x80000000u;
							}
						}
					}
				}
			}
		}

		void machine::store(const operand& op, const reg4& value, const bool saturate_result)
		{
			reg4* target = nullptr;
			switch (op.kind)
			{
			case operand_kind::null:
				return;
			case operand_kind::temp:
				target = &this->temps_.at(op.index[0]);
				break;
			case operand_kind::output:
				target = &this->outputs_.at(op.index[0]);
				break;
			case operand_kind::indexable_temp:
			{
				auto& arr = this->indexable_.at(op.index[0]);
				for (auto lane = 0u; lane < lane_count; lane++)
				{
					if (!(this->active_ & (lane_mask(1) << lane)))
					{
						continue;
					}
					const auto element = this->read_index(op, lane);
					if (element >= arr.size())
					{
						continue;
					}
					for (auto c = 0; c < 4; c++)
					{
						if (op.mask & (1u << c))
						{
							auto bits = value[c][lane];
							if (saturate_result)
							{
								bits = as_bits(saturate(as_float(bits)));
							}
							arr[element][c][lane] = bits;
						}
					}
				}
				return;
			}
			default:
				throw std::runtime_error("shader: unsupported destination operand");
			}

			for (auto c = 0; c < 4; c++)
			{
				if (!(op.mask & (1u << c)))
				{
					continue;
				}
				auto& dst = (*target)[c];
				const auto& src = value[c];
				for (auto lane = 0u; lane < lane_count; lane++)
				{
					if (this->active_ & (lane_mask(1) << lane))
					{
						dst[lane] = saturate_result ? as_bits(saturate(as_float(src[lane]))) : src[lane];
					}
				}
			}
		}

		void machine::sample(const instruction& ins, const std::size_t index)
		{
			const auto opcode = ins.opcode;
			const auto& dst = ins.operands[0];
			const auto& res = ins.operands[2];

			if (res.kind != operand_kind::resource)
			{
				this->fail(index, "sample from a non-texture resource");
			}
			const auto slot = res.index[0];
			const auto* source = slot < this->textures_.size() ? this->textures_[slot] : nullptr;
			if (!source)
			{
				if (this->unbound_zero_)
				{
					this->store(dst, reg4{}, false);
					return;
				}
				this->fail(index, "sample from an unbound texture");
			}

			reg4 coord{};
			this->fetch(ins.operands[1], coord, opcode == D3D10_SB_OPCODE_LD);

			if (this->probing_ && opcode != D3D10_SB_OPCODE_LD && (this->active_ & 0xF) == 0xF)
			{
				this->probes_.push_back({ slot,
					as_float(coord[0][1]) - as_float(coord[0][0]), as_float(coord[1][1]) - as_float(coord[1][0]),
					as_float(coord[0][2]) - as_float(coord[0][0]), as_float(coord[1][2]) - as_float(coord[1][0]),
					as_float(coord[0][0]), as_float(coord[1][0]), ins.offsets[0], ins.offsets[1] });
			}

			reg4 texel{};
			if (opcode == D3D10_SB_OPCODE_LD)
			{
				for (auto lane = 0u; lane < lane_count; lane++)
				{
					float out[4];
					source->load(static_cast<std::int32_t>(coord[0][lane]) + ins.offsets[0],
						static_cast<std::int32_t>(coord[1][lane]) + ins.offsets[1],
						static_cast<std::int32_t>(coord[3][lane]), out);
					for (auto c = 0; c < 4; c++)
					{
						texel[c][lane] = as_bits(out[c]);
					}
				}
			}
			else
			{
				std::uint32_t width, height, levels;
				source->dimensions(0, width, height, levels);

				std::array<float, lane_count> lod{};
				if (opcode == D3D10_SB_OPCODE_SAMPLE || opcode == D3D10_SB_OPCODE_SAMPLE_B)
				{
					// implicit derivatives inside each 2x2 quad (coarse, as the hardware selects
					// one level of detail per quad)
					for (auto quad = 0u; quad < lane_count / 4; quad++)
					{
						const auto l0 = quad * 4;
						const auto dudx = (as_float(coord[0][l0 + 1]) - as_float(coord[0][l0])) * width;
						const auto dvdx = (as_float(coord[1][l0 + 1]) - as_float(coord[1][l0])) * height;
						const auto dudy = (as_float(coord[0][l0 + 2]) - as_float(coord[0][l0])) * width;
						const auto dvdy = (as_float(coord[1][l0 + 2]) - as_float(coord[1][l0])) * height;
						const auto rho = std::max(std::sqrt(dudx * dudx + dvdx * dvdx), std::sqrt(dudy * dudy + dvdy * dvdy));
						const auto level = rho > 0.0f ? std::log2(rho) : -1000.0f;
						for (auto k = 0u; k < 4; k++)
						{
							lod[l0 + k] = level;
						}
					}
					if (opcode == D3D10_SB_OPCODE_SAMPLE_B)
					{
						reg4 bias{};
						this->fetch(ins.operands[4], bias, false);
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							lod[lane] += as_float(bias[0][lane]);
						}
					}
				}
				else if (opcode == D3D10_SB_OPCODE_SAMPLE_L)
				{
					reg4 level{};
					this->fetch(ins.operands[4], level, false);
					for (auto lane = 0u; lane < lane_count; lane++)
					{
						lod[lane] = as_float(level[0][lane]);
					}
				}
				else if (opcode == D3D10_SB_OPCODE_SAMPLE_D)
				{
					reg4 ddx{}, ddy{};
					this->fetch(ins.operands[4], ddx, false);
					this->fetch(ins.operands[5], ddy, false);
					for (auto lane = 0u; lane < lane_count; lane++)
					{
						const auto dudx = as_float(ddx[0][lane]) * width;
						const auto dvdx = as_float(ddx[1][lane]) * height;
						const auto dudy = as_float(ddy[0][lane]) * width;
						const auto dvdy = as_float(ddy[1][lane]) * height;
						const auto rho = std::max(std::sqrt(dudx * dudx + dvdx * dvdx), std::sqrt(dudy * dudy + dvdy * dvdy));
						lod[lane] = rho > 0.0f ? std::log2(rho) : -1000.0f;
					}
				}
				else
				{
					this->fail(index, "unsupported sample variant");
				}

				for (auto lane = 0u; lane < lane_count; lane++)
				{
					float out[4];
					source->sample(as_float(coord[0][lane]), as_float(coord[1][lane]), std::max(lod[lane], 0.0f),
						ins.offsets[0], ins.offsets[1], out);
					for (auto c = 0; c < 4; c++)
					{
						texel[c][lane] = as_bits(out[c]);
					}
				}
			}

			// the resource operand's swizzle picks the returned components
			reg4 result{};
			for (auto c = 0; c < 4; c++)
			{
				result[c] = texel[res.swizzle[c]];
			}
			this->store(dst, result, ins.saturate);
		}

		void machine::run()
		{
			const auto& code = this->program_->instructions();

			struct if_frame
			{
				lane_mask restore;
				lane_mask else_lanes;
			};

			struct loop_frame
			{
				std::size_t start;
				lane_mask restore;
				lane_mask broken;
				lane_mask continued;
				std::uint32_t iterations;
			};

			std::vector<if_frame> ifs;
			std::vector<loop_frame> loops;
			lane_mask finished = 0; // returned or discarded

			this->active_ = all_lanes;
			this->discarded_ = 0;
			for (auto& reg : this->outputs_)
			{
				for (auto& c : reg)
				{
					c.fill(0);
				}
			}

			const auto lanes_where = [&](const operand& op, const bool nonzero) -> lane_mask
			{
				reg4 v{};
				this->fetch(op, v, true);
				lane_mask m = 0;
				for (auto lane = 0u; lane < lane_count; lane++)
				{
					if ((v[0][lane] != 0) == nonzero)
					{
						m |= lane_mask(1) << lane;
					}
				}
				return m;
			};

			const auto skip_to_matching = [&](std::size_t i, const bool stop_at_else) -> std::size_t
			{
				auto depth = 0;
				for (auto k = i + 1; k < code.size(); k++)
				{
					const auto op = code[k].opcode;
					if (op == D3D10_SB_OPCODE_IF)
					{
						depth++;
					}
					else if (op == D3D10_SB_OPCODE_ENDIF)
					{
						if (depth == 0)
						{
							return k;
						}
						depth--;
					}
					else if (op == D3D10_SB_OPCODE_ELSE && depth == 0 && stop_at_else)
					{
						return k;
					}
				}
				throw std::runtime_error("shader: unterminated if");
			};

			const auto end_of_loop = [&](std::size_t i) -> std::size_t
			{
				auto depth = 0;
				for (auto k = i + 1; k < code.size(); k++)
				{
					if (code[k].opcode == D3D10_SB_OPCODE_LOOP)
					{
						depth++;
					}
					else if (code[k].opcode == D3D10_SB_OPCODE_ENDLOOP)
					{
						if (depth == 0)
						{
							return k;
						}
						depth--;
					}
				}
				throw std::runtime_error("shader: unterminated loop");
			};

			reg4 a{}, b{}, c{}, d{}, r0{}, r1{};

			for (std::size_t i = 0; i < code.size(); i++)
			{
				const auto& ins = code[i];
				const auto op = ins.opcode;

				// control flow is evaluated even when no lane is active, to keep the stacks right
				switch (op)
				{
				case D3D10_SB_OPCODE_IF:
				{
					const auto taken = lanes_where(ins.operands[0], ins.test_nonzero) & this->active_;
					ifs.push_back({ this->active_, this->active_ & ~taken });
					this->active_ = taken;
					if (!this->active_)
					{
						// jump straight to the else (or endif) branch
						i = skip_to_matching(i, true) - 1;
					}
					continue;
				}
				case D3D10_SB_OPCODE_ELSE:
				{
					if (ifs.empty())
					{
						this->fail(i, "else without if");
					}
					this->active_ = ifs.back().else_lanes & ~finished;
					for (const auto& loop : loops)
					{
						this->active_ &= ~(loop.broken | loop.continued);
					}
					if (!this->active_)
					{
						i = skip_to_matching(i, false) - 1;
					}
					continue;
				}
				case D3D10_SB_OPCODE_ENDIF:
				{
					if (ifs.empty())
					{
						this->fail(i, "endif without if");
					}
					this->active_ = ifs.back().restore & ~finished;
					for (const auto& loop : loops)
					{
						this->active_ &= ~(loop.broken | loop.continued);
					}
					ifs.pop_back();
					continue;
				}
				case D3D10_SB_OPCODE_LOOP:
					loops.push_back({ i, this->active_, 0, 0, 0 });
					if (!this->active_)
					{
						i = end_of_loop(i);
						this->active_ = loops.back().restore & ~finished;
						loops.pop_back();
					}
					continue;
				case D3D10_SB_OPCODE_BREAK:
				case D3D10_SB_OPCODE_BREAKC:
				{
					if (loops.empty())
					{
						this->fail(i, "break outside a loop");
					}
					auto lanes = this->active_;
					if (op == D3D10_SB_OPCODE_BREAKC)
					{
						lanes &= lanes_where(ins.operands[0], ins.test_nonzero);
					}
					loops.back().broken |= lanes;
					this->active_ &= ~lanes;
					continue;
				}
				case D3D10_SB_OPCODE_CONTINUE:
				case D3D10_SB_OPCODE_CONTINUEC:
				{
					if (loops.empty())
					{
						this->fail(i, "continue outside a loop");
					}
					auto lanes = this->active_;
					if (op == D3D10_SB_OPCODE_CONTINUEC)
					{
						lanes &= lanes_where(ins.operands[0], ins.test_nonzero);
					}
					loops.back().continued |= lanes;
					this->active_ &= ~lanes;
					continue;
				}
				case D3D10_SB_OPCODE_ENDLOOP:
				{
					if (loops.empty())
					{
						this->fail(i, "endloop without loop");
					}
					auto& loop = loops.back();
					const auto running = (this->active_ | loop.continued) & ~finished;
					loop.continued = 0;
					if (running && ++loop.iterations < 1u << 20)
					{
						this->active_ = running;
						i = loop.start;
						continue;
					}
					if (running)
					{
						this->fail(i, "loop did not terminate");
					}
					this->active_ = loop.restore & ~finished;
					loops.pop_back();
					for (const auto& outer : loops)
					{
						this->active_ &= ~(outer.broken | outer.continued);
					}
					continue;
				}
				case D3D10_SB_OPCODE_RET:
					finished |= this->active_;
					this->active_ = 0;
					continue;
				case D3D10_SB_OPCODE_RETC:
				{
					const auto lanes = this->active_ & lanes_where(ins.operands[0], ins.test_nonzero);
					finished |= lanes;
					this->active_ &= ~lanes;
					continue;
				}
				case D3D10_SB_OPCODE_DISCARD:
				{
					const auto lanes = this->active_ & lanes_where(ins.operands[0], ins.test_nonzero);
					this->discarded_ |= lanes;
					// a discarded pixel keeps running as a helper for its quad's derivatives, but
					// nothing it computes afterwards is written
					continue;
				}
				case D3D10_SB_OPCODE_NOP:
				case D3D10_SB_OPCODE_LABEL:
					continue;
				case D3D10_SB_OPCODE_SWITCH:
				case D3D10_SB_OPCODE_CASE:
				case D3D10_SB_OPCODE_DEFAULT:
				case D3D10_SB_OPCODE_ENDSWITCH:
				case D3D10_SB_OPCODE_CALL:
				case D3D10_SB_OPCODE_CALLC:
					this->fail(i, "unsupported control flow");
				}

				if (!this->active_)
				{
					continue;
				}

				const auto integer = is_integer_op(op);
				const auto& ops = ins.operands;

				const auto unary_float = [&](auto fn)
				{
					this->fetch(ops[1], a, false);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = as_bits(fn(as_float(a[k][lane])));
						}
					}
					this->store(ops[0], r0, ins.saturate);
				};

				const auto binary_float = [&](auto fn)
				{
					this->fetch(ops[1], a, false);
					this->fetch(ops[2], b, false);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = as_bits(fn(as_float(a[k][lane]), as_float(b[k][lane])));
						}
					}
					this->store(ops[0], r0, ins.saturate);
				};

				const auto compare_float = [&](auto fn)
				{
					this->fetch(ops[1], a, false);
					this->fetch(ops[2], b, false);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = fn(as_float(a[k][lane]), as_float(b[k][lane])) ? 0xFFFFFFFFu : 0u;
						}
					}
					this->store(ops[0], r0, false);
				};

				const auto unary_int = [&](auto fn)
				{
					this->fetch(ops[1], a, true);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = fn(a[k][lane]);
						}
					}
					this->store(ops[0], r0, false);
				};

				const auto binary_int = [&](auto fn)
				{
					this->fetch(ops[1], a, true);
					this->fetch(ops[2], b, true);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = fn(a[k][lane], b[k][lane]);
						}
					}
					this->store(ops[0], r0, false);
				};

				const auto dot = [&](const int n)
				{
					this->fetch(ops[1], a, false);
					this->fetch(ops[2], b, false);
					for (auto lane = 0u; lane < lane_count; lane++)
					{
						auto sum = 0.0f;
						for (auto k = 0; k < n; k++)
						{
							sum += as_float(a[k][lane]) * as_float(b[k][lane]);
						}
						const auto bits = as_bits(sum);
						r0[0][lane] = r0[1][lane] = r0[2][lane] = r0[3][lane] = bits;
					}
					this->store(ops[0], r0, ins.saturate);
				};

				switch (op)
				{
				case D3D10_SB_OPCODE_MOV:
					this->fetch(ops[1], a, false);
					this->store(ops[0], a, ins.saturate);
					break;
				case D3D10_SB_OPCODE_MOVC:
					this->fetch(ops[1], a, true);
					this->fetch(ops[2], b, false);
					this->fetch(ops[3], c, false);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = a[k][lane] ? b[k][lane] : c[k][lane];
						}
					}
					this->store(ops[0], r0, ins.saturate);
					break;
				case D3D10_SB_OPCODE_ADD: binary_float([](float x, float y) { return x + y; }); break;
				case D3D10_SB_OPCODE_MUL: binary_float([](float x, float y) { return x * y; }); break;
				case D3D10_SB_OPCODE_DIV: binary_float([](float x, float y) { return x / y; }); break;
				case D3D10_SB_OPCODE_MIN: binary_float([](float x, float y) { return std::fmin(x, y); }); break;
				case D3D10_SB_OPCODE_MAX: binary_float([](float x, float y) { return std::fmax(x, y); }); break;
				case D3D10_SB_OPCODE_MAD:
					this->fetch(ops[1], a, false);
					this->fetch(ops[2], b, false);
					this->fetch(ops[3], c, false);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = as_bits(as_float(a[k][lane]) * as_float(b[k][lane]) + as_float(c[k][lane]));
						}
					}
					this->store(ops[0], r0, ins.saturate);
					break;
				case D3D10_SB_OPCODE_DP2: dot(2); break;
				case D3D10_SB_OPCODE_DP3: dot(3); break;
				case D3D10_SB_OPCODE_DP4: dot(4); break;
				case D3D10_SB_OPCODE_EXP: unary_float([](float x) { return std::exp2(x); }); break;
				case D3D10_SB_OPCODE_LOG: unary_float([](float x) { return std::log2(x); }); break;
				case D3D10_SB_OPCODE_RSQ: unary_float([](float x) { return 1.0f / std::sqrt(x); }); break;
				case D3D11_SB_OPCODE_RCP: unary_float([](float x) { return 1.0f / x; }); break;
				case D3D10_SB_OPCODE_SQRT: unary_float([](float x) { return std::sqrt(x); }); break;
				case D3D10_SB_OPCODE_FRC: unary_float([](float x) { return x - std::floor(x); }); break;
				case D3D10_SB_OPCODE_ROUND_NE: unary_float([](float x) { return round_even(x); }); break;
				case D3D10_SB_OPCODE_ROUND_NI: unary_float([](float x) { return std::floor(x); }); break;
				case D3D10_SB_OPCODE_ROUND_PI: unary_float([](float x) { return std::ceil(x); }); break;
				case D3D10_SB_OPCODE_ROUND_Z: unary_float([](float x) { return std::trunc(x); }); break;
				case D3D10_SB_OPCODE_SINCOS:
					this->fetch(ops[2], a, false);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							const auto x = as_float(a[k][lane]);
							r0[k][lane] = as_bits(std::sin(x));
							r1[k][lane] = as_bits(std::cos(x));
						}
					}
					this->store(ops[0], r0, ins.saturate);
					this->store(ops[1], r1, ins.saturate);
					break;
				case D3D10_SB_OPCODE_EQ: compare_float([](float x, float y) { return x == y; }); break;
				case D3D10_SB_OPCODE_NE: compare_float([](float x, float y) { return x != y; }); break;
				case D3D10_SB_OPCODE_LT: compare_float([](float x, float y) { return x < y; }); break;
				case D3D10_SB_OPCODE_GE: compare_float([](float x, float y) { return x >= y; }); break;
				case D3D10_SB_OPCODE_FTOU:
					this->fetch(ops[1], a, false);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = ftou(as_float(a[k][lane]));
						}
					}
					this->store(ops[0], r0, false);
					break;
				case D3D10_SB_OPCODE_FTOI:
					this->fetch(ops[1], a, false);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = static_cast<std::uint32_t>(ftoi(as_float(a[k][lane])));
						}
					}
					this->store(ops[0], r0, false);
					break;
				case D3D10_SB_OPCODE_UTOF:
					this->fetch(ops[1], a, true);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = as_bits(static_cast<float>(a[k][lane]));
						}
					}
					this->store(ops[0], r0, ins.saturate);
					break;
				case D3D10_SB_OPCODE_ITOF:
					this->fetch(ops[1], a, true);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = as_bits(static_cast<float>(static_cast<std::int32_t>(a[k][lane])));
						}
					}
					this->store(ops[0], r0, ins.saturate);
					break;
				case D3D10_SB_OPCODE_AND: binary_int([](std::uint32_t x, std::uint32_t y) { return x & y; }); break;
				case D3D10_SB_OPCODE_OR: binary_int([](std::uint32_t x, std::uint32_t y) { return x | y; }); break;
				case D3D10_SB_OPCODE_XOR: binary_int([](std::uint32_t x, std::uint32_t y) { return x ^ y; }); break;
				case D3D10_SB_OPCODE_NOT: unary_int([](std::uint32_t x) { return ~x; }); break;
				case D3D10_SB_OPCODE_INEG: unary_int([](std::uint32_t x) { return static_cast<std::uint32_t>(-static_cast<std::int32_t>(x)); }); break;
				case D3D10_SB_OPCODE_IADD: binary_int([](std::uint32_t x, std::uint32_t y) { return x + y; }); break;
				case D3D10_SB_OPCODE_ISHL: binary_int([](std::uint32_t x, std::uint32_t y) { return x << (y & 31u); }); break;
				case D3D10_SB_OPCODE_USHR: binary_int([](std::uint32_t x, std::uint32_t y) { return x >> (y & 31u); }); break;
				case D3D10_SB_OPCODE_ISHR: binary_int([](std::uint32_t x, std::uint32_t y) { return static_cast<std::uint32_t>(static_cast<std::int32_t>(x) >> (y & 31u)); }); break;
				case D3D10_SB_OPCODE_IMAX: binary_int([](std::uint32_t x, std::uint32_t y) { return static_cast<std::uint32_t>(std::max(static_cast<std::int32_t>(x), static_cast<std::int32_t>(y))); }); break;
				case D3D10_SB_OPCODE_IMIN: binary_int([](std::uint32_t x, std::uint32_t y) { return static_cast<std::uint32_t>(std::min(static_cast<std::int32_t>(x), static_cast<std::int32_t>(y))); }); break;
				case D3D10_SB_OPCODE_UMAX: binary_int([](std::uint32_t x, std::uint32_t y) { return std::max(x, y); }); break;
				case D3D10_SB_OPCODE_UMIN: binary_int([](std::uint32_t x, std::uint32_t y) { return std::min(x, y); }); break;
				case D3D10_SB_OPCODE_IEQ: binary_int([](std::uint32_t x, std::uint32_t y) { return x == y ? 0xFFFFFFFFu : 0u; }); break;
				case D3D10_SB_OPCODE_INE: binary_int([](std::uint32_t x, std::uint32_t y) { return x != y ? 0xFFFFFFFFu : 0u; }); break;
				case D3D10_SB_OPCODE_IGE: binary_int([](std::uint32_t x, std::uint32_t y) { return static_cast<std::int32_t>(x) >= static_cast<std::int32_t>(y) ? 0xFFFFFFFFu : 0u; }); break;
				case D3D10_SB_OPCODE_ILT: binary_int([](std::uint32_t x, std::uint32_t y) { return static_cast<std::int32_t>(x) < static_cast<std::int32_t>(y) ? 0xFFFFFFFFu : 0u; }); break;
				case D3D10_SB_OPCODE_UGE: binary_int([](std::uint32_t x, std::uint32_t y) { return x >= y ? 0xFFFFFFFFu : 0u; }); break;
				case D3D10_SB_OPCODE_ULT: binary_int([](std::uint32_t x, std::uint32_t y) { return x < y ? 0xFFFFFFFFu : 0u; }); break;
				case D3D10_SB_OPCODE_IMAD:
				case D3D10_SB_OPCODE_UMAD:
					this->fetch(ops[1], a, true);
					this->fetch(ops[2], b, true);
					this->fetch(ops[3], c, true);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = a[k][lane] * b[k][lane] + c[k][lane];
						}
					}
					this->store(ops[0], r0, false);
					break;
				case D3D10_SB_OPCODE_IMUL:
				case D3D10_SB_OPCODE_UMUL:
					this->fetch(ops[2], a, true);
					this->fetch(ops[3], b, true);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							std::uint64_t product;
							if (op == D3D10_SB_OPCODE_IMUL)
							{
								product = static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(a[k][lane])) *
									static_cast<std::int64_t>(static_cast<std::int32_t>(b[k][lane])));
							}
							else
							{
								product = static_cast<std::uint64_t>(a[k][lane]) * b[k][lane];
							}
							r0[k][lane] = static_cast<std::uint32_t>(product >> 32);
							r1[k][lane] = static_cast<std::uint32_t>(product);
						}
					}
					this->store(ops[0], r0, false);
					this->store(ops[1], r1, false);
					break;
				case D3D10_SB_OPCODE_UDIV:
					this->fetch(ops[2], a, true);
					this->fetch(ops[3], b, true);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							const auto x = a[k][lane], y = b[k][lane];
							r0[k][lane] = y ? x / y : 0xFFFFFFFFu;
							r1[k][lane] = y ? x % y : 0xFFFFFFFFu;
						}
					}
					this->store(ops[0], r0, false);
					this->store(ops[1], r1, false);
					break;
				case D3D11_SB_OPCODE_UBFE:
				case D3D11_SB_OPCODE_IBFE:
					this->fetch(ops[1], a, true);
					this->fetch(ops[2], b, true);
					this->fetch(ops[3], c, true);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							const auto width = a[k][lane] & 31u;
							const auto offset = b[k][lane] & 31u;
							const auto value = c[k][lane];
							if (width == 0)
							{
								r0[k][lane] = 0;
							}
							else if (width + offset < 32)
							{
								const auto shifted = value << (32 - (width + offset));
								r0[k][lane] = op == D3D11_SB_OPCODE_UBFE ? shifted >> (32 - width)
									: static_cast<std::uint32_t>(static_cast<std::int32_t>(shifted) >> (32 - width));
							}
							else
							{
								r0[k][lane] = op == D3D11_SB_OPCODE_UBFE ? value >> offset
									: static_cast<std::uint32_t>(static_cast<std::int32_t>(value) >> offset);
							}
						}
					}
					this->store(ops[0], r0, false);
					break;
				case D3D11_SB_OPCODE_BFI:
					this->fetch(ops[1], a, true);
					this->fetch(ops[2], b, true);
					this->fetch(ops[3], c, true);
					this->fetch(ops[4], d, true);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							const auto width = a[k][lane] & 31u;
							const auto offset = b[k][lane] & 31u;
							const auto mask = (((1u << width) - 1u) << offset) & 0xFFFFFFFFu;
							r0[k][lane] = ((c[k][lane] << offset) & mask) | (d[k][lane] & ~mask);
						}
					}
					this->store(ops[0], r0, false);
					break;
				case D3D11_SB_OPCODE_BFREV: unary_int([](std::uint32_t x)
				{
					std::uint32_t r = 0;
					for (auto bit = 0; bit < 32; bit++)
					{
						r |= ((x >> bit) & 1u) << (31 - bit);
					}
					return r;
				}); break;
				case D3D11_SB_OPCODE_COUNTBITS: unary_int([](std::uint32_t x) { return static_cast<std::uint32_t>(std::popcount(x)); }); break;
				case D3D11_SB_OPCODE_FIRSTBIT_LO: unary_int([](std::uint32_t x) { return x ? static_cast<std::uint32_t>(std::countr_zero(x)) : 0xFFFFFFFFu; }); break;
				case D3D11_SB_OPCODE_FIRSTBIT_HI: unary_int([](std::uint32_t x) { return x ? static_cast<std::uint32_t>(std::countl_zero(x)) : 0xFFFFFFFFu; }); break;
				case D3D11_SB_OPCODE_FIRSTBIT_SHI: unary_int([](std::uint32_t x)
				{
					const auto v = (x & 0x80000000u) ? ~x : x;
					return v ? static_cast<std::uint32_t>(std::countl_zero(v)) : 0xFFFFFFFFu;
				}); break;
				case D3D11_SB_OPCODE_F16TOF32: unary_int([](std::uint32_t x) { return as_bits(half_to_float(x & 0xFFFFu)); }); break;
				case D3D11_SB_OPCODE_F32TOF16:
					this->fetch(ops[1], a, false);
					for (auto k = 0; k < 4; k++)
					{
						for (auto lane = 0u; lane < lane_count; lane++)
						{
							r0[k][lane] = float_to_half(as_float(a[k][lane]));
						}
					}
					this->store(ops[0], r0, false);
					break;
				case D3D10_SB_OPCODE_DERIV_RTX:
				case D3D11_SB_OPCODE_DERIV_RTX_COARSE:
				case D3D11_SB_OPCODE_DERIV_RTX_FINE:
				case D3D10_SB_OPCODE_DERIV_RTY:
				case D3D11_SB_OPCODE_DERIV_RTY_COARSE:
				case D3D11_SB_OPCODE_DERIV_RTY_FINE:
				{
					this->fetch(ops[1], a, false);
					const auto is_x = op == D3D10_SB_OPCODE_DERIV_RTX || op == D3D11_SB_OPCODE_DERIV_RTX_COARSE || op == D3D11_SB_OPCODE_DERIV_RTX_FINE;
					const auto fine = op == D3D11_SB_OPCODE_DERIV_RTX_FINE || op == D3D11_SB_OPCODE_DERIV_RTY_FINE;
					for (auto k = 0; k < 4; k++)
					{
						for (auto quad = 0u; quad < lane_count / 4; quad++)
						{
							const auto l0 = quad * 4;
							const auto v = [&](std::uint32_t l) { return as_float(a[k][l0 + l]); };
							for (auto l = 0u; l < 4; l++)
							{
								float result;
								if (is_x)
								{
									const auto row = fine ? (l & 2u) : 0u;
									result = v(row + 1) - v(row);
								}
								else
								{
									const auto col = fine ? (l & 1u) : 0u;
									result = v(col + 2) - v(col);
								}
								r0[k][l0 + l] = as_bits(result);
							}
						}
					}
					this->store(ops[0], r0, ins.saturate);
					break;
				}
				case D3D10_SB_OPCODE_SAMPLE:
				case D3D10_SB_OPCODE_SAMPLE_L:
				case D3D10_SB_OPCODE_SAMPLE_B:
				case D3D10_SB_OPCODE_SAMPLE_D:
				case D3D10_SB_OPCODE_LD:
					this->sample(ins, i);
					break;
				case D3D10_SB_OPCODE_RESINFO:
				{
					const auto& res = ops[2];
					const auto* source = res.index[0] < this->textures_.size() ? this->textures_[res.index[0]] : nullptr;
					if (!source)
					{
						if (this->unbound_zero_)
						{
							this->store(ops[0], reg4{}, false);
							break;
						}
						this->fail(i, "resinfo on an unbound texture");
					}
					this->fetch(ops[1], a, true);
					for (auto lane = 0u; lane < lane_count; lane++)
					{
						std::uint32_t width, height, levels;
						source->dimensions(a[0][lane], width, height, levels);
						const float values[4] = { static_cast<float>(width), static_cast<float>(height), 1.0f, static_cast<float>(levels) };
						for (auto k = 0; k < 4; k++)
						{
							r0[k][lane] = as_bits(values[k]);
						}
					}
					reg4 swizzled{};
					for (auto k = 0; k < 4; k++)
					{
						swizzled[k] = r0[res.swizzle[k]];
					}
					this->store(ops[0], swizzled, false);
					break;
				}
				case D3D11_SB_OPCODE_LD_STRUCTURED:
				{
					// ld_structured dst, element, byte offset, t#: four dwords from element * stride + offset, the
					// resource swizzle picking the returned components
					const auto& res = ops.at(3);
					const auto buffer = res.kind == operand_kind::resource ? this->buffers_.find(res.index[0]) : this->buffers_.end();
					if (buffer == this->buffers_.end())
					{
						if (!this->unbound_zero_)
						{
							this->fail(i, "ld_structured from an unbound buffer");
						}
						this->store(ops[0], reg4{}, false);
						break;
					}
					reg4 element{}, offset{};
					this->fetch(ops[1], element, true);
					this->fetch(ops[2], offset, true);
					const auto& data = buffer->second.data;
					reg4 loaded{};
					for (auto lane = 0u; lane < lane_count; lane++)
					{
						const auto address = static_cast<std::uint64_t>(element[0][lane]) * buffer->second.stride + offset[0][lane];
						for (auto k = 0u; k < 4; k++)
						{
							const auto at = address + 4 * k;
							if (at + 4 <= data.size())
							{
								std::memcpy(&loaded[k][lane], data.data() + at, 4);
							}
						}
					}
					reg4 result{};
					for (auto component = 0; component < 4; component++)
					{
						result[component] = loaded[res.swizzle[component]];
					}
					this->store(ops[0], result, false);
					break;
				}
				// resource reads the machine only runs against nothing bound (the operand holding the resource
				// differs: D3D11 functional spec, section 22)
				case D3D10_SB_OPCODE_SAMPLE_C:
				case D3D10_SB_OPCODE_SAMPLE_C_LZ:
				case D3D10_1_SB_OPCODE_GATHER4:
				case D3D11_SB_OPCODE_GATHER4_C:
				case D3D10_1_SB_OPCODE_LOD:
				case D3D11_SB_OPCODE_LD_RAW:
				case D3D11_SB_OPCODE_GATHER4_PO:
				case D3D11_SB_OPCODE_GATHER4_PO_C:
				case D3D11_SB_OPCODE_BUFINFO:
				case D3D10_1_SB_OPCODE_SAMPLE_INFO:
				{
					std::size_t res_operand = 2;
					if (op == D3D11_SB_OPCODE_GATHER4_PO || op == D3D11_SB_OPCODE_GATHER4_PO_C)
					{
						res_operand = 3;
					}
					else if (op == D3D11_SB_OPCODE_BUFINFO || op == D3D10_1_SB_OPCODE_SAMPLE_INFO)
					{
						res_operand = 1;
					}
					const auto& res = ops.at(res_operand);
					const auto bound = res.kind == operand_kind::resource && res.index[0] < this->textures_.size()
						&& this->textures_[res.index[0]];
					if (bound || !this->unbound_zero_)
					{
						this->fail(i, "unsupported resource instruction");
					}
					this->store(ops[0], reg4{}, false);
					break;
				}
				default:
					this->fail(i, "unsupported instruction");
				}

				(void)integer;
			}
		}

		taint_result analyse(const program& prog)
		{
			using tex_set = taint_result::texture_set;
			using in_set = taint_result::input_set;

			struct value
			{
				tex_set tex;
				in_set in;

				value& operator|=(const value& other)
				{
					this->tex |= other.tex;
					this->in |= other.in;
					return *this;
				}
			};

			using reg = std::array<value, 4>;

			std::uint32_t max_output = 0;
			for (const auto& e : prog.outputs())
			{
				max_output = std::max(max_output, e.reg + 1);
			}

			std::vector<reg> temps(prog.temp_count());
			std::vector<std::vector<reg>> indexable(prog.indexable_temps().size());
			for (auto i = 0u; i < indexable.size(); i++)
			{
				indexable[i].resize(prog.indexable_temps()[i].second);
			}
			std::vector<reg> outputs(max_output);

			taint_result result{};

			const auto source = [&](const operand& op) -> reg
			{
				reg out{};
				reg raw{};
				switch (op.kind)
				{
				case operand_kind::temp:
					raw = temps.at(op.index[0]);
					break;
				case operand_kind::output:
					raw = outputs.at(op.index[0]);
					break;
				case operand_kind::input:
					for (auto c = 0; c < 4; c++)
					{
						if (op.index[0] < 32)
						{
							raw[c].in.set(op.index[0] * 4 + c);
						}
					}
					break;
				case operand_kind::indexable_temp:
					for (const auto& element : indexable.at(op.index[0]))
					{
						for (auto c = 0; c < 4; c++)
						{
							raw[c] |= element[c];
						}
					}
					break;
				default:
					return out; // constants carry no taint
				}
				for (auto c = 0; c < 4; c++)
				{
					out[c] = raw[op.swizzle[c]];
				}
				return out;
			};

			const auto store = [&](const operand& op, const reg& v)
			{
				reg* target = nullptr;
				switch (op.kind)
				{
				case operand_kind::temp:
					target = &temps.at(op.index[0]);
					break;
				case operand_kind::output:
					target = &outputs.at(op.index[0]);
					break;
				case operand_kind::indexable_temp:
					for (auto& element : indexable.at(op.index[0]))
					{
						for (auto c = 0; c < 4; c++)
						{
							if (op.mask & (1u << c))
							{
								element[c] |= v[c];
							}
						}
					}
					return;
				default:
					return;
				}
				for (auto c = 0; c < 4; c++)
				{
					if (op.mask & (1u << c))
					{
						(*target)[c] = v[c];
					}
				}
			};

			const auto all_of = [](const reg& r, const int n)
			{
				value v{};
				for (auto c = 0; c < n; c++)
				{
					v |= r[c];
				}
				return v;
			};

			for (const auto& ins : prog.instructions())
			{
				const auto op = ins.opcode;
				const auto& ops = ins.operands;

				switch (op)
				{
				case D3D10_SB_OPCODE_DISCARD:
				{
					const auto cond = source(ops[0]);
					result.discard_textures |= cond[0].tex;
					result.discard_inputs |= cond[0].in;
					continue;
				}
				case D3D10_SB_OPCODE_IF:
				case D3D10_SB_OPCODE_ELSE:
				case D3D10_SB_OPCODE_ENDIF:
				case D3D10_SB_OPCODE_LOOP:
				case D3D10_SB_OPCODE_ENDLOOP:
				case D3D10_SB_OPCODE_BREAK:
				case D3D10_SB_OPCODE_BREAKC:
				case D3D10_SB_OPCODE_CONTINUE:
				case D3D10_SB_OPCODE_CONTINUEC:
				case D3D10_SB_OPCODE_RET:
				case D3D10_SB_OPCODE_RETC:
				case D3D10_SB_OPCODE_NOP:
				case D3D10_SB_OPCODE_LABEL:
				case D3D10_SB_OPCODE_SWITCH:
				case D3D10_SB_OPCODE_CASE:
				case D3D10_SB_OPCODE_DEFAULT:
				case D3D10_SB_OPCODE_ENDSWITCH:
					continue;
				case D3D10_SB_OPCODE_SAMPLE:
				case D3D10_SB_OPCODE_SAMPLE_L:
				case D3D10_SB_OPCODE_SAMPLE_B:
				case D3D10_SB_OPCODE_SAMPLE_D:
				case D3D10_SB_OPCODE_SAMPLE_C:
				case D3D10_SB_OPCODE_SAMPLE_C_LZ:
				case D3D10_SB_OPCODE_LD:
				{
					const auto coord = all_of(source(ops[1]), 4);
					const auto& res = ops[2];
					reg out{};
					for (auto c = 0; c < 4; c++)
					{
						out[c] = coord;
						if (res.kind == operand_kind::resource && res.index[0] < 128)
						{
							out[c].tex.set(res.index[0] * 4 + res.swizzle[c]);
						}
					}
					store(ops[0], out);
					continue;
				}
				case D3D10_SB_OPCODE_DP2:
				case D3D10_SB_OPCODE_DP3:
				case D3D10_SB_OPCODE_DP4:
				{
					const auto n = op == D3D10_SB_OPCODE_DP2 ? 2 : op == D3D10_SB_OPCODE_DP3 ? 3 : 4;
					auto v = all_of(source(ops[1]), n);
					v |= all_of(source(ops[2]), n);
					reg out{};
					out.fill(v);
					store(ops[0], out);
					continue;
				}
				case D3D10_SB_OPCODE_SINCOS:
				{
					const auto s = source(ops[2]);
					store(ops[0], s);
					store(ops[1], s);
					continue;
				}
				case D3D10_SB_OPCODE_IMUL:
				case D3D10_SB_OPCODE_UMUL:
				case D3D10_SB_OPCODE_UDIV:
				{
					auto s = source(ops[2]);
					const auto t = source(ops[3]);
					for (auto c = 0; c < 4; c++)
					{
						s[c] |= t[c];
					}
					store(ops[0], s);
					store(ops[1], s);
					continue;
				}
				case D3D10_SB_OPCODE_RESINFO:
					store(ops[0], reg{});
					continue;
				}

				if (ops.empty())
				{
					continue;
				}

				// component-wise: destination component c depends on component c of every source
				reg out{};
				for (auto k = 1u; k < ops.size(); k++)
				{
					const auto s = source(ops[k]);
					for (auto c = 0; c < 4; c++)
					{
						out[c] |= s[c];
					}
				}
				store(ops[0], out);
			}

			result.output_textures.resize(max_output);
			result.output_inputs.resize(max_output);
			for (auto r = 0u; r < max_output; r++)
			{
				for (auto c = 0; c < 4; c++)
				{
					result.output_textures[r][c] = outputs[r][c].tex;
					result.output_inputs[r][c] = outputs[r][c].in;
				}
			}
			return result;
		}

		std::array<std::vector<colour_source>, 2> vertex_colour(const program& vs)
		{
			std::array<std::vector<colour_source>, 2> out;
			const auto* input = vs.find_input("COLOR", 0);
			const auto& instructions = vs.instructions();

			// top-level instructions only: a write inside a block may not run
			std::vector<bool> top(instructions.size());
			auto depth = 0;
			for (std::size_t i = 0; i < instructions.size(); i++)
			{
				const auto op = instructions[i].opcode;
				if (op == D3D10_SB_OPCODE_ENDIF || op == D3D10_SB_OPCODE_ENDLOOP || op == D3D10_SB_OPCODE_ENDSWITCH)
				{
					depth--;
				}
				top[i] = depth == 0;
				if (op == D3D10_SB_OPCODE_IF || op == D3D10_SB_OPCODE_LOOP || op == D3D10_SB_OPCODE_SWITCH)
				{
					depth++;
				}
			}

			// the source component an operand gives destination component c
			const auto component = [](const operand& op, const std::uint32_t c) -> std::uint32_t
			{
				return op.component_count == 1 ? op.swizzle[0] : op.swizzle[c];
			};
			// the last write of register (kind, index) component c before instruction `at`, if at the top level
			const auto last_write = [&](const operand_kind kind, const std::uint32_t index, const std::uint32_t c,
				const std::size_t at) -> std::optional<std::size_t>
			{
				for (auto i = at; i-- > 0;)
				{
					const auto& ins = instructions[i];
					if (!ins.operands.empty() && ins.operands[0].kind == kind && ins.operands[0].index[0] == index
						&& (ins.operands[0].mask & (1u << c)))
					{
						return top[i] ? std::optional<std::size_t>(i) : std::nullopt;
					}
				}
				return std::nullopt;
			};
			// the vertex colour channel an operand reads as destination component c, if it reads the COLOR input
			const auto input_channel = [&](const operand& op, const std::uint32_t c) -> std::optional<std::uint8_t>
			{
				if (!input || op.kind != operand_kind::input || op.index[0] != input->reg || op.relative)
				{
					return std::nullopt;
				}
				return static_cast<std::uint8_t>(component(op, c));
			};

			for (const auto& e : vs.outputs())
			{
				if (_stricmp(e.semantic.data(), "COLOR") || e.semantic_index > 1)
				{
					continue;
				}
				auto& sources = out[e.semantic_index];
				for (auto c = 0u; c < 4; c++)
				{
					if (!(e.mask & (1u << c)))
					{
						continue;
					}
					colour_source s{};
					const auto at = last_write(operand_kind::output, e.reg, c, instructions.size());
					if (at)
					{
						const auto& ins = instructions[*at];
						const auto& src = ins.operands.size() > 1 ? ins.operands[1] : ins.operands[0];
						if (ins.opcode == D3D10_SB_OPCODE_MOV && !ins.saturate && src.modifier == 0)
						{
							if (const auto channel = input_channel(src, c))
							{
								s = { colour_source::kind::channel, *channel };
							}
							else if (src.kind == operand_kind::immediate && std::bit_cast<float>(src.immediate[component(src, c)]) == 1.0f)
							{
								s = { colour_source::kind::one, 0 };
							}
						}
						else if (ins.opcode == D3D10_SB_OPCODE_EXP && !ins.saturate && src.kind == operand_kind::temp && src.modifier == 0)
						{
							// exp o, t: t = l * 2.2 (either order), l = log |v.channel|
							const auto tc = component(src, c);
							const auto mul_at = last_write(operand_kind::temp, src.index[0], tc, *at);
							const auto& mul = mul_at ? instructions[*mul_at] : ins;
							if (mul_at && mul.opcode == D3D10_SB_OPCODE_MUL && !mul.saturate && mul.operands.size() == 3)
							{
								for (auto k = 1u; k <= 2 && s.what == colour_source::kind::unknown; k++)
								{
									const auto& lit = mul.operands[3 - k];
									const auto& log = mul.operands[k];
									if (lit.kind != operand_kind::immediate || std::fabs(std::bit_cast<float>(lit.immediate[component(lit, tc)]) - 2.2f) > 1e-6f
										|| log.kind != operand_kind::temp || log.modifier != 0)
									{
										continue;
									}
									const auto lc = component(log, tc);
									const auto log_at = last_write(operand_kind::temp, log.index[0], lc, *mul_at);
									if (!log_at || instructions[*log_at].opcode != D3D10_SB_OPCODE_LOG || instructions[*log_at].saturate)
									{
										continue;
									}
									const auto& v = instructions[*log_at].operands[1];
									if (v.modifier == 2)
									{
										if (const auto channel = input_channel(v, lc))
										{
											s = { colour_source::kind::linearised, *channel };
										}
									}
								}
							}
						}
					}
					sources.push_back(s);
				}
			}
			return out;
		}
	}
}
