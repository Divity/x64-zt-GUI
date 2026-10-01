#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "gpu_eval.hpp"

#include "assets/gfximage.hpp"

#include <DirectXTex.h>
#include <d3dcompiler.h>
#include <utils/string.hpp>

#include <bit>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace zonetool::t7
{
	namespace converter::iw7::gpu_eval
	{
		namespace
		{
			using Microsoft::WRL::ComPtr;

			constexpr std::uint32_t opcode_discard = 13; // D3D10_SB_OPCODE_DISCARD
			constexpr std::uint32_t opcode_customdata = 53; // D3D10_SB_OPCODE_CUSTOMDATA (its length is the next token)
			constexpr std::uint32_t opcode_nop = 58; // D3D10_SB_OPCODE_NOP

			// uploaded images a device keeps between materials before its cache is emptied
			constexpr std::size_t image_budget = 384ull << 20;

			// DirectXTex's BC7 encoder shaders (deps/extra/DirectXTex/Shaders/Compiled, built from DirectXTex's own
			// shader sources) and the layouts its driver (BCDirectCompute.cpp) gives them
			namespace bc7
			{
#include "BC7Encode_TryMode456CS.inc"
#include "BC7Encode_TryMode137CS.inc"
#include "BC7Encode_EncodeBlockCS.inc"

				struct constants
				{
					std::uint32_t tex_width;
					std::uint32_t num_block_x;
					std::uint32_t format;
					std::uint32_t mode_id;
					std::uint32_t start_block_id;
					std::uint32_t num_total_blocks;
					float alpha_weight;
					std::uint32_t reserved;
				};
				static_assert(sizeof(constants) == 32);

				constexpr std::uint32_t block_bytes = 16; // BufferBC6HBC7: uint color[4]
				constexpr std::uint32_t batch = 16384; // blocks per dispatch round (mode 1/3/7 runs a group per block)
			}

			// ---- the DXBC container checksum --------------------------------------------------------

			void md5_transform(std::uint32_t state[4], const std::uint8_t block[64])
			{
				static constexpr std::uint32_t k[64] = {
					0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
					0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
					0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
					0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
					0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
					0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
					0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
					0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
				};
				static constexpr std::uint8_t r[64] = {
					7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
					5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
					4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
					6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
				};

				std::uint32_t w[16];
				std::memcpy(w, block, 64);
				auto a = state[0], b = state[1], c = state[2], d = state[3];
				for (auto i = 0u; i < 64; i++)
				{
					std::uint32_t f, g;
					if (i < 16)
					{
						f = (b & c) | (~b & d);
						g = i;
					}
					else if (i < 32)
					{
						f = (d & b) | (~d & c);
						g = (5 * i + 1) % 16;
					}
					else if (i < 48)
					{
						f = b ^ c ^ d;
						g = (3 * i + 5) % 16;
					}
					else
					{
						f = c ^ (b | ~d);
						g = (7 * i) % 16;
					}
					const auto t = d;
					d = c;
					c = b;
					b = b + std::rotl(a + f + k[i] + w[g], r[i]);
					a = t;
				}
				state[0] += a;
				state[1] += b;
				state[2] += c;
				state[3] += d;
			}

			// MD5 over the container from byte 20, with its own last block: the bit count first, the rest of the
			// data after it and (size * 2) | 1 as the last word (a whole extra block when the rest leaves no
			// room). Checked against the checksum the original program carries before anything is patched.
			std::array<std::uint32_t, 4> dxbc_checksum(const std::uint8_t* data, const std::size_t size)
			{
				std::uint32_t state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };
				const auto* p = data + 20;
				const auto n = size - 20;
				const auto whole = n & ~static_cast<std::size_t>(63);
				for (std::size_t at = 0; at < whole; at += 64)
				{
					md5_transform(state, p + at);
				}

				const auto left = n - whole;
				const auto bits = static_cast<std::uint32_t>(n * 8);
				const auto last = static_cast<std::uint32_t>(n * 2) | 1u;
				std::uint8_t block[64]{};
				if (left >= 56)
				{
					std::memcpy(block, p + whole, left);
					block[left] = 0x80;
					md5_transform(state, block);
					std::memset(block, 0, sizeof(block));
					std::memcpy(block, &bits, 4);
					std::memcpy(block + 60, &last, 4);
					md5_transform(state, block);
				}
				else
				{
					std::memcpy(block, &bits, 4);
					std::memcpy(block + 4, p + whole, left);
					block[4 + left] = 0x80;
					std::memcpy(block + 60, &last, 4);
					md5_transform(state, block);
				}
				return { state[0], state[1], state[2], state[3] };
			}

			std::uint32_t read_u32(const std::vector<std::uint8_t>& data, const std::size_t at)
			{
				if (at + 4 > data.size())
				{
					throw std::runtime_error("read past the end of the shader container");
				}
				std::uint32_t value;
				std::memcpy(&value, data.data() + at, 4);
				return value;
			}

			// The program with every discard replaced by no-ops. The interpreter keeps what a discarded texel
			// computes (the bake stores it; coverage comes from its own texture), where the GPU would drop it.
			std::vector<std::uint8_t> without_discard(const std::uint8_t* bytecode, const std::size_t size)
			{
				std::vector<std::uint8_t> code(bytecode, bytecode + size);
				if (size < 32 || std::memcmp(bytecode, "DXBC", 4) != 0 || read_u32(code, 24) != size)
				{
					throw std::runtime_error("not a DXBC container");
				}

				auto discards = 0u;
				const auto chunk_count = read_u32(code, 28);
				for (auto c = 0u; c < chunk_count; c++)
				{
					const auto offset = read_u32(code, 32 + c * 4);
					const auto fourcc = read_u32(code, offset);
					if (fourcc != 'XEHS' && fourcc != 'RDHS')
					{
						continue;
					}
					const auto base = static_cast<std::size_t>(offset) + 8;
					const auto length = read_u32(code, base + 4); // tokens, the version and length tokens included
					if (static_cast<std::size_t>(length) * 4 > read_u32(code, offset + 4))
					{
						throw std::runtime_error("program length runs past its chunk");
					}
					for (auto i = 2u; i < length;)
					{
						const auto token = read_u32(code, base + i * 4);
						const auto opcode = token & 0x7FFu;
						const auto count = opcode == opcode_customdata ? read_u32(code, base + (i + 1) * 4) : (token >> 24) & 0x7Fu;
						if (!count || i + count > length)
						{
							throw std::runtime_error(utils::string::va("malformed instruction at token %u", i));
						}
						if (opcode == opcode_discard)
						{
							constexpr auto nop = opcode_nop | (1u << 24);
							for (auto k = 0u; k < count; k++)
							{
								std::memcpy(code.data() + base + (i + k) * 4, &nop, 4);
							}
							discards++;
						}
						i += count;
					}
				}
				if (!discards)
				{
					return code;
				}

				std::array<std::uint32_t, 4> stored;
				std::memcpy(stored.data(), bytecode + 4, 16);
				if (dxbc_checksum(bytecode, size) != stored)
				{
					throw std::runtime_error("the container checksum could not be reproduced, so the program cannot be patched");
				}
				const auto sum = dxbc_checksum(code.data(), code.size());
				std::memcpy(code.data() + 4, sum.data(), 16);
				return code;
			}

			// ---- the vertex shader ------------------------------------------------------------------

			std::string bits_literal(const input_element& e, const float value)
			{
				const auto bits = std::bit_cast<std::uint32_t>(value);
				switch (e.component_type)
				{
				case 1: // uint: the interpreter writes the float's bits whatever the element's type
					return utils::string::va("0x%08Xu", bits);
				case 2:
					return utils::string::va("asint(0x%08Xu)", bits);
				default:
					return utils::string::va("asfloat(0x%08Xu)", bits);
				}
			}

			// One output per input element, in the same order, semantics and widths: the compiler packs them into
			// the registers and components BO3's own vertex shader did (checked against the pixel shader's
			// signature once compiled).
			std::string vertex_source(const std::vector<input_element>& inputs)
			{
				std::string fields;
				std::string body;
				for (auto i = 0u; i < inputs.size(); i++)
				{
					const auto& e = inputs[i];
					if (e.kind == input_kind::system)
					{
						continue;
					}
					if (!e.mask)
					{
						throw std::runtime_error(utils::string::va("input %s%u has no components", e.semantic.data(), e.semantic_index));
					}
					const auto first = static_cast<std::uint32_t>(std::countr_zero(e.mask));
					const auto width = static_cast<std::uint32_t>(std::popcount(e.mask));
					if (e.mask != ((1u << width) - 1u) << first)
					{
						throw std::runtime_error(utils::string::va("input %s%u has a split mask 0x%X", e.semantic.data(), e.semantic_index, e.mask));
					}

					const char* base = e.component_type == 1 ? "uint" : (e.component_type == 2 ? "int" : "float");
					if (e.kind == input_kind::position)
					{
						fields += utils::string::va("\tfloat4 e%u : SV_Position;\n", i);
						body += utils::string::va("\to.e%u = float4(s.x * 2.0 - 1.0, 1.0 - s.y * 2.0, 0.5, 1.0);\n", i);
						continue;
					}
					if (e.system_value != 0)
					{
						throw std::runtime_error(utils::string::va("input %s%u is system value %u", e.semantic.data(), e.semantic_index, e.system_value));
					}
					// D3D10_SB_INTERPOLATION_MODE: 1 constant, 2 linear, 3 linear centroid, 4 linear noperspective, 5 linear
					// noperspective centroid, 6 linear sample, 7 linear noperspective sample
					static constexpr const char* modifiers[] = { "", "nointerpolation ", "", "centroid ", "noperspective ",
						"noperspective centroid ", "sample ", "noperspective sample " };
					const auto* modifier = e.interpolation < std::size(modifiers) ? modifiers[e.interpolation] : "";
					fields += utils::string::va("\t%s%s%s e%u : %s%u;\n", modifier, base, width > 1 ? std::to_string(width).data() : "", i,
						e.semantic.data(), e.semantic_index);

					std::string value;
					for (auto j = 0u; j < width; j++)
					{
						if (j)
						{
							value += ", ";
						}
						if (e.kind == input_kind::uv && j < 2)
						{
							value += j == 0 ? "uv.x" : "uv.y";
						}
						else if (e.kind == input_kind::uv)
						{
							value += bits_literal(e, 0.0f);
						}
						else
						{
							value += bits_literal(e, e.value[first + j]);
						}
					}
					body += utils::string::va("\to.e%u = %s%s(%s);\n", i, base, width > 1 ? std::to_string(width).data() : "", value.data());
				}

				return "cbuffer bake : register(b0)\n{\n\tfloat4 origin_step; // uv at the region's corner, uv per texel\n"
					"\tfloat4 extent; // region texels\n};\n\nstruct bake_out\n{\n" + fields + "};\n\n"
					"bake_out main(uint id : SV_VertexID)\n{\n"
					"\t// one triangle over the viewport: s = 0 at its corner, 2 at twice its size\n"
					"\tconst float2 s = float2(id == 1 ? 2.0 : 0.0, id == 2 ? 2.0 : 0.0);\n"
					"\tconst float2 uv = origin_step.xy + s * extent.xy * origin_step.zw;\n"
					"\tbake_out o;\n" + body + "\treturn o;\n}\n";
			}

			// the compiled vertex shader's outputs must land where the pixel shader reads its inputs
			void check_linkage(ID3DBlob* blob, const std::vector<input_element>& inputs)
			{
				ComPtr<ID3D11ShaderReflection> reflection;
				if (FAILED(D3DReflect(blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&reflection))))
				{
					throw std::runtime_error("the generated vertex shader cannot be reflected");
				}
				D3D11_SHADER_DESC desc{};
				reflection->GetDesc(&desc);
				for (const auto& e : inputs)
				{
					if (e.kind == input_kind::system)
					{
						continue;
					}
					auto matched = false;
					for (auto i = 0u; i < desc.OutputParameters; i++)
					{
						D3D11_SIGNATURE_PARAMETER_DESC out{};
						reflection->GetOutputParameterDesc(i, &out);
						if (_stricmp(out.SemanticName, e.semantic.data()) || out.SemanticIndex != e.semantic_index)
						{
							continue;
						}
						if (out.Register != e.reg || (out.Mask & e.mask) != e.mask)
						{
							throw std::runtime_error(utils::string::va("%s%u lands in o%u mask 0x%X, the pixel shader reads v%u mask 0x%X",
								e.semantic.data(), e.semantic_index, out.Register, out.Mask, e.reg, e.mask));
						}
						matched = true;
					}
					if (!matched)
					{
						throw std::runtime_error(utils::string::va("the generated vertex shader has no %s%u", e.semantic.data(), e.semantic_index));
					}
				}
			}

			D3D11_TEXTURE_ADDRESS_MODE address(const material_texture::address_mode mode)
			{
				switch (mode)
				{
				case material_texture::address_mode::clamp: return D3D11_TEXTURE_ADDRESS_CLAMP;
				case material_texture::address_mode::mirror: return D3D11_TEXTURE_ADDRESS_MIRROR;
				default: return D3D11_TEXTURE_ADDRESS_WRAP;
				}
			}

			bool exact_unorm8(const float v)
			{
				const auto scaled = v * 255.0f;
				return v >= 0.0f && v <= 1.0f && scaled == std::round(scaled);
			}

			// ---- packing BO3 gbuffer targets into IW7 texels ----------------------------------------

			// world_material_bake.cpp's decode_gbuffer and pack_texel (with pack_colour, octahedral, iw7_gloss and
			// to_byte), the same arithmetic in the same order; the bake compares the two on its first packed tile
			constexpr char pack_source[] = R"hlsl(
Texture2D<float4> rt0 : register(t0);
Texture2D<float4> rt1 : register(t1);
Texture2D<float4> rt2 : register(t2);
Texture2D<float4> rt2_other : register(t3); // the parity pass: texel x drawn at x + 1
RWTexture2D<uint4> cs_out : register(u0);
RWTexture2D<uint4> ng_out : register(u1);
RWTexture2D<uint> covered_out : register(u2);
RWStructuredBuffer<uint> flags_out : register(u3);

cbuffer pack_constants : register(b0)
{
	uint2 tile_origin; // the tile's first texel in the level
	uint2 tile_size;
	uint options;
	float underlying_gloss;
	uint2 unused;
};

static const uint opt_decal = 1u;
static const uint opt_writes_normal = 2u;
static const uint opt_writes_gloss = 4u;
static const uint opt_writes_specular = 8u;
static const uint opt_writes_occlusion = 16u;
static const uint opt_occlusion = 32u;
static const uint opt_coloured_specular = 64u;
static const uint opt_raw = 128u;

// std::lround of the clamped value times 255: halves round up
uint to_byte(float v)
{
	const float x = saturate(v) * 255.0f;
	const uint r = (uint)x;
	return (x - (float)r >= 0.5f) ? r + 1u : r;
}

float unpremultiply(float value, float alpha)
{
	return alpha > (1.0f / 1024.0f) ? value / alpha : value;
}

float3 decode_normal(float4 enc)
{
	const uint basis = (uint)max(0.0f, enc.w * 3.0f + 0.5f);
	const float px = (enc.x * 2.0f - 1.0f) * 0.85f;
	const float py = (enc.y * 2.0f - 1.0f) * 0.85f;
	const float d = px * px + py * py;
	const float s = sqrt(max(0.0f, 2.0f - d));
	const float qx = s * px;
	const float qy = s * py;

	const float axis_x = (basis & 2u) != 0u ? -1.0f : 1.0f;
	const float axis_y = (basis & 1u) != 0u ? -1.0f : 1.0f;
	const float axis_z = ((basis & 1u) != 0u) != ((basis & 2u) != 0u) ? -1.0f : 1.0f;

	const float k = (1.0f - d) * 0.577350f;
	const float kx = qx * 0.408248f;
	const float ky = qy * 0.707107f;
	float3 n;
	n.x = axis_x * k + axis_x * kx - axis_x * ky;
	n.y = axis_y * k - 2.0f * axis_y * kx;
	n.z = axis_z * k + axis_z * kx + axis_z * ky;
	const float len = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
	if (len > 0.0f)
	{
		n.x /= len;
		n.y /= len;
		n.z /= len;
	}
	return n;
}

float2 octahedral(float3 n)
{
	float nz = n.z;
	if (nz < 0.0f)
	{
		nz = 0.0f;
	}
	const float sum = abs(n.x) + abs(n.y) + nz;
	const float px = sum > 0.0f ? n.x / sum : 0.0f;
	const float py = sum > 0.0f ? n.y / sum : 0.0f;
	return float2((px + py) * 0.5f + 0.5f, (px - py) * 0.5f + 0.5f);
}

float iw7_gloss(float bo3_gloss)
{
	const float t = 1.0f - 17.0f * bo3_gloss / 20.0f;
	return 1.0f - t * t;
}

static const int metal_first_byte = 26;
static const int metal_candidates = 256 - metal_first_byte;

float metal_error(float3 albedo, float3 f0, int a_byte, out uint4 candidate)
{
	const float a = (float)a_byte / 255.0f;
	const float m = clamp((a - 0.1f) / 0.9f, 0.0f, 1.0f);
	float error = 0.0f;
	candidate = uint4(0u, 0u, 0u, (uint)a_byte);
	[unroll] for (int c = 0; c < 3; c++)
	{
		const float denom = (1.0f - m) * (1.0f - m) + m * m;
		const float colour = clamp(((1.0f - m) * albedo[c] + m * (f0[c] - 0.1f)) / denom, 0.0f, 1.0f);
		const uint q = to_byte(sqrt(colour));
		const float cq = ((float)q / 255.0f) * ((float)q / 255.0f);
		const float ed = cq * (1.0f - m) - albedo[c];
		const float ef = 0.1f + m * cq - f0[c];
		error += ed * ed + ef * ef;
		candidate[c] = q;
	}
	return error;
}

float metal_bound(float3 albedo, float3 f0, int k)
{
	const float a = (float)(metal_first_byte + k) / 255.0f;
	const float m = clamp((a - 0.1f) / 0.9f, 0.0f, 1.0f);
	const float denom = (1.0f - m) * (1.0f - m) + m * m;
	float b = 0.0f;
	[unroll] for (int c = 0; c < 3; c++)
	{
		const float cross = m * albedo[c] - (1.0f - m) * (f0[c] - 0.1f);
		b += cross * cross / denom;
	}
	return b;
}

uint4 pack_colour(float3 albedo, float3 f0)
{
	const bool grey = abs(f0.x - f0.y) < 1e-3f && abs(f0.y - f0.z) < 1e-3f;
	if (grey && f0.y <= 0.1f)
	{
		return uint4(to_byte(sqrt(max(albedo.x, 0.0f))), to_byte(sqrt(max(albedo.y, 0.0f))), to_byte(sqrt(max(albedo.z, 0.0f))), to_byte(f0.y));
	}

	int first = 0;
	float first_bound = metal_bound(albedo, f0, 0);
	[loop] for (int k = 1; k < metal_candidates; k++)
	{
		const float b = metal_bound(albedo, f0, k);
		if (b < first_bound)
		{
			first_bound = b;
			first = k;
		}
	}

	uint4 best;
	float best_error = metal_error(albedo, f0, metal_first_byte + first, best);
	int best_k = first;
	[loop] for (int j = 0; j < metal_candidates; j++)
	{
		if (j == first || metal_bound(albedo, f0, j) > best_error * 1.0001f + 1e-12f)
		{
			continue;
		}
		uint4 candidate;
		const float error = metal_error(albedo, f0, metal_first_byte + j, candidate);
		if (error < best_error || (error == best_error && j < best_k))
		{
			best_error = error;
			best_k = j;
			best = candidate;
		}
	}

	// also compare with the best dielectric
	{
		float error = 0.0f;
		const float f = min((f0.x + f0.y + f0.z) / 3.0f, 0.1f);
		uint4 candidate = uint4(0u, 0u, 0u, to_byte(f));
		[unroll] for (int c = 0; c < 3; c++)
		{
			candidate[c] = to_byte(sqrt(max(albedo[c], 0.0f)));
			const float ef = f - f0[c];
			error += ef * ef;
		}
		if (error <= best_error)
		{
			best = candidate;
		}
	}
	return best;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= tile_size.x || id.y >= tile_size.y)
	{
		return;
	}
	const uint2 texel = tile_origin + id.xy;
	const bool decal = (options & opt_decal) != 0u;
	const bool raw = (options & opt_raw) != 0u;

	const float4 t0 = rt0[id.xy];
	const float4 t1 = rt1[id.xy];
	const float4 t2 = rt2[id.xy];
	const float a0 = decal ? t0.w : 1.0f;
	float3 albedo;
	float3 normal;
	float gloss;
	float3 f0;
	float occlusion;
	float shading_model;
	if (raw)
	{
		// decode_surface
		albedo = t0.xyz;
		normal = t1.xyz;
		const float len = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
		if (len > 0.0f)
		{
			normal.x /= len;
			normal.y /= len;
			normal.z /= len;
		}
		gloss = t1.w;
		f0 = t2.xyz;
		occlusion = t2.w;
		shading_model = 1.0f / 3.0f;
	}
	else
	{
		// decode_gbuffer
		const float a1 = decal ? t1.w : 1.0f;
		const float a2 = decal ? t2.w : 1.0f;
		shading_model = decal ? 1.0f / 3.0f : t2.w;

		albedo = float3(unpremultiply(t0.x, a0), unpremultiply(t0.y, a0), unpremultiply(t0.z, a0));
		const float4 enc = float4(unpremultiply(t1.x, a1), unpremultiply(t1.y, a1), unpremultiply(t1.z, a1), decal ? 0.0f : t1.w);
		normal = decode_normal(enc);

		const float z = enc.z;
		const bool model_flag = z >= 0.5f;
		gloss = clamp((z - (model_flag ? 0.5f : 0.001466f)) * 2.009823f, 0.0f, 1.0f);

		const float y_luma = unpremultiply(t2.x, a2);
		const float chroma_here = (unpremultiply(t2.y, a2) - 0.5f) * 2.0f;
		float chroma_other = 0.0f;
		if ((options & opt_coloured_specular) != 0u)
		{
			const float4 o = rt2_other[uint2(id.x + 1u, id.y)];
			chroma_other = (unpremultiply(o.y, decal ? o.w : 1.0f) - 0.5f) * 2.0f;
		}
		const bool same_parity = (texel.x & 1u) == (texel.y & 1u);
		const float rb = same_parity ? chroma_here : chroma_other;
		const float g_minus_rb = same_parity ? chroma_other : chroma_here;
		const float sum_rb = 2.0f * y_luma - g_minus_rb;
		f0.y = y_luma + g_minus_rb * 0.5f;
		f0.x = (sum_rb + rb) * 0.5f;
		f0.z = (sum_rb - rb) * 0.5f;
		occlusion = unpremultiply(t2.z, a2);
	}

	// pack_texel
	if ((options & opt_writes_normal) == 0u)
	{
		normal = float3(0.0f, 0.0f, 1.0f);
	}
	if ((options & opt_writes_gloss) == 0u)
	{
		gloss = underlying_gloss;
	}
	if ((options & opt_writes_specular) == 0u)
	{
		f0 = float3(0.04f, 0.04f, 0.04f);
	}
	if ((options & opt_writes_occlusion) == 0u)
	{
		occlusion = 1.0f;
	}
	if (!decal && abs(shading_model - 2.0f / 3.0f) < 0.01f)
	{
		f0 = float3(0.04f, 0.04f, 0.04f);
		uint previous;
		InterlockedOr(flags_out[0], 1u, previous);
	}

	cs_out[texel] = pack_colour(albedo, f0);
	const float2 oct = octahedral(normal);
	ng_out[texel] = uint4(to_byte(iw7_gloss(gloss)), to_byte(oct.x), (options & opt_occlusion) != 0u ? to_byte(occlusion) : 255u, to_byte(oct.y));
	if (decal)
	{
		covered_out[texel] = a0 > (1.0f / 1024.0f) ? 1u : 0u;
	}
	else if (raw)
	{
		covered_out[texel] = to_byte(t0.w);
	}
}
)hlsl";

			// compiled once for every device
			std::mutex pack_code_mutex;
			ComPtr<ID3DBlob> pack_code;
			std::string pack_code_error;

			ID3DBlob* compile_pack_shader(std::string& error)
			{
				std::lock_guard _(pack_code_mutex);
				if (!pack_code && pack_code_error.empty())
				{
					ComPtr<ID3DBlob> errors;
					if (FAILED(D3DCompile(pack_source, sizeof(pack_source) - 1, "bake_pack", nullptr, nullptr, "main", "cs_5_0",
						D3DCOMPILE_IEEE_STRICTNESS, 0, pack_code.GetAddressOf(), errors.GetAddressOf())))
					{
						pack_code.Reset();
						pack_code_error = errors ? static_cast<const char*>(errors->GetBufferPointer()) : "no message";
					}
				}
				error = pack_code_error;
				return pack_code.Get();
			}
		}

		std::array<std::uint32_t, 4> container_checksum(const std::uint8_t* data, const std::size_t size)
		{
			if (size < 32 || std::memcmp(data, "DXBC", 4) != 0)
			{
				throw std::runtime_error("not a DXBC container");
			}
			return dxbc_checksum(data, size);
		}

		std::unique_ptr<context> context::create()
		{
			std::unique_ptr<context> c(new context());
			const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };
			if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, levels, 1, D3D11_SDK_VERSION,
				c->device_.GetAddressOf(), nullptr, c->context_.GetAddressOf())))
			{
				return nullptr;
			}

			// one texel column more than a tile: the parity-shifted pass draws one texel to the right
			D3D11_TEXTURE2D_DESC target{};
			target.Width = max_tile + 1;
			target.Height = max_tile;
			target.MipLevels = 1;
			target.ArraySize = 1;
			target.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			target.SampleDesc.Count = 1;
			target.Usage = D3D11_USAGE_DEFAULT;
			target.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

			auto staging = target;
			staging.Usage = D3D11_USAGE_STAGING;
			staging.BindFlags = 0;
			staging.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

			for (auto i = 0; i < 4; i++)
			{
				if (FAILED(c->device_->CreateTexture2D(&target, nullptr, c->targets_[i].GetAddressOf()))
					|| FAILED(c->device_->CreateRenderTargetView(c->targets_[i].Get(), nullptr, c->target_views_[i].GetAddressOf()))
					|| FAILED(c->device_->CreateShaderResourceView(c->targets_[i].Get(), nullptr, c->target_srvs_[i].GetAddressOf())))
				{
					return nullptr;
				}
			}
			for (auto i = 0; i < 3; i++)
			{
				if (FAILED(c->device_->CreateTexture2D(&staging, nullptr, c->staging_[i].GetAddressOf())))
				{
					return nullptr;
				}
			}

			D3D11_BUFFER_DESC constants{};
			constants.ByteWidth = 32;
			constants.Usage = D3D11_USAGE_DYNAMIC;
			constants.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constants.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_RASTERIZER_DESC rasterizer{};
			rasterizer.FillMode = D3D11_FILL_SOLID;
			rasterizer.CullMode = D3D11_CULL_NONE;
			rasterizer.DepthClipEnable = TRUE;

			D3D11_DEPTH_STENCIL_DESC depth{};
			depth.DepthEnable = FALSE;
			depth.StencilEnable = FALSE;

			if (FAILED(c->device_->CreateBuffer(&constants, nullptr, c->tile_constants_.GetAddressOf()))
				|| FAILED(c->device_->CreateRasterizerState(&rasterizer, c->rasterizer_.GetAddressOf()))
				|| FAILED(c->device_->CreateDepthStencilState(&depth, c->depth_.GetAddressOf())))
			{
				return nullptr;
			}
			return c;
		}

		context::~context() = default;

		std::vector<std::uint8_t> context::bc7_level(const std::uint32_t w, const std::uint32_t h, const std::uint8_t* rgba)
		{
			D3D11_TEXTURE2D_DESC desc{};
			desc.Width = w;
			desc.Height = h;
			desc.MipLevels = 1;
			desc.ArraySize = 1;
			desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			desc.SampleDesc.Count = 1;
			desc.Usage = D3D11_USAGE_IMMUTABLE;
			desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			const D3D11_SUBRESOURCE_DATA init{ rgba, w * 4, w * h * 4 };
			ComPtr<ID3D11Texture2D> source;
			ComPtr<ID3D11ShaderResourceView> source_view;
			if (FAILED(this->device_->CreateTexture2D(&desc, &init, source.GetAddressOf()))
				|| FAILED(this->device_->CreateShaderResourceView(source.Get(), nullptr, source_view.GetAddressOf())))
			{
				throw std::runtime_error(utils::string::va("a %ux%u BC7 source texture could not be created", w, h));
			}
			return this->bc7_blocks(w, h, source_view.Get());
		}

		std::vector<std::uint8_t> context::bc7_blocks(const std::uint32_t w, const std::uint32_t h, ID3D11ShaderResourceView* const source)
		{
			auto* ctx = this->context_.Get();
			const auto blocks_x = std::max(1u, (w + 3) / 4);
			const auto blocks_y = std::max(1u, (h + 3) / 4);
			const auto blocks = blocks_x * blocks_y;

			if (!this->bc7_mode456_)
			{
				if (FAILED(this->device_->CreateComputeShader(bc7::BC7Encode_TryMode456CS, sizeof(bc7::BC7Encode_TryMode456CS), nullptr,
						this->bc7_mode456_.GetAddressOf()))
					|| FAILED(this->device_->CreateComputeShader(bc7::BC7Encode_TryMode137CS, sizeof(bc7::BC7Encode_TryMode137CS), nullptr,
						this->bc7_mode137_.GetAddressOf()))
					|| FAILED(this->device_->CreateComputeShader(bc7::BC7Encode_EncodeBlockCS, sizeof(bc7::BC7Encode_EncodeBlockCS), nullptr,
						this->bc7_encode_.GetAddressOf())))
				{
					throw std::runtime_error("the BC7 encoder shaders could not be created");
				}
				D3D11_BUFFER_DESC desc{};
				desc.ByteWidth = sizeof(bc7::constants);
				desc.Usage = D3D11_USAGE_DYNAMIC;
				desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
				desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				if (FAILED(this->device_->CreateBuffer(&desc, nullptr, this->bc7_constants_.GetAddressOf())))
				{
					throw std::runtime_error("the BC7 encoder constants could not be created");
				}
			}

			if (blocks > this->bc7_capacity_)
			{
				D3D11_BUFFER_DESC desc{};
				desc.ByteWidth = blocks * bc7::block_bytes;
				desc.Usage = D3D11_USAGE_DEFAULT;
				desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
				desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
				desc.StructureByteStride = bc7::block_bytes;
				D3D11_SHADER_RESOURCE_VIEW_DESC view{};
				view.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
				view.Buffer.NumElements = blocks;
				D3D11_UNORDERED_ACCESS_VIEW_DESC uav{};
				uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
				uav.Buffer.NumElements = blocks;
				for (auto i = 0; i < 3; i++)
				{
					if (FAILED(this->device_->CreateBuffer(&desc, nullptr, this->bc7_buffers_[i].ReleaseAndGetAddressOf()))
						|| FAILED(this->device_->CreateShaderResourceView(this->bc7_buffers_[i].Get(), &view, this->bc7_views_[i].ReleaseAndGetAddressOf()))
						|| FAILED(this->device_->CreateUnorderedAccessView(this->bc7_buffers_[i].Get(), &uav, this->bc7_uavs_[i].ReleaseAndGetAddressOf())))
					{
						throw std::runtime_error(utils::string::va("BC7 encoder buffers for %u blocks could not be created", blocks));
					}
				}
				D3D11_BUFFER_DESC readback{};
				readback.ByteWidth = blocks * bc7::block_bytes;
				readback.Usage = D3D11_USAGE_STAGING;
				readback.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
				if (FAILED(this->device_->CreateBuffer(&readback, nullptr, this->bc7_readback_.ReleaseAndGetAddressOf())))
				{
					throw std::runtime_error(utils::string::va("a BC7 readback buffer for %u blocks could not be created", blocks));
				}
				this->bc7_capacity_ = blocks;
			}

			const auto set_constants = [&](const std::uint32_t mode, const std::uint32_t start)
			{
				D3D11_MAPPED_SUBRESOURCE mapped{};
				if (FAILED(ctx->Map(this->bc7_constants_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
				{
					throw std::runtime_error("the BC7 encoder constants could not be mapped");
				}
				const bc7::constants c{ w, blocks_x, static_cast<std::uint32_t>(DXGI_FORMAT_BC7_UNORM), mode, start, blocks, 1.0f, 0 };
				std::memcpy(mapped.pData, &c, sizeof(c));
				ctx->Unmap(this->bc7_constants_.Get(), 0);
			};
			// as DirectXTex's RunComputeShader: the output is unbound before the views are swapped
			const auto dispatch = [&](ID3D11ComputeShader* shader, ID3D11ShaderResourceView* errors, ID3D11UnorderedAccessView* output,
				const std::uint32_t groups)
			{
				ID3D11UnorderedAccessView* none = nullptr;
				ctx->CSSetUnorderedAccessViews(0, 1, &none, nullptr);
				ctx->CSSetShader(shader, nullptr, 0);
				ID3D11ShaderResourceView* views[2] = { source, errors };
				ctx->CSSetShaderResources(0, 2, views);
				ctx->CSSetUnorderedAccessViews(0, 1, &output, nullptr);
				ctx->CSSetConstantBuffers(0, 1, this->bc7_constants_.GetAddressOf());
				ctx->Dispatch(groups, 1, 1);
			};

			for (auto start = 0u; start < blocks; start += bc7::batch)
			{
				const auto count = std::min(bc7::batch, blocks - start);
				// modes 4, 5, 6 -> error 1; mode 1: error 1 -> 2, mode 3: 2 -> 1, mode 7: 1 -> 2; encode from error 2
				set_constants(0, start);
				dispatch(this->bc7_mode456_.Get(), nullptr, this->bc7_uavs_[1].Get(), std::max((count + 3) / 4, 1u));
				constexpr std::uint32_t modes[] = { 1, 3, 7 };
				for (auto i = 0u; i < 3; i++)
				{
					set_constants(modes[i], start);
					dispatch(this->bc7_mode137_.Get(), (i & 1) ? this->bc7_views_[2].Get() : this->bc7_views_[1].Get(),
						(i & 1) ? this->bc7_uavs_[1].Get() : this->bc7_uavs_[2].Get(), count);
				}
				dispatch(this->bc7_encode_.Get(), this->bc7_views_[2].Get(), this->bc7_uavs_[0].Get(), std::max((count + 3) / 4, 1u));
			}

			ID3D11UnorderedAccessView* no_uav = nullptr;
			ctx->CSSetUnorderedAccessViews(0, 1, &no_uav, nullptr);
			ID3D11ShaderResourceView* no_views[2] = {};
			ctx->CSSetShaderResources(0, 2, no_views);

			const D3D11_BOX box{ 0, 0, 0, blocks * bc7::block_bytes, 1, 1 };
			ctx->CopySubresourceRegion(this->bc7_readback_.Get(), 0, 0, 0, 0, this->bc7_buffers_[0].Get(), 0, &box);
			D3D11_MAPPED_SUBRESOURCE mapped{};
			const auto hr = ctx->Map(this->bc7_readback_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
			if (FAILED(hr))
			{
				throw std::runtime_error(utils::string::va("the BC7 blocks could not be read back (0x%08X)", static_cast<unsigned int>(hr)));
			}
			std::vector<std::uint8_t> out(static_cast<const std::uint8_t*>(mapped.pData),
				static_cast<const std::uint8_t*>(mapped.pData) + static_cast<std::size_t>(blocks) * bc7::block_bytes);
			ctx->Unmap(this->bc7_readback_.Get(), 0);
			return out;
		}

		std::vector<std::vector<std::uint8_t>> context::compress_bc7(const std::uint32_t width, const std::uint32_t height,
			const std::vector<std::vector<std::uint8_t>>& levels)
		{
			std::vector<DirectX::Image> images(levels.size());
			for (auto l = 0u; l < levels.size(); l++)
			{
				auto& image = images[l];
				image.width = std::max(1u, width >> l);
				image.height = std::max(1u, height >> l);
				image.format = DXGI_FORMAT_R8G8B8A8_UNORM;
				image.rowPitch = image.width * 4;
				image.slicePitch = image.rowPitch * image.height;
				image.pixels = const_cast<std::uint8_t*>(levels[l].data());
			}
			// DirectXTex's encoding of levels [first, end)
			const auto directxtex = [&](const std::size_t first)
			{
				DirectX::TexMetadata meta{};
				meta.width = images[first].width;
				meta.height = images[first].height;
				meta.depth = 1;
				meta.arraySize = 1;
				meta.mipLevels = images.size() - first;
				meta.format = DXGI_FORMAT_R8G8B8A8_UNORM;
				meta.dimension = DirectX::TEX_DIMENSION_TEXTURE2D;
				DirectX::ScratchImage compressed;
				const auto hr = DirectX::Compress(this->device_.Get(), images.data() + first, images.size() - first, meta, DXGI_FORMAT_BC7_UNORM,
					DirectX::TEX_COMPRESS_DEFAULT, 1.0f, compressed);
				if (FAILED(hr))
				{
					throw std::runtime_error(utils::string::va("GPU block compression failed (0x%08X)", static_cast<unsigned int>(hr)));
				}
				std::vector<std::vector<std::uint8_t>> out(images.size() - first);
				for (auto l = 0u; l < out.size(); l++)
				{
					const auto* image = compressed.GetImage(l, 0, 0);
					out[l].assign(image->pixels, image->pixels + image->slicePitch);
				}
				return out;
			};

			if (this->bc7_checked_ < 0)
			{
				return directxtex(0);
			}

			std::vector<std::vector<std::uint8_t>> out(levels.size());
			for (auto l = 0u; l < levels.size(); l++)
			{
				out[l] = this->bc7_level(static_cast<std::uint32_t>(images[l].width), static_cast<std::uint32_t>(images[l].height), levels[l].data());
			}

			if (this->bc7_checked_ == 0)
			{
				// the levels up to 1024 x 1024 (a 1024 level takes four of our batches and 1024 of DirectXTex's)
				auto first = 0u;
				while (first + 1 < images.size() && std::max(images[first].width, images[first].height) > 1024)
				{
					first++;
				}
				const auto reference = directxtex(first);
				auto same = true;
				for (auto l = first; same && l < out.size(); l++)
				{
					same = reference[l - first] == out[l];
				}
				this->bc7_checked_ = same ? 1 : -1;
				if (!same)
				{
					ZONETOOL_WARNING("BC7: the batched encoder differs from DirectXTex on a %zux%zu chain; DirectXTex encodes the rest",
						images[first].width, images[first].height);
					return directxtex(0);
				}
				ZONETOOL_INFO("BC7: the batched encoder matches DirectXTex byte for byte on a %zux%zu chain (%zu levels)",
					images[first].width, images[first].height, out.size() - first);
			}
			return out;
		}

		void context::trim()
		{
			if (this->image_bytes_ > image_budget)
			{
				this->images_.clear();
				this->image_bytes_ = 0;
			}
		}

		ID3D11ShaderResourceView* context::image_view(const GfxImage* image)
		{
			const auto found = this->images_.find(image);
			if (found != this->images_.end())
			{
				return found->second.Get();
			}

			ComPtr<ID3D11Texture2D> texture;
			std::size_t bytes = 0;

			// the image as BO3 stores it, when the device samples that format
			gfximage::image_pixels pixels{};
			if (gfximage::get_pixels(image, pixels) && pixels.faces == 1 && pixels.depth == 1 && pixels.levels)
			{
				UINT support = 0;
				const auto sampled = SUCCEEDED(this->device_->CheckFormatSupport(pixels.format, &support))
					&& (support & D3D11_FORMAT_SUPPORT_TEXTURE2D) && (support & D3D11_FORMAT_SUPPORT_SHADER_SAMPLE);
				// Direct3D 11 wants block-compressed textures a whole number of blocks wide and high at the top
				const auto blocks = !DirectX::IsCompressed(pixels.format) || (pixels.width % 4 == 0 && pixels.height % 4 == 0);
				if (sampled && blocks)
				{
					std::vector<D3D11_SUBRESOURCE_DATA> init(pixels.levels);
					std::size_t offset = 0;
					auto whole = true;
					for (auto l = 0u; l < pixels.levels; l++)
					{
						std::size_t row = 0, slice = 0;
						if (FAILED(DirectX::ComputePitch(pixels.format, std::max(1u, pixels.width >> l), std::max(1u, pixels.height >> l), row, slice))
							|| offset + slice > pixels.data.size())
						{
							whole = false;
							break;
						}
						init[l].pSysMem = pixels.data.data() + offset;
						init[l].SysMemPitch = static_cast<UINT>(row);
						init[l].SysMemSlicePitch = static_cast<UINT>(slice);
						offset += slice;
					}

					D3D11_TEXTURE2D_DESC desc{};
					desc.Width = pixels.width;
					desc.Height = pixels.height;
					desc.MipLevels = pixels.levels;
					desc.ArraySize = 1;
					desc.Format = pixels.format;
					desc.SampleDesc.Count = 1;
					desc.Usage = D3D11_USAGE_IMMUTABLE;
					desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
					if (whole && SUCCEEDED(this->device_->CreateTexture2D(&desc, init.data(), texture.GetAddressOf())))
					{
						bytes = offset;
					}
				}
			}

			// otherwise the interpreter's decode of it (8-bit rgba keeping the sRGB bytes, or float)
			if (!texture)
			{
				const auto decoded = material_texture::decode(image);
				if (!decoded)
				{
					return nullptr;
				}
				std::vector<D3D11_SUBRESOURCE_DATA> init(decoded->level_count());
				for (auto l = 0u; l < decoded->level_count(); l++)
				{
					std::uint32_t w, h;
					decoded->level_size(l, w, h);
					init[l].pSysMem = decoded->is_float ? static_cast<const void*>(decoded->levels_f[l].data()) : decoded->levels8[l].data();
					init[l].SysMemPitch = w * (decoded->is_float ? 16u : 4u);
					init[l].SysMemSlicePitch = init[l].SysMemPitch * h;
					bytes += init[l].SysMemSlicePitch;
				}
				D3D11_TEXTURE2D_DESC desc{};
				desc.Width = decoded->width;
				desc.Height = decoded->height;
				desc.MipLevels = decoded->level_count();
				desc.ArraySize = 1;
				desc.Format = decoded->is_float ? DXGI_FORMAT_R32G32B32A32_FLOAT
					: (decoded->srgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM);
				desc.SampleDesc.Count = 1;
				desc.Usage = D3D11_USAGE_IMMUTABLE;
				desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
				if (FAILED(this->device_->CreateTexture2D(&desc, init.data(), texture.GetAddressOf())))
				{
					return nullptr;
				}
			}

			ComPtr<ID3D11ShaderResourceView> view;
			if (FAILED(this->device_->CreateShaderResourceView(texture.Get(), nullptr, view.GetAddressOf())))
			{
				return nullptr;
			}
			this->image_bytes_ += bytes;
			return (this->images_[image] = view).Get();
		}

		ID3D11ShaderResourceView* context::constant_view(const float value[4], const std::uint32_t width, const std::uint32_t height)
		{
			std::vector<float> key(value, value + 4);
			key.push_back(static_cast<float>(width));
			key.push_back(static_cast<float>(height));
			const auto found = this->constants_.find(key);
			if (found != this->constants_.end())
			{
				return found->second.Get();
			}

			// 8-bit when every component is exactly a byte value, float otherwise: the reads return `value` exactly
			const auto bytes = exact_unorm8(value[0]) && exact_unorm8(value[1]) && exact_unorm8(value[2]) && exact_unorm8(value[3]);
			const auto texel_size = bytes ? 4u : 16u;
			std::vector<std::uint8_t> data(static_cast<std::size_t>(width) * height * texel_size);
			for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; i++)
			{
				if (bytes)
				{
					for (auto c = 0; c < 4; c++)
					{
						data[i * 4 + c] = static_cast<std::uint8_t>(std::lround(value[c] * 255.0f));
					}
				}
				else
				{
					std::memcpy(&data[i * 16], value, 16);
				}
			}

			D3D11_TEXTURE2D_DESC desc{};
			desc.Width = width;
			desc.Height = height;
			desc.MipLevels = 1;
			desc.ArraySize = 1;
			desc.Format = bytes ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R32G32B32A32_FLOAT;
			desc.SampleDesc.Count = 1;
			desc.Usage = D3D11_USAGE_IMMUTABLE;
			desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			const D3D11_SUBRESOURCE_DATA init{ data.data(), width * texel_size, width * height * texel_size };

			ComPtr<ID3D11Texture2D> texture;
			ComPtr<ID3D11ShaderResourceView> view;
			if (FAILED(this->device_->CreateTexture2D(&desc, &init, texture.GetAddressOf()))
				|| FAILED(this->device_->CreateShaderResourceView(texture.Get(), nullptr, view.GetAddressOf())))
			{
				return nullptr;
			}
			return (this->constants_[key] = view).Get();
		}

		ID3D11SamplerState* context::sampler_state(const material_texture::address_mode u, const material_texture::address_mode v)
		{
			const std::pair<int, int> key{ static_cast<int>(u), static_cast<int>(v) };
			const auto found = this->samplers_.find(key);
			if (found != this->samplers_.end())
			{
				return found->second.Get();
			}

			// the interpreter's filtering: trilinear, no bias, every level
			D3D11_SAMPLER_DESC desc{};
			desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
			desc.AddressU = address(u);
			desc.AddressV = address(v);
			desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
			desc.MaxAnisotropy = 1;
			desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
			desc.MinLOD = 0.0f;
			desc.MaxLOD = D3D11_FLOAT32_MAX;
			ComPtr<ID3D11SamplerState> state;
			if (FAILED(this->device_->CreateSamplerState(&desc, state.GetAddressOf())))
			{
				return nullptr;
			}
			return (this->samplers_[key] = state).Get();
		}

		bool context::bind(const program& p, std::string& why, bool& shader_failed)
		{
			this->pixel_shader_ = nullptr;
			this->vertex_shader_ = nullptr;
			shader_failed = true;
			try
			{
				auto& ps = this->pixel_shaders_[p.bytecode];
				if (!ps)
				{
					const auto code = without_discard(p.bytecode, p.bytecode_size);
					if (FAILED(this->device_->CreatePixelShader(code.data(), code.size(), nullptr, ps.GetAddressOf())))
					{
						this->pixel_shaders_.erase(p.bytecode);
						throw std::runtime_error("the device does not accept the pixel shader");
					}
				}

				const auto source = vertex_source(p.inputs);
				auto& vs = this->vertex_shaders_[source];
				if (!vs)
				{
					ComPtr<ID3DBlob> blob, errors;
					if (FAILED(D3DCompile(source.data(), source.size(), "bake_vs", nullptr, nullptr, "main", "vs_5_0", 0, 0,
						blob.GetAddressOf(), errors.GetAddressOf())))
					{
						this->vertex_shaders_.erase(source);
						throw std::runtime_error(utils::string::va("the vertex shader does not compile: %s",
							errors ? static_cast<const char*>(errors->GetBufferPointer()) : "no message"));
					}
					try
					{
						check_linkage(blob.Get(), p.inputs);
					}
					catch (...)
					{
						this->vertex_shaders_.erase(source);
						throw;
					}
					if (FAILED(this->device_->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, vs.GetAddressOf())))
					{
						this->vertex_shaders_.erase(source);
						throw std::runtime_error("the device does not accept the vertex shader");
					}
				}

				// the material's data; every slot is set: what an earlier program bound must not leak into this one
				// (unbound reads are zero)
				shader_failed = false;
				this->cbuffers_.clear();
				this->cbuffer_slots_.assign(D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, nullptr);
				for (const auto& [slot, data] : p.cbuffers)
				{
					if (slot >= this->cbuffer_slots_.size())
					{
						throw std::runtime_error(utils::string::va("constant buffer slot %u", slot));
					}
					std::vector<std::uint8_t> padded(*data);
					padded.resize(std::max<std::size_t>(16, (padded.size() + 15) & ~static_cast<std::size_t>(15)), 0);
					D3D11_BUFFER_DESC desc{};
					desc.ByteWidth = static_cast<UINT>(padded.size());
					desc.Usage = D3D11_USAGE_IMMUTABLE;
					desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
					const D3D11_SUBRESOURCE_DATA init{ padded.data(), 0, 0 };
					ComPtr<ID3D11Buffer> buffer;
					if (FAILED(this->device_->CreateBuffer(&desc, &init, buffer.GetAddressOf())))
					{
						throw std::runtime_error(utils::string::va("constant buffer %u could not be created", slot));
					}
					this->cbuffer_slots_[slot] = buffer.Get();
					this->cbuffers_.emplace_back(std::move(buffer));
				}

				this->view_slots_.assign(D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, nullptr);
				for (const auto& t : p.textures)
				{
					if (t.slot >= this->view_slots_.size())
					{
						throw std::runtime_error(utils::string::va("texture slot %u", t.slot));
					}
					auto* view = t.constant ? this->constant_view(t.value, t.width, t.height) : this->image_view(t.image);
					if (!view)
					{
						throw std::runtime_error(utils::string::va("texture %u (%s) could not be uploaded", t.slot,
							t.constant ? "a constant" : t.image->name));
					}
					this->view_slots_[t.slot] = view;
				}
				if (p.pixel_constant_slot)
				{
					auto* view = this->constant_view(p.pixel_constant, max_tile + 1, max_tile);
					if (!view || *p.pixel_constant_slot >= this->view_slots_.size())
					{
						throw std::runtime_error("the per-pixel constant texture could not be created");
					}
					this->view_slots_[*p.pixel_constant_slot] = view;
				}

				this->sampler_slots_.assign(D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, nullptr);
				for (const auto& s : p.samplers)
				{
					auto* state = this->sampler_state(s.u, s.v);
					if (!state || s.slot >= this->sampler_slots_.size())
					{
						throw std::runtime_error(utils::string::va("sampler %u could not be created", s.slot));
					}
					this->sampler_slots_[s.slot] = state;
				}

				this->pixel_shader_ = ps.Get();
				this->vertex_shader_ = vs.Get();
				return true;
			}
			catch (const std::exception& e)
			{
				why = e.what();
				return false;
			}
		}

		void context::run(const level& l, const region& r, const std::uint32_t parity_shift, const std::uint32_t target_mask,
			std::vector<float> (&out)[3])
		{
			if (!this->pixel_shader_ || !this->vertex_shader_)
			{
				throw std::runtime_error("gpu evaluation without a bound program");
			}
			if (!r.width || !r.height || r.width > max_tile || r.height > max_tile || parity_shift > 1)
			{
				throw std::runtime_error(utils::string::va("gpu evaluation of a %ux%u region", r.width, r.height));
			}

			this->set_tile_constants(l, r);
			ID3D11RenderTargetView* const views[3] = { this->target_views_[0].Get(), this->target_views_[1].Get(), this->target_views_[2].Get() };
			this->draw_pass(parity_shift, r, views);

			auto* ctx = this->context_.Get();
			const D3D11_BOX box{ parity_shift, 0, 0, parity_shift + r.width, r.height, 1 };
			for (auto t = 0u; t < 3; t++)
			{
				if (target_mask & (1u << t))
				{
					ctx->CopySubresourceRegion(this->staging_[t].Get(), 0, 0, 0, 0, this->targets_[t].Get(), 0, &box);
				}
			}
			for (auto t = 0u; t < 3; t++)
			{
				if (!(target_mask & (1u << t)))
				{
					out[t].clear();
					continue;
				}
				D3D11_MAPPED_SUBRESOURCE read{};
				const auto hr = ctx->Map(this->staging_[t].Get(), 0, D3D11_MAP_READ, 0, &read);
				if (FAILED(hr))
				{
					throw std::runtime_error(utils::string::va("render target %u could not be read back (0x%08X)", t, static_cast<unsigned int>(hr)));
				}
				out[t].resize(static_cast<std::size_t>(r.width) * r.height * 4);
				for (auto y = 0u; y < r.height; y++)
				{
					std::memcpy(&out[t][static_cast<std::size_t>(y) * r.width * 4], static_cast<const std::uint8_t*>(read.pData) + static_cast<std::size_t>(y) * read.RowPitch,
						static_cast<std::size_t>(r.width) * 16);
				}
				ctx->Unmap(this->staging_[t].Get(), 0);
			}
		}

		void context::set_tile_constants(const level& l, const region& r)
		{
			auto* ctx = this->context_.Get();
			D3D11_MAPPED_SUBRESOURCE mapped{};
			if (FAILED(ctx->Map(this->tile_constants_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
			{
				throw std::runtime_error("the tile constants could not be mapped");
			}
			const auto du = l.span_u / static_cast<float>(l.width);
			const auto dv = l.span_v / static_cast<float>(l.height);
			const float constants[8] = { l.origin_u + static_cast<float>(r.x) * du, l.origin_v + static_cast<float>(r.y) * dv, du, dv,
				static_cast<float>(r.width), static_cast<float>(r.height), 0.0f, 0.0f };
			std::memcpy(mapped.pData, constants, sizeof(constants));
			ctx->Unmap(this->tile_constants_.Get(), 0);
		}

		void context::draw_pass(const std::uint32_t parity_shift, const region& r, ID3D11RenderTargetView* const (&views)[3])
		{
			auto* ctx = this->context_.Get();
			ctx->IASetInputLayout(nullptr);
			ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			ctx->VSSetShader(this->vertex_shader_, nullptr, 0);
			ctx->VSSetConstantBuffers(0, 1, this->tile_constants_.GetAddressOf());
			ctx->PSSetShader(this->pixel_shader_, nullptr, 0);
			ctx->PSSetConstantBuffers(0, static_cast<UINT>(this->cbuffer_slots_.size()), this->cbuffer_slots_.data());
			ctx->PSSetShaderResources(0, static_cast<UINT>(this->view_slots_.size()), this->view_slots_.data());
			ctx->PSSetSamplers(0, static_cast<UINT>(this->sampler_slots_.size()), this->sampler_slots_.data());
			ctx->RSSetState(this->rasterizer_.Get());
			ctx->OMSetDepthStencilState(this->depth_.Get(), 0);
			ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);

			const D3D11_VIEWPORT viewport{ static_cast<float>(parity_shift), 0.0f, static_cast<float>(r.width), static_cast<float>(r.height), 0.0f, 1.0f };
			ctx->RSSetViewports(1, &viewport);

			// the interpreter starts every output at zero
			constexpr float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			for (auto* view : views)
			{
				if (view)
				{
					ctx->ClearRenderTargetView(view, zero);
				}
			}
			ctx->OMSetRenderTargets(3, views, nullptr);
			ctx->Draw(3, 0);
			ctx->OMSetRenderTargets(0, nullptr, nullptr);
		}

		void context::draw(const level& l, const region& r, const bool other_parity)
		{
			if (!this->pixel_shader_ || !this->vertex_shader_)
			{
				throw std::runtime_error("gpu evaluation without a bound program");
			}
			if (!r.width || !r.height || r.width > max_tile || r.height > max_tile)
			{
				throw std::runtime_error(utils::string::va("gpu evaluation of a %ux%u region", r.width, r.height));
			}
			this->set_tile_constants(l, r);
			ID3D11RenderTargetView* const same[3] = { this->target_views_[0].Get(), this->target_views_[1].Get(), this->target_views_[2].Get() };
			this->draw_pass(0, r, same);
			if (other_parity)
			{
				ID3D11RenderTargetView* const other[3] = { nullptr, nullptr, this->target_views_[3].Get() };
				this->draw_pass(1, r, other);
			}
		}

		std::vector<std::uint8_t> context::read_texture(ID3D11Texture2D* const texture, const DXGI_FORMAT format, const std::uint32_t texel_size,
			const region& r)
		{
			auto* ctx = this->context_.Get();
			D3D11_TEXTURE2D_DESC desc{};
			desc.Width = r.width;
			desc.Height = r.height;
			desc.MipLevels = 1;
			desc.ArraySize = 1;
			desc.Format = format;
			desc.SampleDesc.Count = 1;
			desc.Usage = D3D11_USAGE_STAGING;
			desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			ComPtr<ID3D11Texture2D> staging;
			if (FAILED(this->device_->CreateTexture2D(&desc, nullptr, staging.GetAddressOf())))
			{
				throw std::runtime_error(utils::string::va("a %ux%u readback texture could not be created", r.width, r.height));
			}
			const D3D11_BOX box{ r.x, r.y, 0, r.x + r.width, r.y + r.height, 1 };
			ctx->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, texture, 0, &box);

			D3D11_MAPPED_SUBRESOURCE read{};
			const auto hr = ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &read);
			if (FAILED(hr))
			{
				throw std::runtime_error(utils::string::va("a %ux%u region could not be read back (0x%08X)", r.width, r.height, static_cast<unsigned int>(hr)));
			}
			const auto row = static_cast<std::size_t>(r.width) * texel_size;
			std::vector<std::uint8_t> out(row * r.height);
			for (auto y = 0u; y < r.height; y++)
			{
				std::memcpy(out.data() + y * row, static_cast<const std::uint8_t*>(read.pData) + static_cast<std::size_t>(y) * read.RowPitch, row);
			}
			ctx->Unmap(staging.Get(), 0);
			return out;
		}

		std::vector<float> context::read_target(const std::uint32_t target, const region& r)
		{
			if (target > 3)
			{
				throw std::runtime_error(utils::string::va("render target %u", target));
			}
			const auto shift = target == 3 ? 1u : 0u;
			const auto bytes = this->read_texture(this->targets_[target].Get(), DXGI_FORMAT_R32G32B32A32_FLOAT, 16, { r.x + shift, r.y, r.width, r.height });
			std::vector<float> out(bytes.size() / 4);
			std::memcpy(out.data(), bytes.data(), bytes.size());
			return out;
		}

		bool context::can_pack()
		{
			if (this->pack_state_)
			{
				return this->pack_state_ > 0;
			}
			this->pack_state_ = -1;

			std::string error;
			auto* code = compile_pack_shader(error);
			if (!code)
			{
				ZONETOOL_ERROR("material bake: the pack shader does not compile, texels are packed on the CPU: %s", error.data());
				return false;
			}

			D3D11_BUFFER_DESC constants{};
			constants.ByteWidth = 32;
			constants.Usage = D3D11_USAGE_DYNAMIC;
			constants.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constants.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_BUFFER_DESC flags{};
			flags.ByteWidth = 16;
			flags.Usage = D3D11_USAGE_DEFAULT;
			flags.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
			flags.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			flags.StructureByteStride = 4;
			D3D11_UNORDERED_ACCESS_VIEW_DESC flags_view{};
			flags_view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			flags_view.Buffer.NumElements = 4;
			const std::uint32_t zero[4] = {};
			const D3D11_SUBRESOURCE_DATA flags_init{ zero, 0, 0 };

			D3D11_BUFFER_DESC readback{};
			readback.ByteWidth = 16;
			readback.Usage = D3D11_USAGE_STAGING;
			readback.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

			if (FAILED(this->device_->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, this->pack_shader_.GetAddressOf()))
				|| FAILED(this->device_->CreateBuffer(&constants, nullptr, this->pack_constants_.GetAddressOf()))
				|| FAILED(this->device_->CreateBuffer(&flags, &flags_init, this->flags_.GetAddressOf()))
				|| FAILED(this->device_->CreateUnorderedAccessView(this->flags_.Get(), &flags_view, this->flags_uav_.GetAddressOf()))
				|| FAILED(this->device_->CreateBuffer(&readback, nullptr, this->flags_readback_.GetAddressOf())))
			{
				ZONETOOL_ERROR("material bake: the device does not take the pack shader or its buffers, texels are packed on the CPU");
				return false;
			}
			this->pack_state_ = 1;
			return true;
		}

		void context::begin_level(const std::uint32_t width, const std::uint32_t height, const bool covered)
		{
			// 8-bit rgba: written as bytes (UINT), read by the BC7 encoder as UNORM
			D3D11_TEXTURE2D_DESC desc{};
			desc.Width = width;
			desc.Height = height;
			desc.MipLevels = 1;
			desc.ArraySize = 1;
			desc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
			desc.SampleDesc.Count = 1;
			desc.Usage = D3D11_USAGE_DEFAULT;
			desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			D3D11_UNORDERED_ACCESS_VIEW_DESC uav{};
			uav.Format = DXGI_FORMAT_R8G8B8A8_UINT;
			uav.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
			D3D11_SHADER_RESOURCE_VIEW_DESC srv{};
			srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
			srv.Texture2D.MipLevels = 1;
			for (auto i = 0; i < 2; i++)
			{
				if (FAILED(this->device_->CreateTexture2D(&desc, nullptr, this->packed_[i].ReleaseAndGetAddressOf()))
					|| FAILED(this->device_->CreateUnorderedAccessView(this->packed_[i].Get(), &uav, this->packed_uavs_[i].ReleaseAndGetAddressOf()))
					|| FAILED(this->device_->CreateShaderResourceView(this->packed_[i].Get(), &srv, this->packed_views_[i].ReleaseAndGetAddressOf())))
				{
					throw std::runtime_error(utils::string::va("a %ux%u packed level could not be created", width, height));
				}
			}

			this->covered_.Reset();
			this->covered_uav_.Reset();
			if (covered)
			{
				auto cover = desc;
				cover.Format = DXGI_FORMAT_R8_UINT;
				cover.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
				if (FAILED(this->device_->CreateTexture2D(&cover, nullptr, this->covered_.GetAddressOf()))
					|| FAILED(this->device_->CreateUnorderedAccessView(this->covered_.Get(), nullptr, this->covered_uav_.GetAddressOf())))
				{
					throw std::runtime_error(utils::string::va("a %ux%u coverage level could not be created", width, height));
				}
			}
			this->level_width_ = width;
			this->level_height_ = height;
		}

		void context::pack(const region& r, const pack_options& o)
		{
			if (this->pack_state_ <= 0 || !this->packed_[0] || r.x + r.width > this->level_width_ || r.y + r.height > this->level_height_)
			{
				throw std::runtime_error("packing without a pack shader or a level");
			}
			if ((o.decal || o.raw) && !this->covered_)
			{
				throw std::runtime_error("a decal or surface level without coverage");
			}
			auto* ctx = this->context_.Get();

			D3D11_MAPPED_SUBRESOURCE mapped{};
			if (FAILED(ctx->Map(this->pack_constants_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
			{
				throw std::runtime_error("the pack constants could not be mapped");
			}
			const auto options = (o.decal ? 1u : 0u) | (o.writes_normal ? 2u : 0u) | (o.writes_gloss ? 4u : 0u) | (o.writes_specular ? 8u : 0u)
				| (o.writes_occlusion ? 16u : 0u) | (o.occlusion ? 32u : 0u) | (o.coloured_specular ? 64u : 0u) | (o.raw ? 128u : 0u);
			std::uint32_t constants[8] = { r.x, r.y, r.width, r.height, options, std::bit_cast<std::uint32_t>(o.underlying_gloss), 0, 0 };
			std::memcpy(mapped.pData, constants, sizeof(constants));
			ctx->Unmap(this->pack_constants_.Get(), 0);

			ID3D11ShaderResourceView* views[4] = { this->target_srvs_[0].Get(), this->target_srvs_[1].Get(), this->target_srvs_[2].Get(),
				this->target_srvs_[3].Get() };
			ID3D11UnorderedAccessView* uavs[4] = { this->packed_uavs_[0].Get(), this->packed_uavs_[1].Get(), this->covered_uav_.Get(),
				this->flags_uav_.Get() };
			ctx->CSSetShader(this->pack_shader_.Get(), nullptr, 0);
			ctx->CSSetConstantBuffers(0, 1, this->pack_constants_.GetAddressOf());
			ctx->CSSetShaderResources(0, 4, views);
			ctx->CSSetUnorderedAccessViews(0, 4, uavs, nullptr);
			ctx->Dispatch((r.width + 7) / 8, (r.height + 7) / 8, 1);

			// the targets are drawn to again next
			ID3D11ShaderResourceView* no_views[4] = {};
			ID3D11UnorderedAccessView* no_uavs[4] = {};
			ctx->CSSetShaderResources(0, 4, no_views);
			ctx->CSSetUnorderedAccessViews(0, 4, no_uavs, nullptr);
		}

		std::vector<std::uint8_t> context::read_packed(const std::uint32_t which, const region& r)
		{
			if (which > 1 || !this->packed_[which])
			{
				throw std::runtime_error("no packed level to read");
			}
			return this->read_texture(this->packed_[which].Get(), DXGI_FORMAT_R8G8B8A8_TYPELESS, 4, r);
		}

		std::vector<std::uint8_t> context::read_covered()
		{
			if (!this->covered_)
			{
				throw std::runtime_error("no coverage level to read");
			}
			return this->read_texture(this->covered_.Get(), DXGI_FORMAT_R8_UINT, 1, { 0, 0, this->level_width_, this->level_height_ });
		}

		std::vector<std::uint8_t> context::compress_packed(const std::uint32_t which)
		{
			if (which > 1 || !this->packed_[which])
			{
				throw std::runtime_error("no packed level to compress");
			}
			if (this->bc7_checked_ != 1)
			{
				// compress_bc7 checks the encoder against DirectXTex on its first chain and has DirectXTex encode when they
				// differ; both take the 8-bit texels
				const auto rgba = this->read_packed(which, { 0, 0, this->level_width_, this->level_height_ });
				return std::move(this->compress_bc7(this->level_width_, this->level_height_, { rgba })[0]);
			}
			return this->bc7_blocks(this->level_width_, this->level_height_, this->packed_views_[which].Get());
		}

		bool context::take_backlit()
		{
			if (!this->flags_)
			{
				return false;
			}
			auto* ctx = this->context_.Get();
			const D3D11_BOX box{ 0, 0, 0, 16, 1, 1 };
			ctx->CopySubresourceRegion(this->flags_readback_.Get(), 0, 0, 0, 0, this->flags_.Get(), 0, &box);
			D3D11_MAPPED_SUBRESOURCE read{};
			if (FAILED(ctx->Map(this->flags_readback_.Get(), 0, D3D11_MAP_READ, 0, &read)))
			{
				throw std::runtime_error("the pack flags could not be read back");
			}
			std::uint32_t value;
			std::memcpy(&value, read.pData, sizeof(value));
			ctx->Unmap(this->flags_readback_.Get(), 0);
			const std::uint32_t zero[4] = {};
			ctx->UpdateSubresource(this->flags_.Get(), 0, nullptr, zero, 0, 0);
			return value != 0;
		}
	}
}
