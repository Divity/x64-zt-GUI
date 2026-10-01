#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "probe_lighting.hpp"

#include "../parallel.hpp"
#include "zonetool/t7/common/xpak.hpp"

#include <DirectXTex.h>
#include <DirectXPackedVector.h>
#include <utils/string.hpp>

// Everything here mirrors BO3 code (client BlackOps3.exe unless noted):
// * deferred_lighting compute shader, reflection probe loop: per probe, a weight from its blend
//   entries (cb11, six planes each, prod saturate(dot(N, d) + W), d = p - cullOrigin), combined by
//   2-bit flags; override probes first, the others only fill what the overrides leave (weight
//   saturate(y - W_override)); each probe adds weight * exposure *
//   (X n'x^2 + Y n'y^2 + Z n'z^2)(uvw) * cube(n', mip 6) with n' the probe-rotated normal and the
//   upper half of the volume's w for negative components; the sum is divided by max(W, 1) and,
//   below W 0.99, the global probe (entry 0, unrotated, cube slice 0) adds (1 - W) * its exposure.
// * packer 0x141CBE570: probes 1..localReflectionProbeCount (0x141CCC610) that are not
//   exploderDisabled[0]; blend entries 0x141C44B80 (planes relative to the probe's cull origin),
//   the cb10 words 0x141C444A0 / 0x141CC1440 (1 / (size_min + size_max), volumeCoordMul/Add,
//   exploderFade[0], exposure, cube slice = probe index); the global probe 0x141C43140 and its
//   exposure 0x141CE01D0 (configs[0].exposure).
// * sun volume choice 0x141D04BA0; texture layout 0x141CA9CE0: mip-major, then array element, then face
//   (each probe's six faces together: the seams between them are continuous, face-major is not);
//   probe volume buffer = X || Y || Z (0x141D0B3C0).
// * samplers: volumes samplerLinearClamp (trilinear, clamp), cubes samplerLinear at lod 6.

namespace zonetool::t7
{
	namespace converter::iw7::probe_lighting
	{
		// D3D cube faces: direction -> (face, s, t) with s, t in [-1, 1], and back
		void cube_face(const float d[3], std::uint32_t& face, float& s, float& t)
		{
			const auto ax = std::fabs(d[0]), ay = std::fabs(d[1]), az = std::fabs(d[2]);
			if (ax >= ay && ax >= az)
			{
				face = d[0] >= 0.0f ? 0 : 1;
				s = (d[0] >= 0.0f ? -d[2] : d[2]) / ax;
				t = -d[1] / ax;
			}
			else if (ay >= az)
			{
				face = d[1] >= 0.0f ? 2 : 3;
				s = d[0] / ay;
				t = (d[1] >= 0.0f ? d[2] : -d[2]) / ay;
			}
			else
			{
				face = d[2] >= 0.0f ? 4 : 5;
				s = (d[2] >= 0.0f ? d[0] : -d[0]) / az;
				t = -d[1] / az;
			}
		}

		void face_direction(const std::uint32_t face, const float s, const float t, float d[3])
		{
			switch (face)
			{
			case 0: d[0] = 1.0f; d[1] = -t; d[2] = -s; break;
			case 1: d[0] = -1.0f; d[1] = -t; d[2] = s; break;
			case 2: d[0] = s; d[1] = 1.0f; d[2] = t; break;
			case 3: d[0] = s; d[1] = -1.0f; d[2] = -t; break;
			case 4: d[0] = s; d[1] = -t; d[2] = 1.0f; break;
			default: d[0] = -s; d[1] = -t; d[2] = -1.0f; break;
			}
		}

		std::vector<std::uint8_t> read_stream_buffer(const StreamWrappedBuffer* buffer, const char* what)
		{
			if (!buffer)
			{
				throw std::runtime_error(utils::string::va("%s has no pixel data buffer", what));
			}
			// resident buffers (the empty sun volume's 1x1x1 dummies) keep their bytes in the zone
			if (buffer->mode == SWM_RESIDENT)
			{
				if (!buffer->data)
				{
					throw std::runtime_error(utils::string::va("%s is resident but has no data", what));
				}
				return { buffer->data, buffer->data + buffer->dataSize };
			}
			auto data = xpak::get_data_for_xpak_key(buffer->xpakEntry.key, buffer->dataSize);
			if (data.size() != buffer->dataSize)
			{
				throw std::runtime_error(utils::string::va("%s: xpak entry %016llX gave %zu of %u bytes", what,
					static_cast<unsigned long long>(buffer->xpakEntry.key), data.size(), buffer->dataSize));
			}
			return data;
		}

		namespace
		{
			constexpr auto cube_lod = 6u;

			struct plane
			{
				float n[3];
				float w;
			};

			struct blend_entry
			{
				plane planes[6];
			};

			struct probe
			{
				float cull_origin[3];
				// the box the tile culling tests (0x141C42440): cullOrigin +- sum |rotation row| * half size
				float bounds_min[3];
				float bounds_max[3];
				float rotation[3][3];
				float inv_size[3];
				float coord_mul[3];
				float coord_add[3];
				float exposure;
				float fade;
				unsigned int slice;
				bool is_override;
				std::vector<blend_entry> entries;
				unsigned int flags;
				float texel_size;
			};

			// one decoded 3D texture, RGB halves
			struct volume_texture
			{
				std::uint32_t width = 0;
				std::uint32_t height = 0;
				std::uint32_t depth = 0;
				std::vector<std::uint16_t> texels;

				void fetch(std::int32_t x, std::int32_t y, std::int32_t z, float out[3]) const
				{
					x = std::clamp<std::int32_t>(x, 0, static_cast<std::int32_t>(width) - 1);
					y = std::clamp<std::int32_t>(y, 0, static_cast<std::int32_t>(height) - 1);
					z = std::clamp<std::int32_t>(z, 0, static_cast<std::int32_t>(depth) - 1);
					const auto* t = &texels[((static_cast<std::size_t>(z) * height + y) * width + x) * 3];
					for (auto c = 0; c < 3; c++)
					{
						out[c] = DirectX::PackedVector::XMConvertHalfToFloat(t[c]);
					}
				}

				// trilinear, clamp addressing
				void sample(const float uvw[3], float out[3]) const
				{
					const float fx = uvw[0] * static_cast<float>(width) - 0.5f;
					const float fy = uvw[1] * static_cast<float>(height) - 0.5f;
					const float fz = uvw[2] * static_cast<float>(depth) - 0.5f;
					const auto x0 = static_cast<std::int32_t>(std::floor(fx));
					const auto y0 = static_cast<std::int32_t>(std::floor(fy));
					const auto z0 = static_cast<std::int32_t>(std::floor(fz));
					const auto ax = fx - static_cast<float>(x0);
					const auto ay = fy - static_cast<float>(y0);
					const auto az = fz - static_cast<float>(z0);
					out[0] = out[1] = out[2] = 0.0f;
					for (auto k = 0; k < 8; k++)
					{
						const auto dx = k & 1, dy = (k >> 1) & 1, dz = (k >> 2) & 1;
						const auto weight = (dx ? ax : 1.0f - ax) * (dy ? ay : 1.0f - ay) * (dz ? az : 1.0f - az);
						float t[3];
						fetch(x0 + dx, y0 + dy, z0 + dz, t);
						for (auto c = 0; c < 3; c++)
						{
							out[c] += weight * t[c];
						}
					}
				}
			};

			// one mip of a cube array, RGB floats: slice, face, row, column
			struct cube_mip
			{
				std::uint32_t size = 0;
				std::uint32_t slices = 0;
				std::vector<float> texels;

				const float* texel(const std::uint32_t slice, const std::uint32_t face, const std::uint32_t x, const std::uint32_t y) const
				{
					return &texels[(((static_cast<std::size_t>(slice) * 6 + face) * size + y) * size + x) * 3];
				}
			};

			// bilinear on a cube level of `width` texels, with the taps that fall off a face taken from the
			// face they fall onto; texel(face, x, y) gives an RGB texel
			template <typename F>
			void bilinear_cube(const std::uint32_t width, const float dir[3], F&& texel, float out[3])
			{
				std::uint32_t face;
				float s, t;
				cube_face(dir, face, s, t);
				const auto size = static_cast<float>(width);
				const auto fx = (s + 1.0f) * 0.5f * size - 0.5f;
				const auto fy = (t + 1.0f) * 0.5f * size - 0.5f;
				const auto x0 = static_cast<std::int32_t>(std::floor(fx));
				const auto y0 = static_cast<std::int32_t>(std::floor(fy));
				const auto ax = fx - static_cast<float>(x0);
				const auto ay = fy - static_cast<float>(y0);
				out[0] = out[1] = out[2] = 0.0f;
				for (auto k = 0; k < 4; k++)
				{
					const auto x = x0 + (k & 1);
					const auto y = y0 + (k >> 1);
					const auto weight = ((k & 1) ? ax : 1.0f - ax) * ((k >> 1) ? ay : 1.0f - ay);
					std::uint32_t tap_face = face;
					auto tx = x, ty = y;
					if (x < 0 || y < 0 || x >= static_cast<std::int32_t>(width) || y >= static_cast<std::int32_t>(width))
					{
						float d[3];
						face_direction(face, (static_cast<float>(x) + 0.5f) / size * 2.0f - 1.0f,
							(static_cast<float>(y) + 0.5f) / size * 2.0f - 1.0f, d);
						float ts, tt;
						cube_face(d, tap_face, ts, tt);
						tx = std::clamp(static_cast<std::int32_t>(std::floor((ts + 1.0f) * 0.5f * size)), 0, static_cast<std::int32_t>(width) - 1);
						ty = std::clamp(static_cast<std::int32_t>(std::floor((tt + 1.0f) * 0.5f * size)), 0, static_cast<std::int32_t>(width) - 1);
					}
					const float* c = texel(tap_face, static_cast<std::uint32_t>(tx), static_cast<std::uint32_t>(ty));
					for (auto i = 0; i < 3; i++)
					{
						out[i] += weight * c[i];
					}
				}
			}

			void sample_cube(const cube_mip& mip, const std::uint32_t slice, const float dir[3], float out[3])
			{
				bilinear_cube(mip.size, dir, [&](const std::uint32_t face, const std::uint32_t x, const std::uint32_t y)
				{
					return mip.texel(slice, face, x, y);
				}, out);
			}

			float dot3(const float* a, const float* b)
			{
				return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
			}

			float saturate(const float v)
			{
				return std::clamp(v, 0.0f, 1.0f);
			}

			// 0x141C44B80: the six planes of one entry (group `group` of a blend's faces, or its box)
			bool blend_planes(const GfxConfig_Probe& config, const GfxConfig_ProbeBlend& blend, const std::uint32_t group, plane out[6])
			{
				const float o[3] = { blend.origin[0] - config.cullOrigin[0], blend.origin[1] - config.cullOrigin[1],
					blend.origin[2] - config.cullOrigin[2] };

				if (blend.faceCount)
				{
					for (auto k = 0u; k < 6; k++)
					{
						const auto index = 6 * group + k;
						if (index >= blend.faceCount)
						{
							out[k] = { { 1.0f, 0.0f, 0.0f }, config.cullRadius * 2.0f - o[0] };
							continue;
						}
						const auto* f = blend.faces[index];
						float n[3];
						for (auto c = 0; c < 3; c++)
						{
							n[c] = f[0] * blend.rotation[0][c] + f[1] * blend.rotation[1][c] + f[2] * blend.rotation[2][c];
						}
						const auto distance = blend.blends[index];
						const auto w = ((blend.negative ? 0.0f : distance) - f[3]) - dot3(o, n);
						const auto scale = 1.0f / (0.1f - distance >= 0.0f ? 0.1f : distance);
						out[k] = { { n[0] * scale, n[1] * scale, n[2] * scale }, w * scale };
					}
					return 6 * group + 6 < blend.faceCount;
				}

				for (auto k = 0; k < 3; k++)
				{
					const auto* r = blend.rotation[k];
					const auto max_scale = 1.0f / (blend.blend_maxs[k] <= 0.01f ? 0.01f : blend.blend_maxs[k]);
					const auto min_scale = 1.0f / (blend.blend_mins[k] <= 0.01f ? 0.01f : blend.blend_mins[k]);
					const float negative[3] = { -r[0], -r[1], -r[2] };
					const auto w_max = (blend.size_max[k] + blend.blend_maxs[k]) - dot3(negative, o);
					const auto w_min = (blend.size_min[k] + blend.blend_mins[k]) - dot3(r, o);
					out[k] = { { negative[0] * max_scale, negative[1] * max_scale, negative[2] * max_scale }, w_max * max_scale };
					out[k + 3] = { { r[0] * min_scale, r[1] * min_scale, r[2] * min_scale }, w_min * min_scale };
				}
				return false;
			}

			float entry_weight(const blend_entry& entry, const float d[3])
			{
				auto w = 1.0f;
				for (const auto& p : entry.planes)
				{
					w *= saturate(dot3(p.n, d) + p.w);
				}
				return w;
			}

			// the shader's blend combination (see the file comment)
			float blend_weight(const probe& pr, const float d[3])
			{
				if (pr.entries.empty())
				{
					return 0.0f;
				}
				const auto w0 = entry_weight(pr.entries[0], d);
				float x, y;
				if (pr.flags & 1)
				{
					x = w0;
					y = 0.0f;
				}
				else
				{
					x = 1.0f;
					y = w0;
				}
				for (auto k = 1u; k < pr.entries.size(); k++)
				{
					const auto bits = (pr.flags >> (2 * k)) & 3u;
					const auto xk = x * entry_weight(pr.entries[k], d);
					const auto combined = (bits & 2) ? (1.0f - xk) * y : std::max(xk, y);
					if (bits & 1)
					{
						x = xk;
					}
					else
					{
						x = 1.0f;
						y = combined;
					}
				}
				return y;
			}

			std::uint32_t bc_bytes(const std::uint32_t width, const std::uint32_t height)
			{
				return std::max(1u, (width + 3) / 4) * std::max(1u, (height + 3) / 4) * 16u;
			}

			// one 3D texture's texels (BC6H for baked volumes, R11G11B10 for the empty sun volume's dummies)
			void decode_volume(const GfxImage* image, const std::uint8_t* data, volume_texture& out)
			{
				if (image->mapType != MAPTYPE_3D)
				{
					throw std::runtime_error(utils::string::va("probe volume %s is map type %d, not a 3D texture", image->name, image->mapType));
				}
				const auto format = static_cast<DXGI_FORMAT>(image->format);
				out.width = image->width;
				out.height = image->height;
				out.depth = image->depth;
				out.texels.resize(static_cast<std::size_t>(out.width) * out.height * out.depth * 3);
				std::size_t row_pitch = 0, slice_bytes = 0;
				if (FAILED(DirectX::ComputePitch(format, out.width, out.height, row_pitch, slice_bytes)))
				{
					throw std::runtime_error(utils::string::va("probe volume %s has format %d", image->name, image->format));
				}
				if (slice_bytes * out.depth > image->totalSize)
				{
					throw std::runtime_error(utils::string::va("probe volume %s: %zu bytes per slice x %u slices exceed its %u bytes",
						image->name, slice_bytes, out.depth, image->totalSize));
				}

				parallel_for(out.depth, [&](const std::uint32_t z, std::uint32_t)
				{
					DirectX::Image src{};
					src.width = out.width;
					src.height = out.height;
					src.format = format;
					src.rowPitch = row_pitch;
					src.slicePitch = slice_bytes;
					src.pixels = const_cast<std::uint8_t*>(data) + static_cast<std::size_t>(z) * slice_bytes;
					DirectX::ScratchImage decoded;
					const auto hr = DirectX::IsCompressed(format)
						? DirectX::Decompress(src, DXGI_FORMAT_R32G32B32A32_FLOAT, decoded)
						: DirectX::Convert(src, DXGI_FORMAT_R32G32B32A32_FLOAT, DirectX::TEX_FILTER_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, decoded);
					if (FAILED(hr))
					{
						throw std::runtime_error(utils::string::va("could not decode slice %u of %s", z, image->name));
					}
					const auto* img = decoded.GetImage(0, 0, 0);
					for (auto y = 0u; y < out.height; y++)
					{
						const auto* row = reinterpret_cast<const float*>(img->pixels + img->rowPitch * y);
						auto* dst = &out.texels[((static_cast<std::size_t>(z) * out.height + y) * out.width) * 3];
						for (auto x = 0u; x < out.width; x++)
						{
							for (auto c = 0; c < 3; c++)
							{
								dst[x * 3 + c] = DirectX::PackedVector::XMConvertFloatToHalf(row[x * 4 + c]);
							}
						}
					}
				});
			}

			// the cube array's mip `lod`, from its mip-major data (mip, then array element, then face)
			void decode_cube_mip(const GfxImage* image, const std::uint8_t* data, const std::size_t size, const std::uint32_t lod, cube_mip& out)
			{
				if (image->format != DXGI_FORMAT_BC6H_UF16 || image->mapType != MAPTYPE_CUBE_ARRAY || image->width != image->height)
				{
					throw std::runtime_error(utils::string::va("probe cubes %s are format %d map type %d %ux%u, not a square BC6H cube array",
						image->name, image->format, image->mapType, image->width, image->height));
				}
				if (lod >= image->levelCount)
				{
					throw std::runtime_error(utils::string::va("probe cubes %s have %u levels, lod %u is needed", image->name, image->levelCount, lod));
				}
				const auto slices = static_cast<std::uint32_t>(image->depth);
				std::size_t offset = 0;
				for (auto m = 0u; m < lod; m++)
				{
					const auto w = std::max(1u, static_cast<std::uint32_t>(image->width) >> m);
					offset += static_cast<std::size_t>(bc_bytes(w, w)) * 6 * slices;
				}
				const auto w = std::max(1u, static_cast<std::uint32_t>(image->width) >> lod);
				const auto face_bytes = bc_bytes(w, w);
				if (offset + static_cast<std::size_t>(face_bytes) * 6 * slices > size)
				{
					throw std::runtime_error(utils::string::va("probe cubes %s: %zu bytes do not reach mip %u", image->name, size, lod));
				}

				out.size = w;
				out.slices = slices;
				out.texels.assign(static_cast<std::size_t>(slices) * 6 * w * w * 3, 0.0f);
				for (auto face = 0u; face < 6; face++)
				{
					for (auto slice = 0u; slice < slices; slice++)
					{
						DirectX::Image src{};
						src.width = w;
						src.height = w;
						src.format = DXGI_FORMAT_BC6H_UF16;
						src.rowPitch = static_cast<std::size_t>(std::max(1u, (w + 3) / 4)) * 16;
						src.slicePitch = face_bytes;
						src.pixels = const_cast<std::uint8_t*>(data) + offset + (static_cast<std::size_t>(slice) * 6 + face) * face_bytes;
						DirectX::ScratchImage decoded;
						if (FAILED(DirectX::Decompress(src, DXGI_FORMAT_R32G32B32A32_FLOAT, decoded)))
						{
							throw std::runtime_error(utils::string::va("could not decode %s slice %u face %u", image->name, slice, face));
						}
						const auto* img = decoded.GetImage(0, 0, 0);
						for (auto y = 0u; y < w; y++)
						{
							const auto* row = reinterpret_cast<const float*>(img->pixels + img->rowPitch * y);
							for (auto x = 0u; x < w; x++)
							{
								auto* dst = const_cast<float*>(out.texel(slice, face, x, y));
								dst[0] = row[x * 4 + 0];
								dst[1] = row[x * 4 + 1];
								dst[2] = row[x * 4 + 2];
							}
						}
					}
				}
			}

			probe make_probe(const GfxReflectionProbe& src, const unsigned int slice)
			{
				probe p{};
				const auto& config = src.config;
				std::memcpy(p.cull_origin, config.cullOrigin, sizeof(p.cull_origin));
				std::memcpy(p.rotation, config.rotation, sizeof(p.rotation));
				for (auto k = 0; k < 3; k++)
				{
					const auto half = std::max(0.0001f, (config.size_min[k] + config.size_max[k]) / 2.0f);
					p.inv_size[k] = 0.5f / half;
				}
				for (auto c = 0; c < 3; c++)
				{
					auto extent = 0.0f;
					for (auto k = 0; k < 3; k++)
					{
						extent += std::fabs(config.rotation[k][c]) * ((config.size_min[k] + config.size_max[k]) / 2.0f);
					}
					p.bounds_min[c] = config.cullOrigin[c] - extent;
					p.bounds_max[c] = config.cullOrigin[c] + extent;
				}
				std::memcpy(p.coord_mul, src.volumeCoordMul, sizeof(p.coord_mul));
				std::memcpy(p.coord_add, src.volumeCoordAdd, sizeof(p.coord_add));
				p.exposure = src.exposure;
				p.fade = src.exploderFade[0];
				p.slice = slice;
				p.is_override = config.isOverride;
				p.flags = 0;

				// the grid's texels sit at t = i / (gridDim - 1) across the box
				auto texel = 0.0f;
				for (auto k = 0; k < 3; k++)
				{
					const auto cells = std::max(1u, config.gridDim[k] > 1 ? config.gridDim[k] - 1 : 1u);
					texel = std::max(texel, (config.size_min[k] + config.size_max[k]) / static_cast<float>(cells));
				}
				// a probe without a box (the empty sun volume's) lights every point the same
				p.texel_size = texel > 0.0f ? texel : std::numeric_limits<float>::infinity();
				return p;
			}

			// one probe's contribution at a point: the X/Y/Z volume colours (both halves of w, picked
			// by the sign of each rotated normal component) are the same for every normal there
			struct probe_at_point
			{
				const probe* p;
				bool rotated;
				float scale;
				float colours[3][2][3]; // axis, positive / negative, rgb
			};

			void prepare_probe(const probe& p, const volume_texture& vx, const volume_texture& vy, const volume_texture& vz,
				const float d[3], const bool rotated, const float scale, probe_at_point& out)
			{
				float local[3];
				for (auto k = 0; k < 3; k++)
				{
					local[k] = rotated ? dot3(d, p.rotation[k]) : d[k];
				}
				float uvw[3];
				for (auto k = 0; k < 3; k++)
				{
					uvw[k] = saturate(local[k] * p.inv_size[k] + 0.5f) * p.coord_mul[k] + p.coord_add[k];
				}
				out.p = &p;
				out.rotated = rotated;
				out.scale = scale;
				const volume_texture* textures[3] = { &vx, &vy, &vz };
				for (auto axis = 0; axis < 3; axis++)
				{
					for (auto half = 0; half < 2; half++)
					{
						const float at[3] = { uvw[0], uvw[1], uvw[2] + (half ? 0.5f : 0.0f) };
						textures[axis]->sample(at, out.colours[axis][half]);
					}
				}
			}

			void add_probe(const probe_at_point& pp, const cube_mip& cubes, const float n[3], float out[3])
			{
				float nl[3];
				for (auto k = 0; k < 3; k++)
				{
					nl[k] = pp.rotated ? dot3(n, pp.p->rotation[k]) : n[k];
				}
				float ambient[3] = { 0.0f, 0.0f, 0.0f };
				for (auto axis = 0; axis < 3; axis++)
				{
					const auto* c = pp.colours[axis][nl[axis] > 0.0f ? 0 : 1];
					const auto w = pp.scale * nl[axis] * nl[axis];
					for (auto i = 0; i < 3; i++)
					{
						ambient[i] += c[i] * w;
					}
				}
				float cube[3];
				sample_cube(cubes, pp.p->slice, nl, cube);
				for (auto i = 0; i < 3; i++)
				{
					out[i] += ambient[i] * cube[i];
				}
			}
		}

		struct evaluator::impl
		{
			struct sun_volume
			{
				std::vector<plane> planes;
				probe global{};
				std::vector<probe> probes; // overrides first, then the rest (BO3's two loops)
				std::size_t override_count = 0;
				const GfxProbeVolumeTextures* volume_images = nullptr;
				const GfxReflectionProbeArray* probe_array = nullptr;
				bool loaded = false;
				volume_texture x, y, z;
				cube_mip cubes;
			};

			std::vector<sun_volume> volumes;
		};

		evaluator::evaluator(const GfxWorld* world, const unsigned int state)
			: impl_(std::make_unique<impl>())
		{
			if (state >= 4)
			{
				throw std::runtime_error("lighting state out of range");
			}

			for (auto v = 0u; v < world->sunVolumeCount; v++)
			{
				const auto& src = world->sunVolumes[v];
				auto& volume = impl_->volumes.emplace_back();

				for (auto i = 0; i < src.planeCount; i++)
				{
					const auto* p = world->sunVolumePlanes[src.planeStart + i];
					volume.planes.push_back({ { p[0], p[1], p[2] }, p[3] });
				}

				const auto& array = src.reflectionProbes[state];
				volume.probe_array = &array;
				volume.volume_images = &src.probeVolumes[state];
				if (!array.configs)
				{
					throw std::runtime_error(utils::string::va("sun volume %u state %u has no reflection probes", v, state));
				}

				volume.global = make_probe(array.configs[0], 0);

				std::vector<probe> overrides, others;
				for (auto i = 1u; i <= array.localReflectionProbeCount; i++)
				{
					const auto& src_probe = array.configs[i];
					if (src_probe.exploderDisabled[0])
					{
						continue;
					}

					auto p = make_probe(src_probe, i);

					// the packer walks non-negative blends, then negative ones
					auto count = 0u;
					for (auto pass = 0u; pass < 2; pass++)
					{
						for (auto j = 0u; j < src_probe.numBlends; j++)
						{
							const auto& blend = array.blends[src_probe.firstBlend + j];
							if (blend.negative != (pass > 0))
							{
								continue;
							}
							for (auto group = 0u;; group++)
							{
								// two flag bits per entry in one 32-bit word (cb10 word 24)
								if (count >= 16)
								{
									throw std::runtime_error(utils::string::va("sun volume %u probe %u has more than 16 blend entries", v, i));
								}
								blend_entry entry{};
								const auto more = blend_planes(src_probe.config, blend, group, entry.planes);
								if (more)
								{
									p.flags += 1u << (2 * count);
								}
								else if (pass > 0 && j > 0)
								{
									p.flags += 1u << (2 * count + 1);
								}
								p.entries.push_back(entry);
								count++;
								if (!more)
								{
									break;
								}
							}
						}
					}
					(p.is_override ? overrides : others).push_back(std::move(p));
				}

				volume.override_count = overrides.size();
				volume.probes = std::move(overrides);
				for (auto& p : others)
				{
					volume.probes.push_back(std::move(p));
				}
			}
		}

		evaluator::~evaluator() = default;

		unsigned int evaluator::volume_count() const
		{
			return static_cast<unsigned int>(impl_->volumes.size());
		}

		unsigned int evaluator::volume_at(const float p[3]) const
		{
			auto closest = 131072.0f;
			auto closest_volume = 0u;
			for (auto v = 0u; v < impl_->volumes.size(); v++)
			{
				auto inside = true;
				for (const auto& pl : impl_->volumes[v].planes)
				{
					const auto dist = dot3(pl.n, p) + pl.w;
					if (closest > std::fabs(dist))
					{
						closest = std::fabs(dist);
						closest_volume = v;
					}
					if (dist > 0.0f)
					{
						inside = false;
					}
				}
				if (inside)
				{
					return v;
				}
			}
			return closest_volume;
		}

		void evaluator::load(const unsigned int volume)
		{
			auto& v = impl_->volumes.at(volume);
			if (v.loaded)
			{
				return;
			}

			const auto* textures = v.volume_images;
			const auto volume_data = read_stream_buffer(textures->pixelData, utils::string::va("sun volume %u probe volumes", volume));
			std::size_t offset = 0;
			const GfxImage* images[3] = { textures->X, textures->Y, textures->Z };
			volume_texture* outputs[3] = { &v.x, &v.y, &v.z };
			for (auto k = 0; k < 3; k++)
			{
				const auto* image = images[k];
				if (!image || offset + image->totalSize > volume_data.size())
				{
					throw std::runtime_error(utils::string::va("sun volume %u probe volume %d does not fit its buffer", volume, k));
				}
				decode_volume(image, volume_data.data() + offset, *outputs[k]);
				offset += image->totalSize;
			}

			const auto* array = v.probe_array;
			const auto cube_data = read_stream_buffer(array->pixelData, utils::string::va("sun volume %u probe cubes", volume));
			decode_cube_mip(array->probeImages, cube_data.data(), cube_data.size(), cube_lod, v.cubes);
			if (v.cubes.slices <= array->localReflectionProbeCount)
			{
				throw std::runtime_error(utils::string::va("sun volume %u: %u cube slices for %u probes", volume, v.cubes.slices,
					array->localReflectionProbeCount + 1));
			}

			v.loaded = true;
		}

		void evaluator::unload(const unsigned int volume)
		{
			auto& v = impl_->volumes.at(volume);
			v.x = {};
			v.y = {};
			v.z = {};
			v.cubes = {};
			v.loaded = false;
		}

		void evaluator::diffuse(const unsigned int volume, const float p[3], const float (*normals)[3], const std::size_t count,
			float (*out)[3]) const
		{
			const auto& v = impl_->volumes.at(volume);
			if (!v.loaded)
			{
				throw std::runtime_error(utils::string::va("sun volume %u is not loaded", volume));
			}

			// weights depend on the position only; overrides come first and the others only fill
			// what they leave (BO3 skips the others entirely once the overrides reach 1)
			std::vector<probe_at_point> active;
			auto total = 0.0f;
			auto override_total = 0.0f;
			for (auto i = 0u; i < v.probes.size(); i++)
			{
				const auto& pr = v.probes[i];
				const auto is_override = i < v.override_count;
				if (!is_override && override_total >= 1.0f)
				{
					break;
				}
				if (p[0] < pr.bounds_min[0] || p[1] < pr.bounds_min[1] || p[2] < pr.bounds_min[2]
					|| p[0] > pr.bounds_max[0] || p[1] > pr.bounds_max[1] || p[2] > pr.bounds_max[2])
				{
					continue;
				}

				const float d[3] = { p[0] - pr.cull_origin[0], p[1] - pr.cull_origin[1], p[2] - pr.cull_origin[2] };
				const auto y = blend_weight(pr, d);
				const auto weight = (is_override ? saturate(y) : saturate(y - override_total)) * pr.fade;
				if (!(weight > 0.0f))
				{
					continue;
				}
				total += weight;
				if (is_override)
				{
					override_total += weight;
				}
				prepare_probe(pr, v.x, v.y, v.z, d, true, weight * pr.exposure, active.emplace_back());
			}

			const auto norm = 1.0f / std::max(total, 1.0f);
			const auto covered = saturate(total);
			probe_at_point global{};
			if (covered < 0.99f)
			{
				const auto& g = v.global;
				const float d[3] = { p[0] - g.cull_origin[0], p[1] - g.cull_origin[1], p[2] - g.cull_origin[2] };
				prepare_probe(g, v.x, v.y, v.z, d, false, (1.0f - covered) * g.exposure, global);
			}

			for (auto k = 0u; k < count; k++)
			{
				float sum[3] = { 0.0f, 0.0f, 0.0f };
				for (const auto& pp : active)
				{
					add_probe(pp, v.cubes, normals[k], sum);
				}
				for (auto c = 0; c < 3; c++)
				{
					out[k][c] = sum[c] * norm;
				}
				if (covered < 0.99f)
				{
					add_probe(global, v.cubes, normals[k], out[k]);
				}
			}
		}

		float evaluator::texel_size(const unsigned int volume, const float p[3]) const
		{
			const auto& v = impl_->volumes.at(volume);
			auto size = v.global.texel_size;
			for (const auto& pr : v.probes)
			{
				if (p[0] < pr.bounds_min[0] || p[1] < pr.bounds_min[1] || p[2] < pr.bounds_min[2]
					|| p[0] > pr.bounds_max[0] || p[1] > pr.bounds_max[1] || p[2] > pr.bounds_max[2])
				{
					continue;
				}
				const float d[3] = { p[0] - pr.cull_origin[0], p[1] - pr.cull_origin[1], p[2] - pr.cull_origin[2] };
				if (blend_weight(pr, d) > 0.0f && pr.fade > 0.0f)
				{
					size = std::min(size, pr.texel_size);
				}
			}
			return size;
		}

		std::vector<std::array<float, 7>> evaluator::local_boxes() const
		{
			std::vector<std::array<float, 7>> out;
			for (const auto& v : impl_->volumes)
			{
				for (const auto& pr : v.probes)
				{
					if (!std::isfinite(pr.texel_size) || !(pr.fade > 0.0f))
					{
						continue;
					}
					out.push_back({ pr.bounds_min[0], pr.bounds_min[1], pr.bounds_min[2], pr.bounds_max[0], pr.bounds_max[1], pr.bounds_max[2],
						pr.texel_size });
				}
			}
			return out;
		}

		probe_cube decode_probe_cube(const GfxImage* image, const std::uint8_t* data, const std::size_t size, const std::uint32_t slice)
		{
			if (image->format != DXGI_FORMAT_BC6H_UF16 || image->mapType != MAPTYPE_CUBE_ARRAY || image->width != image->height)
			{
				throw std::runtime_error(utils::string::va("probe cubes %s are format %d map type %d %ux%u, not a square BC6H cube array",
					image->name, image->format, image->mapType, image->width, image->height));
			}
			const auto slices = static_cast<std::uint32_t>(image->depth);
			if (slice >= slices)
			{
				throw std::runtime_error(utils::string::va("probe cubes %s have %u slices, slice %u is needed", image->name, slices, slice));
			}

			// mip-major data: for each level, each array element's six faces (face seams measured continuous)
			probe_cube cube;
			cube.size = image->width;
			std::size_t offset = 0;
			for (auto m = 0u; m < image->levelCount; m++)
			{
				const auto w = std::max(1u, static_cast<std::uint32_t>(image->width) >> m);
				const auto face_bytes = bc_bytes(w, w);
				if (offset + static_cast<std::size_t>(face_bytes) * 6 * slices > size)
				{
					throw std::runtime_error(utils::string::va("probe cubes %s: %zu bytes do not reach level %u", image->name, size, m));
				}
				auto& level = cube.levels.emplace_back(static_cast<std::size_t>(6) * w * w * 3, 0.0f);
				for (auto face = 0u; face < 6; face++)
				{
					DirectX::Image src{};
					src.width = w;
					src.height = w;
					src.format = DXGI_FORMAT_BC6H_UF16;
					src.rowPitch = static_cast<std::size_t>(std::max(1u, (w + 3) / 4)) * 16;
					src.slicePitch = face_bytes;
					src.pixels = const_cast<std::uint8_t*>(data) + offset + (static_cast<std::size_t>(slice) * 6 + face) * face_bytes;
					DirectX::ScratchImage decoded;
					if (FAILED(DirectX::Decompress(src, DXGI_FORMAT_R32G32B32A32_FLOAT, decoded)))
					{
						throw std::runtime_error(utils::string::va("could not decode %s slice %u face %u level %u", image->name, slice, face, m));
					}
					const auto* img = decoded.GetImage(0, 0, 0);
					for (auto y = 0u; y < w; y++)
					{
						const auto* row = reinterpret_cast<const float*>(img->pixels + img->rowPitch * y);
						auto* dst = &level[((static_cast<std::size_t>(face) * w + y) * w) * 3];
						for (auto x = 0u; x < w; x++)
						{
							dst[x * 3 + 0] = row[x * 4 + 0];
							dst[x * 3 + 1] = row[x * 4 + 1];
							dst[x * 3 + 2] = row[x * 4 + 2];
						}
					}
				}
				offset += static_cast<std::size_t>(face_bytes) * 6 * slices;
			}
			return cube;
		}

		void probe_cube::sample(const float lod, const float dir[3], float out[3]) const
		{
			const auto top = static_cast<std::uint32_t>(levels.size() - 1);
			const auto l = std::clamp(lod, 0.0f, static_cast<float>(top));
			const auto l0 = static_cast<std::uint32_t>(std::floor(l));
			const auto l1 = std::min(l0 + 1, top);
			const auto blend = l - static_cast<float>(l0);
			const auto level_sample = [&](const std::uint32_t level, float rgb[3])
			{
				const auto w = std::max(1u, size >> level);
				const auto& texels = levels[level];
				bilinear_cube(w, dir, [&](const std::uint32_t face, const std::uint32_t x, const std::uint32_t y)
				{
					return &texels[((static_cast<std::size_t>(face) * w + y) * w + x) * 3];
				}, rgb);
			};
			float a[3];
			level_sample(l0, a);
			if (blend > 0.0f && l1 != l0)
			{
				float b[3];
				level_sample(l1, b);
				for (auto c = 0; c < 3; c++)
				{
					a[c] += (b[c] - a[c]) * blend;
				}
			}
			std::memcpy(out, a, sizeof(a));
		}
	}
}
