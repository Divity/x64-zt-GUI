#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "material_texture.hpp"

#include "gfximage.hpp"
#include "../parallel.hpp"

#include <DirectXTex.h>

// Sampling follows the D3D11 functional specification (7.18): texel centres at (i + 0.5) / size,
// bilinear weights on the four surrounding texels, linear interpolation between the two nearest
// mips for a fractional level of detail, and sRGB-to-linear conversion of each texel before it is
// filtered (IEC 61966-2-1: c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ^ 2.4).

namespace zonetool::t7
{
	namespace converter::iw7::material_texture
	{
		namespace
		{
			// one entry per image, decoded once however many bakes ask for it at the same time
			struct cache_entry
			{
				std::once_flag once;
				std::shared_ptr<const decoded> value;
			};

			std::mutex cache_mutex;
			std::unordered_map<std::string, std::shared_ptr<cache_entry>> cache;

			const std::array<float, 256>& srgb_to_linear()
			{
				static const auto table = []
				{
					std::array<float, 256> t{};
					for (auto i = 0; i < 256; i++)
					{
						const auto c = static_cast<float>(i) / 255.0f;
						t[i] = c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
					}
					return t;
				}();
				return table;
			}

			bool is_srgb(const DXGI_FORMAT format)
			{
				switch (format)
				{
				case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
				case DXGI_FORMAT_BC1_UNORM_SRGB:
				case DXGI_FORMAT_BC2_UNORM_SRGB:
				case DXGI_FORMAT_BC3_UNORM_SRGB:
				case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
				case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
				case DXGI_FORMAT_BC7_UNORM_SRGB:
					return true;
				}
				return false;
			}

			bool is_hdr(const DXGI_FORMAT format)
			{
				switch (format)
				{
				case DXGI_FORMAT_BC6H_UF16:
				case DXGI_FORMAT_BC6H_SF16:
				case DXGI_FORMAT_R16G16B16A16_FLOAT:
				case DXGI_FORMAT_R32G32B32A32_FLOAT:
				case DXGI_FORMAT_R11G11B10_FLOAT:
				case DXGI_FORMAT_R16_FLOAT:
				case DXGI_FORMAT_R16G16_FLOAT:
					return true;
				}
				return false;
			}

			std::uint32_t wrap_coord(std::int32_t i, std::uint32_t size, address_mode mode)
			{
				const auto n = static_cast<std::int32_t>(size);
				switch (mode)
				{
				case address_mode::clamp:
					return static_cast<std::uint32_t>(std::clamp(i, 0, n - 1));
				case address_mode::mirror:
				{
					const auto period = 2 * n;
					auto m = i % period;
					if (m < 0)
					{
						m += period;
					}
					return static_cast<std::uint32_t>(m < n ? m : period - 1 - m);
				}
				default:
				{
					auto m = i % n;
					if (m < 0)
					{
						m += n;
					}
					return static_cast<std::uint32_t>(m);
				}
				}
			}
		}

		void decoded::fetch(const std::uint32_t level, const std::uint32_t x, const std::uint32_t y, float out[4]) const
		{
			std::uint32_t w, h;
			this->level_size(level, w, h);
			const auto index = (static_cast<std::size_t>(y) * w + x) * 4;
			if (this->is_float)
			{
				const auto* p = &this->levels_f[level][index];
				out[0] = p[0];
				out[1] = p[1];
				out[2] = p[2];
				out[3] = p[3];
				return;
			}

			const auto* p = &this->levels8[level][index];
			if (this->srgb)
			{
				const auto& lut = srgb_to_linear();
				out[0] = lut[p[0]];
				out[1] = lut[p[1]];
				out[2] = lut[p[2]];
			}
			else
			{
				out[0] = p[0] / 255.0f;
				out[1] = p[1] / 255.0f;
				out[2] = p[2] / 255.0f;
			}
			out[3] = p[3] / 255.0f;
		}

		namespace
		{
			// one level converted to `target` in strips of 64 rows at once (whole blocks: 64 is a multiple of 4);
			// DirectXTex's own parallel path needs OpenMP
			HRESULT convert_level(const DirectX::Image& src, const DXGI_FORMAT target, std::vector<std::uint8_t>& out, const std::size_t texel_size)
			{
				constexpr auto strip_rows = 64u;
				const auto w = static_cast<std::uint32_t>(src.width);
				const auto h = static_cast<std::uint32_t>(src.height);
				const auto strips = (h + strip_rows - 1) / strip_rows;
				out.resize(static_cast<std::size_t>(w) * h * texel_size);

				std::atomic<HRESULT> result{ S_OK };
				parallel_for(strips, [&](const std::uint32_t strip, std::uint32_t)
				{
					const auto y0 = strip * strip_rows;
					const auto rows = std::min(strip_rows, h - y0);
					const auto compressed = DirectX::IsCompressed(src.format);
					// a compressed row pitch covers a row of 4x4 blocks
					const auto source_row = compressed ? y0 / 4 : y0;

					DirectX::Image part = src;
					part.height = rows;
					part.pixels = src.pixels + static_cast<std::size_t>(source_row) * src.rowPitch;
					part.slicePitch = compressed ? src.rowPitch * ((rows + 3) / 4) : src.rowPitch * rows;

					DirectX::ScratchImage converted;
					HRESULT hr;
					if (compressed)
					{
						hr = DirectX::Decompress(part, target, converted);
					}
					else if (src.format == target)
					{
						hr = converted.InitializeFromImage(part);
					}
					else
					{
						hr = DirectX::Convert(part, target, DirectX::TEX_FILTER_POINT | DirectX::TEX_FILTER_SRGB_IN | DirectX::TEX_FILTER_SRGB_OUT,
							DirectX::TEX_THRESHOLD_DEFAULT, converted);
					}
					if (FAILED(hr))
					{
						result = hr;
						return;
					}
					const auto* img = converted.GetImage(0, 0, 0);
					for (auto y = 0u; y < rows; y++)
					{
						std::memcpy(&out[(static_cast<std::size_t>(y0) + y) * w * texel_size], img->pixels + y * img->rowPitch, w * texel_size);
					}
				});
				return result;
			}

			std::shared_ptr<const decoded> decode_image(const GfxImage* image)
			{
				const std::string name = image->name;
				std::shared_ptr<decoded> result;

				gfximage::image_pixels pixels{};
				if (gfximage::get_pixels(image, pixels) && pixels.faces == 1 && pixels.depth == 1)
				{
					auto out = std::make_shared<decoded>();
					out->name = name;
					out->source_format = pixels.format;
					out->srgb = is_srgb(pixels.format);
					out->is_float = is_hdr(pixels.format);
					out->width = pixels.width;
					out->height = pixels.height;

					const auto target = out->is_float ? DXGI_FORMAT_R32G32B32A32_FLOAT
						: (out->srgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM);

					std::size_t offset = 0;
					auto ok = true;
					for (auto level = 0u; level < pixels.levels && ok; level++)
					{
						const auto w = std::max(1u, pixels.width >> level);
						const auto h = std::max(1u, pixels.height >> level);

						std::size_t row_pitch = 0, slice_pitch = 0;
						if (FAILED(DirectX::ComputePitch(pixels.format, w, h, row_pitch, slice_pitch)) ||
							offset + slice_pitch > pixels.data.size())
						{
							ok = false;
							break;
						}

						DirectX::Image src{};
						src.width = w;
						src.height = h;
						src.format = pixels.format;
						src.rowPitch = row_pitch;
						src.slicePitch = slice_pitch;
						src.pixels = pixels.data.data() + offset;
						offset += slice_pitch;

						std::vector<std::uint8_t> level_bytes;
						if (FAILED(convert_level(src, target, level_bytes, out->is_float ? 16 : 4)))
						{
							ok = false;
							break;
						}
						if (out->is_float)
						{
							std::vector<float> level_data(static_cast<std::size_t>(w) * h * 4);
							std::memcpy(level_data.data(), level_bytes.data(), level_bytes.size());
							out->levels_f.emplace_back(std::move(level_data));
						}
						else
						{
							out->levels8.emplace_back(std::move(level_bytes));
						}
					}

					if (ok && out->level_count())
					{
						result = out;
					}
					else
					{
						ZONETOOL_WARNING("material texture \"%s\": could not decode format %d", image->name, pixels.format);
					}
				}
				return result;
			}
		}

		std::shared_ptr<const decoded> decode(const GfxImage* image)
		{
			if (!image || !image->name)
			{
				return nullptr;
			}

			std::shared_ptr<cache_entry> entry;
			{
				std::lock_guard _(cache_mutex);
				auto& slot = cache[image->name];
				if (!slot)
				{
					slot = std::make_shared<cache_entry>();
				}
				entry = slot;
			}
			std::call_once(entry->once, [&]
			{
				entry->value = decode_image(image);
			});
			return entry->value;
		}

		void clear_cache()
		{
			std::lock_guard _(cache_mutex);
			cache.clear();
		}

		texture::texture(std::shared_ptr<const decoded> image, const address_mode u, const address_mode v)
			: image_(std::move(image)), u_(u), v_(v)
		{
		}

		void texture::dimensions(const std::uint32_t mip, std::uint32_t& width, std::uint32_t& height, std::uint32_t& levels) const
		{
			this->image_->level_size(std::min(mip, this->image_->level_count() - 1), width, height);
			levels = this->image_->level_count();
		}

		void texture::override_channel(const std::uint32_t channel, const float value)
		{
			this->overrides_[channel] = value;
		}

		void texture::apply_overrides(float out[4]) const
		{
			for (auto c = 0; c < 4; c++)
			{
				if (this->overrides_[c].has_value())
				{
					out[c] = this->overrides_[c].value();
				}
			}
		}

		void texture::bilinear(const std::uint32_t level, const float u, const float v, const std::int32_t offset_u,
			const std::int32_t offset_v, float out[4]) const
		{
			std::uint32_t w, h;
			this->image_->level_size(level, w, h);

			const auto x = u * static_cast<float>(w) - 0.5f;
			const auto y = v * static_cast<float>(h) - 0.5f;
			const auto x0f = std::floor(x);
			const auto y0f = std::floor(y);
			const auto fx = x - x0f;
			const auto fy = y - y0f;
			const auto x0 = static_cast<std::int32_t>(x0f) + offset_u;
			const auto y0 = static_cast<std::int32_t>(y0f) + offset_v;

			const auto xa = wrap_coord(x0, w, this->u_);
			const auto xb = wrap_coord(x0 + 1, w, this->u_);
			const auto ya = wrap_coord(y0, h, this->v_);
			const auto yb = wrap_coord(y0 + 1, h, this->v_);

			float t00[4], t10[4], t01[4], t11[4];
			this->image_->fetch(level, xa, ya, t00);
			this->image_->fetch(level, xb, ya, t10);
			this->image_->fetch(level, xa, yb, t01);
			this->image_->fetch(level, xb, yb, t11);

			for (auto c = 0; c < 4; c++)
			{
				const auto top = t00[c] + (t10[c] - t00[c]) * fx;
				const auto bottom = t01[c] + (t11[c] - t01[c]) * fx;
				out[c] = top + (bottom - top) * fy;
			}
		}

		void texture::sample(const float u, const float v, const float lod, const std::int32_t offset_u,
			const std::int32_t offset_v, float out[4]) const
		{
			const auto last = static_cast<float>(this->image_->level_count() - 1);
			const auto level = std::clamp(lod, 0.0f, last);
			const auto l0 = static_cast<std::uint32_t>(level);
			const auto frac = level - static_cast<float>(l0);

			this->bilinear(l0, u, v, offset_u, offset_v, out);
			if (frac > 0.0f && l0 + 1 < this->image_->level_count())
			{
				float next[4];
				this->bilinear(l0 + 1, u, v, offset_u, offset_v, next);
				for (auto c = 0; c < 4; c++)
				{
					out[c] += (next[c] - out[c]) * frac;
				}
			}
			this->apply_overrides(out);
		}

		void texture::load(const std::int32_t x, const std::int32_t y, const std::int32_t mip, float out[4]) const
		{
			const auto level = static_cast<std::uint32_t>(std::clamp<std::int32_t>(mip, 0, static_cast<std::int32_t>(this->image_->level_count()) - 1));
			std::uint32_t w, h;
			this->image_->level_size(level, w, h);
			// out-of-range loads return zero (D3D11 3.4.4.1)
			if (x < 0 || y < 0 || static_cast<std::uint32_t>(x) >= w || static_cast<std::uint32_t>(y) >= h)
			{
				out[0] = out[1] = out[2] = out[3] = 0.0f;
				return;
			}
			this->image_->fetch(level, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), out);
			this->apply_overrides(out);
		}

		constant_texture::constant_texture(const float r, const float g, const float b, const float a,
			const std::uint32_t width, const std::uint32_t height)
			: value_{ r, g, b, a }, width_(width), height_(height)
		{
		}

		void constant_texture::dimensions(const std::uint32_t, std::uint32_t& width, std::uint32_t& height, std::uint32_t& levels) const
		{
			width = this->width_;
			height = this->height_;
			levels = 1;
		}

		void constant_texture::sample(float, float, float, std::int32_t, std::int32_t, float out[4]) const
		{
			std::memcpy(out, this->value_, sizeof(this->value_));
		}

		void constant_texture::load(std::int32_t, std::int32_t, std::int32_t, float out[4]) const
		{
			std::memcpy(out, this->value_, sizeof(this->value_));
		}
	}
}
