#pragma once

#include "../shader_eval.hpp"

namespace zonetool::t7
{
	namespace converter::iw7::material_texture
	{
		// A BO3 image decoded to 8-bit RGBA (or 32-bit float RGBA for HDR formats), whole mip chain.
		// sRGB images keep their stored bytes; sampling converts them to linear the way the GPU
		// does for an _SRGB view (before filtering).
		struct decoded
		{
			std::string name;
			DXGI_FORMAT source_format = DXGI_FORMAT_UNKNOWN;
			bool srgb = false;
			bool is_float = false;
			std::uint32_t width = 0;
			std::uint32_t height = 0;
			std::vector<std::vector<std::uint8_t>> levels8;
			std::vector<std::vector<float>> levels_f;

			std::uint32_t level_count() const
			{
				return static_cast<std::uint32_t>(this->is_float ? this->levels_f.size() : this->levels8.size());
			}

			void level_size(std::uint32_t level, std::uint32_t& w, std::uint32_t& h) const
			{
				w = std::max(1u, this->width >> level);
				h = std::max(1u, this->height >> level);
			}

			// texel of `level` as linear floats (x, y already wrapped/clamped)
			void fetch(std::uint32_t level, std::uint32_t x, std::uint32_t y, float out[4]) const;
		};

		// Decodes a registered BO3 image (see gfximage::register_image). Results are cached by
		// image name; nullptr when the image has no pixel data or a format that cannot be decoded.
		std::shared_ptr<const decoded> decode(const GfxImage* image);
		void clear_cache();

		enum class address_mode : std::uint8_t
		{
			wrap,
			clamp,
			mirror,
		};

		class texture : public shader_eval::texture_source
		{
		public:
			texture(std::shared_ptr<const decoded> image, address_mode u, address_mode v);

			void dimensions(std::uint32_t mip, std::uint32_t& width, std::uint32_t& height, std::uint32_t& levels) const override;
			void sample(float u, float v, float lod, std::int32_t offset_u, std::int32_t offset_v, float out[4]) const override;
			void load(std::int32_t x, std::int32_t y, std::int32_t mip, float out[4]) const override;

			// replace channel c of everything this texture returns with a constant
			void override_channel(std::uint32_t channel, float value);

			const decoded& image() const { return *this->image_; }

		private:
			void bilinear(std::uint32_t level, float u, float v, std::int32_t offset_u, std::int32_t offset_v, float out[4]) const;
			void apply_overrides(float out[4]) const;

			std::shared_ptr<const decoded> image_;
			address_mode u_;
			address_mode v_;
			std::array<std::optional<float>, 4> overrides_;
		};

		// A texture that returns the same value everywhere (bound in place of a real one to take
		// its contribution out of an evaluation).
		class constant_texture : public shader_eval::texture_source
		{
		public:
			constant_texture(float r, float g, float b, float a, std::uint32_t width = 1, std::uint32_t height = 1);

			void dimensions(std::uint32_t mip, std::uint32_t& width, std::uint32_t& height, std::uint32_t& levels) const override;
			void sample(float u, float v, float lod, std::int32_t offset_u, std::int32_t offset_v, float out[4]) const override;
			void load(std::int32_t x, std::int32_t y, std::int32_t mip, float out[4]) const override;

		private:
			float value_[4];
			std::uint32_t width_;
			std::uint32_t height_;
		};
	}
}
