#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "gfximage.hpp"

#include "zonetool/t7/common/xpak.hpp"
#include "zonetool/t7/functions.hpp"

#include <DirectXTex.h>

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace gfximage
		{
			namespace
			{
				std::string clean_name(const std::string& name)
				{
					auto new_name = name;

					for (auto i = 0u; i < name.size(); i++)
					{
						switch (new_name[i])
						{
						case '*':
							new_name[i] = '_';
							break;
						}
					}

					return new_name;
				}

				DXGI_FORMAT convert_format(DXGI_FORMAT format)
				{
					switch (format)
					{
					case DXGI_FORMAT_BC1_UNORM_SRGB:
						return DXGI_FORMAT_BC1_UNORM;
					case DXGI_FORMAT_BC2_UNORM_SRGB:
						return DXGI_FORMAT_BC2_UNORM;
					case DXGI_FORMAT_BC3_UNORM_SRGB:
						return DXGI_FORMAT_BC3_UNORM;
					case DXGI_FORMAT_BC7_UNORM_SRGB:
						return DXGI_FORMAT_BC7_UNORM;
					}

					return format;
				}

				bool get_largest_streamed_part(GfxImage* asset, unsigned int* part_index, unsigned int* part_size)
				{
					auto found = false;
					auto largest_width = 0u;

					const auto count = std::min<unsigned int>(asset->streamedPartCount, 4);
					for (auto i = 0u; i < count; i++)
					{
						const auto* part = &asset->streamedParts[i];
						if (!part->xpakEntry.key || part->width <= largest_width)
						{
							continue;
						}

						const auto previous_size = i == 0
							? 0u
							: asset->streamedParts[i - 1].levelCountAndSize.pixelSize;

						largest_width = part->width;
						*part_index = i;
						*part_size = part->levelCountAndSize.pixelSize - previous_size;
						found = true;
					}

					return found;
				}

				bool extract_source(GfxImage* asset, std::vector<std::uint8_t>& pixels,
					std::size_t& width, std::size_t& height, DXGI_FORMAT& format)
				{
					if (!asset || !asset->name)
					{
						return false;
					}

					width = asset->width;
					height = asset->height;

					auto part_index = 0u;
					auto part_size = 0u;

					if (get_largest_streamed_part(asset, &part_index, &part_size) && part_size)
					{
						const auto* part = &asset->streamedParts[part_index];
						pixels = xpak::get_data_for_xpak_key(part->xpakEntry.key, part_size);
						if (pixels.empty())
						{
							return false;
						}
						width = part->width;
						height = part->height;
					}
					else if (asset->pixels && asset->totalSize)
					{
						pixels.assign(asset->pixels, asset->pixels + asset->totalSize);
					}
					else if (asset->fallbackPixels && asset->fallbackSize)
					{
						pixels.assign(asset->fallbackPixels, asset->fallbackPixels + asset->fallbackSize);
						width = asset->width1;
						height = asset->height1;
					}
					else
					{
						return false;
					}

					format = convert_format(asset->format);
					return true;
				}

				bool load_rgba(GfxImage* asset, DirectX::ScratchImage& out)
				{
					std::vector<std::uint8_t> pixels;
					std::size_t width = 0, height = 0;
					DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
					if (!extract_source(asset, pixels, width, height, format))
					{
						return false;
					}

					std::size_t row_pitch = 0, slice_pitch = 0;
					if (FAILED(DirectX::ComputePitch(format, width, height, row_pitch, slice_pitch)))
					{
						return false;
					}
					if (pixels.size() < slice_pitch)
					{
						return false;
					}

					DirectX::Image src{};
					src.width = width;
					src.height = height;
					src.format = format;
					src.rowPitch = row_pitch;
					src.slicePitch = slice_pitch;
					src.pixels = pixels.data();

					if (DirectX::IsCompressed(format))
					{
						return SUCCEEDED(DirectX::Decompress(src, DXGI_FORMAT_R8G8B8A8_UNORM, out));
					}
					if (format == DXGI_FORMAT_R8G8B8A8_UNORM)
					{
						return SUCCEEDED(out.InitializeFromImage(src));
					}
					return SUCCEEDED(DirectX::Convert(src, DXGI_FORMAT_R8G8B8A8_UNORM,
						DirectX::TEX_FILTER_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, out));
				}

				bool save_dds(const DirectX::Image* image, const std::string& name)
				{
					DirectX::TexMetadata metadata{};
					metadata.width = image->width;
					metadata.height = image->height;
					metadata.depth = 1;
					metadata.arraySize = 1;
					metadata.mipLevels = 1;
					metadata.format = image->format;
					metadata.dimension = DirectX::TEX_DIMENSION::TEX_DIMENSION_TEXTURE2D;

					const auto parent_path = filesystem::get_dump_path() + "images\\";
					if (!std::filesystem::exists(parent_path))
					{
						std::filesystem::create_directories(parent_path);
					}
					const auto path = parent_path + clean_name(name) + ".dds";
					const std::wstring wpath(path.begin(), path.end());
					return SUCCEEDED(DirectX::SaveToDDSFile(image, 1, metadata,
						DirectX::DDS_FLAGS_NONE, wpath.data()));
				}

				bool try_pack_spec_gloss(GfxImage* asset)
				{
					const std::string name = asset->name;
					if (name.size() < 3 || name.substr(name.size() - 2) != "_s")
					{
						return false;
					}

					const std::string g_name = name.substr(0, name.size() - 2) + "_g";
					if (!zonetool::t7::DB_FindXAssetEntry(ASSET_TYPE_IMAGE, g_name.data(), false))
					{
						return false;
					}

					auto* g_asset = zonetool::t7::DB_FindXAssetHeader(
						ASSET_TYPE_IMAGE, g_name.data(), false, -1).image;

					if (!g_asset)
					{
						return false;
					}

					DirectX::ScratchImage s_scratch, g_scratch;
					if (!load_rgba(asset, s_scratch) || !load_rgba(g_asset, g_scratch))
					{
						ZONETOOL_WARNING("gloss pack: failed to decode \"%s\" or \"%s\"", name.data(), g_name.data());
						return false;
					}

					auto* s = s_scratch.GetImage(0, 0, 0);
					const auto* g = g_scratch.GetImage(0, 0, 0);

					for (std::size_t y = 0; y < s->height; y++)
					{
						auto* srow = s->pixels + y * s->rowPitch;
						const auto gy = (g->height == s->height) ? y : (y * g->height / s->height);
						const auto* grow = g->pixels + gy * g->rowPitch;
						for (std::size_t x = 0; x < s->width; x++)
						{
							const auto gx = (g->width == s->width) ? x : (x * g->width / s->width);
							srow[x * 4 + 3] = grow[gx * 4 + 0];
						}
					}

					if (!save_dds(s, name))
					{
						return false;
					}

					ZONETOOL_INFO("packed gloss \"%s\" into \"%s\" alpha", g_name.data(), name.data());
					return true;
				}
			}

			void dump(GfxImage* asset)
			{
				if (!asset->name)
				{
					return;
				}

				if (try_pack_spec_gloss(asset))
				{
					return;
				}

				std::vector<std::uint8_t> pixels;
				std::size_t width = 0, height = 0;
				DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
				if (!extract_source(asset, pixels, width, height, format))
				{
					ZONETOOL_WARNING("image \"%s\" has no pixel data", asset->name);
					return;
				}

				std::size_t row_pitch = 0;
				std::size_t slice_pitch = 0;
				if (FAILED(DirectX::ComputePitch(format, width, height, row_pitch, slice_pitch)))
				{
					ZONETOOL_WARNING("image \"%s\" has an unsupported format (%i)", asset->name, format);
					return;
				}

				if (pixels.size() < slice_pitch)
				{
					ZONETOOL_WARNING("image \"%s\" only has %u of the %u bytes it needs", asset->name,
						static_cast<unsigned int>(pixels.size()), static_cast<unsigned int>(slice_pitch));
					return;
				}

				DirectX::Image image{};
				image.width = width;
				image.height = height;
				image.format = format;
				image.rowPitch = row_pitch;
				image.slicePitch = slice_pitch;
				image.pixels = pixels.data();

				if (!save_dds(&image, asset->name))
				{
					ZONETOOL_WARNING("failed to dump image \"%s\"", asset->name);
				}
			}
		}
	}
}
