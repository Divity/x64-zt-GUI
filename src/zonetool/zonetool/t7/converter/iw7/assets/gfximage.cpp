#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "gfximage.hpp"

#include "zonetool/t7/common/xpak.hpp"
#include "zonetool/t7/functions.hpp"

#include "zonetool/iw7/assets/gfximage.hpp"

#include <DirectXTex.h>

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace gfximage
		{
			namespace
			{
				std::recursive_mutex registry_mutex;
				std::deque<GfxImage> image_copies;
				std::deque<std::string> image_names;
				std::unordered_map<std::string, GfxImage*> images_by_name;
				std::unordered_set<std::string> dumped_images;
				// images whose dump waits for the zone's delayed pixel data (see flush_dumps)
				std::vector<GfxImage*> pending_dumps;

				// Spec and gloss halves waiting for their partner; see pack_spec_gloss.
				std::unordered_map<std::string, image_pixels> pending_spec;
				std::unordered_map<std::string, image_pixels> pending_gloss;

				std::string clean_name(const std::string& name)
				{
					auto new_name = name;
					for (auto& c : new_name)
					{
						if (c == '*')
						{
							c = '_';
						}
					}
					return new_name;
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

				// IW7 keeps colour maps in UNORM formats and flags the gamma instead
				// (IMG_DISK_FLAG_GAMMA_SRGB), which is what every stock colour map does.
				DXGI_FORMAT to_unorm(const DXGI_FORMAT format)
				{
					switch (format)
					{
					case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM;
					case DXGI_FORMAT_BC1_UNORM_SRGB: return DXGI_FORMAT_BC1_UNORM;
					case DXGI_FORMAT_BC2_UNORM_SRGB: return DXGI_FORMAT_BC2_UNORM;
					case DXGI_FORMAT_BC3_UNORM_SRGB: return DXGI_FORMAT_BC3_UNORM;
					case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM;
					case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8X8_UNORM;
					case DXGI_FORMAT_BC7_UNORM_SRGB: return DXGI_FORMAT_BC7_UNORM;
					}
					return format;
				}

				std::size_t level_bytes(const DXGI_FORMAT format, const std::uint32_t width,
					const std::uint32_t height, const std::uint32_t level)
				{
					std::size_t row_pitch = 0, slice_pitch = 0;
					if (FAILED(DirectX::ComputePitch(format, std::max(1u, width >> level),
						std::max(1u, height >> level), row_pitch, slice_pitch)))
					{
						return 0;
					}
					return slice_pitch;
				}

				// bytes of mips [first, first + count) of one face/slice
				std::size_t range_bytes(const DXGI_FORMAT format, const std::uint32_t width,
					const std::uint32_t height, const std::uint32_t depth, const std::uint32_t first,
					const std::uint32_t count)
				{
					std::size_t total = 0;
					for (auto l = first; l < first + count; l++)
					{
						total += level_bytes(format, width, height, l) * std::max(1u, depth >> l);
					}
					return total;
				}

				std::uint32_t face_count(const GfxImage* asset)
				{
					return (asset->mapType == MAPTYPE_CUBE || asset->mapType == MAPTYPE_CUBE_ARRAY) ? 6 : 1;
				}

				// levels of a full chain down to 1x1
				std::uint32_t mip_levels(std::uint32_t width, std::uint32_t height)
				{
					auto levels = 1u;
					while (width > 1 || height > 1)
					{
						width = std::max(1u, width >> 1);
						height = std::max(1u, height >> 1);
						levels++;
					}
					return levels;
				}

				struct streamed_part
				{
					std::uint32_t index;
					std::uint32_t levels;
					std::uint32_t bytes;
					std::uint32_t width;
					std::uint32_t height;
					std::uint64_t key;
				};

				// Every streamed part carries the size of the chain *up to and including* itself,
				// so a part's own payload is the difference to the previous part (Greyhound's
				// GameBlackOps3::LoadXImage does the same for the largest one).
				std::vector<streamed_part> get_streamed_parts(const GfxImage* asset)
				{
					std::vector<streamed_part> parts;
					const auto count = std::min<std::uint32_t>(asset->streamedPartCount, 4);
					std::uint32_t previous_bytes = 0;
					for (auto i = 0u; i < count; i++)
					{
						const auto& part = asset->streamedParts[i];
						const std::uint32_t bytes = part.levelCountAndSize.pixelSize;
						if (part.xpakEntry.key && part.width && bytes > previous_bytes)
						{
							parts.push_back({ i, part.levelCountAndSize.levelCount, bytes - previous_bytes,
								part.width, part.height, part.xpakEntry.key });
						}
						previous_bytes = std::max(previous_bytes, bytes);
					}

					std::sort(parts.begin(), parts.end(), [](const streamed_part& a, const streamed_part& b)
					{
						return a.width > b.width;
					});
					return parts;
				}

				bool assemble_streamed(const GfxImage* asset, image_pixels& out)
				{
					const auto parts = get_streamed_parts(asset);
					if (parts.empty())
					{
						return false;
					}

					const auto format = asset->format;
					const auto faces = face_count(asset);
					const auto& largest = parts.front();
					const auto width = largest.width;
					const auto height = largest.height;
					const auto depth = std::max<std::uint32_t>(1, asset->depth);

					std::vector<std::vector<std::uint8_t>> payloads;
					for (const auto& part : parts)
					{
						auto data = xpak::get_data_for_xpak_key(part.key, part.bytes);
						if (data.size() != part.bytes)
						{
							break;
						}
						payloads.emplace_back(std::move(data));
					}

					if (payloads.empty())
					{
						return false;
					}

					// levels each loaded part contributes, largest first
					std::vector<std::uint32_t> part_levels;
					std::uint32_t total_levels = 0;
					for (auto i = 0u; i < payloads.size(); i++)
					{
						const auto below = i + 1 < parts.size() ? parts[i + 1].levels : 0u;
						const auto own = parts[i].levels > below ? parts[i].levels - below : 0u;
						part_levels.push_back(own);
						total_levels += own;
					}

					// The per-part payloads must be exactly the mips they claim, face by face. The smallest
					// part can carry the chain on to 1x1 past the image's level count; BO3 creates the
					// texture with the image's level count, so those extra levels go.
					std::vector<std::size_t> face_stride(payloads.size());
					auto first_level = 0u;
					auto consistent = true;
					for (auto i = 0u; i < payloads.size() && consistent; i++)
					{
						const auto own = range_bytes(format, width, height, depth, first_level, part_levels[i]);
						consistent = part_levels[i] && payloads[i].size() % faces == 0;
						if (!consistent)
						{
							break;
						}
						face_stride[i] = payloads[i].size() / faces;
						if (face_stride[i] != own)
						{
							auto whole_levels = false;
							const auto smallest = i + 1 == parts.size();
							for (auto extra = 1u; smallest && first_level + part_levels[i] + extra <= mip_levels(width, height); extra++)
							{
								whole_levels |= range_bytes(format, width, height, depth, first_level, part_levels[i] + extra) == face_stride[i];
							}
							consistent = whole_levels;
						}
						first_level += part_levels[i];
					}
					if (consistent && payloads.size() == parts.size() && total_levels != asset->levelCount)
					{
						consistent = false;
					}

					if (!consistent)
					{
						// keep what is certain: the largest part starts at mip 0, so take as many
						// whole levels as its bytes cover
						const auto& data = payloads.front();
						if (faces != 1)
						{
							ZONETOOL_WARNING("image \"%s\": streamed parts do not line up with a %u-face mip chain "
								"of %ux%u (format %d)", asset->name, faces, width, height, format);
							return false;
						}

						auto levels = 0u;
						std::size_t covered = 0;
						while (levels < 16)
						{
							const auto next = level_bytes(format, width, height, levels) * std::max(1u, depth >> levels);
							if (!next || covered + next > data.size())
							{
								break;
							}
							covered += next;
							levels++;
						}

						if (!levels)
						{
							ZONETOOL_WARNING("image \"%s\": largest streamed part holds %zu bytes, less than mip 0 of "
								"%ux%u (format %d)", asset->name, data.size(), width, height, format);
							return false;
						}

						ZONETOOL_WARNING("image \"%s\": streamed parts do not line up with their mip ranges, keeping "
							"%u level(s) of the largest part", asset->name, levels);
						out.data.assign(data.begin(), data.begin() + covered);
						out.levels = levels;
					}
					else
					{
						// each part stores its mips face-major; the output wants every face's whole chain
						// in one run (D3D subresource order), so interleave per face
						out.data.clear();
						out.data.reserve(range_bytes(format, width, height, depth, 0, total_levels) * faces);
						for (auto face = 0u; face < faces; face++)
						{
							auto level = 0u;
							for (auto i = 0u; i < payloads.size(); i++)
							{
								const auto face_bytes = range_bytes(format, width, height, depth, level, part_levels[i]);
								const auto* src = payloads[i].data() + face * face_stride[i];
								out.data.insert(out.data.end(), src, src + face_bytes);
								level += part_levels[i];
							}
						}
						out.levels = total_levels;
					}

					out.width = width;
					out.height = height;
					out.depth = depth;
					out.faces = faces;
					out.format = format;
					return true;
				}

				// Not streamed: the loaded mip lives in memory with its own dimensions
				// (GfxImage::width1/height1, Greyhound's LoadedMipWidth/Height).
				bool assemble_loaded(const GfxImage* asset, image_pixels& out)
				{
					const auto* pixels = asset->pixels;
					const auto size = asset->totalSize;
					if (!pixels || !size)
					{
						return false;
					}

					const auto width = asset->width1 ? asset->width1 : asset->width;
					const auto height = asset->height1 ? asset->height1 : asset->height;
					const auto depth = std::max<std::uint32_t>(1, asset->depth);
					const auto faces = face_count(asset);

					auto levels = 0u;
					std::size_t covered = 0;
					while (covered * faces < size && levels < 16)
					{
						covered += level_bytes(asset->format, width, height, levels) * std::max(1u, depth >> levels);
						levels++;
					}

					if (covered * faces != size)
					{
						ZONETOOL_WARNING("image \"%s\": loaded mip data is %u bytes, not a whole mip chain of %ux%u "
							"(format %d)", asset->name, size, width, height, asset->format);
						return false;
					}

					out.data.assign(pixels, pixels + size);
					out.width = width;
					out.height = height;
					out.depth = depth;
					out.faces = faces;
					out.levels = levels;
					out.format = asset->format;
					return true;
				}

				bool assemble(const GfxImage* asset, image_pixels& out)
				{
					return assemble_streamed(asset, out) || assemble_loaded(asset, out);
				}

				zonetool::iw7::TextureSemantic convert_semantic(const GfxImage* asset)
				{
					switch (asset->semantic)
					{
					case IMG_SEMANTIC_2D: return zonetool::iw7::TS_2D;
					case IMG_SEMANTIC_DIFFUSE_MAP:
					case IMG_SEMANTIC_EFFECT_MAP:
					case IMG_SEMANTIC_CAMO_MAP:
					case IMG_SEMANTIC_EMBLEM:
						return zonetool::iw7::TS_COLOR_MAP;
					case IMG_SEMANTIC_NORMAL_MAP: return zonetool::iw7::TS_NORMAL_MAP;
					case IMG_SEMANTIC_SPECULAR_MASK:
					case IMG_SEMANTIC_SPECULAR_MAP:
					case IMG_SEMANTIC_GLOSS_MAP:
						return zonetool::iw7::TS_SPECULAR_MAP;
					case IMG_SEMANTIC_OCCLUSION_MAP: return zonetool::iw7::TS_SPECULAR_OCCLUSION_MAP;
					case IMG_SEMANTIC_REVEAL_MAP:
					case IMG_SEMANTIC_THICKNESS_MAP:
						return zonetool::iw7::TS_ALPHA_REVEAL_THICKNESS_MAP;
					}
					return zonetool::iw7::TS_FUNCTION;
				}

				zonetool::iw7::MapType convert_map_type(const GfxImage* asset, std::uint32_t* flags)
				{
					switch (asset->mapType)
					{
					case MAPTYPE_2D_ARRAY:
						*flags |= zonetool::iw7::IMG_DISK_FLAG_MAPTYPE_ARRAY;
						return zonetool::iw7::MAPTYPE_ARRAY;
					case MAPTYPE_3D:
						*flags |= zonetool::iw7::IMG_DISK_FLAG_MAPTYPE_3D;
						return zonetool::iw7::MAPTYPE_3D;
					case MAPTYPE_CUBE:
						*flags |= zonetool::iw7::IMG_DISK_FLAG_MAPTYPE_CUBE;
						return zonetool::iw7::MAPTYPE_CUBE;
					case MAPTYPE_CUBE_ARRAY:
						*flags |= zonetool::iw7::IMG_DISK_FLAG_MAPTYPE_CUBE_ARRAY;
						return zonetool::iw7::MAPTYPE_CUBE_ARRAY;
					}
					return zonetool::iw7::MAPTYPE_2D;
				}

				void write_iw7_image(const std::string& name, const GfxImage* source, image_pixels& pixels,
					const zonetool::iw7::TextureSemantic semantic)
				{
					std::uint32_t flags = 0;

					zonetool::iw7::GfxImage image{};
					image.mapType = convert_map_type(source, &flags);
					if (is_srgb(pixels.format))
					{
						flags |= zonetool::iw7::IMG_DISK_FLAG_GAMMA_SRGB;
					}
					if (pixels.levels <= 1)
					{
						flags |= zonetool::iw7::IMG_DISK_FLAG_NOMIPMAPS;
					}

					image.imageFormat = to_unorm(pixels.format);
					image.flags = flags;
					image.semantic = semantic;
					image.category = zonetool::iw7::IMG_CATEGORY_LOAD_FROM_FILE;
					image.dataLen1 = static_cast<unsigned int>(pixels.data.size());
					image.dataLen2 = image.dataLen1;
					image.width = static_cast<unsigned short>(pixels.width);
					image.height = static_cast<unsigned short>(pixels.height);
					image.depth = static_cast<unsigned short>(pixels.depth);
					image.numElements = 1;
					image.levelCount = static_cast<unsigned char>(pixels.levels);
					image.streamed = 0;
					image.pixelData = pixels.data.data();
					image.name = name.data();

					zonetool::iw7::gfx_image::dump(&image);

					std::lock_guard _(registry_mutex);
					dumped_images.insert(name);
				}

				std::string suffix_of(const std::string& name)
				{
					const auto pos = name.find_last_of('_');
					return pos == std::string::npos ? std::string{} : name.substr(pos + 1);
				}

				std::string base_of(const std::string& name)
				{
					const auto pos = name.find_last_of('_');
					return pos == std::string::npos ? std::string{} : name.substr(0, pos);
				}

				bool decompress_rgba(const image_pixels& pixels, DirectX::ScratchImage& out)
				{
					std::size_t row_pitch = 0, slice_pitch = 0;
					if (FAILED(DirectX::ComputePitch(pixels.format, pixels.width, pixels.height, row_pitch, slice_pitch)))
					{
						return false;
					}

					DirectX::Image src{};
					src.width = pixels.width;
					src.height = pixels.height;
					src.format = pixels.format;
					src.rowPitch = row_pitch;
					src.slicePitch = slice_pitch;
					src.pixels = const_cast<std::uint8_t*>(pixels.data.data());

					if (DirectX::IsCompressed(pixels.format))
					{
						return SUCCEEDED(DirectX::Decompress(src, DXGI_FORMAT_R8G8B8A8_UNORM, out));
					}
					if (to_unorm(pixels.format) == DXGI_FORMAT_R8G8B8A8_UNORM)
					{
						src.format = DXGI_FORMAT_R8G8B8A8_UNORM;
						return SUCCEEDED(out.InitializeFromImage(src));
					}
					return SUCCEEDED(DirectX::Convert(src, DXGI_FORMAT_R8G8B8A8_UNORM, DirectX::TEX_FILTER_DEFAULT,
						DirectX::TEX_THRESHOLD_DEFAULT, out));
				}

				// BO3 splits specular colour (_s) and gloss (_g) into two images; IW7's specular slot
				// is one RGBA texture that reads gloss from alpha. Write RGB = _s, A = _g under the
				// _s name, with a full mip chain, keeping the gamma of the _s source.
				void pack_spec_gloss(const std::string& spec_name, const GfxImage* spec_source,
					const image_pixels& spec, const image_pixels& gloss)
				{
					DirectX::ScratchImage spec_rgba, gloss_rgba;
					if (!decompress_rgba(spec, spec_rgba) || !decompress_rgba(gloss, gloss_rgba))
					{
						ZONETOOL_WARNING("image \"%s\": could not decode spec/gloss pair for packing", spec_name.data());
						return;
					}

					auto* s = spec_rgba.GetImage(0, 0, 0);
					const auto* g = gloss_rgba.GetImage(0, 0, 0);
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

					DirectX::ScratchImage mips;
					if (FAILED(DirectX::GenerateMipMaps(*s, DirectX::TEX_FILTER_DEFAULT, 0, mips)))
					{
						ZONETOOL_WARNING("image \"%s\": mip generation failed for the packed spec/gloss", spec_name.data());
						return;
					}

					image_pixels packed{};
					packed.data.assign(mips.GetPixels(), mips.GetPixels() + mips.GetPixelsSize());
					packed.width = static_cast<std::uint32_t>(s->width);
					packed.height = static_cast<std::uint32_t>(s->height);
					packed.levels = static_cast<std::uint32_t>(mips.GetMetadata().mipLevels);
					packed.format = is_srgb(spec.format) ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;

					write_iw7_image(spec_name, spec_source, packed, zonetool::iw7::TS_SPECULAR_MAP);
					ZONETOOL_INFO("packed gloss into \"%s\" alpha (%ux%u, %u levels)", spec_name.data(),
						packed.width, packed.height, packed.levels);
				}
			}

			GfxImage* register_image(GfxImage* image)
			{
				if (!image)
				{
					return image;
				}

				std::lock_guard _(registry_mutex);

				const auto* raw_name = image->name;
				if (!raw_name || !*raw_name)
				{
					return image;
				}

				// a reference to an image loaded earlier resolves to that copy
				if (*raw_name == ',')
				{
					const auto found = images_by_name.find(raw_name + 1);
					return found != images_by_name.end() ? found->second : image;
				}

				const std::string name = raw_name;
				const auto existing = images_by_name.find(name);
				if (existing != images_by_name.end())
				{
					return existing->second;
				}

				auto& copy = image_copies.emplace_back(*image);
				copy.name = image_names.emplace_back(name).data();
				// Load_GfxImage (0x1401B1C90): streaming images only reserve block 8 (filled at
				// runtime); delayLoadPixels go to block 4, read just before DB_FinishLoadXFile, except
				// render targets, which get no bytes; everything else goes to block 6.
				if (image->streaming || (image->delayLoadPixels && image->category == IMG_CATEGORY_RENDERTARGET))
				{
					copy.pixels = nullptr;
				}
				copy.fallbackPixels = nullptr;
				images_by_name.emplace(name, &copy);
				return &copy;
			}

			bool get_pixels(const GfxImage* image, image_pixels& out)
			{
				return image && assemble(image, out);
			}

			GfxImage* find_image(const std::string& name)
			{
				std::lock_guard _(registry_mutex);
				const auto found = images_by_name.find(name);
				return found != images_by_name.end() ? found->second : nullptr;
			}

			bool image_dumped(const std::string& name)
			{
				{
					std::lock_guard _(registry_mutex);
					if (dumped_images.contains(name))
					{
						return true;
					}
				}

				const auto base = filesystem::get_dump_path() + "images\\" + clean_name(name);
				return std::filesystem::exists(base + ".iw7Image") || std::filesystem::exists(base + ".dds");
			}

			void clear_registry()
			{
				std::lock_guard _(registry_mutex);
				images_by_name.clear();
				image_copies.clear();
				image_names.clear();
				dumped_images.clear();
				pending_dumps.clear();
				pending_spec.clear();
				pending_gloss.clear();
			}

			void dump(GfxImage* asset)
			{
				if (!asset->name || !*asset->name || *asset->name == ',')
				{
					return;
				}

				// the delayed pixel data is only read after the last asset of the zone was added
				std::lock_guard _(registry_mutex);
				pending_dumps.push_back(asset);
			}

			void write_image(GfxImage* asset)
			{
				const std::string name = asset->name;

				image_pixels pixels{};
				if (!assemble(asset, pixels))
				{
					ZONETOOL_WARNING("image \"%s\" has no pixel data (%u streamed parts, loaded mip %ux%u, %u bytes)",
						asset->name, asset->streamedPartCount, asset->width1, asset->height1, asset->totalSize);
					return;
				}

				const auto semantic = convert_semantic(asset);
				const auto suffix = suffix_of(name);
				const auto base = base_of(name);

				if (suffix == "s" || suffix == "g")
				{
					std::lock_guard _(registry_mutex);

					if (suffix == "s")
					{
						const auto gloss = pending_gloss.find(base);
						if (gloss != pending_gloss.end())
						{
							pack_spec_gloss(name, asset, pixels, gloss->second);
							pending_gloss.erase(gloss);
							return;
						}
						write_iw7_image(name, asset, pixels, semantic);
						pending_spec.emplace(base, std::move(pixels));
						return;
					}

					write_iw7_image(name, asset, pixels, semantic);

					const auto spec = pending_spec.find(base);
					if (spec != pending_spec.end())
					{
						// _s was written unpacked before its gloss arrived; rewrite it packed
						const auto* spec_source = find_image(base + "_s");
						pack_spec_gloss(base + "_s", spec_source ? spec_source : asset, spec->second, pixels);
						pending_spec.erase(spec);
						return;
					}

					pending_gloss.emplace(base, std::move(pixels));
					return;
				}

				write_iw7_image(name, asset, pixels, semantic);
			}

			void flush_dumps()
			{
				std::vector<GfxImage*> images;
				{
					std::lock_guard _(registry_mutex);
					images.swap(pending_dumps);
				}

				for (auto* image : images)
				{
					write_image(image);
				}
			}
		}
	}
}
