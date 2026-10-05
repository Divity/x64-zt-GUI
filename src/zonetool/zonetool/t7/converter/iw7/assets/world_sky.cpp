#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "world_sky.hpp"

#include "probe_lighting.hpp"
#include "zonetool/t7/converter/iw7/map_common.hpp"
#include "world_material_bake.hpp"
#include "world_techset_donors.hpp"
#include "../parallel.hpp"
#include "../gpu.hpp"

#include "zonetool/iw7/assets/gfximage.hpp"

#include <DirectXTex.h>
#include <utils/string.hpp>

// BO3 (sky_latlong_hdr shaders; per-scene constant 41 skyRotationTransition written at 0x141CE63F8..0x141CE6427):
// * xy = sin, cos of a = (GfxSkyBox::rotation + dvar 0x42C821AD, default 0) * pi / 180 + atan2(-f.y, -f.x), f the
//   forward vector of the lighting state's sun angles (GfxGlobalLightSettings pitch, yaw; BO3 keeps -f); z = the
//   skybox size, negative flipping z; w = dvar 0x57DE83E6 (default 0), a blend with the panorama flipped upside down.
// * the vertex shader turns the view direction d: t = (d.x cos a - d.y sin a, d.x sin a + d.y cos a, d.z); the pixel
//   shader samples the panorama at u = 0.5 - atan2(t.y, t.x) / 2 pi, v = acos(t.z) / pi; the main pass scales it by colour
//   matrices (cb1[51..56]) and cb0[10].x, which light it as the sun does (sky_scale below).
// IW7 (w_sky, stock data): the cube is sampled along the view direction through the code constants
// SKY_ROTATION_MATRIX_INVERSE_R0..2 (identity unless a script turns the sky) and scaled by r_hdrSkyIntensity x
// r_hdrSkyColorTint from the vision. Stock skies are BC6H cubes, colour map semantic, sampler state 243. BO3 exposes the
// sky with the scene, so its texels take map::bo3_light_scale like every other BO3 radiance, on top of the sun's.

namespace zonetool::t7
{
	namespace converter::iw7::world_sky
	{
		namespace
		{
			constexpr float deg2rad = 0.0174532925f; // BO3's 0x3C8EFA35
			constexpr std::uint32_t supersample = 2;
			constexpr std::uint32_t max_face = 2048;
			constexpr float pi = 3.14159265358979f;

			struct panorama
			{
				std::uint32_t width = 0;
				std::uint32_t height = 0;
				std::vector<float> rgb;

				// bilinear, wrapping in u, clamped in v (texel centres at (i + 0.5) / size)
				void sample(const float u, const float v, float out[3]) const
				{
					const auto x = u * static_cast<float>(width) - 0.5f;
					const auto y = std::clamp(v * static_cast<float>(height) - 0.5f, 0.0f, static_cast<float>(height - 1));
					const auto x0 = static_cast<std::int64_t>(std::floor(x));
					const auto y0 = static_cast<std::int64_t>(std::floor(y));
					const auto fx = x - static_cast<float>(x0);
					const auto fy = y - static_cast<float>(y0);
					const auto wrap = [&](const std::int64_t i)
					{
						const auto w = static_cast<std::int64_t>(width);
						return static_cast<std::size_t>(((i % w) + w) % w);
					};
					const auto row = [&](const std::int64_t j)
					{
						return static_cast<std::size_t>(std::clamp<std::int64_t>(j, 0, static_cast<std::int64_t>(height) - 1));
					};
					const std::size_t xs[2] = { wrap(x0), wrap(x0 + 1) };
					const std::size_t ys[2] = { row(y0), row(y0 + 1) };
					for (auto c = 0; c < 3; c++)
					{
						const auto a = rgb[(ys[0] * width + xs[0]) * 3 + c] * (1.0f - fx) + rgb[(ys[0] * width + xs[1]) * 3 + c] * fx;
						const auto b = rgb[(ys[1] * width + xs[0]) * 3 + c] * (1.0f - fx) + rgb[(ys[1] * width + xs[1]) * 3 + c] * fx;
						out[c] = a * (1.0f - fy) + b * fy;
					}
				}
			};

			panorama decode(const GfxSkyBoxImage& source)
			{
				const auto* image = source.image;
				const auto format = static_cast<DXGI_FORMAT>(image->format);
				if (image->mapType != MAPTYPE_2D && image->mapType != MAPTYPE_NONE)
				{
					throw std::runtime_error(utils::string::va("skybox %s is map type %d, not a 2D panorama", image->name, image->mapType));
				}
				std::size_t row_pitch = 0, slice_bytes = 0;
				if (FAILED(DirectX::ComputePitch(format, image->width, image->height, row_pitch, slice_bytes)))
				{
					throw std::runtime_error(utils::string::va("skybox %s has format %d", image->name, image->format));
				}
				const auto data = probe_lighting::read_stream_buffer(&source.pixelData, utils::string::va("skybox %s", image->name));
				if (data.size() < slice_bytes)
				{
					throw std::runtime_error(utils::string::va("skybox %s: %zu bytes, level 0 needs %zu", image->name, data.size(), slice_bytes));
				}

				DirectX::Image src{};
				src.width = image->width;
				src.height = image->height;
				src.format = format;
				src.rowPitch = row_pitch;
				src.slicePitch = slice_bytes;
				src.pixels = const_cast<std::uint8_t*>(data.data());
				DirectX::ScratchImage decoded;
				const auto hr = DirectX::IsCompressed(format)
					? DirectX::Decompress(src, DXGI_FORMAT_R32G32B32A32_FLOAT, decoded)
					: DirectX::Convert(src, DXGI_FORMAT_R32G32B32A32_FLOAT, DirectX::TEX_FILTER_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, decoded);
				if (FAILED(hr))
				{
					throw std::runtime_error(utils::string::va("could not decode skybox %s (0x%08X)", image->name, static_cast<unsigned int>(hr)));
				}

				panorama out;
				out.width = image->width;
				out.height = image->height;
				out.rgb.resize(static_cast<std::size_t>(out.width) * out.height * 3);
				const auto* img = decoded.GetImage(0, 0, 0);
				parallel_for(out.height, [&](const std::uint32_t y, std::uint32_t)
				{
					const auto* row = reinterpret_cast<const float*>(img->pixels + img->rowPitch * y);
					for (auto x = 0u; x < out.width; x++)
					{
						for (auto c = 0; c < 3; c++)
						{
							out.rgb[(static_cast<std::size_t>(y) * out.width + x) * 3 + c] = row[x * 4 + c];
						}
					}
				});
				return out;
			}

			std::string strip_prefix(std::string name, const std::string& prefix)
			{
				if (name.rfind(prefix, 0) == 0)
				{
					name = name.substr(prefix.size());
				}
				for (auto& c : name)
				{
					if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
					{
						c = '_';
					}
				}
				return name;
			}
		}

		std::optional<sky> convert(const GfxWorld* asset)
		{
			const auto start = std::chrono::steady_clock::now();

			// the sky of the sun volumes a camera can be in (a volume without planes only takes what the others
			// leave), the main volume's first (map::main_sun_volume); a map that swaps skies by script keeps it
			const GfxSkyBox* box = nullptr;
			const GfxGlobalLightSettings* sun = nullptr;
			const auto main = map::main_sun_volume(asset);
			for (auto k = 0u; k < asset->sunVolumeCount; k++)
			{
				const auto v = k == 0 ? main : (k <= main ? k - 1 : k);
				const auto& volume = asset->sunVolumes[v];
				const auto& candidate = volume.skyboxes[0];
				if (!volume.planeCount || !candidate.image || !candidate.image->image)
				{
					continue;
				}
				const auto& settings = volume.sun.settings[map::lighting_state()];
				if (!box)
				{
					box = &candidate;
					sun = &settings;
					continue;
				}
				if (candidate.image != box->image || candidate.rotation != box->rotation || candidate.size != box->size
					|| settings.pitch != sun->pitch || settings.yaw != sun->yaw)
				{
					ZONETOOL_WARNING("sky: sun volume %u has another skybox or sun than the main one (map::main_sun_volume); IW7 draws one sky, the main volume's", v);
				}
			}
			if (!box)
			{
				ZONETOOL_INFO("sky: the map has no skybox");
				return {};
			}

			// BO3's turn about z
			const auto pitch = sun->pitch * deg2rad, yaw = sun->yaw * deg2rad;
			const float forward[3] = { std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), -std::sin(pitch) };
			const auto angle = box->rotation * deg2rad + std::atan2(-forward[1], -forward[0]);
			const auto sin_a = std::sin(angle), cos_a = std::cos(angle);
			const auto flip = box->size < 0.0f;
			// BO3's sky pass (sky_latlong_hdr) brightens the panorama by the lighting state's sun, untinted: as bright as
			// panorama x sun intensity x the luminance of its colour (zm_prototype: clouds 1.47 x 724 x 0.54 against moonlit snow
			// ~250, 2.3x, as BO3 shows it, 2.21x; tinted by the moon's blue colour it came out bluer than BO3's grey sky, k92)
			const auto sun_luminance = sun->intensity * (0.2126f * sun->color[0] + 0.7152f * sun->color[1] + 0.0722f * sun->color[2]);
			const float sky_scale[3] = { sun_luminance, sun_luminance, sun_luminance };

			if (box->model && box->model->meshMaterials)
			{
				const auto& mm = box->model->meshMaterials[0];
				for (auto i = 0; i < mm.numMaterials; i++)
				{
					const auto* m = mm.materials[i];
					if (!m)
					{
						continue;
					}
					ZONETOOL_INFO("sky: model %s material %s techset %s, %u constants", box->model->name, m->name,
						m->techniqueSet ? m->techniqueSet->name : "?", static_cast<unsigned int>(m->constantCount));
					for (auto k = 0; k < m->constantCount && m->constantTable; k++)
					{
						const auto& c = m->constantTable[k];
						ZONETOOL_INFO("sky:   constant %.12s (0x%08X) %g %g %g %g", c.name, c.nameHash, c.literal[0], c.literal[1], c.literal[2], c.literal[3]);
					}
				}
			}
			const auto source = decode(*box->image);
			{
				// the panorama's brightness in BO3 units (upper half, by solid angle; the brightest 1%)
				double sum = 0.0, weight = 0.0, zenith = 0.0, zenith_w = 0.0;
				std::vector<float> lum;
				for (auto y = 0u; y < source.height / 2; y++)
				{
					const auto w = std::sin((y + 0.5) / source.height * pi);
					for (auto x = 0u; x < source.width; x += 4)
					{
						const auto* c = &source.rgb[(static_cast<std::size_t>(y) * source.width + x) * 3];
						const auto l = 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
						sum += l * w;
						weight += w;
						if (y < source.height / 8)
						{
							zenith += l * w;
							zenith_w += w;
						}
						lum.push_back(l);
					}
				}
				std::ranges::sort(lum);
				ZONETOOL_INFO("sky: panorama luminance (BO3 units): upper half mean %g, top 45 degrees %g, 99th percentile %g, max %g",
					sum / std::max(weight, 1e-9), zenith / std::max(zenith_w, 1e-9), lum.empty() ? 0.0f : lum[lum.size() * 99 / 100],
					lum.empty() ? 0.0f : lum.back());
			}
			auto face = 1u;
			while (face * 2 <= std::min(max_face, source.width / 4))
			{
				face *= 2;
			}
			auto levels = 1u;
			while ((face >> levels) > 0)
			{
				levels++;
			}

			DirectX::ScratchImage cube;
			if (FAILED(cube.InitializeCube(DXGI_FORMAT_R32G32B32A32_FLOAT, face, face, 1, levels)))
			{
				throw std::runtime_error("could not allocate the sky cube");
			}
			for (auto f = 0u; f < 6; f++)
			{
				const auto* img = cube.GetImage(0, f, 0);
				parallel_for(face, [&](const std::uint32_t y, std::uint32_t)
				{
					auto* row = reinterpret_cast<float*>(img->pixels + img->rowPitch * y);
					for (auto x = 0u; x < face; x++)
					{
						float sum[3] = { 0.0f, 0.0f, 0.0f };
						for (auto sy = 0u; sy < supersample; sy++)
						{
							for (auto sx = 0u; sx < supersample; sx++)
							{
								const auto s = (static_cast<float>(x) + (static_cast<float>(sx) + 0.5f) / supersample) / face * 2.0f - 1.0f;
								const auto t = (static_cast<float>(y) + (static_cast<float>(sy) + 0.5f) / supersample) / face * 2.0f - 1.0f;
								float d[3];
								probe_lighting::face_direction(f, s, t, d);
								const auto len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
								const float turned[3] = { (d[0] * cos_a - d[1] * sin_a) / len, (d[0] * sin_a + d[1] * cos_a) / len,
									(flip ? -d[2] : d[2]) / len };
								const auto u = 0.5f - std::atan2(turned[1], turned[0]) / (2.0f * pi);
								const auto v = std::acos(std::clamp(turned[2], -1.0f, 1.0f)) / pi;
								float rgb[3];
								source.sample(u, v, rgb);
								for (auto c = 0; c < 3; c++)
								{
									sum[c] += rgb[c];
								}
							}
						}
						for (auto c = 0; c < 3; c++)
						{
							row[x * 4 + c] = sum[c] * sky_scale[c] * map::bo3_light_scale / static_cast<float>(supersample * supersample);
						}
						row[x * 4 + 3] = 1.0f;
					}
				});
				// the smaller levels: 2x2 averages of the level above
				for (auto m = 1u; m < levels; m++)
				{
					const auto* above = cube.GetImage(m - 1, f, 0);
					const auto* level = cube.GetImage(m, f, 0);
					const auto size = std::max(1u, face >> m);
					for (auto y = 0u; y < size; y++)
					{
						auto* row = reinterpret_cast<float*>(level->pixels + level->rowPitch * y);
						for (auto x = 0u; x < size; x++)
						{
							for (auto c = 0; c < 4; c++)
							{
								auto sum = 0.0f;
								for (auto k = 0; k < 4; k++)
								{
									const auto* r = reinterpret_cast<const float*>(above->pixels + above->rowPitch * (y * 2 + (k >> 1)));
									sum += r[(x * 2 + (k & 1)) * 4 + c];
								}
								row[x * 4 + c] = sum * 0.25f;
							}
						}
					}
				}
			}

			DirectX::ScratchImage compressed;
			HRESULT hr;
			if (auto* device = gpu_device())
			{
				std::lock_guard _(gpu_mutex);
				hr = DirectX::Compress(device, cube.GetImages(), cube.GetImageCount(), cube.GetMetadata(), DXGI_FORMAT_BC6H_UF16,
					DirectX::TEX_COMPRESS_DEFAULT, 1.0f, compressed);
			}
			else
			{
				hr = DirectX::Compress(cube.GetImages(), cube.GetImageCount(), cube.GetMetadata(), DXGI_FORMAT_BC6H_UF16,
					DirectX::TEX_COMPRESS_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, compressed);
			}
			if (FAILED(hr))
			{
				throw std::runtime_error(utils::string::va("sky cube block compression failed (0x%08X)", static_cast<unsigned int>(hr)));
			}

			const auto* model_name = box->model && box->model->name ? box->model->name : "skybox";
			sky out;
			out.image = "sky_" + strip_prefix(model_name, "skybox_");
			out.material = "w/" + out.image;
			{
				// DirectXTex orders a cube's images by face, then level: IW7's order (the reflection probe arrays)
				std::vector<std::uint8_t> pixels(compressed.GetPixels(), compressed.GetPixels() + compressed.GetPixelsSize());
				zonetool::iw7::GfxImage image{};
				image.imageFormat = DXGI_FORMAT_BC6H_UF16;
				image.flags = zonetool::iw7::IMG_DISK_FLAG_MAPTYPE_CUBE | zonetool::iw7::IMG_DISK_FLAG_GAMMA_SRGB
					| zonetool::iw7::IMG_DISK_FLAG_CLAMP_U | zonetool::iw7::IMG_DISK_FLAG_CLAMP_V;
				image.mapType = zonetool::iw7::MAPTYPE_CUBE;
				image.semantic = zonetool::iw7::TS_COLOR_MAP;
				image.category = zonetool::iw7::IMG_CATEGORY_LOAD_FROM_FILE;
				image.dataLen1 = static_cast<unsigned int>(pixels.size());
				image.dataLen2 = image.dataLen1;
				image.width = static_cast<unsigned short>(face);
				image.height = static_cast<unsigned short>(face);
				image.depth = 1;
				image.numElements = 1;
				image.levelCount = static_cast<unsigned char>(levels);
				image.streamed = 0;
				image.pixelData = pixels.data();
				image.name = out.image.data();
				zonetool::iw7::gfx_image::dump(&image);
			}

			const auto* donor = world_techset_donors::find("w_sky");
			if (!donor || donor->texture_count != 1)
			{
				throw std::runtime_error("the w_sky donor material is missing or has more than one texture");
			}
			world_material::bake::write_stock_material(out.material, "w_sky", { { donor->textures[0].type_hash, out.image } });
			out.sort_key = donor->sort_key;
			out.sampler_state = donor->textures[0].sampler_state;

			const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
			ZONETOOL_INFO("sky: %s (%ux%u %s) as %s, a %u^2 x %u level BC6H cube turned %.2f degrees (skybox %.2f + sun yaw), material %s, %.0f s",
				model_name, source.width, source.height, box->image->image->name, out.image.data(), face, levels,
				angle / deg2rad, box->rotation, out.material.data(), seconds);
			return out;
		}

		std::vector<std::uint8_t> surfaces(const zonetool::iw7::GfxWorld* world)
		{
			std::vector<std::uint8_t> out(world->surfaceCount, 0);
			for (auto i = 0; i < world->skyCount; i++)
			{
				const auto& sky = world->skies[i];
				for (auto k = 0; k < sky.skySurfCount; k++)
				{
					out[world->dpvs.sortedSurfIndex[sky.skyStartSurfs[k]]] = 1;
				}
			}
			return out;
		}
	}
}
