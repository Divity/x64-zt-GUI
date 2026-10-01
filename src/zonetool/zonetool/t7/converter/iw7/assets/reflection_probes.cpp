#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "reflection_probes.hpp"

#include "probe_lighting.hpp"
#include "zonetool/t7/converter/iw7/map_common.hpp"
#include "../parallel.hpp"
#include "../gpu.hpp"

#include "zonetool/iw7/assets/gfximage.hpp"

#include <DirectXTex.h>
#include <DirectXPackedVector.h>
#include <utils/string.hpp>

// BO3 (deferred_lighting CS; probe packer 0x141C444A0 / 0x141CBE570):
// * every sun volume has its own probes per lighting state: a global probe (entry 0) and local ones, each
//   with a capture origin, a rotation, blend volumes (boxes or convex face sets that add or subtract
//   influence) and a parallax region (the probe box, or up to six world planes with inward normals);
// * reflection = cube(rotation * (hit - origin), 5 (1 - g) + sqrt(2 t / |hit - origin|) - 0.8448) * weight
//   * exposure on a 256^2 x 7 level cube array (g gloss, t the distance to the parallax hit); the global
//   probe fills the weight the local ones leave, without parallax or rotation.
// IW7 (lit world pixel shader; iw7_ship sub_E20580 / sub_E1FA80 / sub_E21D40):
// * one 128^2 x 8 level BC6H cube array, every cube stored face by face with its levels in order;
// * instances are oriented boxes with a feather, walked in descending priority, each taking
//   min(remaining weight, its own); the parallax box sits on the instance's axes;
// * lod = 5 sqrt(1 - g') with the converted gloss g' = 1 - (1 - 17 g / 20)^2 (world_material) is 5 - 4.25 g,
//   so IW7 level m holds BO3's cube at lod 1.17647 m - 0.8824 + c, c = sqrt(2) - 0.8448 at the capture point;
// * reflections are scaled by clamp(lightmap luminance along R / probe SH, 0, 3); the SH is refilled from the
//   light grid at the probe origins whenever the map has one, the shipped values are the same L1 fit of BO3's
//   diffuse probe lighting there.

namespace zonetool::t7
{
	namespace converter::iw7::reflection_probes
	{
		namespace
		{
			constexpr std::uint32_t cube_size = 128;
			constexpr std::uint32_t cube_levels = 8;
			// D3D11 cube arrays hold at most 2048 faces
			constexpr std::size_t max_cubes = 2048 / 6;
			constexpr auto supersample = 4u;
			// the blend distances BO3 clamps to (0x141C44B80)
			constexpr float min_box_blend = 0.01f;
			constexpr float min_face_blend = 0.1f;
			// BO3's distance for a missing parallax plane
			constexpr float unbounded = 65504.0f;
			constexpr float fallback_half_size = 262144.0f;
			constexpr float override_priority = 1.0e6f;
			constexpr float global_priority = -1.0e9f;
			constexpr auto sh_normals = 64u;

			float bo3_lod(const std::uint32_t level)
			{
				return 1.17647f * static_cast<float>(level) - 0.8824f + (std::sqrt(2.0f) - 0.8448f);
			}

			float dot(const float* a, const float* b)
			{
				return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
			}

			using plane = std::array<float, 4>; // dot(n, x) >= w inside

			struct source
			{
				unsigned int volume;
				unsigned int index; // in the volume's probe configs, 0 = its global probe
			};

			// Fibonacci sphere
			std::vector<std::array<float, 3>> sphere_directions(const std::uint32_t count)
			{
				std::vector<std::array<float, 3>> out(count);
				const auto golden = 3.14159265f * (3.0f - std::sqrt(5.0f));
				for (auto i = 0u; i < count; i++)
				{
					const auto z = 1.0f - 2.0f * (static_cast<float>(i) + 0.5f) / static_cast<float>(count);
					const auto r = std::sqrt(std::max(0.0f, 1.0f - z * z));
					const auto phi = golden * static_cast<float>(i);
					out[i] = { r * std::cos(phi), r * std::sin(phi), z };
				}
				return out;
			}

			// no direction leaves the region for good: some plane faces against each of a dense set of directions
			bool bounded(const std::vector<plane>& planes)
			{
				static const auto directions = sphere_directions(256);
				for (const auto& d : directions)
				{
					auto blocked = false;
					for (const auto& p : planes)
					{
						if (dot(p.data(), d.data()) < -1e-4f)
						{
							blocked = true;
							break;
						}
					}
					if (!blocked)
					{
						return false;
					}
				}
				return true;
			}

			// bounding box of the convex region of `planes`, in the frame of `axes` (rows) around `origin`
			bool region_bounds(const std::vector<plane>& planes, const float origin[3], const float axes[3][3], float lo[3], float hi[3])
			{
				if (planes.size() < 4 || !bounded(planes))
				{
					return false;
				}
				for (auto k = 0; k < 3; k++)
				{
					lo[k] = std::numeric_limits<float>::max();
					hi[k] = -std::numeric_limits<float>::max();
				}
				auto corners = 0u;
				for (auto i = 0u; i < planes.size(); i++)
				{
					for (auto j = i + 1; j < planes.size(); j++)
					{
						for (auto k = j + 1; k < planes.size(); k++)
						{
							const auto& a = planes[i];
							const auto& b = planes[j];
							const auto& c = planes[k];
							// Cramer on n_a x = w_a, n_b x = w_b, n_c x = w_c
							const double bc[3] = { static_cast<double>(b[1]) * c[2] - static_cast<double>(b[2]) * c[1],
								static_cast<double>(b[2]) * c[0] - static_cast<double>(b[0]) * c[2],
								static_cast<double>(b[0]) * c[1] - static_cast<double>(b[1]) * c[0] };
							const auto det = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
							if (std::fabs(det) < 1e-9)
							{
								continue;
							}
							const double ca[3] = { static_cast<double>(c[1]) * a[2] - static_cast<double>(c[2]) * a[1],
								static_cast<double>(c[2]) * a[0] - static_cast<double>(c[0]) * a[2],
								static_cast<double>(c[0]) * a[1] - static_cast<double>(c[1]) * a[0] };
							const double ab[3] = { static_cast<double>(a[1]) * b[2] - static_cast<double>(a[2]) * b[1],
								static_cast<double>(a[2]) * b[0] - static_cast<double>(a[0]) * b[2],
								static_cast<double>(a[0]) * b[1] - static_cast<double>(a[1]) * b[0] };
							float x[3];
							for (auto m = 0; m < 3; m++)
							{
								x[m] = static_cast<float>((a[3] * bc[m] + b[3] * ca[m] + c[3] * ab[m]) / det);
							}
							auto inside = true;
							for (const auto& p : planes)
							{
								if (dot(p.data(), x) < p[3] - 1e-2f * std::max(1.0f, std::fabs(p[3])))
								{
									inside = false;
									break;
								}
							}
							if (!inside)
							{
								continue;
							}
							const float rel[3] = { x[0] - origin[0], x[1] - origin[1], x[2] - origin[2] };
							for (auto m = 0; m < 3; m++)
							{
								const auto u = dot(rel, axes[m]);
								lo[m] = std::min(lo[m], u);
								hi[m] = std::max(hi[m], u);
							}
							corners++;
						}
					}
				}
				return corners >= 4;
			}

			// the probe's parallax region as world planes (0x141C444A0): the probe box without planes
			std::vector<plane> parallax_planes(const GfxConfig_Probe& c)
			{
				std::vector<plane> out;
				if (!c.planeCount)
				{
					for (auto m = 0; m < 3; m++)
					{
						const auto* a = c.rotation[m];
						const auto h = (c.size_min[m] + c.size_max[m]) / 2.0f;
						const auto centre = dot(a, c.cullOrigin);
						out.push_back({ a[0], a[1], a[2], centre - h });
						out.push_back({ -a[0], -a[1], -a[2], -centre - h });
					}
					return out;
				}
				for (auto k = 0u; k < std::min(c.planeCount, 6u); k++)
				{
					const auto* p = c.wldClipPlanes[k];
					out.push_back({ std::clamp(p[0], -1.0f, 1.0f), std::clamp(p[1], -1.0f, 1.0f), std::clamp(p[2], -1.0f, 1.0f), p[3] });
				}
				return out;
			}

			struct stats
			{
				unsigned int instances = 0;
				unsigned int negative_blends = 0;
				unsigned int face_blends = 0;
				unsigned int open_face_blends = 0;
				unsigned int oblique_planes = 0;
				unsigned int rotated_blends = 0;
			};

			// the IW7 parallax box of an instance (sub_E20580: [-(half + neg), half + pos] around the centre on
			// the instance axes): each plane goes to the axis nearest its normal, crossing it where the plane
			// crosses that axis through the capture origin (exact for planes along the axes)
			void set_parallax(zonetool::iw7::GfxReflectionProbeInstance& inst, const std::vector<plane>& planes, const float origin[3], stats& st)
			{
				const float* axes[3] = { inst.volumeObb.xAxis, inst.volumeObb.yAxis, inst.volumeObb.zAxis };
				float lo[3] = { -unbounded, -unbounded, -unbounded };
				float hi[3] = { unbounded, unbounded, unbounded };
				const float rel[3] = { origin[0] - inst.volumeObb.center[0], origin[1] - inst.volumeObb.center[1], origin[2] - inst.volumeObb.center[2] };
				for (const auto& p : planes)
				{
					auto m = 0;
					for (auto k = 1; k < 3; k++)
					{
						if (std::fabs(dot(p.data(), axes[k])) > std::fabs(dot(p.data(), axes[m])))
						{
							m = k;
						}
					}
					const auto a = dot(p.data(), axes[m]);
					if (std::fabs(a) < 0.999f)
					{
						st.oblique_planes++;
					}
					const auto t = (p[3] - dot(p.data(), origin)) / a;
					const auto x = dot(rel, axes[m]) + t;
					if (a > 0.0f)
					{
						lo[m] = std::max(lo[m], x);
					}
					else
					{
						hi[m] = std::min(hi[m], x);
					}
				}
				for (auto m = 0; m < 3; m++)
				{
					inst.expandProjectionNeg[m] = -lo[m] - inst.volumeObb.halfSize[m];
					inst.expandProjectionPos[m] = hi[m] - inst.volumeObb.halfSize[m];
				}
			}

			void identity_instance(zonetool::iw7::GfxReflectionProbeInstance& inst, const unsigned int image)
			{
				std::memset(&inst, 0, sizeof(inst));
				inst.probeImageIndex = static_cast<unsigned short>(image);
				inst.probeRotation[3] = 1.0f;
			}

			// one instance per blend that adds influence: its outer box (inner box plus blend distances) on
			// its own axes, the mean blend distance as the feather
			void local_instances(const GfxReflectionProbeArray& array, const GfxReflectionProbe& probe, const unsigned int image,
				std::vector<zonetool::iw7::GfxReflectionProbeInstance>& out, stats& st)
			{
				const auto& c = probe.config;
				const auto parallax = parallax_planes(c);
				for (auto j = 0u; j < probe.numBlends; j++)
				{
					const auto& blend = array.blends[probe.firstBlend + j];
					if (blend.negative)
					{
						// IW7 cannot take weight away; the region is left to the probes around it
						st.negative_blends++;
						continue;
					}

					auto rotated = false;
					for (auto m = 0; m < 3 && !rotated; m++)
					{
						for (auto k = 0; k < 3; k++)
						{
							rotated |= std::fabs(blend.rotation[m][k] - c.rotation[m][k]) > 1e-3f;
						}
					}
					st.rotated_blends += rotated;

					float lo[3], hi[3], feather[3];
					auto box = true;
					if (blend.faceCount)
					{
						st.face_blends++;
						std::vector<plane> faces;
						auto distance = 0.0f;
						for (auto f = 0u; f < std::min(blend.faceCount, 24u); f++)
						{
							const auto* face = blend.faces[f];
							float n[3];
							for (auto k = 0; k < 3; k++)
							{
								n[k] = face[0] * blend.rotation[0][k] + face[1] * blend.rotation[1][k] + face[2] * blend.rotation[2][k];
							}
							const auto d = std::max(blend.blends[f], min_face_blend);
							// weight 1 at dot(n, x - blend origin) >= face.w, 0 at face.w - d (0x141C44B80)
							faces.push_back({ n[0], n[1], n[2], face[3] - d + dot(n, blend.origin) });
							distance += d;
						}
						distance /= static_cast<float>(faces.size());
						if (region_bounds(faces, blend.origin, blend.rotation, lo, hi))
						{
							box = false;
							feather[0] = feather[1] = feather[2] = distance;
						}
						else
						{
							st.open_face_blends++;
						}
					}
					if (box)
					{
						for (auto m = 0; m < 3; m++)
						{
							const auto bmin = std::max(blend.blend_mins[m], min_box_blend);
							const auto bmax = std::max(blend.blend_maxs[m], min_box_blend);
							lo[m] = -(blend.size_min[m] + bmin);
							hi[m] = blend.size_max[m] + bmax;
							feather[m] = (bmin + bmax) / 2.0f;
						}
					}

					auto& inst = out.emplace_back();
					identity_instance(inst, image);
					std::memcpy(inst.probePosition, c.origin, sizeof(inst.probePosition));
					float* axes[3] = { inst.volumeObb.xAxis, inst.volumeObb.yAxis, inst.volumeObb.zAxis };
					float centre_local[3];
					for (auto m = 0; m < 3; m++)
					{
						std::memcpy(axes[m], blend.rotation[m], sizeof(float) * 3);
						centre_local[m] = (lo[m] + hi[m]) / 2.0f;
						inst.volumeObb.halfSize[m] = std::max((hi[m] - lo[m]) / 2.0f, min_box_blend);
						// the weight has to reach 1 somewhere
						inst.feather[m] = std::clamp(feather[m], min_box_blend, inst.volumeObb.halfSize[m]);
					}
					for (auto k = 0; k < 3; k++)
					{
						inst.volumeObb.center[k] = blend.origin[k] + centre_local[0] * axes[0][k] + centre_local[1] * axes[1][k]
							+ centre_local[2] * axes[2][k];
					}
					// overrides first, then smaller volumes before larger ones
					inst.priority = (c.isOverride ? override_priority : 0.0f)
						- (inst.volumeObb.halfSize[0] + inst.volumeObb.halfSize[1] + inst.volumeObb.halfSize[2]);
					set_parallax(inst, parallax, c.origin, st);
					st.instances++;
				}
			}

			// the global probe of a sun volume: the volume's region (or everything), after every local probe,
			// no parallax
			void global_instance(const GfxWorld* asset, const unsigned int volume, const GfxReflectionProbe& probe, const unsigned int image,
				std::vector<zonetool::iw7::GfxReflectionProbeInstance>& out)
			{
				const auto& sv = asset->sunVolumes[volume];
				std::vector<plane> planes;
				for (auto i = 0; i < sv.planeCount; i++)
				{
					// inside: dot(n, p) + w <= 0 (0x141D04BA0)
					const auto* p = asset->sunVolumePlanes[sv.planeStart + i];
					planes.push_back({ -p[0], -p[1], -p[2], p[3] });
				}
				auto& inst = out.emplace_back();
				identity_instance(inst, image);
				std::memcpy(inst.probePosition, probe.config.origin, sizeof(inst.probePosition));
				inst.volumeObb.xAxis[0] = inst.volumeObb.yAxis[1] = inst.volumeObb.zAxis[2] = 1.0f;
				const float zero[3] = {};
				const float world_axes[3][3] = { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };
				float lo[3], hi[3];
				if (!planes.empty() && region_bounds(planes, zero, world_axes, lo, hi))
				{
					for (auto k = 0; k < 3; k++)
					{
						inst.volumeObb.center[k] = (lo[k] + hi[k]) / 2.0f;
						inst.volumeObb.halfSize[k] = std::max((hi[k] - lo[k]) / 2.0f, min_box_blend);
					}
				}
				else
				{
					for (auto k = 0; k < 3; k++)
					{
						inst.volumeObb.halfSize[k] = fallback_half_size;
					}
				}
				// sun volumes are hard partitions, taken in order
				inst.feather[0] = inst.feather[1] = inst.feather[2] = min_box_blend;
				inst.priority = global_priority - 1.0e6f * static_cast<float>(volume);
				inst.volumeFlags = 1;
			}

			// IW7 level m of one cube: BO3's cube at bo3_lod(m) over each texel's footprint, probe rotation and
			// exposure applied
			void resample(const probe_lighting::probe_cube& cube, const float (*rotation)[3], const float exposure, DirectX::ScratchImage& out)
			{
				if (FAILED(out.InitializeCube(DXGI_FORMAT_R32G32B32A32_FLOAT, cube_size, cube_size, 1, cube_levels)))
				{
					throw std::runtime_error("could not allocate a probe cube");
				}
				for (auto m = 0u; m < cube_levels; m++)
				{
					const auto size = cube_size >> m;
					const auto lod = bo3_lod(m);
					for (auto face = 0u; face < 6; face++)
					{
						const auto* img = out.GetImage(m, face, 0);
						for (auto y = 0u; y < size; y++)
						{
							auto* row = reinterpret_cast<float*>(img->pixels + img->rowPitch * y);
							for (auto x = 0u; x < size; x++)
							{
								float sum[3] = { 0.0f, 0.0f, 0.0f };
								for (auto sy = 0u; sy < supersample; sy++)
								{
									for (auto sx = 0u; sx < supersample; sx++)
									{
										const auto s = (static_cast<float>(x) + (static_cast<float>(sx) + 0.5f) / supersample) / size * 2.0f - 1.0f;
										const auto t = (static_cast<float>(y) + (static_cast<float>(sy) + 0.5f) / supersample) / size * 2.0f - 1.0f;
										float d[3];
										probe_lighting::face_direction(face, s, t, d);
										const auto len = std::sqrt(dot(d, d));
										for (auto& v : d)
										{
											v /= len;
										}
										float local[3];
										for (auto k = 0; k < 3; k++)
										{
											local[k] = rotation ? dot(d, rotation[k]) : d[k];
										}
										float rgb[3];
										cube.sample(lod, local, rgb);
										for (auto c = 0; c < 3; c++)
										{
											sum[c] += rgb[c];
										}
									}
								}
								for (auto c = 0; c < 3; c++)
								{
									row[x * 4 + c] = sum[c] * exposure / static_cast<float>(supersample * supersample);
								}
								row[x * 4 + 3] = 1.0f;
							}
						}
					}
				}
			}

			std::vector<std::uint8_t> encode(const DirectX::ScratchImage& cube)
			{
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
					throw std::runtime_error(utils::string::va("probe cube block compression failed (0x%08X)", static_cast<unsigned int>(hr)));
				}
				// DirectXTex orders a cube's images by face, then level: IW7's order
				return { compressed.GetPixels(), compressed.GetPixels() + compressed.GetPixelsSize() };
			}

			// the shader's SH (lit world PS, t33 dwords 0-1 as halves): lum(n) = .282095 h(d0.lo)
			// + .325735 (z h(d1.lo) - x h(d1.hi) - y h(d0.hi)), fitted to BO3's diffuse luminance at p
			zonetool::iw7::GfxReflectionProbeSampleData sample_data(const probe_lighting::evaluator& lighting, const unsigned int volume,
				const float p[3])
			{
				static const auto normals = sphere_directions(sh_normals);
				std::vector<std::array<float, 3>> values(sh_normals);
				lighting.diffuse(volume, p, reinterpret_cast<const float(*)[3]>(normals.data()), sh_normals,
					reinterpret_cast<float(*)[3]>(values.data()));
				double ata[4][4]{}, atb[4]{};
				for (auto i = 0u; i < sh_normals; i++)
				{
					const double row[4] = { 1.0, normals[i][0], normals[i][1], normals[i][2] };
					const auto l = 0.2126 * values[i][0] + 0.7152 * values[i][1] + 0.0722 * values[i][2];
					for (auto r = 0; r < 4; r++)
					{
						for (auto c = 0; c < 4; c++)
						{
							ata[r][c] += row[r] * row[c];
						}
						atb[r] += row[r] * l;
					}
				}
				double m[4][5];
				for (auto r = 0; r < 4; r++)
				{
					std::memcpy(m[r], ata[r], sizeof(ata[r]));
					m[r][4] = atb[r];
				}
				for (auto c = 0; c < 4; c++)
				{
					auto pivot = c;
					for (auto r = c + 1; r < 4; r++)
					{
						if (std::fabs(m[r][c]) > std::fabs(m[pivot][c]))
						{
							pivot = r;
						}
					}
					std::swap(m[c], m[pivot]);
					for (auto r = 0; r < 4; r++)
					{
						if (r != c && m[c][c] != 0.0)
						{
							const auto f = m[r][c] / m[c][c];
							for (auto k = c; k < 5; k++)
							{
								m[r][k] -= f * m[c][k];
							}
						}
					}
				}
				double fit[4];
				for (auto r = 0; r < 4; r++)
				{
					fit[r] = m[r][r] != 0.0 ? m[r][4] / m[r][r] : 0.0;
				}
				const auto half = [](const double v) { return DirectX::PackedVector::XMConvertFloatToHalf(static_cast<float>(v)); };
				const auto sh0 = fit[0] / 0.282095, sh_x = -fit[1] / 0.325735, sh_y = -fit[2] / 0.325735, sh_z = fit[3] / 0.325735;
				zonetool::iw7::GfxReflectionProbeSampleData out{};
				const std::uint32_t dwords[2] = { static_cast<std::uint32_t>(half(sh0)) | (static_cast<std::uint32_t>(half(sh_y)) << 16),
					static_cast<std::uint32_t>(half(sh_z)) | (static_cast<std::uint32_t>(half(sh_x)) << 16) };
				std::memcpy(out.unk, dwords, sizeof(dwords));
				return out;
			}
		}

		void convert(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, utils::memory::allocator& allocator)
		{
			const auto start = std::chrono::steady_clock::now();
			probe_lighting::evaluator lighting(asset, 0);

			// ---- which BO3 probes become IW7 probes ------------------------------------------------------
			std::vector<source> probes;
			std::unordered_map<std::uint64_t, std::vector<source>> copies;
			std::vector<std::uint64_t> copy_order;
			auto disabled = 0u, without_blends = 0u;
			for (auto v = 0u; v < asset->sunVolumeCount; v++)
			{
				const auto& array = asset->sunVolumes[v].reflectionProbes[0];
				for (auto i = 1u; i <= array.localReflectionProbeCount; i++)
				{
					const auto& probe = array.configs[i];
					if (probe.exploderDisabled[0])
					{
						disabled++;
						continue;
					}
					auto adds = false;
					for (auto j = 0u; j < probe.numBlends; j++)
					{
						adds |= !array.blends[probe.firstBlend + j].negative;
					}
					if (!adds)
					{
						without_blends++;
						continue;
					}
					// a physical probe repeats in every sun volume it reaches, captured in each
					const auto key = probe.config.guid ? static_cast<std::uint64_t>(probe.config.guid)
						: (1ull << 32) | (static_cast<std::uint64_t>(v) << 16) | i;
					auto& list = copies[key];
					if (list.empty())
					{
						copy_order.push_back(key);
					}
					list.push_back({ v, i });
				}
			}
			auto moved = 0u;
			for (const auto key : copy_order)
			{
				const auto& list = copies[key];
				// the copy from the sun volume BO3 lights its capture point with
				const auto& first = asset->sunVolumes[list[0].volume].reflectionProbes[0].configs[list[0].index];
				const auto home = lighting.volume_at(first.config.origin);
				auto chosen = list[0];
				for (const auto& s : list)
				{
					if (s.volume == home)
					{
						chosen = s;
						break;
					}
				}
				moved += chosen.volume != list[0].volume;
				probes.push_back(chosen);
			}
			const auto local_count = static_cast<unsigned int>(probes.size());
			for (auto v = 0u; v < asset->sunVolumeCount; v++)
			{
				probes.push_back({ v, 0 });
			}
			if (probes.size() > max_cubes)
			{
				ZONETOOL_FATAL("reflection probes: %zu probes do not fit one cube array (%zu at most)", probes.size(), max_cubes);
			}

			const auto config_of = [&](const source& s) -> const GfxReflectionProbe&
			{
				return asset->sunVolumes[s.volume].reflectionProbes[0].configs[s.index];
			};

			// ---- instances ---------------------------------------------------------------------------------
			stats st{};
			std::vector<zonetool::iw7::GfxReflectionProbeInstance> instances;
			std::vector<std::vector<unsigned int>> probe_instances(probes.size());
			for (auto k = 0u; k < probes.size(); k++)
			{
				const auto before = instances.size();
				const auto& s = probes[k];
				if (k < local_count)
				{
					local_instances(asset->sunVolumes[s.volume].reflectionProbes[0], config_of(s), k, instances, st);
				}
				else
				{
					global_instance(asset, s.volume, config_of(s), k, instances);
				}
				for (auto i = before; i < instances.size(); i++)
				{
					probe_instances[k].push_back(static_cast<unsigned int>(i));
				}
			}

			// ---- cubes: every volume's pixel buffer once, its probes in parallel ------------------------
			std::vector<std::vector<std::uint8_t>> cubes(probes.size());
			for (auto v = 0u; v < asset->sunVolumeCount; v++)
			{
				std::vector<unsigned int> mine;
				for (auto k = 0u; k < probes.size(); k++)
				{
					if (probes[k].volume == v)
					{
						mine.push_back(k);
					}
				}
				if (mine.empty())
				{
					continue;
				}
				const auto& array = asset->sunVolumes[v].reflectionProbes[0];
				const auto data = probe_lighting::read_stream_buffer(array.pixelData, utils::string::va("sun volume %u probe cubes", v));
				parallel_for(static_cast<std::uint32_t>(mine.size()), [&](const std::uint32_t i, std::uint32_t)
				{
					const auto k = mine[i];
					const auto& s = probes[k];
					const auto& probe = array.configs[s.index];
					const auto cube = probe_lighting::decode_probe_cube(array.probeImages, data.data(), data.size(), s.index);
					DirectX::ScratchImage levels;
					// the global probe is sampled without rotation
					resample(cube, s.index ? probe.config.rotation : nullptr, probe.exposure * map::bo3_light_scale, levels);
					cubes[k] = encode(levels);
				});
			}

			std::size_t total = 0;
			for (const auto& c : cubes)
			{
				total += c.size();
			}
			std::vector<std::uint8_t> pixels;
			pixels.reserve(total);
			for (const auto& c : cubes)
			{
				pixels.insert(pixels.end(), c.begin(), c.end());
			}

			const std::string image_name = "*reflection_probe_array";
			{
				zonetool::iw7::GfxImage image{};
				image.imageFormat = DXGI_FORMAT_BC6H_UF16;
				image.flags = 0x28301; // stock arrays (IMG_DISK_FLAG_MAPTYPE_CUBE_ARRAY | 0x301)
				image.mapType = zonetool::iw7::MAPTYPE_CUBE_ARRAY;
				image.semantic = zonetool::iw7::TS_FUNCTION;
				image.category = zonetool::iw7::IMG_CATEGORY_AUTO_GENERATED;
				image.dataLen1 = static_cast<unsigned int>(pixels.size());
				image.dataLen2 = image.dataLen1;
				image.width = cube_size;
				image.height = cube_size;
				image.depth = 1;
				image.numElements = static_cast<unsigned short>(probes.size());
				image.levelCount = cube_levels;
				image.streamed = 0;
				image.pixelData = pixels.data();
				image.name = image_name.data();
				zonetool::iw7::gfx_image::dump(&image);
			}

			// ---- the SH IW7 normalizes reflections with, grouped by the sun volume lighting each origin ------
			std::vector<zonetool::iw7::GfxReflectionProbeSampleData> samples(probes.size());
			std::vector<std::vector<unsigned int>> by_volume(lighting.volume_count());
			for (auto k = 0u; k < probes.size(); k++)
			{
				by_volume[lighting.volume_at(config_of(probes[k]).config.origin)].push_back(k);
			}
			for (auto v = 0u; v < by_volume.size(); v++)
			{
				if (by_volume[v].empty())
				{
					continue;
				}
				lighting.load(v);
				for (const auto k : by_volume[v])
				{
					samples[k] = sample_data(lighting, v, config_of(probes[k]).config.origin);
				}
				lighting.unload(v);
			}

			// ---- the world's probe data ------------------------------------------------------------------
			auto& data = world->draw.reflectionProbeData;
			data.reflectionProbeCount = static_cast<unsigned int>(probes.size());
			data.sharedReflectionProbeCount = 0;
			data.reflectionProbes = allocator.allocate_array<zonetool::iw7::GfxReflectionProbe>(probes.size());
			for (auto k = 0u; k < probes.size(); k++)
			{
				auto& dst = data.reflectionProbes[k];
				std::memcpy(dst.origin, config_of(probes[k]).config.origin, sizeof(dst.origin));
				dst.probeInstanceCount = static_cast<unsigned int>(probe_instances[k].size());
				dst.probeInstances = allocator.allocate_array<unsigned int>(std::max<std::size_t>(1, probe_instances[k].size()));
				std::memcpy(dst.probeInstances, probe_instances[k].data(), sizeof(unsigned int) * probe_instances[k].size());
				dst.probeRelightingIndex = std::numeric_limits<unsigned int>::max();
			}
			data.reflectionProbeArrayImage = allocator.allocate<zonetool::iw7::GfxImage>();
			data.reflectionProbeArrayImage->name = allocator.duplicate_string(image_name);
			data.probeRelightingCount = 0;
			data.probeRelightingData = nullptr;
			data.reflectionProbeGBufferImageCount = 0;
			data.reflectionProbeGBufferImages = nullptr;
			data.reflectionProbeGBufferTextures = nullptr;
			data.reflectionProbeInstanceCount = static_cast<unsigned int>(instances.size());
			data.reflectionProbeInstances = allocator.allocate_array<zonetool::iw7::GfxReflectionProbeInstance>(instances.size());
			std::memcpy(data.reflectionProbeInstances, instances.data(), sizeof(zonetool::iw7::GfxReflectionProbeInstance) * instances.size());
			data.reflectionProbeLightgridSampleData = allocator.allocate_array<zonetool::iw7::GfxReflectionProbeSampleData>(probes.size());
			std::memcpy(data.reflectionProbeLightgridSampleData, samples.data(), sizeof(zonetool::iw7::GfxReflectionProbeSampleData) * samples.size());
			world->dpvs.reflectionProbeVisDataCount = (data.reflectionProbeInstanceCount + 31) >> 5;

			const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
			ZONETOOL_INFO("reflection probes: %u local probes (%u taken from the sun volume holding their capture point, %u disabled, "
				"%u without an adding blend) + %u global, %u instances (%u face blends, %u of them open, %u negative blends left out, "
				"%u blends rotated off their probe, %u oblique parallax planes), %zu byte cube array, %.0f s", local_count, moved,
				disabled, without_blends, asset->sunVolumeCount, static_cast<unsigned int>(instances.size()), st.face_blends,
				st.open_face_blends, st.negative_blends, st.rotated_blends, st.oblique_planes, pixels.size(), seconds);
		}
	}
}
