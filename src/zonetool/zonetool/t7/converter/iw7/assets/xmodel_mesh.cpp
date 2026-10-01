#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "xmodel_mesh.hpp"
#include "xmodel.hpp"

#include "zonetool/iw7/assets/xsurface.hpp"

#include "zonetool/t7/common/xpak.hpp"
#include "zonetool/t7/converter/iw7/model_offset.hpp"

#include "game/shared.hpp"

#include <DirectXPackedVector.h>

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace xmodel_mesh
		{
			namespace
			{
				// Both games stream normals and tangents as R10G10B10A2_UNORM with B = cross(N, T) * s. BO3 decodes a
				// component as (k - 512) / 511 and s = 2w - 1 from the tangent's alpha; IW7 decodes 2k / 1023 - 1 and
				// s = (w > 0) ? -1 : +1, so BO3's alpha 3 becomes IW7's 0 and 0 becomes 3.
				float t7_unit_component(const std::uint32_t k)
				{
					return (static_cast<float>(k) - 512.0f) / 511.0f;
				}

				void t7_unpack_unit(const std::uint32_t packed, float* out)
				{
					out[0] = t7_unit_component(packed & 0x3FF);
					out[1] = t7_unit_component((packed >> 10) & 0x3FF);
					out[2] = t7_unit_component((packed >> 20) & 0x3FF);
				}

				std::uint32_t iw7_unit_component(const float v)
				{
					const auto clamped = std::clamp(v, -1.0f, 1.0f);
					return static_cast<std::uint32_t>(std::floor((clamped + 1.0f) * 0.5f * 1023.0f + 0.5f));
				}

				std::uint32_t iw7_pack_unit(const float* v, const std::uint32_t alpha_bits)
				{
					return iw7_unit_component(v[0]) | (iw7_unit_component(v[1]) << 10)
						| (iw7_unit_component(v[2]) << 20) | ((alpha_bits & 3) << 30);
				}

				void normalize(float* v)
				{
					const auto length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
					if (length > 0.0f)
					{
						v[0] /= length;
						v[1] /= length;
						v[2] /= length;
					}
				}

				// Both games stream texture coordinates as R16G16_FLOAT (u first); BO3's are copied bit for bit
				// unless the surface's baked material covers other texture coordinates than BO3's [0, 1).
				std::uint32_t texcoord(const GfxStreamVertex& vertex, const surface_transform& transform)
				{
					std::uint32_t packed;
					std::memcpy(&packed, &vertex.UVUPosition, sizeof(packed));
					if (transform.origin[0] == 0.0f && transform.origin[1] == 0.0f && transform.span[0] == 1.0f && transform.span[1] == 1.0f)
					{
						return packed;
					}
					const auto u = (DirectX::PackedVector::XMConvertHalfToFloat(vertex.UVUPosition) - transform.origin[0]) / transform.span[0];
					const auto v = (DirectX::PackedVector::XMConvertHalfToFloat(vertex.UVVPosition) - transform.origin[1]) / transform.span[1];
					return DirectX::PackedVector::XMConvertFloatToHalf(u)
						| (static_cast<std::uint32_t>(DirectX::PackedVector::XMConvertFloatToHalf(v)) << 16);
				}

				// The mesh's vertex data: in the zone when it loaded with it, else streamed from its xpak (kept in
				// `streamed`). Null when there is none or the xpak has no data for it.
				const byte* mesh_data(XModelMesh* asset, std::vector<std::uint8_t>& streamed)
				{
					if (!asset->shared || !asset->shared->dataSize)
					{
						return nullptr;
					}
					if ((asset->shared->flags & 0x1) == 0) // loaded
					{
						return asset->shared->data;
					}
					streamed = xpak::get_data_for_xpak_key(asset->xpakEntry.key, asset->shared->dataSize);
					if (streamed.empty())
					{
						return nullptr;
					}
					assert(static_cast<uint32_t>(streamed.size()) == asset->shared->dataSize);
					return streamed.data();
				}

				// meshes seen loading, by name (the struct copied: the loader's block need not outlive the load), and
				// the copies written of each, with the texture transforms, the bones and the name each was written with
				struct written_copy
				{
					surface_transforms transforms;
					bone_map bones;
					std::string name;
				};
				std::map<std::string, XModelMesh> pending;
				std::unordered_map<std::string, std::vector<written_copy>> written;
			}

			namespace
			{
				void sort_heaviest_first(vertex_bones& v)
				{
					for (auto i = 1u; i < v.count; i++)
					{
						for (auto k = i; k > 0 && v.weight[k] > v.weight[k - 1]; k--)
						{
							std::swap(v.weight[k], v.weight[k - 1]);
							std::swap(v.bone[k], v.bone[k - 1]);
						}
					}
				}

				// a vertex's BO3 bones with weight (a vertex whose four weights are 0 is wholly on its first bone)
				vertex_bones bones_of(const GfxStreamWeight& w)
				{
					const std::uint16_t id[4] = { w.WeightID1, w.WeightID2, w.WeightID3, w.WeightID4 };
					const std::uint8_t value[4] = { w.WeightVal1, w.WeightVal2, w.WeightVal3, w.WeightVal4 };
					vertex_bones out{};
					auto sum = 0.0f;
					for (auto k = 0; k < 4; k++)
					{
						if (value[k])
						{
							out.bone[out.count] = id[k];
							out.weight[out.count] = value[k];
							sum += value[k];
							out.count++;
						}
					}
					if (!out.count)
					{
						out.bone[0] = id[0];
						out.weight[0] = 1.0f;
						out.count = 1;
						return out;
					}
					for (auto k = 0u; k < out.count; k++)
					{
						out.weight[k] /= sum;
					}
					sort_heaviest_first(out);
					return out;
				}

				// the vertex's bones as IW7's: remapped, and merged where two became one
				vertex_bones remapped(const vertex_bones& v, const bone_map& bones)
				{
					if (bones.empty())
					{
						return v;
					}
					vertex_bones out{};
					for (auto k = 0u; k < v.count; k++)
					{
						const auto bone = bones(v.bone[k]);
						auto found = false;
						for (auto i = 0u; i < out.count && !found; i++)
						{
							if (out.bone[i] == bone)
							{
								out.weight[i] += v.weight[k];
								found = true;
							}
						}
						if (!found)
						{
							out.bone[out.count] = bone;
							out.weight[out.count] = v.weight[k];
							out.count++;
						}
					}
					sort_heaviest_first(out);
					return out;
				}

				void set_part_bit(int (&bits)[8], const std::uint16_t bone, const char* mesh)
				{
					if (bone >= 256)
					{
						ZONETOOL_FATAL("mesh \"%s\": bone %u is past IW7's part bits (256 bones)", mesh, bone);
					}
					bits[bone >> 5] |= static_cast<int>(0x80000000u >> (bone & 31));
				}

				// A surface's part bits in IW7 are exactly the bones its vertices use (a mesh's are its surfaces' union).
				// BO3's can lack used bones, so they are rebuilt from the bones written (bit 0x80000000 >> b % 32 of word
				// b / 32); a surface with no vertex bones at all keeps BO3's.
				void surface_part_bits(const XSurface* surf, const byte* data, const bone_map& bones, int (&bits)[8], const char* mesh)
				{
					std::memset(bits, 0, sizeof(bits));
					auto any = false;
					if ((surf->flags & XSURFACE_FLAG_SKINNED) != 0 && surf->shared && surf->shared->dataSize && data)
					{
						const auto* weights = reinterpret_cast<const GfxStreamWeight*>(data + surf->shared->skinWeightsOffset
							+ surf->baseVertOffset * sizeof(GfxStreamWeight));
						for (auto j = 0u; j < surf->vertCount; j++)
						{
							const auto v = remapped(bones_of(weights[j]), bones);
							for (auto k = 0u; k < v.count; k++)
							{
								set_part_bit(bits, v.bone[k], mesh);
								any = true;
							}
						}
					}
					for (auto j = 0; j < surf->vertListCount; j++)
					{
						set_part_bit(bits, bones(static_cast<std::uint16_t>(surf->vertList[j].boneOffset)), mesh);
						any = true;
					}
					if (any)
					{
						return;
					}
					for (auto b = 0u; b < std::size(surf->partBits) * 32; b++)
					{
						if (static_cast<std::uint32_t>(surf->partBits[b >> 5]) & (0x80000000u >> (b & 31)))
						{
							set_part_bit(bits, bones(static_cast<std::uint16_t>(b)), mesh);
						}
					}
				}
			}

			// IW7's blend records (XSurface blendVerts) of a skinned surface, grouped as BO3 orders its vertices, by how many of
			// WeightVal2-4 are set: per vertex its primary bone, then each other bone and that bone's weight x 65536 (IW7 gives
			// the primary the rest), then 0. The bones are the
			// vertex's in IW7's bones, heaviest first; a record with more slots than bones left names the primary again at
			// weight 0, so each record stays with its vertex.
			void write_blend_verts(zonetool::iw7::XSurface* surf, utils::memory::allocator& mem, const GfxStreamWeight* weights,
				const bone_map& bones)
			{
				std::vector<std::uint16_t> levels[4];
				for (auto j = 0u; j < surf->vertCount; j++)
				{
					const auto& w = weights[j];
					const auto level = (w.WeightVal2 ? 1u : 0u) + (w.WeightVal3 ? 1u : 0u) + (w.WeightVal4 ? 1u : 0u);
					const auto v = remapped(bones_of(w), bones);
					auto& out = levels[level];
					out.push_back(v.bone[0]);
					for (auto k = 1u; k <= level; k++)
					{
						const auto has = k < v.count;
						out.push_back(has ? v.bone[k] : v.bone[0]);
						out.push_back(has ? static_cast<std::uint16_t>(std::min(65535L, std::lround(v.weight[k] * 65536.0f))) : std::uint16_t{ 0 });
					}
					out.push_back(0);
					surf->blendVertCounts[level]++;
				}

				std::size_t size = 0;
				for (const auto& level : levels)
				{
					size += level.size();
				}
				if (!size)
				{
					return;
				}
				surf->blendVertSize = static_cast<unsigned int>(size * sizeof(std::uint16_t));
				surf->blendVerts = mem.manual_allocate<zonetool::iw7::XBlendInfo>(surf->blendVertSize);
				std::size_t at = 0;
				for (const auto& level : levels)
				{
					std::memcpy(surf->blendVerts + at, level.data(), level.size() * sizeof(std::uint16_t));
					at += level.size();
				}
			}

			zonetool::iw7::XModelSurfs* convert(XModelMesh* asset, utils::memory::allocator& allocator, const surface_transforms& transforms,
				const bone_map& bones)
			{
				if (!transforms.empty() && transforms.size() != asset->numSurfs)
				{
					ZONETOOL_FATAL("mesh \"%s\": %zu surface transforms for %u surfaces", asset->name, transforms.size(), asset->numSurfs);
				}

				const auto new_asset = allocator.allocate<zonetool::iw7::XModelSurfs>();

				REINTERPRET_CAST_SAFE(name);

				const auto& offset = model_offset::get(asset->name ? asset->name : "");

				byte* data = nullptr;
				if (asset->shared && asset->shared->dataSize)
				{
					if ((asset->shared->flags & 0x1) == 0) // loaded
					{
						data = asset->shared->data;
					}
					else // streamed
					{
						auto xpak_data = xpak::get_data_for_xpak_key(asset->xpakEntry.key, asset->shared->dataSize);
						if (xpak_data.empty())
						{
							ZONETOOL_ERROR("error getting xmodelmesh data from xpak for \"%s\"\n", asset->name);
							return nullptr;
						}

						assert(static_cast<uint32_t>(xpak_data.size()) == asset->shared->dataSize);
						data = allocator.allocate_array<byte>(asset->shared->dataSize);
						memcpy(data, xpak_data.data(), asset->shared->dataSize);
					}
				}
				
				new_asset->numsurfs = asset->numSurfs;
				new_asset->surfs = allocator.allocate_array<zonetool::iw7::XSurface>(new_asset->numsurfs);
				for (auto i = 0; i < new_asset->numsurfs; i++)
				{
					auto* surf = &asset->surfs[i];
					auto* new_surf = &new_asset->surfs[i];

					new_surf->flags = 0;
					new_surf->flags |= ((surf->flags & zonetool::t7::XSURFACE_FLAG_SKINNED) != 0) ? zonetool::iw7::SURF_FLAG_SKINNED : 0;
					//new_surf->flags |= zonetool::iw7::SURF_FLAG_VERTCOL_NONE;
					new_surf->flags |= zonetool::iw7::SURF_FLAG_SELF_VISIBILITY;

					new_surf->vertCount = surf->vertCount;
					new_surf->triCount = surf->triCount;
					new_surf->rigidVertListCount = surf->vertListCount;

					new_surf->subdivLevelCount = 0;

					if (surf->shared && surf->shared->dataSize && data)
					{
						auto* verts = reinterpret_cast<GfxStreamVertex*>(data + surf->shared->vertsOffset + surf->baseVertOffset * sizeof(GfxStreamVertex));
						auto* indices = reinterpret_cast<GfxStreamFace*>(data + surf->shared->indicesOffset + surf->baseIndexOffset * 2);
						auto* positions = reinterpret_cast<vec3_t*>(data + surf->shared->posOffset + surf->baseVertOffset * sizeof(vec3_t));
						auto* weights = reinterpret_cast<GfxStreamWeight*>(data + surf->shared->skinWeightsOffset + surf->baseVertOffset * sizeof(GfxStreamWeight));

						new_surf->verts0.packedVerts0 = allocator.allocate_array<zonetool::iw7::GfxPackedVertex>(new_surf->vertCount);

						for (auto j = 0; j < new_surf->vertCount; j++)
						{
							new_surf->verts0.packedVerts0[j].xyz[0] = positions[j][0];
							new_surf->verts0.packedVerts0[j].xyz[1] = positions[j][1];
							new_surf->verts0.packedVerts0[j].xyz[2] = positions[j][2];

							if (offset.valid)
							{
								model_offset::apply_point(offset, new_surf->verts0.packedVerts0[j].xyz);
							}

							// IW7 reads the model vertex colour as R8G8B8A8_UNORM at byte 16 (0x1BC31A0)
							const auto& transform = transforms.empty() ? surface_transform{} : transforms[i];
							new_surf->verts0.packedVerts0[j].color.array[0] = transform.white_rgb ? 255 : verts[j].Color[0];
							new_surf->verts0.packedVerts0[j].color.array[1] = transform.white_rgb ? 255 : verts[j].Color[1];
							new_surf->verts0.packedVerts0[j].color.array[2] = transform.white_rgb ? 255 : verts[j].Color[2];
							new_surf->verts0.packedVerts0[j].color.array[3] = verts[j].Color[3];

							new_surf->verts0.packedVerts0[j].texCoord.packed = texcoord(verts[j], transform);

							float n[3], t[3];
							t7_unpack_unit(static_cast<std::uint32_t>(verts[j].VertexNormal), n);
							t7_unpack_unit(static_cast<std::uint32_t>(verts[j].VertexTangent), t);
							model_offset::apply_dir(offset, n);
							model_offset::apply_dir(offset, t);
							normalize(n);
							normalize(t);

							const auto t7_sign_bits = static_cast<std::uint32_t>(verts[j].VertexTangent) >> 30;
							new_surf->verts0.packedVerts0[j].normal.packed = iw7_pack_unit(n, 3);
							new_surf->verts0.packedVerts0[j].tangent.packed = iw7_pack_unit(t, t7_sign_bits >= 2 ? 0 : 3);

							float default_visibility[4] = { 0.0f, 0.0f, 1.0f, 0.0f };
							new_surf->verts0.packedVerts0[j].selfVisibility.packed = self_visibility::XSurfacePackSelfVisibility(default_visibility);
						}

						new_surf->triIndices = reinterpret_cast<zonetool::iw7::Face*>(indices);

						if ((surf->flags & XSURFACE_FLAG_SKINNED) != 0)
						{
							write_blend_verts(new_surf, allocator, weights, bones);
						}
					}

					static_assert(sizeof(zonetool::iw7::XSurfaceCollisionTree) == sizeof(zonetool::t7::XSurfaceCollisionTree));
					static_assert(sizeof(zonetool::iw7::XSurfaceCollisionNode) == sizeof(zonetool::t7::XSurfaceCollisionNode));
					static_assert(sizeof(zonetool::iw7::XSurfaceCollisionLeaf) == sizeof(zonetool::t7::XSurfaceCollisionLeaf));

					new_surf->rigidVertLists = allocator.allocate_array<zonetool::iw7::XRigidVertList>(surf->vertListCount);
					for (int j = 0; j < surf->vertListCount; j++)
					{
						// BO3's rigid list boneOffset is a plain bone index, as is IW7's boneOffsetIndex
						new_surf->rigidVertLists[j].boneOffsetIndex = bones(static_cast<std::uint16_t>(surf->vertList[j].boneOffset));
						new_surf->rigidVertLists[j].vertCount = surf->vertList[j].vertCount;
						new_surf->rigidVertLists[j].triOffset = surf->vertList[j].triOffset;
						new_surf->rigidVertLists[j].triCount = surf->vertList[j].triCount;

						new_surf->rigidVertLists[j].collisionTree = reinterpret_cast<zonetool::iw7::XSurfaceCollisionTree*>(surf->vertList[j].collisionTree);
					}

					surface_part_bits(surf, data, bones, new_surf->partBits, asset->name);
					for (auto k = 0; k < 8; k++)
					{
						new_asset->partBits[k] |= new_surf->partBits[k];
					}
				}

				return new_asset;
			}

			std::array<int, 8> part_bits(XModelMesh* asset, const bone_map& bones)
			{
				std::array<int, 8> out{};
				std::vector<std::uint8_t> streamed;
				const auto* data = mesh_data(asset, streamed);
				for (auto i = 0; i < asset->numSurfs; i++)
				{
					int bits[8];
					surface_part_bits(&asset->surfs[i], data, bones, bits, asset->name);
					for (auto k = 0; k < 8; k++)
					{
						out[k] |= bits[k];
					}
				}
				return out;
			}

			void for_each_vertex(XModelMesh* asset, const std::function<void(const float* xyz, const vertex_bones& bones)>& fn)
			{
				std::vector<std::uint8_t> streamed;
				const auto* data = mesh_data(asset, streamed);
				if (!data)
				{
					return;
				}
				for (auto i = 0; i < asset->numSurfs; i++)
				{
					const auto* surf = &asset->surfs[i];
					if (!surf->shared || !surf->shared->dataSize)
					{
						continue;
					}
					const auto* positions = reinterpret_cast<const vec3_t*>(data + surf->shared->posOffset + surf->baseVertOffset * sizeof(vec3_t));
					if ((surf->flags & XSURFACE_FLAG_SKINNED) != 0)
					{
						const auto* weights = reinterpret_cast<const GfxStreamWeight*>(data + surf->shared->skinWeightsOffset
							+ surf->baseVertOffset * sizeof(GfxStreamWeight));
						for (auto j = 0; j < surf->vertCount; j++)
						{
							fn(positions[j], bones_of(weights[j]));
						}
						continue;
					}
					// a rigid surface: its vertex lists, in order, each on one bone (a surface without lists on the root)
					auto first = 0;
					for (auto l = 0; l < std::max<int>(1, surf->vertListCount); l++)
					{
						vertex_bones rigid{};
						rigid.bone[0] = surf->vertListCount ? static_cast<std::uint16_t>(surf->vertList[l].boneOffset) : 0;
						rigid.weight[0] = 1.0f;
						rigid.count = 1;
						const auto count = surf->vertListCount ? surf->vertList[l].vertCount : surf->vertCount;
						for (auto j = first; j < first + count && j < surf->vertCount; j++)
						{
							fn(positions[j], rigid);
						}
						first += count;
					}
				}
			}

			bool rigid_with_data(XModelMesh* asset)
			{
				std::vector<std::uint8_t> streamed;
				if (!asset || !mesh_data(asset, streamed))
				{
					return false;
				}
				for (auto i = 0; i < asset->numSurfs; i++)
				{
					const auto* surf = &asset->surfs[i];
					if ((surf->flags & XSURFACE_FLAG_SKINNED) != 0 || !surf->shared || !surf->shared->dataSize)
					{
						return false;
					}
				}
				return true;
			}

			bool append_placed(XModelMesh* asset, const unsigned int surface, const surface_transform& transform, const float* origin,
				const float (*axis)[3], const float scale, const float* center, std::vector<zonetool::iw7::GfxPackedVertex>& verts,
				std::vector<zonetool::iw7::Face>& faces, const float visibility)
			{
				if (surface >= static_cast<unsigned int>(asset->numSurfs))
				{
					return false;
				}
				const auto* surf = &asset->surfs[surface];
				if ((surf->flags & XSURFACE_FLAG_SKINNED) != 0 || !surf->shared || !surf->shared->dataSize
					|| verts.size() + surf->vertCount > 0xFFFF)
				{
					return false;
				}
				std::vector<std::uint8_t> streamed;
				const auto* data = mesh_data(asset, streamed);
				if (!data)
				{
					return false;
				}
				const auto* stream = reinterpret_cast<const GfxStreamVertex*>(data + surf->shared->vertsOffset + surf->baseVertOffset * sizeof(GfxStreamVertex));
				const auto* indices = reinterpret_cast<const GfxStreamFace*>(data + surf->shared->indicesOffset + surf->baseIndexOffset * 2);
				const auto* positions = reinterpret_cast<const vec3_t*>(data + surf->shared->posOffset + surf->baseVertOffset * sizeof(vec3_t));
				const auto& offset = model_offset::get(asset->name ? asset->name : "");

				const auto place_dir = [&](const float* d, float* out)
				{
					for (auto c = 0; c < 3; c++)
					{
						out[c] = d[0] * axis[0][c] + d[1] * axis[1][c] + d[2] * axis[2][c];
					}
					normalize(out);
				};

				const auto first = static_cast<std::uint16_t>(verts.size());
				for (auto j = 0; j < surf->vertCount; j++)
				{
					zonetool::iw7::GfxPackedVertex v{};
					float p[3] = { positions[j][0], positions[j][1], positions[j][2] };
					if (offset.valid)
					{
						model_offset::apply_point(offset, p);
					}
					for (auto c = 0; c < 3; c++)
					{
						v.xyz[c] = origin[c] + scale * (p[0] * axis[0][c] + p[1] * axis[1][c] + p[2] * axis[2][c]) - center[c];
					}

					v.color.array[0] = transform.white_rgb ? 255 : stream[j].Color[0];
					v.color.array[1] = transform.white_rgb ? 255 : stream[j].Color[1];
					v.color.array[2] = transform.white_rgb ? 255 : stream[j].Color[2];
					v.color.array[3] = stream[j].Color[3];
					v.texCoord.packed = texcoord(stream[j], transform);

					float n[3], t[3], wn[3], wt[3];
					t7_unpack_unit(static_cast<std::uint32_t>(stream[j].VertexNormal), n);
					t7_unpack_unit(static_cast<std::uint32_t>(stream[j].VertexTangent), t);
					model_offset::apply_dir(offset, n);
					model_offset::apply_dir(offset, t);
					place_dir(n, wn);
					place_dir(t, wt);
					const auto t7_sign_bits = static_cast<std::uint32_t>(stream[j].VertexTangent) >> 30;
					v.normal.packed = iw7_pack_unit(wn, 3);
					v.tangent.packed = iw7_pack_unit(wt, t7_sign_bits >= 2 ? 0 : 3);

					float placed_visibility[4] = { 0.0f, 0.0f, visibility, 0.0f };
					v.selfVisibility.packed = self_visibility::XSurfacePackSelfVisibility(placed_visibility);
					verts.push_back(v);
				}
				for (auto t = 0; t < surf->triCount; t++)
				{
					const auto& f = indices[t];
					faces.push_back({ static_cast<std::uint16_t>(first + f.Index1), static_cast<std::uint16_t>(first + f.Index2),
						static_cast<std::uint16_t>(first + f.Index3) });
				}
				return true;
			}

			bool append_posed(XModelMesh* asset, const unsigned int surface, const surface_transform& transform,
				const std::vector<std::array<float, 12>>& skin, const float* origin, const float (*axis)[3], const float scale,
				const float* center, std::vector<zonetool::iw7::GfxPackedVertex>& verts, std::vector<zonetool::iw7::Face>& faces)
			{
				if (surface >= static_cast<unsigned int>(asset->numSurfs))
				{
					return false;
				}
				const auto* surf = &asset->surfs[surface];
				if (!surf->shared || !surf->shared->dataSize || verts.size() + surf->vertCount > 0xFFFF)
				{
					return false;
				}
				std::vector<std::uint8_t> streamed;
				const auto* data = mesh_data(asset, streamed);
				if (!data)
				{
					return false;
				}
				const auto* stream = reinterpret_cast<const GfxStreamVertex*>(data + surf->shared->vertsOffset + surf->baseVertOffset * sizeof(GfxStreamVertex));
				const auto* indices = reinterpret_cast<const GfxStreamFace*>(data + surf->shared->indicesOffset + surf->baseIndexOffset * 2);
				const auto* positions = reinterpret_cast<const vec3_t*>(data + surf->shared->posOffset + surf->baseVertOffset * sizeof(vec3_t));
				const auto* weights = reinterpret_cast<const GfxStreamWeight*>(data + surf->shared->skinWeightsOffset
					+ surf->baseVertOffset * sizeof(GfxStreamWeight));
				const auto skinned = (surf->flags & XSURFACE_FLAG_SKINNED) != 0;

				// each vertex's bones: skinned surfaces' weights, rigid ones' vertex lists in order
				std::vector<vertex_bones> vertex_bone(surf->vertCount);
				if (skinned)
				{
					for (auto j = 0; j < surf->vertCount; j++)
					{
						vertex_bone[j] = bones_of(weights[j]);
					}
				}
				else
				{
					auto first = 0;
					for (auto l = 0; l < std::max<int>(1, surf->vertListCount); l++)
					{
						vertex_bones rigid{};
						rigid.bone[0] = surf->vertListCount ? static_cast<std::uint16_t>(surf->vertList[l].boneOffset) : 0;
						rigid.weight[0] = 1.0f;
						rigid.count = 1;
						const auto n = surf->vertListCount ? surf->vertList[l].vertCount : surf->vertCount;
						for (auto j = first; j < first + n && j < surf->vertCount; j++)
						{
							vertex_bone[j] = rigid;
						}
						first += n;
					}
				}

				const auto skin_point = [&](const vertex_bones& b, const float* p, float* out)
				{
					out[0] = out[1] = out[2] = 0.0f;
					for (auto k = 0u; k < b.count; k++)
					{
						const auto& m = skin.at(b.bone[k]);
						for (auto r = 0; r < 3; r++)
						{
							out[r] += b.weight[k] * (m[r * 4 + 0] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3]);
						}
					}
				};
				const auto skin_dir = [&](const vertex_bones& b, const float* d, float* out)
				{
					out[0] = out[1] = out[2] = 0.0f;
					for (auto k = 0u; k < b.count; k++)
					{
						const auto& m = skin.at(b.bone[k]);
						for (auto r = 0; r < 3; r++)
						{
							out[r] += b.weight[k] * (m[r * 4 + 0] * d[0] + m[r * 4 + 1] * d[1] + m[r * 4 + 2] * d[2]);
						}
					}
				};
				const auto place_dir = [&](const float* d, float* out)
				{
					for (auto c = 0; c < 3; c++)
					{
						out[c] = d[0] * axis[0][c] + d[1] * axis[1][c] + d[2] * axis[2][c];
					}
					normalize(out);
				};

				const auto first_vertex = static_cast<std::uint16_t>(verts.size());
				for (auto j = 0; j < surf->vertCount; j++)
				{
					zonetool::iw7::GfxPackedVertex v{};
					float posed[3];
					skin_point(vertex_bone[j], positions[j], posed);
					for (auto c = 0; c < 3; c++)
					{
						v.xyz[c] = origin[c] + scale * (posed[0] * axis[0][c] + posed[1] * axis[1][c] + posed[2] * axis[2][c]) - center[c];
					}
					v.color.array[0] = transform.white_rgb ? 255 : stream[j].Color[0];
					v.color.array[1] = transform.white_rgb ? 255 : stream[j].Color[1];
					v.color.array[2] = transform.white_rgb ? 255 : stream[j].Color[2];
					v.color.array[3] = stream[j].Color[3];
					v.texCoord.packed = texcoord(stream[j], transform);

					float n[3], t[3], sn[3], st[3], wn[3], wt[3];
					t7_unpack_unit(static_cast<std::uint32_t>(stream[j].VertexNormal), n);
					t7_unpack_unit(static_cast<std::uint32_t>(stream[j].VertexTangent), t);
					skin_dir(vertex_bone[j], n, sn);
					skin_dir(vertex_bone[j], t, st);
					place_dir(sn, wn);
					place_dir(st, wt);
					const auto t7_sign_bits = static_cast<std::uint32_t>(stream[j].VertexTangent) >> 30;
					v.normal.packed = iw7_pack_unit(wn, 3);
					v.tangent.packed = iw7_pack_unit(wt, t7_sign_bits >= 2 ? 0 : 3);

					float default_visibility[4] = { 0.0f, 0.0f, 1.0f, 0.0f };
					v.selfVisibility.packed = self_visibility::XSurfacePackSelfVisibility(default_visibility);
					verts.push_back(v);
				}
				for (auto t = 0; t < surf->triCount; t++)
				{
					const auto& f = indices[t];
					faces.push_back({ static_cast<std::uint16_t>(first_vertex + f.Index1), static_cast<std::uint16_t>(first_vertex + f.Index2),
						static_cast<std::uint16_t>(first_vertex + f.Index3) });
				}
				return true;
			}

			void dump(XModelMesh* asset)
			{
				if (asset && asset->name)
				{
					pending.insert_or_assign(asset->name, *asset);
				}
			}

			std::vector<world_material::surface_usage> surface_usage(XModelMesh* asset)
			{
				std::vector<world_material::surface_usage> out(asset->numSurfs);
				std::vector<std::uint8_t> streamed;
				const auto* data = mesh_data(asset, streamed);
				if (!data)
				{
					return out;
				}
				for (auto i = 0; i < asset->numSurfs; i++)
				{
					const auto* surf = &asset->surfs[i];
					if (!surf->shared || !surf->shared->dataSize)
					{
						continue;
					}
					const auto* verts = reinterpret_cast<const GfxStreamVertex*>(data + surf->shared->vertsOffset
						+ surf->baseVertOffset * sizeof(GfxStreamVertex));
					for (auto j = 0; j < surf->vertCount; j++)
					{
						out[i].uv.add(DirectX::PackedVector::XMConvertHalfToFloat(verts[j].UVUPosition),
							DirectX::PackedVector::XMConvertHalfToFloat(verts[j].UVVPosition));
						out[i].colour.add(verts[j].Color);
					}
				}
				return out;
			}

			std::string dump(XModelMesh* asset, const surface_transforms& transforms, const bone_map& bones)
			{
				auto& copies = written[asset->name];
				for (const auto& copy : copies)
				{
					if (copy.transforms == transforms && copy.bones == bones)
					{
						return copy.name;
					}
				}

				const auto base = xmodel::iw7_name(asset->name);
				const std::string name = copies.empty() ? base : utils::string::va("%s_uv%zu", base.data(), copies.size());
				utils::memory::allocator allocator;
				const auto converted_asset = convert(asset, allocator, transforms, bones);
				if (!converted_asset)
				{
					return name;
				}
				converted_asset->name = allocator.duplicate_string(name);
				zonetool::iw7::xsurface::dump(converted_asset);
				copies.push_back({ transforms, bones, name });
				return name;
			}

			void dump_remaining()
			{
				for (auto& [name, mesh] : pending)
				{
					const auto copies = written.find(name);
					if (copies == written.end() || copies->second.empty())
					{
						dump(&mesh, {});
					}
				}
				pending.clear();
				written.clear();
			}
		}
	}
}
