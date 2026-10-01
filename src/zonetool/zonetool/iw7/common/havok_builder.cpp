#include <std_include.hpp>
#include "havok_builder.hpp"

#include "zonetool/utils/utils.hpp"

#include <array>
#include <algorithm>
#include <tuple>
#include <utility>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <map>
#include <functional>

namespace zonetool::iw7
{
	namespace havok
	{
		namespace builder
		{
			namespace
			{

				constexpr auto HK_MAGIC0 = 0x57E0E057u;
				constexpr auto HK_MAGIC1 = 0x10C0C010u;
				constexpr auto HK_FILE_VERSION = 11;
				constexpr auto HK_HEADER_SIZE = 64;
				constexpr auto HK_SECTION_HEADER_SIZE = 64;
				constexpr auto HK_MAX_PREDICATE = 21;

				constexpr auto MAX_SHARED_INDICES_PER_SECTION = 255;
				constexpr auto MAX_VERTEX_INDEX = 255;
				constexpr auto KEY_SECTION_SHIFT = 8;

				constexpr auto WORLD_SHAPE_CONTENTS = 0x28033ED1u;

				constexpr auto DEFAULT_MATERIAL_CRC = 0x1AB7BC33u;
				constexpr auto MAX_PRIMITIVES_PER_SECTION = 127;

				constexpr auto SIZEOF_COMPRESSED_MESH_SHAPE = 160;
				constexpr auto SIZEOF_COMPRESSED_MESH_SHAPE_DATA = 256;
				constexpr auto SIZEOF_MESH_TREE = 160;
				constexpr auto SIZEOF_SECTION = 96;
				constexpr auto SIZEOF_PRIMITIVE = 4;
				constexpr auto SIZEOF_DATA_RUN = 4;
				constexpr auto SIZEOF_NODE_TOP = 5;
				constexpr auto SIZEOF_NODE_SECTION = 4;

				constexpr auto SIG_HK_CLASS = 0x33D42383u;
				constexpr auto SIG_HK_CLASS_MEMBER = 0xB0EFA719u;
				constexpr auto SIG_HK_CLASS_ENUM = 0x8A3609CFu;
				constexpr auto SIG_HK_CLASS_ENUM_ITEM = 0xCE6F8A6Cu;
				constexpr auto SIG_SHAPE_LIST = 0xC909A395u;
				constexpr auto SIG_COMPRESSED_MESH_SHAPE = 0x1318CC9Fu;
				constexpr auto SIG_COMPRESSED_MESH_SHAPE_DATA = 0x54FD8D57u;
				constexpr auto SIG_DYNAMIC_COMPOUND_SHAPE = 0x408A0623u;
				constexpr auto SIG_CONVEX_POLYTOPE_SHAPE = 0x6097F158u;
				constexpr auto SIG_CONVEX_POLYTOPE_CONNECTIVITY = 0xB51806FDu;
				constexpr auto SIG_DYNAMIC_COMPOUND_SHAPE_DATA = 0xF33DC3CCu;
				constexpr auto SIG_PHYSICS_ASSET = 0x0DFB2195u;
				constexpr auto SIG_PHYSICS_SYSTEM_DATA = 0xB26317A4u;
				constexpr auto SIG_XMODEL_LOD = 0x93FC3CBDu;
				constexpr auto SIG_REF_COUNTED_PROPERTIES = 0x7C574867u;
				constexpr auto SIG_SHAPE_MASS_PROPERTIES = 0xE9191728u;

				constexpr auto SIZEOF_DYNAMIC_COMPOUND_SHAPE = 208;
				constexpr auto SIZEOF_DYNAMIC_COMPOUND_SHAPE_DATA = 56;
				constexpr auto SIZEOF_CONVEX_POLYTOPE_SHAPE = 96;
				constexpr auto SIZEOF_CONVEX_POLYTOPE_CONNECTIVITY = 48;
				constexpr auto SIZEOF_SHAPE_INSTANCE = 128;
				constexpr auto SIZEOF_DYNAMIC_TREE_NODE = 32;

				constexpr auto INT24_W_BASE = 0x3F000000u;
				constexpr auto SHAPE_INSTANCE_FLAGS_W = INT24_W_BASE | 0x40u;

				constexpr std::size_t padded_vertex_count(const std::size_t count)
				{
					return (count + 3) / 4 * 4;
				}

				constexpr auto CONVEX_SHAPE_FLAGS = 0x0143u;
				constexpr auto CONVEX_DISPATCH_TYPE = 1;
				constexpr auto COMPOUND_SHAPE_FLAGS = 0x0004u;
				constexpr auto COMPOUND_DISPATCH_TYPE = 2;

				struct byte_buffer
				{
					std::vector<std::uint8_t> data;

					std::size_t size() const { return this->data.size(); }

					void align(const std::size_t alignment, const std::uint8_t fill = 0)
					{
						while (this->data.size() % alignment)
						{
							this->data.push_back(fill);
						}
					}

					void write(const void* src, const std::size_t count)
					{
						const auto* p = static_cast<const std::uint8_t*>(src);
						this->data.insert(this->data.end(), p, p + count);
					}

					template <typename T> void write(const T& value)
					{
						this->write(&value, sizeof(T));
					}

					void fill(const std::size_t count, const std::uint8_t value)
					{
						this->data.insert(this->data.end(), count, value);
					}

					std::size_t reserve(const std::size_t count)
					{
						const auto offset = this->data.size();
						this->fill(count, 0);
						return offset;
					}

					template <typename T> void patch(const std::size_t offset, const T& value)
					{
						std::memcpy(this->data.data() + offset, &value, sizeof(T));
					}
				};

				constexpr auto HK_ARRAY_DONT_DEALLOCATE = 0x80000000u;

				struct hk_array_ref
				{
					std::size_t field_offset = 0;
					std::size_t data_offset = 0;
					int count = 0;
				};

				void write_hk_array_header(byte_buffer& buf, const int count)
				{
					buf.reserve(8);
					buf.write<std::int32_t>(count);
					buf.write<std::uint32_t>(count | HK_ARRAY_DONT_DEALLOCATE);
				}

				constexpr int PACKED_BITS[3] = {11, 11, 10};

				constexpr int SHARED_BITS[3] = {21, 21, 22};

				std::uint64_t pack_shared_vertex(const float* pos, const float* mins,
					const float* maxs)
				{
					std::uint64_t packed = 0;
					auto shift = 0;

					for (auto i = 0; i < 3; i++)
					{
						const auto extent = maxs[i] - mins[i];
						const auto max_value = (1u << SHARED_BITS[i]) - 1u;
						const auto step = extent * std::ldexp(1.0f, -SHARED_BITS[i]);

						auto raw = 0u;
						if (step > 0.0f)
						{
							const auto q = std::lround((pos[i] - mins[i]) / step);
							raw = static_cast<std::uint32_t>(
								std::clamp<long>(q, 0, static_cast<long>(max_value)));
						}

						packed |= static_cast<std::uint64_t>(raw) << shift;
						shift += SHARED_BITS[i];
					}

					return packed;
				}

				void unpack_shared_vertex(const std::uint64_t packed, const float* mins,
					const float* maxs, float(&out)[3])
				{
					auto shift = 0;
					for (auto i = 0; i < 3; i++)
					{
						const auto max_value = (1ull << SHARED_BITS[i]) - 1ull;
						const auto raw = (packed >> shift) & max_value;
						const auto step = (maxs[i] - mins[i]) * std::ldexp(1.0f, -SHARED_BITS[i]);
						out[i] = mins[i] + static_cast<float>(raw) * step;
						shift += SHARED_BITS[i];
					}
				}

				std::uint32_t pack_vertex(const float* pos, const float* codec_parms)
				{
					std::uint32_t packed = 0;
					int shift = 0;

					for (auto i = 0; i < 3; i++)
					{
						const auto scale = codec_parms[3 + i];
						const auto max_value = (1 << PACKED_BITS[i]) - 1;

						auto raw = 0;
						if (scale > 0.0f)
						{
							raw = static_cast<int>(std::lround((pos[i] - codec_parms[i]) / scale));
						}

						raw = std::clamp(raw, 0, max_value);
						packed |= static_cast<std::uint32_t>(raw) << shift;
						shift += PACKED_BITS[i];
					}

					return packed;
				}

				void unpack_vertex(const std::uint32_t packed, const float* codec_parms,
					float(&out)[3])
				{
					auto shift = 0;
					for (auto i = 0; i < 3; i++)
					{
						const auto mask = (1u << PACKED_BITS[i]) - 1u;
						const auto raw = (packed >> shift) & mask;
						out[i] = codec_parms[i]
							+ static_cast<float>(raw) * codec_parms[3 + i];
						shift += PACKED_BITS[i];
					}
				}

				enum class vertex_storage
				{
					shared,
					packed,
				};

				vertex_storage selected_vertex_storage()
				{
					const auto* env = std::getenv("ZT_HAVOK_VERTEX_STORAGE");
					if (!env || !env[0] || !std::strcmp(env, "shared"))
					{
						return vertex_storage::shared;
					}

					if (!std::strcmp(env, "packed"))
					{
						return vertex_storage::packed;
					}

					ZONETOOL_WARNING("havok: ZT_HAVOK_VERTEX_STORAGE=\"%s\" is invalid; "
						"using shared vertices", env);
					return vertex_storage::shared;
				}

				constexpr auto CUSTOM_PRIMITIVE_CONVEX = 0x2u;
				constexpr auto MAX_CONVEX_VERTICES = 255u;
				constexpr auto CONVEX_FLAT_TOLERANCE = 1e-4f;
				constexpr std::size_t SHARED_VERTEX_PAGE_SIZE = 0x10000;

				enum class slot_kind : std::uint8_t
				{
					vertex,
					convex_record,
					convex_start,
				};

				struct build_section
				{
					std::vector<std::array<float, 3>> verts;
					std::vector<slot_kind> slot_kinds;
					std::vector<std::array<std::uint8_t, 4>> primitives;
					std::vector<std::uint16_t> tags;
					std::vector<int> contents;
					std::vector<std::uint32_t> material_crcs;
					std::vector<std::uint64_t> user_data;
					std::vector<bool> quads;
					std::vector<int> custom_index;
					std::vector<std::vector<std::array<float, 3>>> convexes;
					float codec_parms[6] = {};
					float mins[3] = {};
					float maxs[3] = {};

					std::size_t listed_vertex_count() const
					{
						return static_cast<std::size_t>(std::count(slot_kinds.begin(),
							slot_kinds.end(), slot_kind::vertex));
					}

					std::size_t pool_vertex_count() const
					{
						auto count = listed_vertex_count();
						for (const auto& run : convexes)
						{
							count += run.size();
						}
						return count;
					}
				};

				void finalise_section_codec(build_section& section)
				{
					for (auto i = 0; i < 3; i++)
					{
						section.mins[i] = FLT_MAX;
						section.maxs[i] = -FLT_MAX;
					}

					const auto grow = [&section](const std::array<float, 3>& v)
					{
						for (auto i = 0; i < 3; i++)
						{
							if (v[i] < section.mins[i])
							{
								section.mins[i] = v[i];
							}
							if (v[i] > section.maxs[i])
							{
								section.maxs[i] = v[i];
							}
						}
					};

					for (auto s = 0u; s < section.verts.size(); s++)
					{
						if (section.slot_kinds[s] == slot_kind::vertex)
						{
							grow(section.verts[s]);
						}
					}

					for (const auto& run : section.convexes)
					{
						for (const auto& v : run)
						{
							grow(v);
						}
					}

					for (auto i = 0; i < 3; i++)
					{
						const auto extent = section.maxs[i] - section.mins[i];
						const auto max_value = static_cast<float>((1 << PACKED_BITS[i]) - 1);

						section.codec_parms[i] = section.mins[i];
						section.codec_parms[3 + i] = (extent > 0.0f) ? (extent / max_value) : 0.0f;
					}
				}

				std::vector<build_section> split_into_sections(const mesh_input& input)
				{
					std::vector<build_section> sections;
					build_section current{};

					const auto flush = [&]
					{
						if (!current.primitives.empty())
						{
							finalise_section_codec(current);
							sections.emplace_back(std::move(current));
							current = {};
						}
					};

					auto degenerate_dropped = 0;

					for (const auto& tri : input.triangles)
					{
						if (current.verts.size() + 4 > MAX_SHARED_INDICES_PER_SECTION ||
							current.primitives.size() >= MAX_PRIMITIVES_PER_SECTION)
						{
							flush();
						}

						const auto same = [](const float* a, const float* b)
						{
							return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
						};
						if (same(tri.verts[0], tri.verts[1]) || same(tri.verts[1], tri.verts[2])
							|| same(tri.verts[0], tri.verts[2]))
						{
							degenerate_dropped++;
							continue;
						}

						auto quad = tri.is_quad;
						if (quad && (same(tri.vert3, tri.verts[0]) || same(tri.vert3, tri.verts[1])
							|| same(tri.vert3, tri.verts[2])))
						{
							quad = false;
						}

						const float* corners[4] = {
							tri.verts[0], tri.verts[1], tri.verts[2], tri.vert3
						};

						std::array<std::uint8_t, 4> indices{};
						for (auto i = 0; i < (quad ? 4 : 3); i++)
						{
							auto found = -1;
							for (auto v = 0u; v < current.verts.size(); v++)
							{
								if (current.slot_kinds[v] == slot_kind::vertex &&
									current.verts[v][0] == corners[i][0] &&
									current.verts[v][1] == corners[i][1] &&
									current.verts[v][2] == corners[i][2])
								{
									found = static_cast<int>(v);
									break;
								}
							}

							if (found < 0)
							{
								found = static_cast<int>(current.verts.size());
								std::array<float, 3> v{};
								std::memcpy(v.data(), corners[i], sizeof(float[3]));
								current.verts.emplace_back(v);
								current.slot_kinds.emplace_back(slot_kind::vertex);
							}

							indices[i] = static_cast<std::uint8_t>(found);
						}

						if (!quad)
						{
							indices[3] = indices[2];
						}

						current.primitives.emplace_back(indices);
						current.quads.emplace_back(quad);
						current.custom_index.emplace_back(-1);
						current.tags.emplace_back(tri.surface_tag);
						current.contents.emplace_back(tri.contents);
						current.material_crcs.emplace_back(tri.material_crc);
						current.user_data.emplace_back(tri.user_data);
					}

					auto convexes_rejected = 0;
					std::map<std::string, int> rejection_reasons;

					for (const auto& cvx : input.convexes)
					{
						if (const auto* reason = convex_rejection(cvx.verts))
						{
							convexes_rejected++;
							rejection_reasons[reason]++;
							continue;
						}

						if (current.verts.size() + 2 > MAX_SHARED_INDICES_PER_SECTION ||
							current.primitives.size() >= MAX_PRIMITIVES_PER_SECTION ||
							current.pool_vertex_count() + cvx.verts.size() > SHARED_VERTEX_PAGE_SIZE)
						{
							flush();
						}

						const auto r = current.verts.size();
						if (r > MAX_VERTEX_INDEX)
						{
							ZONETOOL_ERROR("havok: convex record slot %zu does not fit a uint8", r);
							continue;
						}

						current.verts.emplace_back(cvx.verts.front());
						current.slot_kinds.emplace_back(slot_kind::convex_record);
						current.verts.emplace_back(cvx.verts.front());
						current.slot_kinds.emplace_back(slot_kind::convex_start);

						const auto rb = static_cast<std::uint8_t>(r);
						current.primitives.push_back({rb, rb, rb, rb});
						current.quads.emplace_back(false);
						current.custom_index.emplace_back(static_cast<int>(current.convexes.size()));
						current.convexes.emplace_back(cvx.verts);
						current.tags.emplace_back(cvx.surface_tag);
						current.contents.emplace_back(cvx.contents);
						current.material_crcs.emplace_back(cvx.material_crc);
						current.user_data.emplace_back(cvx.user_data);
					}

					flush();

					if (degenerate_dropped)
					{
						ZONETOOL_INFO("havok: dropped %d degenerate triangle(s); emitting them "
							"would have produced primitives the runtime reads as custom "
							"primitives", degenerate_dropped);
					}

					if (convexes_rejected)
					{
						std::string reasons;
						for (const auto& [reason, count] : rejection_reasons)
						{
							reasons += (reasons.empty() ? "" : ", ") + std::to_string(count)
								+ " " + reason;
						}
						ZONETOOL_WARNING("havok: rejected %d of %zu convex(es), which are NOT in "
							"the mesh (%s)", convexes_rejected, input.convexes.size(),
							reasons.c_str());
					}

					return sections;
				}

				constexpr auto CONTENTS_FILTER_MASK = 0xC7FFBFFFu;

				inline std::uint32_t filter_contents(const std::uint32_t contents)
				{
					return contents & CONTENTS_FILTER_MASK;
				}

				inline std::uint32_t filter_lod_contents(const std::uint32_t contents)
				{
					return contents & (CONTENTS_FILTER_MASK | 0x20000000u);
				}

				constexpr auto NIBBLE_MAX = 15;
				constexpr auto NIBBLE_SCALE = 226.0f;

				bool loose_bvh_enabled()
				{
					const auto* env = std::getenv("ZT_HAVOK_BVH_LOOSE");
					return env && env[0] == '1';
				}

				int quantise_inset(const float inset, const float extent)
				{
					if (!(extent > 0.0f) || !(inset > 0.0f))
					{
						return 0;
					}

					const auto ratio = std::clamp(inset / extent, 0.0f, 1.0f);
					auto n = static_cast<int>(std::floor(std::sqrt(ratio * NIBBLE_SCALE)));
					n = std::clamp(n, 0, NIBBLE_MAX);

					while (n > 0 && (static_cast<float>(n * n) / NIBBLE_SCALE) * extent > inset)
					{
						n--;
					}
					return n;
				}

				float decode_inset(const int nibble, const float extent)
				{
					return (static_cast<float>(nibble * nibble) / NIBBLE_SCALE) * extent;
				}

				struct aabb
				{
					float lo[3];
					float hi[3];

					void reset()
					{
						for (auto i = 0; i < 3; i++)
						{
							lo[i] = FLT_MAX;
							hi[i] = -FLT_MAX;
						}
					}

					void add(const aabb& other)
					{
						for (auto i = 0; i < 3; i++)
						{
							lo[i] = std::min(lo[i], other.lo[i]);
							hi[i] = std::max(hi[i], other.hi[i]);
						}
					}
				};

				aabb encode_node_aabb(std::uint8_t* xyz, const aabb& parent, const aabb& child)
				{
					if (loose_bvh_enabled())
					{
						xyz[0] = xyz[1] = xyz[2] = 0;
						return parent;
					}

					aabb decoded{};
					for (auto i = 0; i < 3; i++)
					{
						const auto extent = parent.hi[i] - parent.lo[i];
						const auto hi_n = quantise_inset(child.lo[i] - parent.lo[i], extent);
						const auto lo_n = quantise_inset(parent.hi[i] - child.hi[i], extent);

						xyz[i] = static_cast<std::uint8_t>((hi_n << 4) | lo_n);
						decoded.lo[i] = parent.lo[i] + decode_inset(hi_n, extent);
						decoded.hi[i] = parent.hi[i] - decode_inset(lo_n, extent);
					}
					return decoded;
				}

				void emit_preorder_tree_top(byte_buffer& buf, const int first_leaf,
					const int leaf_count, std::vector<int>& leaf_nodes,
					const std::vector<aabb>& leaf_boxes, const aabb& parent)
				{
					const auto self = static_cast<int>(buf.size() / SIZEOF_NODE_TOP);

					aabb box{};
					box.reset();
					for (auto i = 0; i < leaf_count; i++)
					{
						box.add(leaf_boxes[first_leaf + i]);
					}

					std::uint8_t xyz[3] = {};
					const auto decoded = encode_node_aabb(xyz, parent, box);

					if (leaf_count <= 1)
					{
						leaf_nodes[first_leaf] = self;
						buf.write(xyz, sizeof(xyz));
						buf.write<std::uint8_t>(static_cast<std::uint8_t>((first_leaf >> 8) & 0x7F));
						buf.write<std::uint8_t>(static_cast<std::uint8_t>(first_leaf & 0xFF));
						return;
					}

					const auto left_count = leaf_count / 2;
					const auto right_count = leaf_count - left_count;

					const auto right_offset = 2 * left_count;
					const auto encoded = right_offset / 2;

					buf.write(xyz, sizeof(xyz));
					buf.write<std::uint8_t>(static_cast<std::uint8_t>(0x80 | ((encoded >> 8) & 0x7F)));
					buf.write<std::uint8_t>(static_cast<std::uint8_t>(encoded & 0xFF));

					emit_preorder_tree_top(buf, first_leaf, left_count, leaf_nodes,
						leaf_boxes, decoded);
					emit_preorder_tree_top(buf, first_leaf + left_count, right_count, leaf_nodes,
						leaf_boxes, decoded);
				}

				void emit_preorder_tree_section(byte_buffer& buf, const int first_leaf,
					const int leaf_count, const std::vector<aabb>& leaf_boxes, const aabb& parent)
				{
					aabb box{};
					box.reset();
					for (auto i = 0; i < leaf_count; i++)
					{
						box.add(leaf_boxes[first_leaf + i]);
					}

					std::uint8_t xyz[3] = {};
					const auto decoded = encode_node_aabb(xyz, parent, box);

					if (leaf_count <= 1)
					{
						if (first_leaf > 0x7F)
						{
							ZONETOOL_ERROR("havok: section BVH leaf primitive index %d does "
								"not fit in the 7-bit field", first_leaf);
						}

						buf.write(xyz, sizeof(xyz));
						buf.write<std::uint8_t>(
							static_cast<std::uint8_t>((first_leaf << 1) & 0xFE));
						return;
					}

					const auto left_count = leaf_count / 2;
					const auto right_count = leaf_count - left_count;
					const auto right_offset = 2 * left_count;

					if (right_offset > 0xFE)
					{
						ZONETOOL_ERROR("havok: section BVH right-child offset %d exceeds the "
							"8-bit field", right_offset);
					}

					buf.write(xyz, sizeof(xyz));
					buf.write<std::uint8_t>(static_cast<std::uint8_t>((right_offset & 0xFE) | 1));

					emit_preorder_tree_section(buf, first_leaf, left_count, leaf_boxes, decoded);
					emit_preorder_tree_section(buf, first_leaf + left_count, right_count,
						leaf_boxes, decoded);
				}

				void emit_data_runs(byte_buffer& buf, const std::vector<std::uint16_t>& tags,
					int& runs_written)
				{
					runs_written = 0;

					for (auto i = 0u; i < tags.size();)
					{
						auto j = i;
						while (j < tags.size() && tags[j] == tags[i] && (j - i) < 0xFF)
						{
							j++;
						}

						buf.write<std::uint16_t>(tags[i]);
						buf.write<std::uint8_t>(static_cast<std::uint8_t>(i));
						buf.write<std::uint8_t>(static_cast<std::uint8_t>(j - i));

						runs_written++;
						i = j;
					}
				}
			}

			const char* convex_rejection(const std::vector<std::array<float, 3>>& verts)
			{
				if (verts.size() < 4)
				{
					return "fewer than 4 vertices";
				}
				if (verts.size() > MAX_CONVEX_VERTICES)
				{
					return "more than 255 vertices";
				}

				for (auto i = 0u; i < verts.size(); i++)
				{
					for (auto j = i + 1; j < verts.size(); j++)
					{
						if (verts[i] == verts[j])
						{
							return "repeated vertices";
						}
					}
				}

				const auto sub = [](const std::array<float, 3>& a, const std::array<float, 3>& b)
				{
					return std::array<double, 3>{
						static_cast<double>(a[0]) - b[0],
						static_cast<double>(a[1]) - b[1],
						static_cast<double>(a[2]) - b[2]};
				};
				const auto cross = [](const std::array<double, 3>& a, const std::array<double, 3>& b)
				{
					return std::array<double, 3>{
						a[1] * b[2] - a[2] * b[1],
						a[2] * b[0] - a[0] * b[2],
						a[0] * b[1] - a[1] * b[0]};
				};
				const auto dot = [](const std::array<double, 3>& a, const std::array<double, 3>& b)
				{
					return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
				};

				const auto width_for = [&](const std::size_t a, const std::size_t b,
					const std::size_t c) -> double
				{
					auto n = cross(sub(verts[b], verts[a]), sub(verts[c], verts[a]));
					const auto length = std::sqrt(dot(n, n));
					if (length < 1e-12)
					{
						return -1.0;
					}
					auto width = 0.0;
					for (const auto& v : verts)
					{
						width = std::max(width, std::fabs(dot(sub(v, verts[a]), n)) / length);
					}
					return width;
				};

				auto any_plane = false;
				if (verts.size() <= 32)
				{
					for (auto a = 0u; a < verts.size(); a++)
					{
						for (auto b = a + 1; b < verts.size(); b++)
						{
							for (auto c = b + 1; c < verts.size(); c++)
							{
								const auto width = width_for(a, b, c);
								if (width < 0.0)
								{
									continue;
								}
								any_plane = true;
								if (width <= CONVEX_FLAT_TOLERANCE)
								{
									return "flat within 1e-4";
								}
							}
						}
					}
				}
				else
				{
					std::size_t b = 0, c = 0;
					auto best = -1.0;
					for (auto i = 1u; i < verts.size(); i++)
					{
						const auto d = sub(verts[i], verts[0]);
						if (dot(d, d) > best)
						{
							best = dot(d, d);
							b = i;
						}
					}
					best = -1.0;
					const auto axis = sub(verts[b], verts[0]);
					for (auto i = 1u; i < verts.size(); i++)
					{
						const auto n = cross(axis, sub(verts[i], verts[0]));
						if (dot(n, n) > best)
						{
							best = dot(n, n);
							c = i;
						}
					}
					const auto width = width_for(0, b, c);
					if (width >= 0.0)
					{
						any_plane = true;
						if (width <= CONVEX_FLAT_TOLERANCE)
						{
							return "flat within 1e-4";
						}
					}
				}

				if (!any_plane)
				{
					return "collinear";
				}

				return nullptr;
			}

			namespace
			{
				// A mesh split into sections, with the counts and bounds its objects need.
				struct prepared_mesh
				{
					std::vector<build_section> sections;
					std::size_t total_customs = 0;
					std::size_t total_prims = 0;
					std::size_t total_verts = 0;
					int num_primitive_keys = 0;
					int mesh_triangles = 0;
					int max_key = 0;
					int bits_per_key = 1;
					float world_min[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
					float world_max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
					float convex_radius = 0.0f;
				};

				bool prepare_mesh(const mesh_input& input, prepared_mesh& mesh)
				{
					if (input.triangles.empty() && input.convexes.empty())
					{
						ZONETOOL_ERROR("havok: refusing to build a mesh blob from 0 triangles "
							"and 0 convexes");
						return false;
					}

					mesh.sections = split_into_sections(input);
					if (mesh.sections.empty())
					{
						ZONETOOL_ERROR("havok: no sections produced");
						return false;
					}
					mesh.convex_radius = input.convex_radius;

					for (const auto& section : mesh.sections)
					{
						mesh.total_customs += section.convexes.size();
					}

					for (auto si = 0u; si < mesh.sections.size(); si++)
					{
						const auto last = static_cast<int>(mesh.sections[si].primitives.size()) - 1;
						const auto key = static_cast<int>(si << KEY_SECTION_SHIFT) | (last << 1) | 1;
						mesh.max_key = std::max(mesh.max_key, key);
					}

					while ((1 << mesh.bits_per_key) <= mesh.max_key)
					{
						mesh.bits_per_key++;
					}

					for (const auto& section : mesh.sections)
					{
						mesh.total_prims += section.primitives.size();
						mesh.total_verts += section.pool_vertex_count();
					}

					for (const auto& section : mesh.sections)
					{
						for (auto pi = 0u; pi < section.primitives.size(); pi++)
						{
							if (section.custom_index[pi] >= 0)
							{
								mesh.num_primitive_keys += 1;
								continue;
							}
							const auto keys = section.quads[pi] ? 2 : 1;
							mesh.num_primitive_keys += keys;
							mesh.mesh_triangles += keys;
						}
					}

					for (const auto& section : mesh.sections)
					{
						for (auto i = 0; i < 3; i++)
						{
							mesh.world_min[i] = std::min(mesh.world_min[i], section.mins[i]);
							mesh.world_max[i] = std::max(mesh.world_max[i], section.maxs[i]);
						}
					}
					return true;
				}

				using mesh_palette = std::vector<std::tuple<int, std::uint32_t, std::uint64_t>>;

				// replaces every primitive's tag with its (contents, material, user data) palette index
				void assign_tags(prepared_mesh& mesh, mesh_palette& palette)
				{
					for (auto& section : mesh.sections)
					{
						for (auto i = 0u; i < section.tags.size(); i++)
						{
							const auto key = std::make_tuple(section.contents[i],
								section.material_crcs[i], section.user_data[i]);

							auto it = std::find(palette.begin(), palette.end(), key);
							if (it == palette.end())
							{
								palette.emplace_back(key);
								it = palette.end() - 1;
							}
							section.tags[i] = static_cast<unsigned short>(
								std::distance(palette.begin(), it));
						}
					}
				}

				struct pending_fixup
				{
					std::size_t src;
					std::size_t dst;
				};

				struct blob_fixups
				{
					std::vector<pending_fixup> local;
					std::vector<std::array<int, 3>> global;
					std::vector<std::pair<std::size_t, int>> virtuals;
				};

				// Writes one hknpCompressedMeshShape and its hknpCompressedMeshShapeData at the end of
				// `buf` and returns the shape's offset. `tagged` = false leaves every primitive untagged
				// (physics assets carry their material in the asset's lookup table instead). 0 on error.
				std::size_t emit_mesh(prepared_mesh& mesh, const vertex_storage vertex_format, const bool tagged,
					byte_buffer& buf, blob_fixups& fixups, const int class_shape, const int class_shape_data)
				{
					auto& sections = mesh.sections;
					const auto section_count = static_cast<int>(sections.size());
					const auto& world_min = mesh.world_min;
					const auto& world_max = mesh.world_max;
					auto& local_fixups = fixups.local;
					auto& global_fixups = fixups.global;
					auto& virtual_fixups = fixups.virtuals;
					const auto align16 = [&] { buf.align(16, 0); };

					aabb world_box{};
					for (auto i = 0; i < 3; i++)
					{
						world_box.lo[i] = world_min[i];
						world_box.hi[i] = world_max[i];
					}

					std::vector<aabb> section_boxes(sections.size());
					for (auto i = 0u; i < sections.size(); i++)
					{
						for (auto c = 0; c < 3; c++)
						{
							section_boxes[i].lo[c] = sections[i].mins[c];
							section_boxes[i].hi[c] = sections[i].maxs[c];
						}
					}

					byte_buffer top_nodes;
					std::vector<int> leaf_nodes(sections.size(), 0);
					emit_preorder_tree_top(top_nodes, 0, section_count, leaf_nodes,
						section_boxes, world_box);

					std::vector<byte_buffer> section_nodes(sections.size());
					for (auto i = 0u; i < sections.size(); i++)
					{
						const auto& section = sections[i];

						std::vector<aabb> prim_boxes(section.primitives.size());
						for (auto p = 0u; p < section.primitives.size(); p++)
						{
							prim_boxes[p].reset();
							if (section.custom_index[p] >= 0)
							{
								for (const auto& v : section.convexes[section.custom_index[p]])
								{
									aabb point{};
									for (auto c = 0; c < 3; c++)
									{
										point.lo[c] = v[c];
										point.hi[c] = v[c];
									}
									prim_boxes[p].add(point);
								}
								continue;
							}
							for (auto k = 0; k < (section.quads[p] ? 4 : 3); k++)
							{
								const auto& v = section.verts[section.primitives[p][k]];
								aabb point{};
								for (auto c = 0; c < 3; c++)
								{
									point.lo[c] = v[c];
									point.hi[c] = v[c];
								}
								prim_boxes[p].add(point);
							}
						}

						emit_preorder_tree_section(section_nodes[i], 0,
							static_cast<int>(section.primitives.size()),
							prim_boxes, section_boxes[i]);
					}

					byte_buffer primitives;
					byte_buffer packed_vertices;
					byte_buffer shared_vertices;
					byte_buffer shared_vertex_index;
					byte_buffer data_runs;
					std::vector<int> section_first_prim(sections.size());
					std::vector<int> section_first_packed(sections.size());
					std::vector<int> section_first_vert(sections.size());
					std::vector<int> section_page(sections.size());
					std::vector<int> section_first_run(sections.size());
					std::vector<int> section_run_count(sections.size());
					auto quantised_duplicate_runs = 0;

					for (auto i = 0u; i < sections.size(); i++)
					{
						auto& section = sections[i];

						section_first_prim[i] = static_cast<int>(primitives.size() / SIZEOF_PRIMITIVE);
						for (const auto& prim : section.primitives)
						{
							primitives.write(prim.data(), 4);
						}

						if (vertex_format == vertex_storage::packed)
						{
							section_first_packed[i] =
								static_cast<int>(packed_vertices.size() / sizeof(std::uint32_t));
							for (const auto& vertex : section.verts)
							{
								packed_vertices.write<std::uint32_t>(
									pack_vertex(vertex.data(), section.codec_parms));
							}
							section_first_vert[i] = 0;
							section_page[i] = 0;
						}
						else
						{
							auto vert_base = shared_vertices.size() / 8;
							const auto vert_needed = section.pool_vertex_count();

							if (vert_needed > 0
								&& ((vert_base + vert_needed - 1) >> 16) != (vert_base >> 16))
							{
								const auto next_page = ((vert_base >> 16) + 1) << 16;
								while (shared_vertices.size() / 8 < next_page)
								{
									shared_vertices.write<std::uint64_t>(0);
								}
								vert_base = next_page;
							}

							section_page[i] = static_cast<int>(vert_base >> 16);

							if (section_page[i] > 0xFF)
							{
								ZONETOOL_ERROR("havok: section %u needs vertex page %d, but the page "
									"field is a uint8 -- the mesh has more than %u shared vertices",
									i, section_page[i], 0x100u << 16);
								return 0;
							}

							section_first_vert[i] = static_cast<int>(shared_vertex_index.size() / 2);

							if (section_first_vert[i] > 0xFFFFFF)
							{
								ZONETOOL_ERROR("havok: sharedVerticesIndex has %d entries, which does "
									"not fit the section's 24-bit first-index field",
									section_first_vert[i]);
								return 0;
							}

							const auto page_base = static_cast<std::size_t>(section_page[i]) << 16;

							const auto listed = section.listed_vertex_count();
							std::vector<std::size_t> run_start(section.convexes.size());
							{
								auto next = vert_base + listed;
								for (auto c = 0u; c < section.convexes.size(); c++)
								{
									run_start[c] = next;
									next += section.convexes[c].size();
								}
							}

							std::vector<int> slot_custom(section.verts.size(), -1);
							for (auto p = 0u; p < section.primitives.size(); p++)
							{
								const auto custom = section.custom_index[p];
								if (custom >= 0)
								{
									slot_custom[section.primitives[p][0]] = custom;
									slot_custom[section.primitives[p][0] + 1u] = custom;
								}
							}

							auto listed_written = 0u;
							for (auto v = 0u; v < section.verts.size(); v++)
							{
								switch (section.slot_kinds[v])
								{
								case slot_kind::vertex:
									shared_vertices.write<std::uint64_t>(
										pack_shared_vertex(section.verts[v].data(), world_min, world_max));
									shared_vertex_index.write<std::uint16_t>(
										static_cast<std::uint16_t>(vert_base + listed_written - page_base));
									listed_written++;
									break;
								case slot_kind::convex_record:
									shared_vertex_index.write<std::uint16_t>(static_cast<std::uint16_t>(
										(section.convexes[slot_custom[v]].size() << 8)
										| CUSTOM_PRIMITIVE_CONVEX));
									break;
								case slot_kind::convex_start:
									shared_vertex_index.write<std::uint16_t>(static_cast<std::uint16_t>(
										run_start[slot_custom[v]] - page_base));
									break;
								}
							}

							for (const auto& run : section.convexes)
							{
								std::vector<std::uint64_t> packed_run;
								packed_run.reserve(run.size());
								for (const auto& vertex : run)
								{
									packed_run.emplace_back(
										pack_shared_vertex(vertex.data(), world_min, world_max));
									shared_vertices.write<std::uint64_t>(packed_run.back());
								}

								std::sort(packed_run.begin(), packed_run.end());
								if (std::adjacent_find(packed_run.begin(), packed_run.end())
									!= packed_run.end())
								{
									quantised_duplicate_runs++;
								}
							}

							if (shared_vertices.size() / 8 != vert_base + vert_needed)
							{
								ZONETOOL_ERROR("havok: section %u wrote %zu pool vertices, expected %zu",
									i, shared_vertices.size() / 8 - vert_base, vert_needed);
								return 0;
							}
						}

						section_first_run[i] = static_cast<int>(data_runs.size() / SIZEOF_DATA_RUN);
						auto runs = 0;
						if (!tagged)
						{
							const std::vector<std::uint16_t> untagged(section.tags.size(), 0xFFFF);
							emit_data_runs(data_runs, untagged, runs);
						}
						else
						{
							emit_data_runs(data_runs, section.tags, runs);
						}
						section_run_count[i] = runs;
					}

					if (quantised_duplicate_runs)
					{
						ZONETOOL_WARNING("havok: %d convex vertex run(s) have two vertices that "
							"quantise to the same shared vertex over this tree domain",
							quantised_duplicate_runs);
					}

					struct simd_node
					{
						float lo[3][4];
						float hi[3][4];
						std::uint32_t data[4];
					};

					struct simd_item
					{
						std::uint32_t key;
						float lo[3];
						float hi[3];
					};

					std::vector<simd_item> items;
					items.reserve(mesh.total_prims);
					for (auto si = 0u; si < sections.size(); si++)
					{
						const auto& section = sections[si];
						for (auto pi = 0u; pi < section.primitives.size(); pi++)
						{
							simd_item item{};
							item.key = static_cast<std::uint32_t>(
								(si << KEY_SECTION_SHIFT) | (pi << 1));
							for (auto c = 0; c < 3; c++)
							{
								item.lo[c] = FLT_MAX;
								item.hi[c] = -FLT_MAX;
							}
							const auto grow = [&item](const std::array<float, 3>& v)
							{
								for (auto c = 0; c < 3; c++)
								{
									item.lo[c] = std::min(item.lo[c], v[c]);
									item.hi[c] = std::max(item.hi[c], v[c]);
								}
							};
							if (section.custom_index[pi] >= 0)
							{
								for (const auto& v : section.convexes[section.custom_index[pi]])
								{
									grow(v);
								}
							}
							else
							{
								for (auto k = 0; k < (section.quads[pi] ? 4 : 3); k++)
								{
									grow(section.verts[section.primitives[pi][k]]);
								}
							}
							items.emplace_back(item);
						}
					}

					std::vector<simd_node> simd_nodes;
					simd_nodes.reserve(items.size());

					const auto clear_node = [](simd_node& node)
					{
						for (auto c = 0; c < 3; c++)
						{
							for (auto s = 0; s < 4; s++)
							{
								node.lo[c][s] = FLT_MAX;
								node.hi[c][s] = -FLT_MAX;
							}
						}
						node.data[0] = node.data[1] = node.data[2] = node.data[3] = 0;
					};

					simd_nodes.emplace_back();
					clear_node(simd_nodes.back());

					const std::function<int(std::vector<simd_item>&)> build_simd =
						[&](std::vector<simd_item>& group) -> int
					{
						const auto self = static_cast<int>(simd_nodes.size());
						simd_nodes.emplace_back();
						clear_node(simd_nodes.back());

						const auto set_slot = [&](const int slot, const float* lo, const float* hi,
							const std::uint32_t data)
						{
							auto& node = simd_nodes[self];
							for (auto c = 0; c < 3; c++)
							{
								node.lo[c][slot] = lo[c];
								node.hi[c][slot] = hi[c];
							}
							node.data[slot] = data;
						};

						if (group.size() <= 4)
						{
							for (auto i = 0u; i < group.size(); i++)
							{
								set_slot(static_cast<int>(i), group[i].lo, group[i].hi,
									(group[i].key << 1) | 1);
							}
							return self;
						}

						float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
						float hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
						for (const auto& item : group)
						{
							for (auto c = 0; c < 3; c++)
							{
								lo[c] = std::min(lo[c], item.lo[c]);
								hi[c] = std::max(hi[c], item.hi[c]);
							}
						}

						auto axis = 0;
						for (auto c = 1; c < 3; c++)
						{
							if ((hi[c] - lo[c]) > (hi[axis] - lo[axis]))
							{
								axis = c;
							}
						}

						std::sort(group.begin(), group.end(),
							[axis](const simd_item& a, const simd_item& b)
							{
								return (a.lo[axis] + a.hi[axis]) < (b.lo[axis] + b.hi[axis]);
							});

						const auto total = group.size();
						for (auto slot = 0; slot < 4; slot++)
						{
							const auto begin = total * slot / 4;
							const auto end = total * (slot + 1) / 4;
							if (begin >= end)
							{
								continue;
							}

							std::vector<simd_item> sub(group.begin() + begin, group.begin() + end);

							float slo[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
							float shi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
							for (const auto& item : sub)
							{
								for (auto c = 0; c < 3; c++)
								{
									slo[c] = std::min(slo[c], item.lo[c]);
									shi[c] = std::max(shi[c], item.hi[c]);
								}
							}

							if (sub.size() == 1)
							{
								set_slot(slot, slo, shi, (sub[0].key << 1) | 1);
							}
							else
							{
								const auto child = build_simd(sub);
								set_slot(slot, slo, shi, static_cast<std::uint32_t>(child) << 1);
							}
						}

						return self;
					};

					build_simd(items);

					const auto shape_offset = buf.size();
					virtual_fixups.emplace_back(shape_offset, class_shape);

					buf.reserve(16);
					buf.write<std::uint16_t>(4);
					buf.write<std::uint8_t>(static_cast<std::uint8_t>(mesh.bits_per_key));
					buf.write<std::uint8_t>(2);
					buf.write<float>(mesh.convex_radius);
					buf.write<std::uint64_t>(0);
					buf.reserve(8);
					buf.reserve(8);
					buf.write<std::uint32_t>(0xFFFFFFFF);
					buf.write<std::uint32_t>(0);
					write_hk_array_header(buf, 0);
					write_hk_array_header(buf, 0);
					buf.write<std::uint32_t>(0xFFFFFFFF);
					buf.reserve(4);
					const auto shape_data_slot = buf.reserve(8);

					const auto interior_bits = mesh.max_key + 1;
					const auto quad_bits = interior_bits / 2;
					const auto interior_words = (interior_bits + 31) / 32;
					const auto quad_words = (quad_bits + 31) / 32;

					const auto quad_field = buf.size();
					write_hk_array_header(buf, quad_words);
					buf.write<std::int32_t>(quad_bits);
					buf.reserve(4);
					const auto interior_field = buf.size();
					write_hk_array_header(buf, interior_words);
					buf.write<std::int32_t>(interior_bits);
					buf.reserve(4);
					buf.write<std::int32_t>(0);
					buf.write<std::int32_t>(0);

					if (buf.size() - shape_offset != SIZEOF_COMPRESSED_MESH_SHAPE)
					{
						ZONETOOL_ERROR("havok: hknpCompressedMeshShape is %zu bytes, expected %d",
							buf.size() - shape_offset, SIZEOF_COMPRESSED_MESH_SHAPE);
						return 0;
					}
					align16();

					std::vector<std::uint32_t> quad_bitfield(quad_words, 0);
					for (auto si = 0u; si < sections.size(); si++)
					{
						const auto& section = sections[si];
						for (auto pi = 0u; pi < section.primitives.size(); pi++)
						{
							if (!section.quads[pi])
							{
								continue;
							}

							float corner[4][3];
							for (auto k = 0; k < 4; k++)
							{
								const auto& v = section.verts[section.primitives[pi][k]];
								if (vertex_format == vertex_storage::packed)
								{
									unpack_vertex(pack_vertex(v.data(), section.codec_parms),
										section.codec_parms, corner[k]);
								}
								else
								{
									unpack_shared_vertex(
										pack_shared_vertex(v.data(), world_min, world_max),
										world_min, world_max, corner[k]);
								}
							}

							float e1[3], e2[3], n[3];
							for (auto c = 0; c < 3; c++)
							{
								e1[c] = corner[1][c] - corner[0][c];
								e2[c] = corner[2][c] - corner[0][c];
							}
							n[0] = e1[1] * e2[2] - e1[2] * e2[1];
							n[1] = e1[2] * e2[0] - e1[0] * e2[2];
							n[2] = e1[0] * e2[1] - e1[1] * e2[0];

							const auto length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
							if (length < 1e-12f)
							{
								continue;
							}

							auto deviation = 0.0f;
							for (auto c = 0; c < 3; c++)
							{
								deviation += (corner[3][c] - corner[0][c]) * n[c] / length;
							}

							if (std::fabs(deviation) > 1e-5f)
							{
								continue;
							}

							const auto bit = static_cast<std::size_t>(
								((si << KEY_SECTION_SHIFT) | (pi << 1)) >> 1);
							if (bit / 32 < quad_bitfield.size())
							{
								quad_bitfield[bit / 32] |= 1u << (bit % 32);
							}
						}
					}

					local_fixups.push_back({quad_field, buf.size()});
					for (const auto word : quad_bitfield)
					{
						buf.write<std::uint32_t>(word);
					}
					align16();
					local_fixups.push_back({interior_field, buf.size()});
					buf.fill(static_cast<std::size_t>(interior_words) * 4, 0);
					align16();

					const auto data_offset = buf.size();
					virtual_fixups.emplace_back(data_offset, class_shape_data);
					global_fixups.push_back({static_cast<int>(shape_data_slot), 2,
						static_cast<int>(data_offset)});

					buf.reserve(16);
					const auto tree_offset = buf.size();

					const auto tree_nodes_field = buf.size();
					write_hk_array_header(buf, static_cast<int>(top_nodes.size() / SIZEOF_NODE_TOP));
					for (auto i = 0; i < 3; i++) buf.write<float>(world_min[i]);
					buf.write<float>(0.0f);
					for (auto i = 0; i < 3; i++) buf.write<float>(world_max[i]);
					buf.write<float>(0.0f);
					buf.write<std::int32_t>(mesh.num_primitive_keys);
					buf.write<std::int32_t>(mesh.bits_per_key);
					buf.write<std::uint32_t>(static_cast<std::uint32_t>(mesh.max_key));
					buf.reserve(4);
					const auto tree_sections_field = buf.size();
					write_hk_array_header(buf, section_count);
					const auto tree_primitives_field = buf.size();
					write_hk_array_header(buf, static_cast<int>(mesh.total_prims));
					const auto tree_svi_field = buf.size();
					write_hk_array_header(buf, static_cast<int>(shared_vertex_index.size() / 2));
					const auto tree_packed_field = buf.size();
					write_hk_array_header(buf, static_cast<int>(packed_vertices.size() / 4));
					const auto tree_shared_field = buf.size();
					write_hk_array_header(buf, static_cast<int>(shared_vertices.size() / 8));
					const auto tree_runs_field = buf.size();
					write_hk_array_header(buf, static_cast<int>(data_runs.size() / SIZEOF_DATA_RUN));

					if (buf.size() - tree_offset != SIZEOF_MESH_TREE)
					{
						ZONETOOL_ERROR("havok: mesh tree is %zu bytes, expected %d",
							buf.size() - tree_offset, SIZEOF_MESH_TREE);
						return 0;
					}

					buf.reserve(8);
					const auto simd_field = buf.size();
					write_hk_array_header(buf, static_cast<int>(simd_nodes.size()));
					write_hk_array_header(buf, 0);
					write_hk_array_header(buf, 0);
					write_hk_array_header(buf, 0);
					buf.reserve(8);

					if (buf.size() - data_offset != SIZEOF_COMPRESSED_MESH_SHAPE_DATA)
					{
						ZONETOOL_ERROR("havok: hknpCompressedMeshShapeData is %zu bytes, expected %d",
							buf.size() - data_offset, SIZEOF_COMPRESSED_MESH_SHAPE_DATA);
						return 0;
					}
					align16();

					local_fixups.push_back({simd_field, buf.size()});
					for (const auto& node : simd_nodes)
					{
						for (auto c = 0; c < 3; c++)
						{
							buf.write(node.lo[c], sizeof(node.lo[c]));
							buf.write(node.hi[c], sizeof(node.hi[c]));
						}
						buf.write(node.data, sizeof(node.data));
					}
					align16();

					local_fixups.push_back({tree_nodes_field, buf.size()});
					buf.write(top_nodes.data.data(), top_nodes.size());
					align16();

					local_fixups.push_back({tree_sections_field, buf.size()});
					std::vector<std::size_t> section_node_field(sections.size());
					for (auto i = 0u; i < sections.size(); i++)
					{
						const auto& section = sections[i];
						const auto base = buf.size();

						section_node_field[i] = buf.size();
						write_hk_array_header(buf,
							static_cast<int>(section_nodes[i].size() / SIZEOF_NODE_SECTION));
						for (auto c = 0; c < 3; c++) buf.write<float>(section.mins[c]);
						buf.write<float>(0.0f);
						for (auto c = 0; c < 3; c++) buf.write<float>(section.maxs[c]);
						buf.write<float>(0.0f);
						if (vertex_format == vertex_storage::packed)
						{
							for (auto c = 0; c < 6; c++)
							{
								buf.write<float>(section.codec_parms[c]);
							}
						}
						else
						{
							for (auto c = 0; c < 3; c++) buf.write<float>(FLT_MAX);
							for (auto c = 0; c < 3; c++) buf.write<float>(-INFINITY);
						}
						buf.write<std::uint32_t>(
							static_cast<std::uint32_t>(section_first_packed[i]));
						buf.write<std::uint32_t>(static_cast<std::uint32_t>(
							(section_first_vert[i] << 8) |
							(vertex_format == vertex_storage::packed ? section.verts.size() : 0)));
						buf.write<std::uint32_t>(static_cast<std::uint32_t>(
							(section_first_prim[i] << 8) | section.primitives.size()));
						buf.write<std::uint32_t>(static_cast<std::uint32_t>(
							(section_first_run[i] << 8) | section_run_count[i]));
						buf.write<std::uint8_t>(vertex_format == vertex_storage::packed
							? static_cast<std::uint8_t>(section.verts.size()) : 0);
						buf.write<std::uint8_t>(vertex_format == vertex_storage::shared
							? static_cast<std::uint8_t>(section.verts.size()) : 0);
						buf.write<std::uint16_t>(static_cast<std::uint16_t>(leaf_nodes[i]));
						buf.write<std::uint8_t>(static_cast<std::uint8_t>(section_page[i]));
						buf.write<std::uint8_t>(0);
						buf.write<std::uint8_t>(0);
						buf.write<std::uint8_t>(0);

						if (buf.size() - base != SIZEOF_SECTION)
						{
							ZONETOOL_ERROR("havok: section is %zu bytes, expected %d",
								buf.size() - base, SIZEOF_SECTION);
							return 0;
						}
					}
					align16();

					for (auto i = 0u; i < sections.size(); i++)
					{
						local_fixups.push_back({section_node_field[i], buf.size()});
						buf.write(section_nodes[i].data.data(), section_nodes[i].size());
						align16();
					}

					local_fixups.push_back({tree_primitives_field, buf.size()});
					buf.write(primitives.data.data(), primitives.size());
					align16();

					if (!shared_vertex_index.data.empty())
					{
						local_fixups.push_back({tree_svi_field, buf.size()});
						buf.write(shared_vertex_index.data.data(), shared_vertex_index.size());
						align16();
					}

					if (!packed_vertices.data.empty())
					{
						local_fixups.push_back({tree_packed_field, buf.size()});
						buf.write(packed_vertices.data.data(), packed_vertices.size());
						align16();
					}

					if (!shared_vertices.data.empty())
					{
						local_fixups.push_back({tree_shared_field, buf.size()});
						buf.write(shared_vertices.data.data(), shared_vertices.size());
						align16();
					}

					local_fixups.push_back({tree_runs_field, buf.size()});
					buf.write(data_runs.data.data(), data_runs.size());
					align16();

					return shape_offset;
				}

				enum class mesh_blob_kind
				{
					world,
					physics_asset,
					xmodel_lod,
				};

				// `meshes` holds one mesh, or for an XModel physics LOD one per bone (`lod_bones`)
				std::vector<std::uint8_t> build_mesh_blob(const mesh_blob_kind kind, const std::vector<const mesh_input*>& meshes,
					const std::vector<std::string>& lod_bones, const physics_asset_input* physics_asset,
					std::vector<shape_tag>* out_tags)
				{
					if (meshes.empty() || (kind != mesh_blob_kind::xmodel_lod && meshes.size() != 1)
						|| (kind == mesh_blob_kind::xmodel_lod && lod_bones.size() != meshes.size()))
					{
						ZONETOOL_ERROR("havok: a %s blob cannot hold %zu mesh(es)", kind == mesh_blob_kind::xmodel_lod
							? "physics LOD" : "single-shape", meshes.size());
						return {};
					}

					std::vector<prepared_mesh> prepared(meshes.size());
					std::size_t total_customs = 0;
					for (auto m = 0u; m < meshes.size(); m++)
					{
						if (!prepare_mesh(*meshes[m], prepared[m]))
						{
							return {};
						}
						total_customs += prepared[m].total_customs;
					}

					auto vertex_format = selected_vertex_storage();
					if (vertex_format == vertex_storage::packed && total_customs)
					{
						ZONETOOL_WARNING("havok: ZT_HAVOK_VERTEX_STORAGE=packed cannot carry %zu "
							"convex custom primitive(s); using shared vertices for this blob",
							total_customs);
						vertex_format = vertex_storage::shared;
					}
					ZONETOOL_INFO("havok: vertex storage %s (ZT_HAVOK_VERTEX_STORAGE)",
						vertex_format == vertex_storage::packed ? "packed" : "shared");

					mesh_palette palette;
					for (auto& mesh : prepared)
					{
						assign_tags(mesh, palette);
					}

					std::vector<shape_tag> tag_records;
					tag_records.reserve(palette.size());
					for (const auto& entry : palette)
					{
						tag_records.emplace_back(shape_tag{
							filter_contents(static_cast<std::uint32_t>(std::get<0>(entry))),
							std::get<1>(entry), std::get<2>(entry)});
					}

					byte_buffer buf;
					blob_fixups fixups;
					auto& local_fixups = fixups.local;
					auto& global_fixups = fixups.global;
					auto& virtual_fixups = fixups.virtuals;

					const auto align16 = [&] { buf.align(16, 0); };

					// the pointer slots each shape's offset goes into
					std::vector<std::vector<std::size_t>> shape_ptr_slots(prepared.size());
					std::size_t shape_list_offset = 0;

					const int class_shape = kind == mesh_blob_kind::physics_asset ? 2 : 1;
					const int class_shape_data = kind == mesh_blob_kind::physics_asset ? 3 : 2;

					const auto tag_count = static_cast<int>(palette.size());
					auto world_contents = WORLD_SHAPE_CONTENTS;
					if (const auto* env = std::getenv("ZT_HAVOK_WORLD_CONTENTS"))
					{
						world_contents = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 16));
						ZONETOOL_INFO("havok: world shape contents overridden to 0x%08X",
							world_contents);
					}

					if (kind == mesh_blob_kind::world)
					{
						const auto& mesh = prepared[0];
						shape_list_offset = buf.size();

						const std::array<int, 10> member_counts = {
							1, 1, 1, 1, 1, 2, 0, tag_count, 1, 1
						};

						std::array<std::size_t, 10> member_field{};
						for (auto i = 0; i < 6; i++)
						{
							member_field[i] = buf.size();
							write_hk_array_header(buf, member_counts[i]);
						}
						buf.write<std::int32_t>(0);
						buf.reserve(4);
						for (auto i = 7; i < 10; i++)
						{
							member_field[i] = buf.size();
							write_hk_array_header(buf, member_counts[i]);
						}

						if (buf.size() - shape_list_offset != 152)
						{
							ZONETOOL_ERROR("havok: HavokPhysicsShapeList is %zu bytes, expected 152",
								buf.size() - shape_list_offset);
							return {};
						}

						align16();

						const auto record_array = [&](const int index, const std::size_t payload)
						{
							local_fixups.push_back({member_field[index], payload});
						};

						const auto shapes_payload = buf.size();
						record_array(0, shapes_payload);
						const auto shape_ptr_slot = buf.reserve(8);
						align16();

						record_array(1, buf.size());
						buf.write<std::int32_t>(0);
						align16();

						record_array(2, buf.size());
						const auto name_ptr_slot = buf.reserve(8);
						align16();
						const auto name_payload = buf.size();
						const char* world_name = "World Entity 0";
						buf.write(world_name, std::strlen(world_name) + 1);
						align16();
						local_fixups.push_back({name_ptr_slot, name_payload});

						record_array(3, buf.size());
						buf.write<std::int32_t>(static_cast<std::int32_t>(mesh.total_verts));
						align16();
						record_array(4, buf.size());
						buf.write<std::int32_t>(static_cast<std::int32_t>(mesh.mesh_triangles));
						align16();

						record_array(5, buf.size());
						for (auto i = 0; i < 3; i++) buf.write<float>(mesh.world_min[i]);
						buf.write<float>(0.0f);
						for (auto i = 0; i < 3; i++) buf.write<float>(mesh.world_max[i]);
						buf.write<float>(0.0f);
						align16();

						record_array(7, buf.size());
						for (auto i = 0u; i < palette.size(); i++)
						{
							buf.write<std::uint32_t>(tag_records[i].collision_filter);
							buf.write<std::uint32_t>(tag_records[i].material_crc);
							buf.write<std::uint16_t>(0xFFFF);
							buf.reserve(6);
							buf.write<std::uint64_t>(tag_records[i].user_data);
						}
						align16();

						record_array(8, buf.size());
						buf.write<std::uint32_t>(world_contents);
						align16();
						record_array(9, buf.size());
						buf.write<std::int32_t>(static_cast<std::int32_t>(mesh.total_customs));
						align16();
						shape_ptr_slots[0].push_back(shape_ptr_slot);
					}
					else if (kind == mesh_blob_kind::xmodel_lod)
					{
						// HavokPhysicsXModelLOD { shapes, bone names, tag table }: the shape pointers, the
						// name pointers followed by the names back to back, then the {userData, contents}
						// table (stock layout, e.g. barrier_park_wall_2_panels_animated)
						const auto count = static_cast<int>(prepared.size());
						virtual_fixups.emplace_back(static_cast<std::size_t>(0), 0);
						const auto shapes_field = buf.size();
						write_hk_array_header(buf, count);
						const auto names_field = buf.size();
						write_hk_array_header(buf, count);
						const auto tag_data_field = buf.size();
						write_hk_array_header(buf, static_cast<int>(palette.size()));

						local_fixups.push_back({shapes_field, buf.size()});
						for (auto i = 0; i < count; i++)
						{
							shape_ptr_slots[i].push_back(buf.reserve(8));
						}
						align16();

						local_fixups.push_back({names_field, buf.size()});
						std::vector<std::size_t> name_ptr_slots;
						for (auto i = 0; i < count; i++)
						{
							name_ptr_slots.push_back(buf.reserve(8));
						}
						for (auto i = 0; i < count; i++)
						{
							local_fixups.push_back({name_ptr_slots[i], buf.size()});
							buf.write(lod_bones[i].c_str(), lod_bones[i].size() + 1);
						}
						align16();

						local_fixups.push_back({tag_data_field, buf.size()});
						for (const auto& entry : palette)
						{
							buf.write<std::uint32_t>(static_cast<std::uint32_t>(std::get<2>(entry)));
							buf.write<std::uint32_t>(filter_lod_contents(
								static_cast<std::uint32_t>(std::get<0>(entry))));
						}
						align16();
					}
					else
					{
						virtual_fixups.emplace_back(static_cast<std::size_t>(0), 0);
						buf.reserve(8);
						const auto system_data_slot = buf.reserve(8);

						std::array<std::size_t, 8> lookup_field{};
						const int lookup_count[8] = {1, 1, 0, 1, 1, 1, 0, 1};
						for (auto i = 0; i < 8; i++)
						{
							lookup_field[i] = buf.size();
							write_hk_array_header(buf, lookup_count[i]);
						}
						if (buf.size() != 144)
						{
							ZONETOOL_ERROR("havok: HavokPhysicsAsset is %zu bytes, expected 144",
								buf.size());
							return {};
						}

						const std::uint32_t lookup_value[8] = {
							physics_asset->body_quality_crc, physics_asset->material_crc,
							0, 0, 0, 0, 0, 0
						};
						for (auto i = 0; i < 8; i++)
						{
							if (!lookup_count[i]) continue;
							local_fixups.push_back({lookup_field[i], buf.size()});
							buf.write<std::uint32_t>(lookup_value[i]);
							align16();
						}

						const auto system_data_offset = buf.size();
						global_fixups.push_back({static_cast<int>(system_data_slot), 2,
							static_cast<int>(system_data_offset)});
						virtual_fixups.emplace_back(system_data_offset, 1);

						buf.reserve(16);
						write_hk_array_header(buf, 0);
						write_hk_array_header(buf, 0);
						write_hk_array_header(buf, 0);
						const auto body_cinfos_field = buf.size();
						write_hk_array_header(buf, 1);
						write_hk_array_header(buf, 0);
						const auto referenced_field = buf.size();
						write_hk_array_header(buf, 1);
						const auto system_name_slot = buf.reserve(8);
						if (buf.size() - system_data_offset != 120)
						{
							ZONETOOL_ERROR("havok: hknpPhysicsSystemData is %zu bytes, expected 120",
								buf.size() - system_data_offset);
							return {};
						}
						align16();

						const auto body_offset = buf.size();
						local_fixups.push_back({body_cinfos_field, body_offset});
						const auto body_shape_slot = buf.reserve(8);
						buf.write<std::int32_t>(0);
						buf.write<std::uint32_t>(physics_asset->body_contents);
						buf.write<std::uint16_t>(0xFFFF);
						buf.write<std::uint8_t>(0xFF);
						buf.reserve(5);
						buf.write<std::uint64_t>(0);
						const auto body_name_slot = buf.reserve(8);
						buf.write<std::uint8_t>(0);
						buf.reserve(7);
						for (auto i = 0; i < 2; i++)
						{
							buf.write<float>(0.0f); buf.write<float>(0.0f);
							buf.write<float>(0.0f); buf.write<float>(1.0f);
						}
						buf.reserve(32);
						buf.write<float>(-1.0f);
						buf.reserve(12);
						buf.write<std::uint16_t>(0xFFFF);
						buf.write<std::uint16_t>(0x0000);
						buf.write<std::uint32_t>(0x7FFFFFFFu);
						buf.write<std::uint32_t>(0x7FFFFFFFu);
						buf.write<std::uint32_t>(0);
						buf.reserve(16);
						if (buf.size() - body_offset != 160)
						{
							ZONETOOL_ERROR("havok: hknpBodyCinfo is %zu bytes, expected 160",
								buf.size() - body_offset);
							return {};
						}

						local_fixups.push_back({body_name_slot, buf.size()});
						buf.write(physics_asset->body_name.c_str(),
							physics_asset->body_name.size() + 1);
						align16();

						local_fixups.push_back({referenced_field, buf.size()});
						const auto referenced_slot = buf.reserve(8);
						align16();

						local_fixups.push_back({system_name_slot, buf.size()});
						const char* system_name = "Default Physics System Data";
						buf.write(system_name, std::strlen(system_name) + 1);
						align16();

						shape_ptr_slots[0].push_back(body_shape_slot);
						shape_ptr_slots[0].push_back(referenced_slot);
					}

					if (kind == mesh_blob_kind::world)
					{
						virtual_fixups.emplace_back(shape_list_offset, 0);
					}

					for (auto m = 0u; m < prepared.size(); m++)
					{
						const auto shape_offset = emit_mesh(prepared[m], vertex_format, kind != mesh_blob_kind::physics_asset, buf,
							fixups, class_shape, class_shape_data);
						if (!shape_offset)
						{
							return {};
						}
						for (const auto slot : shape_ptr_slots[m])
						{
							global_fixups.push_back({static_cast<int>(slot), 2,
								static_cast<int>(shape_offset)});
						}
					}

					const auto data_size = buf.size();

					byte_buffer names;
					const auto write_name = [&](const std::uint32_t sig, const char* name)
					{
						names.write<std::uint32_t>(sig);
						names.write<std::uint8_t>(0x09);
						const auto offset = names.size();
						names.write(name, std::strlen(name) + 1);
						return offset;
					};

					write_name(SIG_HK_CLASS, "hkClass");
					write_name(SIG_HK_CLASS_MEMBER, "hkClassMember");
					write_name(SIG_HK_CLASS_ENUM, "hkClassEnum");
					write_name(SIG_HK_CLASS_ENUM_ITEM, "hkClassEnumItem");

					std::array<std::size_t, 4> name_offsets{};
					if (kind == mesh_blob_kind::xmodel_lod)
					{
						name_offsets[0] = write_name(SIG_XMODEL_LOD, "HavokPhysicsXModelLOD");
						name_offsets[1] = write_name(SIG_COMPRESSED_MESH_SHAPE,
							"hknpCompressedMeshShape");
						name_offsets[2] = write_name(SIG_COMPRESSED_MESH_SHAPE_DATA,
							"hknpCompressedMeshShapeData");
					}
					else if (kind == mesh_blob_kind::world)
					{
						name_offsets[0] = write_name(SIG_SHAPE_LIST, "HavokPhysicsShapeList");
						name_offsets[1] = write_name(SIG_COMPRESSED_MESH_SHAPE,
							"hknpCompressedMeshShape");
						name_offsets[2] = write_name(SIG_COMPRESSED_MESH_SHAPE_DATA,
							"hknpCompressedMeshShapeData");
					}
					else
					{
						name_offsets[0] = write_name(SIG_PHYSICS_ASSET, "HavokPhysicsAsset");
						name_offsets[1] = write_name(SIG_PHYSICS_SYSTEM_DATA,
							"hknpPhysicsSystemData");
						name_offsets[2] = write_name(SIG_COMPRESSED_MESH_SHAPE,
							"hknpCompressedMeshShape");
						name_offsets[3] = write_name(SIG_COMPRESSED_MESH_SHAPE_DATA,
							"hknpCompressedMeshShapeData");
					}
					names.align(16, 0xFF);

					std::sort(local_fixups.begin(), local_fixups.end(),
						[](const pending_fixup& a, const pending_fixup& b)
						{
							return a.dst < b.dst;
						});

					byte_buffer fixup_bytes;
					for (const auto& fixup : local_fixups)
					{
						fixup_bytes.write<std::int32_t>(static_cast<std::int32_t>(fixup.src));
						fixup_bytes.write<std::int32_t>(static_cast<std::int32_t>(fixup.dst));
					}
					fixup_bytes.align(16, 0xFF);
					const auto local_size = fixup_bytes.size();

					std::sort(global_fixups.begin(), global_fixups.end(),
						[](const std::array<int, 3>& a, const std::array<int, 3>& b)
						{
							return a[0] < b[0];
						});
					for (const auto& fixup : global_fixups)
					{
						fixup_bytes.write<std::int32_t>(fixup[0]);
						fixup_bytes.write<std::int32_t>(fixup[1]);
						fixup_bytes.write<std::int32_t>(fixup[2]);
					}
					fixup_bytes.align(16, 0xFF);
					const auto global_size = fixup_bytes.size() - local_size;

					for (const auto& fixup : virtual_fixups)
					{
						fixup_bytes.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
						fixup_bytes.write<std::int32_t>(0);
						fixup_bytes.write<std::int32_t>(static_cast<std::int32_t>(name_offsets[fixup.second]));
					}
					fixup_bytes.align(16, 0xFF);
					const auto virtual_size = fixup_bytes.size() - local_size - global_size;

					byte_buffer file;
					file.write<std::uint32_t>(HK_MAGIC0);
					file.write<std::uint32_t>(HK_MAGIC1);
					file.write<std::int32_t>(0);
					file.write<std::int32_t>(HK_FILE_VERSION);
					file.write<std::uint8_t>(8);
					file.write<std::uint8_t>(1);
					file.write<std::uint8_t>(0);
					file.write<std::uint8_t>(1);
					file.write<std::int32_t>(3);
					file.write<std::int32_t>(2);
					file.write<std::int32_t>(0);
					file.write<std::int32_t>(0);
					file.write<std::int32_t>(static_cast<std::int32_t>(name_offsets[0]));
					const char* version = "hk_2014.2.5-r1";
					const auto version_length = std::strlen(version) + 1;
					file.write(version, version_length);
					file.fill(16 - version_length, 0xFF);
					file.write<std::int32_t>(0);
					file.write<std::uint16_t>(HK_MAX_PREDICATE);
					file.write<std::uint16_t>(0);

					const auto names_start = HK_HEADER_SIZE + 3 * HK_SECTION_HEADER_SIZE;
					const auto data_start = names_start + names.size();

					const auto write_section_header = [&](const char* tag, const std::size_t abs,
						const std::size_t payload, const std::size_t local, const std::size_t global,
						const std::size_t virt)
					{
						char name[19] = {};
						std::strncpy(name, tag, sizeof(name));
						file.write(name, sizeof(name));
						file.write<std::uint8_t>(0xFF);
						file.write<std::int32_t>(static_cast<std::int32_t>(abs));
						file.write<std::int32_t>(static_cast<std::int32_t>(payload));
						file.write<std::int32_t>(static_cast<std::int32_t>(payload + local));
						file.write<std::int32_t>(static_cast<std::int32_t>(payload + local + global));
						const auto end = payload + local + global + virt;
						file.write<std::int32_t>(static_cast<std::int32_t>(end));
						file.write<std::int32_t>(static_cast<std::int32_t>(end));
						file.write<std::int32_t>(static_cast<std::int32_t>(end));
						file.fill(16, 0xFF);
					};

					write_section_header("__classnames__", names_start, names.size(), 0, 0, 0);
					write_section_header("__types__", data_start, 0, 0, 0, 0);
					write_section_header("__data__", data_start, data_size, local_size, global_size,
						virtual_size);

					file.write(names.data.data(), names.size());
					file.write(buf.data.data(), buf.size());
					file.write(fixup_bytes.data.data(), fixup_bytes.size());

					for (auto i = 0u; i < palette.size(); i++)
					{
						ZONETOOL_INFO("  tag %2u  filter 0x%08X (raw 0x%08X)  crc 0x%08X  "
							"userData 0x%016llX", i,
							tag_records[i].collision_filter,
							static_cast<std::uint32_t>(std::get<0>(palette[i])),
							tag_records[i].material_crc,
							static_cast<unsigned long long>(tag_records[i].user_data));
					}

					std::size_t input_triangles = 0;
					auto mesh_triangles = 0;
					auto section_count = 0;
					for (auto m = 0u; m < prepared.size(); m++)
					{
						input_triangles += meshes[m]->triangles.size();
						mesh_triangles += prepared[m].mesh_triangles;
						section_count += static_cast<int>(prepared[m].sections.size());
					}
					ZONETOOL_INFO("havok: %s built -- %zu input triangles (%d mesh triangles), "
						"%zu convex custom primitives, %d sections, %d surface tags, "
						"contents 0x%08X, %zu bytes",
						kind == mesh_blob_kind::physics_asset ? "model physics asset" : "world shape",
						input_triangles, mesh_triangles, total_customs, section_count,
						tag_count, world_contents, file.size());

					if (out_tags)
					{
						*out_tags = tag_records;
					}

					return file.data;
				}
			}

			std::vector<std::uint8_t> build_world_shape(const mesh_input& input,
				std::vector<shape_tag>* out_tags)
			{
				return build_mesh_blob(mesh_blob_kind::world, {&input}, {}, nullptr, out_tags);
			}

			std::vector<std::uint8_t> build_model_physics_asset(const mesh_input& input,
				const physics_asset_input& physics_asset)
			{
				return build_mesh_blob(mesh_blob_kind::physics_asset, {&input}, {}, &physics_asset, nullptr);
			}

			std::vector<std::uint8_t> build_model_physics_lod(const mesh_input& input,
				const std::string& bone_name)
			{
				return build_mesh_blob(mesh_blob_kind::xmodel_lod, {&input}, {bone_name}, nullptr, nullptr);
			}

			std::vector<std::uint8_t> build_model_physics_lod(const std::vector<lod_shape>& shapes)
			{
				std::vector<const mesh_input*> meshes;
				std::vector<std::string> bones;
				for (const auto& shape : shapes)
				{
					meshes.push_back(&shape.mesh);
					bones.push_back(shape.bone_name);
				}
				return build_mesh_blob(mesh_blob_kind::xmodel_lod, meshes, bones, nullptr, nullptr);
			}

			std::vector<std::uint8_t> build_ents_shape_list(const ents_input& input,
				ents_tag_merge* out_merge)
			{
				struct half_edge
				{
					std::uint16_t face = 0;
					std::uint8_t edge = 0;
				};

				struct tree_node
				{
					float mn[3] = {0.0f, 0.0f, 0.0f};
					float mx[3] = {0.0f, 0.0f, 0.0f};
					std::uint16_t parent = 0;
					std::uint32_t data = 0;
				};

				byte_buffer buf;
				std::vector<std::pair<std::size_t, std::size_t>> local_fixups;
				std::vector<std::pair<std::size_t, std::size_t>> global_fixups;
				std::vector<std::pair<std::size_t, int>> virtual_fixups;

				const auto align16 = [&] { buf.align(16, 0); };

				std::vector<const ents_shape*> shapes;
				for (const auto& shape : input.shapes)
				{
					if (shape.convexes.empty())
					{
						ZONETOOL_ERROR("havok: ents shape %zu (\"%s\") has no convexes -- "
							"writing it would shift every later shape index",
							shapes.size(), shape.name.c_str());
						return {};
					}
					shapes.emplace_back(&shape);
				}

				const auto shape_count = static_cast<int>(shapes.size());

				std::vector<shape_tag> palette = input.world_tags;
				const auto prefix_size = palette.size();
				std::vector<bool> prefix_used(prefix_size, false);
				const auto tag_for = [&](const ents_shape& shape)
				{
					const shape_tag key{
						filter_contents(static_cast<std::uint32_t>(shape.contents)),
						shape.material_crc, shape.user_data};
					for (auto i = 0u; i < palette.size(); i++)
					{
						if (palette[i] == key)
						{
							if (i < prefix_size)
							{
								prefix_used[i] = true;
							}
							return static_cast<std::uint16_t>(i);
						}
					}
					palette.emplace_back(key);
					return static_cast<std::uint16_t>(palette.size() - 1);
				};

				for (const auto* shape : shapes)
				{
					tag_for(*shape);
				}

				if (palette.empty())
				{
					palette.emplace_back(shape_tag{filter_contents(1), DEFAULT_MATERIAL_CRC, 0});
				}

				const std::array<int, 10> member_counts = {
					shape_count, shape_count, shape_count, shape_count, shape_count,
					shape_count * 2, 0, static_cast<int>(palette.size()), shape_count,
					shape_count
				};

				std::array<std::size_t, 10> member_field{};
				for (auto i = 0; i < 6; i++)
				{
					member_field[i] = buf.size();
					write_hk_array_header(buf, member_counts[i]);
				}
				buf.write<std::int32_t>(0);
				buf.reserve(4);
				for (auto i = 7; i < 10; i++)
				{
					member_field[i] = buf.size();
					write_hk_array_header(buf, member_counts[i]);
				}

				if (buf.size() != 152)
				{
					ZONETOOL_ERROR("havok: ents HavokPhysicsShapeList is %zu bytes, expected 152",
						buf.size());
					return {};
				}

				virtual_fixups.emplace_back(static_cast<std::size_t>(0), 0);
				align16();

				const auto record_array = [&](const int index, const std::size_t payload)
				{
					if (member_counts[index] > 0)
					{
						local_fixups.emplace_back(member_field[index], payload);
					}
				};

				std::vector<std::size_t> shape_ptr_slots;
				record_array(0, buf.size());
				for (auto i = 0; i < shape_count; i++)
				{
					shape_ptr_slots.emplace_back(buf.reserve(8));
				}
				align16();

				record_array(1, buf.size());
				for (auto i = 0; i < shape_count; i++)
				{
					buf.write<std::int32_t>(i);
				}
				align16();

				record_array(2, buf.size());
				std::vector<std::size_t> name_ptr_slots;
				for (auto i = 0; i < shape_count; i++)
				{
					name_ptr_slots.emplace_back(buf.reserve(8));
				}
				align16();
				for (auto i = 0; i < shape_count; i++)
				{
					local_fixups.emplace_back(name_ptr_slots[i], buf.size());
					const auto& name = shapes[i]->name;
					buf.write(name.c_str(), name.size() + 1);

					if (buf.size() & 1)
					{
						buf.write<std::uint8_t>(0);
					}
				}
				align16();

				record_array(3, buf.size());
				for (const auto* shape : shapes)
				{
					auto verts = 0;
					for (const auto& convex : shape->convexes)
					{
						verts += static_cast<int>(padded_vertex_count(convex.verts.size()));
					}
					buf.write<std::int32_t>(verts);
				}
				align16();

				record_array(4, buf.size());
				for (auto i = 0; i < shape_count; i++)
				{
					buf.write<std::int32_t>(0);
				}
				align16();

				record_array(5, buf.size());
				buf.reserve(static_cast<std::size_t>(shape_count) * 32);
				align16();

				record_array(7, buf.size());
				for (const auto& entry : palette)
				{
					buf.write<std::uint32_t>(entry.collision_filter);
					buf.write<std::uint32_t>(entry.material_crc);
					buf.write<std::uint16_t>(0xFFFF);
					buf.reserve(6);
					buf.write<std::uint64_t>(entry.user_data);
				}
				align16();

				record_array(8, buf.size());
				for (const auto* shape : shapes)
				{
					buf.write<std::uint32_t>(
						filter_contents(shape->entity_contents));
				}
				align16();

				record_array(9, buf.size());
				for (const auto* shape : shapes)
				{
					buf.write<std::int32_t>(static_cast<std::int32_t>(shape->convexes.size()));
				}
				align16();

				for (auto s = 0; s < shape_count; s++)
				{
					const auto& shape = *shapes[s];
					const auto instance_count = static_cast<int>(shape.convexes.size());

					const auto compound_offset = buf.size();
					global_fixups.emplace_back(shape_ptr_slots[s], compound_offset);
					virtual_fixups.emplace_back(compound_offset, 1);

					buf.reserve(16);
					buf.write<std::uint16_t>(static_cast<std::uint16_t>(COMPOUND_SHAPE_FLAGS));
					auto key_bits = 0;
					for (auto n = instance_count; n > 0; n >>= 1)
					{
						key_bits++;
					}
					buf.write<std::uint8_t>(static_cast<std::uint8_t>(key_bits));
					buf.write<std::uint8_t>(static_cast<std::uint8_t>(COMPOUND_DISPATCH_TYPE));
					buf.write<float>(0.0f);
					buf.write<std::uint64_t>(0);
					buf.reserve(8);
					buf.reserve(8);
					buf.write<std::uint32_t>(0xFFFFFFFFu);
					buf.write<std::uint32_t>(0);
					write_hk_array_header(buf, 0);
					write_hk_array_header(buf, 0);
					buf.write<std::uint32_t>(0xFFFFFFFFu);
					buf.reserve(4);

					if (buf.size() - compound_offset != 96)
					{
						ZONETOOL_ERROR("havok: compound header is %zu bytes, expected 96",
							buf.size() - compound_offset);
						return {};
					}

					const auto instances_field = buf.size();
					write_hk_array_header(buf, instance_count);
					buf.write<std::int32_t>(-1);
					buf.reserve(4);
					buf.reserve(8);

					const auto aabb_slot = buf.reserve(32);
					buf.write<std::uint8_t>(1);
					buf.reserve(7);
					buf.reserve(16);
					buf.reserve(8);
					const auto bvd_slot = buf.reserve(8);
					buf.reserve(8);

					if (buf.size() - compound_offset != SIZEOF_DYNAMIC_COMPOUND_SHAPE)
					{
						ZONETOOL_ERROR("havok: hknpDynamicCompoundShape is %zu bytes, expected %d",
							buf.size() - compound_offset, SIZEOF_DYNAMIC_COMPOUND_SHAPE);
						return {};
					}

					local_fixups.emplace_back(instances_field, buf.size());
					std::vector<std::size_t> instance_shape_slots;
					std::vector<std::size_t> instance_offsets;
					for (auto i = 0; i < instance_count; i++)
					{
						const auto instance_offset = buf.size();
						instance_offsets.emplace_back(instance_offset);
						buf.write<float>(1.0f); buf.write<float>(0.0f);
						buf.write<float>(0.0f); buf.write<std::uint32_t>(SHAPE_INSTANCE_FLAGS_W);
						buf.write<float>(0.0f); buf.write<float>(1.0f);
						buf.write<float>(0.0f); buf.write<float>(0.0f);
						buf.write<float>(0.0f); buf.write<float>(0.0f);
						buf.write<float>(1.0f); buf.write<float>(0.0f);
						buf.write<float>(0.0f); buf.write<float>(0.0f);
						buf.write<float>(0.0f); buf.write<std::uint32_t>(INT24_W_BASE);
						for (auto c = 0; c < 4; c++)
						{
							buf.write<float>(1.0f);
						}
						instance_shape_slots.emplace_back(buf.reserve(8));
						buf.write<std::uint16_t>(tag_for(shape));
						buf.write<std::uint16_t>(0xFFFF);
						buf.reserve(36);

						if (buf.size() - instance_offset != SIZEOF_SHAPE_INSTANCE)
						{
							ZONETOOL_ERROR("havok: hknpShapeInstance is %zu bytes, expected %d",
								buf.size() - instance_offset, SIZEOF_SHAPE_INSTANCE);
							return {};
						}
					}
					align16();

					float shape_min[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
					float shape_max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
					std::vector<std::array<float, 6>> convex_bounds;

					for (auto c = 0; c < instance_count; c++)
					{
						const auto& convex = shape.convexes[c];

						auto index_total = 0u;
						for (const auto& face : convex.faces)
						{
							index_total += static_cast<unsigned int>(face.indices.size());
						}

						const auto vertex_count = padded_vertex_count(convex.verts.size());

						if (vertex_count > 255 || convex.faces.size() > 0xFFFF
							|| index_total > 0xFFFF)
						{
							ZONETOOL_ERROR("havok: convex %d of shape %d exceeds the format "
								"limits (%zu verts, %zu faces, %u indices)", c, s,
								convex.verts.size(), convex.faces.size(), index_total);
							return {};
						}

						const auto convex_offset = buf.size();
						global_fixups.emplace_back(instance_shape_slots[c], convex_offset);
						virtual_fixups.emplace_back(convex_offset, 2);

						buf.reserve(16);
						buf.write<std::uint16_t>(static_cast<std::uint16_t>(CONVEX_SHAPE_FLAGS));
						buf.write<std::uint8_t>(0);
						buf.write<std::uint8_t>(static_cast<std::uint8_t>(CONVEX_DISPATCH_TYPE));
						buf.write<float>(0.0f);
						buf.write<std::uint64_t>(0);
						buf.reserve(8);
						buf.reserve(8);
						const auto vertices_field = buf.size();
						buf.reserve(4);
						buf.reserve(12);
						const auto planes_field = buf.size();
						buf.reserve(4);
						const auto faces_field = buf.size();
						buf.reserve(4);
						const auto indices_field = buf.size();
						buf.reserve(4);
						buf.reserve(4);
						const auto connectivity_slot = buf.reserve(8);
						buf.reserve(8);

						if (buf.size() - convex_offset != SIZEOF_CONVEX_POLYTOPE_SHAPE)
						{
							ZONETOOL_ERROR("havok: hknpConvexPolytopeShape is %zu bytes, "
								"expected %d", buf.size() - convex_offset,
								SIZEOF_CONVEX_POLYTOPE_SHAPE);
							return {};
						}

						std::vector<half_edge> vertex_edges(vertex_count);
						std::vector<bool> vertex_seen(vertex_count, false);
						std::vector<half_edge> face_links(index_total);
						std::map<std::pair<std::uint8_t, std::uint8_t>, half_edge> directed;

						for (auto f = 0u; f < convex.faces.size(); f++)
						{
							const auto& face = convex.faces[f];
							for (auto e = 0u; e < face.indices.size(); e++)
							{
								const auto from = face.indices[e];
								const auto to = face.indices[(e + 1) % face.indices.size()];
								const half_edge self{
									static_cast<std::uint16_t>(f),
									static_cast<std::uint8_t>(e)
								};

								if (from < vertex_seen.size() && !vertex_seen[from])
								{
									vertex_seen[from] = true;
									vertex_edges[from] = self;
								}

								directed[{from, to}] = self;
							}
						}

						auto global = 0u;
						auto unmatched = 0;
						for (auto f = 0u; f < convex.faces.size(); f++)
						{
							const auto& face = convex.faces[f];
							for (auto e = 0u; e < face.indices.size(); e++)
							{
								const auto from = face.indices[e];
								const auto to = face.indices[(e + 1) % face.indices.size()];
								const auto twin = directed.find({to, from});
								if (twin != directed.end())
								{
									face_links[global] = twin->second;
								}
								else
								{
									face_links[global] = {
										static_cast<std::uint16_t>(f),
										static_cast<std::uint8_t>(e)
									};
									unmatched++;
								}
								global++;
							}
						}

						if (unmatched)
						{
							ZONETOOL_WARNING("havok: convex %d of shape %d has %d unpaired "
								"edge(s) -- hull is not closed", c, s, unmatched);
						}

						const auto write_rel_array = [&](const std::size_t field,
							const std::size_t count)
						{
							buf.patch<std::uint16_t>(field, static_cast<std::uint16_t>(count));
							buf.patch<std::uint16_t>(field + 2,
								static_cast<std::uint16_t>(buf.size() - field));
						};

						float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
						float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

						write_rel_array(vertices_field, vertex_count);
						for (auto j = 0u; j < vertex_count; j++)
						{
							const auto source = std::min<std::size_t>(
								j, convex.verts.size() - 1);
							const auto& v = convex.verts[source];
							for (auto k = 0; k < 3; k++)
							{
								const auto value = v[k] * input.scale;
								buf.write<float>(value);
								mn[k] = std::min(mn[k], value);
								mx[k] = std::max(mx[k], value);
							}
							buf.write<std::uint32_t>(
								0x3F000000u | static_cast<std::uint32_t>(source));
						}
						align16();

						write_rel_array(planes_field, convex.faces.size());
						for (const auto& face : convex.faces)
						{
							for (auto k = 0; k < 3; k++)
							{
								buf.write<float>(face.plane[k]);
							}
							buf.write<float>(-face.plane[3] * input.scale);
						}
						align16();

						write_rel_array(faces_field, convex.faces.size());
						auto first_index = 0u;
						for (auto f = 0u; f < convex.faces.size(); f++)
						{
							const auto& face = convex.faces[f];
							buf.write<std::uint16_t>(static_cast<std::uint16_t>(first_index));
							buf.write<std::uint8_t>(
								static_cast<std::uint8_t>(face.indices.size()));

							auto smallest = 45.0f;
							for (auto e = 0u; e < face.indices.size(); e++)
							{
								const auto& twin = face_links[first_index + e];
								if (twin.face == f || twin.face >= convex.faces.size())
								{
									continue;
								}

								const auto& other = convex.faces[twin.face].plane;
								auto dot = face.plane[0] * other[0] + face.plane[1] * other[1]
									+ face.plane[2] * other[2];
								dot = std::max(-1.0f, std::min(1.0f, dot));

								const auto dihedral = 180.0f
									- static_cast<float>(std::acos(dot) * 57.29577951308232);
								smallest = std::min(smallest, dihedral * 0.5f);
							}

							const auto half_radians = smallest * 0.017453292519943295f;
							auto quantised = static_cast<int>(
								(half_radians - 1.1920929e-7f) * 41720.875f + 0.5f);
							quantised = std::max(0, std::min(65535, quantised)) >> 8;
							buf.write<std::uint8_t>(static_cast<std::uint8_t>(quantised));

							first_index += static_cast<unsigned int>(face.indices.size());
						}
						align16();

						write_rel_array(indices_field, index_total);
						for (const auto& face : convex.faces)
						{
							for (const auto index : face.indices)
							{
								buf.write<std::uint8_t>(index);
							}
						}
						align16();

						for (auto j = convex.verts.size(); j < vertex_count; j++)
						{
							vertex_edges[j] = vertex_edges[convex.verts.size() - 1];
						}

						const auto connectivity_offset = buf.size();
						global_fixups.emplace_back(connectivity_slot, connectivity_offset);
						virtual_fixups.emplace_back(connectivity_offset, 3);

						buf.reserve(16);
						const auto vertex_edges_field = buf.size();
						write_hk_array_header(buf, static_cast<int>(vertex_edges.size()));
						const auto face_links_field = buf.size();
						write_hk_array_header(buf, static_cast<int>(face_links.size()));

						if (buf.size() - connectivity_offset
							!= SIZEOF_CONVEX_POLYTOPE_CONNECTIVITY)
						{
							ZONETOOL_ERROR("havok: connectivity is %zu bytes, expected %d",
								buf.size() - connectivity_offset,
								SIZEOF_CONVEX_POLYTOPE_CONNECTIVITY);
							return {};
						}

						const auto write_edges = [&](const std::size_t field,
							const std::vector<half_edge>& edges)
						{
							if (edges.empty())
							{
								return;
							}
							local_fixups.emplace_back(field, buf.size());
							for (const auto& edge : edges)
							{
								buf.write<std::uint16_t>(edge.face);
								buf.write<std::uint8_t>(edge.edge);
								buf.write<std::uint8_t>(0);
							}
							align16();
						};

						write_edges(vertex_edges_field, vertex_edges);
						write_edges(face_links_field, face_links);

						convex_bounds.push_back({mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]});
						for (auto k = 0; k < 3; k++)
						{
							shape_min[k] = std::min(shape_min[k], mn[k]);
							shape_max[k] = std::max(shape_max[k], mx[k]);
						}
					}

					for (auto k = 0; k < 3; k++)
					{
						buf.patch<float>(aabb_slot + k * 4, shape_min[k]);
						buf.patch<float>(aabb_slot + 16 + k * 4, shape_max[k]);
					}
					buf.patch<std::uint32_t>(aabb_slot + 12, INT24_W_BASE);
					buf.patch<float>(aabb_slot + 28, 0.0f);

					const auto bvd_offset = buf.size();
					global_fixups.emplace_back(bvd_slot, bvd_offset);
					virtual_fixups.emplace_back(bvd_offset, 4);

					buf.reserve(16);
					const auto nodes_field = buf.size();
					const auto node_count = 2 * instance_count + 1;
					write_hk_array_header(buf, node_count);
					buf.write<std::int32_t>(2 * instance_count);
					buf.reserve(4);
					buf.write<std::int32_t>(instance_count);
					buf.reserve(4);
					buf.write<std::int32_t>(1);
					buf.reserve(4);

					if (buf.size() - bvd_offset != SIZEOF_DYNAMIC_COMPOUND_SHAPE_DATA)
					{
						ZONETOOL_ERROR("havok: hknpDynamicCompoundShapeData is %zu bytes, "
							"expected %d", buf.size() - bvd_offset,
							SIZEOF_DYNAMIC_COMPOUND_SHAPE_DATA);
						return {};
					}

					align16();

					std::vector<tree_node> nodes(node_count);
					auto next_node = 1;

					std::vector<int> order(instance_count);
					for (auto i = 0; i < instance_count; i++)
					{
						order[i] = i;
					}

					std::function<int(int, int, std::uint16_t)> build_tree =
						[&](const int first, const int count, const std::uint16_t parent) -> int
					{
						const auto self = next_node++;
						auto& node = nodes[self];
						node.parent = parent;

						for (auto k = 0; k < 3; k++)
						{
							node.mn[k] = FLT_MAX;
							node.mx[k] = -FLT_MAX;
						}
						for (auto i = 0; i < count; i++)
						{
							const auto& b = convex_bounds[order[first + i]];
							for (auto k = 0; k < 3; k++)
							{
								node.mn[k] = std::min(node.mn[k], b[k]);
								node.mx[k] = std::max(node.mx[k], b[k + 3]);
							}
						}

						if (count == 1)
						{
							node.data = static_cast<std::uint32_t>(order[first]) << 16;
							return self;
						}

						auto axis = 0;
						auto widest = node.mx[0] - node.mn[0];
						for (auto k = 1; k < 3; k++)
						{
							if (node.mx[k] - node.mn[k] > widest)
							{
								widest = node.mx[k] - node.mn[k];
								axis = k;
							}
						}

						const auto centre = [&](const int index)
						{
							const auto& b = convex_bounds[index];
							return (b[axis] + b[axis + 3]) * 0.5f;
						};
						std::sort(order.begin() + first, order.begin() + first + count,
							[&](const int a, const int b) { return centre(a) < centre(b); });

						const auto half = count / 2;
						const auto left = build_tree(first, half,
							static_cast<std::uint16_t>(self));
						const auto right = build_tree(first + half, count - half,
							static_cast<std::uint16_t>(self));
						nodes[self].data = static_cast<std::uint32_t>(left)
							| (static_cast<std::uint32_t>(right) << 16);
						return self;
					};

					if (instance_count > 0)
					{
						build_tree(0, instance_count, 0);
					}

					for (auto n = 1; n < node_count - 1; n++)
					{
						const auto& node = nodes[n];
						if ((node.data & 0xFFFF) != 0)
						{
							continue;
						}
						const auto instance = static_cast<int>(node.data >> 16);
						if (instance < instance_count)
						{
							buf.patch<std::uint32_t>(instance_offsets[instance] + 48 + 12,
								INT24_W_BASE | static_cast<std::uint32_t>(n));
						}
					}

					local_fixups.emplace_back(nodes_field, buf.size());
					for (auto n = 0; n < node_count; n++)
					{
						const auto& node = nodes[n];
						const auto used = (n != 0 && n != node_count - 1);
						for (auto k = 0; k < 3; k++)
						{
							buf.write<float>(used ? node.mn[k] : 0.0f);
						}
						buf.write<std::uint16_t>(used ? node.parent : 0);
						buf.write<std::uint16_t>(used ? 0x3F00 : 0);
						for (auto k = 0; k < 3; k++)
						{
							buf.write<float>(used ? node.mx[k] : 0.0f);
						}
						buf.write<std::uint32_t>(used ? node.data : 0);
					}
					align16();
				}

				const auto data_size = buf.size();

				byte_buffer names;
				const auto write_name = [&](const std::uint32_t sig, const char* name)
				{
					names.write<std::uint32_t>(sig);
					names.write<std::uint8_t>(0x09);
					const auto offset = names.size();
					names.write(name, std::strlen(name) + 1);
					return offset;
				};

				write_name(SIG_HK_CLASS, "hkClass");
				write_name(SIG_HK_CLASS_MEMBER, "hkClassMember");
				write_name(SIG_HK_CLASS_ENUM, "hkClassEnum");
				write_name(SIG_HK_CLASS_ENUM_ITEM, "hkClassEnumItem");

				std::array<std::size_t, 5> name_offsets{};
				name_offsets[0] = write_name(SIG_SHAPE_LIST, "HavokPhysicsShapeList");
				if (shape_count > 0)
				{
					name_offsets[1] = write_name(SIG_DYNAMIC_COMPOUND_SHAPE,
						"hknpDynamicCompoundShape");
					name_offsets[2] = write_name(SIG_CONVEX_POLYTOPE_SHAPE,
						"hknpConvexPolytopeShape");
					name_offsets[3] = write_name(SIG_CONVEX_POLYTOPE_CONNECTIVITY,
						"hknpConvexPolytopeShapeConnectivity");
					name_offsets[4] = write_name(SIG_DYNAMIC_COMPOUND_SHAPE_DATA,
						"hknpDynamicCompoundShapeData");
				}
				names.align(16, 0xFF);

				std::sort(local_fixups.begin(), local_fixups.end(),
					[](const std::pair<std::size_t, std::size_t>& a,
						const std::pair<std::size_t, std::size_t>& b)
					{
						return a.second < b.second;
					});

				byte_buffer fixups;
				for (const auto& fixup : local_fixups)
				{
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.second));
				}
				fixups.align(16, 0xFF);
				const auto local_size = fixups.size();

				std::sort(global_fixups.begin(), global_fixups.end());
				for (const auto& fixup : global_fixups)
				{
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
					fixups.write<std::int32_t>(2);
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.second));
				}
				fixups.align(16, 0xFF);
				const auto global_size = fixups.size() - local_size;

				std::sort(virtual_fixups.begin(), virtual_fixups.end());
				for (const auto& fixup : virtual_fixups)
				{
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
					fixups.write<std::int32_t>(0);
					fixups.write<std::int32_t>(
						static_cast<std::int32_t>(name_offsets[fixup.second]));
				}
				fixups.align(16, 0xFF);
				const auto virtual_size = fixups.size() - local_size - global_size;

				byte_buffer file;
				file.write<std::uint32_t>(HK_MAGIC0);
				file.write<std::uint32_t>(HK_MAGIC1);
				file.write<std::int32_t>(0);
				file.write<std::int32_t>(HK_FILE_VERSION);
				file.write<std::uint8_t>(8);
				file.write<std::uint8_t>(1);
				file.write<std::uint8_t>(0);
				file.write<std::uint8_t>(1);
				file.write<std::int32_t>(3);
				file.write<std::int32_t>(2);
				file.write<std::int32_t>(0);
				file.write<std::int32_t>(0);
				file.write<std::int32_t>(static_cast<std::int32_t>(name_offsets[0]));
				const char* version = "hk_2014.2.5-r1";
				const auto version_length = std::strlen(version) + 1;
				file.write(version, version_length);
				file.fill(16 - version_length, 0xFF);
				file.write<std::int32_t>(0);
				file.write<std::uint16_t>(HK_MAX_PREDICATE);
				file.write<std::uint16_t>(0);

				const auto names_start = HK_HEADER_SIZE + 3 * HK_SECTION_HEADER_SIZE;
				const auto data_start = names_start + names.size();

				const auto write_section_header = [&](const char* tag, const std::size_t abs,
					const std::size_t payload, const std::size_t local, const std::size_t global,
					const std::size_t virt)
				{
					char name[19] = {};
					std::strncpy(name, tag, sizeof(name));
					file.write(name, sizeof(name));
					file.write<std::uint8_t>(0xFF);
					file.write<std::int32_t>(static_cast<std::int32_t>(abs));
					file.write<std::int32_t>(static_cast<std::int32_t>(payload));
					file.write<std::int32_t>(static_cast<std::int32_t>(payload + local));
					file.write<std::int32_t>(static_cast<std::int32_t>(payload + local + global));
					const auto end = payload + local + global + virt;
					file.write<std::int32_t>(static_cast<std::int32_t>(end));
					file.write<std::int32_t>(static_cast<std::int32_t>(end));
					file.write<std::int32_t>(static_cast<std::int32_t>(end));
					file.fill(16, 0xFF);
				};

				write_section_header("__classnames__", names_start, names.size(), 0, 0, 0);
				write_section_header("__types__", data_start, 0, 0, 0, 0);
				write_section_header("__data__", data_start, data_size, local_size, global_size,
					virtual_size);

				file.write(names.data.data(), names.size());
				file.write(buf.data.data(), buf.size());
				file.write(fixups.data.data(), fixups.size());

				auto convex_total = 0;
				for (const auto* shape : shapes)
				{
					convex_total += static_cast<int>(shape->convexes.size());
				}

				std::size_t reused = 0;
				for (const auto used : prefix_used)
				{
					reused += used ? 1 : 0;
				}

				ZONETOOL_INFO("havok: ents shape list built -- %d shapes, %d convexes, "
					"%zu surface tags (%zu inherited from the world table, %zu of those in "
					"use, %zu appended), %zu bytes", shape_count, convex_total, palette.size(),
					prefix_size, reused, palette.size() - prefix_size, file.size());

				if (out_merge)
				{
					out_merge->prefix = prefix_size;
					out_merge->total = palette.size();
					out_merge->reused = reused;
					out_merge->appended = palette.size() - prefix_size;
				}

				return file.data;
			}

			std::vector<std::uint8_t> build_physics_asset(const physics_asset_input& input)
			{
				constexpr auto BOX_CONVEX_RADIUS = 0.01f;

				constexpr auto DUMMY_CONVEX_FLAGS = 0x01C3u;

				byte_buffer buf;
				std::vector<std::pair<std::size_t, std::size_t>> local_fixups;
				std::vector<std::pair<std::size_t, std::size_t>> global_fixups;
				std::vector<std::pair<std::size_t, int>> virtual_fixups;

				const auto align16 = [&] { buf.align(16, 0); };

				virtual_fixups.emplace_back(static_cast<std::size_t>(0), 0);
				buf.reserve(8);
				const auto system_data_slot = buf.reserve(8);
				std::array<std::size_t, 8> lookup_field{};
				const int lookup_count[8] = {1, 1, 0, 1, 1, 1, 0, 1};
				for (auto i = 0; i < 8; i++)
				{
					lookup_field[i] = buf.size();
					write_hk_array_header(buf, lookup_count[i]);
				}

				if (buf.size() != 144)
				{
					ZONETOOL_ERROR("havok: HavokPhysicsAsset is %zu bytes, expected 144",
						buf.size());
					return {};
				}

				const std::uint32_t lookup_value[8] = {
					input.body_quality_crc, input.material_crc, 0, 0, 0, 0, 0, 0
				};
				for (auto i = 0; i < 8; i++)
				{
					if (!lookup_count[i])
					{
						continue;
					}
					local_fixups.emplace_back(lookup_field[i], buf.size());
					buf.write<std::uint32_t>(lookup_value[i]);
					align16();
				}

				const auto system_data_offset = buf.size();
				global_fixups.emplace_back(system_data_slot, system_data_offset);
				virtual_fixups.emplace_back(system_data_offset, 1);

				buf.reserve(16);
				write_hk_array_header(buf, 0);
				write_hk_array_header(buf, 0);
				write_hk_array_header(buf, 0);
				const auto body_cinfos_field = buf.size();
				write_hk_array_header(buf, 1);
				write_hk_array_header(buf, 0);
				const auto referenced_field = buf.size();
				write_hk_array_header(buf, 1);
				const auto system_name_slot = buf.reserve(8);

				if (buf.size() - system_data_offset != 120)
				{
					ZONETOOL_ERROR("havok: hknpPhysicsSystemData is %zu bytes, expected 120",
						buf.size() - system_data_offset);
					return {};
				}

				align16();

				const auto body_offset = buf.size();
				local_fixups.emplace_back(body_cinfos_field, body_offset);

				const auto body_shape_slot = buf.reserve(8);
				buf.write<std::int32_t>(0);
				buf.write<std::uint32_t>(input.body_contents);
				buf.write<std::uint16_t>(0xFFFF);
				buf.write<std::uint8_t>(0xFF);
				buf.reserve(5);
				buf.write<std::uint64_t>(0);
				const auto body_name_slot = buf.reserve(8);
				buf.write<std::uint8_t>(0);
				buf.reserve(7);
				for (auto i = 0; i < 2; i++)
				{
					buf.write<float>(0.0f);
					buf.write<float>(0.0f);
					buf.write<float>(0.0f);
					buf.write<float>(1.0f);
				}
				buf.reserve(32);
				buf.write<float>(-1.0f);
				buf.reserve(12);
				buf.write<std::uint16_t>(0xFFFF);
				buf.write<std::uint16_t>(0x0000);
				buf.write<std::uint32_t>(0x7FFFFFFFu);
				buf.write<std::uint32_t>(0x7FFFFFFFu);
				buf.write<std::uint32_t>(0);
				buf.reserve(16);

				if (buf.size() - body_offset != 160)
				{
					ZONETOOL_ERROR("havok: hknpBodyCinfo is %zu bytes, expected 160",
						buf.size() - body_offset);
					return {};
				}

				local_fixups.emplace_back(body_name_slot, buf.size());
				buf.write(input.body_name.c_str(), input.body_name.size() + 1);
				align16();

				local_fixups.emplace_back(referenced_field, buf.size());
				const auto referenced_slot = buf.reserve(8);
				align16();

				local_fixups.emplace_back(system_name_slot, buf.size());
				const char* system_name = "Default Physics System Data";
				buf.write(system_name, std::strlen(system_name) + 1);
				align16();

				const auto convex_offset = buf.size();
				global_fixups.emplace_back(body_shape_slot, convex_offset);
				global_fixups.emplace_back(referenced_slot, convex_offset);
				virtual_fixups.emplace_back(convex_offset, 2);

				buf.reserve(16);
				buf.write<std::uint16_t>(static_cast<std::uint16_t>(DUMMY_CONVEX_FLAGS));
				buf.write<std::uint8_t>(0);
				buf.write<std::uint8_t>(static_cast<std::uint8_t>(CONVEX_DISPATCH_TYPE));
				buf.write<float>(BOX_CONVEX_RADIUS);
				buf.write<std::uint64_t>(0);
				const auto properties_slot = buf.reserve(8);
				buf.reserve(8);
				const auto vertices_field = buf.size();
				buf.reserve(4);
				buf.reserve(12);
				const auto planes_field = buf.size();
				buf.reserve(4);
				const auto faces_field = buf.size();
				buf.reserve(4);
				const auto indices_field = buf.size();
				buf.reserve(4);
				buf.reserve(4);
				buf.reserve(8);
				buf.reserve(8);

				if (buf.size() - convex_offset != SIZEOF_CONVEX_POLYTOPE_SHAPE)
				{
					ZONETOOL_ERROR("havok: dummy hknpConvexPolytopeShape is %zu bytes, "
						"expected %d", buf.size() - convex_offset,
						SIZEOF_CONVEX_POLYTOPE_SHAPE);
					return {};
				}

				const auto write_rel_array = [&](const std::size_t field,
					const std::size_t count)
				{
					buf.patch<std::uint16_t>(field, static_cast<std::uint16_t>(count));
					buf.patch<std::uint16_t>(field + 2,
						static_cast<std::uint16_t>(buf.size() - field));
				};

				static const std::uint32_t DUMMY_VERTS[32] = {
					0x3F1D701E, 0x3F456FFC, 0x3EEAE07E, 0x3F000000,
					0x3F1D701E, 0xBF456FFC, 0x3EEAE07E, 0x3F000001,
					0x3F1D701E, 0x3F456FFC, 0xBEEAE07E, 0x3F000002,
					0xBF1D701E, 0x3F456FFC, 0x3EEAE07E, 0x3F000003,
					0x3F1D701E, 0xBF456FFC, 0xBEEAE07E, 0x3F000004,
					0xBF1D701E, 0x3F456FFC, 0xBEEAE07E, 0x3F000005,
					0xBF1D701E, 0xBF456FFC, 0x3EEAE07E, 0x3F000006,
					0xBF1D701E, 0xBF456FFC, 0xBEEAE07E, 0x3F000007,
				};
				write_rel_array(vertices_field, 8);
				for (const auto word : DUMMY_VERTS)
				{
					buf.write<std::uint32_t>(word);
				}
				align16();

				static const std::uint32_t DUMMY_PLANES[24] = {
					0x00000000, 0x00000000, 0x3F800000, 0xBEEAE07E,
					0x80000000, 0x00000000, 0xBF800000, 0xBEEAE07E,
					0x3F800000, 0x00000000, 0x00000000, 0xBF1D701E,
					0xBF800000, 0x80000000, 0x00000000, 0xBF1D701E,
					0x00000000, 0x3F800000, 0x00000000, 0xBF456FFC,
					0x80000000, 0xBF800000, 0x00000000, 0xBF456FFC,
				};
				write_rel_array(planes_field, 6);
				for (const auto word : DUMMY_PLANES)
				{
					buf.write<std::uint32_t>(word);
				}
				align16();

				write_rel_array(faces_field, 6);
				for (auto i = 0; i < 6; i++)
				{
					buf.write<std::uint16_t>(static_cast<std::uint16_t>(i * 4));
					buf.write<std::uint8_t>(4);
					buf.write<std::uint8_t>(127);
				}
				align16();

				static const std::uint8_t FACE_INDICES[24] = {
					0, 3, 6, 1,  7, 5, 2, 4,  0, 1, 4, 2,
					3, 5, 7, 6,  5, 3, 0, 2,  7, 4, 1, 6,
				};
				write_rel_array(indices_field, 24);
				for (const auto index : FACE_INDICES)
				{
					buf.write<std::uint8_t>(index);
				}
				align16();

				const auto properties_offset = buf.size();
				global_fixups.emplace_back(properties_slot, properties_offset);
				virtual_fixups.emplace_back(properties_offset, 3);
				const auto entries_field = buf.size();
				write_hk_array_header(buf, 1);

				local_fixups.emplace_back(entries_field, buf.size());
				const auto mass_slot = buf.reserve(8);
				buf.write<std::uint16_t>(0xF100);
				buf.write<std::uint16_t>(0);
				buf.reserve(4);
				align16();

				const auto mass_offset = buf.size();
				global_fixups.emplace_back(mass_slot, mass_offset);
				virtual_fixups.emplace_back(mass_offset, 4);
				buf.reserve(16);
				static const std::uint8_t COMPRESSED_MASS[32] = {
					0x36, 0x5D, 0x00, 0x00, 0x00, 0x00, 0x80, 0x22,
					0x68, 0x42, 0xD4, 0x30, 0x14, 0x50, 0x00, 0x31,
					0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x30, 0xF5,
					0x00, 0x00, 0xA0, 0x40, 0xD1, 0x5D, 0xEA, 0x3F,
				};
				buf.write(COMPRESSED_MASS, sizeof(COMPRESSED_MASS));

				if (buf.size() - mass_offset != 48)
				{
					ZONETOOL_ERROR("havok: hknpShapeMassProperties is %zu bytes, expected 48",
						buf.size() - mass_offset);
					return {};
				}

				align16();
				const auto data_size = buf.size();

				byte_buffer names;
				const auto write_name = [&](const std::uint32_t sig, const char* name)
				{
					names.write<std::uint32_t>(sig);
					names.write<std::uint8_t>(0x09);
					const auto offset = names.size();
					names.write(name, std::strlen(name) + 1);
					return offset;
				};

				write_name(SIG_HK_CLASS, "hkClass");
				write_name(SIG_HK_CLASS_MEMBER, "hkClassMember");
				write_name(SIG_HK_CLASS_ENUM, "hkClassEnum");
				write_name(SIG_HK_CLASS_ENUM_ITEM, "hkClassEnumItem");

				std::array<std::size_t, 5> name_offsets{};
				name_offsets[0] = write_name(SIG_PHYSICS_ASSET, "HavokPhysicsAsset");
				name_offsets[1] = write_name(SIG_PHYSICS_SYSTEM_DATA, "hknpPhysicsSystemData");
				name_offsets[2] = write_name(SIG_CONVEX_POLYTOPE_SHAPE,
					"hknpConvexPolytopeShape");
				name_offsets[3] = write_name(SIG_REF_COUNTED_PROPERTIES,
					"hkRefCountedProperties");
				name_offsets[4] = write_name(SIG_SHAPE_MASS_PROPERTIES,
					"hknpShapeMassProperties");
				names.align(16, 0xFF);

				std::sort(local_fixups.begin(), local_fixups.end(),
					[](const std::pair<std::size_t, std::size_t>& a,
						const std::pair<std::size_t, std::size_t>& b)
					{
						return a.second < b.second;
					});
				byte_buffer fixups;
				for (const auto& fixup : local_fixups)
				{
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.second));
				}
				fixups.align(16, 0xFF);
				const auto local_size = fixups.size();

				std::sort(global_fixups.begin(), global_fixups.end());
				for (const auto& fixup : global_fixups)
				{
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
					fixups.write<std::int32_t>(2);
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.second));
				}
				fixups.align(16, 0xFF);
				const auto global_size = fixups.size() - local_size;

				std::sort(virtual_fixups.begin(), virtual_fixups.end());
				for (const auto& fixup : virtual_fixups)
				{
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
					fixups.write<std::int32_t>(0);
					fixups.write<std::int32_t>(
						static_cast<std::int32_t>(name_offsets[fixup.second]));
				}
				fixups.align(16, 0xFF);
				const auto virtual_size = fixups.size() - local_size - global_size;

				byte_buffer file;
				file.write<std::uint32_t>(HK_MAGIC0);
				file.write<std::uint32_t>(HK_MAGIC1);
				file.write<std::int32_t>(0);
				file.write<std::int32_t>(HK_FILE_VERSION);
				file.write<std::uint8_t>(8);
				file.write<std::uint8_t>(1);
				file.write<std::uint8_t>(0);
				file.write<std::uint8_t>(1);
				file.write<std::int32_t>(3);
				file.write<std::int32_t>(2);
				file.write<std::int32_t>(0);
				file.write<std::int32_t>(0);
				file.write<std::int32_t>(static_cast<std::int32_t>(name_offsets[0]));
				const char* version = "hk_2014.2.5-r1";
				const auto version_length = std::strlen(version) + 1;
				file.write(version, version_length);
				file.fill(16 - version_length, 0xFF);
				file.write<std::int32_t>(0);
				file.write<std::uint16_t>(HK_MAX_PREDICATE);
				file.write<std::uint16_t>(0);

				const auto names_start = HK_HEADER_SIZE + 3 * HK_SECTION_HEADER_SIZE;
				const auto data_start = names_start + names.size();

				const auto write_section_header = [&](const char* tag, const std::size_t abs,
					const std::size_t payload, const std::size_t local, const std::size_t global,
					const std::size_t virt)
				{
					char name[19] = {};
					std::strncpy(name, tag, sizeof(name));
					file.write(name, sizeof(name));
					file.write<std::uint8_t>(0xFF);
					file.write<std::int32_t>(static_cast<std::int32_t>(abs));
					file.write<std::int32_t>(static_cast<std::int32_t>(payload));
					file.write<std::int32_t>(static_cast<std::int32_t>(payload + local));
					file.write<std::int32_t>(static_cast<std::int32_t>(payload + local + global));
					const auto end = payload + local + global + virt;
					file.write<std::int32_t>(static_cast<std::int32_t>(end));
					file.write<std::int32_t>(static_cast<std::int32_t>(end));
					file.write<std::int32_t>(static_cast<std::int32_t>(end));
					file.fill(16, 0xFF);
				};

				write_section_header("__classnames__", names_start, names.size(), 0, 0, 0);
				write_section_header("__types__", data_start, 0, 0, 0, 0);
				write_section_header("__data__", data_start, data_size, local_size, global_size,
					virtual_size);

				file.write(names.data.data(), names.size());
				file.write(buf.data.data(), buf.size());
				file.write(fixups.data.data(), fixups.size());

				ZONETOOL_INFO("havok: physics asset \"%s\" built -- %zu bytes",
					input.body_name.c_str(), file.size());

				return file.data;
			}

			namespace
			{
				struct mass_properties
				{
					float volume = 0.0f;
					float mass = 0.0f;
					float com[3] = {0.0f, 0.0f, 0.0f};
					float inertia[3][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
				};

				mass_properties polytope_mass(const polytope& convex, const float scale)
				{
					constexpr float mult[10] = {
						1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 24.0f, 1.0f / 24.0f,
						1.0f / 60.0f, 1.0f / 60.0f, 1.0f / 60.0f,
						1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f
					};
					double intg[10] = {};

					const auto subexpr = [](const double w0, const double w1, const double w2,
						double& f1, double& f2, double& f3, double& g0, double& g1, double& g2)
					{
						const auto temp0 = w0 + w1;
						f1 = temp0 + w2;
						const auto temp1 = w0 * w0;
						const auto temp2 = temp1 + w1 * temp0;
						f2 = temp2 + w2 * f1;
						f3 = w0 * temp1 + w1 * temp2 + w2 * f2;
						g0 = f2 + w0 * (f1 + w0);
						g1 = f2 + w1 * (f1 + w1);
						g2 = f2 + w2 * (f1 + w2);
					};

					for (const auto& face : convex.faces)
					{
						for (std::size_t t = 2; t < face.indices.size(); t++)
						{
							const std::size_t corner[3] = {
								face.indices[0], face.indices[t - 1], face.indices[t]
							};
							if (corner[0] >= convex.verts.size() || corner[1] >= convex.verts.size()
								|| corner[2] >= convex.verts.size())
							{
								continue;
							}

							double p[3][3];
							for (auto c = 0; c < 3; c++)
							{
								for (auto k = 0; k < 3; k++)
								{
									p[c][k] = static_cast<double>(convex.verts[corner[c]][k]) * scale;
								}
							}

							const double a1 = p[1][0] - p[0][0], b1 = p[1][1] - p[0][1], c1 = p[1][2] - p[0][2];
							const double a2 = p[2][0] - p[0][0], b2 = p[2][1] - p[0][1], c2 = p[2][2] - p[0][2];
							const double d0 = b1 * c2 - b2 * c1;
							const double d1 = a2 * c1 - a1 * c2;
							const double d2 = a1 * b2 - a2 * b1;

							double f1x, f2x, f3x, g0x, g1x, g2x;
							double f1y, f2y, f3y, g0y, g1y, g2y;
							double f1z, f2z, f3z, g0z, g1z, g2z;
							subexpr(p[0][0], p[1][0], p[2][0], f1x, f2x, f3x, g0x, g1x, g2x);
							subexpr(p[0][1], p[1][1], p[2][1], f1y, f2y, f3y, g0y, g1y, g2y);
							subexpr(p[0][2], p[1][2], p[2][2], f1z, f2z, f3z, g0z, g1z, g2z);

							intg[0] += d0 * f1x;
							intg[1] += d0 * f2x;
							intg[2] += d1 * f2y;
							intg[3] += d2 * f2z;
							intg[4] += d0 * f3x;
							intg[5] += d1 * f3y;
							intg[6] += d2 * f3z;
							intg[7] += d0 * (p[0][1] * g0x + p[1][1] * g1x + p[2][1] * g2x);
							intg[8] += d1 * (p[0][2] * g0y + p[1][2] * g1y + p[2][2] * g2y);
							intg[9] += d2 * (p[0][0] * g0z + p[1][0] * g1z + p[2][0] * g2z);
						}
					}

					for (auto i = 0; i < 10; i++)
					{
						intg[i] *= mult[i];
					}

					mass_properties out{};
					const auto volume = intg[0];
					if (volume <= 1e-12)
					{
						return out;
					}

					const double cx = intg[1] / volume, cy = intg[2] / volume, cz = intg[3] / volume;
					const double ixx = intg[5] + intg[6] - volume * (cy * cy + cz * cz);
					const double iyy = intg[4] + intg[6] - volume * (cz * cz + cx * cx);
					const double izz = intg[4] + intg[5] - volume * (cx * cx + cy * cy);
					const double ixy = -(intg[7] - volume * cx * cy);
					const double iyz = -(intg[8] - volume * cy * cz);
					const double ixz = -(intg[9] - volume * cz * cx);

					out.volume = static_cast<float>(volume);
					out.mass = static_cast<float>(volume);
					out.com[0] = static_cast<float>(cx);
					out.com[1] = static_cast<float>(cy);
					out.com[2] = static_cast<float>(cz);
					out.inertia[0][0] = static_cast<float>(ixx);
					out.inertia[1][1] = static_cast<float>(iyy);
					out.inertia[2][2] = static_cast<float>(izz);
					out.inertia[0][1] = out.inertia[1][0] = static_cast<float>(ixy);
					out.inertia[1][2] = out.inertia[2][1] = static_cast<float>(iyz);
					out.inertia[0][2] = out.inertia[2][0] = static_cast<float>(ixz);
					return out;
				}

				void set_mass(mass_properties& mp, const float mass)
				{
					if (mp.volume <= 0.0f)
					{
						mp.mass = mass;
						return;
					}
					const auto factor = mass / mp.volume;
					mp.mass = mass;
					for (auto& row : mp.inertia)
					{
						for (auto& value : row)
						{
							value *= factor;
						}
					}
				}

				mass_properties combine_mass(const std::vector<mass_properties>& parts)
				{
					mass_properties out{};
					for (const auto& p : parts)
					{
						out.volume += p.volume;
						out.mass += p.mass;
						for (auto k = 0; k < 3; k++)
						{
							out.com[k] += p.com[k] * p.mass;
						}
					}
					if (out.mass <= 0.0f)
					{
						return out;
					}
					for (auto k = 0; k < 3; k++)
					{
						out.com[k] /= out.mass;
					}
					for (const auto& p : parts)
					{
						const float d[3] = {p.com[0] - out.com[0], p.com[1] - out.com[1], p.com[2] - out.com[2]};
						const auto dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
						for (auto i = 0; i < 3; i++)
						{
							for (auto j = 0; j < 3; j++)
							{
								out.inertia[i][j] += p.inertia[i][j]
									+ p.mass * ((i == j ? dd : 0.0f) - d[i] * d[j]);
							}
						}
					}
					return out;
				}

				void diagonalise(const float (&m)[3][3], float (&diag)[3], float (&axes)[3][3])
				{
					double a[3][3], v[3][3];
					for (auto i = 0; i < 3; i++)
					{
						for (auto j = 0; j < 3; j++)
						{
							a[i][j] = m[i][j];
							v[i][j] = i == j ? 1.0 : 0.0;
						}
					}

					for (auto sweep = 0; sweep < 50; sweep++)
					{
						const auto off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
						if (off < 1e-24)
						{
							break;
						}
						for (auto p = 0; p < 3; p++)
						{
							for (auto q = p + 1; q < 3; q++)
							{
								if (std::fabs(a[p][q]) < 1e-30)
								{
									continue;
								}
								const auto theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
								const auto t = (theta >= 0.0 ? 1.0 : -1.0)
									/ (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
								const auto c = 1.0 / std::sqrt(t * t + 1.0);
								const auto s = t * c;
								for (auto k = 0; k < 3; k++)
								{
									const auto akp = a[k][p], akq = a[k][q];
									a[k][p] = c * akp - s * akq;
									a[k][q] = s * akp + c * akq;
								}
								for (auto k = 0; k < 3; k++)
								{
									const auto apk = a[p][k], aqk = a[q][k];
									a[p][k] = c * apk - s * aqk;
									a[q][k] = s * apk + c * aqk;
								}
								for (auto k = 0; k < 3; k++)
								{
									const auto vkp = v[k][p], vkq = v[k][q];
									v[k][p] = c * vkp - s * vkq;
									v[k][q] = s * vkp + c * vkq;
								}
							}
						}
					}

					const double det = v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1])
						- v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0])
						+ v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]);
					if (det < 0.0)
					{
						for (auto k = 0; k < 3; k++)
						{
							v[k][2] = -v[k][2];
						}
					}

					for (auto i = 0; i < 3; i++)
					{
						diag[i] = static_cast<float>(std::max(a[i][i], 0.0));
						for (auto j = 0; j < 3; j++)
						{
							axes[i][j] = static_cast<float>(v[i][j]);
						}
					}
				}

				void quat_from_axes(const float (&r)[3][3], float (&q)[4])
				{
					const auto trace = r[0][0] + r[1][1] + r[2][2];
					if (trace > 0.0f)
					{
						const auto s = std::sqrt(trace + 1.0f) * 2.0f;
						q[3] = 0.25f * s;
						q[0] = (r[2][1] - r[1][2]) / s;
						q[1] = (r[0][2] - r[2][0]) / s;
						q[2] = (r[1][0] - r[0][1]) / s;
					}
					else if (r[0][0] > r[1][1] && r[0][0] > r[2][2])
					{
						const auto s = std::sqrt(1.0f + r[0][0] - r[1][1] - r[2][2]) * 2.0f;
						q[3] = (r[2][1] - r[1][2]) / s;
						q[0] = 0.25f * s;
						q[1] = (r[0][1] + r[1][0]) / s;
						q[2] = (r[0][2] + r[2][0]) / s;
					}
					else if (r[1][1] > r[2][2])
					{
						const auto s = std::sqrt(1.0f + r[1][1] - r[0][0] - r[2][2]) * 2.0f;
						q[3] = (r[0][2] - r[2][0]) / s;
						q[0] = (r[0][1] + r[1][0]) / s;
						q[1] = 0.25f * s;
						q[2] = (r[1][2] + r[2][1]) / s;
					}
					else
					{
						const auto s = std::sqrt(1.0f + r[2][2] - r[0][0] - r[1][1]) * 2.0f;
						q[3] = (r[1][0] - r[0][1]) / s;
						q[0] = (r[0][2] + r[2][0]) / s;
						q[1] = (r[1][2] + r[2][1]) / s;
						q[2] = 0.25f * s;
					}
					const auto length = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
					if (length > 0.0f)
					{
						for (auto& c : q)
						{
							c /= length;
						}
					}
				}

				void pack_vector3(const float* v, std::uint16_t (&out)[4])
				{
					auto largest = 0.0f;
					for (auto k = 0; k < 3; k++)
					{
						largest = std::max(largest, std::fabs(v[k]));
					}
					auto k = -46;
					if (largest > 0.0f)
					{
						int exponent = 0;
						std::frexp(largest / 32767.0f, &exponent);
						k = exponent - 16;
					}
					const auto scale = std::ldexp(1.0f, 16 + k);
					for (auto i = 0; i < 3; i++)
					{
						const auto lane = static_cast<int>(std::lround(v[i] / scale));
						out[i] = static_cast<std::uint16_t>(static_cast<std::int16_t>(
							std::max(-32767, std::min(32767, lane))));
					}
					out[3] = static_cast<std::uint16_t>((127 + k) << 7);
				}

				void pack_quat(const float (&q)[4], std::uint16_t (&out)[4])
				{
					for (auto i = 0; i < 4; i++)
					{
						const auto scaled = static_cast<std::int32_t>(q[i] * 1966080000.0f);
						const auto shifted = static_cast<std::uint32_t>(scaled) + 0x80000000u;
						out[i] = static_cast<std::uint16_t>(shifted >> 16);
					}
				}

				struct compressed_mass
				{
					std::uint16_t com[4];
					std::uint16_t inertia[4];
					std::uint16_t axes[4];
					float mass;
					float volume;
					float diag[3];
					float quat[4];
				};

				compressed_mass compress_mass(const mass_properties& mp)
				{
					compressed_mass out{};
					float axes[3][3];
					diagonalise(mp.inertia, out.diag, axes);
					quat_from_axes(axes, out.quat);
					pack_vector3(mp.com, out.com);
					pack_vector3(out.diag, out.inertia);
					pack_quat(out.quat, out.axes);
					out.mass = mp.mass;
					out.volume = mp.volume;
					return out;
				}
			}

			std::vector<std::uint8_t> build_dynamic_physics_asset(const dynamic_physics_asset_input& input)
			{
				struct half_edge
				{
					std::uint16_t face = 0;
					std::uint8_t edge = 0;
				};

				constexpr auto MOTION_INFINITE = 0x5F7FFFF0u;
				constexpr auto CHILD_DENSITY = 1000.0f;

				if (input.convexes.empty())
				{
					ZONETOOL_ERROR("havok: dynamic asset \"%s\" has no convexes", input.body_name.c_str());
					return {};
				}

				byte_buffer buf;
				std::vector<std::pair<std::size_t, std::size_t>> local_fixups;
				std::vector<std::pair<std::size_t, std::size_t>> global_fixups;
				std::vector<std::pair<std::size_t, int>> virtual_fixups;
				const auto align16 = [&] { buf.align(16, 0); };

				enum : int
				{
					CLASS_ASSET, CLASS_SYSTEM_DATA, CLASS_CONVEX, CLASS_PROPERTIES,
					CLASS_MASS, CLASS_CONNECTIVITY, CLASS_COMPOUND, CLASS_COMPOUND_DATA,
					CLASS_COUNT
				};

				const auto compound = input.convexes.size() > 1;
				const auto instance_count = static_cast<int>(input.convexes.size());

				std::vector<mass_properties> child_mass;
				for (const auto& convex : input.convexes)
				{
					child_mass.emplace_back(polytope_mass(convex, input.scale));
				}
				auto body_mass = combine_mass(child_mass);
				if (body_mass.volume <= 0.0f)
				{
					ZONETOOL_WARNING("havok: dynamic asset \"%s\" has no volume -- hull(s) not "
						"closed?", input.body_name.c_str());
				}
				set_mass(body_mass, input.mass);
				for (auto& child : child_mass)
				{
					set_mass(child, child.volume * CHILD_DENSITY);
				}
				const auto body_compressed = compress_mass(body_mass);

				virtual_fixups.emplace_back(static_cast<std::size_t>(0), CLASS_ASSET);
				buf.reserve(8);
				const auto system_data_slot = buf.reserve(8);
				std::array<std::size_t, 8> lookup_field{};
				const int lookup_count[8] = {1, 1, 1, 1, 1, 1, 0, 1};
				for (auto i = 0; i < 8; i++)
				{
					lookup_field[i] = buf.size();
					write_hk_array_header(buf, lookup_count[i]);
				}
				const std::uint32_t lookup_value[8] = {
					input.body_quality_crc, input.material_crc, input.motion_properties_crc, 0, 0, 0, 0, 0
				};
				for (auto i = 0; i < 8; i++)
				{
					if (!lookup_count[i])
					{
						continue;
					}
					local_fixups.emplace_back(lookup_field[i], buf.size());
					buf.write<std::uint32_t>(lookup_value[i]);
					align16();
				}

				const auto system_data_offset = buf.size();
				global_fixups.emplace_back(system_data_slot, system_data_offset);
				virtual_fixups.emplace_back(system_data_offset, CLASS_SYSTEM_DATA);
				buf.reserve(16);
				write_hk_array_header(buf, 0);
				write_hk_array_header(buf, 0);
				const auto motion_cinfos_field = buf.size();
				write_hk_array_header(buf, 1);
				const auto body_cinfos_field = buf.size();
				write_hk_array_header(buf, 1);
				write_hk_array_header(buf, 0);
				const auto referenced_field = buf.size();
				write_hk_array_header(buf, 1);
				const auto system_name_slot = buf.reserve(8);
				align16();

				local_fixups.emplace_back(motion_cinfos_field, buf.size());
				buf.write<std::uint16_t>(0xFFFF);
				buf.write<std::uint8_t>(1);
				buf.write<std::uint8_t>(0);
				buf.write<float>(body_mass.mass > 0.0f ? 1.0f / body_mass.mass : 0.0f);
				buf.write<std::uint32_t>(MOTION_INFINITE);
				buf.write<std::uint32_t>(MOTION_INFINITE);
				for (auto k = 0; k < 3; k++)
				{
					buf.write<float>(body_compressed.diag[k] > 0.0f ? 1.0f / body_compressed.diag[k] : 0.0f);
				}
				buf.write<float>(1.0f);
				for (auto k = 0; k < 3; k++)
				{
					buf.write<float>(body_mass.com[k]);
				}
				buf.write<float>(0.0f);
				for (auto k = 0; k < 4; k++)
				{
					buf.write<float>(body_compressed.quat[k]);
				}
				buf.reserve(32);
				align16();

				const auto body_offset = buf.size();
				local_fixups.emplace_back(body_cinfos_field, body_offset);
				const auto body_shape_slot = buf.reserve(8);
				buf.write<std::int32_t>(0);
				buf.write<std::uint32_t>(input.body_contents);
				buf.write<std::uint16_t>(0xFFFF);
				buf.write<std::uint8_t>(0xFF);
				buf.reserve(5);
				buf.write<std::uint64_t>(0);
				const auto body_name_slot = buf.reserve(8);
				buf.write<std::uint8_t>(0);
				buf.reserve(7);
				for (auto i = 0; i < 2; i++)
				{
					buf.write<float>(0.0f);
					buf.write<float>(0.0f);
					buf.write<float>(0.0f);
					buf.write<float>(1.0f);
				}
				buf.reserve(32);
				buf.write<float>(-1.0f);
				buf.reserve(12);
				buf.write<std::uint16_t>(0xFFFF);
				buf.write<std::uint16_t>(0);
				buf.write<std::uint32_t>(0x7FFFFFFFu);
				buf.write<std::uint32_t>(0);
				buf.write<std::uint32_t>(0);
				buf.reserve(16);
				if (buf.size() - body_offset != 160)
				{
					ZONETOOL_ERROR("havok: hknpBodyCinfo is %zu bytes, expected 160", buf.size() - body_offset);
					return {};
				}

				local_fixups.emplace_back(body_name_slot, buf.size());
				buf.write(input.body_name.c_str(), input.body_name.size() + 1);
				align16();

				local_fixups.emplace_back(referenced_field, buf.size());
				const auto referenced_slot = buf.reserve(8);
				align16();

				local_fixups.emplace_back(system_name_slot, buf.size());
				const char* system_name = "Default Physics System Data";
				buf.write(system_name, std::strlen(system_name) + 1);
				align16();

				const auto write_rel_array = [&](const std::size_t field, const std::size_t count)
				{
					buf.patch<std::uint16_t>(field, static_cast<std::uint16_t>(count));
					buf.patch<std::uint16_t>(field + 2, static_cast<std::uint16_t>(buf.size() - field));
				};

				const auto write_mass_properties = [&](const std::size_t properties_slot,
					const compressed_mass& cm)
				{
					const auto properties_offset = buf.size();
					global_fixups.emplace_back(properties_slot, properties_offset);
					virtual_fixups.emplace_back(properties_offset, CLASS_PROPERTIES);
					const auto entries_field = buf.size();
					write_hk_array_header(buf, 1);
					local_fixups.emplace_back(entries_field, buf.size());
					const auto mass_slot = buf.reserve(8);
					buf.write<std::uint16_t>(0xF100);
					buf.write<std::uint16_t>(0);
					buf.reserve(4);
					align16();

					const auto mass_offset = buf.size();
					global_fixups.emplace_back(mass_slot, mass_offset);
					virtual_fixups.emplace_back(mass_offset, CLASS_MASS);
					buf.reserve(16);
					for (const auto w : cm.com) buf.write<std::uint16_t>(w);
					for (const auto w : cm.inertia) buf.write<std::uint16_t>(w);
					for (const auto w : cm.axes) buf.write<std::uint16_t>(w);
					buf.write<float>(cm.mass);
					buf.write<float>(cm.volume);
					align16();
				};

				const auto write_convex = [&](const polytope& convex, const compressed_mass& cm,
					const int index, float (&mn)[3], float (&mx)[3]) -> std::size_t
				{
					auto index_total = 0u;
					for (const auto& face : convex.faces)
					{
						index_total += static_cast<unsigned int>(face.indices.size());
					}
					const auto vertex_count = padded_vertex_count(convex.verts.size());
					if (convex.verts.empty() || vertex_count > 255 || convex.faces.size() > 0xFFFF
						|| index_total > 0xFFFF)
					{
						ZONETOOL_ERROR("havok: convex %d of \"%s\" exceeds the format limits "
							"(%zu verts, %zu faces, %u indices)", index, input.body_name.c_str(),
							convex.verts.size(), convex.faces.size(), index_total);
						return 0;
					}

					const auto convex_offset = buf.size();
					virtual_fixups.emplace_back(convex_offset, CLASS_CONVEX);
					buf.reserve(16);
					buf.write<std::uint16_t>(static_cast<std::uint16_t>(CONVEX_SHAPE_FLAGS));
					buf.write<std::uint8_t>(0);
					buf.write<std::uint8_t>(static_cast<std::uint8_t>(CONVEX_DISPATCH_TYPE));
					buf.write<float>(0.0f);
					buf.write<std::uint64_t>(0);
					const auto properties_slot = buf.reserve(8);
					buf.reserve(8);
					const auto vertices_field = buf.size();
					buf.reserve(4);
					buf.reserve(12);
					const auto planes_field = buf.size();
					buf.reserve(4);
					const auto faces_field = buf.size();
					buf.reserve(4);
					const auto indices_field = buf.size();
					buf.reserve(4);
					buf.reserve(4);
					const auto connectivity_slot = buf.reserve(8);
					buf.reserve(8);
					if (buf.size() - convex_offset != SIZEOF_CONVEX_POLYTOPE_SHAPE)
					{
						ZONETOOL_ERROR("havok: hknpConvexPolytopeShape is %zu bytes, expected %d",
							buf.size() - convex_offset, SIZEOF_CONVEX_POLYTOPE_SHAPE);
						return 0;
					}

					std::vector<half_edge> vertex_edges(vertex_count);
					std::vector<bool> vertex_seen(vertex_count, false);
					std::vector<half_edge> face_links(index_total);
					std::map<std::pair<std::uint8_t, std::uint8_t>, half_edge> directed;
					for (auto f = 0u; f < convex.faces.size(); f++)
					{
						const auto& face = convex.faces[f];
						for (auto e = 0u; e < face.indices.size(); e++)
						{
							const auto from = face.indices[e];
							const auto to = face.indices[(e + 1) % face.indices.size()];
							const half_edge self{static_cast<std::uint16_t>(f), static_cast<std::uint8_t>(e)};
							if (from < vertex_seen.size() && !vertex_seen[from])
							{
								vertex_seen[from] = true;
								vertex_edges[from] = self;
							}
							directed[{from, to}] = self;
						}
					}
					auto global = 0u;
					auto unmatched = 0;
					for (auto f = 0u; f < convex.faces.size(); f++)
					{
						const auto& face = convex.faces[f];
						for (auto e = 0u; e < face.indices.size(); e++)
						{
							const auto from = face.indices[e];
							const auto to = face.indices[(e + 1) % face.indices.size()];
							const auto twin = directed.find({to, from});
							if (twin != directed.end())
							{
								face_links[global] = twin->second;
							}
							else
							{
								face_links[global] = {static_cast<std::uint16_t>(f), static_cast<std::uint8_t>(e)};
								unmatched++;
							}
							global++;
						}
					}
					if (unmatched)
					{
						ZONETOOL_WARNING("havok: convex %d of \"%s\" has %d unpaired edge(s) -- "
							"hull is not closed", index, input.body_name.c_str(), unmatched);
					}

					write_rel_array(vertices_field, vertex_count);
					for (auto j = 0u; j < vertex_count; j++)
					{
						const auto source = std::min<std::size_t>(j, convex.verts.size() - 1);
						const auto& v = convex.verts[source];
						for (auto k = 0; k < 3; k++)
						{
							const auto value = v[k] * input.scale;
							buf.write<float>(value);
							mn[k] = std::min(mn[k], value);
							mx[k] = std::max(mx[k], value);
						}
						buf.write<std::uint32_t>(INT24_W_BASE | static_cast<std::uint32_t>(source));
					}
					align16();

					write_rel_array(planes_field, convex.faces.size());
					for (const auto& face : convex.faces)
					{
						for (auto k = 0; k < 3; k++)
						{
							buf.write<float>(face.plane[k]);
						}
						buf.write<float>(-face.plane[3] * input.scale);
					}
					align16();

					write_rel_array(faces_field, convex.faces.size());
					auto first_index = 0u;
					for (auto f = 0u; f < convex.faces.size(); f++)
					{
						const auto& face = convex.faces[f];
						buf.write<std::uint16_t>(static_cast<std::uint16_t>(first_index));
						buf.write<std::uint8_t>(static_cast<std::uint8_t>(face.indices.size()));
						auto smallest = 45.0f;
						for (auto e = 0u; e < face.indices.size(); e++)
						{
							const auto& twin = face_links[first_index + e];
							if (twin.face == f || twin.face >= convex.faces.size())
							{
								continue;
							}
							const auto& other = convex.faces[twin.face].plane;
							auto dot = face.plane[0] * other[0] + face.plane[1] * other[1] + face.plane[2] * other[2];
							dot = std::max(-1.0f, std::min(1.0f, dot));
							const auto dihedral = 180.0f - static_cast<float>(std::acos(dot) * 57.29577951308232);
							smallest = std::min(smallest, dihedral * 0.5f);
						}
						const auto half_radians = smallest * 0.017453292519943295f;
						auto quantised = static_cast<int>((half_radians - 1.1920929e-7f) * 41720.875f + 0.5f);
						quantised = std::max(0, std::min(65535, quantised)) >> 8;
						buf.write<std::uint8_t>(static_cast<std::uint8_t>(quantised));
						first_index += static_cast<unsigned int>(face.indices.size());
					}
					align16();

					write_rel_array(indices_field, index_total);
					for (const auto& face : convex.faces)
					{
						for (const auto idx : face.indices)
						{
							buf.write<std::uint8_t>(idx);
						}
					}
					align16();

					write_mass_properties(properties_slot, cm);

					for (auto j = convex.verts.size(); j < vertex_count; j++)
					{
						vertex_edges[j] = vertex_edges[convex.verts.size() - 1];
					}
					const auto connectivity_offset = buf.size();
					global_fixups.emplace_back(connectivity_slot, connectivity_offset);
					virtual_fixups.emplace_back(connectivity_offset, CLASS_CONNECTIVITY);
					buf.reserve(16);
					const auto vertex_edges_field = buf.size();
					write_hk_array_header(buf, static_cast<int>(vertex_edges.size()));
					const auto face_links_field = buf.size();
					write_hk_array_header(buf, static_cast<int>(face_links.size()));
					const auto write_edges = [&](const std::size_t field, const std::vector<half_edge>& edges)
					{
						if (edges.empty())
						{
							return;
						}
						local_fixups.emplace_back(field, buf.size());
						for (const auto& edge : edges)
						{
							buf.write<std::uint16_t>(edge.face);
							buf.write<std::uint8_t>(edge.edge);
							buf.write<std::uint8_t>(0);
						}
						align16();
					};
					write_edges(vertex_edges_field, vertex_edges);
					write_edges(face_links_field, face_links);
					return convex_offset;
				};

				std::size_t root_offset = 0;
				if (!compound)
				{
					float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
					root_offset = write_convex(input.convexes[0], body_compressed, 0, mn, mx);
					if (!root_offset)
					{
						return {};
					}
				}
				else
				{
					const auto compound_offset = buf.size();
					root_offset = compound_offset;
					virtual_fixups.emplace_back(compound_offset, CLASS_COMPOUND);
					buf.reserve(16);
					buf.write<std::uint16_t>(static_cast<std::uint16_t>(COMPOUND_SHAPE_FLAGS));
					auto key_bits = 0;
					for (auto n = instance_count; n > 0; n >>= 1)
					{
						key_bits++;
					}
					buf.write<std::uint8_t>(static_cast<std::uint8_t>(key_bits));
					buf.write<std::uint8_t>(static_cast<std::uint8_t>(COMPOUND_DISPATCH_TYPE));
					buf.write<float>(0.0f);
					buf.write<std::uint64_t>(0);
					const auto compound_properties_slot = buf.reserve(8);
					buf.reserve(8);
					buf.write<std::uint32_t>(0xFFFFFFFFu);
					buf.write<std::uint32_t>(0);
					write_hk_array_header(buf, 0);
					write_hk_array_header(buf, 0);
					buf.write<std::uint32_t>(0xFFFFFFFFu);
					buf.reserve(4);
					const auto instances_field = buf.size();
					write_hk_array_header(buf, instance_count);
					buf.write<std::int32_t>(-1);
					buf.reserve(4);
					buf.reserve(8);
					const auto aabb_slot = buf.reserve(32);
					buf.write<std::uint8_t>(1);
					buf.reserve(7);
					buf.reserve(16);
					buf.reserve(8);
					const auto bvd_slot = buf.reserve(8);
					buf.reserve(8);
					if (buf.size() - compound_offset != SIZEOF_DYNAMIC_COMPOUND_SHAPE)
					{
						ZONETOOL_ERROR("havok: hknpDynamicCompoundShape is %zu bytes, expected %d",
							buf.size() - compound_offset, SIZEOF_DYNAMIC_COMPOUND_SHAPE);
						return {};
					}

					local_fixups.emplace_back(instances_field, buf.size());
					std::vector<std::size_t> instance_shape_slots;
					std::vector<std::size_t> instance_offsets;
					for (auto i = 0; i < instance_count; i++)
					{
						instance_offsets.emplace_back(buf.size());
						buf.write<float>(1.0f); buf.write<float>(0.0f);
						buf.write<float>(0.0f); buf.write<std::uint32_t>(SHAPE_INSTANCE_FLAGS_W);
						buf.write<float>(0.0f); buf.write<float>(1.0f);
						buf.write<float>(0.0f); buf.write<float>(0.0f);
						buf.write<float>(0.0f); buf.write<float>(0.0f);
						buf.write<float>(1.0f); buf.write<float>(0.0f);
						buf.write<float>(0.0f); buf.write<float>(0.0f);
						buf.write<float>(0.0f); buf.write<std::uint32_t>(INT24_W_BASE);
						for (auto c = 0; c < 4; c++)
						{
							buf.write<float>(1.0f);
						}
						instance_shape_slots.emplace_back(buf.reserve(8));
						buf.write<std::uint16_t>(0xFFFF);
						buf.write<std::uint16_t>(0xFFFF);
						buf.reserve(36);
					}
					align16();

					write_mass_properties(compound_properties_slot, body_compressed);

					float shape_min[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
					float shape_max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
					std::vector<std::array<float, 6>> convex_bounds;
					for (auto c = 0; c < instance_count; c++)
					{
						float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
						const auto convex_offset = write_convex(input.convexes[c],
							compress_mass(child_mass[c]), c, mn, mx);
						if (!convex_offset)
						{
							return {};
						}
						global_fixups.emplace_back(instance_shape_slots[c], convex_offset);
						convex_bounds.push_back({mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]});
						for (auto k = 0; k < 3; k++)
						{
							shape_min[k] = std::min(shape_min[k], mn[k]);
							shape_max[k] = std::max(shape_max[k], mx[k]);
						}
					}
					for (auto k = 0; k < 3; k++)
					{
						buf.patch<float>(aabb_slot + k * 4, shape_min[k]);
						buf.patch<float>(aabb_slot + 16 + k * 4, shape_max[k]);
					}
					buf.patch<std::uint32_t>(aabb_slot + 12,
						INT24_W_BASE | static_cast<std::uint32_t>(instance_count - 1));
					buf.patch<float>(aabb_slot + 28, 0.0f);

					struct tree_node
					{
						float mn[3] = {0.0f, 0.0f, 0.0f};
						float mx[3] = {0.0f, 0.0f, 0.0f};
						std::uint16_t parent = 0;
						std::uint32_t data = 0;
					};
					const auto bvd_offset = buf.size();
					global_fixups.emplace_back(bvd_slot, bvd_offset);
					virtual_fixups.emplace_back(bvd_offset, CLASS_COMPOUND_DATA);
					buf.reserve(16);
					const auto nodes_field = buf.size();
					const auto node_count = 2 * instance_count + 1;
					write_hk_array_header(buf, node_count);
					buf.write<std::int32_t>(2 * instance_count);
					buf.reserve(4);
					buf.write<std::int32_t>(instance_count);
					buf.reserve(4);
					buf.write<std::int32_t>(1);
					buf.reserve(4);
					if (buf.size() - bvd_offset != SIZEOF_DYNAMIC_COMPOUND_SHAPE_DATA)
					{
						ZONETOOL_ERROR("havok: hknpDynamicCompoundShapeData is %zu bytes, expected %d",
							buf.size() - bvd_offset, SIZEOF_DYNAMIC_COMPOUND_SHAPE_DATA);
						return {};
					}
					align16();

					std::vector<tree_node> nodes(node_count);
					auto next_node = 1;
					std::vector<int> order(instance_count);
					for (auto i = 0; i < instance_count; i++)
					{
						order[i] = i;
					}
					std::function<int(int, int, std::uint16_t)> build_tree =
						[&](const int first, const int count, const std::uint16_t parent) -> int
					{
						const auto self = next_node++;
						auto& node = nodes[self];
						node.parent = parent;
						for (auto k = 0; k < 3; k++)
						{
							node.mn[k] = FLT_MAX;
							node.mx[k] = -FLT_MAX;
						}
						for (auto i = 0; i < count; i++)
						{
							const auto& b = convex_bounds[order[first + i]];
							for (auto k = 0; k < 3; k++)
							{
								node.mn[k] = std::min(node.mn[k], b[k]);
								node.mx[k] = std::max(node.mx[k], b[k + 3]);
							}
						}
						if (count == 1)
						{
							node.data = static_cast<std::uint32_t>(order[first]) << 16;
							return self;
						}
						auto axis = 0;
						auto widest = node.mx[0] - node.mn[0];
						for (auto k = 1; k < 3; k++)
						{
							if (node.mx[k] - node.mn[k] > widest)
							{
								widest = node.mx[k] - node.mn[k];
								axis = k;
							}
						}
						const auto centre = [&](const int idx)
						{
							const auto& b = convex_bounds[idx];
							return (b[axis] + b[axis + 3]) * 0.5f;
						};
						std::sort(order.begin() + first, order.begin() + first + count,
							[&](const int a, const int b) { return centre(a) < centre(b); });
						const auto half = count / 2;
						const auto left = build_tree(first, half, static_cast<std::uint16_t>(self));
						const auto right = build_tree(first + half, count - half, static_cast<std::uint16_t>(self));
						nodes[self].data = static_cast<std::uint32_t>(left) | (static_cast<std::uint32_t>(right) << 16);
						return self;
					};
					build_tree(0, instance_count, 0);

					for (auto n = 1; n < node_count - 1; n++)
					{
						const auto& node = nodes[n];
						if ((node.data & 0xFFFF) != 0)
						{
							continue;
						}
						const auto instance = static_cast<int>(node.data >> 16);
						if (instance < instance_count)
						{
							buf.patch<std::uint32_t>(instance_offsets[instance] + 48 + 12,
								INT24_W_BASE | static_cast<std::uint32_t>(n));
						}
					}

					local_fixups.emplace_back(nodes_field, buf.size());
					for (auto n = 0; n < node_count; n++)
					{
						const auto& node = nodes[n];
						const auto used = (n != 0 && n != node_count - 1);
						for (auto k = 0; k < 3; k++)
						{
							buf.write<float>(used ? node.mn[k] : 0.0f);
						}
						buf.write<std::uint16_t>(used ? node.parent : 0);
						buf.write<std::uint16_t>(used ? 0x3F00 : 0);
						for (auto k = 0; k < 3; k++)
						{
							buf.write<float>(used ? node.mx[k] : 0.0f);
						}
						buf.write<std::uint32_t>(used ? node.data : 0);
					}
					align16();
				}

				global_fixups.emplace_back(body_shape_slot, root_offset);
				global_fixups.emplace_back(referenced_slot, root_offset);
				const auto data_size = buf.size();

				byte_buffer names;
				const auto write_name = [&](const std::uint32_t sig, const char* name)
				{
					names.write<std::uint32_t>(sig);
					names.write<std::uint8_t>(0x09);
					const auto offset = names.size();
					names.write(name, std::strlen(name) + 1);
					return offset;
				};
				write_name(SIG_HK_CLASS, "hkClass");
				write_name(SIG_HK_CLASS_MEMBER, "hkClassMember");
				write_name(SIG_HK_CLASS_ENUM, "hkClassEnum");
				write_name(SIG_HK_CLASS_ENUM_ITEM, "hkClassEnumItem");
				std::array<std::size_t, CLASS_COUNT> name_offsets{};
				name_offsets[CLASS_ASSET] = write_name(SIG_PHYSICS_ASSET, "HavokPhysicsAsset");
				name_offsets[CLASS_SYSTEM_DATA] = write_name(SIG_PHYSICS_SYSTEM_DATA, "hknpPhysicsSystemData");
				if (compound)
				{
					name_offsets[CLASS_COMPOUND] = write_name(SIG_DYNAMIC_COMPOUND_SHAPE, "hknpDynamicCompoundShape");
				}
				name_offsets[CLASS_PROPERTIES] = write_name(SIG_REF_COUNTED_PROPERTIES, "hkRefCountedProperties");
				name_offsets[CLASS_MASS] = write_name(SIG_SHAPE_MASS_PROPERTIES, "hknpShapeMassProperties");
				name_offsets[CLASS_CONVEX] = write_name(SIG_CONVEX_POLYTOPE_SHAPE, "hknpConvexPolytopeShape");
				name_offsets[CLASS_CONNECTIVITY] = write_name(SIG_CONVEX_POLYTOPE_CONNECTIVITY, "hknpConvexPolytopeShapeConnectivity");
				if (compound)
				{
					name_offsets[CLASS_COMPOUND_DATA] = write_name(SIG_DYNAMIC_COMPOUND_SHAPE_DATA, "hknpDynamicCompoundShapeData");
				}
				names.align(16, 0xFF);

				std::sort(local_fixups.begin(), local_fixups.end(),
					[](const std::pair<std::size_t, std::size_t>& a, const std::pair<std::size_t, std::size_t>& b)
					{
						return a.second < b.second;
					});
				byte_buffer fixups;
				for (const auto& fixup : local_fixups)
				{
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.second));
				}
				fixups.align(16, 0xFF);
				const auto local_size = fixups.size();
				std::sort(global_fixups.begin(), global_fixups.end());
				for (const auto& fixup : global_fixups)
				{
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
					fixups.write<std::int32_t>(2);
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.second));
				}
				fixups.align(16, 0xFF);
				const auto global_size = fixups.size() - local_size;
				std::sort(virtual_fixups.begin(), virtual_fixups.end());
				for (const auto& fixup : virtual_fixups)
				{
					fixups.write<std::int32_t>(static_cast<std::int32_t>(fixup.first));
					fixups.write<std::int32_t>(0);
					fixups.write<std::int32_t>(static_cast<std::int32_t>(name_offsets[fixup.second]));
				}
				fixups.align(16, 0xFF);
				const auto virtual_size = fixups.size() - local_size - global_size;

				byte_buffer file;
				file.write<std::uint32_t>(HK_MAGIC0);
				file.write<std::uint32_t>(HK_MAGIC1);
				file.write<std::int32_t>(0);
				file.write<std::int32_t>(HK_FILE_VERSION);
				file.write<std::uint8_t>(8);
				file.write<std::uint8_t>(1);
				file.write<std::uint8_t>(0);
				file.write<std::uint8_t>(1);
				file.write<std::int32_t>(3);
				file.write<std::int32_t>(2);
				file.write<std::int32_t>(0);
				file.write<std::int32_t>(0);
				file.write<std::int32_t>(static_cast<std::int32_t>(name_offsets[CLASS_ASSET]));
				const char* version = "hk_2014.2.5-r1";
				const auto version_length = std::strlen(version) + 1;
				file.write(version, version_length);
				file.fill(16 - version_length, 0xFF);
				file.write<std::int32_t>(0);
				file.write<std::uint16_t>(HK_MAX_PREDICATE);
				file.write<std::uint16_t>(0);

				const auto names_start = HK_HEADER_SIZE + 3 * HK_SECTION_HEADER_SIZE;
				const auto data_start = names_start + names.size();
				const auto write_section_header = [&](const char* tag, const std::size_t abs,
					const std::size_t payload, const std::size_t local, const std::size_t global,
					const std::size_t virt)
				{
					char name[19] = {};
					std::strncpy(name, tag, sizeof(name));
					file.write(name, sizeof(name));
					file.write<std::uint8_t>(0xFF);
					file.write<std::int32_t>(static_cast<std::int32_t>(abs));
					file.write<std::int32_t>(static_cast<std::int32_t>(payload));
					file.write<std::int32_t>(static_cast<std::int32_t>(payload + local));
					file.write<std::int32_t>(static_cast<std::int32_t>(payload + local + global));
					const auto end = payload + local + global + virt;
					file.write<std::int32_t>(static_cast<std::int32_t>(end));
					file.write<std::int32_t>(static_cast<std::int32_t>(end));
					file.write<std::int32_t>(static_cast<std::int32_t>(end));
					file.fill(16, 0xFF);
				};
				write_section_header("__classnames__", names_start, names.size(), 0, 0, 0);
				write_section_header("__types__", data_start, 0, 0, 0, 0);
				write_section_header("__data__", data_start, data_size, local_size, global_size, virtual_size);
				file.write(names.data.data(), names.size());
				file.write(buf.data.data(), buf.size());
				file.write(fixups.data.data(), fixups.size());

				ZONETOOL_INFO("havok: dynamic physics asset \"%s\" built -- %d convex(es), mass %.2f, "
					"volume %.4f, %zu bytes", input.body_name.c_str(), instance_count,
					body_mass.mass, body_mass.volume, file.size());
				return file.data;
			}
		}
	}
}
