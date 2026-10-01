#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "navmesh.hpp"

#include <utils/string.hpp>

#include <deque>
#include <numbers>
#include <sstream>

// BO3 navigation is Havok AI (hk_2014.2.0): the navmesh asset holds six serialized Havok packfiles (hkai_main.cpp,
// loaded by 0x140444F70). Blob 3 carries the hkaiNavMesh: convex faces over half-edges, vertices in Havok units
// (meters: BO3's physics code multiplies world positions by 0.0254, 0x140E091C4). IW7 uses BabelFlux NavPower.
//
// hkaiNavMesh layouts, from the hkClass reflection data in BlackOps3_UnrankedDedicatedServer (members 0x140F3C670,
// face hkClass 0x14B3B1690, edge hkClass 0x14B3B16E0, edge flag enum 0x140F3C418):
//   hkaiNavMesh (0xB0): faces +0x10, edges +0x20, vertices (hkVector4) +0x30 (hkArray: data, i32 size, i32 capacity)
//   Face (0x10): startEdgeIndex i32 +0, startUserEdgeIndex i32 +4, numEdges i16 +8, numUserEdges i16 +10, cluster +12
//   Edge (0x14): a i32 +0, b i32 +4, oppositeEdge u32 +8, oppositeFace u32 +12, flags u8 +16
//
// IW7 NavPower image (graph v43), as stock maps ship it: areas are convex in
// XY, CCW from +Z and planar (the engine takes an area's plane from vertex 0, 1 and the basis vertex); adjacent areas
// share exact reversed edge endpoints; clusters of at most 50 areas are split by 0x8000 portal edges; a KD tree over
// the areas follows them; checksum = little-endian u32 word sum of the image after its 24-byte header.

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace navmesh
		{
			namespace
			{
				constexpr double world_per_havok = 1.0 / 0.0254;
				constexpr double weld = 1.0 / 64.0; // Havok keeps some vertices twice at one position
				constexpr float planar_tolerance = 0.5f; // world units a vertex may sit off its area's plane
				constexpr float convex_epsilon = 1e-4f; // XY cross products below this count as collinear
				constexpr std::size_t cluster_size = 50;
				constexpr float build_scale = 72.0f;
				constexpr float kd_epsilon = 0.0001f * build_scale;
				constexpr std::uint32_t no_edge = 0xFFFFFFFFu;
				constexpr double max_area_slope = 60.0; // degrees; with max_area_rise, a rise IW7's zombies cannot walk
				constexpr float max_area_rise = 20.0f;   // world units: twice IW7's 10 unit step

				// the last 128 bytes of every stock NavGraph header (NavPower's own constant)
				constexpr std::uint8_t navpower_constant[128] = {
					0xbf, 0x14, 0xd6, 0x7e, 0x2d, 0xdc, 0x8e, 0x66, 0x83, 0xef, 0x57, 0x49, 0x61, 0xff, 0x69, 0x8f,
					0x61, 0xcd, 0xd1, 0x1e, 0x9d, 0x9c, 0x16, 0x72, 0x72, 0xe6, 0x1d, 0xf0, 0x84, 0x4f, 0x4a, 0x77,
					0x02, 0xd7, 0xe8, 0x39, 0x2c, 0x53, 0xcb, 0xc9, 0x12, 0x1e, 0x33, 0x74, 0x9e, 0x0c, 0xf4, 0xd5,
					0xd4, 0x9f, 0xd4, 0xa4, 0x59, 0x7e, 0x35, 0xcf, 0x32, 0x22, 0xf4, 0xcc, 0xcf, 0xd3, 0x90, 0x2d,
					0x48, 0xd3, 0x8f, 0x75, 0xe6, 0xd9, 0x1d, 0x2a, 0xe5, 0xc0, 0xf7, 0x2b, 0x78, 0x81, 0x87, 0x44,
					0x0e, 0x5f, 0x50, 0x00, 0xd4, 0x61, 0x8d, 0xbe, 0x7b, 0x05, 0x15, 0x07, 0x3b, 0x33, 0x82, 0x1f,
					0x18, 0x70, 0x92, 0xda, 0x64, 0x54, 0xce, 0xb1, 0x85, 0x3e, 0x69, 0x15, 0xf8, 0x46, 0x6a, 0x04,
					0x96, 0x73, 0x0e, 0xd9, 0x16, 0x2f, 0x67, 0x68, 0xd4, 0xf7, 0x4a, 0x4a, 0xd0, 0x57, 0x68, 0x76,
				};

				using vec3 = std::array<float, 3>;

				vec3 sub(const vec3& a, const vec3& b)
				{
					return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
				}

				vec3 cross(const vec3& a, const vec3& b)
				{
					return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
				}

				double dot(const vec3& a, const vec3& b)
				{
					return static_cast<double>(a[0]) * b[0] + static_cast<double>(a[1]) * b[1] + static_cast<double>(a[2]) * b[2];
				}

				double length(const vec3& a)
				{
					return std::sqrt(dot(a, a));
				}

				// ---- Havok packfile ---------------------------------------------------------------------

				// A hk_2014 packfile of the x64 layout (8-byte pointers, little endian): its __data__ section, the
				// section's local pointer fixups and each object's class name (virtual fixups)
				class packfile
				{
				public:
					packfile(const std::uint8_t* data, const std::size_t size)
						: data_(data), size_(size)
					{
						if (size < 0x40 || this->u32(0) != 0x57E0E057 || this->u32(4) != 0x10C0C010)
						{
							throw std::runtime_error("not a Havok packfile");
						}
						if (this->i32(12) != 11 || data[0x10] != 8 || data[0x11] != 1)
						{
							throw std::runtime_error(utils::string::va("Havok packfile version %d, %u-byte pointers", this->i32(12), data[0x10]));
						}
						const auto sections = this->i32(0x14);
						auto at = 0x40u + static_cast<std::uint16_t>(this->i16(0x3E));
						for (auto s = 0; s < sections; s++, at += 0x40)
						{
							section sec{};
							sec.tag.assign(reinterpret_cast<const char*>(data + at), strnlen(reinterpret_cast<const char*>(data + at), 19));
							for (auto k = 0; k < 7; k++)
							{
								sec.fields[k] = this->i32(at + 20 + k * 4);
							}
							this->sections_.push_back(sec);
						}

						const auto* data_section = this->find_section("__data__");
						const auto start = static_cast<std::uint32_t>(data_section->fields[0]);
						this->data_start_ = start;
						this->data_size_ = static_cast<std::uint32_t>(data_section->fields[1]);
						for (auto p = start + data_section->fields[1]; p + 8 <= start + data_section->fields[2]; p += 8)
						{
							const auto src = this->i32(p);
							if (src != -1)
							{
								this->fixups_[static_cast<std::uint32_t>(src)] = static_cast<std::uint32_t>(this->i32(p + 4));
							}
						}
						for (auto p = start + data_section->fields[3]; p + 12 <= start + data_section->fields[4]; p += 12)
						{
							const auto src = this->i32(p);
							if (src == -1)
							{
								continue;
							}
							const auto& names = this->sections_.at(static_cast<std::size_t>(this->i32(p + 4)));
							const auto name_at = static_cast<std::uint32_t>(names.fields[0] + this->i32(p + 8));
							this->objects_.emplace_back(static_cast<std::uint32_t>(src),
								std::string(reinterpret_cast<const char*>(data + name_at), strnlen(reinterpret_cast<const char*>(data + name_at), 256)));
						}
					}

					// the __data__ offset of the only object of `cls`
					std::uint32_t object(const std::string& cls) const
					{
						std::optional<std::uint32_t> found;
						for (const auto& [offset, name] : this->objects_)
						{
							if (name == cls)
							{
								if (found)
								{
									throw std::runtime_error(utils::string::va("more than one %s", cls.data()));
								}
								found = offset;
							}
						}
						if (!found)
						{
							throw std::runtime_error(utils::string::va("no %s", cls.data()));
						}
						return *found;
					}

					// the hkArray at a __data__ offset: its elements' offset and count
					std::pair<std::uint32_t, std::uint32_t> array(const std::uint32_t at) const
					{
						const auto count = this->d_i32(at + 8);
						if (count <= 0)
						{
							return { 0, 0 };
						}
						const auto found = this->fixups_.find(at);
						if (found == this->fixups_.end())
						{
							throw std::runtime_error(utils::string::va("array at 0x%X has no data", at));
						}
						return { found->second, static_cast<std::uint32_t>(count) };
					}

					std::int32_t d_i32(const std::uint32_t at) const
					{
						return this->i32(this->data_start_ + this->check(at, 4));
					}

					std::int16_t d_i16(const std::uint32_t at) const
					{
						return this->i16(this->data_start_ + this->check(at, 2));
					}

					std::uint8_t d_u8(const std::uint32_t at) const
					{
						return this->data_[this->data_start_ + this->check(at, 1)];
					}

					float d_f32(const std::uint32_t at) const
					{
						float value;
						std::memcpy(&value, this->data_ + this->data_start_ + this->check(at, 4), 4);
						return value;
					}

				private:
					struct section
					{
						std::string tag;
						std::int32_t fields[7]; // start, local, global, virtual, exports, imports, end
					};

					std::uint32_t check(const std::uint32_t at, const std::uint32_t size) const
					{
						if (at + size > this->data_size_)
						{
							throw std::runtime_error(utils::string::va("read at 0x%X past the Havok data (%u bytes)", at, this->data_size_));
						}
						return at;
					}

					const section* find_section(const char* tag) const
					{
						for (const auto& s : this->sections_)
						{
							if (s.tag == tag)
							{
								return &s;
							}
						}
						throw std::runtime_error(utils::string::va("Havok packfile without %s", tag));
					}

					std::uint32_t u32(const std::size_t at) const
					{
						if (at + 4 > this->size_)
						{
							throw std::runtime_error("read past the Havok packfile");
						}
						std::uint32_t value;
						std::memcpy(&value, this->data_ + at, 4);
						return value;
					}

					std::int32_t i32(const std::size_t at) const
					{
						return static_cast<std::int32_t>(this->u32(at));
					}

					std::int16_t i16(const std::size_t at) const
					{
						if (at + 2 > this->size_)
						{
							throw std::runtime_error("read past the Havok packfile");
						}
						std::int16_t value;
						std::memcpy(&value, this->data_ + at, 2);
						return value;
					}

					const std::uint8_t* data_;
					std::size_t size_;
					std::vector<section> sections_;
					std::uint32_t data_start_ = 0;
					std::uint32_t data_size_ = 0;
					std::unordered_map<std::uint32_t, std::uint32_t> fixups_;
					std::vector<std::pair<std::uint32_t, std::string>> objects_;
				};

				// ---- BO3 navmesh ------------------------------------------------------------------------

				struct havok_mesh
				{
					std::vector<vec3> vertices; // world units
					std::vector<std::vector<std::uint32_t>> faces; // vertex indices, in edge order
				};

				havok_mesh read_mesh(const HavokBlob& blob)
				{
					const packfile pf(reinterpret_cast<const std::uint8_t*>(blob.data), blob.size);
					const auto mesh = pf.object("hkaiNavMesh");
					const auto [faces_at, face_count] = pf.array(mesh + 0x10);
					const auto [edges_at, edge_count] = pf.array(mesh + 0x20);
					const auto [vertices_at, vertex_count] = pf.array(mesh + 0x30);

					havok_mesh out{};
					for (auto v = 0u; v < vertex_count; v++)
					{
						vec3 p{};
						for (auto k = 0; k < 3; k++)
						{
							p[k] = static_cast<float>(pf.d_f32(vertices_at + v * 16 + k * 4) * world_per_havok);
						}
						out.vertices.push_back(p);
					}
					for (auto f = 0u; f < face_count; f++)
					{
						const auto at = faces_at + f * 16;
						const auto start = pf.d_i32(at);
						const auto count = pf.d_i16(at + 8);
						std::vector<std::uint32_t> loop;
						for (auto k = 0; k < count; k++)
						{
							const auto edge = static_cast<std::uint32_t>(start + k);
							if (edge >= edge_count)
							{
								throw std::runtime_error(utils::string::va("face %u edge %u of %u", f, edge, edge_count));
							}
							const auto a = static_cast<std::uint32_t>(pf.d_i32(edges_at + edge * 20));
							if (a >= vertex_count)
							{
								throw std::runtime_error(utils::string::va("edge %u vertex %u of %u", edge, a, vertex_count));
							}
							loop.push_back(a);
						}
						out.faces.push_back(std::move(loop));
					}
					return out;
				}

				// ---- polygon geometry -------------------------------------------------------------------

				double area_xy(const std::vector<vec3>& p)
				{
					double s = 0.0;
					for (std::size_t i = 0; i < p.size(); i++)
					{
						const auto& a = p[i];
						const auto& b = p[(i + 1) % p.size()];
						s += static_cast<double>(a[0]) * b[1] - static_cast<double>(b[0]) * a[1];
					}
					return s * 0.5;
				}

				double planar_deviation(const std::vector<vec3>& p)
				{
					// Newell normal through the vertex average
					double n[3] = {};
					for (std::size_t i = 0; i < p.size(); i++)
					{
						const auto& a = p[i];
						const auto& b = p[(i + 1) % p.size()];
						n[0] += (static_cast<double>(a[1]) - b[1]) * (static_cast<double>(a[2]) + b[2]);
						n[1] += (static_cast<double>(a[2]) - b[2]) * (static_cast<double>(a[0]) + b[0]);
						n[2] += (static_cast<double>(a[0]) - b[0]) * (static_cast<double>(a[1]) + b[1]);
					}
					const auto l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
					if (l <= 0.0)
					{
						return 0.0;
					}
					double c[3] = {};
					for (const auto& v : p)
					{
						for (auto k = 0; k < 3; k++)
						{
							c[k] += v[k];
						}
					}
					auto worst = 0.0;
					for (const auto& v : p)
					{
						auto d = 0.0;
						for (auto k = 0; k < 3; k++)
						{
							d += (v[k] - c[k] / static_cast<double>(p.size())) * n[k] / l;
						}
						worst = std::max(worst, std::fabs(d));
					}
					return worst;
				}

				bool convex_ccw_xy(const std::vector<vec3>& p)
				{
					if (area_xy(p) <= 0.0)
					{
						return false;
					}
					for (std::size_t i = 0; i < p.size(); i++)
					{
						const auto& a = p[i];
						const auto& b = p[(i + 1) % p.size()];
						const auto& c = p[(i + 2) % p.size()];
						const auto cr = (static_cast<double>(b[0]) - a[0]) * (static_cast<double>(c[1]) - b[1])
							- (static_cast<double>(b[1]) - a[1]) * (static_cast<double>(c[0]) - b[0]);
						if (cr < -convex_epsilon)
						{
							return false;
						}
					}
					return true;
				}

				std::vector<vec3> points(const std::vector<std::uint32_t>& loop, const std::vector<vec3>& vertices)
				{
					std::vector<vec3> out;
					for (const auto i : loop)
					{
						out.push_back(vertices[i]);
					}
					return out;
				}

				double turn(const vec3& a, const vec3& b, const vec3& c)
				{
					return (static_cast<double>(b[0]) - a[0]) * (static_cast<double>(c[1]) - b[1])
						- (static_cast<double>(b[1]) - a[1]) * (static_cast<double>(c[0]) - b[0]);
				}

				// triangles of a simple CCW polygon (ear clipping in XY; a degenerate remainder is fanned)
				std::vector<std::vector<std::uint32_t>> ear_clip(std::vector<std::uint32_t> idx, const std::vector<vec3>& v)
				{
					std::vector<std::vector<std::uint32_t>> tris;
					while (idx.size() > 3)
					{
						const auto n = idx.size();
						auto clipped = false;
						for (std::size_t i = 0; i < n && !clipped; i++)
						{
							const auto a = idx[(i + n - 1) % n];
							const auto b = idx[i];
							const auto c = idx[(i + 1) % n];
							if (turn(v[a], v[b], v[c]) <= 0.0)
							{
								continue;
							}
							auto inside = false;
							for (const auto j : idx)
							{
								if (j == a || j == b || j == c)
								{
									continue;
								}
								const auto& A = v[a];
								const auto& B = v[b];
								const auto& C = v[c];
								const auto& P = v[j];
								const auto d1 = (static_cast<double>(B[0]) - A[0]) * (static_cast<double>(P[1]) - A[1]) - (static_cast<double>(B[1]) - A[1]) * (static_cast<double>(P[0]) - A[0]);
								const auto d2 = (static_cast<double>(C[0]) - B[0]) * (static_cast<double>(P[1]) - B[1]) - (static_cast<double>(C[1]) - B[1]) * (static_cast<double>(P[0]) - B[0]);
								const auto d3 = (static_cast<double>(A[0]) - C[0]) * (static_cast<double>(P[1]) - C[1]) - (static_cast<double>(A[1]) - C[1]) * (static_cast<double>(P[0]) - C[0]);
								if (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0)
								{
									inside = true;
									break;
								}
							}
							if (inside)
							{
								continue;
							}
							tris.push_back({ a, b, c });
							idx.erase(idx.begin() + static_cast<std::ptrdiff_t>(i));
							clipped = true;
						}
						if (!clipped)
						{
							for (std::size_t k = 1; k + 1 < idx.size(); k++)
							{
								tris.push_back({ idx[0], idx[k], idx[k + 1] });
							}
							return tris;
						}
					}
					tris.push_back(idx);
					return tris;
				}

				// merges triangles sharing an edge while the union stays convex, CCW and planar
				std::vector<std::vector<std::uint32_t>> merge(std::vector<std::vector<std::uint32_t>> polys, const std::vector<vec3>& v)
				{
					auto changed = true;
					while (changed)
					{
						changed = false;
						for (std::size_t i = 0; i < polys.size() && !changed; i++)
						{
							if (polys[i].empty())
							{
								continue;
							}
							for (std::size_t j = 0; j < polys.size() && !changed; j++)
							{
								if (i == j || polys[j].empty())
								{
									continue;
								}
								const auto& a = polys[i];
								const auto& b = polys[j];
								for (std::size_t ai = 0; ai < a.size() && !changed; ai++)
								{
									const auto u = a[ai];
									const auto w = a[(ai + 1) % a.size()];
									const auto bj = std::find(b.begin(), b.end(), w);
									if (bj == b.end())
									{
										continue;
									}
									const auto k = static_cast<std::size_t>(bj - b.begin());
									if (b[(k + 1) % b.size()] != u)
									{
										continue;
									}
									std::vector<std::uint32_t> merged(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(ai + 1));
									for (std::size_t r = 0; r + 2 < b.size(); r++)
									{
										merged.push_back(b[(k + 2 + r) % b.size()]);
									}
									merged.insert(merged.end(), a.begin() + static_cast<std::ptrdiff_t>(ai + 1), a.end());
									std::unordered_set<std::uint32_t> distinct(merged.begin(), merged.end());
									const auto p = points(merged, v);
									if (distinct.size() == merged.size() && convex_ccw_xy(p) && planar_deviation(p) <= planar_tolerance)
									{
										polys[i] = std::move(merged);
										polys[j].clear();
										changed = true;
									}
								}
							}
						}
					}
					std::vector<std::vector<std::uint32_t>> out;
					for (auto& p : polys)
					{
						if (!p.empty())
						{
							out.push_back(std::move(p));
						}
					}
					return out;
				}

				// NavKit CalculateCentroid: the polygon's area centroid in its plane (normal from v0, v1, v2)
				vec3 centroid(const std::vector<vec3>& p)
				{
					const auto average = [&]()
					{
						vec3 c{};
						for (const auto& v : p)
						{
							for (auto k = 0; k < 3; k++)
							{
								c[k] += v[k] / static_cast<float>(p.size());
							}
						}
						return c;
					};
					const auto n = cross(sub(p[2], p[0]), sub(p[1], p[0]));
					const auto ln = length(n);
					const auto e = sub(p[1], p[0]);
					const auto le = length(e);
					if (ln <= 0.0 || le <= 0.0)
					{
						return average();
					}
					const vec3 nrm{ static_cast<float>(n[0] / ln), static_cast<float>(n[1] / ln), static_cast<float>(n[2] / ln) };
					const vec3 u{ static_cast<float>(e[0] / le), static_cast<float>(e[1] / le), static_cast<float>(e[2] / le) };
					auto w = cross(u, nrm);
					const auto lw = length(w);
					const vec3 v{ static_cast<float>(w[0] / lw), static_cast<float>(w[1] / lw), static_cast<float>(w[2] / lw) };
					std::vector<std::pair<double, double>> uv;
					for (const auto& q : p)
					{
						uv.emplace_back(dot(sub(q, p[0]), u), dot(sub(q, p[0]), v));
					}
					auto s = 0.0;
					for (std::size_t i = 0; i < uv.size(); i++)
					{
						const auto& a = uv[i];
						const auto& b = uv[(i + 1) % uv.size()];
						s += a.first * b.second - b.first * a.second;
					}
					if (std::fabs(s) < 2e-9)
					{
						return average();
					}
					const auto area = std::fabs(s / 2.0);
					auto sx = 0.0, sy = 0.0;
					for (std::size_t i = 0; i < uv.size(); i++)
					{
						const auto& a = uv[i];
						const auto& b = uv[(i + 1) % uv.size()];
						const auto d = a.first * b.second - b.first * a.second;
						sx += (a.first + b.first) * d;
						sy += (a.second + b.second) * d;
					}
					const auto cu = sx / (6.0 * area);
					const auto cv = sy / (6.0 * area);
					return { static_cast<float>(p[0][0] + u[0] * cu + v[0] * cv), static_cast<float>(p[0][1] + u[1] * cu + v[1] * cv),
						static_cast<float>(p[0][2] + u[2] * cu + v[2] * cv) };
				}

				// NavKit CalculateBasisVert: the vertex (index >= 2) farthest from the line through v0 and v1
				std::uint32_t basis_vertex(const std::vector<vec3>& p)
				{
					std::uint32_t best = 2;
					auto best_distance = -1.0;
					const auto d01 = sub(p[1], p[0]);
					const auto l2 = dot(d01, d01);
					for (std::uint32_t i = 2; i < p.size(); i++)
					{
						const auto t = l2 > 0.0 ? dot(sub(p[i], p[0]), d01) / l2 : 0.0;
						const vec3 foot{ static_cast<float>(p[0][0] + d01[0] * t), static_cast<float>(p[0][1] + d01[1] * t),
							static_cast<float>(p[0][2] + d01[2] * t) };
						const auto d = length(sub(p[i], foot));
						if (d > best_distance)
						{
							best = i;
							best_distance = d;
						}
					}
					return best;
				}

				// ---- the NavPower image -----------------------------------------------------------------

				struct nav_areas
				{
					std::vector<vec3> vertices;
					std::vector<std::vector<std::uint32_t>> areas;
					std::vector<std::vector<std::int32_t>> adjacent; // per edge: the neighbour area or -1
					std::vector<std::uint32_t> cluster;
					std::vector<std::uint32_t> component; // per area: its connected component
					std::vector<std::uint32_t> order; // storage order: components in breadth-first order
					std::size_t components = 0;
					std::size_t largest_component = 0;
					std::size_t clusters = 0;
					std::size_t faces_split = 0;
					std::size_t faces_dropped = 0;
					std::size_t steep_dropped = 0;
					std::size_t edge_conflicts = 0;
				};

				nav_areas build_areas(const havok_mesh& mesh)
				{
					nav_areas out{};

					// weld; every test runs on the float values the image holds
					std::map<std::array<long long, 3>, std::uint32_t> welded;
					std::vector<std::uint32_t> remap;
					for (const auto& p : mesh.vertices)
					{
						const std::array<long long, 3> key{ std::llround(p[0] / weld), std::llround(p[1] / weld), std::llround(p[2] / weld) };
						const auto [at, added] = welded.emplace(key, static_cast<std::uint32_t>(out.vertices.size()));
						if (added)
						{
							out.vertices.push_back(p);
						}
						remap.push_back(at->second);
					}

					// faces: welded loops without repeats, no zero area, no duplicate, no clockwise one
					std::set<std::vector<std::uint32_t>> seen;
					std::vector<std::vector<std::uint32_t>> faces;
					for (const auto& face : mesh.faces)
					{
						std::vector<std::uint32_t> loop;
						for (const auto v : face)
						{
							const auto w = remap[v];
							if (loop.empty() || loop.back() != w)
							{
								loop.push_back(w);
							}
						}
						while (loop.size() > 1 && loop.front() == loop.back())
						{
							loop.pop_back();
						}
						std::unordered_set<std::uint32_t> distinct(loop.begin(), loop.end());
						const auto p = points(loop, out.vertices);
						auto key = loop;
						std::sort(key.begin(), key.end());
						if (loop.size() < 3 || distinct.size() != loop.size() || std::fabs(area_xy(p)) < 1e-3 || !seen.insert(key).second
							|| area_xy(p) < 0.0)
						{
							out.faces_dropped++;
							continue;
						}
						faces.push_back(std::move(loop));
					}

					// what NavPower cannot take (not planar, not convex in XY) is split into convex planar pieces
					for (auto& loop : faces)
					{
						const auto p = points(loop, out.vertices);
						if (convex_ccw_xy(p) && planar_deviation(p) <= planar_tolerance)
						{
							out.areas.push_back(std::move(loop));
							continue;
						}
						out.faces_split++;
						std::vector<std::vector<std::uint32_t>> tris;
						for (auto& t : ear_clip(loop, out.vertices))
						{
							if (area_xy(points(t, out.vertices)) > 1e-6)
							{
								tris.push_back(std::move(t));
							}
						}
						for (auto& piece : merge(std::move(tris), out.vertices))
						{
							out.areas.push_back(std::move(piece));
						}
					}

					// BO3's Havok mesh joins ground across near-vertical rises with narrow steep areas. IW7 builds its graphs
					// with a 10 unit step (stock stepHeight), so its zombies would path up them and get stuck at the foot.
					// Such an area is left out; its neighbours' edges become boundaries.
					std::erase_if(out.areas, [&](const std::vector<std::uint32_t>& loop)
					{
						const auto p = points(loop, out.vertices);
						double nx = 0.0, ny = 0.0, nz = 0.0;
						auto lo = p[0][2], hi = p[0][2];
						for (std::size_t k = 0; k < p.size(); k++)
						{
							const auto& a = p[k];
							const auto& b = p[(k + 1) % p.size()];
							nx += (static_cast<double>(a[1]) - b[1]) * (static_cast<double>(a[2]) + b[2]);
							ny += (static_cast<double>(a[2]) - b[2]) * (static_cast<double>(a[0]) + b[0]);
							nz += (static_cast<double>(a[0]) - b[0]) * (static_cast<double>(a[1]) + b[1]);
							lo = std::min(lo, a[2]);
							hi = std::max(hi, a[2]);
						}
						const auto slope = std::atan2(std::hypot(nx, ny), std::fabs(nz)) * 180.0 / std::numbers::pi;
						const auto steep = slope > max_area_slope && hi - lo > max_area_rise;
						out.steep_dropped += steep ? 1 : 0;
						return steep;
					});

					// adjacency: an edge u -> v and exactly one v -> u
					std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::pair<std::uint32_t, std::uint32_t>>> directed;
					for (std::uint32_t a = 0; a < out.areas.size(); a++)
					{
						const auto& loop = out.areas[a];
						for (std::uint32_t k = 0; k < loop.size(); k++)
						{
							directed[{ loop[k], loop[(k + 1) % loop.size()] }].emplace_back(a, k);
						}
					}
					out.adjacent.resize(out.areas.size());
					for (std::uint32_t a = 0; a < out.areas.size(); a++)
					{
						out.adjacent[a].assign(out.areas[a].size(), -1);
					}
					for (const auto& [edge, owners] : directed)
					{
						const auto twin = directed.find({ edge.second, edge.first });
						if (twin == directed.end())
						{
							continue;
						}
						if (owners.size() == 1 && twin->second.size() == 1)
						{
							out.adjacent[owners[0].first][owners[0].second] = static_cast<std::int32_t>(twin->second[0].first);
						}
						else
						{
							out.edge_conflicts++;
						}
					}

					// components breadth first (the storage order), then clusters of at most 50 grown inside each
					const auto n = out.areas.size();
					std::vector<std::uint8_t> visited(n, 0);
					out.cluster.assign(n, no_edge);
					for (std::uint32_t s = 0; s < n; s++)
					{
						if (visited[s])
						{
							continue;
						}
						std::vector<std::uint32_t> component;
						std::deque<std::uint32_t> queue{ s };
						visited[s] = 1;
						while (!queue.empty())
						{
							const auto x = queue.front();
							queue.pop_front();
							component.push_back(x);
							for (const auto y : out.adjacent[x])
							{
								if (y >= 0 && !visited[y])
								{
									visited[y] = 1;
									queue.push_back(static_cast<std::uint32_t>(y));
								}
							}
						}
						out.component.resize(n, 0);
						for (const auto x : component)
						{
							out.component[x] = static_cast<std::uint32_t>(out.components);
						}
						out.components++;
						out.largest_component = std::max(out.largest_component, component.size());
						out.order.insert(out.order.end(), component.begin(), component.end());

						for (const auto seed : component)
						{
							if (out.cluster[seed] != no_edge)
							{
								continue;
							}
							const auto id = static_cast<std::uint32_t>(out.clusters++);
							std::size_t members = 0;
							std::deque<std::uint32_t> grow{ seed };
							out.cluster[seed] = id;
							while (!grow.empty() && members < cluster_size)
							{
								const auto x = grow.front();
								grow.pop_front();
								members++;
								for (const auto y : out.adjacent[x])
								{
									if (y >= 0 && out.cluster[y] == no_edge && members + grow.size() < cluster_size)
									{
										out.cluster[y] = id;
										grow.push_back(static_cast<std::uint32_t>(y));
									}
								}
							}
						}
					}
					return out;
				}

				template <typename T>
				void put(std::vector<std::uint8_t>& out, const T& value)
				{
					const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
					out.insert(out.end(), bytes, bytes + sizeof(T));
				}

				template <typename T>
				void put_at(std::vector<std::uint8_t>& out, const std::size_t at, const T& value)
				{
					std::memcpy(out.data() + at, &value, sizeof(T));
				}

				std::vector<std::uint8_t> navpower_image(const nav_areas& nav)
				{
					std::unordered_map<std::uint32_t, std::uint64_t> offsets; // area -> offset from the graph start
					std::uint64_t at = 0x150;
					for (const auto a : nav.order)
					{
						offsets[a] = at;
						at += 72 + 32 * nav.areas[a].size();
					}
					const auto area_bytes = static_cast<std::uint32_t>(at - 0x150);

					std::vector<std::uint8_t> areas;
					vec3 lo{ FLT_MAX, FLT_MAX, FLT_MAX };
					vec3 hi{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
					for (const auto a : nav.order)
					{
						const auto p = points(nav.areas[a], nav.vertices);
						for (const auto& v : p)
						{
							for (auto k = 0; k < 3; k++)
							{
								lo[k] = std::min(lo[k], v[k]);
								hi[k] = std::max(hi[k], v[k]);
							}
						}
						const auto c = centroid(p);
						auto radius = 0.0;
						for (const auto& v : p)
						{
							radius = std::max(radius, length(sub(c, v)));
						}
						for (auto i = 0; i < 4; i++)
						{
							put<std::uint64_t>(areas, 0); // runtime pointers
						}
						put(areas, c);
						put(areas, static_cast<float>(radius));
						put<std::uint32_t>(areas, 0xFFFFFFFFu); // searchCost
						put<std::uint32_t>(areas, 0); // usageFlags
						put<std::uint32_t>(areas, (static_cast<std::uint32_t>(p.size()) & 0x7Fu) | (0x3FFFFu << 7)); // edges, island unset
						put<std::uint32_t>(areas, 0x00110000u | (basis_vertex(p) << 24)); // cost multipliers 1, basis vertex
						put<std::uint32_t>(areas, 0);
						put<std::uint32_t>(areas, 0);
						for (std::size_t k = 0; k < p.size(); k++)
						{
							const auto neighbour = nav.adjacent[a][k];
							const auto& v0 = p[k];
							const auto& v1 = p[(k + 1) % p.size()];
							if (neighbour >= 0)
							{
								put<std::uint64_t>(areas, offsets.at(static_cast<std::uint32_t>(neighbour)));
								put(areas, v0);
								put<std::uint32_t>(areas, nav.cluster[neighbour] != nav.cluster[a] ? 0xFFFF8000u : 0xFFFF0000u);
								put<std::uint32_t>(areas, static_cast<std::uint32_t>(length(sub(v0, v1)) * 1000.0 / build_scale) + 1);
							}
							else
							{
								put<std::uint64_t>(areas, 0);
								put(areas, v0);
								put<std::uint32_t>(areas, 0xFFFF0000u);
								put<std::uint32_t>(areas, 0);
							}
							put<std::uint32_t>(areas, 0);
						}
					}
					if (areas.size() != area_bytes)
					{
						throw std::runtime_error("NavPower areas: size mismatch");
					}

					// KD tree: split the longest extent of the node's vertex box, balanced by count (by centroid)
					std::vector<vec3> area_min(nav.areas.size()), area_max(nav.areas.size());
					std::vector<vec3> area_mid(nav.areas.size());
					for (std::uint32_t a = 0; a < nav.areas.size(); a++)
					{
						area_min[a] = { FLT_MAX, FLT_MAX, FLT_MAX };
						area_max[a] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
						vec3 sum{};
						for (const auto v : nav.areas[a])
						{
							for (auto k = 0; k < 3; k++)
							{
								area_min[a][k] = std::min(area_min[a][k], nav.vertices[v][k]);
								area_max[a][k] = std::max(area_max[a][k], nav.vertices[v][k]);
								sum[k] += nav.vertices[v][k];
							}
						}
						for (auto k = 0; k < 3; k++)
						{
							area_mid[a][k] = sum[k] / static_cast<float>(nav.areas[a].size());
						}
					}
					std::vector<std::uint8_t> kd;
					const auto bounds = [&](const std::uint32_t* items, const std::size_t count, vec3& bmin, vec3& bmax)
					{
						bmin = { FLT_MAX, FLT_MAX, FLT_MAX };
						bmax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
						for (std::size_t i = 0; i < count; i++)
						{
							for (auto k = 0; k < 3; k++)
							{
								bmin[k] = std::min(bmin[k], area_min[items[i]][k]);
								bmax[k] = std::max(bmax[k], area_max[items[i]][k]);
							}
						}
					};
					std::function<void(std::uint32_t*, std::size_t)> build = [&](std::uint32_t* items, const std::size_t count)
					{
						if (count == 1)
						{
							put<std::uint32_t>(kd, 0x80000000u | static_cast<std::uint32_t>(offsets.at(items[0])));
							return;
						}
						vec3 bmin, bmax;
						bounds(items, count, bmin, bmax);
						auto axis = 0;
						for (auto k = 1; k < 3; k++)
						{
							if (bmax[k] - bmin[k] > bmax[axis] - bmin[axis])
							{
								axis = k;
							}
						}
						std::sort(items, items + count, [&](const std::uint32_t x, const std::uint32_t y)
						{
							return area_mid[x][axis] != area_mid[y][axis] ? area_mid[x][axis] < area_mid[y][axis] : x < y;
						});
						const auto half = (count + 1) / 2;
						vec3 lmin, lmax, rmin, rmax;
						bounds(items, half, lmin, lmax);
						bounds(items + half, count - half, rmin, rmax);
						const auto node = kd.size();
						kd.resize(kd.size() + 12);
						build(items, half);
						const auto right = static_cast<std::uint32_t>(kd.size() - node);
						build(items + half, count - half);
						if (right >= (1u << 28))
						{
							throw std::runtime_error("NavPower KD tree: offset overflow");
						}
						put_at<std::uint32_t>(kd, node, (static_cast<std::uint32_t>(axis) << 28) | right);
						put_at<float>(kd, node + 4, lmax[axis] + kd_epsilon);
						put_at<float>(kd, node + 8, rmin[axis] - kd_epsilon);
					};
					auto items = nav.order;
					build(items.data(), items.size());
					const auto count = nav.order.size();
					if (kd.size() != 12 * (count - 1) + 4 * count)
					{
						throw std::runtime_error("NavPower KD tree: size mismatch");
					}

					std::vector<std::uint8_t> graph;
					const auto kd_bytes = static_cast<std::uint32_t>(28 + kd.size());
					put<std::uint32_t>(graph, 43); // version
					put<std::uint32_t>(graph, 0); // layer
					put<std::uint32_t>(graph, area_bytes);
					put<std::uint32_t>(graph, kd_bytes);
					put<std::uint32_t>(graph, 0);
					put<std::uint32_t>(graph, 0);
					put<std::uint32_t>(graph, 56);
					put<std::uint32_t>(graph, 0); // link records: off-mesh links come from aipaths
					put<std::uint32_t>(graph, 0x150 + area_bytes + kd_bytes);
					put<float>(graph, build_scale);
					put<float>(graph, 5.0f); // voxel size
					put<float>(graph, 15.0f); // agent radius
					put<float>(graph, 10.0f); // step height
					put<float>(graph, 72.0f); // agent height
					put(graph, lo);
					put(graph, hi);
					put<std::uint32_t>(graph, 2); // up axis Z
					graph.resize(graph.size() + 124, 0);
					graph.insert(graph.end(), std::begin(navpower_constant), std::end(navpower_constant));
					if (graph.size() != 0x150)
					{
						throw std::runtime_error("NavPower header: size mismatch");
					}
					graph.insert(graph.end(), areas.begin(), areas.end());
					for (auto k = 0; k < 3; k++)
					{
						put<float>(graph, lo[k] - kd_epsilon);
					}
					for (auto k = 0; k < 3; k++)
					{
						put<float>(graph, hi[k] + kd_epsilon);
					}
					put<std::uint32_t>(graph, static_cast<std::uint32_t>(kd.size()));
					graph.insert(graph.end(), kd.begin(), kd.end());

					std::vector<std::uint8_t> payload;
					put<std::uint32_t>(payload, 0x10000); // section id
					put<std::uint32_t>(payload, static_cast<std::uint32_t>(12 + graph.size()));
					put<std::uint32_t>(payload, 1);
					put<std::uint32_t>(payload, 0); // NavSet: endian flag, version 43, one graph
					put<std::uint32_t>(payload, 43);
					put<std::uint32_t>(payload, 1);
					payload.insert(payload.end(), graph.begin(), graph.end());

					std::uint32_t checksum = 0;
					for (std::size_t i = 0; i + 4 <= payload.size(); i += 4)
					{
						std::uint32_t word;
						std::memcpy(&word, payload.data() + i, 4);
						checksum += word;
					}
					std::vector<std::uint8_t> image;
					put<std::uint32_t>(image, 0); // endian flag
					put<std::uint32_t>(image, 2); // version
					put<std::uint32_t>(image, static_cast<std::uint32_t>(payload.size()));
					put<std::uint32_t>(image, checksum);
					put<std::uint32_t>(image, 0); // runtime flags
					put<std::uint32_t>(image, 0x01000001u); // constant flags
					image.insert(image.end(), payload.begin(), payload.end());
					return image;
				}

				// ---- off-mesh links: BO3's traversals -----------------------------------------------------

				// One directed traversal. BO3 pairs a node_negotiation_begin with the node_negotiation_end it targets, and
				// each carries the animscript of the move that starts on it (e.g. jump_up_128 on a begin with
				// jump_down_128 on its end), so a pair is up to two links.
				struct traversal
				{
					vec3 start;
					vec3 end;
					float yaw; // degrees, the start node's
					std::string bo3; // BO3 animscript
					std::string state; // IW7 zombie ASM state
				};

				vec3 parse_vec3(const std::string* text)
				{
					vec3 v{};
					if (text)
					{
						std::istringstream in(*text);
						in >> v[0] >> v[1] >> v[2];
					}
					return v;
				}

				// "jump_up<a>_down<b>": up over a ledge then down, a and b in inches
				bool over_ledge(const std::string& anim)
				{
					constexpr std::string_view up = "jump_up";
					if (!anim.starts_with(up))
					{
						return false;
					}
					auto at = up.size();
					const auto digits = [&]()
					{
						const auto first = at;
						while (at < anim.size() && std::isdigit(static_cast<unsigned char>(anim[at])))
						{
							at++;
						}
						return at > first;
					};
					if (!digits() || anim.compare(at, 5, "_down") != 0)
					{
						return false;
					}
					at += 5;
					return digits() && at == anim.size();
				}

				// The IW7 zombie ASM state (zombie_asm_animclass) for a BO3 traversal, by what the move does: dz is the
				// end's height over the start's, run the horizontal distance. The thresholds follow the dz ranges stock
				// aipaths use each state for; jump_down_fast is the jump down the zombie script scales to any drop. No
				// state for BO3's teleport moves.
				std::optional<std::string> iw7_state(const std::string& anim, const float dz, const float run)
				{
					if (anim.empty() || anim == "teleport")
					{
						return {};
					}
					if (anim.starts_with("mantle_over_"))
					{
						return std::atoi(anim.data() + 12) <= 48 ? "mantle_40_over_extended" : "wall_over_40_flex";
					}
					if (anim.starts_with("jump_across"))
					{
						return run > 300.0f ? "jump_across_196" : "jump_across_100";
					}
					if (anim.find("dockrailing") != std::string::npos)
					{
						return dz >= 0.0f ? "jump_up_128_over_40" : "over_40_down_128";
					}
					if (over_ledge(anim))
					{
						if (dz >= 100.0f)
						{
							return "jump_up_128_over_40";
						}
						if (dz > 20.0f)
						{
							return "jump_up_56_over_40";
						}
						if (dz >= -20.0f)
						{
							return "wall_over_40";
						}
						return dz > -100.0f ? "jump_over_30_out_30_down_48" : "over_40_down_128";
					}
					if (anim.starts_with("jump_down"))
					{
						return dz >= -48.0f ? "jump_down_40" : "jump_down_fast";
					}
					if (anim.starts_with("jump_up") || anim.find("climb_up") != std::string::npos)
					{
						if (dz <= 44.0f)
						{
							return "jump_up_40";
						}
						return dz <= 100.0f ? "jump_up_56" : "jump_up_128";
					}
					return {};
				}

				std::vector<traversal> read_traversals(const std::string& t7_entities, std::map<std::string, unsigned int>& skipped)
				{
					const auto ents = map_entities::parse(t7_entities);
					std::unordered_map<std::string, const map_entities::entity*> ends;
					for (const auto& e : ents)
					{
						const auto* targetname = e.get("targetname");
						if (e.is("classname", "node_negotiation_end") && targetname)
						{
							ends[*targetname] = &e;
						}
					}

					std::vector<traversal> out;
					const auto add = [&](const map_entities::entity& from, const map_entities::entity& to)
					{
						const auto* anim = from.get("animscript");
						const auto start = parse_vec3(from.get("origin"));
						const auto end = parse_vec3(to.get("origin"));
						const auto dz = end[2] - start[2];
						const auto run = static_cast<float>(std::hypot(end[0] - start[0], end[1] - start[1]));
						const auto state = iw7_state(anim ? *anim : std::string{}, dz, run);
						if (!state)
						{
							skipped[anim && !anim->empty() ? *anim : "(no animscript)"]++;
							return;
						}
						out.push_back({ start, end, parse_vec3(from.get("angles"))[1], anim ? *anim : std::string{}, *state });
					};
					for (const auto& e : ents)
					{
						if (!e.is("classname", "node_negotiation_begin"))
						{
							continue;
						}
						const auto* target = e.get("target");
						const auto found = target ? ends.find(*target) : ends.end();
						if (found == ends.end())
						{
							skipped["(begin without an end)"]++;
							continue;
						}
						add(e, *found->second);
						add(*found->second, e);
					}
					return out;
				}

				// ---- aipaths: the links' negotiation nodes ------------------------------------------------

				struct node_tree
				{
					int axis = -1;
					float dist = 0.0f;
					std::size_t child[2]{};
					std::vector<std::uint16_t> nodes;
				};

				// the node tree of the stock aipaths (root first, children after their parent): a leaf under 4 nodes or
				// where the larger XY extent is at most 192, else split at that extent's midpoint
				std::size_t build_tree(std::vector<node_tree>& tree, std::vector<std::uint16_t> items, const std::vector<vec3>& origins)
				{
					const auto index = tree.size();
					tree.emplace_back();
					if (items.size() >= 4)
					{
						float lo[2] = { FLT_MAX, FLT_MAX }, hi[2] = { -FLT_MAX, -FLT_MAX };
						for (const auto i : items)
						{
							for (auto k = 0; k < 2; k++)
							{
								lo[k] = std::min(lo[k], origins[i][k]);
								hi[k] = std::max(hi[k], origins[i][k]);
							}
						}
						const auto axis = (hi[1] - lo[1]) > (hi[0] - lo[0]) ? 1 : 0;
						if (hi[axis] - lo[axis] > 192.0f)
						{
							const auto dist = (hi[axis] + lo[axis]) * 0.5f;
							std::vector<std::uint16_t> left, right;
							for (const auto i : items)
							{
								(origins[i][axis] < dist ? left : right).push_back(i);
							}
							if (!left.empty() && !right.empty())
							{
								const auto l = build_tree(tree, std::move(left), origins);
								const auto r = build_tree(tree, std::move(right), origins);
								tree[index].axis = axis;
								tree[index].dist = dist;
								tree[index].child[0] = l;
								tree[index].child[1] = r;
								return index;
							}
						}
					}
					tree[index].nodes = std::move(items);
					return index;
				}

				// Fixed nodes: a negotiation begin (target, animscript, a negotiation link to its end) and end
				// (targetname) per link, like stock CP; then the 64 dynamic slots and one zone node, as the stock
				// layout counts them (nodeCount = fixed + maxDynamicSpawnedNodeCount + zoneCount); visibility, exposure
				// and zone bytes sized as stock and empty: zombies path on the navmesh.
				void write_aipaths(const std::string& bsp_name, const std::vector<traversal>& links, const std::vector<std::string>& names)
				{
					namespace iw7 = zonetool::iw7;
					constexpr auto dynamic = 64u;
					const auto fixed = static_cast<std::uint32_t>(links.size() * 2);
					const auto count = fixed + dynamic + 1;
					if (count > 0xFFFF)
					{
						throw std::runtime_error(utils::string::va("%u path nodes", count));
					}

					std::vector<iw7::pathnode_t> nodes(count);
					std::vector<iw7::pathlink_s> negotiation(links.size());
					std::vector<vec3> origins(fixed);
					for (auto& n : nodes)
					{
						n.constant.wOverlapNode[0] = 0xFFFF;
						n.constant.wOverlapNode[1] = 0xFFFF;
					}
					for (std::size_t k = 0; k < links.size(); k++)
					{
						const auto& l = links[k];
						const auto yaw = l.yaw * std::numbers::pi_v<float> / 180.0f;
						auto& begin = nodes[2 * k].constant;
						auto& end = nodes[2 * k + 1].constant;
						begin.type = iw7::NODE_NEGOTIATION_BEGIN;
						end.type = iw7::NODE_NEGOTIATION_END;
						for (auto* c : { &begin, &end })
						{
							c->spawnflags = 0x8000;
							c->orientation.yaw_orient.fLocalAngle = l.yaw;
							c->orientation.yaw_orient.localForward[0] = std::cos(yaw);
							c->orientation.yaw_orient.localForward[1] = std::sin(yaw);
						}
						std::memcpy(begin.vLocalOrigin, l.start.data(), 12);
						std::memcpy(end.vLocalOrigin, l.end.data(), 12);
						origins[2 * k] = l.start;
						origins[2 * k + 1] = l.end;
						negotiation[k].fDist = static_cast<float>(std::hypot(l.end[0] - l.start[0], l.end[1] - l.start[1]));
						negotiation[k].nodeNum = static_cast<unsigned short>(2 * k + 1);
						negotiation[k].negotiationLink = 1;
						begin.totalLinkCount = 1;
						begin.Links = &negotiation[k];
					}
					auto& zone = nodes[fixed + dynamic].constant;
					zone.spawnflags = 0x100;
					if (fixed)
					{
						std::memcpy(zone.vLocalOrigin, origins[0].data(), 12);
					}

					std::vector<std::uint16_t> items(fixed);
					for (std::uint32_t i = 0; i < fixed; i++)
					{
						items[i] = static_cast<std::uint16_t>(i);
					}
					std::vector<node_tree> tree;
					build_tree(tree, std::move(items), origins);
					std::vector<iw7::pathnode_tree_t> trees(tree.size());
					for (std::size_t i = 0; i < tree.size(); i++)
					{
						trees[i].axis = tree[i].axis;
						trees[i].dist = tree[i].dist;
						if (tree[i].axis < 0)
						{
							trees[i].u.s.nodeCount = static_cast<int>(tree[i].nodes.size());
							trees[i].u.s.nodes = tree[i].nodes.data();
						}
						else
						{
							trees[i].u.child[0] = &trees[tree[i].child[0]];
							trees[i].u.child[1] = &trees[tree[i].child[1]];
						}
					}

					std::vector<char> vis((static_cast<std::size_t>(count) * (count - 1) / 2 + 7) / 8, 0);
					std::vector<char> exposure((static_cast<std::size_t>(count) * 265 + 7) / 8, 0);
					std::vector<char> no_peek(static_cast<std::size_t>(count) * 2, 0);
					std::vector<char> zones(count, 0);

					iw7::PathData data{};
					data.name = bsp_name.data();
					data.nodeCount = count;
					data.nodes = nodes.data();
					data.parentIndexResolved = false;
					data.version = 22;
					data.visBytes = static_cast<int>(vis.size());
					data.pathVis = vis.data();
					data.nodeTreeCount = static_cast<int>(trees.size());
					data.nodeTree = trees.data();
					data.exposureBytes = static_cast<int>(exposure.size());
					data.pathExposure = exposure.data();
					data.noPeekVisBytes = static_cast<int>(no_peek.size());
					data.pathNoPeekVis = no_peek.data();
					data.zoneCount = 1;
					data.zonesBytes = static_cast<int>(zones.size());
					data.pathZones = zones.data();
					data.fixedNodeCount = static_cast<int>(fixed);
					data.maxDynamicSpawnedNodeCount = static_cast<int>(dynamic);

					// the records of x64-zt's path_data::dump / parse, with this conversion's strings
					assetmanager::dumper write;
					if (!write.open(bsp_name + ".aipaths"))
					{
						throw std::runtime_error(utils::string::va("could not write %s.aipaths", bsp_name.data()));
					}
					write.dump_single(&data);
					write.dump_string(data.name);
					write.dump_array(data.nodes, data.nodeCount);
					for (std::uint32_t i = 0; i < count; i++)
					{
						const auto link = i < fixed ? i / 2 : 0;
						const auto begin = i < fixed && i % 2 == 0;
						const auto end = i < fixed && i % 2 == 1;
						write.dump_string(end ? names[link].data() : nullptr); // targetname
						write.dump_string(static_cast<const char*>(nullptr)); // script_linkName
						write.dump_string(static_cast<const char*>(nullptr)); // script_noteworthy
						write.dump_string(begin ? names[link].data() : nullptr); // target
						write.dump_string(begin ? links[link].state.data() : nullptr); // animscript
						write.dump_string(static_cast<const char*>(nullptr)); // parent
						write.dump_array(nodes[i].constant.Links, nodes[i].constant.totalLinkCount);
					}
					write.dump_array(data.pathVis, data.visBytes);
					write.dump_array(data.nodeTree, data.nodeTreeCount);
					const std::function<void(iw7::pathnode_tree_t*)> children = [&](iw7::pathnode_tree_t* t)
					{
						write.dump_single(t);
						if (t->axis < 0)
						{
							write.dump_array(t->u.s.nodes, t->u.s.nodeCount);
							return;
						}
						children(t->u.child[0]);
						children(t->u.child[1]);
					};
					for (auto& t : trees)
					{
						if (t.axis < 0)
						{
							write.dump_array(t.u.s.nodes, t.u.s.nodeCount);
							continue;
						}
						children(t.u.child[0]);
						children(t.u.child[1]);
					}
					write.dump_array(static_cast<iw7::PathDynamicNodeGroup*>(nullptr), 0);
					write.dump_array(data.pathExposure, data.exposureBytes);
					write.dump_array(data.pathNoPeekVis, data.noPeekVisBytes);
					write.dump_array(data.pathZones, data.zonesBytes);
					write.dump_array(static_cast<char*>(nullptr), 0); // dynamic states
					write.close();
				}

				// re-reads the image: the walk over the areas, symmetric adjacency with reversed endpoints
				void check_image(const std::vector<std::uint8_t>& image)
				{
					const auto u32 = [&](const std::size_t at)
					{
						std::uint32_t v;
						std::memcpy(&v, image.data() + at, 4);
						return v;
					};
					constexpr std::size_t g = 0x30;
					const auto area_bytes = u32(g + 8);
					struct edge
					{
						std::uint64_t adjacent;
						vec3 pos;
						std::uint32_t flags;
					};
					std::map<std::uint64_t, std::vector<edge>> areas;
					std::uint64_t at = 0x150;
					while (at < 0x150 + area_bytes)
					{
						const auto count = u32(g + at + 56) & 0x7Fu;
						auto& edges = areas[at];
						for (auto k = 0u; k < count; k++)
						{
							edge e{};
							std::memcpy(&e.adjacent, image.data() + g + at + 72 + 32 * k, 8);
							std::memcpy(e.pos.data(), image.data() + g + at + 72 + 32 * k + 8, 12);
							e.flags = u32(g + at + 72 + 32 * k + 20);
							edges.push_back(e);
						}
						at += 72 + 32 * count;
					}
					if (at != 0x150 + area_bytes)
					{
						throw std::runtime_error("NavPower check: the area walk does not end at the KD tree");
					}
					for (const auto& [offset, edges] : areas)
					{
						for (std::size_t k = 0; k < edges.size(); k++)
						{
							if (!edges[k].adjacent)
							{
								continue;
							}
							const auto other = areas.find(edges[k].adjacent);
							if (other == areas.end())
							{
								throw std::runtime_error("NavPower check: an edge points past the areas");
							}
							const auto& next = edges[(k + 1) % edges.size()].pos;
							auto twin = false;
							for (std::size_t j = 0; j < other->second.size(); j++)
							{
								const auto& e = other->second[j];
								twin |= e.adjacent == offset && e.pos == next && other->second[(j + 1) % other->second.size()].pos == edges[k].pos
									&& e.flags == edges[k].flags;
							}
							if (!twin)
							{
								throw std::runtime_error("NavPower check: an adjacency is not mirrored");
							}
						}
						std::vector<vec3> p;
						for (const auto& e : edges)
						{
							p.push_back(e.pos);
						}
						if (!convex_ccw_xy(p) || planar_deviation(p) > planar_tolerance + 1e-3)
						{
							throw std::runtime_error("NavPower check: an area is not convex, CCW and planar");
						}
					}
				}
			}

			std::vector<map_entities::entity> convert(const NavMeshData* asset, const std::string& bsp_name, const std::string& t7_entities)
			{
				const auto mesh = read_mesh(asset->blobs[3]);
				const auto nav = build_areas(mesh);
				const auto image = navpower_image(nav);
				check_image(image);

				std::map<std::string, unsigned int> skipped;
				const auto links = read_traversals(t7_entities, skipped);
				std::vector<std::string> names;
				std::vector<zonetool::iw7::nav_link_creation_data_s> records(links.size());
				std::vector<map_entities::entity> nodes;
				std::map<std::string, unsigned int> states;
				for (std::size_t k = 0; k < links.size(); k++)
				{
					const auto& l = links[k];
					names.push_back(utils::string::va("t7_traversal_%zu", k));
					auto& r = records[k];
					std::memcpy(r.m_Start.m_Pt1, l.start.data(), 12);
					std::memcpy(r.m_Start.m_Pt2, l.start.data(), 12);
					std::memcpy(r.m_End.m_Pt1, l.end.data(), 12);
					std::memcpy(r.m_End.m_Pt2, l.end.data(), 12);
					r.m_UsageFlags = 0xEF; // stock's value for a negotiation node without a unit type: zombies (0x40) included
					r.m_PenaltyMult = 1.0f;
					r.m_bBidirectional = false;
					states[l.state]++;

					const auto angles = utils::string::va("0 %g 0", l.yaw);
					map_entities::entity begin{}, end{};
					begin.set("classname", "node_negotiation_begin");
					begin.set("origin", utils::string::va("%g %g %g", l.start[0], l.start[1], l.start[2]));
					begin.set("angles", angles);
					begin.set("target", names.back());
					begin.set("animscript", l.state);
					end.set("classname", "node_negotiation_end");
					end.set("origin", utils::string::va("%g %g %g", l.end[0], l.end[1], l.end[2]));
					end.set("angles", angles);
					end.set("targetname", names.back());
					nodes.push_back(std::move(begin));
					nodes.push_back(std::move(end));
				}
				if (!links.empty())
				{
					write_aipaths(bsp_name, links, names);
				}

				zonetool::iw7::nav_resource_s resource{};
				resource.modelIdx = -1;
				resource.graphSize = static_cast<int>(image.size());
				resource.pGraphBuffer = reinterpret_cast<char*>(const_cast<std::uint8_t*>(image.data()));

				zonetool::iw7::NavMeshData data{};
				data.name = bsp_name.data();
				data.version = 15;
				data.numNavResources = 1;
				data.navResources = &resource;
				// the reader takes the link records' strings by this count (x64-zt iw7 nav_mesh::parse)
				data.numLinkCreationData = static_cast<int>(records.size());
				data.linkCreationData = records.data();

				assetmanager::dumper write;
				if (!write.open(bsp_name + ".navmesh"))
				{
					throw std::runtime_error(utils::string::va("could not write %s.navmesh", bsp_name.data()));
				}
				write.dump_single(&data);
				write.dump_string(data.name);
				write.dump_array(data.navResources, data.numNavResources);
				write.dump_string(static_cast<const char*>(nullptr)); // targetName
				write.dump_array(resource.pGraphBuffer, resource.graphSize);
				write.dump_array(static_cast<zonetool::iw7::nav_obstacle_bounds_s*>(nullptr), 0);
				write.dump_array(static_cast<zonetool::iw7::nav_glass_bounds_s*>(nullptr), 0);
				write.dump_array(static_cast<zonetool::iw7::nav_modifier_s*>(nullptr), 0);
				write.dump_array(records.data(), static_cast<std::uint32_t>(records.size()));
				for (std::size_t k = 0; k < records.size(); k++)
				{
					write.dump_string(links[k].state.data()); // m_Animscript
					write.dump_string(names[k].data()); // m_Target: the begin node's target
					write.dump_string(static_cast<const char*>(nullptr)); // m_Parent
				}
				write.dump_array(static_cast<float*>(nullptr), 0); // volume seeds
				write.dump_array(static_cast<zonetool::iw7::nav_raw_volume_s*>(nullptr), 0);
				write.dump_array(static_cast<zonetool::iw7::nav_raw_custom_volume_s*>(nullptr), 0);
				write.close();

				// ZT_NAV_EXPORT=<file>: the areas (vertex loops) and their components as json, for offline checks
				if (const char* out_path = std::getenv("ZT_NAV_EXPORT"))
				{
					if (auto* f = std::fopen(out_path, "w"))
					{
						std::fprintf(f, "{\"vertices\":[");
						for (std::size_t k = 0; k < nav.vertices.size(); k++)
						{
							std::fprintf(f, "%s[%g,%g,%g]", k ? "," : "", nav.vertices[k][0], nav.vertices[k][1], nav.vertices[k][2]);
						}
						std::fprintf(f, "],\"areas\":[");
						for (std::size_t a = 0; a < nav.areas.size(); a++)
						{
							std::fprintf(f, "%s{\"c\":%u,\"v\":[", a ? "," : "", nav.component[a]);
							for (std::size_t k = 0; k < nav.areas[a].size(); k++)
							{
								std::fprintf(f, "%s%u", k ? "," : "", nav.areas[a][k]);
							}
							std::fprintf(f, "]}");
						}
						std::fprintf(f, "]}");
						std::fclose(f);
					}
				}

				ZONETOOL_INFO("navmesh \"%s\": %zu BO3 faces (%zu split into planar convex areas, %zu dropped, %zu steep rises left out) -> %zu NavPower areas in %zu "
					"components (largest %zu), %zu clusters, %zu edge conflicts left open, %zu-byte image",
					bsp_name.data(), mesh.faces.size(), nav.faces_split, nav.faces_dropped, nav.steep_dropped, nav.areas.size(), nav.components,
					nav.largest_component, nav.clusters, nav.edge_conflicts, image.size());

				std::string state_text, skipped_text;
				for (const auto& [state, count] : states)
				{
					state_text += utils::string::va(" %s %u", state.data(), count);
				}
				for (const auto& [anim, count] : skipped)
				{
					skipped_text += utils::string::va(" %s %u", anim.data(), count);
				}
				ZONETOOL_INFO("navmesh \"%s\": %zu off-mesh links from BO3's traversals (%zu aipaths negotiation nodes):%s; no link for:%s",
					bsp_name.data(), links.size(), nodes.size(), state_text.data(), skipped_text.empty() ? " -" : skipped_text.data());
				return nodes;
			}
		}
	}
}
