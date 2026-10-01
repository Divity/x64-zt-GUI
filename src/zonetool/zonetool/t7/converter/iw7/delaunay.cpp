#include <std_include.hpp>
#include "delaunay.hpp"

namespace zonetool::t7
{
	namespace converter::iw7::delaunay
	{
		namespace
		{
			// dot(cross(b - a, c - a), d - a)
			double orient(const point& a, const point& b, const point& c, const point& d)
			{
				const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
				const double v[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
				const double w[3] = { d[0] - a[0], d[1] - a[1], d[2] - a[2] };
				return (u[1] * v[2] - u[2] * v[1]) * w[0] + (u[2] * v[0] - u[0] * v[2]) * w[1] + (u[0] * v[1] - u[1] * v[0]) * w[2];
			}

			double det3(const double* x, const double* y, const double* z)
			{
				return x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0]) + x[2] * (y[0] * z[1] - y[1] * z[0]);
			}

			// the lifted determinant: negative when e lies inside the circumsphere of the positively oriented
			// a, b, c, d (checked numerically for this orientation)
			double in_sphere(const point& a, const point& b, const point& c, const point& d, const point& e)
			{
				double r[4][4];
				const point* p[4] = { &a, &b, &c, &d };
				for (auto i = 0; i < 4; i++)
				{
					for (auto k = 0; k < 3; k++)
					{
						r[i][k] = (*p[i])[k] - e[k];
					}
					r[i][3] = r[i][0] * r[i][0] + r[i][1] * r[i][1] + r[i][2] * r[i][2];
				}
				return -r[0][3] * det3(r[1], r[2], r[3]) + r[1][3] * det3(r[0], r[2], r[3])
					- r[2][3] * det3(r[0], r[1], r[3]) + r[3][3] * det3(r[0], r[1], r[2]);
			}

			std::uint64_t spread(std::uint64_t v)
			{
				v &= 0x1FFFFF;
				v = (v | v << 32) & 0x1F00000000FFFFull;
				v = (v | v << 16) & 0x1F0000FF0000FFull;
				v = (v | v << 8) & 0x100F00F00F00F00Full;
				v = (v | v << 4) & 0x10C30C30C30C30C3ull;
				v = (v | v << 2) & 0x1249249249249249ull;
				return v;
			}

			std::uint64_t face_key(std::uint32_t a, std::uint32_t b, std::uint32_t c)
			{
				if (a > b) std::swap(a, b);
				if (b > c) std::swap(b, c);
				if (a > b) std::swap(a, b);
				return (static_cast<std::uint64_t>(a) << 42) ^ (static_cast<std::uint64_t>(b) << 21) ^ c;
			}

			struct builder
			{
				std::vector<point> pts;
				std::vector<std::array<std::uint32_t, 4>> v;
				std::vector<std::array<std::uint32_t, 4>> n;
				std::vector<std::uint8_t> alive;
				std::vector<std::uint32_t> stamp;
				std::vector<std::uint32_t> free_list;
				std::uint32_t current = 0;
				std::uint32_t last = 0;

				struct pending
				{
					std::uint64_t key;
					std::uint32_t tet;
					int face;
				};
				std::vector<std::uint32_t> cavity;
				std::vector<pending> edges;

				std::uint32_t add(const std::array<std::uint32_t, 4>& verts)
				{
					std::uint32_t t;
					if (!free_list.empty())
					{
						t = free_list.back();
						free_list.pop_back();
						v[t] = verts;
						n[t] = { none, none, none, none };
						alive[t] = 1;
						stamp[t] = 0;
					}
					else
					{
						t = static_cast<std::uint32_t>(v.size());
						v.push_back(verts);
						n.push_back({ none, none, none, none });
						alive.push_back(1);
						stamp.push_back(0);
					}
					return t;
				}

				double orient_of(const std::array<std::uint32_t, 4>& t) const
				{
					return orient(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]);
				}

				std::uint32_t locate(const point& p)
				{
					auto t = last;
					for (std::uint64_t steps = 0;; steps++)
					{
						if (steps > 50000000)
						{
							throw std::runtime_error("delaunay: point location does not terminate");
						}
						const auto first = static_cast<int>(steps & 3);
						auto next = none;
						for (auto i = 0; i < 4 && next == none; i++)
						{
							const auto k = (first + i) & 3;
							point q[4] = { pts[v[t][0]], pts[v[t][1]], pts[v[t][2]], pts[v[t][3]] };
							q[k] = p;
							if (orient(q[0], q[1], q[2], q[3]) < 0.0)
							{
								next = n[t][k];
								if (next == none)
								{
									throw std::runtime_error("delaunay: a point lies outside the super box");
								}
							}
						}
						if (next == none)
						{
							return t;
						}
						t = next;
					}
				}

				void insert(const std::uint32_t pi)
				{
					const auto p = pts[pi];
					const auto start = locate(p);
					current++;

					cavity.clear();
					cavity.push_back(start);
					stamp[start] = current;
					for (std::size_t i = 0; i < cavity.size(); i++)
					{
						const auto c = cavity[i];
						for (auto k = 0; k < 4; k++)
						{
							const auto nb = n[c][k];
							if (nb == none || stamp[nb] == current)
							{
								continue;
							}
							const auto& q = v[nb];
							if (in_sphere(pts[q[0]], pts[q[1]], pts[q[2]], pts[q[3]], p) < 0.0)
							{
								stamp[nb] = current;
								cavity.push_back(nb);
							}
						}
					}

					edges.clear();
					auto created = none;
					for (const auto c : cavity)
					{
						for (auto k = 0; k < 4; k++)
						{
							const auto nb = n[c][k];
							if (nb != none && stamp[nb] == current)
							{
								continue;
							}
							auto verts = v[c];
							verts[k] = pi;
							if (!(orient_of(verts) > 0.0))
							{
								throw std::runtime_error("delaunay: the cavity of a point is not star-shaped around it");
							}
							const auto t = add(verts);
							n[t][k] = nb;
							if (nb != none)
							{
								for (auto j = 0; j < 4; j++)
								{
									if (n[nb][j] == c)
									{
										n[nb][j] = t;
										break;
									}
								}
							}
							if (created == none)
							{
								created = t;
							}
							for (auto j = 0; j < 4; j++)
							{
								if (j == k)
								{
									continue;
								}
								// the face opposite j holds the new point and the two vertices other than j and k
								auto a = none, b = none;
								for (auto i = 0; i < 4; i++)
								{
									if (i == j || i == k)
									{
										continue;
									}
									(a == none ? a : b) = verts[i];
								}
								edges.push_back({ (static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b), t, j });
							}
						}
					}

					// every edge of the cavity boundary is shared by exactly two of its faces
					std::sort(edges.begin(), edges.end(), [](const pending& l, const pending& r) { return l.key < r.key; });
					for (std::size_t i = 0; i < edges.size(); i += 2)
					{
						if (i + 1 >= edges.size() || edges[i].key != edges[i + 1].key
							|| (i + 2 < edges.size() && edges[i + 2].key == edges[i].key))
						{
							throw std::runtime_error("delaunay: the cavity boundary of a point is not a closed surface");
						}
						n[edges[i].tet][edges[i].face] = edges[i + 1].tet;
						n[edges[i + 1].tet][edges[i + 1].face] = edges[i].tet;
					}

					for (const auto c : cavity)
					{
						alive[c] = 0;
						free_list.push_back(c);
					}
					last = created;
				}
			};
		}

		mesh tetrahedralize(const std::vector<point>& points)
		{
			const auto count = static_cast<std::uint32_t>(points.size());
			if (count < 4)
			{
				throw std::runtime_error("delaunay: fewer than four points");
			}

			builder b;
			b.pts = points;

			double lo[3] = { DBL_MAX, DBL_MAX, DBL_MAX }, hi[3] = { -DBL_MAX, -DBL_MAX, -DBL_MAX };
			for (const auto& p : points)
			{
				for (auto k = 0; k < 3; k++)
				{
					lo[k] = std::min(lo[k], p[k]);
					hi[k] = std::max(hi[k], p[k]);
				}
			}
			const auto half = std::max({ hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1.0 }) * 0.5;
			const double centre[3] = { (lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5 };
			// a box far enough out that its corners rarely take part in the hull's circumspheres
			const auto extent = half * 16.0;
			for (auto i = 0u; i < 8; i++)
			{
				b.pts.push_back({ centre[0] + ((i & 1) ? extent : -extent), centre[1] + ((i & 2) ? extent : -extent),
					centre[2] + ((i & 4) ? extent : -extent) });
			}

			// the box as six tetrahedra around its diagonal
			static constexpr int kuhn[6][4] = { { 0, 1, 3, 7 }, { 0, 3, 2, 7 }, { 0, 2, 6, 7 }, { 0, 6, 4, 7 }, { 0, 4, 5, 7 }, { 0, 5, 1, 7 } };
			std::unordered_map<std::uint64_t, std::pair<std::uint32_t, int>> open_faces;
			for (const auto& k : kuhn)
			{
				std::array<std::uint32_t, 4> verts = { count + k[0], count + k[1], count + k[2], count + k[3] };
				if (b.orient_of(verts) < 0.0)
				{
					std::swap(verts[1], verts[2]);
				}
				const auto t = b.add(verts);
				for (auto f = 0; f < 4; f++)
				{
					std::uint32_t fv[3];
					auto m = 0;
					for (auto i = 0; i < 4; i++)
					{
						if (i != f)
						{
							fv[m++] = verts[i];
						}
					}
					const auto key = face_key(fv[0], fv[1], fv[2]);
					const auto found = open_faces.find(key);
					if (found == open_faces.end())
					{
						open_faces.emplace(key, std::make_pair(t, f));
					}
					else
					{
						b.n[t][f] = found->second.first;
						b.n[found->second.first][found->second.second] = t;
						open_faces.erase(found);
					}
				}
			}

			// insertion in Morton order keeps each point near the last cavity
			std::vector<std::pair<std::uint64_t, std::uint32_t>> order(count);
			const auto scale = 2097151.0 / std::max({ hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1.0 });
			for (auto i = 0u; i < count; i++)
			{
				const auto x = static_cast<std::uint32_t>((points[i][0] - lo[0]) * scale);
				const auto y = static_cast<std::uint32_t>((points[i][1] - lo[1]) * scale);
				const auto z = static_cast<std::uint32_t>((points[i][2] - lo[2]) * scale);
				order[i] = { spread(x) | spread(y) << 1 | spread(z) << 2, i };
			}
			std::sort(order.begin(), order.end());
			for (const auto& [code, i] : order)
			{
				b.insert(i);
			}

			// drop the box, renumber, and check the result
			mesh out;
			std::vector<std::uint32_t> remap(b.v.size(), none);
			for (auto t = 0u; t < b.v.size(); t++)
			{
				if (!b.alive[t])
				{
					continue;
				}
				const auto& verts = b.v[t];
				if (verts[0] >= count || verts[1] >= count || verts[2] >= count || verts[3] >= count)
				{
					continue;
				}
				remap[t] = static_cast<std::uint32_t>(out.tets.size());
				out.tets.push_back(verts);
			}
			out.neighbours.resize(out.tets.size());
			for (auto t = 0u; t < b.v.size(); t++)
			{
				if (remap[t] == none)
				{
					continue;
				}
				for (auto k = 0; k < 4; k++)
				{
					const auto u = b.n[t][k];
					out.neighbours[remap[t]][k] = u == none ? none : remap[u];
				}
			}

			for (auto t = 0u; t < out.tets.size(); t++)
			{
				const auto& verts = out.tets[t];
				if (!(orient(points[verts[0]], points[verts[1]], points[verts[2]], points[verts[3]]) > 0.0))
				{
					throw std::runtime_error("delaunay: a tetrahedron is not positively oriented");
				}
				for (auto k = 0; k < 4; k++)
				{
					const auto u = out.neighbours[t][k];
					if (u == none)
					{
						continue;
					}
					auto back = -1;
					for (auto j = 0; j < 4; j++)
					{
						if (out.neighbours[u][j] == t)
						{
							back = j;
						}
					}
					if (back < 0)
					{
						throw std::runtime_error("delaunay: a neighbour relation is not mutual");
					}
					auto shared = 0;
					for (auto i = 0; i < 4; i++)
					{
						for (auto j = 0; j < 4; j++)
						{
							shared += i != k && j != back && verts[i] == out.tets[u][j];
						}
					}
					if (shared != 3)
					{
						throw std::runtime_error("delaunay: neighbours do not share the face opposite their vertex");
					}
				}
			}
			return out;
		}

		std::uint32_t locate(const mesh& m, const std::vector<point>& points, const point& p, std::uint32_t start)
		{
			auto t = start;
			for (std::uint64_t steps = 0; t != none; steps++)
			{
				if (steps > 1000000)
				{
					return none;
				}
				const auto& verts = m.tets[t];
				const auto first = static_cast<int>(steps & 3);
				auto next = t;
				for (auto i = 0; i < 4 && next == t; i++)
				{
					const auto k = (first + i) & 3;
					point q[4] = { points[verts[0]], points[verts[1]], points[verts[2]], points[verts[3]] };
					q[k] = p;
					if (orient(q[0], q[1], q[2], q[3]) < 0.0)
					{
						next = m.neighbours[t][k];
					}
				}
				if (next == t)
				{
					return t;
				}
				t = next;
			}
			return none;
		}

		std::array<double, 4> barycentric(const mesh& m, const std::vector<point>& points, const std::uint32_t t, const point& p)
		{
			const auto& verts = m.tets[t];
			const auto total = orient(points[verts[0]], points[verts[1]], points[verts[2]], points[verts[3]]);
			std::array<double, 4> out{};
			for (auto k = 0; k < 4; k++)
			{
				point q[4] = { points[verts[0]], points[verts[1]], points[verts[2]], points[verts[3]] };
				q[k] = p;
				out[k] = orient(q[0], q[1], q[2], q[3]) / total;
			}
			return out;
		}
	}
}
