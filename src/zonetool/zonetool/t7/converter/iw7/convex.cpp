#include <std_include.hpp>
#include "convex.hpp"

#include <numeric>

namespace zonetool::t7
{
	namespace converter::iw7::convex
	{
		namespace
		{
			constexpr double plane_epsilon = 0.01;
			constexpr double weld_epsilon = 0.05;
			constexpr double hull_extent = 131072.0;

			// The clipping runs in double: in float the huge base quad leaves ~0.01 of rounding per clip, which
			// planes meeting at a shallow angle amplify past the weld distance.
			using point = std::array<double, 3>;
			using dplane = std::array<double, 4>;

			double dot(const double* a, const double* b)
			{
				return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
			}

			// a quad on the plane, far larger than any map
			std::vector<point> base_winding(const dplane& p)
			{
				auto axis = 0;
				for (auto i = 1; i < 3; i++)
				{
					if (std::fabs(p[i]) < std::fabs(p[axis]))
					{
						axis = i;
					}
				}
				double up[3] = { 0.0, 0.0, 0.0 };
				up[axis] = 1.0;
				const auto d = dot(up, p.data());
				for (auto i = 0; i < 3; i++)
				{
					up[i] -= d * p[i];
				}
				const auto len = std::sqrt(dot(up, up));
				if (len < 1e-6)
				{
					return {};
				}
				for (auto& c : up)
				{
					c /= len;
				}
				const double right[3] = { up[1] * p[2] - up[2] * p[1], up[2] * p[0] - up[0] * p[2], up[0] * p[1] - up[1] * p[0] };
				std::vector<point> w(4);
				for (auto c = 0; c < 4; c++)
				{
					const auto su = (c == 0 || c == 3) ? -hull_extent : hull_extent;
					const auto sr = c < 2 ? -hull_extent : hull_extent;
					for (auto i = 0; i < 3; i++)
					{
						w[c][i] = p[i] * p[3] + up[i] * su + right[i] * sr;
					}
				}
				return w;
			}

			// keeps the part of the polygon inside the plane
			void clip(std::vector<point>& w, const dplane& p)
			{
				if (w.empty())
				{
					return;
				}
				std::vector<double> dists(w.size());
				for (auto i = 0u; i < w.size(); i++)
				{
					dists[i] = dot(w[i].data(), p.data()) - p[3];
				}
				std::vector<point> out;
				for (auto i = 0u; i < w.size(); i++)
				{
					const auto j = (i + 1) % w.size();
					const auto di = dists[i], dj = dists[j];
					if (di <= plane_epsilon)
					{
						out.push_back(w[i]);
					}
					if ((di > plane_epsilon && dj < -plane_epsilon) || (di < -plane_epsilon && dj > plane_epsilon))
					{
						const auto t = di / (di - dj);
						point mid{};
						for (auto c = 0; c < 3; c++)
						{
							mid[c] = w[i][c] + t * (w[j][c] - w[i][c]);
						}
						out.push_back(mid);
					}
				}
				w = std::move(out);
			}

			unsigned int weld(std::vector<point>& vertices, const point& p)
			{
				for (auto i = 0u; i < vertices.size(); i++)
				{
					if (std::fabs(vertices[i][0] - p[0]) <= weld_epsilon && std::fabs(vertices[i][1] - p[1]) <= weld_epsilon
						&& std::fabs(vertices[i][2] - p[2]) <= weld_epsilon)
					{
						return i;
					}
				}
				vertices.push_back(p);
				return static_cast<unsigned int>(vertices.size() - 1);
			}

			// ---- closed hulls -----------------------------------------------------------------------------

			// a point this far above a triangle sees it
			constexpr double visible_epsilon = 1e-6;
			// a corner this close to a plane lies on it: just above the corner weld, so corners the weld moved
			// still count as on their planes
			constexpr double on_plane_epsilon = weld_epsilon + 0.01;

			using triangle = std::array<unsigned int, 3>;

			void cross(const double* a, const double* b, double* out)
			{
				out[0] = a[1] * b[2] - a[2] * b[1];
				out[1] = a[2] * b[0] - a[0] * b[2];
				out[2] = a[0] * b[1] - a[1] * b[0];
			}

			// unit normal of a triangle (zero when it has no area)
			std::array<double, 3> triangle_normal(const std::vector<point>& pts, const triangle& t)
			{
				double e1[3], e2[3], n[3];
				for (auto k = 0; k < 3; k++)
				{
					e1[k] = pts[t[1]][k] - pts[t[0]][k];
					e2[k] = pts[t[2]][k] - pts[t[0]][k];
				}
				cross(e1, e2, n);
				const auto length = std::sqrt(dot(n, n));
				if (length < 1e-12)
				{
					return { 0.0, 0.0, 0.0 };
				}
				return { n[0] / length, n[1] / length, n[2] / length };
			}

			// Incremental hull: a tetrahedron, then every point outside the hull (farthest first) replaces the
			// triangles it sees, grown from the one it sees best so they stay one patch, with a fan to their
			// boundary. Triangles are counter-clockwise seen from outside.
			bool triangle_hull(const std::vector<point>& pts, std::vector<triangle>& out)
			{
				const auto count = static_cast<unsigned int>(pts.size());
				if (count < 4)
				{
					return false;
				}

				const auto distance = [&](const unsigned int a, const unsigned int b)
				{
					const double d[3] = { pts[a][0] - pts[b][0], pts[a][1] - pts[b][1], pts[a][2] - pts[b][2] };
					return std::sqrt(dot(d, d));
				};

				auto i0 = 0u;
				for (auto i = 1u; i < count; i++)
				{
					if (pts[i][0] < pts[i0][0])
					{
						i0 = i;
					}
				}
				auto i1 = i0;
				for (auto i = 0u; i < count; i++)
				{
					if (distance(i, i0) > distance(i1, i0))
					{
						i1 = i;
					}
				}
				const double axis[3] = { pts[i1][0] - pts[i0][0], pts[i1][1] - pts[i0][1], pts[i1][2] - pts[i0][2] };
				const auto axis_length = std::sqrt(dot(axis, axis));
				if (axis_length < 1e-6)
				{
					return false;
				}
				auto i2 = i0;
				auto best = 0.0;
				for (auto i = 0u; i < count; i++)
				{
					const double d[3] = { pts[i][0] - pts[i0][0], pts[i][1] - pts[i0][1], pts[i][2] - pts[i0][2] };
					double c[3];
					cross(d, axis, c);
					const auto off_line = std::sqrt(dot(c, c)) / axis_length;
					if (off_line > best)
					{
						best = off_line;
						i2 = i;
					}
				}
				if (best < 1e-6)
				{
					return false;
				}
				const auto base = triangle_normal(pts, { i0, i1, i2 });
				auto i3 = i0;
				best = 0.0;
				for (auto i = 0u; i < count; i++)
				{
					const double d[3] = { pts[i][0] - pts[i0][0], pts[i][1] - pts[i0][1], pts[i][2] - pts[i0][2] };
					if (std::fabs(dot(d, base.data())) > std::fabs(best))
					{
						best = dot(d, base.data());
						i3 = i;
					}
				}
				if (std::fabs(best) < 1e-6)
				{
					return false; // flat
				}
				if (best > 0.0)
				{
					std::swap(i1, i2); // (i0, i1, i2) faces away from i3
				}

				struct hull_triangle
				{
					triangle v;
					std::array<double, 3> n;
					double d;
					bool alive;
				};
				std::vector<hull_triangle> tris;
				std::map<std::pair<unsigned int, unsigned int>, unsigned int> edge_owner;

				const auto add = [&](const unsigned int a, const unsigned int b, const unsigned int c)
				{
					hull_triangle t{ { a, b, c }, {}, 0.0, true };
					t.n = triangle_normal(pts, t.v);
					t.d = dot(t.n.data(), pts[a].data());
					const auto index = static_cast<unsigned int>(tris.size());
					edge_owner[{ a, b }] = index;
					edge_owner[{ b, c }] = index;
					edge_owner[{ c, a }] = index;
					tris.push_back(t);
				};

				add(i0, i1, i2);
				add(i0, i3, i1);
				add(i1, i3, i2);
				add(i2, i3, i0);

				double centre[3] = { 0.0, 0.0, 0.0 };
				for (const auto& p : pts)
				{
					for (auto k = 0; k < 3; k++)
					{
						centre[k] += p[k] / count;
					}
				}
				std::vector<unsigned int> order(count);
				std::iota(order.begin(), order.end(), 0u);
				std::stable_sort(order.begin(), order.end(), [&](const unsigned int a, const unsigned int b)
				{
					const double da[3] = { pts[a][0] - centre[0], pts[a][1] - centre[1], pts[a][2] - centre[2] };
					const double db[3] = { pts[b][0] - centre[0], pts[b][1] - centre[1], pts[b][2] - centre[2] };
					return dot(da, da) > dot(db, db);
				});

				for (const auto p : order)
				{
					if (p == i0 || p == i1 || p == i2 || p == i3)
					{
						continue;
					}

					auto top = ~0u;
					auto top_height = visible_epsilon;
					for (auto t = 0u; t < tris.size(); t++)
					{
						if (!tris[t].alive)
						{
							continue;
						}
						const auto height = dot(tris[t].n.data(), pts[p].data()) - tris[t].d;
						if (height > top_height)
						{
							top_height = height;
							top = t;
						}
					}
					if (top == ~0u)
					{
						continue; // inside
					}

					std::set<unsigned int> visible{ top };
					std::vector<unsigned int> stack{ top };
					while (!stack.empty())
					{
						const auto t = stack.back();
						stack.pop_back();
						for (auto e = 0; e < 3; e++)
						{
							const auto twin = edge_owner.find({ tris[t].v[(e + 1) % 3], tris[t].v[e] });
							if (twin == edge_owner.end())
							{
								return false;
							}
							const auto& other = tris[twin->second];
							if (!visible.contains(twin->second)
								&& dot(other.n.data(), pts[p].data()) - other.d > visible_epsilon)
							{
								visible.insert(twin->second);
								stack.push_back(twin->second);
							}
						}
					}

					std::vector<std::pair<unsigned int, unsigned int>> horizon;
					for (const auto t : visible)
					{
						for (auto e = 0; e < 3; e++)
						{
							const auto a = tris[t].v[e];
							const auto b = tris[t].v[(e + 1) % 3];
							if (!visible.contains(edge_owner.at({ b, a })))
							{
								horizon.emplace_back(a, b);
							}
						}
					}
					for (const auto t : visible)
					{
						tris[t].alive = false;
						for (auto e = 0; e < 3; e++)
						{
							edge_owner.erase({ tris[t].v[e], tris[t].v[(e + 1) % 3] });
						}
					}
					for (const auto& [a, b] : horizon)
					{
						add(a, b, p);
					}
				}

				out.clear();
				for (const auto& t : tris)
				{
					if (t.alive)
					{
						out.push_back(t.v);
					}
				}
				// closed: every directed edge has its twin
				for (const auto& t : out)
				{
					for (auto e = 0; e < 3; e++)
					{
						if (!edge_owner.contains({ t[(e + 1) % 3], t[e] }))
						{
							return false;
						}
					}
				}
				return out.size() >= 4;
			}
		}

		bool from_planes(const std::vector<plane>& planes, hull& out)
		{
			out = {};

			std::vector<dplane> dplanes(planes.size());
			for (auto i = 0u; i < planes.size(); i++)
			{
				for (auto c = 0; c < 4; c++)
				{
					dplanes[i][c] = planes[i][c];
				}
			}

			std::vector<point> vertices;
			for (auto i = 0u; i < dplanes.size(); i++)
			{
				auto w = base_winding(dplanes[i]);
				for (auto j = 0u; j < dplanes.size() && !w.empty(); j++)
				{
					if (i != j)
					{
						clip(w, dplanes[j]);
					}
				}
				if (w.size() < 3)
				{
					continue;
				}

				face f{};
				f.p = planes[i];
				const auto committed = vertices.size();
				for (const auto& p : w)
				{
					f.indices.push_back(weld(vertices, p));
				}
				f.indices.erase(std::unique(f.indices.begin(), f.indices.end()), f.indices.end());
				while (f.indices.size() > 1 && f.indices.front() == f.indices.back())
				{
					f.indices.pop_back();
				}
				auto distinct = f.indices;
				std::sort(distinct.begin(), distinct.end());
				distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
				if (distinct.size() < 3 || distinct.size() != f.indices.size())
				{
					vertices.resize(committed);
					continue;
				}

				// counter-clockwise seen from outside (the polygon's normal along the plane's)
				double area[3] = { 0.0, 0.0, 0.0 };
				for (auto k = 0u; k < w.size(); k++)
				{
					const auto& a = w[k];
					const auto& b = w[(k + 1) % w.size()];
					area[0] += a[1] * b[2] - a[2] * b[1];
					area[1] += a[2] * b[0] - a[0] * b[2];
					area[2] += a[0] * b[1] - a[1] * b[0];
				}
				if (dot(area, dplanes[i].data()) < 0.0)
				{
					std::reverse(f.indices.begin(), f.indices.end());
				}
				out.faces.push_back(std::move(f));
			}

			// a face reaching the huge base quad's edge means the solid is open on that side
			for (const auto& v : vertices)
			{
				for (const auto c : v)
				{
					if (std::fabs(c) > hull_extent * 0.5)
					{
						out = {};
						return false;
					}
				}
			}

			out.vertices.reserve(vertices.size());
			for (const auto& v : vertices)
			{
				out.vertices.push_back({static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])});
			}
			return out.faces.size() >= 4 && out.vertices.size() >= 4;
		}

		bool closed_from_planes(const std::vector<plane>& planes, hull& out)
		{
			out = {};

			hull clipped{};
			if (!from_planes(planes, clipped))
			{
				return false;
			}
			std::vector<point> pts;
			for (const auto& v : clipped.vertices)
			{
				pts.push_back({ v[0], v[1], v[2] });
			}

			std::vector<triangle> tris;
			if (!triangle_hull(pts, tris))
			{
				return false;
			}

			// each triangle on the plane its corners lie on that it faces most; a triangle on none keeps its own
			std::vector<std::array<double, 4>> face_planes;
			for (const auto& p : planes)
			{
				face_planes.push_back({ p[0], p[1], p[2], p[3] });
			}
			std::vector<unsigned int> assigned(tris.size());
			for (auto t = 0u; t < tris.size(); t++)
			{
				const auto n = triangle_normal(pts, tris[t]);
				const auto flat = n[0] == 0.0 && n[1] == 0.0 && n[2] == 0.0;
				auto best = ~0u;
				auto best_score = -DBL_MAX;
				for (auto k = 0u; k < planes.size(); k++)
				{
					auto worst = 0.0;
					for (const auto c : tris[t])
					{
						worst = std::max(worst, std::fabs(dot(face_planes[k].data(), pts[c].data()) - face_planes[k][3]));
					}
					if (worst > on_plane_epsilon)
					{
						continue;
					}
					// a triangle without area has no facing: take the plane its corners are closest to
					const auto score = flat ? -worst : dot(face_planes[k].data(), n.data());
					if (!flat && score <= 0.0)
					{
						continue;
					}
					if (score > best_score)
					{
						best_score = score;
						best = k;
					}
				}
				if (best == ~0u)
				{
					if (flat)
					{
						return false;
					}
					best = static_cast<unsigned int>(face_planes.size());
					face_planes.push_back({ n[0], n[1], n[2], dot(n.data(), pts[tris[t][0]].data()) });
				}
				assigned[t] = best;
			}

			// faces: the connected triangles on one plane
			std::map<std::pair<unsigned int, unsigned int>, unsigned int> edge_owner;
			for (auto t = 0u; t < tris.size(); t++)
			{
				for (auto e = 0; e < 3; e++)
				{
					edge_owner[{ tris[t][e], tris[t][(e + 1) % 3] }] = t;
				}
			}
			std::vector<unsigned int> parent(tris.size());
			std::iota(parent.begin(), parent.end(), 0u);
			const auto root = [&](unsigned int t)
			{
				while (parent[t] != t)
				{
					parent[t] = parent[parent[t]];
					t = parent[t];
				}
				return t;
			};
			for (auto t = 0u; t < tris.size(); t++)
			{
				for (auto e = 0; e < 3; e++)
				{
					const auto twin = edge_owner.at({ tris[t][(e + 1) % 3], tris[t][e] });
					if (assigned[twin] == assigned[t])
					{
						parent[root(t)] = root(twin);
					}
				}
			}
			std::map<unsigned int, std::vector<unsigned int>> regions;
			for (auto t = 0u; t < tris.size(); t++)
			{
				regions[root(t)].push_back(t);
			}

			// each face's outline: its triangles' edges whose twin is in another face, one loop
			struct polygon
			{
				unsigned int plane;
				std::vector<unsigned int> loop;
			};
			std::vector<polygon> polygons;
			for (const auto& [r, members] : regions)
			{
				std::map<unsigned int, unsigned int> next;
				for (const auto t : members)
				{
					for (auto e = 0; e < 3; e++)
					{
						const auto a = tris[t][e];
						const auto b = tris[t][(e + 1) % 3];
						if (root(edge_owner.at({ b, a })) == r)
						{
							continue;
						}
						if (!next.emplace(a, b).second)
						{
							return false; // the outline touches itself
						}
					}
				}
				polygon poly{ assigned[members.front()], {} };
				poly.loop.push_back(next.begin()->first);
				while (poly.loop.size() <= next.size())
				{
					const auto to = next.find(poly.loop.back());
					if (to == next.end())
					{
						return false;
					}
					if (to->second == poly.loop.front())
					{
						break;
					}
					poly.loop.push_back(to->second);
				}
				if (poly.loop.size() != next.size())
				{
					return false; // several outlines
				}
				polygons.push_back(std::move(poly));
			}

			// a corner on exactly two faces lies on the edge between them: leave it out of both, unless that
			// would leave either with fewer than three corners
			for (auto changed = true; changed;)
			{
				changed = false;
				std::map<unsigned int, std::vector<unsigned int>> faces_of;
				for (auto f = 0u; f < polygons.size(); f++)
				{
					for (const auto v : polygons[f].loop)
					{
						faces_of[v].push_back(f);
					}
				}
				for (const auto& [v, faces] : faces_of)
				{
					if (faces.size() == 2 && polygons[faces[0]].loop.size() > 3 && polygons[faces[1]].loop.size() > 3)
					{
						for (const auto f : faces)
						{
							auto& loop = polygons[f].loop;
							loop.erase(std::remove(loop.begin(), loop.end(), v), loop.end());
						}
						changed = true;
						break;
					}
				}
			}

			// only the corners the faces use, renumbered
			std::vector<unsigned int> remap(pts.size(), ~0u);
			for (const auto& poly : polygons)
			{
				face f{};
				const auto& fp = face_planes[poly.plane];
				f.p = { static_cast<float>(fp[0]), static_cast<float>(fp[1]), static_cast<float>(fp[2]), static_cast<float>(fp[3]) };
				for (const auto v : poly.loop)
				{
					if (remap[v] == ~0u)
					{
						remap[v] = static_cast<unsigned int>(out.vertices.size());
						out.vertices.push_back({ static_cast<float>(pts[v][0]), static_cast<float>(pts[v][1]), static_cast<float>(pts[v][2]) });
					}
					f.indices.push_back(remap[v]);
				}
				out.faces.push_back(std::move(f));
			}

			// closed: every directed edge has its twin exactly once
			std::map<std::pair<unsigned int, unsigned int>, int> edges;
			for (const auto& f : out.faces)
			{
				for (auto e = 0u; e < f.indices.size(); e++)
				{
					edges[{ f.indices[e], f.indices[(e + 1) % f.indices.size()] }]++;
				}
			}
			for (const auto& [edge, uses] : edges)
			{
				const auto twin = edges.find({ edge.second, edge.first });
				if (uses != 1 || twin == edges.end() || twin->second != 1)
				{
					out = {};
					return false;
				}
			}
			return out.faces.size() >= 4 && out.vertices.size() >= 4;
		}
	}
}
