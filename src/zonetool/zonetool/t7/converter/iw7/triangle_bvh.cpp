#include <std_include.hpp>
#include "triangle_bvh.hpp"

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace
		{
			constexpr std::uint32_t leaf_triangles = 4;
			constexpr std::uint32_t bins = 16;
			// a depth-first walk holds at most depth + 1 nodes
			constexpr std::uint32_t max_depth = 250;
			constexpr std::uint32_t stack_size = max_depth + 2;
		}

		triangle_bvh::triangle_bvh(std::vector<float> triangles)
			: tris_(std::move(triangles))
		{
			const auto count = static_cast<std::uint32_t>(tris_.size() / 9);
			order_.resize(count);
			std::vector<float> centre(static_cast<std::size_t>(count) * 3);
			std::vector<float> box(static_cast<std::size_t>(count) * 6);
			for (auto i = 0u; i < count; i++)
			{
				order_[i] = i;
				const auto* t = &tris_[static_cast<std::size_t>(i) * 9];
				for (auto k = 0; k < 3; k++)
				{
					const auto lo = std::min({ t[k], t[3 + k], t[6 + k] });
					const auto hi = std::max({ t[k], t[3 + k], t[6 + k] });
					box[i * 6 + k] = lo;
					box[i * 6 + 3 + k] = hi;
					centre[i * 3 + k] = (lo + hi) * 0.5f;
				}
			}
			if (!count)
			{
				return;
			}

			nodes_.reserve(static_cast<std::size_t>(count) * 2 / leaf_triangles + 1);
			struct task
			{
				std::uint32_t node, first, count, depth;
			};
			std::vector<task> stack;
			nodes_.push_back({});
			stack.push_back({ 0, 0, count, 0 });
			while (!stack.empty())
			{
				const auto [index, first, n, depth] = stack.back();
				stack.pop_back();
				if (depth > max_depth)
				{
					throw std::runtime_error("triangle bvh: the tree grew deeper than its traversal stack");
				}

				float lo[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, hi[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
				float clo[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, chi[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
				for (auto i = first; i < first + n; i++)
				{
					const auto t = order_[i];
					for (auto k = 0; k < 3; k++)
					{
						lo[k] = std::min(lo[k], box[t * 6 + k]);
						hi[k] = std::max(hi[k], box[t * 6 + 3 + k]);
						clo[k] = std::min(clo[k], centre[t * 3 + k]);
						chi[k] = std::max(chi[k], centre[t * 3 + k]);
					}
				}
				auto& out = nodes_[index];
				std::memcpy(out.lo, lo, sizeof(lo));
				std::memcpy(out.hi, hi, sizeof(hi));

				auto axis = 0;
				for (auto k = 1; k < 3; k++)
				{
					if (chi[k] - clo[k] > chi[axis] - clo[axis])
					{
						axis = k;
					}
				}
				const auto extent = chi[axis] - clo[axis];
				if (n <= leaf_triangles || !(extent > 0.0f))
				{
					out.first = first;
					out.count = n;
					continue;
				}

				// binned surface area heuristic along the widest centroid axis
				struct bin
				{
					float lo[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, hi[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
					std::uint32_t count = 0;
				};
				bin b[bins];
				const auto bin_of = [&](const std::uint32_t t)
				{
					const auto x = (centre[t * 3 + axis] - clo[axis]) / extent * static_cast<float>(bins);
					return std::min(bins - 1, static_cast<std::uint32_t>(std::max(0.0f, x)));
				};
				for (auto i = first; i < first + n; i++)
				{
					const auto t = order_[i];
					auto& target = b[bin_of(t)];
					target.count++;
					for (auto k = 0; k < 3; k++)
					{
						target.lo[k] = std::min(target.lo[k], box[t * 6 + k]);
						target.hi[k] = std::max(target.hi[k], box[t * 6 + 3 + k]);
					}
				}
				const auto area = [](const float* l, const float* h)
				{
					const float d[3] = { h[0] - l[0], h[1] - l[1], h[2] - l[2] };
					return d[0] * d[1] + d[1] * d[2] + d[2] * d[0];
				};
				float left_area[bins], right_area[bins];
				std::uint32_t left_count[bins], right_count[bins];
				{
					bin acc{};
					for (auto i = 0u; i < bins; i++)
					{
						acc.count += b[i].count;
						for (auto k = 0; k < 3; k++)
						{
							acc.lo[k] = std::min(acc.lo[k], b[i].lo[k]);
							acc.hi[k] = std::max(acc.hi[k], b[i].hi[k]);
						}
						left_count[i] = acc.count;
						left_area[i] = acc.count ? area(acc.lo, acc.hi) : 0.0f;
					}
				}
				{
					bin acc{};
					for (auto i = bins; i-- > 0;)
					{
						acc.count += b[i].count;
						for (auto k = 0; k < 3; k++)
						{
							acc.lo[k] = std::min(acc.lo[k], b[i].lo[k]);
							acc.hi[k] = std::max(acc.hi[k], b[i].hi[k]);
						}
						right_count[i] = acc.count;
						right_area[i] = acc.count ? area(acc.lo, acc.hi) : 0.0f;
					}
				}
				auto best = bins;
				auto best_cost = FLT_MAX;
				for (auto i = 0u; i + 1 < bins; i++)
				{
					if (!left_count[i] || !right_count[i + 1])
					{
						continue;
					}
					const auto cost = left_area[i] * static_cast<float>(left_count[i]) + right_area[i + 1] * static_cast<float>(right_count[i + 1]);
					if (cost < best_cost)
					{
						best_cost = cost;
						best = i;
					}
				}

				auto middle = first;
				if (best < bins)
				{
					middle = static_cast<std::uint32_t>(std::partition(order_.begin() + first, order_.begin() + first + n,
						[&](const std::uint32_t t) { return bin_of(t) <= best; }) - order_.begin());
				}
				if (middle == first || middle == first + n)
				{
					// all centroids in one bin: split the list in the middle along the axis
					middle = first + n / 2;
					std::nth_element(order_.begin() + first, order_.begin() + middle, order_.begin() + first + n,
						[&](const std::uint32_t l, const std::uint32_t r) { return centre[l * 3 + axis] < centre[r * 3 + axis]; });
				}

				const auto left = static_cast<std::uint32_t>(nodes_.size());
				nodes_.push_back({});
				nodes_.push_back({});
				nodes_[index].first = left;
				nodes_[index].count = 0;
				stack.push_back({ left, first, middle - first, depth + 1 });
				stack.push_back({ left + 1, middle, first + n - middle, depth + 1 });
			}
		}

		bool triangle_bvh::any_in_box(const float lo[3], const float hi[3]) const
		{
			if (nodes_.empty())
			{
				return false;
			}
			std::uint32_t stack[stack_size];
			auto top = 0u;
			stack[top++] = 0;
			while (top)
			{
				const auto& n = nodes_[stack[--top]];
				if (n.lo[0] > hi[0] || n.hi[0] < lo[0] || n.lo[1] > hi[1] || n.hi[1] < lo[1] || n.lo[2] > hi[2] || n.hi[2] < lo[2])
				{
					continue;
				}
				if (!n.count)
				{
					stack[top++] = n.first;
					stack[top++] = n.first + 1;
					continue;
				}
				for (auto i = n.first; i < n.first + n.count; i++)
				{
					const auto* t = &tris_[static_cast<std::size_t>(order_[i]) * 9];
					auto inside = true;
					for (auto k = 0; k < 3 && inside; k++)
					{
						inside = std::min({ t[k], t[3 + k], t[6 + k] }) <= hi[k] && std::max({ t[k], t[3 + k], t[6 + k] }) >= lo[k];
					}
					if (inside)
					{
						return true;
					}
				}
			}
			return false;
		}

		float triangle_bvh::last_crossing(const double a[3], const double b[3], const double t_min, const double t_max) const
		{
			if (nodes_.empty())
			{
				return -1.0f;
			}
			const double d[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
			double inv[3];
			for (auto k = 0; k < 3; k++)
			{
				inv[k] = d[k] != 0.0 ? 1.0 / d[k] : std::numeric_limits<double>::infinity();
			}

			auto best = -1.0;
			std::uint32_t stack[stack_size];
			auto top = 0u;
			stack[top++] = 0;
			while (top)
			{
				const auto& n = nodes_[stack[--top]];
				// slab test of the part of the segment that can still beat the best hit
				auto t0 = std::max(t_min, best), t1 = t_max;
				for (auto k = 0; k < 3 && t0 <= t1; k++)
				{
					if (d[k] == 0.0)
					{
						if (a[k] < n.lo[k] || a[k] > n.hi[k])
						{
							t0 = 1.0;
							t1 = 0.0;
						}
						continue;
					}
					auto n0 = (n.lo[k] - a[k]) * inv[k];
					auto n1 = (n.hi[k] - a[k]) * inv[k];
					if (n0 > n1)
					{
						std::swap(n0, n1);
					}
					t0 = std::max(t0, n0);
					t1 = std::min(t1, n1);
				}
				if (t0 > t1)
				{
					continue;
				}
				if (!n.count)
				{
					stack[top++] = n.first;
					stack[top++] = n.first + 1;
					continue;
				}
				for (auto i = n.first; i < n.first + n.count; i++)
				{
					// Moller-Trumbore, double sided
					const auto* t = &tris_[static_cast<std::size_t>(order_[i]) * 9];
					const double e1[3] = { t[3] - static_cast<double>(t[0]), t[4] - static_cast<double>(t[1]), t[5] - static_cast<double>(t[2]) };
					const double e2[3] = { t[6] - static_cast<double>(t[0]), t[7] - static_cast<double>(t[1]), t[8] - static_cast<double>(t[2]) };
					const double p[3] = { d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0] };
					const auto det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
					if (std::fabs(det) < 1e-12)
					{
						continue;
					}
					const auto inv_det = 1.0 / det;
					const double s[3] = { a[0] - t[0], a[1] - t[1], a[2] - t[2] };
					const auto u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv_det;
					if (u < 0.0 || u > 1.0)
					{
						continue;
					}
					const double q[3] = { s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0] };
					const auto v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv_det;
					if (v < 0.0 || u + v > 1.0)
					{
						continue;
					}
					const auto hit = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv_det;
					if (hit > t_min && hit < t_max && hit > best)
					{
						best = hit;
					}
				}
			}
			return static_cast<float>(best);
		}

		bool triangle_bvh::blocked(const double a[3], const double b[3], const double t_min, const double t_max,
			const std::function<bool(std::uint32_t triangle, double u, double v)>& opaque) const
		{
			if (nodes_.empty())
			{
				return false;
			}
			const double d[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
			double inv[3];
			for (auto k = 0; k < 3; k++)
			{
				inv[k] = d[k] != 0.0 ? 1.0 / d[k] : std::numeric_limits<double>::infinity();
			}
			std::uint32_t stack[stack_size];
			auto top = 0u;
			stack[top++] = 0;
			while (top)
			{
				const auto& n = nodes_[stack[--top]];
				auto t0 = t_min, t1 = t_max;
				for (auto k = 0; k < 3 && t0 <= t1; k++)
				{
					if (d[k] == 0.0)
					{
						if (a[k] < n.lo[k] || a[k] > n.hi[k])
						{
							t0 = 1.0;
							t1 = 0.0;
						}
						continue;
					}
					auto n0 = (n.lo[k] - a[k]) * inv[k];
					auto n1 = (n.hi[k] - a[k]) * inv[k];
					if (n0 > n1)
					{
						std::swap(n0, n1);
					}
					t0 = std::max(t0, n0);
					t1 = std::min(t1, n1);
				}
				if (t0 > t1)
				{
					continue;
				}
				if (!n.count)
				{
					stack[top++] = n.first;
					stack[top++] = n.first + 1;
					continue;
				}
				for (auto i = n.first; i < n.first + n.count; i++)
				{
					// Moller-Trumbore, double sided
					const auto index = order_[i];
					const auto* t = &tris_[static_cast<std::size_t>(index) * 9];
					const double e1[3] = { t[3] - static_cast<double>(t[0]), t[4] - static_cast<double>(t[1]), t[5] - static_cast<double>(t[2]) };
					const double e2[3] = { t[6] - static_cast<double>(t[0]), t[7] - static_cast<double>(t[1]), t[8] - static_cast<double>(t[2]) };
					const double p[3] = { d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0] };
					const auto det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
					if (std::fabs(det) < 1e-12)
					{
						continue;
					}
					const auto inv_det = 1.0 / det;
					const double s[3] = { a[0] - t[0], a[1] - t[1], a[2] - t[2] };
					const auto u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv_det;
					if (u < 0.0 || u > 1.0)
					{
						continue;
					}
					const double q[3] = { s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0] };
					const auto v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv_det;
					if (v < 0.0 || u + v > 1.0)
					{
						continue;
					}
					const auto hit = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv_det;
					if (hit > t_min && hit < t_max && opaque(index, u, v))
					{
						return true;
					}
				}
			}
			return false;
		}
	}
}
