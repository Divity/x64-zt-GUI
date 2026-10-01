#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		// A bounding volume hierarchy over a triangle soup, for segment queries against static geometry
		// (the light grid's probe visibility). Triangles are double sided.
		class triangle_bvh
		{
		public:
			// nine floats per triangle
			explicit triangle_bvh(std::vector<float> triangles);

			std::size_t size() const
			{
				return tris_.size() / 9;
			}

			// whether any triangle's box overlaps the box [lo, hi]
			bool any_in_box(const float lo[3], const float hi[3]) const;

			// the largest t in (t_min, t_max) at which the segment a + t (b - a) crosses a triangle, or -1
			float last_crossing(const double a[3], const double b[3], double t_min, double t_max) const;

			// whether the segment crosses, in (t_min, t_max), a triangle `opaque` accepts: it is given the triangle's index
			// (its place in the constructor's list) and the crossing's barycentric (u, v) on it (point = p0 + u e1 + v e2)
			bool blocked(const double a[3], const double b[3], double t_min, double t_max,
				const std::function<bool(std::uint32_t triangle, double u, double v)>& opaque) const;

		private:
			struct node
			{
				float lo[3];
				float hi[3];
				std::uint32_t first; // leaf: first triangle in order_; inner: left child (right = first + 1)
				std::uint32_t count; // 0 for inner nodes
			};

			std::vector<float> tris_;
			std::vector<std::uint32_t> order_;
			std::vector<node> nodes_;
		};
	}
}
