#pragma once

#include <array>
#include <vector>

namespace zonetool::t7
{
	namespace converter::iw7::delaunay
	{
		constexpr std::uint32_t none = 0xFFFFFFFF;

		using point = std::array<double, 3>;

		struct mesh
		{
			// positively oriented: dot(cross(v1 - v0, v2 - v0), v3 - v0) > 0 (IW7's light grid convention)
			std::vector<std::array<std::uint32_t, 4>> tets;
			// [k]: the tetrahedron across the face opposite vertex k, `none` on the hull
			std::vector<std::array<std::uint32_t, 4>> neighbours;
		};

		// Delaunay tetrahedralization (Bowyer-Watson in a super box that is removed afterwards). The points
		// must be in general position up to double rounding: jitter lattice points first.
		mesh tetrahedralize(const std::vector<point>& points);

		// the tetrahedron containing p, walking from `start`; `none` once the walk leaves the hull
		std::uint32_t locate(const mesh& m, const std::vector<point>& points, const point& p, std::uint32_t start);

		// barycentric coordinates of p in tetrahedron t (weights of its four vertices)
		std::array<double, 4> barycentric(const mesh& m, const std::vector<point>& points, std::uint32_t t, const point& p);
	}
}
