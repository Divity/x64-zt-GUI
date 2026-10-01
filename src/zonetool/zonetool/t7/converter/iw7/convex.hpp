#pragma once

#include <array>
#include <vector>

namespace zonetool::t7
{
	namespace converter::iw7::convex
	{
		// plane { n.x, n.y, n.z, d }: the inside is dot(n, x) <= d
		using plane = std::array<float, 4>;

		struct face
		{
			plane p;
			std::vector<unsigned int> indices; // counter-clockwise seen from outside
		};

		struct hull
		{
			std::vector<std::array<float, 3>> vertices;
			std::vector<face> faces;
		};

		// the convex solid bounded by the planes (winding clipper: every plane's face is a huge quad clipped
		// by all the others); false when it is empty or unbounded
		bool from_planes(const std::vector<plane>& planes, hull& out);

		// the same solid as a closed hull, every edge bordering exactly two faces (from_planes' per-face welding can
		// leave T-junctions where planes meet almost at a point): an incremental triangle hull of from_planes'
		// corners, closed by construction, with the triangles merged onto the planes they lie on (a triangle on none
		// of them within the corner weld keeps its own plane); false when the solid is empty, unbounded or flat
		bool closed_from_planes(const std::vector<plane>& planes, hull& out);
	}
}
