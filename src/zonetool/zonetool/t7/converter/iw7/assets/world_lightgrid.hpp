#pragma once

#include "world_lightmap.hpp"

namespace zonetool::t7
{
	namespace converter::iw7::world_lightgrid
	{
		// IW7's light grid from BO3's probe lighting of lighting state 0: the GPU tetrahedral probe grid (dynamic
		// models, particles, and the SH IW7 normalizes reflection probes with) with its probe visibility, the
		// static models' probe samples, and the voxel tree with its per-voxel light lists. Runs after
		// world_lights (light hulls). occluders[i]: static surface i blocks light (opaque, not alpha tested), for the probe
		// visibility; sun_blockers: the lightmap's, for the sun's visibility (coefficient 27).
		void build(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, const zonetool::iw7::GfxWorldTransientZone* zone,
			const std::vector<std::uint8_t>& occluders, const world_lightmap::sun_blockers& sun_blockers,
			utils::memory::allocator& allocator);
	}
}
