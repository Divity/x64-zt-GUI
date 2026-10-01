#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::reflection_probes
	{
		// IW7 reflection probes from BO3's lighting state 0: one probe (a slice of the "*reflection_probe_array"
		// cube array) per distinct BO3 local probe, from the sun volume holding its capture point, with one
		// instance per blend volume, and each sun volume's global probe as the fallback over that volume.
		// Fills world->draw.reflectionProbeData and dpvs.reflectionProbeVisDataCount and dumps the cube array.
		void convert(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, utils::memory::allocator& allocator);
	}
}
