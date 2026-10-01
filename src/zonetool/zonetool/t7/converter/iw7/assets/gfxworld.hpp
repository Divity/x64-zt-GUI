#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace gfxworld
		{
			// Needs the converted ComWorld (comworld::converted()) for everything tied to primary
			// lights, so the light list has to be converted first.
			zonetool::iw7::GfxWorld* convert(GfxWorld* asset, utils::memory::allocator& allocator);
			void dump(GfxWorld* asset);

			// BO3's worldspawn umbraSmallestHole (the gap its Umbra bake keeps open); 0: the tome generator's default
			void set_umbra_smallest_hole(float units);
		}
	}
}
