#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace comworld
		{
			// `world` supplies the sun (IW7 primary light 1); BO3 keeps it in the GfxWorld's sun
			// volumes. It may be null, in which case the sun is left dark.
			zonetool::iw7::ComWorld* convert(ComWorld* asset, const GfxWorld* world, utils::memory::allocator& allocator);
			void dump(ComWorld* asset, const GfxWorld* world);
			// ZT_LIGHTDEF_EXPORT: the flickering lights' entities, cookie slices and list (in the dump)
			void export_flicker(const GfxWorld* world);

			// The IW7 light list built for the last dumped ComWorld. T7 light i is IW7 light
			// remap_light(i); 0 means it was dropped.
			const zonetool::iw7::ComWorld* converted();
			const ComWorld* source();
			unsigned int remap_light(unsigned int t7_index);
		}
	}
}
