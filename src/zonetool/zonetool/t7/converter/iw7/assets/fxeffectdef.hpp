#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace fxeffectdef
		{
			zonetool::iw7::FxEffectDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator);
			void dump(FxEffectDef* asset);
		}
	}
}
