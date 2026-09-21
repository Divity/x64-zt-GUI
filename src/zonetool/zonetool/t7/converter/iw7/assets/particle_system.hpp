#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace particlesystem
		{
			zonetool::iw7::ParticleSystemDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator);
			void dump(FxEffectDef* asset);
		}
	}
}
