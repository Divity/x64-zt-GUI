#pragma once

#include "effect_material.hpp"

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace particlesystem
		{
			// the GfxLightDef every converted light element draws with (stock effect lights' own; resident in IW7)
			inline constexpr const char* light_def = "light_fx_default";

			// What a converted effect references by name.
			struct references
			{
				// the IW7 name of a BO3 effect: the converted effect itself, its runners' children and the effects its
				// particles spawn (on death, on impact, as they fly)
				std::function<std::string(const std::string&)> effect;
				// the IW7 material a sprite element draws a BO3 material with (empty name: the visual is left out)
				std::function<effect_material::converted(const Material*, const FxElemDef*)> material;
			};

			// the references of an effect dumped on its own (dumpzone): BO3's names, the legacy effect materials
			const references& standalone_references();

			// `children`: where the effects a conversion makes of its own go (an aimed spot light's light, spawned by its
			// runner); without it such lights stay on the effect's axis
			zonetool::iw7::ParticleSystemDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator, const references& refs,
				std::vector<zonetool::iw7::ParticleSystemDef*>* children);
			zonetool::iw7::ParticleSystemDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator, const references& refs);
			zonetool::iw7::ParticleSystemDef* convert(FxEffectDef* asset, utils::memory::allocator& allocator);
			void dump(FxEffectDef* asset);
			void dump(FxEffectDef* asset, const references& refs);
		}
	}
}
