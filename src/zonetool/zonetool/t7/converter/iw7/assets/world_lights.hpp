#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::world_lights
	{
		// IW7's per-light world data for the converted primary lights: the frustum light hulls lights are
		// binned into clusters with (BO3's cull box applied), light view frustums of shadowed spots, shadow
		// casters, the tree of lights with dynamic shadows, and the per-surface / per-static-model light
		// lists. Runs after the surfaces, bounds and static models are final.
		// static surfaces (IW7 index) that cast the sun's shadow but no local light's: the world groups' (a light with dynamic
		// shadows redraws its casters every frame); before build
		void set_sun_only_casters(std::vector<char> surfaces);

		void build(zonetool::iw7::GfxWorld* world, utils::memory::allocator& allocator);
	}
}
