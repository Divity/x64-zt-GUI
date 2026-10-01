#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::world_lights
	{
		// IW7's per-light world data for the converted primary lights: the frustum light hulls lights are
		// binned into clusters with (BO3's cull box applied), light view frustums of shadowed spots, shadow
		// casters, the tree of lights with dynamic shadows, and the per-surface / per-static-model light
		// lists. Runs after the surfaces, bounds and static models are final.
		void build(zonetool::iw7::GfxWorld* world, utils::memory::allocator& allocator);
	}
}
