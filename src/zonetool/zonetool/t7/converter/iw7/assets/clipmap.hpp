#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace clipmap
		{
			// Converts the collision (clipMap_t, with its MapEnts) and writes the IW7 clipMap, MapEnts and the
			// two dummy PhysicsAssets the entity shapes are bound through.
			void dump(clipMap_t* asset);
		}
	}
}
