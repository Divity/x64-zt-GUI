#pragma once

#include "map_entities.hpp"

namespace zonetool::t7
{
	namespace converter::iw7::zombies_ents
	{
		// BO3 zombies (zm_) map entities -> the layout the stock IW7 zombies (CP) scripts read: zone volumes become
		// spawn volumes, their spawn / riser locations static spawners, the initial spawn points and respawn points
		// the CP player starts and respawn locations, plus the intermission camera CP expects.
		void convert(std::vector<map_entities::entity>& ents);
	}
}
