#pragma once

#include "map_entities.hpp"

namespace zonetool::t7
{
	namespace converter::iw7::map_effects
	{
		// the XModels the effects BO3's `fx` entities place draw (model elements, their children's included): the map
		// converts them, and their materials, with its placed models
		std::vector<const XModel*> models(const std::vector<map_entities::entity>& ents);

		// Converts the effects BO3's `fx` entities place (and every effect those spawn: runner children, the effects of
		// their particles' death, impact and flight) and their materials, and writes the scripts IW7 places them with:
		// scripts/cp/maps/<map>/<map>_fx.gsc (the LoadFX table) and gen/<map>_fx.gsc (the createfx placements), which the
		// IW7 client parses as text and the level script runs. Returns the zone rows (the placed effects, the scripts).
		std::vector<std::string> convert(const std::vector<map_entities::entity>& ents, const std::string& map);
	}
}
