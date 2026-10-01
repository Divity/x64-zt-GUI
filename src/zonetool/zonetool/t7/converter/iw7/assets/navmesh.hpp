#pragma once

#include "../map_entities.hpp"

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace navmesh
		{
			// BO3's Havok AI navmesh (maps/<map>_navmesh) as IW7's NavPower navmesh <bsp_name>.navmesh, and the BO3
			// traversals of `t7_entities` (node_negotiation_begin / _end pairs) as IW7 off-mesh links: records in the
			// navmesh, negotiation nodes in <bsp_name>.aipaths and the node entities returned, which the map's ents
			// carry in this order (IW7 fills each aipaths fixed node slot from the next node_ entity). Throws when the
			// BO3 data cannot be read or the result breaks a NavPower rule.
			std::vector<map_entities::entity> convert(const NavMeshData* asset, const std::string& bsp_name, const std::string& t7_entities);
		}
	}
}
