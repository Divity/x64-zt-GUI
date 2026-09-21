#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace material
		{
			// the name a converted material is written under, models and
			// effects have to ask for them by the same name
			std::string get_converted_name(Material* asset);
			void dump(Material* asset);
		}
	}
}
