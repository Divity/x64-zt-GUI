#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace material
		{
			std::string get_converted_name(Material* asset);
			void dump(Material* asset);

			// The IW7 name of a BO3 model material: mo/ and its name, with its folder kept when it is not mc/. BO3
			// keeps same-named materials apart by their folder, IW7's model materials all live in mo/.
			std::string model_material_name(const std::string& bo3_name);

			// model material `name` is drawn as `shared` (an identical material): model_material_name answers `shared`
			void set_shared(const std::string& name, const std::string& shared);
		}
	}
}
