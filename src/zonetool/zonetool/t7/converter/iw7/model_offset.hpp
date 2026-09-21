#pragma once

#include <string>
#include <unordered_map>

namespace zonetool::t7
{
	namespace converter::iw7
	{
		// A rigid transform applied to a converted model's geometry, bone matrices
		// and normals. T7 weapon models are authored around tag_weapon, IW7 hangs
		// viewmodels off j_gun, so a ported weapon needs its geometry re-authored
		// around j_gun or it will not sit in the hands.
		//
		// Offsets are read from "t7_to_iw7_model_offsets.json" next to the exe:
		//   { "wpn_t7_zmb_thundergun_view": { "quat": [x,y,z,w], "trans": [x,y,z] } }
		// Keys match the start of the model or mesh name, so LOD suffixes still hit.
		namespace model_offset
		{
			struct offset_t
			{
				bool valid = false;
				float rot[3][3]{};
				float trans[3]{};
				// optional bone renames, e.g. a T7 weapon root "tag_weapon" -> "j_gun",
				// because IW7 parents tag_weapon to tag_torso (body space) while the
				// viewmodel weapon belongs on j_gun under the wrist
				std::unordered_map<std::string, std::string> bones;
			};

			const offset_t& get(const std::string& name);

			// full transform - use for vertex positions and bone translations
			void apply_point(const offset_t& off, float* xyz);
			// rotation only - use for normals, tangents and bone quaternions
			void apply_dir(const offset_t& off, float* xyz);
			void apply_quat(const offset_t& off, float* quat);
		}
	}
}
