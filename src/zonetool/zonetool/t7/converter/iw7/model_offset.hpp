#pragma once

#include <string>
#include <unordered_map>

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace model_offset
		{
			struct offset_t
			{
				bool valid = false;
				float rot[3][3]{};
				float trans[3]{};
				std::unordered_map<std::string, std::string> bones;
			};

			const offset_t& get(const std::string& name);

			void apply_point(const offset_t& off, float* xyz);
			void apply_dir(const offset_t& off, float* xyz);
			void apply_quat(const offset_t& off, float* quat);
		}
	}
}
