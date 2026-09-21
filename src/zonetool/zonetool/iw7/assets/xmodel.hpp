#pragma once
#include "../zonetool.hpp"

namespace zonetool::iw7
{
	class xmodel : public asset_interface
	{
	private:
		std::string name_;
		XModel* asset_ = nullptr;

		std::vector<std::pair<scr_string_t*, const char*>> script_strings;
		void add_script_string(scr_string_t* ptr, const char* str);
		const char* get_script_string(scr_string_t* ptr);

	public:
		XModel* parse(std::string name, zone_memory* mem);

		void init(const std::string& name, zone_memory* mem) override;
		void prepare(zone_buffer* buf, zone_memory* mem) override;
		void load_depending(zone_base* zone) override;

		void* pointer() override { return asset_; }
		bool referenced() override { return name_.starts_with(","); }
		std::string name() override;
		std::int32_t type() override;
		void write(zone_base* zone, zone_buffer* buffer) override;

		static void dump(XModel* asset);

		// Optional bone-name remap applied while dumping. A ported model can carry a
		// source-engine bone name that IW7 resolves somewhere else in the hierarchy:
		// a T7 weapon roots at tag_weapon, which IW7 parents to tag_torso (body
		// space), while IW7 expects j_gun under the wrist. Converters set this around
		// a dump call and clear it afterwards.
		using bone_name_remap_t = std::unordered_map<std::string, std::string>;
		static void set_bone_name_remap(const bone_name_remap_t* remap);
	};
}