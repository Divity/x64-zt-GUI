#pragma once

#include "game/mode.hpp"
#include <string>
#include <vector>
#include <optional>
#include <unordered_set>

namespace gui
{
	namespace commands
	{
		void load_zone(const std::string& zone);
		void unload_zones();
		void verify_zone(const std::string& zone);
		void dump_zone(const std::string& zone, const game::game_mode target = game::game_mode::none, const std::optional<std::unordered_set<std::string>>& asset_filter = {});
		void dump_asset(const std::string& type, const std::string& name);
		void dump_map(const std::string& map, const game::game_mode target = game::game_mode::none, const std::optional<std::unordered_set<std::string>>& asset_filter = {}, bool skip_common = false);
		void build_zone(const std::string& zone);
		void dump_csv(const std::string& zone);
		void generate_csv(const std::string& map, bool is_sp = false);
		void iterate_zones();
		void dump_matching_zones(const std::string& pattern, const game::game_mode target = game::game_mode::none, const std::optional<std::unordered_set<std::string>>& asset_filter = {});
		void build_xmodel(const std::string& name);

		std::vector<std::string> discover_fastfiles();
		std::vector<std::string> discover_map_zones();
		std::vector<std::string> discover_csv_files();
		std::vector<std::string> discover_zonetool_maps();
		std::vector<std::string> get_available_asset_types();
	}

}

