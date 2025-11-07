#include <std_include.hpp>
#include "gui_commands.hpp"

#include "game/mode.hpp"
#include "zonetool/utils/utils.hpp"
#include "zonetool/h1/zonetool.hpp"
#include "zonetool/h2/zonetool.hpp"
#include "zonetool/s1/zonetool.hpp"
#include "zonetool/iw6/zonetool.hpp"
#include "zonetool/iw7/zonetool.hpp"
#include "zonetool/t7/zonetool.hpp"
#include "component/h1/command.hpp"
#include "component/h2/command.hpp"
#include "component/s1/command.hpp"
#include "component/iw6/command.hpp"
#include "component/iw7/command.hpp"
#include "component/t7/command.hpp"

#include <utils/io.hpp>
#include <utils/string.hpp>

namespace gui
{
	namespace commands
	{
		std::vector<std::string> discover_fastfiles()
		{
			std::vector<std::string> fastfiles;
			std::unordered_set<std::string> seen;

			const auto scan_directory = [&](const std::string& path)
			{
				if (!utils::io::directory_exists(path))
				{
					return;
				}

				try
				{
					for (const auto& entry : std::filesystem::directory_iterator(path))
					{
						if (entry.is_regular_file() && entry.path().extension() == ".ff")
						{
							const auto zone = entry.path().stem().string();
							if (seen.find(zone) == seen.end())
							{
								fastfiles.push_back(zone);
								seen.insert(zone);
							}
						}
					}
				}
				catch (...)
				{
				}
			};

			scan_directory("zone");
			scan_directory("zone/english");
			scan_directory(".");

		std::sort(fastfiles.begin(), fastfiles.end());
		return fastfiles;
	}

	std::vector<std::string> discover_map_zones()
	{
		std::vector<std::string> map_zones;
		std::unordered_set<std::string> seen;

	const auto is_map_zone = [](const std::string& zone) -> bool
	{
		if (zone.find("mp_body_") != std::string::npos ||
			zone.find("mp_head_") != std::string::npos ||
			zone.find("mp_legs_") != std::string::npos ||
			zone.find("mp_hands_") != std::string::npos ||
			zone.find("mp_viewhands_") != std::string::npos ||
			zone.find("mp_weapon_") != std::string::npos)
		{
			return false;
		}
		
		if (zone.starts_with("mp_") || zone.starts_with("cp_") || zone.find('_') == std::string::npos)
		{
			return true;
		}
		
		return false;
	};

		const auto scan_directory = [&](const std::string& path)
		{
			if (!utils::io::directory_exists(path))
			{
				return;
			}

			try
			{
				for (const auto& entry : std::filesystem::directory_iterator(path))
				{
					if (entry.is_regular_file() && entry.path().extension() == ".ff")
					{
						const auto zone = entry.path().stem().string();
						if (seen.find(zone) == seen.end() && is_map_zone(zone))
						{
							map_zones.push_back(zone);
							seen.insert(zone);
						}
					}
				}
			}
			catch (...)
			{
			}
		};

		scan_directory("zone");
		scan_directory("zone/english");
		scan_directory(".");

		std::sort(map_zones.begin(), map_zones.end());
		return map_zones;
	}

	std::vector<std::string> discover_csv_files()
	{
		std::vector<std::string> csv_files;

		const auto scan_directory = [&](const std::string& path)
		{
			if (!utils::io::directory_exists(path))
			{
				return;
			}

			try
			{
				for (const auto& entry : std::filesystem::directory_iterator(path))
				{
					if (entry.is_regular_file() && entry.path().extension() == ".csv")
					{
						const auto csv_name = entry.path().stem().string();
						csv_files.push_back(csv_name);
					}
				}
			}
			catch (...)
			{
			}
		};

		scan_directory("zone_source");

		std::sort(csv_files.begin(), csv_files.end());
		return csv_files;
	}

	std::vector<std::string> discover_zonetool_maps()
	{
		std::vector<std::string> maps;
		const std::string zonetool_dir = "zonetool";

		if (!utils::io::directory_exists(zonetool_dir))
		{
			return maps;
		}

		try
		{
			for (const auto& entry : std::filesystem::directory_iterator(zonetool_dir))
			{
				if (entry.is_directory())
				{
					const auto map_name = entry.path().filename().string();
					maps.push_back(map_name);
				}
			}
		}
		catch (...)
		{
		}

		std::sort(maps.begin(), maps.end());
		return maps;
	}

	void load_zone(const std::string& zone)
	{
		const auto mode = game::get_mode();
		const std::string command = "loadzone "s + zone;

		switch (mode)
		{
		case game::h1:
			::h1::command::execute(command);
			break;
		case game::h2:
			::h2::command::execute(command);
			break;
		case game::s1:
			::s1::command::execute(command);
			break;
		case game::iw6:
			::iw6::command::execute(command);
			break;
		case game::iw7:
			::iw7::command::execute(command);
			break;
		case game::t7:
			::t7::command::execute(command);
			break;
		default:
			break;
		}
	}

		void unload_zones()
		{
			const auto mode = game::get_mode();
			const std::string command = "unloadzones"s;

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				::h2::command::execute(command);
				break;
			case game::s1:
				::s1::command::execute(command);
				break;
			case game::iw6:
				::iw6::command::execute(command);
				break;
			case game::iw7:
				::iw7::command::execute(command);
				break;
			case game::t7:
				::t7::command::execute(command);
				break;
			default:
				break;
			}
		}

		void verify_zone(const std::string& zone)
		{
			const auto mode = game::get_mode();
			const std::string command = "verifyzone "s + zone;

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				::h2::command::execute(command);
				break;
			case game::s1:
				::s1::command::execute(command);
				break;
			case game::iw6:
				::iw6::command::execute(command);
				break;
			case game::iw7:
				::iw7::command::execute(command);
				break;
			case game::t7:
				::t7::command::execute(command);
				break;
			default:
				break;
			}
		}

		void dump_zone(const std::string& zone, const game::game_mode target, const std::optional<std::unordered_set<std::string>>& asset_filter)
		{
			const auto mode = game::get_mode();
			auto target_mode = target == game::game_mode::none ? mode : target;

			std::string command = "dumpzone "s;
			if (target_mode != mode)
			{
				command += game::get_mode_as_string(target_mode) + " "s;
			}
			command += zone;

			if (asset_filter.has_value() && !asset_filter.value().empty())
			{
				std::string filter_str;
				for (const auto& type_str : asset_filter.value())
				{
					if (!filter_str.empty())
					{
						filter_str += ","s;
					}
					filter_str += type_str;
				}
				command += " "s + filter_str;
			}

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				::h2::command::execute(command);
				break;
			case game::s1:
				::s1::command::execute(command);
				break;
			case game::iw6:
				::iw6::command::execute(command);
				break;
			case game::iw7:
				::iw7::command::execute(command);
				break;
			case game::t7:
				::t7::command::execute(command);
				break;
			default:
				break;
			}
		}

		void dump_asset(const std::string& type, const std::string& name)
		{
			const auto mode = game::get_mode();
			const std::string command = "dumpasset "s + type + " "s + name;

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				::h2::command::execute(command);
				break;
			case game::s1:
				::s1::command::execute(command);
				break;
			case game::iw6:
				::iw6::command::execute(command);
				break;
			case game::iw7:
				::iw7::command::execute(command);
				break;
			case game::t7:
				::t7::command::execute(command);
				break;
			default:
				break;
			}
		}

		void dump_map(const std::string& map, const game::game_mode target, const std::optional<std::unordered_set<std::string>>& asset_filter, bool skip_common)
		{
			const auto mode = game::get_mode();
			auto target_mode = target == game::game_mode::none ? mode : target;

			std::string command = "dumpmap "s;
			if (target_mode != mode)
			{
				command += game::get_mode_as_string(target_mode) + " "s;
			}
			command += map;

			if (asset_filter.has_value() && !asset_filter.value().empty())
			{
				std::string filter_str;
				for (const auto& type_str : asset_filter.value())
				{
					if (!filter_str.empty())
					{
						filter_str += ","s;
					}
					filter_str += type_str;
				}
				command += " "s + filter_str;
			}

			command += " "s + (skip_common ? "true"s : "false"s);

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				::h2::command::execute(command);
				break;
			case game::s1:
				::s1::command::execute(command);
				break;
			case game::iw6:
				::iw6::command::execute(command);
				break;
			case game::iw7:
				::iw7::command::execute(command);
				break;
			case game::t7:
				::t7::command::execute(command);
				break;
			default:
				break;
			}
		}

		void build_zone(const std::string& zone)
		{
			const auto mode = game::get_mode();
			const std::string command = "buildzone "s + zone;

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				::h2::command::execute(command);
				break;
			case game::s1:
				::s1::command::execute(command);
				break;
			case game::iw6:
				::iw6::command::execute(command);
				break;
			case game::iw7:
				::iw7::command::execute(command);
				break;
			case game::t7:
				::t7::command::execute(command);
				break;
			default:
				break;
			}
		}

		void dump_csv(const std::string& zone)
		{
			const auto mode = game::get_mode();
			const std::string command = "dumpcsv "s + zone;

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				::h2::command::execute(command);
				break;
			case game::s1:
				::s1::command::execute(command);
				break;
			case game::iw6:
				::iw6::command::execute(command);
				break;
			case game::iw7:
				::iw7::command::execute(command);
				break;
			case game::t7:
				::t7::command::execute(command);
				break;
			default:
				break;
			}
		}

		void generate_csv(const std::string& map, bool is_sp)
		{
			const auto mode = game::get_mode();
			std::string command = "generatecsv "s + map;
			if (is_sp)
			{
				command += " sp"s;
			}

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				::h2::command::execute(command);
				break;
			case game::s1:
				::s1::command::execute(command);
				break;
			case game::iw6:
				::iw6::command::execute(command);
				break;
			case game::iw7:
				::iw7::command::execute(command);
				break;
			case game::t7:
				// Not available in T7
				break;
			default:
				break;
			}
		}

		void iterate_zones()
		{
			const auto mode = game::get_mode();
			const std::string command = "iteratezones"s;

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				// Not available in H2
				break;
			case game::s1:
				// Not available in S1
				break;
			case game::iw6:
				// Not available in IW6
				break;
			case game::iw7:
				::iw7::command::execute(command);
				break;
			case game::t7:
				::t7::command::execute(command);
				break;
			default:
				break;
			}
		}

		void dump_matching_zones(const std::string& pattern, const game::game_mode target, const std::optional<std::unordered_set<std::string>>& asset_filter)
		{
			const auto mode = game::get_mode();
			auto target_mode = target == game::game_mode::none ? mode : target;

			std::string command = "dumpmatchingzones "s;
			if (target_mode != mode)
			{
				command += game::get_mode_as_string(target_mode) + " "s;
			}
			command += pattern;

			if (asset_filter.has_value() && !asset_filter.value().empty())
			{
				std::string filter_str;
				for (const auto& type_str : asset_filter.value())
				{
					if (!filter_str.empty())
					{
						filter_str += ","s;
					}
					filter_str += type_str;
				}
				command += " "s + filter_str;
			}

			switch (mode)
			{
			case game::h1:
				::h1::command::execute(command);
				break;
			case game::h2:
				// Not available in H2
				break;
			case game::s1:
				::s1::command::execute(command);
				break;
			case game::iw6:
				// Not available in IW6
				break;
			case game::iw7:
				// Not available in IW7
				break;
			case game::t7:
				// Not available in T7
				break;
			default:
				break;
			}
		}

		void build_xmodel(const std::string& name)
		{
			const auto mode = game::get_mode();
			const std::string command = "buildxmodel "s + name;

			switch (mode)
			{
			case game::h1:
				// Not available in H1
				break;
			case game::h2:
				::h2::command::execute(command);
				break;
			case game::s1:
				// Not available in S1
				break;
			case game::iw6:
				// Not available in IW6
				break;
			case game::iw7:
				// Not available in IW7
				break;
			case game::t7:
				// Not available in T7
				break;
			default:
				break;
			}
		}
	}
}

