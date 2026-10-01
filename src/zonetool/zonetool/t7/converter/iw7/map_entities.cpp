#include <std_include.hpp>
#include "include.hpp"
#include "map_entities.hpp"

#include <utils/string.hpp>

// What an IW7 MP / CP server does with a map entity (iw7_ship):
// * G_SpawnEntitiesFromString 0x140409E90 parses one entity at a time (G_ParseSpawnVars 0x140B1E790: at most 64
//   pairs and 2048 value characters an entity, else ERR_DROP) and hands it to G_CallSpawn 0x140409A80;
// * G_CallSpawn sends classnames starting "node_" to the path node list (0x140AEA040: each takes the next aipaths
//   node slot, so a map carries exactly its aipaths' fixed nodes, in order: BO3's go, the converted aipaths'
//   negotiation nodes are added), "info_vehicle_node" to the vehicle paths, "actor_*" to the SP actor spawner (nothing in MP / CP),
//   "weapon_<name>" to a weapon pickup, and looks the rest up in two tables of spawn functions (0x1414723A0,
//   0x141472460; the per-mode tables 0x143F25A50 / 0x143F25A60 are empty in MP and CP, 0x140B63D40);
// * a classname in neither table gets a gentity that stays allocated with no type (0x140409CC8), which is how
//   stock uses mp_global_intermission; every BO3-only entity would hold one of the map's gentities for good;
// * light / light_spot / light_omni without a "pl#" and reflection_probe without an "index" free themselves
//   (0x1404003E0, 0x140400690).

namespace zonetool::t7
{
	namespace converter::iw7::map_entities
	{
		namespace
		{
			// the two spawn tables, 0x1414723A0 (12) and 0x141472460 (17)
			const std::unordered_set<std::string> iw7_spawned =
			{
				"info_notnull", "info_volume_grapple", "info_volume", "trigger_radius", "trigger_rotatable_radius",
				"sound_blend", "script_model", "script_origin", "script_weapon", "script_vehicle_collmap", "script_item",
				"script_character",
				"trigger_use_touch", "trigger_use", "trigger_multiple", "trigger_disk", "trigger_hurt", "trigger_once",
				"trigger_damage", "light", "light_spot", "light_omni", "misc_mg42", "misc_turret", "script_brushmodel",
				"script_struct", "script_vehicle", "reflection_probe", "physicsvolume",
			};

			// BO3 entities whose content the conversion already carried into the IW7 world (lights, reflection probes,
			// fog / exposure / sun / vista / grime / outdoor volumes, umbra and export volumes, build lights), that
			// IW7 draws another way (fx: client effects, IW7 places them from the level's createfx script) or that
			// only BO3's navigation and tools read
			bool render_or_tool(const std::string& classname)
			{
				static const std::unordered_set<std::string> names =
				{
					"fx", "light", "light_spot", "light_omni", "reflection_probe", "umbra_volume", "export_volume",
					"build_light", "pbg_box", "nav_volume_passable", "heli_height_lock",
				};
				return names.contains(classname) || classname.starts_with("volume_");
			}

			// IW7 spawn functions that BO3 data does not fit: BO3 vehicle types are not IW7 vehicles
			bool bo3_meaning(const std::string& classname)
			{
				return classname == "script_vehicle" || classname == "info_vehicle_node" || classname.starts_with("actor_");
			}

			std::vector<entity> path_nodes;
			std::vector<entity> script_models;
			std::vector<entity> light_entities;
		}

		void set_path_nodes(std::vector<entity> nodes)
		{
			path_nodes = std::move(nodes);
		}

		void set_script_models(std::vector<entity> models)
		{
			script_models = std::move(models);
		}

		void set_light_entities(std::vector<entity> lights)
		{
			light_entities = std::move(lights);
		}

		std::vector<entity> parse(const std::string& text)
		{
			std::vector<entity> out;
			std::size_t cursor = 0;
			while (true)
			{
				const auto open = text.find('{', cursor);
				if (open == std::string::npos)
				{
					break;
				}
				const auto close = text.find('}', open);
				if (close == std::string::npos)
				{
					throw std::runtime_error("map entities: unterminated entity");
				}
				entity e{};
				const auto body = text.substr(open + 1, close - open - 1);
				std::size_t start = 0;
				while (start < body.size())
				{
					auto end = body.find('\n', start);
					if (end == std::string::npos)
					{
						end = body.size();
					}
					const auto line = body.substr(start, end - start);
					start = end + 1;
					const auto first = line.find_first_not_of(" \t\r");
					if (first == std::string::npos)
					{
						continue;
					}
					key_value kv{};
					std::size_t after_key;
					if (line[first] == '"')
					{
						const auto k1 = line.find('"', first + 1);
						if (k1 == std::string::npos)
						{
							throw std::runtime_error("map entities: unterminated key in \"" + line + "\"");
						}
						kv.key = line.substr(first + 1, k1 - first - 1);
						after_key = k1 + 1;
					}
					else
					{
						const auto k1 = line.find_first_of(" \t", first);
						if (k1 == std::string::npos)
						{
							throw std::runtime_error("map entities: key without a value in \"" + line + "\"");
						}
						kv.key = line.substr(first, k1 - first);
						kv.quoted = false;
						after_key = k1;
					}
					const auto v0 = line.find('"', after_key);
					const auto v1 = v0 == std::string::npos ? v0 : line.find('"', v0 + 1);
					if (v1 == std::string::npos)
					{
						throw std::runtime_error("map entities: key without a quoted value in \"" + line + "\"");
					}
					kv.value = line.substr(v0 + 1, v1 - v0 - 1);
					e.keys.push_back(std::move(kv));
				}
				out.push_back(std::move(e));
				cursor = close + 1;
			}
			return out;
		}

		std::string write(const std::vector<entity>& ents)
		{
			std::string out;
			for (const auto& e : ents)
			{
				out += "{\n";
				for (const auto& kv : e.keys)
				{
					out += (kv.quoted ? "\"" + kv.key + "\"" : kv.key) + " \"" + kv.value + "\"\n";
				}
				out += "}\n";
			}
			return out;
		}

		void keep_spawnable(std::vector<entity>& ents)
		{
			std::map<std::string, unsigned int> dropped, structs;
			auto nodes = 0u;
			std::vector<entity> kept;
			kept.reserve(ents.size());
			for (auto i = 0u; i < ents.size(); i++)
			{
				auto& e = ents[i];
				const auto* found = e.get("classname");
				const auto classname = found ? *found : std::string{};
				if (i == 0 && classname == "worldspawn")
				{
					kept.push_back(std::move(e));
					continue;
				}
				if (classname.starts_with("node_"))
				{
					nodes++;
					continue;
				}
				if (render_or_tool(classname))
				{
					dropped[classname]++;
					continue;
				}
				if (!bo3_meaning(classname) && iw7_spawned.contains(classname))
				{
					kept.push_back(std::move(e));
					continue;
				}
				// struct data for the level scripts, no gentity
				structs[classname]++;
				e.set("script_type", classname);
				e.set("classname", "script_struct");
				kept.push_back(std::move(e));
			}
			ents = std::move(kept);
			ents.insert(ents.end(), path_nodes.begin(), path_nodes.end());
			ents.insert(ents.end(), script_models.begin(), script_models.end());
			ents.insert(ents.end(), light_entities.begin(), light_entities.end());

			std::string dropped_text, struct_text;
			auto dropped_count = 0u, struct_count = 0u;
			for (const auto& [name, count] : dropped)
			{
				dropped_text += utils::string::va(" %s %u", name.data(), count);
				dropped_count += count;
			}
			for (const auto& [name, count] : structs)
			{
				struct_text += utils::string::va(" %s %u", name.data(), count);
				struct_count += count;
			}
			ZONETOOL_INFO("map entities: %zu kept; %u BO3 path nodes replaced by the %zu of the converted aipaths; %u render / tool "
				"entities left out:%s; %u BO3-only entities as script_structs (BO3 classname in script_type):%s", ents.size(), nodes,
				path_nodes.size(), dropped_count, dropped_text.data(), struct_count, struct_text.data());
		}
	}
}
