#include <std_include.hpp>
#include "include.hpp"
#include "zombies_ents.hpp"

#include <utils/string.hpp>

// BO3 zombies:
// * a zone is an info_volume with script_noteworthy "player_volume", targetname the zone, target the targetname of
//   its spawner structs;
// * the spawner structs' script_noteworthy is their role: spawn_location and riser_location are where generic zombies
//   appear, the other roles serve other enemies;
// * the round start positions are script_structs "initial_spawn_points", respawns script_structs
//   "player_respawn_point".
// IW7 zombies (stock scripts/cp/zombies/zombies_spawning.gsc and stock map ents):
// * zones are info_volumes with targetname "spawn_volume"; their basename is script_linkname (a "pfN_" prefix
//   dropped), their target names the spawners (getentarray / getstructarray of self.target);
// * spawners are script_structs with script_noteworthy "static"; script_animation "spawn_ground" makes a zombie rise
//   (drop_to_ground), script_parameters "ground_spawn_no_boards" / "no_boards" as stock non-window spawners;
// * a volume's respawn locations are the "player_respawn_loc" structs inside it (ispointinvolume);
// * the start positions are "default_player_start" structs; cp_globallogic takes the intermission camera from the
//   first "mp_global_intermission" entity (origin + angles).

namespace zonetool::t7
{
	namespace converter::iw7::zombies_ents
	{
		namespace
		{
			using map_entities::entity;

			std::array<float, 3> vec3(const std::string& s)
			{
				std::array<float, 3> v{};
				std::istringstream in(s);
				in >> v[0] >> v[1] >> v[2];
				return v;
			}
		}

		void convert(std::vector<entity>& ents)
		{
			// zones -> spawn volumes
			std::unordered_set<std::string> spawner_names;
			auto volumes = 0u;
			for (auto& e : ents)
			{
				if (!e.is("classname", "info_volume") || !e.is("script_noteworthy", "player_volume"))
				{
					continue;
				}
				const auto* zone = e.get("targetname");
				if (!zone || zone->empty())
				{
					throw std::runtime_error("map entities: a BO3 zone volume without a targetname");
				}
				e.set("script_linkname", *zone);
				e.set("targetname", "spawn_volume");
				e.erase("script_noteworthy");
				if (const auto* target = e.get("target"))
				{
					spawner_names.insert(*target);
				}
				volumes++;
			}

			// generic zombie spawn points -> static spawners
			auto grounded = 0u, standing = 0u;
			for (auto& e : ents)
			{
				const auto* name = e.get("targetname");
				if (!e.is("classname", "script_struct") || !name || !spawner_names.contains(*name))
				{
					continue;
				}
				if (e.is("script_noteworthy", "riser_location"))
				{
					e.set("script_noteworthy", "static");
					e.set("script_animation", "spawn_ground");
					e.set("script_parameters", "ground_spawn_no_boards");
					grounded++;
				}
				else if (e.is("script_noteworthy", "spawn_location"))
				{
					e.set("script_noteworthy", "static");
					e.set("script_parameters", "no_boards");
					standing++;
				}
			}

			// player starts and respawn locations
			const entity* first_start = nullptr;
			auto starts = 0u, respawns = 0u;
			for (auto& e : ents)
			{
				if (e.is("classname", "script_struct") && e.is("targetname", "initial_spawn_points"))
				{
					e.set("targetname", "default_player_start");
					if (!e.get("angles"))
					{
						e.set("angles", "0 0 0");
					}
					starts++;
				}
				else if (e.is("classname", "script_struct") && e.is("targetname", "player_respawn_point"))
				{
					e.set("targetname", "player_respawn_loc");
					respawns++;
				}
			}
			for (const auto& e : ents)
			{
				if (e.is("targetname", "default_player_start"))
				{
					first_start = &e;
					break;
				}
			}
			if (!first_start || !first_start->get("origin"))
			{
				throw std::runtime_error("map entities: no BO3 initial spawn point for the CP player starts");
			}

			// the intermission camera: eye height over the first start, facing as the start does
			auto camera = vec3(*first_start->get("origin"));
			entity intermission{};
			intermission.set("classname", "mp_global_intermission");
			intermission.set("origin", utils::string::va("%g %g %g", camera[0], camera[1], camera[2] + 64.0f));
			intermission.set("angles", *first_start->get("angles"));
			ents.push_back(std::move(intermission));

			ZONETOOL_INFO("map entities: %u zones -> spawn volumes, %u rising + %u standing static spawners, %u player starts, "
				"%u respawn locations, intermission added", volumes, grounded, standing, starts, respawns);
		}
	}
}
