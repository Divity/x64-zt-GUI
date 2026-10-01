#include <std_include.hpp>
#include "include.hpp"
#include "map_common.hpp"

#include "assets/clipmap.hpp"
#include "assets/comworld.hpp"
#include "assets/gfxworld.hpp"
#include "assets/navmesh.hpp"
#include "assets/world_material.hpp"
#include "map_effects.hpp"
#include "map_entities.hpp"
#include "assets/xmodel.hpp"
#include "assets/xmodel_mesh.hpp"
#include "assets/static_model_clusters.hpp"
#include "parallel.hpp"

#include "zonetool/t7/functions.hpp"
#include "zonetool/iw7/assets/fxworld.hpp"
#include "zonetool/iw7/assets/glassworld.hpp"

namespace zonetool::t7
{
	namespace converter::iw7::map
	{
		namespace
		{
			// The loader hands DB_AddXAsset the asset struct in zone memory; the struct is copied
			// so the conversion does not depend on that block, the data it points to stays in
			// the zone for as long as the zone is loaded.
			template <typename T>
			struct pending_asset
			{
				bool present = false;
				T asset{};

				void set(void* header)
				{
					std::memcpy(&asset, header, sizeof(T));
					present = true;
				}

				T* get()
				{
					return present ? &asset : nullptr;
				}

				void clear()
				{
					present = false;
				}
			};

			pending_asset<clipMap_t> pending_clipmap;
			pending_asset<ComWorld> pending_comworld;
			pending_asset<GameWorld> pending_gameworld;
			pending_asset<MapEnts> pending_mapents;
			pending_asset<GfxWorld> pending_gfxworld;
			pending_asset<NavMeshData> pending_navmesh;
			// The zone's own XModels wait for the map too, so their materials bake inside the map's model material
			// atlases. By name: the header the loader hands over does not outlive the load, the asset entry does
			std::vector<std::string> pending_model_names;

			std::vector<XModel*> zone_models()
			{
				std::vector<XModel*> out;
				for (const auto& name : pending_model_names)
				{
					const auto* entry = zonetool::t7::DB_FindXAssetEntry(ASSET_TYPE_XMODEL, name.data(), false);
					if (entry && entry->asset.header.model)
					{
						out.push_back(entry->asset.header.model);
					}
				}
				return out;
			}

			// The XModels the map places: its static models and the models its entities name ("*n" are brush
			// models, "?n" trigger volumes).
			std::vector<const XModel*> placed_models(const GfxWorld* gfx_world, const MapEnts* map_ents)
			{
				std::vector<const XModel*> out;
				std::unordered_set<const XModel*> seen;
				const auto add = [&](const XModel* model)
				{
					if (model && seen.insert(model).second)
					{
						out.push_back(model);
					}
				};

				if (gfx_world)
				{
					for (auto i = 0u; i < gfx_world->dpvs.smodelCount; i++)
					{
						add(gfx_world->dpvs.smodelDrawInsts[i].model);
					}
				}

				if (map_ents && map_ents->entityString)
				{
					const std::string ents(map_ents->entityString, strnlen(map_ents->entityString, map_ents->numEntityChars));
					static const std::regex entity_block(R"re(\{[^{}]*\})re");
					static const std::regex model_key(R"re("model" "([^"]*)")re");
					std::set<std::string> missing;
					for (std::sregex_iterator e(ents.begin(), ents.end(), entity_block), end; e != end; ++e)
					{
						const auto entity = e->str();
						std::smatch model;
						if (!std::regex_search(entity, model, model_key))
						{
							continue;
						}
						const auto name = model[1].str();
						if (name.empty() || name[0] == '*' || name[0] == '?')
						{
							continue;
						}
						const auto* entry = zonetool::t7::DB_FindXAssetEntry(ASSET_TYPE_XMODEL, name.data(), false);
						if (!entry || !entry->asset.header.model)
						{
							// a script_struct or an actor spawner (actor_*: its aitype picks the character) is never spawned
							// with its model: that is Radiant's preview, loaded only when a script spawns it
							if (entity.find("\"classname\" \"script_struct\"") == std::string::npos
								&& entity.find("\"classname\" \"actor_") == std::string::npos)
							{
								missing.insert(name);
							}
							continue;
						}
						add(entry->asset.header.model);
					}
					for (const auto& name : missing)
					{
						ZONETOOL_WARNING("map entity model \"%s\" is not loaded", name.data());
					}
				}

				return out;
			}

			// The texture coordinates and vertex colours each placed model material's surfaces use: every surface of every
			// LOD the converted models keep, each mesh read once.
			std::unordered_map<const Material*, world_material::surface_usage> used_vertex_data(const std::vector<const XModel*>& placed)
			{
				const auto start = std::chrono::steady_clock::now();
				std::vector<std::pair<XModelMesh*, std::vector<const Material*>>> lods;
				std::unordered_map<XModelMesh*, std::uint32_t> mesh_index;
				std::vector<XModelMesh*> meshes;
				for (const auto* model : placed)
				{
					for (auto& lod : xmodel::lod_meshes(model))
					{
						if (mesh_index.emplace(lod.first, static_cast<std::uint32_t>(meshes.size())).second)
						{
							meshes.push_back(lod.first);
						}
						lods.emplace_back(std::move(lod));
					}
				}

				std::vector<std::vector<world_material::surface_usage>> usage(meshes.size());
				parallel_for(static_cast<std::uint32_t>(meshes.size()), [&](const std::uint32_t i, std::uint32_t)
				{
					usage[i] = xmodel_mesh::surface_usage(meshes[i]);
				});

				std::unordered_map<const Material*, world_material::surface_usage> used;
				for (const auto& [mesh, materials] : lods)
				{
					const auto& surfaces = usage[mesh_index[mesh]];
					for (auto s = 0u; s < materials.size() && s < surfaces.size(); s++)
					{
						if (materials[s])
						{
							used[materials[s]].add(surfaces[s]);
						}
					}
				}
				ZONETOOL_INFO("model materials: texture coordinates and vertex colours of %zu meshes read (%.1f s)", meshes.size(),
					std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
				return used;
			}
		}

		namespace
		{
			// "maps/zm/<name>.d3dbsp" -> "<name>"
			std::string t7_map_name(const std::string& t7_bsp_name)
			{
				auto name = t7_bsp_name;

				const auto slash = name.find_last_of("/\\");
				if (slash != std::string::npos)
				{
					name = name.substr(slash + 1);
				}

				constexpr std::string_view extension = ".d3dbsp";
				if (name.ends_with(extension))
				{
					name.resize(name.size() - extension.size());
				}

				return name;
			}
		}

		bool zombies_map(const std::string& t7_bsp_name)
		{
			return t7_map_name(t7_bsp_name).starts_with("zm_");
		}

		std::string map_name(const std::string& t7_bsp_name)
		{
			// A BO3 zombies map becomes an IW7 zombies (CP) map, and IW7 decides that by the map's name: the level
			// and fx scripts load from scripts/cp/maps/<map>/ only when the name starts with "cp_" (0x140CDB930,
			// strncmp(map, "cp_", 3), used by 0x140B5C650 and 0x140768AE0), scripts/mp/maps/ otherwise.
			const auto name = t7_map_name(t7_bsp_name);
			return zombies_map(t7_bsp_name) ? "cp_" + name : name;
		}

		std::string bsp_name(const std::string& t7_bsp_name)
		{
			const auto name = map_name(t7_bsp_name);

			std::string folder;
			if (name.starts_with("cp_"))
			{
				folder = "cp/";
			}
			else if (name.starts_with("mp_"))
			{
				folder = "mp/";
			}

			return "maps/" + folder + name + ".d3dbsp";
		}

		bool remember(const XAssetType type, void* header)
		{
			if (!header)
			{
				return false;
			}

			switch (type)
			{
			case ASSET_TYPE_CLIPMAP:
				pending_clipmap.set(header);
				return true;
			case ASSET_TYPE_COMWORLD:
				pending_comworld.set(header);
				return true;
			case ASSET_TYPE_GAMEWORLD:
				pending_gameworld.set(header);
				return true;
			case ASSET_TYPE_MAP_ENTS:
				pending_mapents.set(header);
				return true;
			case ASSET_TYPE_GFXWORLD:
				pending_gfxworld.set(header);
				return true;
			case ASSET_TYPE_NAVMESH:
				pending_navmesh.set(header);
				return true;
			case ASSET_TYPE_XMODEL:
				if (const auto* model = static_cast<XModel*>(header); model->name)
				{
					pending_model_names.emplace_back(model->name);
					return true;
				}
				return false;
			default:
				return false;
			}
		}

		void convert_pending()
		{
			auto* gfx_world = pending_gfxworld.get();

			// the IW7 zone rows of what this conversion wrote (zone_source\<map>.csv in the dump)
			std::vector<std::string> model_rows;
			std::vector<std::string> map_rows;
			std::vector<std::string> effect_rows;

			// the map's entities: the effects its `fx` entities place are converted with it
			std::vector<map_entities::entity> ents;
			if (const auto* map_ents = pending_mapents.get(); map_ents && map_ents->entityString)
			{
				ents = map_entities::parse(std::string(map_ents->entityString, strnlen(map_ents->entityString, map_ents->numEntityChars)));
			}

			// the light list first: the sun comes from the GfxWorld's sun volumes and the GfxWorld
			// conversion maps BO3 light indices through it
			if (auto* com_world = pending_comworld.get())
			{
				comworld::dump(com_world, gfx_world);
				map_rows.emplace_back("com_map," + bsp_name(com_world->name));
			}

			if (gfx_world)
			{
				if (!ents.empty())
				{
					if (const auto* hole = ents[0].get("umbraSmallestHole"))
					{
						gfxworld::set_umbra_smallest_hole(std::strtof(hole->data(), nullptr));
					}
				}
				gfxworld::dump(gfx_world);
				// after the GfxWorld: its conversion classifies the materials and plans their bakes
				world_material::dump_all(gfx_world);

				// the placed models' materials, after the world's (whose decal layers set the model decal keys); the models the
				// placed effects draw are converted with them
				auto placed = placed_models(gfx_world, pending_mapents.get());
				for (const auto* model : map_effects::models(ents))
				{
					if (std::ranges::find(placed, model) == placed.end())
					{
						placed.push_back(model);
					}
				}
				// with the zone's own models (pending_models), whose materials bake with the placed ones'
				auto with_zone = placed;
				for (const auto* model : zone_models())
				{
					if (std::ranges::find(with_zone, model) == with_zone.end())
					{
						with_zone.push_back(model);
					}
				}
				std::vector<const Material*> model_materials;
				std::unordered_set<const Material*> seen;
				for (const auto* model : with_zone)
				{
					for (const auto* material : xmodel::materials(model))
					{
						if (seen.insert(material).second)
						{
							model_materials.push_back(material);
						}
					}
				}
				world_material::dump_models(model_materials, used_vertex_data(with_zone));

				// the merged static models, now that their members' materials (and baked texture coordinates) are known
				static_model_clusters::dump_all();

				// a model placed only inside merged static models needs no XModel of its own; one an entity or an effect
				// places does
				std::unordered_set<const XModel*> by_entity;
				for (const auto* model : placed_models(nullptr, pending_mapents.get()))
				{
					by_entity.insert(model);
				}
				for (const auto* model : map_effects::models(ents))
				{
					by_entity.insert(model);
				}
				const auto needed = [&](const XModel* model)
				{
					return by_entity.contains(model) || !static_model_clusters::only_clustered(model);
				};

				// IW7 loads a map's FxWorld and GlassWorld by the BSP name with the rest of its assets. BO3 has no
				// breakable-glass world asset, so both are written empty (an all-zero glass system and a G_GlassData
				// with no pieces).
				const auto name = bsp_name(gfx_world->name);
				zonetool::iw7::FxWorld fx_world{};
				fx_world.name = name.data();
				zonetool::iw7::fx_world::dump(&fx_world);

				zonetool::iw7::G_GlassData glass_data{};
				zonetool::iw7::GlassWorld glass_world{};
				glass_world.name = name.data();
				glass_world.g_glassData = &glass_data;
				zonetool::iw7::glass_world::dump(&glass_world);

				// The placed models are part of the map: every one the zone's own XModel dump did not write (none when the
				// dump leaves XModels out, or a static model from another loaded zone) is written here.
				// IW7 drops the whole map when a static model's XModel is missing (StaticModels_CreateClipmapShapes
				// 0x140574CF0, "This level has MiscModel errors").
				auto added = 0u;
				for (const auto* model : placed)
				{
					if (xmodel::dumped(model->name) || !needed(model))
					{
						continue;
					}
					const auto* entry = DB_FindXAssetEntry(ASSET_TYPE_XMODEL, model->name, false);
					if (entry && entry->placeholder)
					{
						ZONETOOL_WARNING("placed model \"%s\" belongs to a zone that was not loaded (placeholder)", model->name);
					}
					xmodel::dump(const_cast<XModel*>(model));
					added++;
				}
				ZONETOOL_INFO("xmodels: %u of the %zu placed models written with the map", added, placed.size());

				// every placed XModel is listed: the map's static models are not resolved as its dependencies
				auto only_merged = 0u;
				for (const auto* model : placed)
				{
					if (!needed(model))
					{
						only_merged++;
						continue;
					}
					model_rows.emplace_back("xmodel," + xmodel::iw7_name(model->name));
				}
				for (const auto& merged : static_model_clusters::model_names())
				{
					model_rows.emplace_back("xmodel," + merged);
				}
				std::sort(model_rows.begin(), model_rows.end());
				ZONETOOL_INFO("xmodels: %u models placed only inside merged static models get no row", only_merged);

				const auto base = map_name(gfx_world->name);
				map_rows.emplace_back("fx_map," + name);
				map_rows.emplace_back("gfx_map," + name);
				map_rows.emplace_back("glass_map," + name);
				map_rows.emplace_back("gfx_map_trzone," + base);
				// the lightmap bake writes one atlas (world_lightmap.cpp)
				map_rows.emplace_back("gfxlightmap,*lightmap0");

				// the placed effects, their materials and the createfx scripts
				effect_rows = map_effects::convert(ents, base);
			}

			if (auto* nav_mesh = pending_navmesh.get(); nav_mesh && gfx_world)
			{
				const auto name = bsp_name(gfx_world->name);
				try
				{
					const auto* map_ents = pending_mapents.get();
					const auto entities = map_ents && map_ents->entityString
						? std::string(map_ents->entityString, strnlen(map_ents->entityString, map_ents->numEntityChars)) : std::string{};
					auto nodes = navmesh::convert(nav_mesh, name, entities);
					map_rows.emplace_back("navmesh," + name);
					if (!nodes.empty())
					{
						map_rows.emplace_back("aipaths," + name);
					}
					// before the clipmap: its conversion writes the ents, which carry the aipaths' node entities
					map_entities::set_path_nodes(std::move(nodes));
				}
				catch (const std::exception& e)
				{
					ZONETOOL_ERROR("navmesh \"%s\": not converted: %s", nav_mesh->name, e.what());
				}
			}

			if (auto* clip_map = pending_clipmap.get())
			{
				clipmap::dump(clip_map);
				map_rows.emplace_back("map_ents," + bsp_name(clip_map->name));
				map_rows.emplace_back("col_map," + bsp_name(clip_map->name));
			}

			if (gfx_world && !map_rows.empty())
			{
				const auto base = map_name(gfx_world->name);
				filesystem::file file("zone_source\\" + base + ".csv");
				file.open("wb");
				if (!file.get_fp())
				{
					ZONETOOL_FATAL("could not write zone_source\\%s.csv", base.data());
				}
				std::fprintf(file.get_fp(), "// %s: the IW7 assets converted from BO3 %s (x64-zt T7 -> IW7)\n", base.data(), gfx_world->name);
				std::fprintf(file.get_fp(), "// every XModel the map places (static models and entity models)\n");
				for (const auto& row : model_rows)
				{
					std::fprintf(file.get_fp(), "%s\n", row.data());
				}
				std::fprintf(file.get_fp(), "// the map assets, loaded by the BSP name\n");
				for (const auto& row : map_rows)
				{
					std::fprintf(file.get_fp(), "%s\n", row.data());
				}
				std::fprintf(file.get_fp(), "// the placed effects (their children, materials and models load with them) and the createfx scripts\n");
				for (const auto& row : effect_rows)
				{
					std::fprintf(file.get_fp(), "%s\n", row.data());
				}
				file.close();
				ZONETOOL_INFO("zone rows: zone_source\\%s.csv (%zu models, %zu map assets, %zu effect rows)", base.data(), model_rows.size(),
					map_rows.size(), effect_rows.size());
			}

			// the zone's own XModels the map did not write, now that the model materials are baked
			for (auto* model : zone_models())
			{
				if (!xmodel::dumped(model->name))
				{
					xmodel::dump(model);
				}
			}
			pending_model_names.clear();

			pending_clipmap.clear();
			pending_comworld.clear();
			pending_gameworld.clear();
			pending_mapents.clear();
			pending_gfxworld.clear();
			pending_navmesh.clear();

			xmodel_mesh::dump_remaining();
			xmodel::clear();
			static_model_clusters::clear();
		}
	}
}
