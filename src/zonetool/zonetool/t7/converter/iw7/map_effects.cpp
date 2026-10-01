#include <std_include.hpp>
#include "include.hpp"
#include "memory_probe.hpp"
#include "map_effects.hpp"

#include "assets/effect_material.hpp"
#include "assets/particle_system.hpp"

#include "zonetool/t7/functions.hpp"

#include <utils/string.hpp>

// BO3 places a map's effects with `fx` entities (the level's fx parser, dedicated server 0x140085650): fxdef, origin,
// angles, delay (the start offset in milliseconds, negative = already running), timescale, fxstate1-4 and exploders
// (exploder_count, exploder_index_N = "ver=2 don=0 doff=0 snd= off=0|1 name=...", 0x140001330). An effect with no
// exploder plays when the level starts. An exploder flips its effects (0x140084CC0, 0x1400845F0): one with off=0 turns
// on don milliseconds after the exploder plays (CG_CreateFX 0x140085490 starts it at that time + delay: prewarmed), one
// with off=1 is on from the start and turns off doff milliseconds after the exploder plays.
//
// IW7 places them with the client's createfx (clientSideEffects, on by default, 0x140768AE0 and the parsers it calls):
// it reads the map's scripts/cp/maps/<map>/<map>_fx.gsc (level._effect[ id ] = LoadFX( path ): id and path under 64
// characters, at most 256 of them) and gen/<map>_fx.gsc (the stock header exactly, then createOneshotEffect /
// createExploder, set_origin_and_angles, v[ "fxid" ], v[ "delay" ] in seconds, v[ "exploder" ] an integer 0-255; at most
// 256 exploder entries) as text, in the stock scripts' format. A one-shot starts at delay (from the level's
// start), an exploder's entry at delay after the exploder plays (0x140766370). The level script runs the LoadFX table
// only, as stock does.

namespace zonetool::t7
{
	namespace converter::iw7::map_effects
	{
		namespace
		{
			constexpr unsigned char elem_type_model = 7;
			constexpr unsigned char elem_type_omni_light = 8;
			constexpr unsigned char elem_type_spot_light = 10;
			constexpr unsigned char elem_type_runner = 14;
			constexpr std::size_t max_fx_path = 63;

			struct placement
			{
				std::string effect;
				float origin[3] = {};
				float angles[3] = {};
				int delay_ms = 0;
				std::string exploder; // empty: none
				bool on_at_start = true;
				int delay_on_ms = 0; // after the exploder plays
			};

			// vfx/t7/<BO3 name>; one longer than the client's 63 characters keeps its start and ends in a hash of the whole
			// BO3 name (FNV-1a), so it stays unique and the same from dump to dump
			std::string iw7_effect_name(const std::string& bo3_name)
			{
				auto name = "vfx/t7/" + bo3_name;
				if (name.size() <= max_fx_path)
				{
					return name;
				}
				auto hash = 2166136261u;
				for (const auto c : bo3_name)
				{
					hash = (hash ^ static_cast<unsigned char>(c)) * 16777619u;
				}
				const auto suffix = utils::string::va("_%08x", hash);
				return name.substr(0, max_fx_path - std::strlen(suffix)) + suffix;
			}

			void parse_vec3(const std::string* text, float out[3])
			{
				if (text)
				{
					std::sscanf(text->data(), "%f %f %f", &out[0], &out[1], &out[2]);
				}
			}

			// "ver=2 don=0 doff=0 snd= off=0 name=..." (the name runs to the end)
			std::map<std::string, std::string> exploder_fields(const std::string& text)
			{
				std::map<std::string, std::string> out;
				std::size_t at = 0;
				while (at < text.size())
				{
					const auto eq = text.find('=', at);
					if (eq == std::string::npos)
					{
						break;
					}
					const auto key = text.substr(at, eq - at);
					if (key == "name")
					{
						out[key] = text.substr(eq + 1);
						break;
					}
					const auto space = text.find(' ', eq + 1);
					out[key] = text.substr(eq + 1, space == std::string::npos ? std::string::npos : space - eq - 1);
					at = space == std::string::npos ? text.size() : space + 1;
				}
				return out;
			}

			// `report`: say what does not convert (once: the placements are read for the models and for the conversion)
			std::vector<placement> placements(const std::vector<map_entities::entity>& ents, const bool report)
			{
				std::vector<placement> out;
				auto turned_off = 0u, not_prewarmed = 0u;
				for (const auto& e : ents)
				{
					if (!e.is("classname", "fx"))
					{
						continue;
					}
					const auto* fxdef = e.get("fxdef");
					if (!fxdef || fxdef->empty())
					{
						continue;
					}
					placement p{};
					p.effect = *fxdef;
					parse_vec3(e.get("origin"), p.origin);
					parse_vec3(e.get("angles"), p.angles);
					if (const auto* delay = e.get("delay"))
					{
						p.delay_ms = std::atoi(delay->data());
					}
					const auto* count = e.get("exploder_count");
					const auto exploders = count ? std::atoi(count->data()) : 0;
					if (exploders > 1 && report)
					{
						ZONETOOL_WARNING("fx %s at (%g, %g, %g): %d exploders; IW7 gives an effect one, the first is kept", fxdef->data(),
							p.origin[0], p.origin[1], p.origin[2], exploders);
					}
					if (exploders > 0)
					{
						if (const auto* first = e.get("exploder_index_0"))
						{
							const auto fields = exploder_fields(*first);
							const auto name = fields.find("name");
							if (name != fields.end() && !name->second.empty())
							{
								p.exploder = name->second;
								const auto off = fields.find("off");
								p.on_at_start = off != fields.end() && off->second == "1";
								if (const auto don = fields.find("don"); don != fields.end())
								{
									p.delay_on_ms = std::atoi(don->second.data());
								}
								if (p.on_at_start)
								{
									turned_off++;
								}
								else if (p.delay_on_ms > 0 && p.delay_ms < 0)
								{
									not_prewarmed++;
								}
								if (const auto snd = fields.find("snd"); snd != fields.end() && !snd->second.empty() && report)
								{
									ZONETOOL_WARNING("fx %s: exploder %s plays sound %s as it turns the effect on; the sound is left out",
										fxdef->data(), p.exploder.data(), snd->second.data());
								}
							}
						}
					}
					out.push_back(std::move(p));
				}
				if (report && turned_off)
				{
					ZONETOOL_WARNING("%u effects are on from the start and BO3 turns them off when their exploder plays (off=1); IW7's "
						"createfx cannot turn a one-shot off, so they stay on", turned_off);
				}
				if (report && not_prewarmed)
				{
					ZONETOOL_WARNING("%u exploder effects turn on after a delay (don) and BO3 prewarms them by their start offset as "
						"well; IW7 gives an exploder entry one start time, so they start on time without the prewarm", not_prewarmed);
				}
				return out;
			}

			const FxEffectDef* find_effect(const std::string& name)
			{
				const auto* entry = DB_FindXAssetEntry(ASSET_TYPE_FX, name.data(), false);
				const auto* effect = entry ? static_cast<const FxEffectDef*>(entry->asset.header.data) : nullptr;
				return probe::readable(effect, sizeof(FxEffectDef)) ? effect : nullptr;
			}

			const char* effect_name(const FxEffectDef* effect)
			{
				return effect && probe::readable(effect, sizeof(void*)) ? probe::terminated(effect->name) : nullptr;
			}

			// the placed effects and every effect they spawn, each once, placed ones first (`report` as placements')
			std::vector<const FxEffectDef*> collect(const std::vector<placement>& placed, const bool report)
			{
				std::vector<const FxEffectDef*> out;
				std::unordered_set<std::string> seen;
				std::deque<std::string> todo;
				for (const auto& p : placed)
				{
					todo.push_back(p.effect);
				}
				while (!todo.empty())
				{
					const auto name = todo.front();
					todo.pop_front();
					if (!seen.insert(name).second)
					{
						continue;
					}
					const auto* effect = find_effect(name);
					if (!effect)
					{
						if (report)
						{
							ZONETOOL_WARNING("effect \"%s\" is not loaded; its placements and spawns are left out", name.data());
						}
						continue;
					}
					out.push_back(effect);
					const auto count = effect->elemDefCountLooping + effect->elemDefCountOneShot + effect->elemDefCountEmission;
					if (!probe::readable(effect->elemDefs, sizeof(FxElemDef) * count))
					{
						continue;
					}
					for (auto i = 0; i < count; i++)
					{
						const auto& elem = effect->elemDefs[i];
						const auto child = [&](const FxEffectDef* ref)
						{
							if (const auto* n = effect_name(ref); n && *n)
							{
								todo.push_back(n);
							}
						};
						if (elem.elemType == elem_type_runner && elem.visualCount)
						{
							for (auto v = 0; v < elem.visualCount; v++)
							{
								const auto& visual = elem.visualCount == 1 ? elem.visuals.instance : elem.visuals.array[v];
								child(visual.effectDef.handle);
							}
						}
						child(elem.effectOnDeath.handle);
						child(elem.effectOnImpact.handle);
						child(elem.effectEmitted.handle);
					}
				}
				return out;
			}

			std::string number(const float v)
			{
				return utils::string::va("%g", v);
			}

			void write_text(const std::string& path, const std::string& text)
			{
				filesystem::file file(path);
				file.open("wb");
				if (!file.get_fp())
				{
					ZONETOOL_FATAL("could not write %s", path.data());
				}
				file.write(text.data(), text.size(), 1);
				file.close();
			}
		}

		std::vector<const XModel*> models(const std::vector<map_entities::entity>& ents)
		{
			std::vector<const XModel*> out;
			std::unordered_set<const XModel*> seen;
			for (const auto* effect : collect(placements(ents, false), false))
			{
				const auto count = effect->elemDefCountLooping + effect->elemDefCountOneShot + effect->elemDefCountEmission;
				if (!probe::readable(effect->elemDefs, sizeof(FxElemDef) * count))
				{
					continue;
				}
				for (auto i = 0; i < count; i++)
				{
					const auto& elem = effect->elemDefs[i];
					if (elem.elemType != elem_type_model)
					{
						continue;
					}
					for (auto v = 0; v < elem.visualCount; v++)
					{
						const auto* model = elem.visualCount == 1 ? elem.visuals.instance.model : elem.visuals.array[v].model;
						const auto* name = model && probe::readable(model, sizeof(void*)) ? probe::terminated(model->name) : nullptr;
						const auto* entry = name ? DB_FindXAssetEntry(ASSET_TYPE_XMODEL, name, false) : nullptr;
						const auto* full = entry ? entry->asset.header.model : nullptr;
						if (!full)
						{
							ZONETOOL_WARNING("effect \"%s\": model \"%s\" is not loaded", effect->name, name ? name : "?");
							continue;
						}
						if (seen.insert(full).second)
						{
							out.push_back(full);
						}
					}
				}
			}
			return out;
		}

		std::vector<std::string> convert(const std::vector<map_entities::entity>& ents, const std::string& map)
		{
			const auto placed = placements(ents, true);
			auto seeds = placed;
			// effects that only scripts play (no fx entity places them): ZT_SCRIPT_EFFECTS, comma separated BO3 names
			if (const auto* names = std::getenv("ZT_SCRIPT_EFFECTS"))
			{
				for (const auto& name : utils::string::split(names, ','))
				{
					if (find_effect(name.data()))
					{
						seeds.push_back({ .effect = name });
					}
					else
					{
						ZONETOOL_WARNING("script effect \"%s\" is not loaded", name.data());
					}
				}
			}
			const auto effects = collect(seeds, true);

			// the converted effects and their materials
			particlesystem::references refs{};
			refs.effect = iw7_effect_name;
			refs.material = effect_material::request;
			for (const auto* effect : effects)
			{
				try
				{
					particlesystem::dump(const_cast<FxEffectDef*>(effect), refs);
				}
				catch (const std::exception& e)
				{
					ZONETOOL_ERROR("effect \"%s\": not converted: %s", effect->name, e.what());
				}
			}
			effect_material::write_all();

			// createfx ids: the effect's name without its folder, the folder kept where two share a name
			std::map<std::string, std::string> ids;
			std::map<std::string, unsigned int> base_count;
			for (const auto& p : seeds)
			{
				if (find_effect(p.effect) && !ids.contains(p.effect))
				{
					ids[p.effect] = "";
					base_count[p.effect.substr(p.effect.find_last_of('/') + 1)]++;
				}
			}
			for (auto& [name, id] : ids)
			{
				auto base = name.substr(name.find_last_of('/') + 1);
				if (base_count[base] > 1)
				{
					base = name;
					std::replace(base.begin(), base.end(), '/', '_');
				}
				id = base;
			}

			// exploders by name, numbered from 1 (IW7 exploder ids are integers 0-255)
			std::map<std::string, unsigned int> exploder_ids;
			for (const auto& p : placed)
			{
				if (!p.exploder.empty() && !p.on_at_start && ids.contains(p.effect))
				{
					exploder_ids.emplace(p.exploder, 0);
				}
			}
			auto next_id = 1u;
			for (auto& [name, id] : exploder_ids)
			{
				id = next_id++;
			}
			if (next_id > 256)
			{
				ZONETOOL_ERROR("%u exploders; IW7 numbers them 0-255", next_id - 1);
			}

			std::vector<std::string> rows;
			// the light template the converted light elements draw with
			const auto lights = std::ranges::any_of(effects, [](const FxEffectDef* effect)
			{
				const auto count = effect->elemDefCountLooping + effect->elemDefCountOneShot + effect->elemDefCountEmission;
				return probe::readable(effect->elemDefs, sizeof(FxElemDef) * count) &&
					std::any_of(effect->elemDefs, effect->elemDefs + count, [](const FxElemDef& elem)
					{
						return elem.elemType == elem_type_omni_light || elem.elemType == elem_type_spot_light;
					});
			});
			if (lights)
			{
				rows.push_back("lightdef,"s + particlesystem::light_def);
			}

			std::string table = "\r\nmain()\r\n{\r\n";
			for (const auto& [name, id] : ids)
			{
				auto path = iw7_effect_name(name) + ".vfx";
				if (path.size() > max_fx_path)
				{
					path = iw7_effect_name(name); // the client drops the extension anyway
				}
				if (path.size() > max_fx_path)
				{
					ZONETOOL_ERROR("effect path %s is longer than the %zu characters IW7's client reads", path.data(), max_fx_path);
				}
				table += utils::string::va("\tlevel._effect[ \"%s\" ] = LoadFX( \"%s\" );\r\n", id.data(), path.data());
				rows.push_back("vfx," + iw7_effect_name(name));
			}
			table += "}\r\n";

			std::string gen = "//_createfx generated. Do not touch!!\r\n#include scripts\\common\\utility;\r\n#include scripts\\common\\createfx;\r\n\r\nmain()\r\n{\r\n";
			auto count = 0u, exploding = 0u;
			std::string entries;
			for (const auto& p : placed)
			{
				const auto id = ids.find(p.effect);
				if (id == ids.end())
				{
					continue;
				}
				const auto exploder = !p.exploder.empty() && !p.on_at_start;
				entries += utils::string::va("\tent = %s( \"%s\" );\r\n", exploder ? "createExploder" : "createOneshotEffect", id->second.data());
				entries += "\tent set_origin_and_angles( (" + number(p.origin[0]) + ", " + number(p.origin[1]) + ", " + number(p.origin[2]) +
					"), (" + number(p.angles[0]) + ", " + number(p.angles[1]) + ", " + number(p.angles[2]) + ") );\r\n";
				entries += utils::string::va("\tent.v[ \"fxid\" ] = \"%s\";\r\n", id->second.data());
				// an exploder entry starts its delay after the exploder plays: BO3's delay-on when it has one, else the
				// start offset (see the header comment)
				const auto delay_ms = exploder && p.delay_on_ms > 0 ? p.delay_on_ms : p.delay_ms;
				entries += "\tent.v[ \"delay\" ] = " + number(static_cast<float>(delay_ms) / 1000.0f) + ";\r\n";
				if (exploder)
				{
					entries += utils::string::va("\tent.v[ \"exploder\" ] = \"%u\";\r\n", exploder_ids[p.exploder]);
					exploding++;
				}
				entries += "\r\n";
				count++;
			}
			gen += utils::string::va("\t// CreateFX fx entities placed: %u\r\n", count) + entries + "}\r\n";
			// the client's createfx tables (0x140766760, 0x1407673F0) keep the first 256 of each and drop the rest
			if (ids.size() > 256)
			{
				ZONETOOL_ERROR("%zu placed effects; IW7's client reads the first 256 LoadFX entries only", ids.size());
			}
			if (exploding > 256)
			{
				ZONETOOL_ERROR("%u exploder entries; IW7's client keeps the first 256 only", exploding);
			}

			const auto folder = "scripts\\cp\\maps\\" + map + "\\";
			write_text(folder + map + "_fx.gsc", table);
			write_text(folder + "gen\\" + map + "_fx.gsc", gen);
			rows.push_back("rawfile,scripts/cp/maps/" + map + "/" + map + "_fx.gsc");
			rows.push_back("rawfile,scripts/cp/maps/" + map + "/gen/" + map + "_fx.gsc");

			std::string exploder_list;
			for (const auto& [name, id] : exploder_ids)
			{
				exploder_list += utils::string::va(" %u=%s", id, name.data());
			}
			ZONETOOL_INFO("map effects: %zu effects converted (%zu placed), %u placements of which %u in exploders:%s", effects.size(),
				ids.size(), count, exploding, exploder_list.data());
			effect_material::clear();
			return rows;
		}
	}
}
