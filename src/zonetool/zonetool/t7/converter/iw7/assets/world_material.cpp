#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "zonetool/t7/converter/iw7/memory_probe.hpp"
#include "world_material.hpp"
#include "world_material_bake.hpp"
#include "material.hpp"
#include "../parallel.hpp"

#include "zonetool/t7/functions.hpp"

#include <utils/string.hpp>

#include <thread>

// Sources:
// * BO3 draw bucket: Material +0x273 is the forward bucket (4 lit opaque, 5 lit trans, 8 emissive fx,
//   17 never drawn). A techset with renderFlags & 2 draws in the gbuffer instead, bucket 2, or 3 when
//   renderFlags & 8 (decal) - sub_141C70B40 in the client.
// * BO3 decal order: the draw list is stable-sorted ascending on the material's GfxSortKey
//   (sub_141C91BF0), whose decalSurfSort field is 63 - layerSortDecal, so layer 22 is drawn first and
//   layer 1 last. The static surface order itself is not by layer.
// * IW7 draws its decal range in ascending material sort key (7 lit decal .. 17 top decal) and
//   reserves 14 for runtime effect decals (sortKeyEffectDecal). Stock world decals use 7-13 and
//   15-17.
// * Shadow casting: BO3 builds surfaceCastsShadow from gameFlags & 0x40 (sub_141C93F90).
// * IW7 shadow-only surfaces: w/shadowcaster and w/caulk_shadow are sort key 2, camera region 11,
//   lightmap index 31 in stock maps.

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace world_material
		{
			namespace
			{
				constexpr unsigned char t7_bucket_lit_opaque = 4;
				constexpr unsigned char t7_bucket_lit_trans = 5;
				// GfxCameraRegion 7, CAMERA_REGION_LIT_POST_RESOLVE: forward-lit surfaces drawn after the scene resolve
				constexpr unsigned char t7_bucket_lit_post_resolve = 7;
				constexpr unsigned char t7_bucket_emissive_fx = 8;
				constexpr unsigned char t7_bucket_not_drawn = 17;

				constexpr unsigned int t7_techset_deferred = 2;
				constexpr unsigned int t7_techset_decal = 8;

				constexpr unsigned int t7_game_flag_casts_shadow = 0x40;

				constexpr unsigned char iw7_region_lit_opaque = 0;
				constexpr unsigned char iw7_region_lit_sss = 1; // every stock mo_ skin (sss) and eye material
				constexpr unsigned char iw7_region_lit_decal = 2;
				constexpr unsigned char iw7_region_lit_trans = 3;
				constexpr unsigned char iw7_region_shadow_only = 11;

				constexpr unsigned char iw7_sort_opaque = 2;
				constexpr unsigned char iw7_sort_trans = 26;
				constexpr unsigned char iw7_sort_emissive_fx = 30;

				// IW7 decal keys available to static decals, bottom to top
				constexpr unsigned char iw7_decal_keys[] = { 7, 8, 9, 10, 11, 12, 13, 15, 16, 17 };

				std::unordered_map<const Material*, info> infos;
				std::unordered_map<const Material*, info> model_infos;
				std::unordered_map<const Material*, uv_bounds> world_usage;
				std::vector<const Material*> world_atlas_order; // plan_world_atlases
				std::map<unsigned char, unsigned char> decal_key_for_layer;

				// BO3's global_invisible materials (shadow-only geometry): a level's copies (global_invisible1, ...) keep a lit techset,
				// which would draw their placeholder texture
				bool global_invisible(const Material* material)
				{
					if (!material || !material->name)
					{
						return false;
					}
					const std::string name = material->name;
					const auto slash = name.find_last_of("/\\");
					return name.compare(slash == std::string::npos ? 0 : slash + 1, 16, "global_invisible") == 0;
				}

				unsigned int techset_flags(const Material* material)
				{
					return material->techniqueSet ? material->techniqueSet->renderFlags : 0;
				}

				// The IW7 key of a model decal: BO3 sorts model and world decals in one list by layer, so a model
				// decal takes the key of its layer's world decals, or of the nearest layer the world uses.
				unsigned char model_decal_key(const unsigned char layer)
				{
					if (decal_key_for_layer.empty())
					{
						return iw7_decal_keys[0];
					}
					const auto exact = decal_key_for_layer.find(layer);
					if (exact != decal_key_for_layer.end())
					{
						return exact->second;
					}
					auto best = decal_key_for_layer.begin();
					for (auto it = decal_key_for_layer.begin(); it != decal_key_for_layer.end(); ++it)
					{
						if (std::abs(it->first - layer) < std::abs(best->first - layer))
						{
							best = it;
						}
					}
					return best->second;
				}

				// Groups the BO3 layers (bottom first) into at most as many groups as IW7 has keys,
				// always merging the adjacent pair that covers the fewest surfaces.
				void assign_decal_keys(const std::map<unsigned char, unsigned int>& surfaces_per_layer)
				{
					decal_key_for_layer.clear();

					// layer 22 is drawn first in BO3, so it goes to the lowest IW7 key
					std::vector<std::pair<std::vector<unsigned char>, unsigned int>> groups;
					for (auto it = surfaces_per_layer.rbegin(); it != surfaces_per_layer.rend(); ++it)
					{
						groups.push_back({ { it->first }, it->second });
					}

					constexpr auto key_count = std::size(iw7_decal_keys);
					while (groups.size() > key_count)
					{
						auto best = 0u;
						for (auto i = 1u; i + 1 < groups.size(); i++)
						{
							if (groups[i].second + groups[i + 1].second < groups[best].second + groups[best + 1].second)
							{
								best = i;
							}
						}

						auto& into = groups[best];
						auto& from = groups[best + 1];
						into.first.insert(into.first.end(), from.first.begin(), from.first.end());
						into.second += from.second;
						groups.erase(groups.begin() + best + 1);
					}

					for (auto g = 0u; g < groups.size(); g++)
					{
						for (const auto layer : groups[g].first)
						{
							decal_key_for_layer[layer] = iw7_decal_keys[g];
						}
					}

					std::string summary;
					for (const auto& [layer, key] : decal_key_for_layer)
					{
						summary += utils::string::va(" %u->%u", layer, key);
					}
					ZONETOOL_INFO("world materials: BO3 decal layer -> IW7 sort key%s", summary.data());
				}

				info classify(const Material* material)
				{
					info result{};
					result.name = material->name;
					if (const auto found = world_usage.find(material); found != world_usage.end())
					{
						result.used_uv = found->second;
					}
					result.casts_shadow = (material->info.gameFlags & t7_game_flag_casts_shadow) != 0;

					const auto bucket = static_cast<unsigned char>(material->cameraRegion);
					const auto flags = techset_flags(material);

					if (bucket == t7_bucket_not_drawn || global_invisible(material))
					{
						result.cls = surface_class::shadow_only;
						result.sort_key = iw7_sort_opaque;
						result.camera_region = iw7_region_shadow_only;
						result.lightmapped = false;
					}
					else if ((flags & t7_techset_deferred) && (flags & t7_techset_decal))
					{
						result.cls = surface_class::decal;
						result.camera_region = iw7_region_lit_decal;
						const auto key = decal_key_for_layer.find(material->info.layerSortDecal);
						if (key == decal_key_for_layer.end())
						{
							ZONETOOL_FATAL("world material \"%s\": decal layer %u was not seen by world_material::prepare",
								material->name, material->info.layerSortDecal);
						}
						result.sort_key = key->second;
					}
					else if ((flags & t7_techset_deferred) || bucket == t7_bucket_lit_opaque)
					{
						result.cls = surface_class::opaque;
						result.sort_key = iw7_sort_opaque;
						result.camera_region = iw7_region_lit_opaque;
					}
					else if (bucket == t7_bucket_lit_trans || bucket == t7_bucket_lit_post_resolve)
					{
						result.cls = surface_class::trans;
						result.sort_key = iw7_sort_trans;
						result.camera_region = iw7_region_lit_trans;
					}
					else if (bucket == t7_bucket_emissive_fx)
					{
						result.cls = surface_class::trans;
						result.sort_key = iw7_sort_emissive_fx;
						result.camera_region = iw7_region_lit_trans;
					}
					else
					{
						ZONETOOL_FATAL("world material \"%s\": unhandled BO3 draw bucket %u (techset flags 0x%X)",
							material->name, bucket, flags);
					}

					// BO3's water (its lit technique draws the refracted scene itself, some from the opaque bucket): IW7's
					// refractive water, which reads the scene drawn before it, a transparent-region techset
					const auto bo3 = bake::bo3_template(material);
					if (bo3.starts_with("lit_water_sim_flow") || bo3.starts_with("water_shore_flow") || bo3.starts_with("lit_transparent_water_flow"))
					{
						result.cls = surface_class::trans;
						result.sort_key = iw7_sort_trans;
						result.camera_region = iw7_region_lit_trans;
					}

					const auto& plan = bake::get_plan(material, result);
					result.uv_period[0] = plan.uv_period[0];
					result.uv_period[1] = plan.uv_period[1];
					result.uv_origin[0] = plan.uv_origin[0];
					result.uv_origin[1] = plan.uv_origin[1];
					result.uv_span[0] = plan.uv_span[0];
					result.uv_span[1] = plan.uv_span[1];
					result.alpha_test = plan.alpha_test;
					result.vertex_alpha = plan.supported && plan.multiply;
					if (plan.supported && plan.reveal_decal)
					{
						// a transparent surface revealed by its vertex alpha, drawn as IW7's world reveal decal: in the decal
						// region, at the top decal key, after the world's decals as BO3 draws its transparent bucket after them
						result.camera_region = iw7_region_lit_decal;
						result.sort_key = iw7_decal_keys[std::size(iw7_decal_keys) - 1];
					}

					return result;
				}

				// the same classes for a model material (the stock mo_ materials use the same sort keys and camera
				// regions as the world's: opaque 2 / region 0, decals 7-17 / region 2, trans 24-31 / region 3)
				info classify_model(const Material* material, const surface_usage* used = nullptr)
				{
					info result{};
					result.model = true;
					result.name = converter::iw7::material::get_converted_name(const_cast<Material*>(material));
					result.casts_shadow = (material->info.gameFlags & t7_game_flag_casts_shadow) != 0;
					result.lightmapped = false;
					if (used)
					{
						result.used_uv = used->uv;
						if (!used->colour.empty())
						{
							result.used_colour = used->colour;
						}
					}

					if (!readable(material->techniqueSet, sizeof(MaterialTechniqueSet)))
					{
						result.techset_loaded = false;
						return result;
					}
					// a technique set of a zone that was not loaded is a placeholder copy of the default one
					if (material->techniqueSet->name)
					{
						const auto* entry = zonetool::t7::DB_FindXAssetEntry(ASSET_TYPE_TECHNIQUE_SET, material->techniqueSet->name, true);
						if (entry && entry->placeholder)
						{
							result.techset_loaded = false;
							return result;
						}
					}

					const auto bucket = static_cast<unsigned char>(material->cameraRegion);
					const auto flags = techset_flags(material);

					if (bucket == t7_bucket_not_drawn || global_invisible(material))
					{
						result.cls = surface_class::shadow_only;
						result.sort_key = iw7_sort_opaque;
						result.camera_region = iw7_region_shadow_only;
					}
					else if ((flags & t7_techset_deferred) && (flags & t7_techset_decal))
					{
						result.cls = surface_class::decal;
						result.camera_region = iw7_region_lit_decal;
						result.sort_key = 0; // set by dump_models, once prepare() has seen the world's decal layers
					}
					else if ((flags & t7_techset_deferred) || bucket == t7_bucket_lit_opaque)
					{
						result.cls = surface_class::opaque;
						result.sort_key = iw7_sort_opaque;
						// BO3's skin and eye become IW7 skin and eye (sss techsets), which IW7 draws in their own region
						const auto bo3 = bake::bo3_template(material);
						result.camera_region = bo3 == "skin" || bo3 == "eye" ? iw7_region_lit_sss : iw7_region_lit_opaque;
					}
					else if (bucket == t7_bucket_lit_trans || bucket == t7_bucket_lit_post_resolve)
					{
						result.cls = surface_class::trans;
						result.sort_key = iw7_sort_trans;
						// an additive forward decal (BO3's decal_emissive, polygon offset): IW7 draws it as its stock emissive
						// overlays, in the decal region at the key of its decal layer (dump_models)
						result.camera_region = bake::additive_decal(material) ? iw7_region_lit_decal : iw7_region_lit_trans;
					}
					else if (bucket == t7_bucket_emissive_fx)
					{
						result.cls = surface_class::trans;
						result.sort_key = iw7_sort_emissive_fx;
						result.camera_region = iw7_region_lit_trans;
					}
					else
					{
						ZONETOOL_FATAL("model material \"%s\": unhandled BO3 draw bucket %u (techset flags 0x%X)",
							material->name, bucket, flags);
					}

					// BO3's water on a model: IW7's UV-animated lit water (bake plan water), opaque as stock
					// water_lake_geneva_close_tsunami (sort key 2, camera region 0)
					const auto bo3 = bake::bo3_template(material);
					if (bo3.starts_with("lit_water_sim_flow") || bo3.starts_with("water_shore_flow") || bo3.starts_with("lit_transparent_water_flow"))
					{
						result.cls = surface_class::opaque;
						result.sort_key = iw7_sort_opaque;
						result.camera_region = iw7_region_lit_opaque;
					}

					const auto& plan = bake::get_plan(material, result);
					result.uv_period[0] = plan.uv_period[0];
					result.uv_period[1] = plan.uv_period[1];
					result.uv_origin[0] = plan.uv_origin[0];
					result.uv_origin[1] = plan.uv_origin[1];
					result.uv_span[0] = plan.uv_span[0];
					result.uv_span[1] = plan.uv_span[1];
					result.alpha_test = plan.alpha_test;
					// the model water's mco_ techset multiplies its albedo by the vertex rgb, which BO3's water does not read
					result.vertex_alpha = plan.supported && (plan.vertex_alpha || plan.multiply || plan.water);

					return result;
				}

				// materials baking at the same time, each worker with its own GPU device: one material leaves the CPU
				// or the GPU idle between its stages (texture reads, evaluation, packing, compression, writing)
				constexpr std::size_t bake_workers = 4;

				struct bake_result
				{
					unsigned int converted = 0;
					std::vector<std::string> failed;
				};

				bake_result bake_all(const std::vector<std::pair<const Material*, info>>& jobs, const char* kind)
				{
					std::vector<std::unique_ptr<bake::worker>> workers;
					for (auto i = 0u; i < std::min(bake_workers, jobs.size()); i++)
					{
						workers.push_back(std::make_unique<bake::worker>());
					}

					std::atomic<std::size_t> next{ 0 };
					std::atomic<unsigned int> converted{ 0 };
					std::mutex failed_mutex;
					std::vector<std::string> failed;
					std::vector<std::thread> threads;
					for (auto& w : workers)
					{
						threads.emplace_back([&, worker = w.get()]
						{
							for (;;)
							{
								const auto i = next++;
								if (i >= jobs.size())
								{
									break;
								}
								const auto& [material, inf] = jobs[i];
								try
								{
									if (bake::write_material(material, inf, *worker))
									{
										converted++;
										continue;
									}
								}
								catch (const std::exception& e)
								{
									ZONETOOL_ERROR("%s material %s: %s", kind, material->name, e.what());
								}
								std::lock_guard _(failed_mutex);
								failed.emplace_back(material->name);
							}
						});
					}
					for (auto& thread : threads)
					{
						thread.join();
					}

					std::sort(failed.begin(), failed.end());
					return { converted.load(), std::move(failed) };
				}
			}

			bool readable(const void* ptr, const std::size_t size)
			{
				return probe::readable(ptr, size);
			}

			void set_world_usage(const Material* material, const uv_bounds& uv)
			{
				world_usage[material] = uv;
			}

			void prepare(const GfxWorld* world, const std::vector<const Material*>& world_props)
			{
				infos.clear();
				world_usage.clear();
				world_atlas_order.clear();
				bake::clear();

				// the decal layers of the world's surfaces, and of the models the GfxWorld draws (static_model_clusters world
				// groups), which take the world's decal keys
				std::map<unsigned char, unsigned int> surfaces_per_layer;
				const auto count_layer = [&](const Material* material)
				{
					const auto flags = techset_flags(material);
					if ((flags & t7_techset_deferred) && (flags & t7_techset_decal))
					{
						surfaces_per_layer[material->info.layerSortDecal]++;
					}
				};
				for (auto s = 0; s < world->surfaceCount; s++)
				{
					count_layer(world->dpvs.surfaces[s].material);
				}
				for (const auto* material : world_props)
				{
					count_layer(material);
				}

				assign_decal_keys(surfaces_per_layer);
			}

			const info& get(const Material* material)
			{
				const auto found = infos.find(material);
				if (found != infos.end())
				{
					return found->second;
				}

				return infos.emplace(material, classify(material)).first->second;
			}

			const info& get_model(const Material* material)
			{
				const auto found = model_infos.find(material);
				if (found != model_infos.end())
				{
					return found->second;
				}

				return model_infos.emplace(material, classify_model(material)).first->second;
			}

			std::tuple<unsigned char, unsigned char, std::string, std::string> model_draw_order(const Material* material)
			{
				if (!material)
				{
					return {};
				}
				const auto found = model_infos.find(material);
				const auto inf = found != model_infos.end() ? found->second : classify_model(material);
				const auto* techset = readable(material->techniqueSet, sizeof(MaterialTechniqueSet)) && material->techniqueSet->name
					? material->techniqueSet->name : "";
				return { inf.camera_region, inf.sort_key, techset, inf.name };
			}

			bool model_blocks_sun(const Material* material)
			{
				if (!material || !(material->info.gameFlags & t7_game_flag_casts_shadow)
					|| !readable(material->techniqueSet, sizeof(MaterialTechniqueSet)))
				{
					return false;
				}
				const auto bucket = static_cast<unsigned char>(material->cameraRegion);
				if (bucket == t7_bucket_not_drawn || global_invisible(material))
				{
					return true; // shadow-only geometry
				}
				const auto flags = techset_flags(material);
				if ((flags & t7_techset_deferred) && (flags & t7_techset_decal))
				{
					return false;
				}
				if (!(flags & t7_techset_deferred) && bucket != t7_bucket_lit_opaque)
				{
					return false;
				}
				// (alpha tested ones too: the caller blocks the sun with them through their alpha, bake::alpha_mask_of)
				return true;
			}

			std::map<std::string, std::vector<std::string>> atlas_report(const std::vector<const Material*>& materials,
				const std::unordered_map<const Material*, surface_usage>& used)
			{
				std::vector<std::string> reasons(materials.size());
				parallel_for(static_cast<std::uint32_t>(materials.size()), [&](const std::uint32_t i, std::uint32_t)
				{
					const auto found = used.find(materials[i]);
					const auto inf = classify_model(materials[i], found != used.end() ? &found->second : nullptr);
					std::uint32_t w, h;
					auto alpha = false;
					std::string why;
					reasons[i] = bake::atlas_tile(materials[i], inf, w, h, alpha, &why) ? "atlased" : why;
				});
				std::map<std::string, std::vector<std::string>> out;
				for (auto i = 0u; i < materials.size(); i++)
				{
					out[reasons[i]].push_back(materials[i]->name);
				}
				return out;
			}

			bool world_drawable(const Material* material)
			{
				if (!material || !readable(material->techniqueSet, sizeof(MaterialTechniqueSet)))
				{
					return false;
				}
				if (material->techniqueSet->name)
				{
					const auto* entry = zonetool::t7::DB_FindXAssetEntry(ASSET_TYPE_TECHNIQUE_SET, material->techniqueSet->name, true);
					if (entry && entry->placeholder)
					{
						return false;
					}
				}
				// BO3's placeholder ($default) and effect materials (the emissive fx bucket, _fx templates) stay on models
				if (!material->name || std::strchr(material->name, '$'))
				{
					return false;
				}
				const auto bucket = static_cast<unsigned char>(material->cameraRegion);
				if (bucket != t7_bucket_not_drawn && bucket != t7_bucket_lit_opaque && bucket != t7_bucket_lit_trans
					&& bucket != t7_bucket_lit_post_resolve && !(techset_flags(material) & t7_techset_deferred))
				{
					return false;
				}
				const auto bo3 = bake::bo3_template(material);
				// (BO3's additive decals have no stock world techset in the transparent region: decal_emissive)
				return bo3 != "skin" && bo3 != "eye" && bo3.find("_fx") == std::string::npos && !bo3.starts_with("decal_emissive");
			}

			using atlas_key = std::tuple<std::uint32_t, std::uint32_t, bool, std::string, bool>;

			std::optional<atlas_key> atlas_key_of(const Material* material, const info& inf)
			{
				std::uint32_t tw, th;
				auto alpha = false;
				if (!bake::atlas_tile(material, inf, tw, th, alpha))
				{
					return {};
				}
				// (not by techset or shadow casting: with iw7-mod's static model lists, 16 times IW7's, the atlas count, images IW7 holds
				// at most 15616 of, matters more than an atlas being one material)
				return atlas_key{ tw, th, alpha, std::string(), false };
			}

			// the tiles an atlas side holds (up to 4096 texels, at most 16)
			std::uint32_t atlas_side(const std::uint32_t size)
			{
				auto n = 1u;
				while (n < 16 && size * n * 2 <= 4096)
				{
					n *= 2;
				}
				return n;
			}

			// atlases of `materials` (those bake::atlas_tile takes), named prefix_<n>: their texture areas in `table` move to their
			// tiles. Returns the members, each atlas's together (an atlas is written once its last tile is baked)
			std::vector<const Material*> plan_atlases(const std::vector<const Material*>& materials,
				std::unordered_map<const Material*, info>& table, const char* prefix, const char* kind)
			{
				// atlases (bake::atlas_tile), of the materials classified here: a model written before took its material's
				// texture coordinates already. Same-size tiles, up to 4096 texels an atlas side (2 x 2 tiles of 2048: a 22 MB streamed part), a power of
				// two up to 16 tiles a side (IW7 loads at most 15616 images: a large map's small props need the larger grids)
				// alpha-tested materials in atlases of their own (with a coverage layer)
				std::map<atlas_key, std::vector<const Material*>> by_tile;
				for (auto i = 0u; i < materials.size(); i++)
				{
					if (const auto key = atlas_key_of(materials[i], table.at(materials[i])))
					{
						by_tile[*key].push_back(materials[i]);
					}
				}
				std::vector<const Material*> atlas_order;
				auto atlas_count = 0u;
				for (const auto& [tile, members] : by_tile)
				{
					const auto side = atlas_side;
					const auto& [tile_width, tile_height, tile_alpha, tile_techset, tile_casts] = tile;
					const auto max_columns = side(tile_width);
					const auto max_rows = side(tile_height);
					for (std::size_t first = 0; first < members.size(); first += static_cast<std::size_t>(max_columns) * max_rows)
					{
						const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(members.size() - first,
							static_cast<std::size_t>(max_columns) * max_rows));
						if (count < 2)
						{
							break;
						}
						auto columns = max_columns;
						auto rows = max_rows;
						while (rows > 1 && columns * rows / 2 >= count)
						{
							rows /= 2;
						}
						while (columns > 1 && columns * rows / 2 >= count)
						{
							columns /= 2;
						}
						bake::atlas_slot slot{};
						slot.cs = utils::string::va("%s_%u_packed_cs", prefix, atlas_count);
						slot.ng = utils::string::va("%s_%u_packed_ng", prefix, atlas_count);
						if (tile_alpha)
						{
							slot.a = utils::string::va("%s_%u_packed_a", prefix, atlas_count);
						}
						slot.columns = columns;
						slot.rows = rows;
						slot.tile_width = tile_width;
						slot.tile_height = tile_height;
						atlas_count++;
						for (auto k = 0u; k < count; k++)
						{
							const auto* material = members[first + k];
							slot.column = k % columns;
							slot.row = k / columns;
							auto& inf = table.at(material);
							// tile texture coordinates t = (uv - origin) / span in [0, 1] -> (place + t) / n
							const std::uint32_t place[2] = { slot.column, slot.row };
							const std::uint32_t n[2] = { columns, rows };
							for (auto axis = 0; axis < 2; axis++)
							{
								const auto span = inf.uv_span[axis];
								auto origin = inf.uv_origin[axis] - static_cast<float>(place[axis]) * span;
								const auto atlas_span = static_cast<float>(n[axis]) * span;
								if (2 * place[axis] >= n[axis] && n[axis] > 1)
								{
									origin += atlas_span;
								}
								inf.uv_origin[axis] = origin;
								inf.uv_span[axis] = atlas_span;
							}
							bake::set_atlas(material, inf, slot);
							atlas_order.push_back(material);
						}
					}
				}
				std::string tile_sizes;
				for (const auto& [tile, members] : by_tile)
				{
					tile_sizes += utils::string::va(" %ux%u%s:%zu", std::get<0>(tile), std::get<1>(tile), std::get<2>(tile) ? "a" : "", members.size());
				}
				ZONETOOL_INFO("%s: %zu share %u atlases (%zu images fewer); tiles%s", kind, atlas_order.size(), atlas_count,
					(atlas_order.size() - atlas_count) * 2, tile_sizes.data());
				return atlas_order;

			}

			std::unordered_map<const Material*, unsigned int> atlas_groups(const std::vector<const Material*>& materials,
				const std::unordered_map<const Material*, surface_usage>& used)
			{
				std::vector<std::optional<atlas_key>> keys(materials.size());
				parallel_for(static_cast<std::uint32_t>(materials.size()), [&](const std::uint32_t i, std::uint32_t)
				{
					const auto found = used.find(materials[i]);
					keys[i] = atlas_key_of(materials[i], classify_model(materials[i], found != used.end() ? &found->second : nullptr));
				});
				std::map<atlas_key, std::vector<const Material*>> by_tile;
				for (auto i = 0u; i < materials.size(); i++)
				{
					if (keys[i])
					{
						by_tile[*keys[i]].push_back(materials[i]);
					}
				}
				std::unordered_map<const Material*, unsigned int> out;
				auto group = 0u;
				for (const auto& [key, members] : by_tile)
				{
					const auto per = static_cast<std::size_t>(atlas_side(std::get<0>(key))) * atlas_side(std::get<1>(key));
					for (std::size_t first = 0; first < members.size(); first += per)
					{
						const auto end = std::min(members.size(), first + per);
						if (end - first < 2)
						{
							break;
						}
						for (auto k = first; k < end; k++)
						{
							out[members[k]] = group;
						}
						group++;
					}
				}
				return out;
			}

			void plan_world_atlases(const std::vector<const Material*>& materials)
			{
				for (const auto* material : materials)
				{
					get(material);
				}
				world_atlas_order = plan_atlases(materials, infos, "t7_watlas", "world materials of placed models");
			}

			void dump_all(const GfxWorld* world)
			{
				// every material the world draws with, in first-use order
				std::vector<const Material*> materials;
				std::unordered_set<const Material*> seen;
				const auto add = [&](const Material* material)
				{
					if (material && seen.insert(material).second)
					{
						materials.push_back(material);
					}
				};
				for (auto s = 0; s < world->surfaceCount; s++)
				{
					add(world->dpvs.surfaces[s].material);
				}
				for (auto i = 0; i < world->materialMemoryCount; i++)
				{
					add(world->materialMemory[i].material);
				}
				// and the models' materials the GfxWorld draws on world surfaces (static_model_clusters world groups), by name
				{
					std::vector<const Material*> props;
					for (const auto& [material, inf] : infos)
					{
						if (!seen.contains(material))
						{
							props.push_back(material);
						}
					}
					std::ranges::sort(props, [](const Material* a, const Material* b)
					{
						return std::strcmp(a->name, b->name) < 0;
					});
					for (const auto* material : props)
					{
						add(material);
					}
				}

				bake::set_underlying_gloss(bake::estimate_underlying_gloss(materials));

				// classified here, one after another: the workers only read the table; the atlas members last, each atlas's
				// together (plan_world_atlases)
				std::vector<std::pair<const Material*, info>> jobs;
				const std::unordered_set<const Material*> in_atlas(world_atlas_order.begin(), world_atlas_order.end());
				for (const auto* material : materials)
				{
					if (!in_atlas.contains(material))
					{
						jobs.emplace_back(material, get(material));
					}
				}
				for (const auto* material : world_atlas_order)
				{
					jobs.emplace_back(material, get(material));
				}

				const auto start = std::chrono::steady_clock::now();
				const auto result = bake_all(jobs, "world");
				bake::flush_atlases();
				const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
				ZONETOOL_INFO("world materials: converted %u of %zu (%.1f MB of images, %.0f s; %s)", result.converted, materials.size(),
					static_cast<double>(bake::image_bytes()) / (1024.0 * 1024.0), seconds, bake::stage_report().data());
				if (!result.failed.empty())
				{
					ZONETOOL_WARNING("world materials: %zu not converted", result.failed.size());
				}
			}

			void dump_models(const std::vector<const Material*>& materials, const std::unordered_map<const Material*, surface_usage>& used)
			{
				// classified on every core first (planning runs each material's shader over a quad); the workers only
				// read the table
				std::vector<info> classified(materials.size());
				std::vector<char> fresh(materials.size());
				for (auto i = 0u; i < materials.size(); i++)
				{
					fresh[i] = !model_infos.contains(materials[i]);
				}
				parallel_for(static_cast<std::uint32_t>(materials.size()), [&](const std::uint32_t i, std::uint32_t)
				{
					if (fresh[i])
					{
						const auto found = used.find(materials[i]);
						classified[i] = classify_model(materials[i], found != used.end() ? &found->second : nullptr);
					}
				});
				for (auto i = 0u; i < materials.size(); i++)
				{
					model_infos.emplace(materials[i], std::move(classified[i]));
				}

				std::vector<const Material*> candidates;
				for (auto i = 0u; i < materials.size(); i++)
				{
					if (fresh[i])
					{
						candidates.push_back(materials[i]);
					}
				}
				const auto atlas_order = plan_atlases(candidates, model_infos, "t7_atlas", "model materials");

				// the atlas members last, each atlas's together (an atlas is written once its last tile is baked)
				std::unordered_set<const Material*> in_atlas(atlas_order.begin(), atlas_order.end());
				std::vector<const Material*> order;
				for (const auto* material : materials)
				{
					if (!in_atlas.contains(material))
					{
						order.push_back(material);
					}
				}
				order.insert(order.end(), atlas_order.begin(), atlas_order.end());

				std::vector<std::pair<const Material*, info>> jobs;
				auto windowed = 0u;
				for (const auto* material : order)
				{
					auto inf = get_model(material);
					if (inf.cls == surface_class::decal || inf.camera_region == iw7_region_lit_decal)
					{
						inf.sort_key = model_decal_key(material->info.layerSortDecal);
					}
					if (bake::get_plan(material, inf).windowed)
					{
						windowed++;
					}
					jobs.emplace_back(material, std::move(inf));
				}
				ZONETOOL_INFO("model materials: %u of %zu bake only the texture area their surfaces use (less than a period)", windowed,
					materials.size());

				const auto start = std::chrono::steady_clock::now();
				const auto bytes_before = bake::image_bytes();
				const auto result = bake_all(jobs, "model");
				bake::flush_atlases();
				bake::share_atlas_materials();
				const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
				ZONETOOL_INFO("model materials: converted %u of %zu (%.1f MB of images, %.0f s; %s)", result.converted, materials.size(),
					static_cast<double>(bake::image_bytes() - bytes_before) / (1024.0 * 1024.0), seconds, bake::stage_report().data());
				if (!result.failed.empty())
				{
					ZONETOOL_WARNING("model materials: %zu not converted", result.failed.size());
				}
			}
		}
	}
}
