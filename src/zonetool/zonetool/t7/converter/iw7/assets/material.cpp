#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "material.hpp"
#include "material_template.hpp"

#include "zonetool/t7/functions.hpp"

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace material
		{
			namespace
			{
				enum map_type
				{
					MAP_SPECULAR,
					MAP_NORMAL,
					MAP_COLOR,
					MAP_OCCLUSION,
					MAP_COUNT,
				};

				struct texture_slot
				{
					unsigned int type_hash;
					unsigned char first_character;
					unsigned char last_character;
					unsigned char semantic;
					const char* placeholder;
					const char* suffixes[4];
				};

				const texture_slot texture_slots[MAP_COUNT] =
				{
					{ 887934131,  's', 'p', 8, "$black", { "s", "sg", "g", nullptr } },
					{ 1507003663, 'n', 'p', 5, "$identitynormalmap", { "n", "nml", nullptr } },
					{ 2695565377, 'c', 'p', 2, "$white", { "c", "col", nullptr } },
					{ 2771134132, 's', 'p', 9, "$white", { "o", "ao", nullptr } },
				};

				bool is_readable(const void* ptr)
				{
					MEMORY_BASIC_INFORMATION mbi{};
					if (!VirtualQuery(ptr, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT)
					{
						return false;
					}

					constexpr auto readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
						PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;

					return (mbi.Protect & readable) != 0 && (mbi.Protect & PAGE_GUARD) == 0;
				}

				const char* safe_name(const char* name)
				{
					if (!name)
					{
						return nullptr;
					}

					MEMORY_BASIC_INFORMATION mbi{};
					if (!VirtualQuery(name, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT)
					{
						return nullptr;
					}

					constexpr auto readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
						PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;

					if ((mbi.Protect & readable) == 0 || (mbi.Protect & PAGE_GUARD) != 0)
					{
						return nullptr;
					}

					const auto* end = static_cast<const char*>(mbi.BaseAddress) + mbi.RegionSize;
					for (const auto* it = name; it < end && (it - name) < 256; it++)
					{
						const auto c = static_cast<unsigned char>(*it);
						if (!c)
						{
							return it > name ? name : nullptr;
						}

						if (c < 0x20 || c > 0x7E)
						{
							return nullptr;
						}
					}

					return nullptr;
				}

				std::string get_suffix(const std::string& name)
				{
					const auto pos = name.find_last_of('_');
					if (pos == std::string::npos)
					{
						return {};
					}

					return name.substr(pos + 1);
				}

				int classify_suffix(const std::string& suffix)
				{
					for (auto i = 0; i < MAP_COUNT; i++)
					{
						for (auto s = 0; texture_slots[i].suffixes[s]; s++)
						{
							if (suffix == texture_slots[i].suffixes[s])
							{
								return i;
							}
						}
					}

					return -1;
				}

				int classify_hash(unsigned int name_hash)
				{
					switch (name_hash)
					{
					case 0xA0AB1041: return MAP_COLOR;
					case 0x59D30D0F: return MAP_NORMAL;
					case 0xEC443804: return MAP_SPECULAR;
					}

					return -1;
				}

				const char* const unused_suffixes[] = { "e", "r", "t", "ir", "mask", "a" };

				bool is_map_suffix(const std::string& suffix)
				{
					if (classify_suffix(suffix) >= 0)
					{
						return true;
					}

					for (const auto* unused : unused_suffixes)
					{
						if (suffix == unused)
						{
							return true;
						}
					}

					return false;
				}

				std::string get_base_name(const std::string& name)
				{
					const auto suffix = get_suffix(name);
					if (suffix.empty() || !is_map_suffix(suffix))
					{
						return {};
					}

					return name.substr(0, name.size() - suffix.size() - 1);
				}

				bool image_exists(const std::string& name)
				{
					if (zonetool::t7::DB_FindXAssetEntry(ASSET_TYPE_IMAGE, name.data(), false))
					{
						return true;
					}

					return std::filesystem::exists(filesystem::get_dump_path() + "images\\" + name + ".dds");
				}

				bool is_effect_material(const std::string& name, const Material* asset)
				{
					if (name.starts_with("ei/") || name.starts_with("ec/") ||
						name.starts_with("el/") || name.starts_with("vd/") ||
						name.starts_with("vdd/") || name.starts_with("gfx_"))
					{
						return true;
					}

					if (asset->techniqueSet && is_readable(asset->techniqueSet))
					{
						const auto* techset_name = safe_name(asset->techniqueSet->name);
						if (!techset_name)
						{
							return false;
						}

						const std::string techset = techset_name;
						return techset.find("effect") != std::string::npos ||
							techset.find("unlit") != std::string::npos;
					}

					return false;
				}

				bool is_emissive_material(const Material* asset)
				{
					if (asset->techniqueSet && is_readable(asset->techniqueSet))
					{
						const auto* techset_name = safe_name(asset->techniqueSet->name);
						if (techset_name && std::string(techset_name).find("emissive") != std::string::npos)
						{
							return true;
						}
					}

					return false;
				}

				std::string find_emissive_map(const Material* asset)
				{
					for (auto i = 0; asset->textureTable && i < asset->textureCount; i++)
					{
						auto* image = asset->textureTable[i].image;
						if (!image || !is_readable(image))
						{
							continue;
						}

						const auto* image_name = safe_name(image->name);
						if (!image_name || *image_name == ',')
						{
							continue;
						}

						if (get_suffix(image_name) == "e")
						{
							return image_name;
						}
					}

					return {};
				}

				std::string get_material_name(const std::string& name)
				{
					auto clean = name;

					const auto pos = clean.find_last_of("/\\");
					if (pos != std::string::npos)
					{
						clean = clean.substr(pos + 1);
					}

					for (auto i = 0u; i < clean.size(); i++)
					{
						switch (clean[i])
						{
						case '*':
						case '|':
						case ':':
						case '?':
						case '<':
						case '>':
						case '\"':
							clean[i] = '_';
							break;
						}
					}

					return clean;
				}

				bool write_file(const std::string& path, const void* data, std::size_t size)
				{
					filesystem::file file(path);
					file.open("wb");

					if (!file.get_fp())
					{
						return false;
					}

					file.write(data, size, 1);
					file.close();

					return true;
				}

				bool write_techset_files(const std::string& techset, const std::string& name)
				{
					const material_template::techset_template* tmpl = nullptr;
					for (const auto& entry : material_template::techset_templates)
					{
						if (techset == entry.techset)
						{
							tmpl = &entry;
							break;
						}
					}

					if (!tmpl)
					{
						return false;
					}

					const auto state = "techsets\\state\\"s + techset + "\\"s + name;
					const auto constants = "techsets\\constantbuffer\\"s + techset + "\\"s + name;

					return write_file(state + ".stateinfo", tmpl->stateinfo.data, tmpl->stateinfo.size)
						&& write_file(state + ".statebits", tmpl->statebits.data, tmpl->statebits.size)
						&& write_file(state + ".statebitsmap", tmpl->statebitsmap.data, tmpl->statebitsmap.size)
						&& write_file(constants + ".cbi", tmpl->constant_buffer_indexes.data,
							tmpl->constant_buffer_indexes.size)
						&& write_file(constants + ".cbt", tmpl->constant_buffer_table.data,
							tmpl->constant_buffer_table.size);
				}

				void collect_maps(const Material* asset, std::string* maps)
				{
					std::string first_image;

					for (auto i = 0; asset->textureTable && i < asset->textureCount; i++)
					{
						auto* image = asset->textureTable[i].image;
						if (!image || !is_readable(image))
						{
							continue;
						}

						const auto* image_name = safe_name(image->name);
						if (!image_name || *image_name == ',')
						{
							continue;
						}

						if (first_image.empty())
						{
							first_image = image_name;
						}

						auto type = classify_hash(asset->textureTable[i].nameHash);
						if (type < 0)
						{
							type = classify_suffix(get_suffix(image_name));
						}

						if (type >= 0 && maps[type].empty())
						{
							maps[type] = image_name;
						}
					}

					for (auto i = 0; i < MAP_COUNT; i++)
					{
						for (auto j = i + 1; j < MAP_COUNT; j++)
						{
							if (!maps[i].empty() && maps[i] == maps[j])
							{
								maps[j].clear();
							}
						}
					}

					std::string base;
					for (auto i = 0; i < MAP_COUNT && base.empty(); i++)
					{
						if (!maps[i].empty())
						{
							base = get_base_name(maps[i]);
						}
					}

					if (base.empty())
					{
						if (!first_image.empty())
						{
							maps[MAP_COLOR] = first_image;
						}
						return;
					}

					std::string fallback;
					for (auto i = 0; i < MAP_COUNT && fallback.empty(); i++)
					{
						fallback = maps[i];
					}

					for (auto i = 0; i < MAP_COUNT; i++)
					{
						if (!maps[i].empty() && classify_suffix(get_suffix(maps[i])) != i)
						{
							maps[i].clear();
						}
					}

					for (auto i = 0; i < MAP_COUNT; i++)
					{
						for (auto s = 0; texture_slots[i].suffixes[s] && maps[i].empty(); s++)
						{
							const auto candidate = base + "_" + texture_slots[i].suffixes[s];
							if (image_exists(candidate))
							{
								maps[i] = candidate;
							}
						}
					}

					if (maps[MAP_COLOR].empty())
					{
						maps[MAP_COLOR] = fallback;
					}
				}
			}

			std::string get_converted_name(Material* asset)
			{
				const auto* name = asset ? safe_name(asset->name) : nullptr;
				if (!name)
				{
					return {};
				}

				if (is_effect_material(name, asset))
				{
					return "el/"s + get_material_name(name);
				}

				if (is_emissive_material(asset))
				{
					return "mo/"s + get_material_name(name);
				}

				return "mo/"s + get_material_name(name);
			}

			void dump(Material* asset)
			{
				if (!asset->name)
				{
					return;
				}

				const auto effect = is_effect_material(asset->name, asset);

				if (!effect && is_emissive_material(asset))
				{
					const std::string techset = "mo_effectunlit_replace_lin_ct_nocast_mkhdr";
					const auto name = "mo/"s + get_material_name(asset->name);

					std::string maps[MAP_COUNT];
					collect_maps(asset, maps);
					auto emissive_map = maps[MAP_COLOR];
					if (emissive_map.empty())
					{
						emissive_map = find_emissive_map(asset);
					}

					ordered_json matdata;
					matdata["techniqueSet->name"] = techset;
					matdata["gameFlags"] = 0;
					matdata["unkFlags"] = 0;
					matdata["sortKey"] = 35;
					matdata["renderFlags"] = 1;
					matdata["textureAtlasRowCount"] = 1;
					matdata["textureAtlasColumnCount"] = 1;
					matdata["textureAtlasFrameBlend"] = 0;
					matdata["textureAtlasAsArray"] = 0;
					matdata["surfaceTypeBits"] = 0;
					matdata["stateFlags"] = 16;
					matdata["cameraRegion"] = 4;
					matdata["materialType"] = 23;
					matdata["assetFlags"] = 0;

					ordered_json constants;
					{
						ordered_json c;
						c["name"] = "textureAtlas";
						c["nameHash"] = 1128936273;
						c["literal"] = { 1.0f, 1.0f, 1.0f, 1.0f };
						constants.push_back(c);
					}
					{
						ordered_json c;
						c["name"] = "featherParms";
						c["nameHash"] = 1300144692;
						c["literal"] = { 0.06666667f, 15.0f, 15.0f, 0.0f };
						constants.push_back(c);
					}
					{
						ordered_json c;
						c["name"] = "colorTint";
						c["nameHash"] = 3054254906;
						c["literal"] = { 1.4f, 1.4f, 1.4f, 1.0f };
						constants.push_back(c);
					}
					matdata["constantTable"] = constants;

					const auto& color_slot = texture_slots[MAP_COLOR];
					ordered_json texture;
					texture["image"] = emissive_map.empty() ? "$black" : emissive_map;
					texture["semantic"] = color_slot.semantic;
					texture["samplerState"] = 19;
					texture["lastCharacter"] = color_slot.last_character;
					texture["firstCharacter"] = color_slot.first_character;
					texture["typeHash"] = color_slot.type_hash;
					matdata["textureTable"] = ordered_json::array({ texture });

					filesystem::file file("materials\\"s + name + ".json"s);
					file.open("wb");
					if (!file.get_fp())
					{
						ZONETOOL_ERROR("could not open material file for \"%s\"", asset->name);
						return;
					}
					file.write(matdata.dump(4));
					file.close();

					if (!write_techset_files(techset, name))
					{
						ZONETOOL_ERROR("could not write techset files for emissive material \"%s\"", asset->name);
					}
					return;
				}

				std::string maps[MAP_COUNT];
				collect_maps(asset, maps);

				bool emit[MAP_COUNT]{};
				const char* techset_name = nullptr;

				if (effect)
				{
					techset_name = "el_effect_add_nofog_ndw_nocast";
					emit[MAP_COLOR] = true;
				}
				else
				{
					emit[MAP_NORMAL] = true;
					emit[MAP_COLOR] = true;

					if (!maps[MAP_SPECULAR].empty() && !maps[MAP_NORMAL].empty() &&
						!maps[MAP_OCCLUSION].empty())
					{
						techset_name = "mo_l_sm_replace_i0c0s0o0n0";
						emit[MAP_SPECULAR] = true;
						emit[MAP_OCCLUSION] = true;
					}
					else if (!maps[MAP_SPECULAR].empty() && !maps[MAP_NORMAL].empty())
					{
						techset_name = "mo_l_sm_replace_i0c0s0o0n0";
						emit[MAP_SPECULAR] = true;
						emit[MAP_OCCLUSION] = true;
					}
					else if (!maps[MAP_SPECULAR].empty())
					{
						techset_name = "mo_l_sm_replace_i0c0s0";
						emit[MAP_SPECULAR] = true;
					}
					else
					{
						techset_name = "mo_l_sm_replace_i0c0";
					}
				}

				const std::string techset = techset_name;
				const auto name = (effect ? "el/"s : "mo/"s) + get_material_name(asset->name);

				ordered_json matdata;

				matdata["techniqueSet->name"] = techset;
				matdata["gameFlags"] = effect ? 0 : 144;
				matdata["unkFlags"] = 0;
				matdata["sortKey"] = effect ? 35 : 2;
				matdata["renderFlags"] = 0;
				matdata["textureAtlasRowCount"] = 1;
				matdata["textureAtlasColumnCount"] = 1;
				matdata["textureAtlasFrameBlend"] = 0;
				matdata["textureAtlasAsArray"] = 0;
				matdata["surfaceTypeBits"] = 0;
				matdata["stateFlags"] = effect ? 16 : 57;
				matdata["cameraRegion"] = effect ? 4 : 0;
				matdata["materialType"] = effect ? 71 : 23;
				matdata["assetFlags"] = 0;

				ordered_json constants;
				if (effect)
				{
					ordered_json texture_atlas;
					texture_atlas["name"] = "textureAtlas";
					texture_atlas["nameHash"] = 1128936273;
					texture_atlas["literal"] = { 1.0f, 1.0f, 1.0f, 1.0f };
					constants.push_back(texture_atlas);
				}

				ordered_json color_tint;
				color_tint["name"] = "colorTint";
				color_tint["nameHash"] = 3054254906;
				color_tint["literal"] = { 1.0f, 1.0f, 1.0f, 1.0f };
				constants.push_back(color_tint);

				if (!effect)
				{
					ordered_json reflection;
					reflection["name"] = "reflectionRa";
					reflection["nameHash"] = 3344177073;
					reflection["literal"] = { 8096.0f, 0.0f, 0.0f, 0.0f };
					constants.push_back(reflection);
				}
				matdata["constantTable"] = constants;

				ordered_json textures;
				for (auto i = 0; i < MAP_COUNT; i++)
				{
					if (!emit[i])
					{
						continue;
					}

					const auto& slot = texture_slots[i];

					ordered_json texture;
					texture["image"] = maps[i].empty() ? slot.placeholder : maps[i];
					texture["semantic"] = slot.semantic;
					texture["samplerState"] = effect ? 20 : 19;
					texture["lastCharacter"] = slot.last_character;
					texture["firstCharacter"] = slot.first_character;
					texture["typeHash"] = slot.type_hash;

					textures.push_back(texture);
				}
				matdata["textureTable"] = textures;

				filesystem::file file("materials\\"s + name + ".json"s);
				file.open("wb");

				if (!file.get_fp())
				{
					ZONETOOL_ERROR("could not open material file for \"%s\"", asset->name);
					return;
				}

				file.write(matdata.dump(4));
				file.close();

				if (!write_techset_files(techset, name))
				{
					ZONETOOL_ERROR("could not write techset files for material \"%s\"", asset->name);
				}
			}
		}
	}
}
