#include <std_include.hpp>
#include "zonetool.hpp"

#include <utils/io.hpp>
#include <utils/flags.hpp>

#include "converter/converter.hpp"

#include "common/xpak.hpp"

namespace zonetool::t7
{
	struct dump_params
	{
		game::game_mode target;
		std::string zone;
		bool valid;
		std::unordered_set<XAssetType> filter;
	};

	zonetool_globals_t globals{};
	std::vector<std::pair<XAssetType, std::string>> referenced_assets;
	std::unordered_set<XAssetType> asset_type_filter;
	std::recursive_mutex dump_lock;

	// last line of dump.log names the asset being dumped when we died
	void dump_trace(const char* type, const char* name)
	{
		static FILE* trace_file = []() -> FILE*
		{
			FILE* fp = nullptr;
			fopen_s(&fp, "dump.log", "w");
			return fp;
		}();

		if (!trace_file)
		{
			return;
		}

		fprintf(trace_file, "%s %s", type, name);
		fputc(10, trace_file);
		fflush(trace_file);
	}

	std::unordered_set<std::pair<std::uint32_t, std::string>, pair_hash<std::uint32_t, std::string>> ignore_assets;

	// Validate names read directly from zone memory.
	const char* safe_asset_name(const char* name)
	{
		if (!name)
		{
			return "";
		}

		MEMORY_BASIC_INFORMATION mbi{};
		if (!VirtualQuery(name, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT)
		{
			return "";
		}

		constexpr auto readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
			PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;

		if ((mbi.Protect & readable) == 0 || (mbi.Protect & PAGE_GUARD) != 0)
		{
			return "";
		}

		const auto* end = static_cast<const char*>(mbi.BaseAddress) + mbi.RegionSize;
		for (const auto* it = name; it < end && (it - name) < 256; it++)
		{
			const auto c = static_cast<unsigned char>(*it);
			if (!c)
			{
				return it > name ? name : "";
			}

			if (c < 0x20 || c > 0x7E)
			{
				return "";
			}
		}

		return "";
	}

	const char* get_asset_name(XAssetType type, void* pointer)
	{
		if (!pointer)
		{
			return "";
		}

		if (type == ASSET_TYPE_IMAGE)
		{
			return safe_asset_name(reinterpret_cast<const GfxImage*>(pointer)->name);
		}

		return safe_asset_name(*reinterpret_cast<const char* const*>(pointer));
	}

	const char* get_asset_name(XAsset* asset)
	{
		return get_asset_name(asset->type, asset->header.data);
	}

	const char* type_to_string(XAssetType type)
	{
		return g_assetNames[type]; sizeof(GfxImage);
	}

	std::int32_t type_to_int(std::string type)
	{
		for (std::int32_t i = 0; i < ASSET_TYPE_COUNT; i++)
		{
			if (g_assetNames[i] == type)
				return i;
		}

		return -1;
	}

	bool is_valid_asset_type(const std::string& type)
	{
		return type_to_int(type) >= 0;
	}

	bool zone_exists(const std::string& zone)
	{
		return DB_FileExists(zone.data(), 0);
	}

	bool is_zone_loaded(const std::string& name)
	{
		int index = *g_zoneCount;
		auto* zone = &g_zones[index];
		auto zone_name = &g_zoneNames[index];

		while (index > 0)
		{
			if (!_stricmp(zone_name->name, name.data()))
			{
				return true;
			}

			index--;
			zone--;
			zone_name--;
		}

		return false;
	}

	bool is_referenced_asset(XAsset* asset)
	{
		if (get_asset_name(asset)[0] == ',')
		{
			return true;
		}
		return false;
	}

	void wait_for_database()
	{
		// wait for database to be ready
		while (!utils::hook::invoke<bool>(0x1405261E0))
		{
			Sleep(5);
		}
	}

	XAssetHeader db_find_x_asset_header(XAssetType type, const char* name)
	{
		return DB_FindXAssetHeader(type, name, 0, -1);
	}

	XAssetHeader db_find_x_asset_header_safe(XAssetType type, const std::string& name)
	{
		return db_find_x_asset_header(type, name.data());
	}

	void dump_asset_h1(XAsset* asset)
	{
#define DUMP_ASSET_REGULAR(__type__,__namespace__,__struct__) \
		if (asset->type == __type__) \
		{ \
			if(IS_DEBUG) ZONETOOL_INFO("Dumping asset \"%s\" of type %s.", get_asset_name(asset), type_to_string(asset->type)); \
			auto asset_ptr = reinterpret_cast<__struct__*>(asset->header.data); \
			__namespace__::dump(asset_ptr); \
		}

#define DUMP_ASSET_NO_CONVERT(__type__,__namespace__,__struct__) \
		if (asset->type == __type__) \
		{ \
			if(IS_DEBUG) ZONETOOL_INFO("Dumping asset \"%s\" of type %s.", get_asset_name(asset), type_to_string(asset->type)); \
			auto asset_ptr = reinterpret_cast<zonetool::h1::__struct__*>(asset->header.data); \
			zonetool::h1::__namespace__::dump(asset_ptr); \
		}

#define DUMP_ASSET_CONVERT(__type__,__namespace__,__struct__) \
		if (asset->type == __type__) \
		{ \
			if(IS_DEBUG) ZONETOOL_INFO("Converting and dumping asset \"%s\" of type %s.", get_asset_name(asset), type_to_string(asset->type)); \
			auto asset_ptr = reinterpret_cast<__struct__*>(asset->header.data); \
			converter::h1::__namespace__::dump(asset_ptr); \
		}

		try
		{
			DUMP_ASSET_CONVERT(ASSET_TYPE_XANIMPARTS, xanim, XAnimParts);
			DUMP_ASSET_CONVERT(ASSET_TYPE_XMODEL, xmodel, XModel);
			DUMP_ASSET_CONVERT(ASSET_TYPE_XMODELMESH, xmodel_mesh, XModelMesh);
		}
		catch (const std::exception& e)
		{
			ZONETOOL_FATAL("A fatal exception occured while dumping zone \"%s\", exception was: \n%s",
				filesystem::get_fastfile().data(), e.what());
		}

#undef DUMP_ASSET_CONVERT
#undef DUMP_ASSET_NO_CONVERT
#undef DUMP_ASSET_REGULAR
	}

	void dump_asset_iw7(XAsset* asset)
	{
#define DUMP_ASSET_REGULAR(__type__,__namespace__,__struct__) \
		if (asset->type == __type__) \
		{ \
			if(IS_DEBUG) ZONETOOL_INFO("Dumping asset \"%s\" of type %s.", get_asset_name(asset), type_to_string(asset->type)); \
			auto asset_ptr = reinterpret_cast<__struct__*>(asset->header.data); \
			__namespace__::dump(asset_ptr); \
		}

#define DUMP_ASSET_NO_CONVERT(__type__,__namespace__,__struct__) \
		if (asset->type == __type__) \
		{ \
			if(IS_DEBUG) ZONETOOL_INFO("Dumping asset \"%s\" of type %s.", get_asset_name(asset), type_to_string(asset->type)); \
			auto asset_ptr = reinterpret_cast<zonetool::iw7::__struct__*>(asset->header.data); \
			zonetool::iw7::__namespace__::dump(asset_ptr); \
		}

#define DUMP_ASSET_CONVERT(__type__,__namespace__,__struct__) \
		if (asset->type == __type__) \
		{ \
			if(IS_DEBUG) ZONETOOL_INFO("Converting and dumping asset \"%s\" of type %s.", get_asset_name(asset), type_to_string(asset->type)); \
			auto asset_ptr = reinterpret_cast<__struct__*>(asset->header.data); \
			converter::iw7::__namespace__::dump(asset_ptr); \
		}

		try
		{
			DUMP_ASSET_CONVERT(ASSET_TYPE_XANIMPARTS, xanim, XAnimParts);
			DUMP_ASSET_CONVERT(ASSET_TYPE_XMODEL, xmodel, XModel);
			DUMP_ASSET_CONVERT(ASSET_TYPE_XMODELMESH, xmodel_mesh, XModelMesh);
			DUMP_ASSET_CONVERT(ASSET_TYPE_IMAGE, gfximage, GfxImage);
			DUMP_ASSET_CONVERT(ASSET_TYPE_MATERIAL, material, Material);
			DUMP_ASSET_CONVERT(ASSET_TYPE_FX, fxeffectdef, FxEffectDef);
			// also emitted as vfx: iw7 weapons only consume FX_COMBINED_VFX, so the
			// legacy .fxe cannot be referenced by one
			DUMP_ASSET_CONVERT(ASSET_TYPE_FX, particlesystem, FxEffectDef);
			if (asset->type == ASSET_TYPE_WEAPON)
			{
				converter::iw7::weapon::dump(asset->header.data, get_asset_name(asset));
			}
		}
		catch (const std::exception& e)
		{
			ZONETOOL_FATAL("A fatal exception occured while dumping zone \"%s\", exception was: \n%s",
				filesystem::get_fastfile().data(), e.what());
		}

#undef DUMP_ASSET_CONVERT
#undef DUMP_ASSET_NO_CONVERT
#undef DUMP_ASSET_REGULAR
	}

	void dump_asset_t7(XAsset* asset)
	{
#define DUMP_ASSET(__type__,___,__struct__) \
		if (asset->type == __type__) \
		{ \
			if(IS_DEBUG) ZONETOOL_INFO("Dumping asset \"%s\" of type %s.", get_asset_name(asset), type_to_string(asset->type)); \
			auto asset_ptr = reinterpret_cast<__struct__*>(asset->header.data); \
			___::dump(asset_ptr); \
		}

		try
		{
			// dump assets
			//DUMP_ASSET(ASSET_TYPE_XMODEL, xmodel, XModel);
		}
		catch (const std::exception& e)
		{
			ZONETOOL_FATAL("A fatal exception occured while dumping zone \"%s\", exception was: \n%s",
				filesystem::get_fastfile().data(), e.what());
		}

#undef DUMP_ASSET
	}

	std::unordered_map<game::game_mode, std::function<void(XAsset*)>> dump_functions =
	{
		{game::t7, dump_asset_t7},
		{game::h1, dump_asset_h1},
		{game::iw7, dump_asset_iw7},
	};

	void dump_asset(XAsset* asset)
	{
		std::lock_guard<std::recursive_mutex> lock(dump_lock);

		dump_trace(type_to_string(asset->type), get_asset_name(asset));

		if (globals.verify)
		{
			ZONETOOL_INFO("Loading asset \"%s\" of type %s.", get_asset_name(asset), type_to_string(asset->type));
		}

		if (globals.dump_csv)
		{
			if (globals.csv_file.get_fp() == nullptr)
			{
				globals.csv_file = filesystem::file(filesystem::get_fastfile() + ".csv");
				globals.csv_file.open("wb");
			}

			// dump assets to disk
			if (globals.csv_file.get_fp())
			{
				std::fprintf(globals.csv_file.get_fp(), "%s,%s\n", type_to_string(asset->type), get_asset_name(asset));
			}
		}

		if (!globals.dump)
		{
			return;
		}

		if (asset_type_filter.size() > 0 && !asset_type_filter.contains(asset->type))
		{
			return;
		}

		// dump referenced later
		if (is_referenced_asset(asset))
		{
			referenced_assets.emplace_back(asset->type, get_asset_name(asset));
			return;
		}

		const auto dump_func = dump_functions.find(globals.target_game);
		if (dump_func == dump_functions.end())
		{
			const auto name = game::get_mode_as_string(globals.target_game);
			ZONETOOL_ERROR("Dump mode \"%s\" is not supported", name.data());
			return;
		}

		dump_func->second(asset);
	}

	void dump_refs()
	{
		dump_trace("phase", "dump_refs begin");
		// remove duplicates
		std::sort(referenced_assets.begin(), referenced_assets.end());
		referenced_assets.erase(std::unique(referenced_assets.begin(),
			referenced_assets.end()), referenced_assets.end());

		for (auto& asset : referenced_assets)
		{
			if (asset.second.length() <= 1)
			{
				continue;
			}

			const auto asset_name = &asset.second[1];

			// DB_FindXAssetHeader synthesizes a default asset for anything that is not
			// loaded, and building one is fatal here for several types, so only take
			// assets that are genuinely resident
			dump_trace("ref-lookup", asset_name);
			const auto* entry = DB_FindXAssetEntry(asset.first, asset_name, false);
			const auto asset_header = entry ? entry->asset.header : XAssetHeader{};

			if (!asset_header.data)
			{
				ZONETOOL_ERROR("Could not find referenced asset \"%s\" of type \"%s\"", asset_name, type_to_string(asset.first));
				continue;
			}

			ZONETOOL_INFO("Dumping additional asset \"%s\" of type \"%s\"", asset_name, type_to_string(asset.first));

			XAsset referenced_asset =
			{
				asset.first,
				asset_header
			};

			dump_asset(&referenced_asset);
		}

		dump_trace("phase", "dump_refs end");
		referenced_assets.clear();
	}

	void stop_dumping()
	{
		std::lock_guard<std::recursive_mutex> lock(dump_lock);

		dump_trace("phase", "stop_dumping");
		globals.verify = false;

		if (globals.dump_csv)
		{
			globals.csv_file.close();
			globals.csv_file = {};
			globals.dump_csv = false;
		}

		if (!globals.dump)
		{
			return;
		}

		dump_refs();

		ZONETOOL_INFO("Zone \"%s\" dumped.", filesystem::get_fastfile().data());

		globals.dump = false;

		dump_trace("phase", "clearing xpak cache");
		xpak::clear_cache();
	
		zonetool::taskbar::clear();
	}

	utils::hook::detour db_add_xasset_hook;
	XAssetHeader db_add_xasset_stub(XAssetType type, XAssetHeader header)
	{
		XAsset xasset =
		{
			type,
			header
		};

		dump_asset(&xasset);

		const auto* name = get_asset_name(&xasset);
		if (name[0] == ',')
		{
			const auto* entry = DB_FindXAssetEntry(type, name + 1, false);
			return entry ? entry->asset.header : header;
		}

		// the engine must not take ownership of images here, letting DB_AddXAsset
		// run for them crashes the zone load
		if (type == ASSET_TYPE_IMAGE)
		{
			return header;
		}

		return db_add_xasset_hook.invoke<XAssetHeader>(type, header);
	}

	utils::hook::detour db_finish_load_x_file_hook;
	void db_finish_load_x_file_stub(void* a1, void* a2)
	{
		const auto ff_name = *reinterpret_cast<const char**>(0x1468FD4A8);
		if (ff_name == filesystem::get_fastfile())
		{
			stop_dumping();
		}
		
		return db_finish_load_x_file_hook.invoke<void>(a1, a2);
	}

	/*void reallocate_asset_pool(const XAssetType type, const unsigned int new_size)
	{
		const size_t element_size = DB_GetXAssetTypeSize(type);

		auto* new_pool = utils::memory::get_allocator()->allocate(new_size * element_size);
		std::memmove(new_pool, g_assetPool[type], g_poolSize[type] * element_size);

		g_assetPool[type] = new_pool;
		g_poolSize[type] = new_size;
	}

	void reallocate_asset_pool_multiplier(const XAssetType type, unsigned int multiplier)
	{
		const auto new_size = multiplier * g_poolSize[type];
		reallocate_asset_pool(type, multiplier * new_size);
	}*/

	bool load_zone(const std::string& name, bool sync = false, bool inform = true)
	{
		if (!zone_exists(name.data()))
		{
			ZONETOOL_INFO("Zone \"%s\" could not be found!", name.data());
			return false;
		}

		wait_for_database();

		if (is_zone_loaded(name))
		{
			ZONETOOL_INFO("zone \"%s\" is already loaded...", name.data());
			return false;
		}

		if (inform)
		{
			ZONETOOL_INFO("Loading zone \"%s\"...", name.data());
		}

		XZoneInfo zone{};
		zone.name = name.data();
		zone.allocFlags = 0x10000;
		zone.freeFlags = 0;
		zone.allocSlot = 0;
		zone.freeSlot = 0;

		zonetool::taskbar::set_indeterminate();

		utils::hook::invoke<void>(0x140506C70);
		DB_LoadXAssets(&zone, 1, static_cast<qboolean>(sync), 0);
		utils::hook::invoke<void>(0x1401360A0);
		if (!utils::hook::invoke<bool>(0x1406FF940))
			utils::hook::invoke<void>(0x14014C610);
		utils::hook::invoke<void>(0x140309FF0);
		utils::hook::invoke<void>(0x14038ED20);

		//utils::hook::copy_string(0x14699D220, name.data());
		//utils::hook::invoke<void>(0x1401DB0B0); // DB_LoadZone

		return true;
	}

	void unload_zones()
	{
		int index = *g_zoneCount;
		auto* zone = &g_zones[index];
		auto zone_name = &g_zoneNames[index];

		while (index > 0)
		{
			if ((g_zones[index].flags & 0x10000) != 0)
			{
				DB_UnloadXZone(zone->index, 0, 0);
			}

			index--;
			zone--;
			zone_name--;
		}
		
		ZONETOOL_INFO("Unloaded loaded zones...");
	}

	void dump_zone(const std::string& name, const game::game_mode target, const std::optional<std::string> fastfile = {})
	{
		if (!zone_exists(name.data()))
		{
			ZONETOOL_INFO("Zone \"%s\" could not be found!", name.data());
			return;
		}

		wait_for_database();

		globals.target_game = target;
		ZONETOOL_INFO("Dumping zone \"%s\"...", name.data());

		if (fastfile.has_value())
		{
			filesystem::set_fastfile(fastfile.value());
		}
		else
		{
			filesystem::set_fastfile(name);
		}

		globals.dump = true;
		globals.dump_csv = true;
		if (!load_zone(name, false, false))
		{
			globals.dump = false;
			globals.dump_csv = false;
			return;
		}

		while (globals.dump)
		{
			Sleep(1);
		}
	}

	void dump_csv(const std::string& name)
	{
		if (!zone_exists(name.data()))
		{
			ZONETOOL_INFO("Zone \"%s\" could not be found!", name.data());
			return;
		}

		wait_for_database();

		ZONETOOL_INFO("Dumping csv \"%s\"...", name.data());

		filesystem::set_fastfile(name);

		globals.dump_csv = true;
		if (!load_zone(name, false, true))
		{
			globals.dump_csv = false;
			return;
		}

		while (globals.dump_csv)
		{
			Sleep(1);
		}

		ZONETOOL_INFO("Csv \"%s\" dumped...", name.data());
	}

	void verify_zone(const std::string& name)
	{
		if (!zone_exists(name.data()))
		{
			ZONETOOL_INFO("Zone \"%s\" could not be found!", name.data());
			return;
		}

		wait_for_database();

		globals.verify = true;
		if (!load_zone(name, false, true))
		{
			globals.verify = false;
		}

		while (globals.verify)
		{
			Sleep(1);
		}
	}

	void iterate_zones()
	{
		const auto iterate_zones_internal = [](const std::string& path)
		{
			for (auto const& dir_entry : std::filesystem::directory_iterator{ path })
			{
				if (dir_entry.is_regular_file() && dir_entry.path().extension() == ".ff")
				{
					const auto zone = dir_entry.path().stem().string();

					load_zone(zone);

					wait_for_database();
					unload_zones();
				}
			}
		};

		const auto zone_path = utils::io::directory_exists("zone") ? "zone/" : "";
		iterate_zones_internal(zone_path);
	}

	void register_commands()
	{
		::t7::command::add("quit", []()
		{
			std::quick_exit(EXIT_SUCCESS);
		});

		::t7::command::add("buildzone", [](const ::t7::command::params& params)
		{
			if (params.size() != 2)
			{
				ZONETOOL_ERROR("usage: buildzone <zone>");
				return;
			}

			ZONETOOL_ERROR("buildzone is not supported");
		});

		::t7::command::add("loadzone", [](const ::t7::command::params& params)
		{
			if (params.size() != 2)
			{
				ZONETOOL_ERROR("usage: loadzone <zone>");
				return;
			}

			load_zone(params.get(1));
		});

		::t7::command::add("unloadzones", []()
		{
			unload_zones();
		});

		::t7::command::add("dumpzone", [](const ::t7::command::params& params)
		{
			if (params.size() < 2)
			{
				ZONETOOL_ERROR("usage: dumpzone <zone>");
				return;
			}

			asset_type_filter.clear();

			if (params.size() >= 3)
			{
				const auto mode = params.get(1);
				const auto dump_target = game::get_mode_from_string(mode);

				if (dump_target == game::none)
				{
					ZONETOOL_ERROR("Invalid dump target \"%s\"", mode);
					return;
				}

				if (!dump_functions.contains(dump_target))
				{
					ZONETOOL_ERROR("Unsupported dump target \"%s\" (%i)", mode, dump_target);
					return;
				}

				if (params.size() >= 4)
				{
					const auto asset_types_str = params.get(3);
					const auto asset_types = utils::string::split(asset_types_str, ',');

					for (const auto& type_str : asset_types)
					{
						const auto type = type_to_int(type_str);
						if (type == -1)
						{
							ZONETOOL_ERROR("Asset type \"%s\" does not exist", type_str.data());
							return;
						}

						asset_type_filter.insert(static_cast<XAssetType>(type));
					}
				}

				dump_zone(params.get(2), dump_target);
			}
			else
			{
				dump_zone(params.get(1), game::t7);
			}
		});

		::t7::command::add("dumpasset", [](const ::t7::command::params& params)
		{
			if (params.size() < 3)
			{
				ZONETOOL_ERROR("usage: dumpasset [target] <type> <name>");
				return;
			}

			auto target = game::t7;
			auto arg = 1;

			if (params.size() >= 4)
			{
				const auto mode = params.get(1);
				target = game::get_mode_from_string(mode);

				if (target == game::none)
				{
					ZONETOOL_ERROR("Invalid dump target \"%s\"", mode);
					return;
				}

				if (!dump_functions.contains(target))
				{
					ZONETOOL_ERROR("Unsupported dump target \"%s\" (%i)", mode, target);
					return;
				}

				arg = 2;
			}

			const auto type_str = params.get(arg);
			const auto type_int = type_to_int(type_str);
			if (type_int == -1)
			{
				ZONETOOL_ERROR("Asset type \"%s\" does not exist", type_str);
				return;
			}

			asset_type_filter.clear();

			const auto type = XAssetType(type_int);
			const auto name = params.get(arg + 1);

			XAsset asset{};
			asset.type = type;

			XAssetHeader header{};

			if (type == ASSET_TYPE_IMAGE)
			{
				// avoid the default-asset path, it is fatal for images
				const auto* entry = DB_FindXAssetEntry(type, name, false);
				if (entry)
				{
					header = entry->asset.header;
				}
			}
			else
			{
				header = db_find_x_asset_header(type, name);
			}
			if (!header.data)
			{
				ZONETOOL_INFO("Asset not found\n");
				return;
			}

			globals.dump = true;
			const auto _0 = gsl::finally([]
			{
				globals.dump = false;
			});

			filesystem::set_fastfile("assets");
			asset.header = header;
			globals.target_game = target;
			dump_asset(&asset);

			ZONETOOL_INFO("Dumped to dump/assets");
		});

		::t7::command::add("dumpcsv", [](const ::t7::command::params& params)
		{
			if (params.size() != 2)
			{
				ZONETOOL_ERROR("usage: dumpcsv <zone>");
				return;
			}

			dump_csv(params.get(1));
		});

		::t7::command::add("dumpassets", [](const ::t7::command::params& params)
		{
			if (params.size() < 3)
			{
				ZONETOOL_ERROR("usage: dumpassets <target> <type> [substring]");
				return;
			}

			const auto mode = params.get(1);
			const auto target = game::get_mode_from_string(mode);
			if (target == game::none || !dump_functions.contains(target))
			{
				ZONETOOL_ERROR("Invalid dump target \"%s\"", mode);
				return;
			}

			const auto type_str = params.get(2);
			const auto type_int = type_to_int(type_str);
			if (type_int == -1)
			{
				ZONETOOL_ERROR("Asset type \"%s\" does not exist", type_str);
				return;
			}

			const auto type = XAssetType(type_int);
			const std::string filter = params.size() >= 4 ? params.get(3) : "";

			asset_type_filter.clear();
			asset_type_filter.insert(type);

			globals.target_game = target;
			globals.dump = true;
			const auto _0 = gsl::finally([]
			{
				globals.dump = false;
			});

			filesystem::set_fastfile("assets");

			auto count = 0;
			for (auto bucket = 0u; bucket < ASSET_HASH_BUCKET_COUNT; bucket++)
			{
				for (auto index = g_assetHashTable[bucket]; index; index = g_assetEntries[index].nextHash)
				{
					auto& entry = g_assetEntries[index];
					if (entry.unloaded || entry.asset.type != type)
					{
						continue;
					}

					const std::string name = get_asset_name(&entry.asset);
					if (name.empty() || (!filter.empty() && name.find(filter) == std::string::npos))
					{
						continue;
					}

					XAsset asset = entry.asset;
					dump_asset(&asset);
					count++;
				}
			}

			ZONETOOL_INFO("Dumped %i assets of type \"%s\" to dump/assets", count,
				type_to_string(type));
		});

		::t7::command::add("listassets", [](const ::t7::command::params& params)
		{
			if (params.size() < 2)
			{
				ZONETOOL_ERROR("usage: listassets <type> [substring]");
				return;
			}

			const auto type_int = type_to_int(params.get(1));
			if (type_int == -1)
			{
				ZONETOOL_ERROR("Asset type \"%s\" does not exist", params.get(1));
				return;
			}

			const auto type = XAssetType(type_int);
			const std::string filter = params.size() >= 3 ? params.get(2) : "";

			auto count = 0;

			for (auto bucket = 0u; bucket < ASSET_HASH_BUCKET_COUNT; bucket++)
			{
				for (auto index = g_assetHashTable[bucket]; index; index = g_assetEntries[index].nextHash)
				{
					auto& entry = g_assetEntries[index];
					if (entry.unloaded || entry.asset.type != type)
					{
						continue;
					}

					const std::string name = get_asset_name(&entry.asset);
					if (!filter.empty() && name.find(filter) == std::string::npos)
					{
						continue;
					}

					ZONETOOL_INFO("%s (zone %u)", name.data(), entry.zoneIndex);
					count++;
				}
			}

			ZONETOOL_INFO("Found %i assets of type \"%s\"", count, type_to_string(type));
		});


		::t7::command::add("verifyzone", [](const ::t7::command::params& params)
		{
			if (params.size() != 2)
			{
				ZONETOOL_ERROR("usage: verifyzone <zone>");
				return;
			}

			verify_zone(params.get(1));
		});

		::t7::command::add("iteratezones", []()
		{
			iterate_zones();
		});
	}

	std::vector<std::string> get_command_line_arguments()
	{
		LPWSTR* szArglist;
		int nArgs;

		szArglist = CommandLineToArgvW(GetCommandLineW(), &nArgs);

		std::vector<std::string> args;
		args.resize(nArgs);

		// convert all args to std::string
		for (int i = 0; i < nArgs; i++)
		{
			auto curArg = std::wstring(szArglist[i]);
			args[i] = std::string(curArg.begin(), curArg.end());
		}

		// return arguments
		return args;
	}

	void handle_params()
	{
		// Execute command line commands
		auto args = get_command_line_arguments();
		if (args.size() > 1)
		{
			for (std::size_t i = 0; i < args.size(); i++)
			{
				if (i < args.size() - 1 && i + 1 < args.size())
				{
					if (args[i] == "-loadzone")
					{
						load_zone(args[i + 1]);
						i++;
					}
					else if (args[i] == "-verifyzone")
					{
						verify_zone(args[i + 1]);
						i++;
					}
					else if (args[i] == "-dumpzone")
					{
						// -dumpzone [target] <zone>, target defaults to t7 so the
						// existing single-argument form keeps working
						auto target = game::t7;
						auto zone_index = i + 1;

						if (i + 2 < args.size())
						{
							const auto maybe_target = game::get_mode_from_string(args[i + 1]);
							if (maybe_target != game::none && dump_functions.contains(maybe_target))
							{
								target = maybe_target;
								zone_index = i + 2;
							}
						}

						dump_zone(args[zone_index], target);
						i = zone_index;
					}
				}
			}

			std::quick_exit(EXIT_SUCCESS);
		}
	}

	void on_exit(void)
	{
		globals.verify = false;
		globals.dump = false;
		globals.dump_csv = false;
		globals.csv_file.close();
	}

	utils::hook::detour doexit_hook;
	void doexit(unsigned int a1, int a2, int a3)
	{
		on_exit();
		doexit_hook.invoke<void>(a1, a2, a3);
	}

	/*void skip_extra_zones_stub(utils::hook::assembler& a)
	{
		const auto skip = a.newLabel();
		const auto original = a.newLabel();

		a.pushad64();
		a.test(esi, 0x10000000); // allocFlags
		a.jnz(skip);

		a.bind(original);
		a.popad64();
		a.call(0x1401DC090);
		a.test(al, al);
		a.jz(0x1401DBC62);
		a.jmp(0x1401DBB5E);

		a.bind(skip);
		a.popad64();
		a.push(r14d);
		a.mov(r14d, 0x10000000);
		a.not_(r14d);
		a.and_(esi, r14d);
		a.pop(r14d);
		a.jmp(0x1401DBC62);
	}*/

	void init_zonetool()
	{
		static bool initialized = false;
		if (initialized) return;
		initialized = true;
		ZONETOOL_INFO("ZoneTool is initializing...");

		// reallocs
		//reallocate_asset_pool_multiplier(ASSET_TYPE_RAWFILE, 2);

		// enable dumping
		db_add_xasset_hook.create(0x1401D4600, &db_add_xasset_stub);

		// stop dumping
		db_finish_load_x_file_hook.create(0x1401D3F00, &db_finish_load_x_file_stub);

		//doexit_hook.create(, doexit);
		atexit(on_exit);

		utils::hook::nop(0x1401D87AE, 5); // nop original "loadzone" command

		//utils::hook::nop(0x1401DBB51, 13);
		//utils::hook::jump(0x1401DBB51, utils::hook::assemble(skip_extra_zones_stub), true);
	}

	void finalize()
	{
		ZONETOOL_INFO("ZoneTool initialization complete!");
	}

	void initialize()
	{
		init_zonetool();
	}

	void start()
	{
		initialize();
		finalize();

		branding();
		register_commands();

		handle_params();
	}
}
