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

	// The zone the current dump is loading. The fast file name the dump writes under can
	// differ from it (-dumpas), and the end of the load is recognised by this one.
	std::string dumping_zone;
	std::unordered_set<XAssetType> asset_type_filter;
	std::recursive_mutex dump_lock;

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

	// A printable name of 1-255 characters. Reading through a bad pointer faults into the handler instead of
	// asking the kernel about the page first: a VirtualQuery per call (several per asset, for every asset of
	// every zone, BO3's own included) was most of the database thread's time while zones loaded.
	bool printable_name(const char* name)
	{
		__try
		{
			for (auto i = 0; i < 256; i++)
			{
				const auto c = static_cast<unsigned char>(name[i]);
				if (!c)
				{
					return i > 0;
				}
				if (c < 0x20 || c > 0x7E)
				{
					return false;
				}
			}
			return false;
		}
		__except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION || GetExceptionCode() == STATUS_GUARD_PAGE_VIOLATION
			? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
		{
			return false;
		}
	}

	const char* safe_asset_name(const char* name)
	{
		return name && printable_name(name) ? name : "";
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
			// map assets depend on each other and are converted once the zone has loaded
			if (converter::iw7::map::remember(asset->type, asset->header.data))
			{
				return;
			}

			DUMP_ASSET_CONVERT(ASSET_TYPE_XANIMPARTS, xanim, XAnimParts);
			DUMP_ASSET_CONVERT(ASSET_TYPE_XMODEL, xmodel, XModel);
			DUMP_ASSET_CONVERT(ASSET_TYPE_XMODELMESH, xmodel_mesh, XModelMesh);
			DUMP_ASSET_CONVERT(ASSET_TYPE_IMAGE, gfximage, GfxImage);
			DUMP_ASSET_CONVERT(ASSET_TYPE_MATERIAL, material, Material);
			DUMP_ASSET_CONVERT(ASSET_TYPE_FX, fxeffectdef, FxEffectDef);
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
			if (asset->type == ASSET_TYPE_IMAGE)
			{
				// an image named with ZT_UI_IMAGE_PREFIX as decoded pixels: ui_images/<image>.raw, a { width, height, depth,
				// faces, levels, DXGI format } u32 header then the pixels (the map's own HUD art, rebuilt in IW7 LUI)
				const auto* image = reinterpret_cast<const GfxImage*>(asset->header.data);
				const auto* prefix = std::getenv("ZT_UI_IMAGE_PREFIX");
				converter::iw7::gfximage::image_pixels pixels;
				if (image && image->name && prefix && !std::strncmp(image->name, prefix, std::strlen(prefix))
					&& converter::iw7::gfximage::get_pixels(image, pixels))
				{
					std::string out(24, '\0');
					const std::uint32_t header[6] = { pixels.width, pixels.height, pixels.depth, pixels.faces, pixels.levels,
						static_cast<std::uint32_t>(pixels.format) };
					std::memcpy(out.data(), header, sizeof(header));
					out.append(reinterpret_cast<const char*>(pixels.data.data()), pixels.data.size());
					std::string file_name = utils::string::va("ui_images/%s.raw", image->name);
					for (auto& c : file_name)
					{
						if (c == '*' || c == '#')
						{
							c = '_';
						}
					}
					filesystem::file file(file_name);
					file.open("wb");
					file.write(out);
					file.close();
				}
			}
			if (asset->type == ASSET_TYPE_SOUND)
			{
				// SndBank: aliasCount +0x20, alias lists +0x28 (0x28 each: name, id, aliases +0x10, count +0x18); an alias is
				// 0xD8 (name +0, assetId +0x38 = SND_HashName(file), file +0x40); written as list,alias,file,assetId,raw hex
				const auto* bank = reinterpret_cast<const std::uint8_t*>(asset->header.data);
				const auto list_count = *reinterpret_cast<const std::uint32_t*>(bank + 0x20);
				const auto* lists = *reinterpret_cast<const std::uint8_t* const*>(bank + 0x28);
				std::string csv;
				for (std::uint32_t l = 0; l < list_count && lists; l++)
				{
					const auto* list = lists + 0x28 * l;
					const auto* list_name = *reinterpret_cast<const char* const*>(list);
					const auto* aliases = *reinterpret_cast<const std::uint8_t* const*>(list + 0x10);
					const auto count = *reinterpret_cast<const std::uint32_t*>(list + 0x18);
					for (std::uint32_t a = 0; a < count && aliases; a++)
					{
						const auto* alias = aliases + 0xD8 * a;
						const auto* name = *reinterpret_cast<const char* const*>(alias);
						const auto* file = *reinterpret_cast<const char* const*>(alias + 0x40);
						const auto id = *reinterpret_cast<const std::uint32_t*>(alias + 0x38);
						csv += utils::string::va("%s,%s,%s,%08X,", list_name ? list_name : "", name ? name : "", file ? file : "", id);
						for (auto b = 0; b < 0xD8; b++)
						{
							csv += utils::string::va("%02X", alias[b]);
						}
						csv += "\n";
					}
				}
				filesystem::file file(utils::string::va("sound_aliases/%s.csv", get_asset_name(asset)));
				file.open("wb");
				file.write(csv);
				file.close();
			}
			if (asset->type == ASSET_TYPE_SOUND && std::getenv("ZT_SOUND_PROBE"))
			{
				const auto readable = [](const void* p, std::size_t n)
				{
					MEMORY_BASIC_INFORMATION mbi{};
					if (!p || !VirtualQuery(p, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
					{
						return false;
					}
					return reinterpret_cast<std::uintptr_t>(p) + n <= reinterpret_cast<std::uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
				};
				std::string out;
				const auto block = [&](const std::uint8_t* at, std::size_t n, int depth, auto&& self) -> void
				{
					if (!readable(at, n))
					{
						return;
					}
					out += utils::string::va("@%p %zu %d\n", at, n, depth);
					out.append(reinterpret_cast<const char*>(at), n);
					out += "\n";
					if (depth >= 2)
					{
						return;
					}
					for (std::size_t o = 0; o + 8 <= n; o += 8)
					{
						const auto* p = *reinterpret_cast<const std::uint8_t* const*>(at + o);
						if (p != at && readable(p, 0x400))
						{
							self(p, 0x400, depth + 1, self);
						}
					}
				};
				block(reinterpret_cast<const std::uint8_t*>(asset->header.data), 0x200, 0, block);
				filesystem::file file(utils::string::va("sound_probe/%s.bin", get_asset_name(asset)));
				file.open("wb");
				file.write(out);
				file.close();
			}
			if (asset->type == ASSET_TYPE_RAWFILE)
			{
				// written raw: { name, len +8, buffer +16 }
				const auto* raw = reinterpret_cast<const std::uint8_t*>(asset->header.data);
				const auto* name = *reinterpret_cast<const char* const*>(raw);
				const auto len = *reinterpret_cast<const std::uint32_t*>(raw + 8);
				const auto* buffer = *reinterpret_cast<const std::uint8_t* const*>(raw + 16);
				if (name && buffer && len)
				{
					filesystem::file file(name);
					file.open("wb");
					file.write(buffer, len, 1);
					file.close();
				}
			}
			if (asset->type == ASSET_TYPE_STRINGTABLE)
			{
				// written as its csv: { name, columns +8, rows +12, cells +16 }, a cell { string, hash } of 16 bytes (measured on
				// zm_waterparkfinale's weapon table: 20 x 57, header row first)
				const auto* raw = reinterpret_cast<const std::uint8_t*>(asset->header.data);
				const auto* name = *reinterpret_cast<const char* const*>(raw);
				const auto columns = *reinterpret_cast<const int*>(raw + 8);
				const auto rows = *reinterpret_cast<const int*>(raw + 12);
				const auto* cells = *reinterpret_cast<const std::uint8_t* const*>(raw + 16);
				std::string csv;
				for (auto r = 0; r < rows && cells; r++)
				{
					for (auto c = 0; c < columns; c++)
					{
						const auto* s = *reinterpret_cast<const char* const*>(cells + 16 * (r * columns + c));
						csv += (c ? "," : "") + std::string(s ? s : "");
					}
					csv += "\n";
				}
				filesystem::file file(name);
				file.open("wb");
				file.write(csv);
				file.close();
			}
			if (asset->type == ASSET_TYPE_SCRIPTPARSETREE)
			{
				// compiled script, written raw: { name, len, buffer } or { name, buffer, len }
				const auto* raw = reinterpret_cast<const std::uint8_t*>(asset->header.data);
				const auto* name = *reinterpret_cast<const char* const*>(raw);
				const std::uint8_t* buffer = nullptr;
				std::size_t len = 0;
				for (const auto [len_at, ptr_at] : {std::pair{8, 16}, std::pair{16, 8}})
				{
					const auto* candidate = *reinterpret_cast<const std::uint8_t* const*>(raw + ptr_at);
					const auto size = *reinterpret_cast<const std::uint32_t*>(raw + len_at);
					if (candidate && size > 8 && !std::memcmp(candidate, "\x80GSC", 4))
					{
						buffer = candidate;
						len = size;
						break;
					}
				}
				if (!buffer)
				{
					ZONETOOL_WARNING("scriptparsetree \"%s\": no GSC buffer found", name);
					return;
				}
				filesystem::file file(std::string(name) + "c");
				file.open("wb");
				file.write(buffer, len, 1);
				file.close();
			}
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

			dump_trace("ref-lookup", asset_name);
			XAssetHeader asset_header{};
			if (asset.first == ASSET_TYPE_IMAGE)
			{
				// images never reach the pool (db_add_xasset_stub keeps its own copies)
				asset_header.image = converter::iw7::gfximage::find_image(asset_name);
			}
			else
			{
				const auto* entry = DB_FindXAssetEntry(asset.first, asset_name, false);
				asset_header = entry ? entry->asset.header : XAssetHeader{};
			}

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

	// the zone brought map assets to convert (converter::iw7::map::remember); finish_dump converts them
	bool map_conversion_pending = false;

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

		if (globals.target_game == game::iw7)
		{
			// DB_LoadXFile reads the delay-loaded blocks right before DB_FinishLoadXFile
			dump_trace("phase", "images");
			converter::iw7::gfximage::flush_dumps();

			// the map assets wait for the end of the load (finish_dump)
			map_conversion_pending = true;
		}

		globals.dump = false;
	}

	// Once the dumped zone has finished loading. BO3 applies the asset overrides it queued while loading a zone in
	// DB_PostLoadXZone (0x1401D8E50), which otherwise only runs when the next zone starts loading: until then an asset of
	// the zone that an earlier zone referenced is the placeholder copy of its type's default that the reference made, and
	// the zone's own pointers to it read that copy.
	void finish_dump()
	{
		std::lock_guard<std::recursive_mutex> lock(dump_lock);

		if (map_conversion_pending)
		{
			map_conversion_pending = false;
			wait_for_database();
			utils::hook::invoke<void>(0x1401D8E50, 0); // DB_PostLoadXZone

			dump_trace("phase", "map assets");
			try
			{
				converter::iw7::map::convert_pending();
			}
			catch (const std::exception& e)
			{
				ZONETOOL_FATAL("A fatal exception occured while converting the map assets of zone \"%s\", exception was: \n%s",
					filesystem::get_fastfile().data(), e.what());
			}
		}

		ZONETOOL_INFO("Zone \"%s\" dumped.", filesystem::get_fastfile().data());

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

		if (type == ASSET_TYPE_IMAGE)
		{
			// Images must not go through the game's own add (it crashes the load), and every
			// one of them is loaded into the same temporary block. Hand the loader a copy we
			// own instead, so each material texture slot keeps pointing at its own image. The
			// copy is also what gets dumped: the header itself is gone by the time the zone's
			// delayed pixel data has been read.
			XAssetHeader image{};
			image.image = converter::iw7::gfximage::register_image(header.image);
			if (header.image && header.image->name && header.image->name[0] != ',')
			{
				xasset.header = image;
			}
			dump_asset(&xasset);
			return image;
		}

		dump_asset(&xasset);

		const auto* name = get_asset_name(&xasset);
		if (name[0] == ',')
		{
			const auto* entry = DB_FindXAssetEntry(type, name + 1, false);
			if (entry)
			{
				return entry->asset.header;
			}

			// A reference to an asset that is not loaded: BO3's own add (0x1401D7960 -> 0x1401D4E20) links it to an
			// unloaded placeholder, a copy of the type's default asset, which the real asset replaces if its zone
			// loads later. An unresolved reference stub faults in DB_FreeUnusedResources (0x1401D6650 -> Mark_XAsset).
			const auto* default_entry = DB_FindDefaultXAssetEntry(type);
			if (default_entry && default_entry->asset.header.data)
			{
				return db_add_xasset_hook.invoke<XAssetHeader>(type, header);
			}
			return header;
		}

		return db_add_xasset_hook.invoke<XAssetHeader>(type, header);
	}

	utils::hook::detour db_finish_load_x_file_hook;
	void db_finish_load_x_file_stub(void* a1, void* a2)
	{
		const auto ff_name = *reinterpret_cast<const char**>(0x1468FD4A8);
		if (ff_name && ff_name == dumping_zone)
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

	// The alloc flags BO3 gives a zone of its core or mode zone tables (40-byte XZoneInfo records, read by 0x1401DBD90:
	// core_ui 0x8, core_mod 0x40, zm_patch 0x10000080, zm_common 0x80, zm_mod 0x100, zm_levelcommon 0x200, ...); 0x10000
	// for any other zone (a map). A map's own patch and localized zones load with its alloc flags and free flags derived
	// from them, which unload every zone sharing 0x10000: the common zones need their own flags to stay loaded.
	// the zones load_zone loaded, for unload_zones
	std::unordered_set<std::string> loaded_zones;

	std::uint32_t zone_alloc_flags(const std::string& name)
	{
		struct zone_table
		{
			std::uintptr_t address;
			std::size_t count;
		};
		static constexpr zone_table tables[] = { { 0x1410C22C0, 10 }, { 0x1410C2450, 4 }, { 0x1410C24F0, 5 }, { 0x1410C25C0, 4 } };
		for (const auto& table : tables)
		{
			for (auto i = 0u; i < table.count; i++)
			{
				const auto* entry = reinterpret_cast<const XZoneInfo*>(table.address + i * 40);
				if (entry->name && !_stricmp(entry->name, name.data()))
				{
					return entry->allocFlags;
				}
			}
		}
		return 0x10000;
	}

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

		// DB_LoadXAssets (0x1401D8740) replaces free flags without 0x40000000 by ones derived from the alloc flags (alloc
		// bit b frees every loaded zone with a bit >= b of the same group: 0xFFF, 0x1F000 or 0x3FE0000) and unloads the
		// loaded zones that have them first. 0x40000000 keeps the free flags as given: nothing is unloaded.
		XZoneInfo zone{};
		zone.name = name.data();
		zone.allocFlags = zone_alloc_flags(name);
		zone.freeFlags = 0x40000000;
		loaded_zones.insert(name);

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
			// the maps (0x10000) and the zones load_zone gave BO3's own flags
			if ((g_zones[index].flags & 0x10000) != 0 || loaded_zones.contains(zone_name->name))
			{
				DB_UnloadXZone(zone->index, 0, 0);
			}

			index--;
			zone--;
			zone_name--;
		}
		loaded_zones.clear();

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

		dumping_zone = name;
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
		finish_dump();
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
			std::optional<std::string> dump_as;

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
					else if (args[i] == "-dumpas")
					{
						// write the next dump to dump/<name>/ instead of dump/<zone>/
						dump_as = args[i + 1];
						i++;
					}
					else if (args[i] == "-dumpzone")
					{
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

						// optional asset type filter after the zone, as in the console command:
						// -dumpzone iw7 <zone> com_map,gfx_map
						asset_type_filter.clear();
						auto last = zone_index;
						if (zone_index + 1 < args.size() && !args[zone_index + 1].starts_with("-"))
						{
							last = zone_index + 1;
							for (const auto& type_str : utils::string::split(args[last], ','))
							{
								const auto type = type_to_int(type_str);
								if (type == -1)
								{
									ZONETOOL_ERROR("Asset type \"%s\" does not exist", type_str.data());
									continue;
								}

								asset_type_filter.insert(static_cast<XAssetType>(type));
							}
						}

						dump_zone(args[zone_index], target, dump_as);
						i = last;
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
		const auto skip = a.new_label();
		const auto original = a.new_label();

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

		// BO3's asset copy (0x1401D91A0: an override, or the placeholder copy of a type's default for a reference to an
		// asset that is not loaded, 0x1401D4E20 with flags 17) releases the Direct3D objects of the technique set or
		// compute shader set it writes over and adds references to the ones it copies. This process has no renderer,
		// and a placeholder lands in a recycled pool slot whose stale technique pointers reach freed memory. The
		// reference counting is skipped.
		utils::hook::set<std::uint8_t>(0x14038EFC0, 0xC3); // technique set: release
		utils::hook::set<std::uint8_t>(0x14038EE60, 0xC3); // technique set: add references
		utils::hook::set<std::uint8_t>(0x14038EAD0, 0xC3); // compute shader set: release
		utils::hook::set<std::uint8_t>(0x14038EA60, 0xC3); // compute shader set: add references

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
