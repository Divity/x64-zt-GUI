#include "std_include.hpp"
#include "zonetool/utils/utils.hpp"
#include "xpak.hpp"

#include "../structs.hpp"
#include "../variables.hpp"

#include "utils/io.hpp"
#include "utils/string.hpp"

#include "lz4.h"

namespace zonetool::t7
{
	namespace xpak
	{
		namespace
		{
#pragma pack(push, 1)
			struct XPakSection
			{
				int64_t itemCount;
				int64_t offset;
				int64_t size;
			};

			struct XPakHeader
			{
				uint32_t Magic;
				uint16_t Unknown1;
				uint16_t Version;
				uint64_t Unknown2;
				uint64_t Size;
				uint64_t FileCount;
				uint64_t DataOffset;
				uint64_t DataSize;
				uint64_t HashCount;
				uint64_t HashOffset;
				uint64_t HashSize;
				uint64_t Unknown3;
				uint64_t UnknownOffset;
				uint64_t Unknown4;
				uint64_t IndexCount;
				uint64_t IndexOffset;
				uint64_t IndexSize;
			};

			struct XPakHashEntry
			{
				uint64_t Key;
				uint64_t Offset;
				uint64_t Size;
			};

			struct XPakDataHeader
			{
				// Count and offset
				uint32_t Count;
				uint32_t Offset;

				// The commands tell what each block of data does
				uint32_t Commands[30];
			};
#pragma pack(pop)

			struct PackageCacheObject
			{
				uint64_t Offset;
				uint64_t CompressedSize;
				uint64_t UncompressedSize;
				std::string PackageFile;
			};

			std::unordered_map<std::string, std::unordered_map<uint64_t, PackageCacheObject>> xpak_cache_map;

			namespace
			{
				size_t align_value(size_t value, unsigned int alignment)
				{
					const auto diff = value % alignment;
					return diff != 0
						? value + (alignment - diff)
						: value;
				}

				const char* align_value(const char* value, unsigned int alignment)
				{
					return reinterpret_cast<const char*>(
						align_value(reinterpret_cast<size_t>(value), alignment));
				}
			}

			std::vector<std::uint8_t> extract(const void* data, const size_t size, const size_t decompressedSize)
			{
				std::vector<std::uint8_t> out_buffer(decompressedSize);

				const auto* base = reinterpret_cast<const char*>(data);
				size_t pos = 0;
				size_t written = 0;

				while (pos + sizeof(XPakDataHeader) <= size && written < decompressedSize)
				{
					XPakDataHeader header{};
					std::memcpy(&header, base + pos, sizeof(header));
					pos += sizeof(header);

					if (header.Count > 30)
					{
						break;
					}

					for (uint32_t i = 0; i < header.Count && written < decompressedSize; i++)
					{
						const size_t blockSize = (header.Commands[i] & 0xFFFFFF);
						const size_t flag = (header.Commands[i] >> 24);

						if (pos + blockSize > size)
						{
							return {};
						}

						const auto* block = base + pos;
						const auto remaining = decompressedSize - written;

						switch (flag)
						{
						case 0x3: // compressed (lz4)
						{
							const auto result = LZ4_decompress_safe(block,
								reinterpret_cast<char*>(out_buffer.data()) + written,
								static_cast<int>(blockSize), static_cast<int>(remaining));
							if (result <= 0)
							{
								return {};
							}
							written += static_cast<size_t>(result);
							break;
						}
						case 0x0: // raw data
						{
							const auto copy_size = std::min(blockSize, remaining);
							std::memcpy(out_buffer.data() + written, block, copy_size);
							written += copy_size;
							break;
						}
						default: // oodle, unused by t7
						{
							return {};
						}
						}

						pos += blockSize;
					}

					pos = (pos + 0x7F) & ~static_cast<size_t>(0x7F);
				}

				out_buffer.resize(written);
				return out_buffer;
			}


			std::vector<std::uint8_t> decompress_xpak_data(std::string compressed_data, size_t decompressedSize)
			{
				try
				{
					return extract(compressed_data.data(), compressed_data.size(), decompressedSize);
				}
				catch (const std::exception& e)
				{
					ZONETOOL_WARNING("failed to decompress xpak data: %s", e.what());
					return {};
				}
			}

			void populate_xpak_cache_internal(const std::string& pak_path)
			{
				if (xpak_cache_map.contains(pak_path)) return;

				filesystem::file file(pak_path);
				file.open("rb", false, false);

				if (!file.get_fp()) return;

				XPakHeader header{};
				file.read(&header);

				if (header.Magic != 0x4950414b)
				{
					ZONETOOL_WARNING("skipping \"%s\", not a valid xpak", pak_path.data());
					file.close();
					return;
				}

				file.seek(header.HashOffset, SEEK_SET);
				XPakHashEntry entry{};

				for (uint64_t i = 0; i < header.HashCount; i++)
				{
					file.read(&entry);

					// Prepare a cache entry
					PackageCacheObject NewObject{};
					// Set data
					NewObject.Offset = header.DataOffset + entry.Offset;
					NewObject.CompressedSize = entry.Size & 0xFFFFFFFFFFFFFF; // 0x80 in last 8 bits in some entries in new XPAKs
					NewObject.UncompressedSize = 0;
					NewObject.PackageFile = pak_path;
					// Append to database

					xpak_cache_map[pak_path].insert(std::make_pair(entry.Key, NewObject));
				}

				file.close();
			}

			void populate_xpak_cache_iterator(const std::string& path)
			{
				if (!std::filesystem::is_directory(path))
				{
					return;
				}

				for (auto const& dir_entry : std::filesystem::directory_iterator{ path })
				{
					if (dir_entry.is_regular_file() && dir_entry.path().extension() == ".xpak")
					{
						const auto pakfile = dir_entry.path().string();

						populate_xpak_cache_internal(pakfile);
					}
				}
			}

			void try_populate_xpak_cache_for_zone(const std::string& name)
			{
				auto pak_name = name + ".xpak";
				auto pak_path = "zone/" + pak_name;

				if (!utils::io::file_exists(pak_path))
				{
					pak_path = "../" + pak_path;
					if (!utils::io::file_exists(pak_path)) return;
					//return;
				}

				populate_xpak_cache_internal(pak_path);
			}

			void populate_xpak_cache_for_loaded_zones()
			{
				int index = *g_zoneCount;
				auto* zone = &g_zones[index];
				auto zone_name = &g_zoneNames[index];

				while (index > 0)
				{
					try_populate_xpak_cache_for_zone(zone_name->name);

					index--;
					zone--;
					zone_name--;
				}
			}

			std::vector<std::uint8_t> get_data(uint64_t key, const unsigned int expected_size)
			{
				for (auto& map : xpak_cache_map)
				{
					if (map.second.contains(key))
					{
						auto obj = map.second[key];
						filesystem::file file(obj.PackageFile);
						file.open("rb", false, false);

						if (!file.get_fp())
						{
							return {};
						}

						std::string buffer;
						buffer.resize(obj.CompressedSize);
						file.seek(obj.Offset, SEEK_SET);
						file.read(buffer.data(), obj.CompressedSize);

						auto data = decompress_xpak_data(buffer, expected_size);
						if (data.size() == expected_size)
						{
							return data;
						}
					}
				}

				return {};
			}
		}

		std::vector<std::uint8_t> get_data_for_xpak_key(uint64_t key, const unsigned int expected_size)
		{
			/*auto zone_count_changed = []() -> bool
			{
				static unsigned int zone_count = 0;
				if (zone_count != *g_zoneCount)
				{
					zone_count = *g_zoneCount;
					return true;
				}
				return false;
			};

			if (zone_count_changed())
			{
				clear_cache();
			}*/

			if (xpak::xpak_cache_map.empty())
			{
				xpak::populate_xpak_cache_iterator("../zone/");
				xpak::populate_xpak_cache_iterator("zone/");

				//xpak::populate_xpak_cache_for_loaded_zones();
			}

			auto data = xpak::get_data(key, expected_size);

			return data;
		}

		void clear_cache()
		{
			xpak::xpak_cache_map.clear();
		}
	}
}