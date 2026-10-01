#pragma once

#include <map>

namespace zonetool::t7
{
	namespace converter::iw7::surface_parms
	{
		// Both games ship the map compiler's infoParm table ({name, clearSolid, surfaceFlags, contents,
		// toolFlags}; BO3 dedi 0x14114BE70, IW7 0x141BBF008). BO3 keeps a 6-bit surface type at bits 20..25
		// (40 types), IW7 a 6-bit type at bits 19..24 (64 types); they are matched by name. Every other
		// surface and contents bit means the same in both except the ones handled here.

		// userData bit 48 of a Havok shape tag: the primitive came from a brush (stock sets it on
		// brush-derived tags)
		constexpr std::uint64_t user_data_brush = 1ull << 48;

		constexpr std::uint32_t bo3_caulk = 0x1000;
		constexpr std::uint32_t bo3_nodraw = 0x80;

		// what a conversion ran into that BO3's own table does not list
		struct stats
		{
			std::map<std::uint32_t, unsigned int> unknown_types;
			std::uint32_t unknown_surface_bits = 0;
			std::uint32_t unknown_contents_bits = 0;
			unsigned int climb = 0;
		};

		// BO3 surface flags -> IW7 surface flags
		std::uint32_t surface_flags(std::uint32_t bo3, stats& stats);

		// BO3 contents -> IW7 contents (0: nothing collides with it in IW7)
		std::uint32_t contents(std::uint32_t bo3, stats& stats);

		// the Havok physics material of an IW7 surface type
		std::uint32_t material_crc(std::uint32_t iw7_flags);

		// warns about everything `stats` holds, prefixed by `who`
		void report(const stats& stats, const char* who);
	}
}
