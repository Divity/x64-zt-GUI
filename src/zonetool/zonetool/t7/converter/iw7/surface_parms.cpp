#include <std_include.hpp>
#include "include.hpp"
#include "surface_parms.hpp"

// Sources:
// * the tables: BO3 dedi 0x14114BE70 (79 rows), IW7 0x141BBF008.
// * contents in IW7 collision: a Havok shape tag's collisionFilterInfo is its contents. BO3's clip values
//   without utilityclip (0x400000, no IW7 parm) are exactly stock IW7's (0x30200 clip, 0x31640 nosight,
//   0x336C0 full, 0x2080 shot-only geometry) as found on stock world shape tags.
// * material CRCs: Havok physics material name CRCs from the shipped physics assets and stock world
//   surfaces of the same substance; a type without a matching material gets the default one, as stock
//   world surfaces mostly do.

namespace zonetool::t7
{
	namespace converter::iw7::surface_parms
	{
		namespace
		{
			constexpr std::uint32_t bo3_type_shift = 20;
			constexpr std::uint32_t iw7_type_shift = 19;

			// BO3 surface type -> IW7 surface type, by name; where IW7 has no type of that name the nearest
			// one is noted
			constexpr std::uint8_t iw7_surface_types[40] =
			{
				0,  // none
				1,  // bark
				2,  // brick
				3,  // carpet -> carpet_solid
				4,  // cloth
				5,  // concrete -> concrete_dry
				6,  // dirt
				7,  // flesh
				8,  // foliage -> foliage_dry
				9,  // glass -> glass_pane
				10, // grass -> grass_short
				11, // gravel
				12, // ice -> ice_solid
				13, // metal -> metal_thick
				15, // mud
				16, // paper
				17, // plaster
				18, // rock
				19, // sand
				20, // snow
				21, // water
				22, // wood -> wood_solid
				23, // asphalt -> asphalt_dry
				24, // ceramic
				25, // plastic
				26, // rubber
				35, // cushion
				27, // fruit
				28, // paintedmetal -> metal_painted
				7,  // player: no IW7 type, flesh
				41, // tallgrass -> grass_tall
				29, // riotshield
				43, // metalthin -> metal_thin
				43, // metalhollow: no IW7 type, metal_thin
				14, // metalcatwalk -> metal_grate
				43, // metalcar: no IW7 type, metal_thin
				40, // glasscar -> glass_vehicle
				39, // glassbulletproof -> glass_solid
				50, // watershallow -> water_knee
				47, // bodyarmor: no IW7 type, robot_armor
			};

			// nodamage slick sky noimpact nomarks nodraw nopenetrate nosteps nonsolid nocastshadow portal
			constexpr std::uint32_t shared_surface_flags = 0x1 | 0x2 | 0x4 | 0x10 | 0x20 | 0x80 | 0x100 | 0x2000
				| 0x4000 | 0x40000 | 0x80000000;
			// every BO3 bit outside the type and traversal fields with a table entry: the shared ones plus
			// noReceiveDynamicShadow 0x800, caulk 0x1000, outdoorOccluder 0x10000, onlyCastSunShadow 0x20000,
			// onlyCastShadow 0x80000, mount 0x04000000 (none has an IW7 parm; in IW7 0x1000 is soft, 0x20000
			// nodlight, 0x04000000 mantleOver)
			constexpr std::uint32_t known_surface_flags = shared_surface_flags | 0x800 | 0x1000 | 0x10000 | 0x20000
				| 0x80000 | 0x04000000 | 0x03F00000 | 0x38000000;

			// solid foliage glass water canshootclip clipmissile vehicleclip itemclip sky ai_nosight clipshot
			// playerclip monsterclip mantle nodrop
			constexpr std::uint32_t shared_contents = 0x1 | 0x2 | 0x10 | 0x20 | 0x40 | 0x80 | 0x200 | 0x400 | 0x800
				| 0x1000 | 0x2000 | 0x10000 | 0x20000 | 0x01000000 | 0x80000000;
			constexpr std::uint32_t bo3_playervehicleclip = 0x40000;
			// umbraTarget 0x80000, utilityclip 0x400000 (no IW7 parm), detail 0x08000000 and structural
			// 0x10000000 (compile-time only; IW7 filters them out of every shape tag)
			constexpr std::uint32_t known_contents = shared_contents | bo3_playervehicleclip | 0x80000 | 0x400000
				| 0x08000000 | 0x10000000;

			constexpr std::uint32_t crc_default = 0x1AB7BC33; // concrete, and the dummies' material
			constexpr std::uint32_t crc_wood = 0x0AD71E4E;
			constexpr std::uint32_t crc_brick = 0xD8B39111;
			constexpr std::uint32_t crc_carpet = 0x04E51705;
			constexpr std::uint32_t crc_cloth = 0xCD123193;
			constexpr std::uint32_t crc_dirt = 0x47EEC5DE;
			constexpr std::uint32_t crc_flesh = 0x4724ADF2;
			constexpr std::uint32_t crc_foliage = 0x63A1FDAD;
			constexpr std::uint32_t crc_glass = 0xF728E572;
			constexpr std::uint32_t crc_rock = 0x0103BCE1;
			constexpr std::uint32_t crc_ice = 0x820231A4;
			constexpr std::uint32_t crc_metal = 0xCBF7A6C4;
			constexpr std::uint32_t crc_mud = 0xAE2DE6F5;
			constexpr std::uint32_t crc_paper = 0xA1F93A3B;
			constexpr std::uint32_t crc_plaster = 0x8D07D363;
			constexpr std::uint32_t crc_sand = 0x96309552;
			constexpr std::uint32_t crc_water = 0x00C9A2F0;
			constexpr std::uint32_t crc_asphalt = 0x46FAE51C;
			constexpr std::uint32_t crc_ceramic = 0xF49C81BF;
			constexpr std::uint32_t crc_plastic = 0x4B02BC9D;
			constexpr std::uint32_t crc_rubber = 0xFFD772CD;
			constexpr std::uint32_t crc_fruit = 0x4FE888BA;
			constexpr std::uint32_t crc_painted_metal = 0xE8F3FA9A;
			constexpr std::uint32_t crc_cushion = 0x98C096F9;

			// IW7 surface type -> Havok physics material
			constexpr std::uint32_t material_crcs[64] =
			{
				crc_default, crc_wood, crc_brick, crc_carpet, crc_cloth, crc_default, crc_dirt, crc_flesh, // none..flesh
				crc_foliage, crc_glass, crc_default, crc_rock, crc_ice, crc_metal, crc_metal, crc_mud, // foliage_dry..mud
				crc_paper, crc_plaster, crc_rock, crc_sand, crc_default, crc_water, crc_wood, crc_asphalt, // paper..asphalt_dry
				crc_ceramic, crc_plastic, crc_rubber, crc_fruit, crc_painted_metal, crc_default, crc_default, crc_asphalt, // ..asphalt_wet
				crc_carpet, crc_carpet, crc_default, crc_cushion, crc_default, crc_foliage, crc_glass, crc_glass, // ..glass_solid
				crc_glass, crc_default, crc_ice, crc_metal, crc_default, crc_default, crc_default, crc_default, // ..robot_armor
				crc_default, crc_default, crc_water, crc_water, crc_wood, crc_default, crc_default, crc_default, // ..user_terrain_3
				crc_default, crc_default, crc_default, crc_default, crc_default, crc_default, crc_default, crc_default, // ..code_reserved
			};
		}

		std::uint32_t surface_flags(const std::uint32_t bo3, stats& stats)
		{
			const auto type = (bo3 >> bo3_type_shift) & 0x3F;
			auto flags = bo3 & shared_surface_flags;
			if (type < std::size(iw7_surface_types))
			{
				flags |= static_cast<std::uint32_t>(iw7_surface_types[type]) << iw7_type_shift;
			}
			else
			{
				stats.unknown_types[type]++;
			}

			// traversal: ladder 1, mantleOn 2, mantleOver 3, climbWall 4, climbPipe 5
			switch ((bo3 >> 27) & 7)
			{
			case 1:
				flags |= 0x8;
				break;
			case 2:
				flags |= 0x02000000;
				break;
			case 3:
				flags |= 0x04000000;
				break;
			case 4:
			case 5:
				stats.climb++;
				break;
			default:
				break;
			}

			stats.unknown_surface_bits |= bo3 & ~known_surface_flags;
			return flags;
		}

		std::uint32_t contents(const std::uint32_t bo3, stats& stats)
		{
			auto out = bo3 & shared_contents;
			if (bo3 & bo3_playervehicleclip)
			{
				out |= 0x200; // vehicleclip
			}
			stats.unknown_contents_bits |= bo3 & ~known_contents;
			return out;
		}

		std::uint32_t material_crc(const std::uint32_t iw7_flags)
		{
			return material_crcs[(iw7_flags >> iw7_type_shift) & 0x3F];
		}

		void report(const stats& stats, const char* who)
		{
			for (const auto& [type, count] : stats.unknown_types)
			{
				ZONETOOL_WARNING("%s: %u surfaces use BO3 surface type %u, which is not in BO3's table", who, count, type);
			}
			if (stats.unknown_surface_bits)
			{
				ZONETOOL_WARNING("%s: BO3 surface flags 0x%08X are not in BO3's table and were dropped", who, stats.unknown_surface_bits);
			}
			if (stats.unknown_contents_bits)
			{
				ZONETOOL_WARNING("%s: BO3 contents 0x%08X are not in BO3's table and were dropped", who, stats.unknown_contents_bits);
			}
			if (stats.climb)
			{
				ZONETOOL_WARNING("%s: %u surfaces are BO3 climb surfaces, which IW7 has no parm for", who, stats.climb);
			}
		}
	}
}
