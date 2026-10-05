#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::map
	{
		// BO3 -> IW7 light units. BO3's auto exposure leaves its radiance scale arbitrary; IW7's vision
		// exposure only works inside the range stock maps light to (stock cp_zmb lightmap luminance is the
		// reference). Every BO3 radiance is multiplied by it, so their ratios stay BO3's.
		constexpr float bo3_light_scale = 6.572e-4f;

		// MapEnts client dynents (script-spawned scriptable slots); GfxWorld dpvsDyn must match
		constexpr unsigned short reserved_dynents = 64;

		// The BO3 lighting state the converted map is frozen in (ZT_LIGHTING_STATE, 0-3, default 0): its lights, sun,
		// probes and sky. A map whose script switches state (power on) is converted in the state it plays in.
		unsigned int lighting_state();

		// The sun volume the level plays in: the one with planes whose exposure grids hold the most probes (a map may keep
		// a leftover volume first: zm_waterparkfinale's volume 0 is a moonlit 2-probe box, its park volume 1). Its sun,
		// sky and look are the map's.
		unsigned int main_sun_volume(const GfxWorld* world);

		// The IW7 map name: "maps/zm/zm_<name>.d3dbsp" -> "cp_<name>" (a zombies map is an IW7 cp_ map)
		std::string map_name(const std::string& t7_bsp_name);

		// a BO3 zombies (zm_) map: converted to IW7 zombies (CP)
		bool zombies_map(const std::string& t7_bsp_name);

		// The name IW7 asks for when told to load `map`: Com_GetBspFilename (iw7_ship 0x140CDB7B0)
		// formats "maps/%s%s.d3dbsp" with "cp/" for cp_ maps, "mp/" for mp_ maps and nothing
		// otherwise.
		std::string bsp_name(const std::string& t7_bsp_name);

		// The map assets reference each other (the GfxWorld needs the converted light list, the
		// sun comes from the GfxWorld, ...), and a zone adds them in its own order. They are
		// collected as they load and converted together once the zone has finished loading.
		// Returns true when the asset type is one of them.
		bool remember(XAssetType type, void* header);
		void convert_pending();
	}
}
