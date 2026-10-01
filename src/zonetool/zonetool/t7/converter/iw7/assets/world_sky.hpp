#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::world_sky
	{
		struct sky
		{
			std::string material; // IW7 w_sky material
			std::string image;    // IW7 BC6H cube
			unsigned char sort_key = 0;      // the material's (stock w_sky)
			unsigned char sampler_state = 0; // the cube's (GfxSky::skySamplerState)
		};

		// BO3 draws lighting state 0's skybox (GfxSunVolume::skyboxes) with sky_latlong_hdr: a latlong
		// panorama sampled along the view direction turned about z. IW7 draws a world surface with w_sky, a cube
		// sampled along the view direction. Writes the cube (the panorama resampled with BO3's turn) and the
		// material under the dump folder; nothing when the map has no skybox.
		std::optional<sky> convert(const GfxWorld* asset);

		// per IW7 surface: 1 for the sky's (GfxSky::skyStartSurfs holds their sorted slots)
		std::vector<std::uint8_t> surfaces(const zonetool::iw7::GfxWorld* world);
	}
}
