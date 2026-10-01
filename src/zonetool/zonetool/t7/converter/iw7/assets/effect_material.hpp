#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::effect_material
	{
		// How an IW7 effect material shades a particle, which decides what the particle converter gives it
		// (particle_system.cpp). BO3's effect templates (ei/effect_*, ec/effect_*) map onto stock IW7 particle techsets:
		enum class shading
		{
			// ei/effect_lit_blend: eq_effect_zfeather_blend_tab_lin_ndw_nocast, lit by IW7's particle lighting
			lit,
			// ei/effect_lit_emissive_blend on elements whose vertex colour is grey: eq_effect_zfeather_blend_tab_lin_ndw_nocast_evc_em,
			// lit plus an emissive map (colour x mask) scaled by the particle's emissive value
			emissive_colour,
			// ei/effect_lit_emissive_blend on elements with a coloured vertex colour: eq_effect_zfeather_blendadd_tab_lin_ndw_nocast_evc_emm,
			// the emission is mask x colour x vertex colour x the particle's emissive value, the lit part x (1 - mask)
			emissive_mask,
			// ei/effect_distortion: eq_distortion_scale_zfeather
			distortion,
			// ec/effect_cloud_*: particle_cloud_blend_lin
			cloud,
		};

		struct converted
		{
			std::string name; // IW7 material; empty when the BO3 material does not convert
			shading shade = shading::lit;
			bool old_hdr_scale = false; // BO3 useOldHDRScale: the emission is the material's hdrScale alone
			// BO3's atlas range mode: the material's atlas is the element's frame range in play order, `atlas_range` BO3
			// frames over `atlas_slots` IW7 frames (a power of two); 0: the BO3 atlas as it is
			std::uint32_t atlas_slots = 0;
			std::uint32_t atlas_range = 0;
		};

		// The IW7 material a BO3 effect element draws `material` with (its type, depth feather and vertex colour pick the
		// variant), queued for write_all. The same BO3 material can become several IW7 materials.
		converted request(const Material* material, const FxElemDef* elem);

		// the IW7 image and material names every requested material writes, for the zone rows
		std::vector<std::string> requested_materials();

		// writes every requested material and its images (once the map's other materials are written)
		void write_all();

		void clear();
	}
}
