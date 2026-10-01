#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "zonetool/t7/converter/iw7/memory_probe.hpp"
#include "effect_material.hpp"
#include "world_material.hpp"
#include "world_material_bake.hpp"

#include "zonetool/t7/functions.hpp"
#include "zonetool/t7/converter/iw7/map_common.hpp"

#include <utils/string.hpp>

// BO3 effect materials as stock IW7 particle materials.
//
// BO3's effect templates multiply a premultiplied sRGB colour map by the linear vertex colour and the particle lighting,
// and feather by scene depth over zFeather units (FxElemDef +0x234). IW7's particle shaders square texture x vertex
// colour (gamma 2) and take the feather as featherParms.x = 1 / depth, so each feather depth is its own IW7 material.
// Emissive variants: _evc_em (blend) bakes the emissive map as colour x mask and drops the vertex colour's hue, used where
// it is grey; _evc_emm (blendadd) bakes mask x texture alpha and keeps the hue, at the cost of the lit part under the mask.
// emissiveParams.x is hdrScale x map::bo3_light_scale; TONEMAP_PARMS.x is 1 with r_EVCompBounds 0.

namespace zonetool::t7
{
	namespace converter::iw7::effect_material
	{
		namespace
		{
			namespace bake = world_material::bake;

			constexpr std::uint32_t slot_colour = 0xA0AB1041; // colorMap
			constexpr std::uint32_t slot_emissive = 0x34614347; // emissiveMap
			constexpr std::uint32_t constant_feather = 1300144692; // featherParms
			constexpr std::uint32_t constant_emissive = 3944623239; // emissiveParams
			constexpr std::uint32_t constant_distortion = 4084951315; // distortionScale

			// BO3's atlas behaviour (0x140207A40): the start frame mode (3: a range), blend between frames, reversed, used
			constexpr unsigned char atlas_start_mask = 0x3;
			constexpr unsigned char atlas_start_range = 0x3;
			constexpr unsigned char atlas_blend = 0x10;
			constexpr unsigned char atlas_reverse = 0x20;
			constexpr unsigned char atlas_enabled = 0x80;

			std::mutex mutex;
			std::map<std::string, bake::effect_material_def> queue; // by IW7 name
			std::unordered_map<const Material*, std::optional<bake::effect_globals>> globals; // empty: they do not read

			// the BO3 template of an effect material: its technique set's name without folder or hash
			std::string effect_template(const Material* material)
			{
				return bake::bo3_template(material);
			}

			// the BO3 name with its folder (materials of the same name live in different folders) as one word
			std::string clean(const std::string& name)
			{
				auto out = name;
				for (auto& c : out)
				{
					if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
					{
						c = '_';
					}
				}
				return out;
			}

			// the element's vertex colour is grey over its whole life, both ends of every random range
			bool grey_vertex_colour(const FxElemDef* elem)
			{
				const auto count = elem->visStateIntervalCount + 1;
				const auto* samples = reinterpret_cast<const std::uint8_t*>(elem->visSamples);
				if (!probe::readable(samples, static_cast<std::size_t>(0x50) * count))
				{
					return false;
				}
				for (auto i = 0; i < count; i++)
				{
					for (const auto half : { 0x00, 0x28 })
					{
						const auto* c = samples + 0x50 * i + half;
						if (std::abs(c[0] - c[1]) > 1 || std::abs(c[1] - c[2]) > 1)
						{
							return false;
						}
					}
				}
				return true;
			}

			std::string feather_suffix(const float depth)
			{
				if (depth <= 0.0f)
				{
					return "_zf0";
				}
				std::string text = utils::string::va("%g", depth);
				for (auto& c : text)
				{
					if (c == '.')
					{
						c = 'p';
					}
				}
				return "_zf" + text;
			}

			// featherParms (1 / depth, depth, depth, as stock); BO3's zFeather 0 feathers over 1 / 60000 units
			std::array<float, 4> feather(const float depth)
			{
				const auto d = depth > 0.0f ? depth : 1.0f / 60000.0f;
				return { 1.0f / d, d, d, std::numeric_limits<float>::quiet_NaN() };
			}

			// the material's constants (its lit shader's $Globals); null, said once, when they do not read
			const bake::effect_globals* globals_of(const Material* material)
			{
				auto found = globals.find(material);
				if (found == globals.end())
				{
					std::optional<bake::effect_globals> g;
					try
					{
						g = bake::read_effect_globals(material);
					}
					catch (const std::exception& e)
					{
						ZONETOOL_WARNING("effect material %s: its constants do not read (%s); its elements are left out", material->name,
							e.what());
					}
					found = globals.emplace(material, g).first;
				}
				return found->second ? &*found->second : nullptr;
			}
		}

		converted request(const Material* material, const FxElemDef* elem)
		{
			converted out{};
			if (!material || !probe::readable(material, sizeof(Material)) || !probe::terminated(material->name))
			{
				return out;
			}
			// the zone's own copy when the element holds a reference placeholder
			if (const auto* entry = DB_FindXAssetEntry(ASSET_TYPE_MATERIAL, material->name, false);
				entry && probe::readable(entry->asset.header.material, sizeof(Material)))
			{
				material = entry->asset.header.material;
			}
			if (!world_material::readable(material->techniqueSet, sizeof(MaterialTechniqueSet)) || !material->techniqueSet->name)
			{
				ZONETOOL_WARNING("effect material %s: its technique set is not loaded; the element is left out", material->name);
				return out;
			}

			const auto tmpl = effect_template(material);
			const auto base = clean(material->name);
			const auto cloud_element = elem->elemType == 6;
			if (elem->elemType == 5)
			{
				ZONETOOL_WARNING("effect material %s: geo trails need IW7's ev_ techsets, which this converter does not write yet; "
					"the element is left out", material->name);
				return out;
			}

			std::lock_guard _(mutex);
			const auto* globals_read = globals_of(material);
			if (!globals_read)
			{
				return out;
			}
			const auto& g = *globals_read;
			bake::effect_material_def def{};
			def.source = material;
			def.frame_blend = (elem->atlas.behavior & atlas_blend) != 0;
			const auto blend_suffix = def.frame_blend ? "_fb"s : ""s;

			// BO3's atlas range mode (0x140207A40): the element plays frames index .. index + range - 1 of its atlas (wrapping
			// over it), mirrored over the whole atlas when reversed. The IW7 material's atlas holds them in play order. IW7
			// rounds an atlas up to a power of two of frames (INIT_ATLAS 0x140D0F950), so a range of another length is
			// spread over the next power of two (each IW7 frame shows the range's frame at the same point of its loop).
			auto key = base;
			const auto& atlas = elem->atlas;
			// BO3 draws an atlas by the element's bits (0x140207A40), the IW7 material by its own grid
			if ((atlas.behavior & atlas_enabled) && ((1u << atlas.colIndexBits) != std::max<unsigned>(1, material->info.textureAtlasColumnCount) ||
				(1u << atlas.rowIndexBits) != std::max<unsigned>(1, material->info.textureAtlasRowCount)))
			{
				ZONETOOL_WARNING("effect material %s: the element's atlas is %ux%u frames, the material's %ux%u", material->name,
					1u << atlas.colIndexBits, 1u << atlas.rowIndexBits, material->info.textureAtlasColumnCount,
					material->info.textureAtlasRowCount);
			}
			// BO3's compute sprite path draws the element's compute visual
			if (const auto* cm = static_cast<const Material*>(elem->computeVisuals.instance.anonymous);
				elem->visualCount == 1 && cm && cm != material && probe::readable(cm, sizeof(Material)) && probe::terminated(cm->name) &&
				std::strcmp(cm->name, material->name))
			{
				ZONETOOL_INFO("effect material %s: its element's compute visual is %s", material->name, cm->name);
			}
			if ((atlas.behavior & atlas_enabled) && (atlas.behavior & atlas_start_mask) == atlas_start_range && atlas.indexRange &&
				!cloud_element)
			{
				const auto total = 1u << (atlas.colIndexBits + atlas.rowIndexBits);
				const auto range = static_cast<std::uint32_t>(atlas.indexRange);
				auto slots = 1u;
				while (slots < range)
				{
					slots <<= 1;
				}
				const auto reverse = (atlas.behavior & atlas_reverse) != 0;
				for (auto s = 0u; s < slots; s++)
				{
					const auto frame = (atlas.index + s * range / slots) & (total - 1);
					def.frames.push_back(reverse ? total - 1 - frame : frame);
				}
				def.source_columns = 1u << atlas.colIndexBits;
				def.source_rows = 1u << atlas.rowIndexBits;
				out.atlas_slots = slots;
				out.atlas_range = range;
				key += utils::string::va("_a%u_%u%s", atlas.index, range, reverse ? "r" : "");
			}
			const auto colour = [&](const std::string& image_suffix)
			{
				return bake::effect_image{ slot_colour, key + image_suffix, bake::effect_texel::colour };
			};

			if (tmpl.starts_with("effect_cloud"))
			{
				if (!cloud_element)
				{
					ZONETOOL_WARNING("effect material %s: a cloud material on a sprite element; the element is left out", material->name);
					return out;
				}
				out.shade = shading::cloud;
				def.name = "ec/t7_" + key;
				def.techset = "particle_cloud_blend_lin";
				def.images = { colour("_t7fc") };
			}
			else if (tmpl == "effect_distortion")
			{
				out.shade = shading::distortion;
				def.name = "eq/t7_" + key + blend_suffix;
				def.techset = "eq_distortion_scale_zfeather";
				def.constants.push_back({ constant_distortion, { g.distortion_scale[0], g.distortion_scale[1], 0.0f, 0.0f } });
				def.images = { { slot_colour, key + "_t7fd", bake::effect_texel::distortion } };
			}
			else if (tmpl.starts_with("effect_lit_emissive_blend"))
			{
				const auto grey = grey_vertex_colour(elem);
				out.shade = grey ? shading::emissive_colour : shading::emissive_mask;
				out.old_hdr_scale = g.old_hdr_scale;
				def.name = "eq/t7_" + key + (grey ? "_em" : "_emm") + blend_suffix + feather_suffix(elem->zFeather);
				def.techset = grey ? "eq_effect_zfeather_blend_tab_lin_ndw_nocast_evc_em" : "eq_effect_zfeather_blendadd_tab_lin_ndw_nocast_evc_emm";
				def.constants.push_back({ constant_feather, feather(elem->zFeather) });
				def.constants.push_back({ constant_emissive, { g.hdr_scale * map::bo3_light_scale, 0.0f, 0.0f, 0.0f } });
				def.images = { colour("_t7fc"), grey
					? bake::effect_image{ slot_emissive, key + "_t7fe", bake::effect_texel::emissive_colour }
					: bake::effect_image{ slot_emissive, key + "_t7fm", bake::effect_texel::emissive_mask } };
			}
			else if (tmpl.starts_with("effect_lit_blend") || tmpl.starts_with("effect_lit_specular_blend"))
			{
				out.shade = shading::lit;
				def.name = "eq/t7_" + key + blend_suffix + feather_suffix(elem->zFeather);
				def.techset = "eq_effect_zfeather_blend_tab_lin_ndw_nocast";
				def.constants.push_back({ constant_feather, feather(elem->zFeather) });
				def.images = { colour("_t7fc") };
			}
			else
			{
				ZONETOOL_WARNING("effect material %s: BO3 template %s has no IW7 mapping; the element is left out", material->name,
					tmpl.data());
				return out;
			}

			out.name = def.name;
			if (queue.emplace(def.name, std::move(def)).second)
			{
				// what the IW7 material approximates, said once
				if (tmpl.starts_with("effect_lit_specular_blend"))
				{
					ZONETOOL_WARNING("effect material %s: BO3's reflection-probe specular (normal map, gloss) is left out; lit blend",
						out.name.data());
				}
				if (out.atlas_slots != out.atlas_range)
				{
					ZONETOOL_WARNING("effect material %s: its element plays %u atlas frames, spread over %u (IW7 counts a power of two "
						"of frames); each shows the frame BO3 shows at the start of its share of the loop", out.name.data(),
						out.atlas_range, out.atlas_slots);
				}
			}
			return out;
		}

		std::vector<std::string> requested_materials()
		{
			std::lock_guard _(mutex);
			std::vector<std::string> names;
			for (const auto& [name, def] : queue)
			{
				names.push_back(name);
			}
			return names;
		}

		void write_all()
		{
			std::lock_guard _(mutex);
			bake::worker worker;
			auto written = 0u;
			for (const auto& [name, def] : queue)
			{
				try
				{
					bake::write_effect_material(def, worker);
					written++;
				}
				catch (const std::exception& e)
				{
					ZONETOOL_ERROR("effect material %s (%s): not written: %s", name.data(), def.source->name, e.what());
				}
			}
			ZONETOOL_INFO("effect materials: %u of %zu written", written, queue.size());
		}

		void clear()
		{
			std::lock_guard _(mutex);
			queue.clear();
			globals.clear();
		}
	}
}
