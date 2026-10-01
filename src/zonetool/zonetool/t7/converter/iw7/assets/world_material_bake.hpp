#pragma once

#include "world_material.hpp"
#include "material_texture.hpp"

namespace zonetool::t7
{
	namespace converter::iw7::gpu_eval
	{
		class context;
	}

	namespace converter::iw7::world_material::bake
	{
		// Where a pixel shader samples a texture, from base texture coordinates (u, v): (m[0] u + m[1] v + m[2],
		// m[3] u + m[4] v + m[5]) and the instruction's texel offsets.
		struct sample_map
		{
			float m[6] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
			std::int32_t offset[2] = { 0, 0 };
		};

		// How one BO3 world material is converted, decided from its pixel shader, textures,
		// constants and render state before anything is evaluated.
		struct plan
		{
			bool supported = false;
			std::string reason; // why not, when unsupported

			std::string techset; // IW7 techset
			std::uint32_t uv_period[2] = { 1, 1 };
			// the base texture coordinates the bake covers: [uv_origin, uv_origin + uv_span) per axis, a whole period
			// from 0 unless the material's surfaces use less of it (info::used_uv)
			float uv_origin[2] = { 0.0f, 0.0f };
			float uv_span[2] = { 1.0f, 1.0f };
			bool windowed = false; // an axis covers only the used area
			std::string size_from; // which texture sets the density along u and v (its tiling rate times its size)
			std::uint32_t width = 0; // baked texture size (mip 0)
			std::uint32_t height = 0;
			std::uint32_t reduced = 0; // times the size was halved to fit a streamed part (below BO3's texel density)

			bool alpha_test = false;
			bool decal = false;
			// a forward-lit transparent material (no gbuffer technique): the surface comes from its lit technique's
			// debug-override permutation cut to its surface values (forward_surface), its alpha is the coverage; with
			// forward_emissive the emissive, and the alpha BO3 draws with, come from the lit shader run without lighting
			bool surface = false;
			std::string surface_sources; // the surface values not labelled: read at the forward decal blend, or constants
			// a forward surface drawn opaque (BO3's forward-lit skin): no alpha, no coverage
			bool opaque_surface = false;
			// BO3 skin: IW7's subsurface scattering (sss) techsets; gtao: the material's enableGTAO, BO3's screen-space
			// GTAO in place of its aoMap (IW7's _gtao skin techset without an occlusion map)
			bool skin = false;
			bool gtao = false;
			// BO3's additive emissive templates (decal_emissive): the lit shader adds nothing but its emission; IW7's unlit
			// add (mkhdr) techset with that emission baked
			bool unlit_emissive = false;
			// BO3's eye: IW7's eye techset (write_eye), its colour map and iris constants converted, nothing baked
			bool eye = false;
			// BO3's water (lit_water_sim_flow_normal*, water_shore_flow*): IW7's refractive water, on models its UV-animated
			// lit water (write_water)
			bool water = false;
			// a decal whose RT0 multiplies the albedo under it (ZERO + SRC_COLOR: wet decals, raindrops) or does
			// not write it (normal-only decals): what RT0 does to the albedo, as IW7's unlit multiply of the lit result (its
			// stock stains' technique): BO3's RT0 at vertex alpha 1, faded by the vertex alpha at run time. IW7's forward
			// decals cannot change the lighting inputs (normal, gloss, specular) of the surface under them, so the rest is
			// not drawn. identity: RT0 not written, a multiply by 1 (nothing drawn)
			bool multiply = false;
			bool identity = false;
			// a model decal whose coverage (or reveal) BO3 takes from a vertex alpha its surfaces have below 1: IW7's vertex
			// colour decal techset (mco_) applies it at run time, and the surfaces' vertex rgb is written white (BO3's decal
			// reads none of it, IW7's multiplies the albedo by it)
			bool vertex_alpha = false;

			// a forward surface whose alpha is its coverage: drawn blended, or alpha-tested (BO3's hair_alphatest discards
			// below 0.5, IW7's alpha test threshold)
			bool surface_coverage() const
			{
				return this->surface && (!this->opaque_surface || this->alpha_test);
			}
			bool premultiplied = false; // BO3 blends it ONE / INV_SRC_ALPHA (IW7 blendadd), else SRC_ALPHA / INV_SRC_ALPHA
			// a world transparent surface revealed by its vertex alpha (reveal): IW7 reveals by the vertex alpha only in its
			// decal techsets (v0), so it is drawn as IW7's world reveal decal, in the decal region after the world's decals
			bool reveal_decal = false;
			// a forward-lit emissive material (no gbuffer technique): the surface comes from the gbuffer shader of
			// `base_techset` (its _emissive-less sibling) fed the material's own constants and textures, the
			// emissive from the material's lit shader run without lighting
			bool forward_emissive = false;
			std::string base_techset;
			bool reveal = false;
			// the coverage (alpha test, decal alpha) and reveal textures are baked from where the shader samples them
			sample_map coverage_map;
			sample_map reveal_map;
			// an alpha test on more than one texture channel: its coverage evaluated from the pixel shader cut at its discard
			// (forward_surface::coverage) instead of sampled from one texture
			bool coverage_program = false;
			bool detail = false; // BO3 detailMap converted to an IW7 detail normal (q0)
			bool flat_detail = false; // the IW7 techset takes a detail normal (q0) the BO3 material does not have: a flat one
			bool occlusion = false; // the material writes a non-constant occlusion
			bool coloured_specular = false; // F0 is not the constant 0.04

			// decals: which gbuffer channels BO3 writes (the technique's per-target write masks);
			// the rest keep the surface underneath, which an IW7 forward decal cannot do
			bool writes_normal = true;
			bool writes_gloss = true;
			bool writes_specular = true;
			bool writes_occlusion = true;
		};

		// `inf` is the material's classification (its class decides whether and how it is baked)
		const plan& get_plan(const Material* material, const info& inf);

		// The texture channel an alpha-tested material's gbuffer program discards by (what the bake's coverage reads), for
		// surfaces that block light only where they are drawn; nullopt when it does not alpha test (or its coverage is not
		// one texture channel). Cached; records no plan.
		struct alpha_mask
		{
			std::shared_ptr<const material_texture::decoded> image;
			std::uint32_t channel = 3;
			material_texture::address_mode u = material_texture::address_mode::wrap;
			material_texture::address_mode v = material_texture::address_mode::wrap;
			sample_map map; // where the shader samples it from the surface's texture coordinates
		};
		std::optional<alpha_mask> alpha_mask_of(const Material* material);

		// the BO3 template a material's technique set is made from: its name without the folder and the #hash
		// (mc/skin#e6142445 -> skin); empty without a named technique set
		std::string bo3_template(const Material* material);

		// a forward (not gbuffer) decal technique set whose lit technique adds its output (SRC_ALPHA or ONE onto ONE) with a
		// polygon offset: BO3's decal_emissive, which IW7 draws as its stock emissive overlays do (unlit add, decal region)
		bool additive_decal(const Material* material);

		// Gloss (BO3 g) given to decal texels whose BO3 material does not write gloss: IW7 lights a
		// decal with its own gloss, BO3 kept the gloss of the surface under it.
		void set_underlying_gloss(float gloss);

		// the median upper gloss (glossRange.y / 17) of the opaque world materials
		float estimate_underlying_gloss(const std::vector<const Material*>& materials);

		// What one bake thread owns: the GPU device its materials are evaluated and block-compressed on (none
		// without a hardware device; both then run on the CPU). Materials bake on several workers at once.
		struct worker
		{
			worker();
			~worker();

			std::unique_ptr<gpu_eval::context> gpu;
			const void* bound = nullptr; // the evaluator whose program is bound on `gpu`
			// the first tile packed on `gpu` is also packed on the CPU: 0 not yet, 1 they agree (texels are packed on the
			// GPU), -1 they differ (texels are packed on the CPU); [0] gbuffer targets, [1] a forward surface's raw targets
			int pack_checked[2] = { 0, 0 };
		};

		// bakes and writes everything for one material; returns false when it was not converted
		bool write_material(const Material* material, const info& inf, worker& w);

		// ---- model material atlases ----------------------------------------------------------------------
		// IW7 stops loading a level whose zones add up to more than 15616 images (the image pool, "Exceeded limit of
		// 15616 'image' assets"; its entries are addressed by index, 0x1403B7590, so the pool cannot grow). A model
		// material whose surfaces stay inside the area its bake covers (none tiles it) shares its cs / ng images with
		// others of the same tile size: each bakes into its tile and its surfaces' texture coordinates are moved there
		// (info::uv_origin, uv_span).
		struct atlas_slot
		{
			std::string cs; // the atlas images
			std::string ng;
			std::uint32_t columns = 1;
			std::uint32_t rows = 1;
			std::uint32_t column = 0;
			std::uint32_t row = 0;
			std::uint32_t tile_width = 0;
			std::uint32_t tile_height = 0;
		};

		// the tile a model material's bake takes in an atlas (its bake size rounded up to a size atlases share), or
		// false when it cannot share one
		bool atlas_tile(const Material* material, const info& inf, std::uint32_t& width, std::uint32_t& height);

		// bakes the material at the slot's tile size into the slot; before the material is baked
		void set_atlas(const Material* material, const info& inf, const atlas_slot& slot);

		// writes the atlases a failed material left incomplete (its tile empty); after every material is baked
		void flush_atlases();

		// writes a material of a stock IW7 techset from its donor (constants, state, flags, sort key) with the
		// given images by texture type hash (the sky's w_sky)
		void write_stock_material(const std::string& name, const std::string& techset,
			const std::vector<std::pair<std::uint32_t, std::string>>& images);

		// ---- effect materials (effect_material.cpp decides what each one becomes) ----------------------------

		// the $Globals of a BO3 effect material's lit technique that its conversion reads
		struct effect_globals
		{
			float hdr_scale = 1.0f; // lit emissive templates: the emission's scale
			bool old_hdr_scale = false; // useOldHDRScale: the emission is hdrScale alone, not scaled per particle
			float desaturation = 0.0f; // desaturationAmount
			std::array<float, 4> levels = { 0.0f, 1.0f, 0.0f, 1.0f }; // levelsControls: input min, max, output min, max
			std::array<float, 2> distortion_scale = { 0.0f, 0.0f }; // distortion template
		};
		effect_globals read_effect_globals(const Material* material);

		// How an effect texture's texels are made from the BO3 material's colour map (BO3 samples it as sRGB,
		// premultiplied) and emissive mask. IW7's particle shaders square the colour they sample (and the vertex colour).
		enum class effect_texel
		{
			colour, // the colour unpremultiplied, desaturation and levels applied as BO3's shader does, rgb as its
			        // square root; alpha kept, rgb of transparent texels filled from their neighbours
			emissive_colour, // square root of colour x emissive mask; alpha 1
			emissive_mask, // emissive mask x alpha in every channel
			distortion, // the colour map's values as BO3's shader reads them (sRGB decoded), linear
		};

		struct effect_image
		{
			std::uint32_t slot = 0; // IW7 texture type hash
			std::string name; // IW7 image
			effect_texel texel = effect_texel::colour;
		};

		struct effect_material_def
		{
			std::string name; // IW7 material
			std::string techset; // stock IW7 techset with a donor
			const Material* source = nullptr;
			// by name hash; the donor's other constants keep their values (textureAtlas: the BO3 material's rows and columns)
			std::vector<std::pair<std::uint32_t, std::array<float, 4>>> constants;
			std::vector<effect_image> images;
			bool frame_blend = false; // IW7 blends atlas frames (textureAtlasFrameBlend): BO3's atlas behaviour 0x10
			// the frames the IW7 atlas holds, in its order, as indices into the BO3 atlas's grid of `source_columns` x
			// `source_rows` frames (BO3's atlas range mode; a power of two of them); empty: the BO3 atlas as it is
			std::vector<std::uint32_t> frames;
			std::uint32_t source_columns = 1;
			std::uint32_t source_rows = 1;
		};

		// writes an effect material: its images made from the BO3 material's maps, the donor's state, constant layout and
		// flags, the BO3 material's atlas
		void write_effect_material(const effect_material_def& def, worker& w);

		// bytes of converted image data written so far
		std::size_t image_bytes();

		// thread-seconds per bake stage since the last report, for the summary
		std::string stage_report();

		void clear();
	}
}
