#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace world_material
		{
			// Where a BO3 world material's surfaces end up in IW7. IW7 keeps the static surfaces in
			// four contiguous ranges picked by the material sort key (opaque 1-6, decal 7-17,
			// trans 18-34, emissive 35-40); shadow-only surfaces sit in the opaque range with
			// camera region 11, as w/shadowcaster does in every stock map.
			enum class surface_class
			{
				opaque,
				decal,
				trans,
				emissive,
				shadow_only,
			};

			// the texture coordinates (BO3 texture units) a model material's surfaces use
			struct uv_bounds
			{
				float min[2] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
				float max[2] = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

				bool empty() const
				{
					return this->min[0] > this->max[0] || this->min[1] > this->max[1];
				}

				void add(const float u, const float v)
				{
					this->min[0] = std::min(this->min[0], u);
					this->min[1] = std::min(this->min[1], v);
					this->max[0] = std::max(this->max[0], u);
					this->max[1] = std::max(this->max[1], v);
				}

				void add(const uv_bounds& other)
				{
					if (!other.empty())
					{
						this->add(other.min[0], other.min[1]);
						this->add(other.max[0], other.max[1]);
					}
				}
			};

			// the vertex colours (the BO3 stream's 8-bit RGBA) a model material's surfaces use
			struct colour_range
			{
				std::uint8_t min[4] = { 255, 255, 255, 255 };
				std::uint8_t max[4] = { 0, 0, 0, 0 };

				bool empty() const
				{
					return this->min[0] > this->max[0];
				}

				bool uniform() const
				{
					return !this->empty() && std::equal(this->min, this->min + 4, this->max);
				}

				void add(const std::uint8_t* rgba)
				{
					for (auto c = 0; c < 4; c++)
					{
						this->min[c] = std::min(this->min[c], rgba[c]);
						this->max[c] = std::max(this->max[c], rgba[c]);
					}
				}

				void add(const colour_range& other)
				{
					if (!other.empty())
					{
						this->add(other.min);
						this->add(other.max);
					}
				}
			};

			// what a model material's surfaces use of their vertex stream
			struct surface_usage
			{
				uv_bounds uv;
				colour_range colour;

				void add(const surface_usage& other)
				{
					this->uv.add(other.uv);
					this->colour.add(other.colour);
				}
			};

			struct info
			{
				std::string name; // the IW7 material
				// an XModel material: IW7 model techsets (mo_), BO3's model vertex inputs, and its textures
				// named apart from the world's
				bool model = false;
				bool techset_loaded = true; // false: its technique set is in a zone that was not loaded
				surface_class cls = surface_class::opaque;
				unsigned char sort_key = 2;
				unsigned char camera_region = 0;
				bool casts_shadow = false;
				bool lightmapped = true;
				bool alpha_test = false; // the converted material discards texels (not an occluder)
				// The converted material's textures cover this many BO3 texture-space units along
				// u and v (a baked composite of textures that tile at different rates repeats only
				// after several units), so the surfaces' uv0 is divided by it. 1 for almost all.
				std::uint32_t uv_period[2] = { 1, 1 };
				// A model material whose surfaces use less than a period bakes only that area: the converted
				// texture coordinates are (uv - uv_origin) / uv_span. World materials: 0 and the period. BO3's water
				// (bake plan water): the span is 1 / normalMapScale, the texture space BO3 samples its maps in.
				float uv_origin[2] = { 0.0f, 0.0f };
				float uv_span[2] = { 1.0f, 1.0f };
				// model materials: the texture coordinates and vertex colours their surfaces use, when known before planning
				std::optional<uv_bounds> used_uv;
				std::optional<colour_range> used_colour;
				// a material whose IW7 techset reads the vertex colour where BO3's reads only its alpha (bake plan vertex_alpha,
				// multiply) or none of it (model water): its surfaces' vertex rgb is written white (xmodel_mesh, gfxworld)
				bool vertex_alpha = false;
			};

			// A material can reference assets of a zone that was not loaded, which leaves those pointers
			// unresolved.
			bool readable(const void* ptr, std::size_t size);

			// Scans every material the world's surfaces use. Decal sort keys depend on which BO3
			// decal layers the map uses, so this has to run before get().
			void prepare(const GfxWorld* world, const std::vector<const Material*>& world_props = {});

			// the texture coordinates the world surfaces of placed models use, for a material no BO3 world surface uses: its
			// world bake covers that area (a composite that repeats after no whole number of units bakes only there).
			// After prepare(), before get() classifies the material
			void set_world_usage(const Material* material, const uv_bounds& uv);

			// Classifies the materials (after set_world_usage) and puts those an atlas can hold in shared atlases (t7_watlas_<n>),
			// moving their texture areas; before the GfxWorld reads them (apply_uv_periods)
			void plan_world_atlases(const std::vector<const Material*>& materials);

			const info& get(const Material* material);

			// An XModel material, named as the converted XModels reference it (material::get_converted_name).
			// Its decal sort key is only known once prepare() has seen the world's decal layers.
			const info& get_model(const Material* material);

			// where IW7 sorts a model material among the static models' (0x140E133F0 sorts by sort key, technique, then name),
			// as its classification and names give it, after its camera region (each region draws from its own lists);
			// classifies without recording (dump_models classifies with usage)
			std::tuple<unsigned char, unsigned char, std::string, std::string> model_draw_order(const Material* material);

			// Whether a model surface of this BO3 material blocks the sun: it casts shadows and is opaque or shadow-only,
			// not alpha tested. Records nothing (get_model's classification decides what dump_models bakes).
			bool model_blocks_sun(const Material* material);

			// Whether a model material can draw on a static world surface with a world techset: not skin or eye (IW7's sss
			// techsets are model techsets). A decal's layer must reach prepare(). Records nothing.
			bool world_drawable(const Material* material);

			// the atlas (an index) each model material that dump_models would put in one joins, classified with `used` without
			// recording: an atlas's members draw as one material
			std::unordered_map<const Material*, unsigned int> atlas_groups(const std::vector<const Material*>& materials,
				const std::unordered_map<const Material*, surface_usage>& used);

			// per reason a model material stays out of the atlases (bake::atlas_tile), the materials; classifies with `used`
			// without recording
			std::map<std::string, std::vector<std::string>> atlas_report(const std::vector<const Material*>& materials,
				const std::unordered_map<const Material*, surface_usage>& used);

			// Converts every material the world's surfaces use: evaluates each BO3 material's own
			// pixel shader to bake IW7 textures, and writes the IW7 material, its images and the
			// techset state/constant buffer files under the dump folder. Runs after prepare().
			void dump_all(const GfxWorld* world);

			// The same for XModel materials (the models the map places); runs after dump_all(). `used`: the texture
			// coordinates (for the bake area) and vertex colours each material's surfaces use (every LOD the converted
			// models keep).
			void dump_models(const std::vector<const Material*>& materials,
				const std::unordered_map<const Material*, surface_usage>& used);
		}
	}
}
