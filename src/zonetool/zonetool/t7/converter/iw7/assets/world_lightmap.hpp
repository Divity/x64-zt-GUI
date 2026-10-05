#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::world_lightmap
	{
		// IW7 lights world surfaces from a lightmap (indirect) plus its primary lights at runtime; BO3
		// has no lightmaps and takes the indirect light from its reflection probes. This gives the
		// converted world's lightmapped static surfaces lightmap coordinates (charts cut the surfaces
		// along their creases, so the transient zone's vertices and the surfaces' vertex ranges and
		// indices are rebuilt), bakes BO3's probe lighting of lighting state 0 into IW7's lightmap
		// images (ambient + directional colour and a tangent-space direction) and writes the lightmap
		// asset. Runs after the surfaces are converted and before anything reads their vertices.
		// What blocks BO3's sun, for IW7's baked sun visibility (the lightmap's primary image and the light grid's
		// coefficient 27): the static surfaces `occluders` flags, the static models that cast (first LOD, placed) and the
		// alpha tested ones where their mask passes IW7's test, as IW7's real-time sun shadow draws them. Built before
		// bake (which rebuilds the world's vertices).
		class sun_blockers
		{
		public:
			sun_blockers(const GfxWorld* asset, const zonetool::iw7::GfxWorld* world, const zonetool::iw7::GfxWorldTransientZone* zone,
				const std::vector<std::uint8_t>& occluders);
			// the same from BO3's world alone, before its static models are planned (static_model_clusters): BO3's opaque static
			// surfaces that cast the sun's shadow, and its static models, each owning its triangles by its BO3 index
			explicit sun_blockers(const GfxWorld* asset);
			~sun_blockers();

			// whether the segment crosses a blocker; ignore_model: a static model whose own triangles do not count (the light
			// grid lights a static model at a point inside it): its IW7 index, or its BO3 index for the BO3-only set
			bool blocked(const double from[3], const double to[3], const std::uint32_t* ignore_model = nullptr) const;

			// what blocks the segment: 0 nothing, 1 an opaque triangle, 2 an alpha tested one, and that triangle's owner (a static
			// model as blocked() names it, ~0u a world surface)
			std::pair<int, std::uint32_t> blocker(const double from[3], const double to[3], const std::uint32_t* ignore_model = nullptr) const;

		private:
			void build(const GfxWorld* asset, const zonetool::iw7::GfxWorld* world, const zonetool::iw7::GfxWorldTransientZone* zone,
				const std::vector<std::uint8_t>* occluders);

			struct impl;
			std::unique_ptr<impl> impl_;
		};

		// the share of the sun (0..1) that reaches p: five rays toward it, 0.5 degrees apart, from one unit along it (the light
		// grid's sun visibility, coefficient 27, for its probes and static model samples)
		float sun_fraction(const sun_blockers& blockers, const float sun_dir[3], const float p[3], const std::uint32_t* ignore_model);

		// blockers: what shadows the sun (the primary image's sun visibility). pieces: static surfaces of placed models (the
		// static models the GfxWorld draws), per surface the first triangle of each placed part; a part's triangles are
		// charted by the axis their normal leans to (six box projections), not by crease: thousands of small parts would
		// each be many charts
		void bake(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, zonetool::iw7::GfxWorldTransientZone* zone,
			const sun_blockers& blockers, const std::unordered_map<unsigned int, std::vector<unsigned int>>& pieces,
			utils::memory::allocator& allocator);
	}
}
