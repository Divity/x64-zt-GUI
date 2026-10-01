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
			~sun_blockers();

			// whether the segment crosses a blocker; ignore_model: an IW7 static model whose own triangles do not count (the
			// light grid lights a static model at a point inside it)
			bool blocked(const double from[3], const double to[3], const std::uint32_t* ignore_model = nullptr) const;

		private:
			struct impl;
			std::unique_ptr<impl> impl_;
		};

		// blockers: what shadows the sun (the primary image's sun visibility)
		void bake(const GfxWorld* asset, zonetool::iw7::GfxWorld* world, zonetool::iw7::GfxWorldTransientZone* zone,
			const sun_blockers& blockers, utils::memory::allocator& allocator);
	}
}
