#pragma once

#include "world_material.hpp"

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace xmodel_mesh
		{
			// per surface, what its converted material changes in its vertices (world_material::info of the material):
			// the BO3 texture coordinates its baked textures cover (uv_origin, uv_span), so the surface's coordinates
			// become (uv - origin) / span, and white_rgb (info vertex_alpha): the IW7 techset reads the vertex colour
			// where BO3's read only its alpha, so the rgb is written white. Empty: BO3's own vertices for all.
			struct surface_transform
			{
				float origin[2] = { 0.0f, 0.0f };
				float span[2] = { 1.0f, 1.0f };
				bool white_rgb = false;

				bool operator==(const surface_transform&) const = default;
			};
			using surface_transforms = std::vector<surface_transform>;

			// a model's IW7 bones where they are not BO3's (xmodel: cosmetic bones merged into their parents): BO3 bone ->
			// IW7 bone (a merged bone -> its parent's) and IW7 bone -> BO3 bone. Empty: BO3's bones as they are.
			struct bone_map
			{
				std::vector<std::uint16_t> to_iw7;
				std::vector<std::uint16_t> to_bo3;

				bool empty() const
				{
					return this->to_iw7.empty();
				}

				std::uint16_t operator()(const std::uint16_t bone) const
				{
					return this->empty() ? bone : this->to_iw7.at(bone);
				}

				bool operator==(const bone_map&) const = default;
			};

			zonetool::iw7::XModelSurfs* convert(XModelMesh* asset, utils::memory::allocator& allocator, const surface_transforms& transforms = {},
				const bone_map& bones = {});

			// the part bits convert() gives the mesh (the IW7 bones its vertices use), for the model LOD that draws it
			std::array<int, 8> part_bits(XModelMesh* asset, const bone_map& bones);

			// the texture coordinates and vertex colours of each surface (empty for a surface without vertex data)
			std::vector<world_material::surface_usage> surface_usage(XModelMesh* asset);

			// a vertex's BO3 bones with weight, heaviest first, the weights summing to 1
			struct vertex_bones
			{
				std::array<std::uint16_t, 4> bone{};
				std::array<float, 4> weight{};
				unsigned int count = 0;
			};

			// the model space position and the bones of each vertex of the mesh (a rigid surface's vertices wholly on their
			// list's bone); nothing for a mesh without vertex data
			void for_each_vertex(XModelMesh* asset, const std::function<void(const float* xyz, const vertex_bones& bones)>& fn);

			// whether every surface of the mesh is rigid and has vertex data (what append_placed can place)
			bool rigid_with_data(XModelMesh* asset);

			// Appends surface `surface` of the mesh as a static model places it (world = origin + scale * (x axis[0] +
			// y axis[1] + z axis[2])), moved by -center, with the texture coordinates and vertex colours convert() writes
			// under `transform`; its faces index the vertices appended. False (nothing appended) when the surface is skinned,
			// has no vertex data or would take `verts` past IW7's 16-bit vertex indices. `visibility` is every vertex's self
			// visibility, the share of the model's probe lighting it takes (0..1).
			bool append_placed(XModelMesh* asset, unsigned int surface, const surface_transform& transform, const float* origin,
				const float (*axis)[3], float scale, const float* center, std::vector<zonetool::iw7::GfxPackedVertex>& verts,
				std::vector<zonetool::iw7::Face>& faces, float visibility = 1.0f);

			// append_placed for a posed mesh (skinned or rigid): every vertex first goes through its BO3 bones' model space
			// skinning matrices (row-major 3x4, `skin[bone]`), weighted, then the placement
			bool append_posed(XModelMesh* asset, unsigned int surface, const surface_transform& transform,
				const std::vector<std::array<float, 12>>& skin, const float* origin, const float (*axis)[3], float scale, const float* center,
				std::vector<zonetool::iw7::GfxPackedVertex>& verts, std::vector<zonetool::iw7::Face>& faces);

			// A mesh is written by the XModel that draws it, which knows its surfaces' materials and bones and loads after
			// it: dump(asset) only remembers the mesh, dump(asset, transforms, bones) writes it, dump_remaining() writes the
			// remembered meshes no XModel wrote. XModels that draw one mesh with materials that change its vertices
			// differently (skins of one prop covering different texture coordinates) or with other bones each get a copy:
			// dump(asset, transforms, bones) returns the name the mesh is written under, the mesh's own for the first.
			void dump(XModelMesh* asset);
			std::string dump(XModelMesh* asset, const surface_transforms& transforms, const bone_map& bones = {});
			void dump_remaining();
		}
	}
}
