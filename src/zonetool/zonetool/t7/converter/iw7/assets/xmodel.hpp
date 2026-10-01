#pragma once

#include "xmodel_mesh.hpp"

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace xmodel
		{
			// `bones`: the model's IW7 bones where they are not BO3's (dump() maps them)
			zonetool::iw7::XModel* convert(XModel* asset, const xmodel_mesh::bone_map& bones, utils::memory::allocator& allocator);
			void dump(XModel* asset);

			// the IW7 name of a BO3 XModel or XModelMesh: its own, but a brush model's "*n", which no file name can hold, is
			// t7_brushmodel_n
			std::string iw7_name(const char* bo3_name);

			// whether dump() wrote this model in the current dump
			bool dumped(const char* name);
			void clear();

			// the materials of the LODs the converted model keeps
			std::vector<const Material*> materials(const XModel* asset);

			// the converted model's LOD count, and the distance IW7 keeps LOD `lod` to (the last LOD's: where it is culled)
			unsigned int lod_count(const XModel* asset);
			float lod_dist(const XModel* asset, unsigned int lod);

			// per LOD the converted model keeps, each surface's BO3 texture mip term (meshMaterials himipInvSqRadii)
			std::vector<std::vector<float>> himip_inv_sq_radii(const XModel* asset);

			// IW7's XModel invHighMipRadius entry for a BO3 himipInvSqRadii: an integer, the inverse radius x 65536.
			// DB_StreamModels (0x140A86AB0) takes (float)entry x distance / 65536 as the surface's texture demand.
			inline unsigned short inv_high_mip_radius(const float inv_sq_radius)
			{
				const auto value = std::sqrt(std::max(inv_sq_radius, 0.0f)) * 65536.0f;
				return static_cast<unsigned short>(std::clamp(std::lround(value), 0l, 65535l));
			}

			// the meshes of the LODs the converted model keeps, each with its surfaces' materials
			std::vector<std::pair<XModelMesh*, std::vector<const Material*>>> lod_meshes(const XModel* asset);
		}
	}
}
