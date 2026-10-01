#pragma once

#include "xmodel_mesh.hpp"

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace xmodel_collision
		{
			// a bone's base pose (baseMat, the same in both games: its rotation and position in model space), to take model
			// space points into the bone's space
			struct bone_frame
			{
				double q[4]; // unit
				double t[3];
			};
			bone_frame frame_of(const zonetool::iw7::DObjAnimMat& mat);
			std::array<float, 3> to_bone(const bone_frame& bone, const std::array<float, 3>& p);

			// BO3 collision surfaces and collmaps -> the IW7 model's collSurfs, contents, physics LOD and
			// PhysicsAsset. Needs the converted model's bones (baseMat) and how they map from BO3's.
			void convert(XModel* asset, zonetool::iw7::XModel* model, const xmodel_mesh::bone_map& bones, utils::memory::allocator& allocator);

			// the collision triangles convert() builds the physics LOD and the mesh PhysicsAsset from: model space,
			// IW7 winding, IW7 contents and surface flags
			struct triangle
			{
				std::array<float, 3> verts[3];
				std::uint32_t contents;
				std::uint32_t flags;
			};
			std::vector<triangle> triangles(XModel* asset);

			// what the converted model adds to IW7's static model physics where it is placed: its PhysicsAsset's
			// bodies (convert() gives every colliding model one) and its physics LOD shapes (one per bone with
			// triangles)
			struct physics_cost
			{
				unsigned int bodies;
				unsigned int lod_shapes;
			};
			physics_cost cost(XModel* asset);
		}
	}
}
