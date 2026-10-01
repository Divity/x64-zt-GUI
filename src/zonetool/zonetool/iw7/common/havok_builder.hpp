#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zonetool::iw7
{
	namespace havok
	{
		namespace builder
		{
			struct shape_tag
			{
				std::uint32_t collision_filter = 0;
				std::uint32_t material_crc = 0;
				std::uint64_t user_data = 0;

				bool operator==(const shape_tag& other) const
				{
					return collision_filter == other.collision_filter
						&& material_crc == other.material_crc
						&& user_data == other.user_data;
				}
			};

			struct triangle
			{
				float verts[3][3];
				bool is_quad = false;
				float vert3[3] = {};
				unsigned short surface_tag;
				int contents = 1;
				std::uint32_t material_crc = 0x1AB7BC33u;
				std::uint64_t user_data = 0;
			};

			struct convex
			{
				std::vector<std::array<float, 3>> verts;
				unsigned short surface_tag;
				int contents = 1;
				std::uint32_t material_crc = 0x1AB7BC33u;
				std::uint64_t user_data = 0;
			};

			const char* convex_rejection(const std::vector<std::array<float, 3>>& verts);

			struct mesh_input
			{
				std::vector<triangle> triangles;
				std::vector<convex> convexes;
				float convex_radius = 0.0f;
			};

			std::vector<std::uint8_t> build_world_shape(const mesh_input& input,
				std::vector<shape_tag>* out_tags = nullptr);

			struct polytope_face
			{
				float plane[4];
				std::vector<std::uint8_t> indices;
			};

			struct polytope
			{
				std::vector<std::array<float, 3>> verts;
				std::vector<polytope_face> faces;
			};

			struct ents_shape
			{
				std::vector<polytope> convexes;
				int contents = 1;
				unsigned int entity_contents = 1;
				std::uint32_t material_crc = 0x1AB7BC33u;
				std::uint64_t user_data = 0;
				std::string name;
			};

			struct ents_input
			{
				std::vector<ents_shape> shapes;

				std::vector<shape_tag> world_tags;

				float scale = 1.0f / 32.0f;
			};

			struct ents_tag_merge
			{
				std::size_t prefix = 0;
				std::size_t total = 0;
				std::size_t reused = 0;
				std::size_t appended = 0;
			};

			std::vector<std::uint8_t> build_ents_shape_list(const ents_input& input,
				ents_tag_merge* out_merge = nullptr);

			struct physics_asset_input
			{
				std::string body_name = "scriptbrushmodeldummy";
				std::uint32_t body_quality_crc = 0x7923E35Cu;
				std::uint32_t material_crc = 0x1AB7BC33u;
				std::uint32_t body_contents = 1;
			};

			std::vector<std::uint8_t> build_physics_asset(const physics_asset_input& input);

			std::vector<std::uint8_t> build_model_physics_asset(const mesh_input& input,
				const physics_asset_input& physics_asset);

			struct dynamic_physics_asset_input
			{
				std::vector<polytope> convexes;
				std::string body_name = "tag_origin";
				float mass = 5.0f;
				std::uint32_t body_contents = 0x3180u;
				std::uint32_t material_crc = 0x1AB7BC33u;
				std::uint32_t body_quality_crc = 0x7923E35Cu;
				std::uint32_t motion_properties_crc = 0x9F53AC92u;
				float scale = 1.0f / 32.0f;
			};

			std::vector<std::uint8_t> build_dynamic_physics_asset(const dynamic_physics_asset_input& input);

			std::vector<std::uint8_t> build_model_physics_lod(const mesh_input& input,
				const std::string& bone_name);

			// an XModel physics LOD shape: the collision of one bone, in that bone's space
			struct lod_shape
			{
				std::string bone_name;
				mesh_input mesh;
			};

			// HavokPhysicsXModelLOD with one compressed mesh per bone and one tag table for all of them
			std::vector<std::uint8_t> build_model_physics_lod(const std::vector<lod_shape>& shapes);
		}
	}
}
