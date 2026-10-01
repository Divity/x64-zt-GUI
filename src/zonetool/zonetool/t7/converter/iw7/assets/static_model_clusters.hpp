#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::static_model_clusters
	{
		// IW7 lists the static models it draws each frame in one record table (iw7_ship 0x140DCE4C0: per model and LOD a
		// record of 12 bytes plus 2 per instance, in a 56 KB buffer, 0x140DCB890) and addresses a record by a 16-bit
		// entry, record byte offset * 4 + surface (0x140DCE680: movzx, shl di, 2), so past 16 KB of records the entries
		// wrap onto other records (wrong or flickering static models) and past 56 KB the rest are dropped. BO3 has no such
		// limit. The decoder (0x140DEBDB0) reads the record at 4 x (entry >> 4) and the surface at entry & 15.
		// So small rigid static models of one area are merged into one IW7 model each, drawn as one static model with
		// every member's geometry at its placement.
		struct cluster
		{
			std::string name;
			std::vector<unsigned int> members; // BO3 static model indices
			float origin[3]{};                  // the IW7 placement: the members' centroid, unrotated, scale 1
			float lighting_origin[3]{};         // where the light grid lights it: its brightest member's centre
			std::vector<float> visibility;      // per member: its probe lighting's luminance against the cluster's (0..1]
			float mins[3]{};                    // world bounds of every member's first LOD
			float maxs[3]{};
			bool casts_shadow = true;
			// a posed static model (BO3 GfxStaticModelDrawInst posedBones: corpses) as one member with its pose baked in:
			// per BO3 bone its model space skinning matrix (row-major 3x4)
			bool posed = false;
			std::vector<std::array<float, 12>> skin;
			// a LOD has more than IW7's 16 static model surfaces: placed as a script_model instead (gfxworld.cpp)
			bool too_many_surfaces = false;
		};

		// IW7 lists a sun shadow cascade's static model casters per surface: 256 entries a draw type (iw7_ship 0x140DCC700
		// sets each list's capacity to 256, 0x140DCF0C0 drops the entry past it, error 25), one entry per surface of each
		// consecutive run of one model; casters past the 256th cast nothing. BO3 has no such limit, so the static models' shadows are drawn by
		// shadow proxies: models that only cast (camera region 11, which the camera's lists skip, 0x140DCE680), each opaque
		// caster triangle of an area in as few surfaces as IW7's 16-bit indices allow, an alpha tested one's with its own
		// material; the models they stand for cast nothing (NO_CAST_SHADOW).
		struct shadow_proxy
		{
			std::string name;
			std::vector<unsigned int> members; // BO3 static model indices
			float origin[3]{};                  // the IW7 placement: the members' centroid, unrotated, scale 1
			float mins[3]{};                    // world bounds of every member's first LOD
			float maxs[3]{};
			float dist = 0.0f;                  // cast until the farthest member's last LOD distance
			bool alpha = false;                 // its members' alpha tested surfaces (else their opaque ones)
		};

		struct plan
		{
			std::vector<cluster> clusters;
			std::vector<int> cluster_of;          // BO3 static model -> cluster, or -1
			std::vector<unsigned int> iw7_index;  // BO3 static model -> the IW7 static model drawing it
			unsigned int iw7_count = 0;           // the unmerged static models (in BO3's order), the clusters, the proxies
			std::vector<shadow_proxy> proxies;
			std::vector<char> proxied;            // BO3 static model -> its shadow is a proxy's (it casts none itself)
			unsigned int proxy_begin = 0;         // the IW7 static model index of the first proxy
		};

		// decides the clusters of the GfxWorld's static models (before its cells and static models are converted)
		void make_plan(const GfxWorld* asset, const std::string& map_name);
		// decides the shadow proxies (after world_material::prepare, before the static models are converted)
		void plan_shadow_proxies(const GfxWorld* asset, const std::string& map_name);
		const plan& current();

		// BO3 static model index lists (cells) as IW7 ones: each one's IW7 static model, once
		std::vector<unsigned short> remap_indices(const unsigned short* indices, unsigned int count);

		// writes each cluster's and shadow proxy's XModel and its LOD meshes; after the model materials (their baked texture
		// coordinates, and the alpha tested materials a proxy copies)
		void dump_all();

		// whether a model is placed only inside clusters (no row of its own needed)
		bool only_clustered(const XModel* model);

		std::vector<std::string> model_names();
		void clear();
	}
}
