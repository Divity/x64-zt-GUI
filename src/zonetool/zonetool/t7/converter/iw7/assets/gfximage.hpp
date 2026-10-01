#pragma once

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace gfximage
		{
			// One image's whole mip chain as BO3 stores it: faces one after another, each face's
			// levels largest first (D3D subresource order), in the image's own format.
			struct image_pixels
			{
				std::vector<std::uint8_t> data;
				std::uint32_t width = 0;
				std::uint32_t height = 0;
				std::uint32_t depth = 1;
				std::uint32_t faces = 1;
				std::uint32_t levels = 1;
				DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
			};

			// Images never reach the asset pool (see db_add_xasset_stub) and the loader reuses one
			// temporary block for each, so a copy of every header is kept to resolve material
			// texture slots. Zone mips are complete from DB_FinishLoadXFile on.
			GfxImage* register_image(GfxImage* image);
			GfxImage* find_image(const std::string& name);
			void clear_registry();

			bool image_dumped(const std::string& name);

			// the full-resolution mip chain: the streamed parts from the xpaks, or the mips the
			// zone carries when the image is not streamed
			bool get_pixels(const GfxImage* image, image_pixels& out);

			// queues the image; flush_dumps writes the queue once the zone's delayed data is in
			void dump(GfxImage* asset);
			void flush_dumps();
		}
	}
}
