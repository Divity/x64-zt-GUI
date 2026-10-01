#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::probe_lighting
	{
		// D3D cube faces: direction -> face and face coordinates s, t in [-1, 1], and back
		void cube_face(const float d[3], std::uint32_t& face, float& s, float& t);
		void face_direction(std::uint32_t face, float s, float t, float d[3]);

		// a StreamWrappedBuffer's bytes: resident ones from the zone, streamed ones from the xpaks
		std::vector<std::uint8_t> read_stream_buffer(const StreamWrappedBuffer* buffer, const char* what);

		// every level of one probe of a BO3 probe cube array, RGB floats per level: face, row, column
		struct probe_cube
		{
			std::uint32_t size = 0; // level 0 width
			std::vector<std::vector<float>> levels;

			// trilinear: seamless bilinear on the two levels around lod
			void sample(float lod, const float dir[3], float out[3]) const;
		};

		probe_cube decode_probe_cube(const GfxImage* image, const std::uint8_t* data, std::size_t size, std::uint32_t slice);

		// BO3's indirect diffuse lighting for one lighting state: the reflection probe term of the
		// deferred lighting pass (deferred_lighting.hlsl, constants packed by the client's 0x141CBE570),
		// evaluated on the CPU from the GfxWorld's sun volumes. The probe volumes and probe cube arrays
		// are read from the xpaks.
		class evaluator
		{
		public:
			evaluator(const GfxWorld* world, unsigned int state);
			~evaluator();

			evaluator(const evaluator&) = delete;
			evaluator& operator=(const evaluator&) = delete;

			unsigned int volume_count() const;

			// the sun volume BO3 lights a camera at p with (0x141D04BA0)
			unsigned int volume_at(const float p[3]) const;

			// loads / frees one sun volume's probe textures (the diffuse calls need it loaded)
			void load(unsigned int volume);
			void unload(unsigned int volume);

			// D: the probe lighting BO3 multiplies by albedo, ambient occlusion and SSAO, at world
			// position p for each of `count` (normal-mapped) world normals
			void diffuse(unsigned int volume, const float p[3], const float (*normals)[3], std::size_t count, float (*out)[3]) const;

			// world size of the finest probe volume texel with weight at p (at least the global
			// probe's): how finely the lighting varies there
			float texel_size(unsigned int volume, const float p[3]) const;

			// the culling boxes (min xyz, max xyz) of every sun volume's local probes, and the world size of
			// each one's probe volume texel: where BO3 has lighting finer than the global probe's
			std::vector<std::array<float, 7>> local_boxes() const;

		private:
			struct impl;
			std::unique_ptr<impl> impl_;
		};
	}
}
