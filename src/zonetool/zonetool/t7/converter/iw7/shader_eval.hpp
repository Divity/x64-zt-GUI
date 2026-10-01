#pragma once

#include <bitset>

// Runs BO3 (DXBC shader model 5) pixel shaders on the CPU so material conversion can use the
// game's own shader math instead of a transcription of it. Programs are decoded with the
// vendored shader-tool reader; the resource definition chunk is parsed here because the vendored
// reader assumes 40-byte resource bindings, which is the shader model 5.1 layout (BO3's RD11
// header declares 32).
//
// A machine executes one program over 64 lanes at once, laid out as 16 quads of 2x2 pixels in
// the same order as the GPU: lane 0 (0,0), 1 (1,0), 2 (0,1), 3 (1,1) of each quad. Implicit
// derivatives (sample, deriv_rt*) are taken inside a quad, like the hardware does.

namespace zonetool::t7
{
	namespace converter::iw7::shader_eval
	{
		constexpr std::uint32_t lane_count = 64;
		using lane_mask = std::uint64_t;
		constexpr lane_mask all_lanes = ~lane_mask(0);

		struct cbuffer_variable
		{
			std::string name;
			std::uint32_t offset;
			std::uint32_t size;
		};

		struct cbuffer_desc
		{
			std::string name;
			std::uint32_t size;
			std::vector<cbuffer_variable> variables;
		};

		// D3D_SHADER_INPUT_TYPE: 0 cbuffer, 1 tbuffer, 2 texture, 3 sampler, 5 structured ...
		struct binding_desc
		{
			std::string name;
			std::uint32_t input_type;
			std::uint32_t dimension;
			std::uint32_t bind_point;
			std::uint32_t bind_count;
		};

		struct signature_desc
		{
			std::string semantic;
			std::uint32_t semantic_index;
			std::uint32_t system_value;
			std::uint32_t component_type;
			std::uint32_t reg;
			std::uint32_t mask;
			// a pixel shader input's D3D10_SB_INTERPOLATION_MODE, from its dcl_input_ps (0 when not declared)
			std::uint32_t interpolation = 0;
		};

		enum class operand_kind : std::uint8_t
		{
			null,
			temp,
			indexable_temp,
			input,
			output,
			immediate,
			cbuffer,
			icb,
			resource,
			sampler,
			other,
		};

		struct operand
		{
			operand_kind kind = operand_kind::null;
			std::uint8_t component_count = 0; // 0, 1 or 4
			std::uint8_t mask = 0; // destination write mask
			std::uint8_t swizzle[4] = { 0, 1, 2, 3 };
			std::uint8_t modifier = 0; // 1 neg, 2 abs, 3 -abs
			std::uint32_t index[2] = { 0, 0 };
			// relative addressing of index[1] (cb#[r.x + n], x#[r.x + n]) or index[0] (icb[r.x + n])
			bool relative = false;
			operand_kind relative_kind = operand_kind::null;
			std::uint32_t relative_register = 0;
			std::uint8_t relative_component = 0;
			std::uint32_t immediate[4] = { 0, 0, 0, 0 };
		};

		struct instruction
		{
			std::uint16_t opcode = 0;
			bool saturate = false;
			bool test_nonzero = false;
			std::int8_t offsets[3] = { 0, 0, 0 };
			std::vector<operand> operands;
		};

		class program
		{
		public:
			// Throws std::runtime_error when the container or an instruction cannot be handled.
			static std::shared_ptr<program> parse(const std::uint8_t* data, std::size_t size);

			const std::vector<cbuffer_desc>& cbuffers() const { return this->cbuffers_; }
			const std::vector<binding_desc>& bindings() const { return this->bindings_; }
			const std::vector<signature_desc>& inputs() const { return this->inputs_; }
			const std::vector<signature_desc>& outputs() const { return this->outputs_; }
			const std::vector<instruction>& instructions() const { return this->instructions_; }

			const binding_desc* find_binding(const std::string& name, std::uint32_t input_type) const;
			const cbuffer_desc* find_cbuffer(const std::string& name) const;
			const signature_desc* find_input(const std::string& semantic, std::uint32_t semantic_index) const;

			std::uint32_t temp_count() const { return this->temp_count_; }
			const std::vector<std::pair<std::uint32_t, std::uint32_t>>& indexable_temps() const { return this->indexable_temps_; }
			const std::vector<std::uint32_t>& immediate_cbuffer() const { return this->icb_; }

			// the disassembly of every instruction, for error messages
			std::string describe(std::size_t instruction_index) const;

		private:
			std::vector<cbuffer_desc> cbuffers_;
			std::vector<binding_desc> bindings_;
			std::vector<signature_desc> inputs_;
			std::vector<signature_desc> outputs_;
			std::vector<instruction> instructions_;
			std::vector<std::string> text_;
			std::uint32_t temp_count_ = 0;
			std::vector<std::pair<std::uint32_t, std::uint32_t>> indexable_temps_; // x# -> element count
			std::vector<std::uint32_t> icb_;
		};

		// One texture as the program sees it. `lod` is the mip level to read (fractional levels
		// blend two mips); the source applies its own addressing and filtering.
		class texture_source
		{
		public:
			virtual ~texture_source() = default;
			virtual void dimensions(std::uint32_t mip, std::uint32_t& width, std::uint32_t& height, std::uint32_t& levels) const = 0;
			virtual void sample(float u, float v, float lod, std::int32_t offset_u, std::int32_t offset_v, float out[4]) const = 0;
			virtual void load(std::int32_t x, std::int32_t y, std::int32_t mip, float out[4]) const = 0;
		};

		// the coordinate derivatives a sample instruction saw in quad 0 (lanes 0-3)
		struct sample_probe
		{
			std::uint32_t texture_slot;
			float dudx;
			float dvdx;
			float dudy;
			float dvdy;
			float u; // lane 0's coordinate
			float v;
			std::int32_t offset_u; // the instruction's texel offsets
			std::int32_t offset_v;
		};

		class machine
		{
		public:
			explicit machine(std::shared_ptr<program> prog);

			void enable_probe(bool enabled) { this->probing_ = enabled; this->probes_.clear(); }
			const std::vector<sample_probe>& probes() const { return this->probes_; }

			// Reads of a resource slot nothing is bound to (textures of any dimension, structured and raw
			// buffers) return zeros instead of failing: a forward shader evaluated with its lighting
			// resources left out. Off by default, so a material shader that reads a texture the material
			// does not have still fails.
			void set_unbound_resources_zero(bool enabled) { this->unbound_zero_ = enabled; }

			void bind_cbuffer(std::uint32_t slot, const void* data, std::size_t size);
			void bind_texture(std::uint32_t slot, const texture_source* source);
			// a structured buffer: ld_structured reads element e at byte e * stride + offset (bytes past the end read 0)
			void bind_structured(std::uint32_t slot, const void* data, std::size_t size, std::uint32_t stride);

			// input register values for every lane, as raw 32-bit patterns
			void set_input(std::uint32_t reg, std::uint32_t component, std::uint32_t lane, std::uint32_t bits);
			void set_input(std::uint32_t reg, std::uint32_t component, std::uint32_t lane, float value);

			// executes the program over all lanes
			void run();

			float output(std::uint32_t reg, std::uint32_t component, std::uint32_t lane) const;
			std::uint32_t output_bits(std::uint32_t reg, std::uint32_t component, std::uint32_t lane) const;
			lane_mask discarded() const { return this->discarded_; }

		private:
			using lanes = std::array<std::uint32_t, lane_count>;
			using reg4 = std::array<lanes, 4>;

			void fetch(const operand& op, reg4& out, bool integer) const;
			void store(const operand& op, const reg4& value, bool saturate);
			std::uint32_t read_index(const operand& op, std::uint32_t lane) const;
			void sample(const instruction& ins, std::size_t index);
			void fail(std::size_t index, const char* why) const;

			std::shared_ptr<program> program_;
			std::vector<reg4> temps_;
			std::vector<std::vector<reg4>> indexable_;
			std::vector<reg4> inputs_;
			std::vector<reg4> outputs_;
			std::vector<std::vector<std::uint32_t>> cbuffers_;
			std::vector<const texture_source*> textures_;
			struct structured_buffer
			{
				std::vector<std::uint8_t> data;
				std::uint32_t stride = 0;
			};
			std::unordered_map<std::uint32_t, structured_buffer> buffers_;
			lane_mask active_ = all_lanes;
			lane_mask discarded_ = 0;
			bool probing_ = false;
			bool unbound_zero_ = false;
			std::vector<sample_probe> probes_;
		};

		// Static data flow: which texture channels and input components can reach each output
		// component, and which reach a discard condition. Control flow is ignored, so the answer
		// is a superset.
		struct taint_result
		{
			// bit = texture_slot * 4 + channel, for texture slots below 128
			using texture_set = std::bitset<512>;
			// bit = input_register * 4 + component, for registers below 32
			using input_set = std::bitset<128>;

			std::vector<std::array<texture_set, 4>> output_textures;
			std::vector<std::array<input_set, 4>> output_inputs;
			texture_set discard_textures;
			input_set discard_inputs;
		};

		taint_result analyse(const program& prog);

		// Where a vertex shader takes a component of a COLOR output from
		struct colour_source
		{
			enum class kind : std::uint8_t
			{
				unknown,
				one, // the literal 1
				channel, // vertex colour channel `channel` (0-3: r, g, b, a) as it is
				linearised, // vertex colour channel `channel` ^ 2.2
			};

			kind what = kind::unknown;
			std::uint8_t channel = 0;
		};

		// For a vertex shader: per COLOR output (semantic index 0 / 1), per component in the element's order, what the
		// program writes it from, read from the last write of the component at the top level: `mov o, v.c` (its COLOR
		// input), `mov o, l(1)`, or `exp o, t` of `mul t, l, 2.2` of `log l, |v.c|` (BO3's model vertex shaders: lit_flag's
		// COLOR0 rgb, lit_alphatest's COLOR1.x the alpha). Anything else is unknown.
		std::array<std::vector<colour_source>, 2> vertex_colour(const program& vs);
	}
}
