#pragma once

// The surface of a BO3 forward-lit pixel shader (the lit technique of transparent and other forward templates, which
// have no gbuffer technique), as BO3 itself labels it.
//
// Draw method 1 of a lit technique is BO3's debug-override permutation: every value a PerSceneConsts debug override
// can replace is blended toward the override by its .w, `mad dst, cb.debugXOverride.w, (override - value), value`:
// debugColorOverride the albedo, debugAlphaOverride the alpha, debugNormalOverride the normal, debugSpecularOverride
// the specular colour (F0), debugGlossOverride the gloss, debugOcclusionOverride the occlusion. With every override's
// .w at 0 each blend's destination holds the material's own value, computed by the game's own code before any
// lighting runs (the blends at the top level of the program).
//
// A template with no specular map has no debugSpecularOverride blend (the glass family): its F0 is a constant, the
// one its forward decal blend puts the decals' specular over, `mad s.xyz, W, l(c, c, c, 0), decal.xyz` with
// W = 1 - the decal coverage (`add W, -coverage, l(1)`); c is taken from there (glass: 0.08, which its lighting also
// uses, 0.08 + 0.92 (1 - V.H)^3.4). A template without forward decals has it folded into its lighting's specular sum
// instead, `mul t, b, l(1 - c)` `mad u, a, l(c), t` (lit_flag_transparent: 0.04).
//
// Any other value without a label is read where that forward decal blend reads it (the template's own value before
// the decals, the one a label holds where there is one): the alpha `mad a', a, W, coverage`, the normal (normalised
// right after), the albedo and the gloss `mad v', v, W, decal`. A template whose blend has no albedo has albedo 0
// (glass_nocolor); a permutation without a single debug-override blend (glass_*_tile) has occlusion 1, as it has no
// occlusion input: the caller checks that every material texture of the lit shader feeds a surface value.
//
// A specular or gloss that neither a label nor a decal blend gives (hair, which labels neither) is read where
// the lighting takes it: F0 at the one top-level `mul f.xyz, x, specColorTint.xyz` of its $Globals, the gloss at the one
// top-level `mul t, g, l(17)` right before `exp` (the specular power 2^(17 g)).
//
// make() cuts that permutation after the last value it reads and writes the values out, BO3's gbuffer channels
// without their encoding: SV_Target0 = albedo rgb + alpha, SV_Target1 = normal xyz (tangent space over the bake's
// identity frame) + gloss (BO3 g), SV_Target2 = F0 rgb + occlusion. The container keeps its resource definitions (the
// interpreter reads them) and is re-signed; the unmodified program must round-trip byte for byte through the writer
// first.

namespace zonetool::t7
{
	namespace converter::iw7::forward_surface
	{
		enum class role : std::uint32_t
		{
			albedo,
			alpha,
			normal,
			specular,
			gloss,
			occlusion,
			count,
		};

		struct surface_program
		{
			std::vector<std::uint8_t> bytecode; // empty when the permutation cannot be cut (see reason)
			std::string reason;
			std::uint32_t labelled = 0; // bit per role found
			std::uint32_t from_decal_blend = 0; // bit per role read at the forward decal blend
			// bit per role read where the lighting takes it, without a label or decal blend (hair: the F0 tinted by its
			// specColorTint, the gloss its specular power 2^(17 g) raises 2 to)
			std::uint32_t anchored = 0;
			std::uint32_t literal = 0; // bit per role that is a constant (the F0 below, albedo 0, occlusion 1)
			std::optional<float> constant_specular; // F0 of a template without a specular label (its decal blend's)
			// the occlusion is labelled in a branch of if/else blocks only, 1 where that branch does not run (skin: its
			// aoMap path, the other being screen-space GTAO): the caller decides which branch its material takes
			bool conditional_occlusion = false;
			std::uint32_t cut_at = 0; // instructions (declarations included) the cut program keeps
		};

		// the permutation's surface program, made once per pixel shader (the result lives as long as the process)
		const surface_program& get(const std::uint8_t* bytecode, std::size_t size);

		// The alpha-test coverage of a gbuffer pixel shader whose discard reads more than one texture channel
		// (lit_flag: colorMap.a x lerp(1, colorMap00.a, COLOR0.x^2)): the program cut at its one top-level
		// `lt c, v, l(0.5)` `discard_nz c`, v written to all of render target 0 (the IW7 alpha test's threshold is 0.5 too).
		// Empty bytecode, with the reason, when the program is not of that form. Made once per pixel shader.
		const surface_program& coverage(const std::uint8_t* bytecode, std::size_t size);
	}
}
