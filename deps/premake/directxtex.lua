directxtex = {
	source = path.join(dependencies.basePath, "DirectXTex/DirectXTex"),
	common = path.join(dependencies.basePath, "DirectXTex/Common"),
	-- the GPU BC6H/BC7 encoder's shaders, compiled by DirectXTex/Shaders/CompileShaders.cmd
	shaders = path.join(dependencies.basePath, "extra/DirectXTex/Shaders/Compiled")
}

function directxtex.import()
	links { "DirectXTex" }
	directxtex.includes()
end

function directxtex.includes()
	includedirs {
		directxtex.source,
		directxtex.common,
		-- DirectXTex's GPU encoder builds with them, and the T7 converter drives the BC7 ones itself (gpu_eval.cpp)
		directxtex.shaders
	}
end

function directxtex.project()
	project "DirectXTex"
		language "C++"

		directxtex.includes()

		files {
			path.join(directxtex.common, "*.h"),
			path.join(directxtex.source, "*.h"),
			path.join(directxtex.source, "*.cpp"),
			path.join(directxtex.source, "*.inl"),
		}

		warnings "Off"
		kind "StaticLib"
end

table.insert(dependencies, directxtex)
