imgui = {
	source = path.join(dependencies.basePath, "imgui"),
	backends = path.join(dependencies.basePath, "imgui/backends"),
}

function imgui.import()
	imgui.includes()
end

function imgui.includes()
	includedirs {
		imgui.source,
		imgui.backends,
	}
end

function imgui.project()
	-- Empty - source files included directly in zonetool project via src/zonetool.lua
	-- This prevents creation of imgui.vcxproj while keeping imgui in dependencies
end

table.insert(dependencies, imgui)
