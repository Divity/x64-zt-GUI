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
end

table.insert(dependencies, imgui)
