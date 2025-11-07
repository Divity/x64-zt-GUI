imgui = {
	source = path.join(dependencies.basePath, "imgui"),
	backends = path.join(dependencies.basePath, "imgui/backends"),
}

function imgui.import()
	links { "imgui" }
	imgui.includes()
end

function imgui.includes()
	includedirs {
		imgui.source,
		imgui.backends,
	}
end

function imgui.project()
	project "imgui"
		language "C++"
		warnings "Off"
		kind "StaticLib"

		imgui.includes()

		files {
			path.join(imgui.source, "*.cpp"),
			path.join(imgui.source, "*.h"),
			path.join(imgui.backends, "imgui_impl_win32.cpp"),
			path.join(imgui.backends, "imgui_impl_win32.h"),
			path.join(imgui.backends, "imgui_impl_dx11.cpp"),
			path.join(imgui.backends, "imgui_impl_dx11.h"),
		}

		defines {
			"IMGUI_IMPL_API=extern `"C`"",
		}
end

table.insert(dependencies, imgui)
