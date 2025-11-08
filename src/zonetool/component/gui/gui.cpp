#include <std_include.hpp>
#include "gui.hpp"
#include "loader/component_loader.hpp"

#include "game/mode.hpp"
#include "component/h1/command.hpp"
#include "component/h2/command.hpp"
#include "component/s1/command.hpp"
#include "component/iw6/command.hpp"
#include "component/iw7/command.hpp"
#include "component/t7/command.hpp"
#include "resource.hpp"

#include <d3d11.h>
#include <dxgi.h>

// ImGui headers (implementation compiled separately via project files)
#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx11.h>

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

#include <utils/thread.hpp>
#include <utils/io.hpp>
#include <utils/hook.hpp>

#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <shlobj.h>
#include <dwmapi.h>

struct ACCENTPOLICY
{
	int32_t nAccentState;
	int32_t nFlags;
	int32_t nColor;
	int32_t nAnimationId;
};

struct WINCOMPATTRDATA
{
	int32_t nAttribute;
	void* pData;
	uint32_t ulSizeOfData;
};

typedef BOOL (WINAPI *pfnSetWindowCompositionAttribute)(HWND, WINCOMPATTRDATA*);

#ifndef DWMWA_SYSTEMBACKDROP_TYPE
#define DWMWA_SYSTEMBACKDROP_TYPE 38
#endif
#ifndef DWMSBT_MICA
#define DWMSBT_MICA 2
#endif

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "dwmapi.lib")

namespace gui
{
	static component* g_gui_component = nullptr;

	component* get_gui_component()
	{
		return g_gui_component;
	}

	component::component()
		: hwnd_(nullptr)
		, hicon_small_(nullptr)
		, hicon_large_(nullptr)
		, device_(nullptr)
		, device_context_(nullptr)
		, swap_chain_(nullptr)
		, render_target_view_(nullptr)
		, running_(false)
		, should_close_(false)
		, selected_fastfile_(-1)
		, selected_map_zone_(-1)
		, selected_csv_(-1)
		, dump_matching_zones_pattern_("")
		, selected_generate_csv_map_(-1)
		, generate_csv_sp_mode_(false)
		, auto_scroll_log_(true)
		, filter_log_errors_warnings_(false)
		, command_input_text_("")
		, operation_in_progress_(false)
		, operation_status_text_("")
		, dump_target_mode_(game::game_mode::none)
		, dump_asset_type_("")
		, dump_asset_name_("")
		, dump_map_skip_common_(false)
		, file_watcher_running_(false)
		, resize_width_(0)
		, resize_height_(0)
		, batch_dump_in_progress_(false)
		, batch_dump_current_(0)
		, batch_dump_total_(0)
		, swap_chain_occluded_(false)
	{
		g_gui_component = this;
	}

	component::~component()
	{
		if (batch_dump_thread_.joinable())
		{
			batch_dump_thread_.join();
		}
		if (g_gui_component == this)
		{
			g_gui_component = nullptr;
		}
		file_watcher_running_ = false;
		if (file_watcher_thread_.joinable())
		{
			file_watcher_thread_.join();
		}
		destroy_device();
		destroy_window();
	}

	namespace
	{
		using printf_func = int(*)(const char* format, ...);
		using vprintf_func = int(*)(const char* format, va_list);
		utils::hook::detour printf_hook;
		printf_func printf_original = nullptr;

	int printf_stub(const char* format, ...)
	{
		char buffer[4096];
		va_list args;
		va_start(args, format);
		const int result = vsnprintf(buffer, sizeof(buffer), format, args);
		va_end(args);

		if (result > 0)
		{
			auto* comp = get_gui_component();
			if (comp)
			{
				comp->add_log_message(buffer);
			}
		}


		va_start(args, format);
		const int orig_result = vprintf(format, args);
		va_end(args);
		return orig_result;
	}
	}

	void component::add_log_message(const std::string& message)
	{
		if (message.empty())
		{
			return;
		}

		std::string msg = message;
		if (msg.back() == '\n')
		{
			msg.pop_back();
		}
		if (!msg.empty() && msg.back() == '\r')
		{
			msg.pop_back();
		}
		if (msg.empty())
		{
			return;
		}

		std::lock_guard<std::mutex> lock(log_mutex_);
		log_messages_.push_back(std::move(msg));
		if (log_messages_.size() > 1000)
		{
			log_messages_.pop_front();
		}
	}

	void component::post_start()
	{

		printf_hook.create(&printf, printf_stub);

		printf_original = reinterpret_cast<printf_func>(printf_hook.get_original());
	}

	void component::post_load()
	{
		::ShowWindow(::GetConsoleWindow(), SW_HIDE);

		running_ = true;
		gui_thread_ = utils::thread::create_named_thread("GUI", [this]()
		{
			this->gui_thread();
		});
	}

	void component::pre_destroy()
	{
		should_close_ = true;
		file_watcher_running_ = false;
		if (file_watcher_thread_.joinable())
		{
			file_watcher_thread_.join();
		}
		if (gui_thread_.joinable())
		{
			gui_thread_.join();
		}
		printf_hook.clear();
		running_ = false;
	}

	void component::gui_thread()
	{
		if (!create_window())
		{
			return;
		}

		if (!create_device())
		{
			destroy_window();
			return;
		}

		ImGui_ImplWin32_EnableDpiAwareness();
		const auto main_scale = ImGui_ImplWin32_GetDpiScaleForMonitor(::MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY));

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ImGui::StyleColorsDark();
	ImGuiStyle& style = ImGui::GetStyle();
	style.ScaleAllSizes(main_scale);

	style.WindowRounding = 0.0f;
	style.ChildRounding = 6.0f;
	style.FrameRounding = 4.0f;
	style.PopupRounding = 4.0f;
	style.ScrollbarRounding = 9.0f;
	style.GrabRounding = 4.0f;
	style.TabRounding = 4.0f;

	style.WindowPadding = ImVec2(12.0f, 12.0f);
	style.FramePadding = ImVec2(10.0f, 6.0f);
	style.ItemSpacing = ImVec2(8.0f, 6.0f);
	style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
	style.IndentSpacing = 20.0f;
	style.ScrollbarSize = 12.0f;
	style.GrabMinSize = 10.0f;

	style.WindowBorderSize = 0.0f;
	style.ChildBorderSize = 1.0f;
	style.PopupBorderSize = 1.0f;
	style.FrameBorderSize = 0.0f;
	style.TabBorderSize = 0.0f;

	style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
	style.ButtonTextAlign = ImVec2(0.5f, 0.5f);

	ImVec4* colors = style.Colors;
	colors[ImGuiCol_Text] = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.5f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.13f, 0.13f, 0.15f, 0.95f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.11f, 0.11f, 0.13f, 0.98f);
	colors[ImGuiCol_Border] = ImVec4(0.25f, 0.25f, 0.27f, 1.00f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.20f, 0.22f, 1.00f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.09f, 0.09f, 0.10f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.09f, 0.09f, 0.10f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.09f, 0.09f, 0.10f, 1.00f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.11f, 0.11f, 0.13f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.06f, 0.06f, 0.07f, 1.00f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.25f, 0.25f, 0.27f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.30f, 0.30f, 0.32f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.35f, 0.35f, 0.37f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.56f, 0.76f, 0.96f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.56f, 0.76f, 0.96f, 1.00f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.66f, 0.82f, 0.99f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.20f, 0.25f, 0.30f, 1.00f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.32f, 0.38f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.32f, 0.38f, 0.44f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.20f, 0.25f, 0.30f, 1.00f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.32f, 0.38f, 1.00f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.32f, 0.38f, 0.44f, 1.00f);
	colors[ImGuiCol_Separator] = ImVec4(0.20f, 0.20f, 0.22f, 1.00f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.26f, 0.32f, 0.38f, 1.00f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.32f, 0.38f, 0.44f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.20f, 0.25f, 0.30f, 1.00f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.32f, 0.38f, 1.00f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.32f, 0.38f, 0.44f, 1.00f);
	colors[ImGuiCol_Tab] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.32f, 0.38f, 1.00f);
	colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.25f, 0.30f, 1.00f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.11f, 0.11f, 0.13f, 1.00f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.56f, 0.76f, 0.96f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.66f, 0.82f, 0.99f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.56f, 0.76f, 0.96f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.66f, 0.82f, 0.99f, 1.00f);
	colors[ImGuiCol_TableHeaderBg] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
	colors[ImGuiCol_TableBorderStrong] = ImVec4(0.20f, 0.20f, 0.22f, 1.00f);
	colors[ImGuiCol_TableBorderLight] = ImVec4(0.18f, 0.18f, 0.20f, 1.00f);
	colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.15f, 0.15f, 0.17f, 0.30f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.32f, 0.38f, 0.50f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(0.56f, 0.76f, 0.96f, 0.95f);
	colors[ImGuiCol_NavHighlight] = ImVec4(0.56f, 0.76f, 0.96f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.56f, 0.76f, 0.96f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.20f, 0.20f, 0.22f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.10f, 0.10f, 0.12f, 0.50f);

		char font_path[MAX_PATH];
		if (SHGetFolderPathA(nullptr, CSIDL_FONTS, nullptr, SHGFP_TYPE_CURRENT, font_path) == S_OK)
		{
			std::string segoe_ui(font_path);
			segoe_ui += "\\segoeui.ttf";
			std::string segoe_ui_bold(font_path);
			segoe_ui_bold += "\\segoeuib.ttf";
			
			ImFontConfig font_config;
			font_config.OversampleH = 2;
			font_config.OversampleV = 2;
			font_config.PixelSnapH = true;
			
			ImFont* font = io.Fonts->AddFontFromFileTTF(segoe_ui.c_str(), 15.0f * main_scale, &font_config);
			if (!font)
			{
				font = io.Fonts->AddFontFromFileTTF(segoe_ui_bold.c_str(), 15.0f * main_scale, &font_config);
			}
			
			if (!font)
			{
				io.Fonts->AddFontDefault();
			}
		}
		else
		{
			io.Fonts->AddFontDefault();
		}

		if (!ImGui_ImplWin32_Init(hwnd_))
		{
			destroy_device();
			destroy_window();
			return;
		}

		if (!ImGui_ImplDX11_Init(device_, device_context_))
		{
			ImGui_ImplWin32_Shutdown();
			destroy_device();
			destroy_window();
			return;
		}

		
		HINSTANCE hInst = ::GetModuleHandle(nullptr);
		HICON hIconSmall = reinterpret_cast<HICON>(::LoadImage(hInst, MAKEINTRESOURCE(ID_ICON), IMAGE_ICON, 16, 16, 0));
		if (!hIconSmall)
		{
			hIconSmall = reinterpret_cast<HICON>(::LoadImage(hInst, MAKEINTRESOURCE(102), IMAGE_ICON, 16, 16, 0));
		}
		
		if (hIconSmall)
		{

			::SendMessage(hwnd_, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hIconSmall));
		}
		

		::ShowWindow(hwnd_, SW_SHOWDEFAULT);
		::UpdateWindow(hwnd_);

		::SetWindowPos(hwnd_, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);

		refresh_fastfiles();
		refresh_map_zones();
		refresh_csv_files();
		refresh_zonetool_maps();

		last_file_check_ = std::chrono::steady_clock::now();
		file_watcher_running_ = true;
		file_watcher_thread_ = std::thread([this]()
		{
			while (file_watcher_running_)
			{
				std::this_thread::sleep_for(std::chrono::seconds(2));
				
				if (!file_watcher_running_)
					break;
				
				const auto now = std::chrono::steady_clock::now();
				const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_file_check_).count();
				
				if (elapsed >= 2)
				{
					refresh_fastfiles();
					refresh_map_zones();
					refresh_csv_files();
					last_file_check_ = now;
				}
			}
		});

		while (running_ && !should_close_)
		{
			MSG msg;
			while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
			{
				::TranslateMessage(&msg);
				::DispatchMessage(&msg);
				if (msg.message == WM_QUIT)
				{
					should_close_ = true;
				}
			}

			if (should_close_)
			{
				break;
			}

			if (swap_chain_occluded_ && swap_chain_->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED)
			{
				::Sleep(10);
				continue;
			}
			swap_chain_occluded_ = false;

			if (resize_width_ != 0 && resize_height_ != 0)
			{
				destroy_render_target();
				swap_chain_->ResizeBuffers(0, resize_width_, resize_height_, DXGI_FORMAT_UNKNOWN, 0);
				create_render_target();


				set_viewport(resize_width_, resize_height_);

				resize_width_ = 0;
				resize_height_ = 0;
			}

			ImGui_ImplDX11_NewFrame();
			ImGui_ImplWin32_NewFrame();
			ImGui::NewFrame();

			render_frame();

			ImGui::Render();
			if (render_target_view_)
			{
				const float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
				device_context_->ClearRenderTargetView(render_target_view_, clear_color);
				ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
			}

			const HRESULT hr = swap_chain_->Present(1, 0);
			swap_chain_occluded_ = (hr == DXGI_STATUS_OCCLUDED);
		}

		ImGui_ImplDX11_Shutdown();
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();

		destroy_device();
		destroy_window();
	}

	bool component::create_window()
	{
		HINSTANCE hInst = ::GetModuleHandle(nullptr);
		
		WNDCLASSEXW wc = {};
		wc.cbSize = sizeof(wc);
		wc.style = CS_CLASSDC;
		wc.lpfnWndProc = wnd_proc;
		wc.hInstance = hInst;
		wc.lpszClassName = L"ZoneToolGUI";
		
		wchar_t exePath[MAX_PATH];
		::GetModuleFileNameW(nullptr, exePath, MAX_PATH);
		
		::ExtractIconExW(exePath, 0, &hicon_large_, &hicon_small_, 1);
		
		if (!hicon_small_)
		{
			hicon_small_ = reinterpret_cast<HICON>(::LoadImage(hInst, MAKEINTRESOURCE(ID_ICON), IMAGE_ICON, 16, 16, 0));
			if (!hicon_small_)
			{
				hicon_small_ = reinterpret_cast<HICON>(::LoadImage(hInst, MAKEINTRESOURCE(102), IMAGE_ICON, 16, 16, 0));
			}
		}
		if (!hicon_large_)
		{
			hicon_large_ = reinterpret_cast<HICON>(::LoadImage(hInst, MAKEINTRESOURCE(ID_ICON), IMAGE_ICON, 32, 32, 0));
			if (!hicon_large_)
			{
				hicon_large_ = reinterpret_cast<HICON>(::LoadImage(hInst, MAKEINTRESOURCE(102), IMAGE_ICON, 32, 32, 0));
			}
		}
		
		wc.hIcon = hicon_large_;
		wc.hIconSm = hicon_small_;
		
		::RegisterClassExW(&wc);

		hwnd_ = ::CreateWindowExW(0, wc.lpszClassName, L"ZoneTool", WS_OVERLAPPEDWINDOW, 100, 100, 1200, 800, nullptr, nullptr, wc.hInstance, nullptr);
		::SetWindowLongPtrW(hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

		if (hwnd_)
		{
			if (hicon_small_)
			{
				::SendMessage(hwnd_, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hicon_small_));
			}
			if (hicon_large_)
			{
				::SendMessage(hwnd_, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hicon_large_));
			}
			
			const BOOL value = TRUE;
			::DwmSetWindowAttribute(hwnd_, DWMWA_USE_IMMERSIVE_DARK_MODE, &value, sizeof(value));
			
			const int backdrop = DWMSBT_MICA;
			::DwmSetWindowAttribute(hwnd_, DWMWA_SYSTEMBACKDROP_TYPE, &backdrop, sizeof(backdrop));
			
			const BOOL extend_frame = TRUE;
			::DwmSetWindowAttribute(hwnd_, 9, &extend_frame, sizeof(extend_frame));
			
			HMODULE user32 = ::LoadLibraryW(L"user32.dll");
			if (user32)
			{
				pfnSetWindowCompositionAttribute set_window_composition_attribute = (pfnSetWindowCompositionAttribute)::GetProcAddress(user32, "SetWindowCompositionAttribute");
				if (set_window_composition_attribute)
				{
					ACCENTPOLICY policy = {};
					policy.nAccentState = 3;
					policy.nFlags = 0;
					policy.nColor = 0;
					policy.nAnimationId = 0;
					
					WINCOMPATTRDATA data = {};
					data.nAttribute = 19;
					data.pData = &policy;
					data.ulSizeOfData = sizeof(ACCENTPOLICY);
					set_window_composition_attribute(hwnd_, &data);
				}
				::FreeLibrary(user32);
			}

			::ShowWindow(hwnd_, SW_HIDE);
			::UpdateWindow(hwnd_);
		}

		return hwnd_ != nullptr;
	}

	void component::destroy_window()
	{
		if (hwnd_)
		{
			::DestroyWindow(hwnd_);
			hwnd_ = nullptr;
			::UnregisterClassW(L"ZoneToolGUI", ::GetModuleHandle(nullptr));
		}
	}

	bool component::create_device()
	{
		RECT rect = {};
		::GetClientRect(hwnd_, &rect);
		const UINT width = rect.right - rect.left;
		const UINT height = rect.bottom - rect.top;

		DXGI_SWAP_CHAIN_DESC sd = {};
		sd.BufferCount = 2;
		sd.BufferDesc.Width = width > 0 ? width : 1200;
		sd.BufferDesc.Height = height > 0 ? height : 800;
		sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		sd.BufferDesc.RefreshRate.Numerator = 60;
		sd.BufferDesc.RefreshRate.Denominator = 1;
		sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		sd.OutputWindow = hwnd_;
		sd.SampleDesc.Count = 1;
		sd.SampleDesc.Quality = 0;
		sd.Windowed = TRUE;
		sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
		sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

		D3D_FEATURE_LEVEL feature_level;
		const D3D_FEATURE_LEVEL feature_level_array[2] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
		HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, feature_level_array, 2, D3D11_SDK_VERSION, &sd, reinterpret_cast<IDXGISwapChain**>(&swap_chain_), &device_, &feature_level, &device_context_);
		if (res == DXGI_ERROR_UNSUPPORTED)
		{
			res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, feature_level_array, 2, D3D11_SDK_VERSION, &sd, reinterpret_cast<IDXGISwapChain**>(&swap_chain_), &device_, &feature_level, &device_context_);
		}
		if (res != S_OK)
		{
			return false;
		}

		create_render_target();

		set_viewport(width, height);

		return true;
	}

	void component::destroy_device()
	{
		destroy_render_target();
		if (swap_chain_)
		{
			swap_chain_->Release();
			swap_chain_ = nullptr;
		}
		if (device_context_)
		{
			device_context_->Release();
			device_context_ = nullptr;
		}
		if (device_)
		{
			device_->Release();
			device_ = nullptr;
		}
	}

	void component::create_render_target()
	{
		destroy_render_target();
		ID3D11Texture2D* back_buffer = nullptr;
		HRESULT res = swap_chain_->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&back_buffer));
		if (res == S_OK && back_buffer)
		{
			res = device_->CreateRenderTargetView(back_buffer, nullptr, &render_target_view_);
			back_buffer->Release();
		}
	}

	void component::destroy_render_target()
	{
		if (render_target_view_)
		{
			render_target_view_->Release();
			render_target_view_ = nullptr;
		}
	}

	void component::set_viewport(UINT width, UINT height)
	{
		if (!render_target_view_ || !device_context_)
		{
			return;
		}

		device_context_->OMSetRenderTargets(1, &render_target_view_, nullptr);

		D3D11_VIEWPORT viewport = {};
		viewport.Width = static_cast<float>(width > 0 ? width : 1200);
		viewport.Height = static_cast<float>(height > 0 ? height : 800);
		viewport.MinDepth = 0.0f;
		viewport.MaxDepth = 1.0f;
		viewport.TopLeftX = 0;
		viewport.TopLeftY = 0;
		device_context_->RSSetViewports(1, &viewport);
	}

	void component::render_frame()
	{
		render_main_window();
	}

	void component::render_main_window()
	{
		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
		ImGui::Begin("ZoneTool", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

		const auto mode = game::get_mode();
		const char* game_name = "Unknown";
		switch (mode)
		{
		case game::h1: game_name = "Modern Warfare Remastered"; break;
		case game::h2: game_name = "Modern Warfare 2 CR"; break;
		case game::s1: game_name = "Advanced Warfare"; break;
		case game::iw6: game_name = "Ghosts"; break;
		case game::iw7: game_name = "Infinite Warfare"; break;
		case game::t7: game_name = "Black Ops 3"; break;
		default: break;
		}

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(16, 12));
		ImGui::BeginChild("Header", ImVec2(0, 60), true, ImGuiWindowFlags_NoScrollbar);
		ImGui::SetCursorPosY(18);
		ImGui::SetWindowFontScale(1.3f);
		ImGui::Text("ZONETOOL");
		ImGui::SetWindowFontScale(1.0f);
		ImGui::SameLine();
		ImGui::SetCursorPosY(22);
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
		ImGui::Text("|");
		ImGui::PopStyleColor();
		ImGui::SameLine();
		ImGui::SetCursorPosY(22);
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.75f, 1.0f));
		ImGui::Text("%s", game_name);
		ImGui::PopStyleColor();
		ImGui::EndChild();
		ImGui::PopStyleVar();

		ImGui::Spacing();

		static int active_tab = 0;
		const float tab_width = 200.0f + 150.0f + 150.0f + 8.0f;
		const float tab_window_width = ImGui::GetContentRegionAvail().x;
		const float tab_center_offset = (tab_window_width - tab_width) / 2.0f;
		if (tab_center_offset > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + tab_center_offset);
		
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(20, 10));
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 0));
		if (ImGui::Button("DUMP", ImVec2(200, 0))) active_tab = 0;
		ImGui::SameLine();
		if (ImGui::Button("BUILD", ImVec2(150, 0))) active_tab = 1;
		ImGui::SameLine();
		if (ImGui::Button("CONSOLE", ImVec2(150, 0))) active_tab = 2;
		ImGui::PopStyleVar(2);

		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Spacing();

		const float card_width = 350.0f;
		const float card_height = 280.0f;
		const float card_spacing = 8.0f;
		const float content_width = (card_width * 3) + (card_spacing * 2);
		const float window_width = ImGui::GetContentRegionAvail().x;
		const float center_offset = (window_width - content_width) / 2.0f;

		if (operation_in_progress_)
		{
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
			ImGui::Text("%s", operation_status_text_.c_str());
			ImGui::PopStyleColor();
			ImGui::Spacing();
			ImGui::ProgressBar(-1.0f, ImVec2(-1, 0));
			ImGui::Spacing();
			ImGui::Separator();
			ImGui::Spacing();
		}

		if (active_tab == 0)
		{
			const auto current_mode = game::get_mode();
			
			if (center_offset > 0) ImGui::SetCursorPosX(center_offset);
			
			ImGui::BeginGroup();
			ImGui::BeginChild("FastfileCard", ImVec2(card_width, card_height), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("FASTFILES");
			ImGui::PopStyleColor();
			ImGui::Spacing();
			ImGui::Separator();
			ImGui::Spacing();
			ImGui::SetNextItemWidth(-80);
			char fastfile_filter_buf[256] = {};
			const size_t fastfile_filter_len = std::min(fastfile_filter_.size(), sizeof(fastfile_filter_buf) - 1);
			std::memcpy(fastfile_filter_buf, fastfile_filter_.data(), fastfile_filter_len);
			fastfile_filter_buf[fastfile_filter_len] = '\0';
			
			std::string fastfile_preview = "Select...";
			{
				std::lock_guard<std::mutex> lock(selected_fastfiles_batch_mutex_);
				if (!selected_fastfiles_batch_.empty())
				{
					fastfile_preview = std::to_string(selected_fastfiles_batch_.size()) + " selected";
				}
				else if (selected_fastfile_ >= 0)
				{
					std::lock_guard<std::mutex> lock_ff(fastfiles_mutex_);
					if (static_cast<size_t>(selected_fastfile_) < fastfiles_.size())
					{
						fastfile_preview = fastfiles_[selected_fastfile_];
					}
				}
			}
			
			const float combo_button_width = ImGui::CalcItemWidth();
			if (ImGui::BeginCombo("##fastfile", fastfile_preview.c_str()))
			{
				ImGui::SetNextItemWidth(combo_button_width);
				if (ImGui::InputText("##fastfile_filter", fastfile_filter_buf, sizeof(fastfile_filter_buf), ImGuiInputTextFlags_AutoSelectAll))
				{
					fastfile_filter_ = fastfile_filter_buf;
				}
				ImGui::Separator();
				
				float button_width = (combo_button_width - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
				if (ImGui::Button("Select All", ImVec2(button_width, 0)))
				{
					std::lock_guard<std::mutex> lock_ff(fastfiles_mutex_);
					std::lock_guard<std::mutex> lock(selected_fastfiles_batch_mutex_);
					std::string filter_lower = fastfile_filter_;
					std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
					for (size_t i = 0; i < fastfiles_.size(); ++i)
					{
						if (!filter_lower.empty())
						{
							std::string item_lower = fastfiles_[i];
							std::transform(item_lower.begin(), item_lower.end(), item_lower.begin(), ::tolower);
							if (item_lower.find(filter_lower) == std::string::npos)
								continue;
						}
						selected_fastfiles_batch_.insert(i);
					}
				}
				ImGui::SameLine();
				if (ImGui::Button("Clear All", ImVec2(button_width, 0)))
				{
					std::lock_guard<std::mutex> lock(selected_fastfiles_batch_mutex_);
					selected_fastfiles_batch_.clear();
					selected_fastfile_ = -1;
				}
				ImGui::Separator();
				
				ImGui::BeginChild("FastfileList", ImVec2(combo_button_width, 200), true);
				
				std::lock_guard<std::mutex> lock(fastfiles_mutex_);
				const size_t size = fastfiles_.size();
				std::string filter_lower = fastfile_filter_;
				std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
				
				for (size_t i = 0; i < size; ++i)
				{
					if (!filter_lower.empty())
					{
						std::string item_lower = fastfiles_[i];
						std::transform(item_lower.begin(), item_lower.end(), item_lower.begin(), ::tolower);
						if (item_lower.find(filter_lower) == std::string::npos)
							continue;
					}
					
					std::lock_guard<std::mutex> lock_batch(selected_fastfiles_batch_mutex_);
					bool is_selected_batch = selected_fastfiles_batch_.find(i) != selected_fastfiles_batch_.end();
					
					if (ImGui::Checkbox(fastfiles_[i].data(), &is_selected_batch))
					{
						if (is_selected_batch)
						{
							selected_fastfiles_batch_.insert(i);
						}
						else
						{
							selected_fastfiles_batch_.erase(i);
							if (selected_fastfile_ == static_cast<int>(i))
							{
								selected_fastfile_ = -1;
							}
						}
					}
				}
				ImGui::EndChild();
				ImGui::EndCombo();
			}
			else
			{
				fastfile_filter_.clear();
			}
			ImGui::SameLine();
			if (ImGui::Button("REFRESH", ImVec2(-1, 0)))
				refresh_fastfiles();
			ImGui::Spacing();
			
			ImGui::Text("Target Mode:");
			ImGui::SetNextItemWidth(-1);
			const char* target_modes[] = { "Current", "H1", "H2", "S1", "IW6", "IW7", "T7" };
			int target_mode_idx = 0;
			if (dump_target_mode_ != game::game_mode::none)
			{
				switch (dump_target_mode_)
				{
				case game::h1: target_mode_idx = 1; break;
				case game::h2: target_mode_idx = 2; break;
				case game::s1: target_mode_idx = 3; break;
				case game::iw6: target_mode_idx = 4; break;
				case game::iw7: target_mode_idx = 5; break;
				case game::t7: target_mode_idx = 6; break;
				default: target_mode_idx = 0; break;
				}
			}
			if (ImGui::Combo("##targetmode", &target_mode_idx, target_modes, IM_ARRAYSIZE(target_modes)))
			{
				switch (target_mode_idx)
				{
				case 0: dump_target_mode_ = game::game_mode::none; break;
				case 1: dump_target_mode_ = game::game_mode::h1; break;
				case 2: dump_target_mode_ = game::game_mode::h2; break;
				case 3: dump_target_mode_ = game::game_mode::s1; break;
				case 4: dump_target_mode_ = game::game_mode::iw6; break;
				case 5: dump_target_mode_ = game::game_mode::iw7; break;
				case 6: dump_target_mode_ = game::game_mode::t7; break;
				}
			}
			
			ImGui::Spacing();
			
			static bool show_asset_filter = false;
			if (ImGui::CollapsingHeader("Asset Filter", ImGuiTreeNodeFlags_DefaultOpen))
			{
				show_asset_filter = true;
				const auto asset_types = commands::get_available_asset_types();
				
				char asset_search_buf[256] = {};
				const size_t asset_search_len = std::min(asset_filter_search_.size(), sizeof(asset_search_buf) - 1);
				std::memcpy(asset_search_buf, asset_filter_search_.data(), asset_search_len);
				asset_search_buf[asset_search_len] = '\0';
				ImGui::SetNextItemWidth(-1);
				if (ImGui::InputText("##asset_search", asset_search_buf, sizeof(asset_search_buf)))
				{
					asset_filter_search_ = asset_search_buf;
				}
				if (ImGui::IsItemHovered())
				{
					ImGui::BeginTooltip();
					ImGui::Text("Filter asset types by name");
					ImGui::EndTooltip();
				}
				
				ImGui::Spacing();
				ImGui::BeginChild("AssetFilterList", ImVec2(-1, 100), true);
				
				std::string filter_lower = asset_filter_search_;
				std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
				
				for (const auto& asset_type : asset_types)
				{
					if (!filter_lower.empty())
					{
						std::string type_lower = asset_type;
						std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(), ::tolower);
						if (type_lower.find(filter_lower) == std::string::npos)
							continue;
					}
					
					bool is_selected = selected_asset_types_.find(asset_type) != selected_asset_types_.end();
					if (ImGui::Checkbox(asset_type.c_str(), &is_selected))
					{
						if (is_selected)
							selected_asset_types_.insert(asset_type);
						else
							selected_asset_types_.erase(asset_type);
					}
				}
				ImGui::EndChild();
				
				if (ImGui::Button("Clear Filter", ImVec2(-1, 0)))
				{
					selected_asset_types_.clear();
				}
			}
			else
			{
				show_asset_filter = false;
			}
			
			ImGui::Spacing();
			
			bool has_batch_selection = false;
			{
				std::lock_guard<std::mutex> lock(selected_fastfiles_batch_mutex_);
				has_batch_selection = !selected_fastfiles_batch_.empty();
			}
			
			if (has_batch_selection && !batch_dump_in_progress_)
			{
				if (ImGui::Button("DUMP SELECTED ZONES", ImVec2(-1, 28)))
				{
					std::lock_guard<std::mutex> lock_batch(selected_fastfiles_batch_mutex_);
					std::lock_guard<std::mutex> lock_ff(fastfiles_mutex_);
					
					std::vector<size_t> zones_to_dump;
					for (const auto& idx : selected_fastfiles_batch_)
					{
						if (idx < fastfiles_.size())
						{
							zones_to_dump.push_back(idx);
						}
					}
					
					if (!zones_to_dump.empty())
					{
						std::vector<std::string> zone_names;
						for (const auto& idx : zones_to_dump)
						{
							if (idx < fastfiles_.size())
							{
								zone_names.push_back(fastfiles_[idx]);
							}
						}
						
						if (!zone_names.empty())
						{
							batch_dump_total_ = zone_names.size();
							batch_dump_current_ = 0;
							batch_dump_in_progress_ = true;
							
							std::optional<std::unordered_set<std::string>> asset_filter = {};
							if (!selected_asset_types_.empty())
							{
								asset_filter = selected_asset_types_;
							}
							
							auto batch_target_mode = dump_target_mode_;
							
							batch_dump_thread_ = std::thread([this, zone_names, asset_filter, batch_target_mode]()
							{
								for (size_t i = 0; i < zone_names.size(); ++i)
								{
									batch_dump_current_ = i + 1;
									{
										std::lock_guard<std::mutex> lock_status(batch_dump_mutex_);
										batch_dump_current_zone_ = zone_names[i];
									}
									
									{
										std::lock_guard<std::mutex> lock_status(operation_status_mutex_);
										operation_status_text_ = "Batch dumping: " + std::to_string(i + 1) + "/" + std::to_string(zone_names.size()) + " - " + zone_names[i];
										operation_in_progress_ = true;
									}
									
									commands::dump_zone(zone_names[i], batch_target_mode, asset_filter);
								}
								
								batch_dump_in_progress_ = false;
								operation_in_progress_ = false;
								{
									std::lock_guard<std::mutex> lock_status(operation_status_mutex_);
									operation_status_text_ = "Batch dump complete: " + std::to_string(zone_names.size()) + " zones dumped";
								}
							});
							batch_dump_thread_.detach();
						}
					}
				}
			}
			else if (selected_fastfile_ >= 0 && !batch_dump_in_progress_)
			{
				if (ImGui::Button("DUMP SELECTED ZONE", ImVec2(-1, 28)))
				{
					std::lock_guard<std::mutex> lock(fastfiles_mutex_);
					const size_t size = fastfiles_.size();
					if (static_cast<size_t>(selected_fastfile_) < size)
					{
						std::optional<std::unordered_set<std::string>> asset_filter = {};
						if (!selected_asset_types_.empty())
						{
							asset_filter = selected_asset_types_;
						}
						commands::dump_zone(fastfiles_[selected_fastfile_], dump_target_mode_, asset_filter);
						operation_in_progress_ = true;
						operation_status_text_ = "Dumping zone: " + fastfiles_[selected_fastfile_];
					}
				}
			}
			else if (batch_dump_in_progress_)
			{
				ImGui::BeginDisabled();
				std::string batch_status;
				{
					std::lock_guard<std::mutex> lock(batch_dump_mutex_);
					batch_status = "DUMPING: " + std::to_string(batch_dump_current_.load()) + "/" + std::to_string(batch_dump_total_.load());
					if (!batch_dump_current_zone_.empty())
					{
						batch_status += " - " + batch_dump_current_zone_;
					}
				}
				ImGui::Button(batch_status.c_str(), ImVec2(-1, 28));
				ImGui::EndDisabled();
				
				float progress = 0.0f;
				if (batch_dump_total_ > 0)
				{
					progress = static_cast<float>(batch_dump_current_) / static_cast<float>(batch_dump_total_);
				}
				ImGui::ProgressBar(progress, ImVec2(-1, 0));
			}
			ImGui::EndChild();
			ImGui::EndGroup();

			ImGui::SameLine(0, card_spacing);

			ImGui::BeginGroup();
			ImGui::BeginChild("ZoneOpsCard", ImVec2(card_width, card_height), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("ZONE OPERATIONS");
			ImGui::PopStyleColor();
			ImGui::Spacing();
			ImGui::Separator();
			ImGui::Spacing();
			if (ImGui::Button("LOAD ZONE", ImVec2(-1, 28)))
			{
				if (selected_fastfile_ >= 0)
				{
					std::lock_guard<std::mutex> lock(fastfiles_mutex_);
					const size_t size = fastfiles_.size();
					if (static_cast<size_t>(selected_fastfile_) < size)
						commands::load_zone(fastfiles_[selected_fastfile_]);
				}
			}
			if (ImGui::Button("UNLOAD ALL ZONES", ImVec2(-1, 28)))
			{
				commands::unload_zones();
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Unloads all zones including default zones\nloaded at startup (common, code_post_gfx, etc.)");
				ImGui::EndTooltip();
			}
			if (ImGui::Button("VERIFY ZONE", ImVec2(-1, 28)))
			{
				if (selected_fastfile_ >= 0)
				{
					std::lock_guard<std::mutex> lock(fastfiles_mutex_);
					const size_t size = fastfiles_.size();
					if (static_cast<size_t>(selected_fastfile_) < size)
						commands::verify_zone(fastfiles_[selected_fastfile_]);
				}
			}
			ImGui::Spacing();
			const bool iterate_zones_available = (current_mode == game::h1 || current_mode == game::iw7 || current_mode == game::t7);
			if (!iterate_zones_available) ImGui::BeginDisabled();
			if (ImGui::Button("ITERATE ZONES", ImVec2(-1, 28)))
			{
				commands::iterate_zones();
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				if (current_mode == game::t7)
				{
					ImGui::Text("Available in H1, IW7, T7.\nIterates through all .ff files in zone/ directory, loading and unloading each.");
				}
				else
				{
					ImGui::Text("Available in H1, IW7, T7.\nIterates through all .ff files in zone/ and zone/english/ directories, loading and unloading each.");
				}
				ImGui::EndTooltip();
			}
			if (!iterate_zones_available) ImGui::EndDisabled();
			ImGui::EndChild();
			ImGui::EndGroup();

			ImGui::SameLine(0, card_spacing);

			ImGui::BeginGroup();
			ImGui::BeginChild("DumpMapCard", ImVec2(card_width, card_height), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("DUMP MAP");
			ImGui::PopStyleColor();
			ImGui::Spacing();
			ImGui::Separator();
			ImGui::Spacing();
			ImGui::SetNextItemWidth(-1);
			char map_zone_filter_buf[256] = {};
			const size_t map_zone_filter_len = std::min(map_zone_filter_.size(), sizeof(map_zone_filter_buf) - 1);
			std::memcpy(map_zone_filter_buf, map_zone_filter_.data(), map_zone_filter_len);
			map_zone_filter_buf[map_zone_filter_len] = '\0';
			
			if (ImGui::BeginCombo("##mapzone", selected_map_zone_ >= 0 ? map_zones_[selected_map_zone_].data() : "Select map..."))
			{
				ImGui::SetNextItemWidth(-1);
				if (ImGui::InputText("##mapzone_filter", map_zone_filter_buf, sizeof(map_zone_filter_buf), ImGuiInputTextFlags_AutoSelectAll))
				{
					map_zone_filter_ = map_zone_filter_buf;
				}
				ImGui::Separator();
				
				std::lock_guard<std::mutex> lock(map_zones_mutex_);
				const size_t size = map_zones_.size();
				std::string filter_lower = map_zone_filter_;
				std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
				
				for (size_t i = 0; i < size; ++i)
				{
					if (!filter_lower.empty())
					{
						std::string item_lower = map_zones_[i];
						std::transform(item_lower.begin(), item_lower.end(), item_lower.begin(), ::tolower);
						if (item_lower.find(filter_lower) == std::string::npos)
							continue;
					}
					
					const bool is_selected = (selected_map_zone_ == static_cast<int>(i));
					if (ImGui::Selectable(map_zones_[i].data(), is_selected))
					{
						selected_map_zone_ = static_cast<int>(i);
						map_zone_filter_.clear();
					}
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			else
			{
				map_zone_filter_.clear();
			}
			
			ImGui::Spacing();
			
			ImGui::Checkbox("Skip Common Assets", &dump_map_skip_common_);
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Skip common assets when dumping maps");
				ImGui::EndTooltip();
			}
			
			ImGui::Spacing();
			
			if (ImGui::Button("DUMP MAP", ImVec2(-1, 28)))
			{
				if (selected_map_zone_ >= 0)
				{
					std::lock_guard<std::mutex> lock(map_zones_mutex_);
					const size_t size = map_zones_.size();
					if (static_cast<size_t>(selected_map_zone_) < size)
					{
						std::optional<std::unordered_set<std::string>> asset_filter = {};
						if (!selected_asset_types_.empty())
						{
							asset_filter = selected_asset_types_;
						}
						commands::dump_map(map_zones_[selected_map_zone_], dump_target_mode_, asset_filter, dump_map_skip_common_);
						operation_in_progress_ = true;
						operation_status_text_ = "Dumping map: " + map_zones_[selected_map_zone_];
					}
				}
			}
			ImGui::EndChild();
			ImGui::EndGroup();

			if (center_offset > 0) ImGui::SetCursorPosX(center_offset);

			const bool dump_csv_available = (current_mode == game::h1 || current_mode == game::s1 || current_mode == game::iw7 || current_mode == game::t7);
			if (!dump_csv_available) ImGui::BeginDisabled();
			ImGui::BeginGroup();
			ImGui::BeginChild("DumpCsvCard", ImVec2(card_width, card_height), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("DUMP CSV");
			ImGui::PopStyleColor();
			ImGui::Spacing();
			ImGui::Separator();
			ImGui::Spacing();
			ImGui::SetNextItemWidth(-1);
			char csv_filter_buf[256] = {};
			const size_t csv_filter_len = std::min(csv_filter_.size(), sizeof(csv_filter_buf) - 1);
			std::memcpy(csv_filter_buf, csv_filter_.data(), csv_filter_len);
			csv_filter_buf[csv_filter_len] = '\0';
			
			if (ImGui::BeginCombo("##csvzone", selected_fastfile_ >= 0 ? fastfiles_[selected_fastfile_].data() : "Select zone..."))
			{
				ImGui::SetNextItemWidth(-1);
				if (ImGui::InputText("##csvzone_filter", csv_filter_buf, sizeof(csv_filter_buf), ImGuiInputTextFlags_AutoSelectAll))
				{
					csv_filter_ = csv_filter_buf;
				}
				ImGui::Separator();
				
				std::lock_guard<std::mutex> lock(fastfiles_mutex_);
				const size_t size = fastfiles_.size();
				std::string filter_lower = csv_filter_;
				std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
				
				for (size_t i = 0; i < size; ++i)
				{
					if (!filter_lower.empty())
					{
						std::string item_lower = fastfiles_[i];
						std::transform(item_lower.begin(), item_lower.end(), item_lower.begin(), ::tolower);
						if (item_lower.find(filter_lower) == std::string::npos)
							continue;
					}
					
					const bool is_selected = (selected_fastfile_ == static_cast<int>(i));
					if (ImGui::Selectable(fastfiles_[i].data(), is_selected))
					{
						selected_fastfile_ = static_cast<int>(i);
						csv_filter_.clear();
					}
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			else
			{
				csv_filter_.clear();
			}
			if (ImGui::Button("DUMP CSV", ImVec2(-1, 28)))
			{
				if (selected_fastfile_ >= 0)
				{
					std::lock_guard<std::mutex> lock(fastfiles_mutex_);
					const size_t size = fastfiles_.size();
					if (static_cast<size_t>(selected_fastfile_) < size)
						commands::dump_csv(fastfiles_[selected_fastfile_]);
				}
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Available in H1, S1, IW7, T7.\nDumps a CSV file listing all assets in a zone.");
				ImGui::EndTooltip();
			}
			ImGui::EndChild();
			ImGui::EndGroup();
			if (!dump_csv_available)
			{
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
				{
					ImGui::BeginTooltip();
					ImGui::Text("Not available in current game mode.\nAvailable in H1, S1, IW7, T7.");
					ImGui::EndTooltip();
				}
				ImGui::EndDisabled();
			}

			ImGui::SameLine(0, card_spacing);

			ImGui::BeginGroup();
			ImGui::BeginChild("DumpAssetCard", ImVec2(card_width, card_height), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("DUMP ASSET");
			ImGui::PopStyleColor();
			ImGui::Spacing();
			ImGui::Separator();
			ImGui::Spacing();
			ImGui::Text("Asset Type:");
			ImGui::SetNextItemWidth(-1);
			const auto asset_types = commands::get_available_asset_types();
			if (ImGui::BeginCombo("##assettype", dump_asset_type_.empty() ? "Select type..." : dump_asset_type_.c_str()))
			{
				std::string filter_lower;
				char asset_type_search_buf[256] = {};
				ImGui::SetNextItemWidth(-1);
				if (ImGui::InputText("##assettype_search", asset_type_search_buf, sizeof(asset_type_search_buf), ImGuiInputTextFlags_AutoSelectAll))
				{
					filter_lower = asset_type_search_buf;
					std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
				}
				ImGui::Separator();
				
				for (const auto& asset_type : asset_types)
				{
					if (!filter_lower.empty())
					{
						std::string type_lower = asset_type;
						std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(), ::tolower);
						if (type_lower.find(filter_lower) == std::string::npos)
							continue;
					}
					
					if (ImGui::Selectable(asset_type.c_str()))
					{
						dump_asset_type_ = asset_type;
					}
				}
				ImGui::EndCombo();
			}
			ImGui::Text("Asset Name:");
			ImGui::SetNextItemWidth(-1);
			char asset_name_buf[256] = {};
			const size_t asset_name_len = std::min(dump_asset_name_.size(), sizeof(asset_name_buf) - 1);
			std::memcpy(asset_name_buf, dump_asset_name_.data(), asset_name_len);
			asset_name_buf[asset_name_len] = '\0';
			if (ImGui::InputText("##assetname", asset_name_buf, sizeof(asset_name_buf)))
			{
				dump_asset_name_ = asset_name_buf;
			}
			ImGui::Spacing();
			if (ImGui::Button("DUMP ASSET", ImVec2(-1, 28)))
			{
				if (!dump_asset_type_.empty() && !dump_asset_name_.empty())
				{
					commands::dump_asset(dump_asset_type_, dump_asset_name_);
					operation_in_progress_ = true;
					operation_status_text_ = "Dumping asset: " + dump_asset_type_ + " / " + dump_asset_name_;
				}
			}
			ImGui::EndChild();
			ImGui::EndGroup();

			ImGui::SameLine(0, card_spacing);

			const bool dump_matching_zones_available = (current_mode == game::h1 || current_mode == game::s1);
			if (!dump_matching_zones_available) ImGui::BeginDisabled();
			ImGui::BeginGroup();
			ImGui::BeginChild("DumpMatchingZonesCard", ImVec2(card_width, card_height), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("DUMP MATCHING ZONES");
			ImGui::PopStyleColor();
			ImGui::Spacing();
			ImGui::Separator();
			ImGui::Spacing();
			ImGui::Text("Regex Pattern:");
			ImGui::SetNextItemWidth(-1);
			char pattern_buf[256] = {};
			const size_t pattern_len = std::min(dump_matching_zones_pattern_.size(), sizeof(pattern_buf) - 1);
			std::memcpy(pattern_buf, dump_matching_zones_pattern_.data(), pattern_len);
			pattern_buf[pattern_len] = '\0';
			if (ImGui::InputText("##pattern", pattern_buf, sizeof(pattern_buf)))
			{
				dump_matching_zones_pattern_ = pattern_buf;
			}
			ImGui::Text("Target Mode:");
			ImGui::SetNextItemWidth(-1);
			const char* target_modes_match[] = { "Current", "H1", "H2", "S1", "IW6", "IW7", "T7" };
			int target_mode_idx_match = 0;
			if (dump_target_mode_ != game::game_mode::none)
			{
				switch (dump_target_mode_)
				{
				case game::h1: target_mode_idx_match = 1; break;
				case game::h2: target_mode_idx_match = 2; break;
				case game::s1: target_mode_idx_match = 3; break;
				case game::iw6: target_mode_idx_match = 4; break;
				case game::iw7: target_mode_idx_match = 5; break;
				case game::t7: target_mode_idx_match = 6; break;
				default: target_mode_idx_match = 0; break;
				}
			}
			if (ImGui::Combo("##targetmode_match", &target_mode_idx_match, target_modes_match, IM_ARRAYSIZE(target_modes_match)))
			{
				switch (target_mode_idx_match)
				{
				case 0: dump_target_mode_ = game::game_mode::none; break;
				case 1: dump_target_mode_ = game::game_mode::h1; break;
				case 2: dump_target_mode_ = game::game_mode::h2; break;
				case 3: dump_target_mode_ = game::game_mode::s1; break;
				case 4: dump_target_mode_ = game::game_mode::iw6; break;
				case 5: dump_target_mode_ = game::game_mode::iw7; break;
				case 6: dump_target_mode_ = game::game_mode::t7; break;
				}
			}
			ImGui::Spacing();
			if (ImGui::Button("DUMP MATCHING", ImVec2(-1, 28)))
			{
				if (!dump_matching_zones_pattern_.empty())
				{
					std::optional<std::unordered_set<std::string>> asset_filter = {};
					if (!selected_asset_types_.empty())
					{
						asset_filter = selected_asset_types_;
					}
					commands::dump_matching_zones(dump_matching_zones_pattern_, dump_target_mode_, asset_filter);
					operation_in_progress_ = true;
					operation_status_text_ = "Dumping matching zones: " + dump_matching_zones_pattern_;
				}
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Available in H1, S1.\nDumps zones matching a regex pattern with optional target game conversion and asset filtering.");
				ImGui::EndTooltip();
			}
			ImGui::EndChild();
			ImGui::EndGroup();
			if (!dump_matching_zones_available)
			{
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
				{
					ImGui::BeginTooltip();
					ImGui::Text("Not available in current game mode.\nAvailable in H1, S1.");
					ImGui::EndTooltip();
				}
				ImGui::EndDisabled();
			}
		}
		else if (active_tab == 1)
		{
			const auto current_mode = game::get_mode();
			const float build_window_width = ImGui::GetContentRegionAvail().x;
			const float build_card_width = build_window_width - 40.0f;
			const float build_card_height = 120.0f;
			

			ImGui::BeginChild("BuildCard", ImVec2(build_card_width, build_card_height), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("BUILD ZONE");
			ImGui::PopStyleColor();
			ImGui::SameLine();
			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 20.0f);
			ImGui::Text("CSV File:");
			ImGui::SameLine();
			ImGui::SetNextItemWidth(300.0f);
			if (ImGui::BeginCombo("##csvfile", selected_csv_ >= 0 ? csv_files_[selected_csv_].data() : "Select CSV..."))
			{
				std::lock_guard<std::mutex> lock(csv_files_mutex_);
				const size_t size = csv_files_.size();
				for (size_t i = 0; i < size; ++i)
				{
					const bool is_selected = (selected_csv_ == static_cast<int>(i));
					if (ImGui::Selectable(csv_files_[i].data(), is_selected))
						selected_csv_ = static_cast<int>(i);
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			ImGui::SameLine();
			ImGui::SetCursorPosX(build_card_width - 150.0f);
			if (ImGui::Button("BUILD ZONE", ImVec2(140.0f, 32)))
			{
				if (selected_csv_ >= 0)
				{
					std::lock_guard<std::mutex> lock(csv_files_mutex_);
					const size_t size = csv_files_.size();
					if (static_cast<size_t>(selected_csv_) < size)
						commands::build_zone(csv_files_[selected_csv_]);
				}
			}
			ImGui::EndChild();

			ImGui::Spacing();

			const bool generate_csv_available = (current_mode == game::h1 || current_mode == game::h2 || current_mode == game::s1 || current_mode == game::iw6 || current_mode == game::iw7);
			if (!generate_csv_available) ImGui::BeginDisabled();
			ImGui::BeginChild("GenerateCsvCard", ImVec2(build_card_width, build_card_height), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("GENERATE CSV");
			ImGui::PopStyleColor();
			ImGui::SameLine();
			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 20.0f);
			ImGui::Text("Map:");
			ImGui::SameLine();
			ImGui::SetNextItemWidth(300.0f);
			if (ImGui::BeginCombo("##generatecsvmap", selected_generate_csv_map_ >= 0 ? zonetool_maps_[selected_generate_csv_map_].data() : "Select map..."))
			{
				std::lock_guard<std::mutex> lock(zonetool_maps_mutex_);
				const size_t size = zonetool_maps_.size();
				for (size_t i = 0; i < size; ++i)
				{
					const bool is_selected = (selected_generate_csv_map_ == static_cast<int>(i));
					if (ImGui::Selectable(zonetool_maps_[i].data(), is_selected))
						selected_generate_csv_map_ = static_cast<int>(i);
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			ImGui::SameLine();
			ImGui::Checkbox("Single Player Mode", &generate_csv_sp_mode_);
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Enable this for single-player maps.");
				ImGui::EndTooltip();
			}
			ImGui::SameLine();
			ImGui::SetCursorPosX(build_card_width - 150.0f);
			if (ImGui::Button("GENERATE CSV", ImVec2(140.0f, 32)))
			{
				if (selected_generate_csv_map_ >= 0)
				{
					std::lock_guard<std::mutex> lock(zonetool_maps_mutex_);
					const size_t size = zonetool_maps_.size();
					if (static_cast<size_t>(selected_generate_csv_map_) < size)
						commands::generate_csv(zonetool_maps_[selected_generate_csv_map_], generate_csv_sp_mode_);
				}
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Available in H1, H2, S1, IW6, IW7.\nGenerates a CSV file for a map by parsing mapents, GSC files, and assets.");
				ImGui::EndTooltip();
			}
			ImGui::EndChild();
			if (!generate_csv_available)
			{
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
				{
					ImGui::BeginTooltip();
					ImGui::Text("Not available in current game mode.\nAvailable in H1, H2, S1, IW6, IW7.");
					ImGui::EndTooltip();
				}
				ImGui::EndDisabled();
			}

			ImGui::Spacing();

			const bool build_xmodel_available = (current_mode == game::h2);
			if (!build_xmodel_available) ImGui::BeginDisabled();
			ImGui::BeginChild("BuildXModelCard", ImVec2(build_card_width, build_card_height), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("BUILD XMODEL");
			ImGui::PopStyleColor();
			ImGui::SameLine();
			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 20.0f);
			ImGui::Text("CSV File:");
			ImGui::SameLine();
			ImGui::SetNextItemWidth(300.0f);
			if (ImGui::BeginCombo("##xmodelcsv", selected_csv_ >= 0 ? csv_files_[selected_csv_].data() : "Select CSV..."))
			{
				std::lock_guard<std::mutex> lock(csv_files_mutex_);
				const size_t size = csv_files_.size();
				for (size_t i = 0; i < size; ++i)
				{
					const bool is_selected = (selected_csv_ == static_cast<int>(i));
					if (ImGui::Selectable(csv_files_[i].data(), is_selected))
						selected_csv_ = static_cast<int>(i);
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			ImGui::SameLine();
			ImGui::SetCursorPosX(build_card_width - 150.0f);
			if (ImGui::Button("BUILD XMODEL", ImVec2(140.0f, 32)))
			{
				if (selected_csv_ >= 0)
				{
					std::lock_guard<std::mutex> lock(csv_files_mutex_);
					const size_t size = csv_files_.size();
					if (static_cast<size_t>(selected_csv_) < size)
						commands::build_xmodel(csv_files_[selected_csv_]);
				}
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Available in H2 only.\nBuilds a composite xmodel from a CSV file in zone_source/.");
				ImGui::EndTooltip();
			}
			ImGui::EndChild();
			if (!build_xmodel_available)
			{
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
				{
					ImGui::BeginTooltip();
					ImGui::Text("Not available in current game mode.\nAvailable in H2 only.");
					ImGui::EndTooltip();
				}
				ImGui::EndDisabled();
			}
		}
		else if (active_tab == 2)
		{
			ImGui::BeginChild("ConsoleCard", ImVec2(0, 0), true);
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
			ImGui::Text("OUTPUT LOG");
			ImGui::PopStyleColor();
			ImGui::SameLine();
			ImGui::Checkbox("Auto-scroll", &auto_scroll_log_);
			ImGui::SameLine();
			ImGui::Checkbox("Show Errors/Warnings Only", &filter_log_errors_warnings_);
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Filter log to show only ERROR and WARNING messages");
				ImGui::EndTooltip();
			}
			ImGui::SameLine();
			if (ImGui::Button("CLEAR"))
			{
				std::lock_guard<std::mutex> lock(log_mutex_);
				log_messages_.clear();
			}
			ImGui::Spacing();
			ImGui::Separator();
			ImGui::Spacing();
			ImGui::BeginChild("LogContent", ImVec2(0, -40), false, ImGuiWindowFlags_HorizontalScrollbar);
			{
				std::vector<std::string> messages_copy;
				{
					std::lock_guard<std::mutex> lock(log_mutex_);
					messages_copy.reserve(log_messages_.size());
					messages_copy.assign(log_messages_.begin(), log_messages_.end());
				}
				
				for (const auto& msg : messages_copy)
				{
					if (filter_log_errors_warnings_)
					{
						if (msg.find("[ ERROR ]") == std::string::npos && 
						    msg.find("[ FATAL ]") == std::string::npos && 
						    msg.find("[ WARNING ]") == std::string::npos)
						{
							continue;
						}
					}
					
					if (msg.find("[ ERROR ]") != std::string::npos || msg.find("[ FATAL ]") != std::string::npos)
					{
						ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
						ImGui::TextUnformatted(msg.c_str());
						ImGui::PopStyleColor();
					}
					else if (msg.find("[ WARNING ]") != std::string::npos)
					{
						ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.4f, 1.0f));
						ImGui::TextUnformatted(msg.c_str());
						ImGui::PopStyleColor();
					}
					else if (msg.find("[ INFO ]") != std::string::npos)
					{
						ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
						ImGui::TextUnformatted(msg.c_str());
						ImGui::PopStyleColor();
					}
					else
					{
						ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.9f, 0.95f, 1.0f));
						ImGui::TextUnformatted(msg.c_str());
						ImGui::PopStyleColor();
					}
				}
				
				if (auto_scroll_log_ && !messages_copy.empty())
					ImGui::SetScrollHereY(1.0f);
			}
			ImGui::EndChild();
			ImGui::Separator();
			ImGui::SetNextItemWidth(-1);
			char command_buf[512] = {};
			const size_t copy_len = std::min(command_input_text_.size(), sizeof(command_buf) - 1);
			std::memcpy(command_buf, command_input_text_.data(), copy_len);
			command_buf[copy_len] = '\0';
			if (ImGui::InputText("##command_input", command_buf, sizeof(command_buf), ImGuiInputTextFlags_EnterReturnsTrue))
			{
				if (command_buf[0] != '\0')
				{
					const std::string command = command_buf;
					command_input_text_ = "";
					std::lock_guard<std::mutex> lock(log_mutex_);
					log_messages_.emplace_back("> ");
					log_messages_.back() += command;
					const auto cmd_mode = game::get_mode();
					switch (cmd_mode)
					{
					case game::h1: ::h1::command::execute(command); break;
					case game::h2: ::h2::command::execute(command); break;
					case game::s1: ::s1::command::execute(command); break;
					case game::iw6: ::iw6::command::execute(command); break;
					case game::iw7: ::iw7::command::execute(command); break;
					case game::t7: ::t7::command::execute(command); break;
					default: break;
					}
				}
			}
			else
				command_input_text_ = command_buf;
			ImGui::EndChild();
		}

		ImGui::End();
	}

	void component::refresh_fastfiles()
	{
		const auto fastfiles = commands::discover_fastfiles();
		std::lock_guard<std::mutex> lock(fastfiles_mutex_);
		fastfiles_ = fastfiles;
		const size_t size = fastfiles_.size();
		if (static_cast<size_t>(selected_fastfile_) >= size)
		{
			selected_fastfile_ = -1;
		}
	}

	void component::refresh_map_zones()
	{
		const auto map_zones = commands::discover_map_zones();
		std::lock_guard<std::mutex> lock(map_zones_mutex_);
		map_zones_ = map_zones;
		const size_t size = map_zones_.size();
		if (static_cast<size_t>(selected_map_zone_) >= size)
		{
			selected_map_zone_ = -1;
		}
	}

	void component::refresh_csv_files()
	{
		const auto csv_files = commands::discover_csv_files();
		std::lock_guard<std::mutex> lock(csv_files_mutex_);
		csv_files_ = csv_files;
		const size_t size = csv_files_.size();
		if (static_cast<size_t>(selected_csv_) >= size)
		{
			selected_csv_ = -1;
		}
	}

	void component::refresh_zonetool_maps()
	{
		const auto maps = commands::discover_zonetool_maps();
		std::lock_guard<std::mutex> lock(zonetool_maps_mutex_);
		zonetool_maps_ = maps;
		const size_t size = zonetool_maps_.size();
		if (static_cast<size_t>(selected_generate_csv_map_) >= size)
		{
			selected_generate_csv_map_ = -1;
		}
	}

	LRESULT WINAPI component::wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
	{
		if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam))
		{
			return true;
		}

		auto* comp = reinterpret_cast<gui::component*>(::GetWindowLongPtrW(hwnd, GWLP_USERDATA));
		if (!comp)
		{
			return ::DefWindowProcW(hwnd, msg, wparam, lparam);
		}

		switch (msg)
		{
		case WM_GETICON:
			if (wparam == ICON_SMALL || wparam == ICON_SMALL2)
			{
				if (comp->hicon_small_)
				{
					return reinterpret_cast<LRESULT>(comp->hicon_small_);
				}
			}
			else if (wparam == ICON_BIG)
			{
				if (comp->hicon_large_)
				{
					return reinterpret_cast<LRESULT>(comp->hicon_large_);
				}
			}
			break;
		case WM_CLOSE:
			comp->should_close_ = true;
			::DestroyWindow(hwnd);
			std::quick_exit(EXIT_SUCCESS);
		case WM_SIZE:
			if (wparam == SIZE_MINIMIZED)
			{
				return 0;
			}
			comp->resize_width_ = static_cast<UINT>(LOWORD(lparam));
			comp->resize_height_ = static_cast<UINT>(HIWORD(lparam));
			return 0;
		case WM_SYSCOMMAND:
			if ((wparam & 0xfff0) == SC_KEYMENU)
			{
				return 0;
			}
			break;
		case WM_DESTROY:
			comp->should_close_ = true;
			::PostQuitMessage(0);
			return 0;
		}

		return ::DefWindowProcW(hwnd, msg, wparam, lparam);
	}
}

REGISTER_COMPONENT(gui::component)

