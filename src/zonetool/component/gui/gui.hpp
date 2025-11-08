#pragma once

#include "loader/component_interface.hpp"
#include "gui_commands.hpp"
#include "game/mode.hpp"

#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_set>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct IDXGISwapChain;
struct ID3D11RenderTargetView;

struct HWND__;
typedef HWND__* HWND;

namespace gui
{
	class component;

	component* get_gui_component();

	class component final : public component_interface
	{
	public:
		component();
		~component() override;

		void post_start() override;
		void post_load() override;
		void pre_destroy() override;

	private:
		void gui_thread();
		bool create_window();
		void destroy_window();
		bool create_device();
		void destroy_device();
		void create_render_target();
		void destroy_render_target();
		void set_viewport(UINT width, UINT height);
		void render_frame();
		void render_main_window();
		void refresh_fastfiles();
		void refresh_map_zones();
		void refresh_csv_files();
		void refresh_zonetool_maps();

		static LRESULT WINAPI wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

		HWND hwnd_;
		HICON hicon_small_;
		HICON hicon_large_;
		ID3D11Device* device_;
		ID3D11DeviceContext* device_context_;
		IDXGISwapChain* swap_chain_;
		ID3D11RenderTargetView* render_target_view_;

		std::thread gui_thread_;
		std::atomic<bool> running_;
		std::atomic<bool> should_close_;

		std::vector<std::string> fastfiles_;
		std::mutex fastfiles_mutex_;
		int selected_fastfile_;
		std::unordered_set<size_t> selected_fastfiles_batch_;
		std::mutex selected_fastfiles_batch_mutex_;
		
		std::atomic<bool> batch_dump_in_progress_;
		std::atomic<size_t> batch_dump_current_;
		std::atomic<size_t> batch_dump_total_;
		std::string batch_dump_current_zone_;
		std::mutex batch_dump_mutex_;
		std::thread batch_dump_thread_;

		std::vector<std::string> map_zones_;
		std::mutex map_zones_mutex_;
		int selected_map_zone_;

		std::vector<std::string> csv_files_;
		std::mutex csv_files_mutex_;
		int selected_csv_;

		std::vector<std::string> zonetool_maps_;
		std::mutex zonetool_maps_mutex_;
		int selected_generate_csv_map_;
		bool generate_csv_sp_mode_;

		std::string dump_matching_zones_pattern_;

		std::deque<std::string> log_messages_;
		std::mutex log_mutex_;
		bool auto_scroll_log_;
		bool filter_log_errors_warnings_;
		std::string command_input_text_;

		std::atomic<bool> operation_in_progress_;
		std::string operation_status_text_;
		std::mutex operation_status_mutex_;

		std::unordered_set<std::string> selected_asset_types_;
		std::string asset_filter_search_;

		game::game_mode dump_target_mode_;

		std::string dump_asset_type_;
		std::string dump_asset_name_;

		bool dump_map_skip_common_;

		std::string fastfile_filter_;
		std::string csv_filter_;
		std::string map_zone_filter_;

		UINT resize_width_;
		UINT resize_height_;
		bool swap_chain_occluded_;

		std::thread file_watcher_thread_;
		std::atomic<bool> file_watcher_running_;
		std::chrono::steady_clock::time_point last_file_check_;

	public:
		void add_log_message(const std::string& message);
	};

}

