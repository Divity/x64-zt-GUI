#pragma once

#include <d3d11.h>
#include <mutex>
#include <wrl/client.h>

#pragma comment(lib, "d3d11.lib")

namespace zonetool::t7
{
	namespace converter::iw7
	{
		// calls on the device's immediate context (DirectXTex's GPU block compressors) take this
		inline std::mutex gpu_mutex;

		// the hardware device BC6H / BC7 are encoded on, null when there is none
		inline ID3D11Device* gpu_device()
		{
			static std::once_flag once;
			static Microsoft::WRL::ComPtr<ID3D11Device> device;
			std::call_once(once, []
			{
				const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };
				if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, levels, 1, D3D11_SDK_VERSION,
					device.GetAddressOf(), nullptr, nullptr)))
				{
					device.Reset();
					ZONETOOL_INFO("no Direct3D 11 device: BC6H and BC7 are encoded on the CPU");
				}
			});
			return device.Get();
		}
	}
}
