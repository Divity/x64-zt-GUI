#pragma once

// Whether BO3 data behind a pointer can be read (loaded zones leave some pointers dangling). Reads the bytes under a
// structured exception handler; VirtualQuery is too slow with this process's large address space map.

namespace zonetool::t7
{
	namespace converter::iw7::probe
	{
		// the first and the last byte of [ptr, ptr + size) can be read
		bool readable(const void* ptr, std::size_t size);

		// `text` when it is 1-255 printable characters (0x20-0x7E) and a NUL, else nullptr
		const char* name(const char* text);

		// `text` when a NUL follows 1-255 characters of any value, else nullptr
		const char* terminated(const char* text);
	}
}
