#include <std_include.hpp>
#include "memory_probe.hpp"

namespace zonetool::t7
{
	namespace converter::iw7::probe
	{
		namespace
		{
			int filter(const unsigned long code)
			{
				return code == EXCEPTION_ACCESS_VIOLATION || code == STATUS_GUARD_PAGE_VIOLATION
					? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH;
			}

			bool touch(const volatile std::uint8_t* first, const volatile std::uint8_t* last)
			{
				__try
				{
					static_cast<void>(*first);
					static_cast<void>(*last);
					return true;
				}
				__except (filter(GetExceptionCode()))
				{
					return false;
				}
			}

			// the string's length (1-255), or -1
			int length(const char* text, const bool printable)
			{
				__try
				{
					for (auto i = 0; i < 256; i++)
					{
						const auto c = static_cast<unsigned char>(text[i]);
						if (!c)
						{
							return i > 0 ? i : -1;
						}
						if (printable && (c < 0x20 || c > 0x7E))
						{
							return -1;
						}
					}
					return -1;
				}
				__except (filter(GetExceptionCode()))
				{
					return -1;
				}
			}
		}

		bool readable(const void* ptr, const std::size_t size)
		{
			if (!ptr)
			{
				return false;
			}
			const auto* first = static_cast<const volatile std::uint8_t*>(ptr);
			return touch(first, first + (size ? size - 1 : 0));
		}

		const char* name(const char* text)
		{
			return text && length(text, true) > 0 ? text : nullptr;
		}

		const char* terminated(const char* text)
		{
			return text && length(text, false) > 0 ? text : nullptr;
		}
	}
}
