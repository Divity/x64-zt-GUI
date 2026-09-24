#include "std_include.hpp"
#include "suit_anim_package.hpp"

namespace zonetool::iw7
{
	void suit_anim_package::init(const std::string& name, zone_memory* mem)
	{
		this->name_ = name;

		if (!this->referenced())
		{
			ZONETOOL_FATAL("suitanimpackage \"%s\" can only be referenced (\",%s\"): stock packages ship in code_post_gfx",
				name.data(), name.data());
		}

		this->asset_ = mem->allocate<SuitAnimPackage>();
		this->asset_->name = mem->duplicate_string(name);
	}

	void suit_anim_package::prepare(zone_buffer* buf, zone_memory* mem)
	{
	}

	void suit_anim_package::load_depending(zone_base* zone)
	{
	}

	std::string suit_anim_package::name()
	{
		return this->name_;
	}

	std::int32_t suit_anim_package::type()
	{
		return ASSET_TYPE_SUITANIMPACKAGE;
	}

	void suit_anim_package::write(zone_base* zone, zone_buffer* buf)
	{
		auto data = this->asset_;
		auto dest = buf->write(data);

		buf->push_stream(XFILE_BLOCK_VIRTUAL);
		dest->name = buf->write_str(this->name());
		buf->pop_stream();
	}
}
