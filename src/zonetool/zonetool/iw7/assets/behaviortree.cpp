#include "std_include.hpp"
#include "behaviortree.hpp"

namespace zonetool::iw7
{
	BehaviorTree* behavior_tree::parse(const std::string& name, zone_memory* mem)
	{
		assetmanager::reader read(mem);

		const auto path = "behaviortree\\"s + name;
		if (!read.open(path))
		{
			return nullptr;
		}

		ZONETOOL_INFO("Parsing behaviortree \"%s\"...", name.data());

		auto* asset = read.read_single<BehaviorTree>();
		asset->name = read.read_string();

		if (asset->nodes)
		{
			asset->nodes = read.read_array<BehaviorTreeNode>();
			for (auto i = 0; i < asset->nodeCount; i++)
			{
				asset->nodes[i].name = read.read_string();
			}
		}

		return asset;
	}

	void behavior_tree::init(const std::string& name, zone_memory* mem)
	{
		this->name_ = name;

		if (this->referenced())
		{
			this->asset_ = mem->allocate<typename std::remove_reference<decltype(*this->asset_)>::type>();
			this->asset_->name = mem->duplicate_string(name);
			return;
		}

		this->asset_ = parse(name, mem);
		if (!this->asset_)
		{
			ZONETOOL_FATAL("behaviortree %s is missing...", name.data());
		}
	}

	void behavior_tree::prepare(zone_buffer*, zone_memory*)
	{
	}

	void behavior_tree::load_depending(zone_base*)
	{
	}

	std::string behavior_tree::name()
	{
		return this->name_;
	}

	std::int32_t behavior_tree::type()
	{
		return ASSET_TYPE_BEHAVIOR_TREE;
	}

	void behavior_tree::write(zone_base*, zone_buffer* buf)
	{
		auto* data = this->asset_;
		auto* dest = buf->write(data);

		buf->push_stream(XFILE_BLOCK_VIRTUAL);

		dest->name = buf->write_str(this->name());

		if (data->nodes)
		{
			buf->align(7);
			auto* dest_nodes = buf->write(data->nodes, data->nodeCount);
			for (auto i = 0; i < data->nodeCount; i++)
			{
				if (data->nodes[i].name)
				{
					dest_nodes[i].name = buf->write_str(data->nodes[i].name);
				}
			}
			buf->clear_pointer(&dest->nodes);
		}

		buf->pop_stream();
	}

	void behavior_tree::dump(BehaviorTree* asset)
	{
		const auto path = "behaviortree\\"s + asset->name;

		assetmanager::dumper dumper;
		if (!dumper.open(path))
		{
			return;
		}

		dumper.dump_single(asset);
		dumper.dump_string(asset->name);

		if (asset->nodes)
		{
			dumper.dump_array(asset->nodes, asset->nodeCount);
			for (auto i = 0; i < asset->nodeCount; i++)
			{
				dumper.dump_string(asset->nodes[i].name);
			}
		}

		dumper.close();
	}
}
