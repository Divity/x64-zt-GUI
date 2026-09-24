#include "std_include.hpp"
#include "suit.hpp"

namespace zonetool::iw7
{
#define SUIT_STRINGS(X) \
	X(doubleJump_sound) \
	X(doubleJump_soundPlayer) \
	X(doubleJump_releaseSound) \
	X(doubleJump_releaseSoundPlayer) \
	X(groundPound_activationSound) \
	X(groundPound_activationSoundPlayer) \
	X(groundPound_landingSound) \
	X(groundPound_landingSoundPlayer)

#define SUIT_ANIM_PACKAGES(X) \
	X(animPackage) \
	X(animPackageL) \
	X(animPackageR) \
	X(animPackageRelaxed) \
	X(animPackageSafe) \
	X(animPackageUnk)

#define SUIT_RUMBLES(X) \
	X(groundPound_activationRumble) \
	X(groundPound_landingRumble) \
	X(landing_rumbleLowHeight) \
	X(landing_rumbleMediumHeight) \
	X(landing_rumbleHighHeight) \
	X(landing_rumbleExtremeHeight) \
	X(footstep_rumble)

	namespace
	{
		template <typename T>
		T* parse_asset_ref(ordered_json& value, zone_memory* mem)
		{
			if (value.is_null() || value.get<std::string>().empty())
			{
				return nullptr;
			}

			auto asset = mem->manual_allocate<T>(sizeof(const char*));
			asset->name = mem->duplicate_string(value.get<std::string>());
			return asset;
		}

		void clear_pointers(SuitDef* asset)
		{
			asset->name = nullptr;
#define CLEAR(__field__) asset->__field__ = nullptr;
			SUIT_STRINGS(CLEAR)
			SUIT_ANIM_PACKAGES(CLEAR)
			SUIT_RUMBLES(CLEAR)
			CLEAR(scriptableDef)
#undef CLEAR
		}
	}

	SuitDef* suit::parse(const std::string& name, zone_memory* mem)
	{
		const auto path = "suit\\"s + name + ".json"s;

		auto file = filesystem::file(path);
		if (!file.exists())
		{
			return nullptr;
		}

		ZONETOOL_INFO("Parsing suit \"%s\"...", name.data());

		file.open("rb");
		ordered_json data = json::parse(file.read_bytes(file.size()));
		file.close();

		const auto raw = data["raw"].get<std::vector<std::uint8_t>>();
		if (raw.size() != sizeof(SuitDef))
		{
			ZONETOOL_FATAL("suit \"%s\": raw is %zu bytes, expected %zu", name.data(), raw.size(), sizeof(SuitDef));
		}

		auto asset = mem->allocate<SuitDef>();
		std::memcpy(asset, raw.data(), sizeof(SuitDef));
		clear_pointers(asset);
		asset->name = mem->duplicate_string(name);

#define PARSE_STRING(__field__) \
		if (!data[#__field__].is_null() && !data[#__field__].get<std::string>().empty()) \
			asset->__field__ = mem->duplicate_string(data[#__field__].get<std::string>());
		SUIT_STRINGS(PARSE_STRING)
#undef PARSE_STRING

#define PARSE_PACKAGE(__field__) asset->__field__ = parse_asset_ref<SuitAnimPackage>(data[#__field__], mem);
		SUIT_ANIM_PACKAGES(PARSE_PACKAGE)
#undef PARSE_PACKAGE

#define PARSE_RUMBLE(__field__) asset->__field__ = parse_asset_ref<RumbleInfo>(data[#__field__], mem);
		SUIT_RUMBLES(PARSE_RUMBLE)
#undef PARSE_RUMBLE

		asset->scriptableDef = parse_asset_ref<ScriptableDef>(data["scriptableDef"], mem);

		return asset;
	}

	void suit::init(const std::string& name, zone_memory* mem)
	{
		this->name_ = name;

		if (this->referenced())
		{
			this->asset_ = mem->allocate<typename std::remove_reference<decltype(*this->asset_)>::type>();
			this->asset_->name = mem->duplicate_string(name);
			return;
		}

		this->asset_ = this->parse(name, mem);
		if (!this->asset_)
		{
			this->asset_ = db_find_x_asset_header_safe(XAssetType(this->type()), this->name().data()).suitDef;
		}
	}

	void suit::prepare(zone_buffer* buf, zone_memory* mem)
	{
	}

	void suit::load_depending(zone_base* zone)
	{
		auto* asset = this->asset_;

#define ADD_PACKAGE(__field__) \
		if (asset->__field__) zone->add_asset_of_type(ASSET_TYPE_SUITANIMPACKAGE, ","s + asset->__field__->name);
		SUIT_ANIM_PACKAGES(ADD_PACKAGE)
#undef ADD_PACKAGE

#define ADD_RUMBLE(__field__) \
		if (asset->__field__) zone->add_asset_of_type(ASSET_TYPE_RUMBLE, asset->__field__->name);
		SUIT_RUMBLES(ADD_RUMBLE)
#undef ADD_RUMBLE

		if (asset->scriptableDef)
		{
			zone->add_asset_of_type(ASSET_TYPE_SCRIPTABLE, asset->scriptableDef->name);
		}
	}

	std::string suit::name()
	{
		return this->name_;
	}

	std::int32_t suit::type()
	{
		return ASSET_TYPE_SUIT;
	}

	void suit::write(zone_base* zone, zone_buffer* buf)
	{
		auto data = this->asset_;
		auto dest = buf->write(data);

		buf->push_stream(XFILE_BLOCK_VIRTUAL);

		dest->name = buf->write_str(this->name());

#define WRITE_STRING(__field__) \
		if (data->__field__) dest->__field__ = buf->write_str(data->__field__);
		WRITE_STRING(doubleJump_sound)
		WRITE_STRING(doubleJump_soundPlayer)
		WRITE_STRING(doubleJump_releaseSound)
		WRITE_STRING(doubleJump_releaseSoundPlayer)
		WRITE_STRING(groundPound_activationSound)
		WRITE_STRING(groundPound_activationSoundPlayer)
		WRITE_STRING(groundPound_landingSound)
		WRITE_STRING(groundPound_landingSoundPlayer)
#undef WRITE_STRING

#define WRITE_PACKAGE(__field__) \
		if (data->__field__) dest->__field__ = reinterpret_cast<SuitAnimPackage*>( \
			zone->get_asset_pointer(ASSET_TYPE_SUITANIMPACKAGE, data->__field__->name));
		SUIT_ANIM_PACKAGES(WRITE_PACKAGE)
#undef WRITE_PACKAGE

#define WRITE_RUMBLE(__field__) \
		if (data->__field__) dest->__field__ = reinterpret_cast<RumbleInfo*>( \
			zone->get_asset_pointer(ASSET_TYPE_RUMBLE, data->__field__->name));
		SUIT_RUMBLES(WRITE_RUMBLE)
#undef WRITE_RUMBLE

		if (data->scriptableDef)
		{
			dest->scriptableDef = reinterpret_cast<ScriptableDef*>(
				zone->get_asset_pointer(ASSET_TYPE_SCRIPTABLE, data->scriptableDef->name));
		}

		buf->pop_stream();
	}

	void suit::dump(SuitDef* asset)
	{
		const auto path = "suit\\"s + asset->name + ".json"s;
		auto file = filesystem::file(path);
		file.open("wb");

		ordered_json data;

		SuitDef scalars = *asset;
		clear_pointers(&scalars);
		const auto* bytes = reinterpret_cast<const std::uint8_t*>(&scalars);
		data["raw"] = std::vector<std::uint8_t>(bytes, bytes + sizeof(SuitDef));

#define DUMP_STRING(__field__) data[#__field__] = asset->__field__ ? asset->__field__ : "";
		SUIT_STRINGS(DUMP_STRING)
#undef DUMP_STRING

#define DUMP_ASSET(__field__) data[#__field__] = asset->__field__ ? asset->__field__->name : "";
		SUIT_ANIM_PACKAGES(DUMP_ASSET)
		SUIT_RUMBLES(DUMP_ASSET)
		DUMP_ASSET(scriptableDef)
#undef DUMP_ASSET

		auto str = data.dump(4);
		data.clear();
		file.write(str);
		file.close();
	}
}
