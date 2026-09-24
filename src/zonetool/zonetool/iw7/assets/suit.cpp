#include "std_include.hpp"
#include "suit.hpp"

namespace zonetool::iw7
{
#define PARSE_STRING(__field__) \
	static_assert(std::is_same_v<decltype(asset->__field__), const char*>, "Field is not of type const char*"); \
	!data[#__field__].is_null() && !data[#__field__].get<std::string>().empty() ? asset->__field__ = mem->duplicate_string(data[#__field__].get<std::string>()) : asset->__field__ = nullptr;

#define PARSE_FIELD(__field__) \
	if (!data[#__field__].is_null()) asset->__field__ = data[#__field__].get<decltype(asset->__field__)>();

#define PARSE_FIELD_ARR(__field__, __size__) \
	if(!data[#__field__].is_null()) \
	{ \
		for (auto idx = 0u; idx < (unsigned int)__size__; idx++) \
		{ \
			asset->__field__[idx] = data[#__field__][idx].get<typename std::remove_reference<decltype(asset->__field__[idx])>::type>(); \
		} \
	}

#define PARSE_ASSET(__field__) \
	if (!data[#__field__].is_null() && !data[#__field__].get<std::string>().empty()) \
	{ \
		asset->__field__ = mem->manual_allocate<typename std::remove_reference<decltype(*asset->__field__)>::type>(sizeof(const char*)); \
		asset->__field__->name = mem->duplicate_string(data[#__field__].get<std::string>()); \
	} \
	else \
	{ \
		asset->__field__ = nullptr; \
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

		auto asset = mem->allocate<SuitDef>();
		asset->name = mem->duplicate_string(name);

		PARSE_FIELD(cam_minVelocityForFovIncrease);
		PARSE_FIELD(cam_maxVelocityForFovIncrease);
		PARSE_FIELD(cam_velocityToDecreaseFov);
		PARSE_FIELD(cam_fovIncreaseAtMaxVelocity);
		PARSE_FIELD(cam_oneOverFovEaseInTimeMs);
		PARSE_FIELD(cam_oneOverFovEaseOutTimeMs);
		PARSE_FIELD(enableIKOverride);
		PARSE_FIELD(player_globalAccelScale);
		PARSE_FIELD(player_sprintSpeedScale);
		PARSE_FIELD(player_sprintUnlimited);
		PARSE_FIELD(player_viewBobScale);
		PARSE_FIELD(jump_slowdownEnable);
		PARSE_FIELD(jump_height);
		PARSE_FIELD(sprintLeap_height);
		PARSE_FIELD(sprintLeap_forwardVelocityScale);
		PARSE_FIELD(sprintLeap_minSprintTimeMs);
		PARSE_FIELD(doubleJump_accel);
		PARSE_FIELD(doubleJump_speed);
		PARSE_FIELD(doubleJump_speedNoBoost);
		PARSE_FIELD(doubleJump_frictionMin);
		PARSE_FIELD(doubleJump_frictionMax);
		PARSE_FIELD(doubleJump_initialUpBoostAccel);
		PARSE_FIELD(doubleJump_standardUpBoostAccel);
		PARSE_FIELD(doubleJump_energyNeededForInitialUpBoost);
		PARSE_FIELD(doubleJump_energyNeededForStandardUpBoost);
		PARSE_FIELD(doubleJump_maxUpwardsVelocity);
		PARSE_FIELD(doubleJump_enableMinZVelocity);
		PARSE_FIELD(doubleJump_minZVelocity);
		PARSE_FIELD(doubleJump_energyInitialCost);
		PARSE_FIELD(doubleJump_boostUpEnergyUseRate);
		PARSE_FIELD(doubleJump_energyUsePerButtonPress);
		PARSE_FIELD(doubleJump_hoverOnly);
		PARSE_FIELD(doubleJump_maxViewPitchDip);
		PARSE_FIELD(doubleJump_maxViewBackwardsPitchDip);
		PARSE_FIELD(doubleJump_maxViewRoll);
		PARSE_FIELD(doubleJump_oneOverEaseInTime);
		PARSE_FIELD(doubleJump_oneOverEaseOutTimeMs);
		PARSE_FIELD(doubleJump_alwaysDipView);
		PARSE_STRING(doubleJump_sound);
		PARSE_STRING(doubleJump_soundPlayer);
		PARSE_STRING(doubleJump_releaseSound);
		PARSE_STRING(doubleJump_releaseSoundPlayer);
		PARSE_FIELD(slide_sprint_penalty_ms);
		PARSE_FIELD(slide_allow_firing);
		PARSE_FIELD(slide_allow_ads);
		PARSE_FIELD(slide_allow_weapon_switch);
		PARSE_FIELD(slide_jump_speed_scale);
		PARSE_FIELD(slide_energy_cost_ratio);
		PARSE_FIELD(slide_energy_unknown01);
		PARSE_FIELD(slide_energy_unknown02);
		PARSE_FIELD(slide_energy_unknown03);
		PARSE_FIELD(slide_max_time_ms);
		PARSE_FIELD(slide_max_time_reduced_ms);
		PARSE_FIELD(slide_max_time_base_ms);
		PARSE_FIELD(wallRun_minZVel);
		PARSE_FIELD(wallRun_minTriggerSpeed);
		PARSE_FIELD(wallRun_minMaintainSpeed);
		PARSE_FIELD(wallRun_delayPeriodMs);
		PARSE_FIELD(wallRun_minJumpHeight);
		PARSE_FIELD(wallRun_maxTimeMs);
		PARSE_FIELD(wallRun_fallStageTimeMs);
		PARSE_FIELD(wallRun_maxHeight);
		PARSE_FIELD(wallRun_jumpHeight);
		PARSE_FIELD(wallRun_jumpVelocity);
		PARSE_FIELD(wallRun_frictionScale);
		PARSE_FIELD(wallRun_speedScale);
		PARSE_FIELD(wallRun_speedScaleADS);
		PARSE_FIELD(wallRun_energyInitialCost);
		PARSE_FIELD(wallRun_energyChangePerSecond);
		PARSE_FIELD(wallRun_energyUnknown01);
		PARSE_FIELD(wallRun_energyUnknown02);
		PARSE_FIELD(suitAnimType);
		PARSE_ASSET(animPackage);
		PARSE_ASSET(animPackageL);
		PARSE_ASSET(animPackageR);
		PARSE_ASSET(animPackageRelaxed);
		PARSE_ASSET(animPackageSafe);
		PARSE_ASSET(animPackageUnk);
		PARSE_ASSET(scriptableDef);
		PARSE_FIELD(viewheight_stand);
		PARSE_FIELD(viewheight_crouch);
		PARSE_FIELD(viewheight_prone);
		PARSE_FIELD(viewheight_laststand);
		PARSE_FIELD(viewheight_dead);
		PARSE_FIELD(viewheight_swim);
		PARSE_FIELD(viewheight_slide);
		PARSE_FIELD(bounds_radius);
		PARSE_FIELD(bounds_height_stand);
		PARSE_FIELD(bounds_height_crouch);
		PARSE_FIELD(bounds_height_prone);
		PARSE_FIELD_ARR(bounds_stand.midPoint, 3);
		PARSE_FIELD_ARR(bounds_stand.halfSize, 3);
		PARSE_FIELD_ARR(bounds_crouch.midPoint, 3);
		PARSE_FIELD_ARR(bounds_crouch.halfSize, 3);
		PARSE_FIELD_ARR(bounds_prone.midPoint, 3);
		PARSE_FIELD_ARR(bounds_prone.halfSize, 3);
		PARSE_FIELD(radialMotionBlur_interpTimeIn);
		PARSE_FIELD(radialMotionBlur_interpTimeOut);
		PARSE_FIELD(radialMotionBlur_sprintMinRadius);
		PARSE_FIELD(radialMotionBlur_sprintMaxRadius);
		PARSE_FIELD(radialMotionBlur_sprintMinStrength);
		PARSE_FIELD(radialMotionBlur_sprintMaxStrength);
		PARSE_FIELD(radialMotionBlur_slideMinRadius);
		PARSE_FIELD(radialMotionBlur_slideMaxRadius);
		PARSE_FIELD(radialMotionBlur_slideMinStrength);
		PARSE_FIELD(radialMotionBlur_slideMaxStrength);
		PARSE_FIELD(radialMotionBlur_doubleJumpMinRadius);
		PARSE_FIELD(radialMotionBlur_doubleJumpMaxRadius);
		PARSE_FIELD(radialMotionBlur_doubleJumpMinStrength);
		PARSE_FIELD(radialMotionBlur_doubleJumpMaxStrength);
		PARSE_FIELD(radialMotionBlur_wallRunMinRadius);
		PARSE_FIELD(radialMotionBlur_wallRunMaxRadius);
		PARSE_FIELD(radialMotionBlur_wallRunMinStrength);
		PARSE_FIELD(radialMotionBlur_wallRunMaxStrength);
		PARSE_FIELD(radialMotionBlur_groundPoundMinRadius);
		PARSE_FIELD(radialMotionBlur_groundPoundMaxRadius);
		PARSE_FIELD(radialMotionBlur_groundPoundMinStrength);
		PARSE_FIELD(radialMotionBlur_groundPoundMaxStrength);
		PARSE_FIELD(radialMotionBlur_rewindMinRadius);
		PARSE_FIELD(radialMotionBlur_rewindMaxRadius);
		PARSE_FIELD(radialMotionBlur_rewindMinStrength);
		PARSE_FIELD(radialMotionBlur_rewindMaxStrength);
		PARSE_FIELD(radialMotionBlur_dodgeMinRadius);
		PARSE_FIELD(radialMotionBlur_dodgeMaxRadius);
		PARSE_FIELD(radialMotionBlur_dodgeMinStrength);
		PARSE_FIELD(radialMotionBlur_dodgeMaxStrength);
		PARSE_ASSET(groundPound_activationRumble);
		PARSE_STRING(groundPound_activationSound);
		PARSE_STRING(groundPound_activationSoundPlayer);
		PARSE_ASSET(groundPound_landingRumble);
		PARSE_STRING(groundPound_landingSound);
		PARSE_STRING(groundPound_landingSoundPlayer);
		PARSE_ASSET(landing_rumbleLowHeight);
		PARSE_ASSET(landing_rumbleMediumHeight);
		PARSE_ASSET(landing_rumbleHighHeight);
		PARSE_ASSET(landing_rumbleExtremeHeight);
		PARSE_FIELD(landing_speedScale);
		PARSE_FIELD(footstep_shakeBroadcastRadiusInches);
		PARSE_FIELD(footstep_shakeDurationMs);
		PARSE_FIELD(footstep_shakeAmplitude);
		PARSE_ASSET(footstep_rumble);

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

#define SUBASSET_DEPENDING(__field__,__type__) \
	if (asset->__field__) \
	{ \
		zone->add_asset_of_type(__type__, asset->__field__->name); \
	}

#define SUBASSET_REFERENCE(__field__,__type__) \
	if (asset->__field__) \
	{ \
		zone->add_asset_of_type(__type__, ","s + asset->__field__->name); \
	}

	void suit::load_depending(zone_base* zone)
	{
		auto* asset = this->asset_;

		SUBASSET_REFERENCE(animPackage, ASSET_TYPE_SUITANIMPACKAGE);
		SUBASSET_REFERENCE(animPackageL, ASSET_TYPE_SUITANIMPACKAGE);
		SUBASSET_REFERENCE(animPackageR, ASSET_TYPE_SUITANIMPACKAGE);
		SUBASSET_REFERENCE(animPackageRelaxed, ASSET_TYPE_SUITANIMPACKAGE);
		SUBASSET_REFERENCE(animPackageSafe, ASSET_TYPE_SUITANIMPACKAGE);
		SUBASSET_REFERENCE(animPackageUnk, ASSET_TYPE_SUITANIMPACKAGE);

		SUBASSET_DEPENDING(scriptableDef, ASSET_TYPE_SCRIPTABLE);

		SUBASSET_DEPENDING(groundPound_activationRumble, ASSET_TYPE_RUMBLE);
		SUBASSET_DEPENDING(groundPound_landingRumble, ASSET_TYPE_RUMBLE);
		SUBASSET_DEPENDING(landing_rumbleLowHeight, ASSET_TYPE_RUMBLE);
		SUBASSET_DEPENDING(landing_rumbleMediumHeight, ASSET_TYPE_RUMBLE);
		SUBASSET_DEPENDING(landing_rumbleHighHeight, ASSET_TYPE_RUMBLE);
		SUBASSET_DEPENDING(landing_rumbleExtremeHeight, ASSET_TYPE_RUMBLE);
		SUBASSET_DEPENDING(footstep_rumble, ASSET_TYPE_RUMBLE);
	}

	std::string suit::name()
	{
		return this->name_;
	}

	std::int32_t suit::type()
	{
		return ASSET_TYPE_SUIT;
	}

#define WRITE_STRING(__field__) \
	if (data->__field__) \
	{ \
		dest->__field__ = buf->write_str(data->__field__); \
	}

#define WRITE_ASSET(__field__,__type__) \
	if (data->__field__) \
	{ \
		dest->__field__ = reinterpret_cast<typename std::remove_reference<decltype(dest->__field__)>::type>(zone->get_asset_pointer(__type__, data->__field__->name)); \
	}

	void suit::write(zone_base* zone, zone_buffer* buf)
	{
		auto data = this->asset_;
		auto dest = buf->write(data);

		buf->push_stream(XFILE_BLOCK_VIRTUAL);

		dest->name = buf->write_str(this->name());

		WRITE_STRING(doubleJump_sound);
		WRITE_STRING(doubleJump_soundPlayer);
		WRITE_STRING(doubleJump_releaseSound);
		WRITE_STRING(doubleJump_releaseSoundPlayer);
		WRITE_STRING(groundPound_activationSound);
		WRITE_STRING(groundPound_activationSoundPlayer);
		WRITE_STRING(groundPound_landingSound);
		WRITE_STRING(groundPound_landingSoundPlayer);

		WRITE_ASSET(animPackage, ASSET_TYPE_SUITANIMPACKAGE);
		WRITE_ASSET(animPackageL, ASSET_TYPE_SUITANIMPACKAGE);
		WRITE_ASSET(animPackageR, ASSET_TYPE_SUITANIMPACKAGE);
		WRITE_ASSET(animPackageRelaxed, ASSET_TYPE_SUITANIMPACKAGE);
		WRITE_ASSET(animPackageSafe, ASSET_TYPE_SUITANIMPACKAGE);
		WRITE_ASSET(animPackageUnk, ASSET_TYPE_SUITANIMPACKAGE);

		WRITE_ASSET(scriptableDef, ASSET_TYPE_SCRIPTABLE);

		WRITE_ASSET(groundPound_activationRumble, ASSET_TYPE_RUMBLE);
		WRITE_ASSET(groundPound_landingRumble, ASSET_TYPE_RUMBLE);
		WRITE_ASSET(landing_rumbleLowHeight, ASSET_TYPE_RUMBLE);
		WRITE_ASSET(landing_rumbleMediumHeight, ASSET_TYPE_RUMBLE);
		WRITE_ASSET(landing_rumbleHighHeight, ASSET_TYPE_RUMBLE);
		WRITE_ASSET(landing_rumbleExtremeHeight, ASSET_TYPE_RUMBLE);
		WRITE_ASSET(footstep_rumble, ASSET_TYPE_RUMBLE);

		buf->pop_stream();
	}

#define DUMP_STRING(__field__) \
	static_assert(std::is_same_v<decltype(asset->__field__), const char*>, "Field is not of type const char*"); \
	asset->__field__ ? data[#__field__] = asset->__field__ : data[#__field__] = "";

#define DUMP_FIELD(__field__) \
	data[#__field__] = asset->__field__;

#define DUMP_FIELD_ARR(__field__, __size__) \
	for (auto idx = 0u; idx < (unsigned int)__size__; idx++) \
	{ \
		data[#__field__][idx] = asset->__field__[idx]; \
	}

#define DUMP_ASSET(__field__) \
	if (asset->__field__) \
	{ \
		data[#__field__] = asset->__field__->name; \
	} \
	else \
	{ \
		data[#__field__] = ""; \
	}

	void suit::dump(SuitDef* asset)
	{
		const auto path = "suit\\"s + asset->name + ".json"s;
		auto file = filesystem::file(path);
		file.open("wb");

		ordered_json data;

		DUMP_FIELD(cam_minVelocityForFovIncrease);
		DUMP_FIELD(cam_maxVelocityForFovIncrease);
		DUMP_FIELD(cam_velocityToDecreaseFov);
		DUMP_FIELD(cam_fovIncreaseAtMaxVelocity);
		DUMP_FIELD(cam_oneOverFovEaseInTimeMs);
		DUMP_FIELD(cam_oneOverFovEaseOutTimeMs);
		DUMP_FIELD(enableIKOverride);
		DUMP_FIELD(player_globalAccelScale);
		DUMP_FIELD(player_sprintSpeedScale);
		DUMP_FIELD(player_sprintUnlimited);
		DUMP_FIELD(player_viewBobScale);
		DUMP_FIELD(jump_slowdownEnable);
		DUMP_FIELD(jump_height);
		DUMP_FIELD(sprintLeap_height);
		DUMP_FIELD(sprintLeap_forwardVelocityScale);
		DUMP_FIELD(sprintLeap_minSprintTimeMs);
		DUMP_FIELD(doubleJump_accel);
		DUMP_FIELD(doubleJump_speed);
		DUMP_FIELD(doubleJump_speedNoBoost);
		DUMP_FIELD(doubleJump_frictionMin);
		DUMP_FIELD(doubleJump_frictionMax);
		DUMP_FIELD(doubleJump_initialUpBoostAccel);
		DUMP_FIELD(doubleJump_standardUpBoostAccel);
		DUMP_FIELD(doubleJump_energyNeededForInitialUpBoost);
		DUMP_FIELD(doubleJump_energyNeededForStandardUpBoost);
		DUMP_FIELD(doubleJump_maxUpwardsVelocity);
		DUMP_FIELD(doubleJump_enableMinZVelocity);
		DUMP_FIELD(doubleJump_minZVelocity);
		DUMP_FIELD(doubleJump_energyInitialCost);
		DUMP_FIELD(doubleJump_boostUpEnergyUseRate);
		DUMP_FIELD(doubleJump_energyUsePerButtonPress);
		DUMP_FIELD(doubleJump_hoverOnly);
		DUMP_FIELD(doubleJump_maxViewPitchDip);
		DUMP_FIELD(doubleJump_maxViewBackwardsPitchDip);
		DUMP_FIELD(doubleJump_maxViewRoll);
		DUMP_FIELD(doubleJump_oneOverEaseInTime);
		DUMP_FIELD(doubleJump_oneOverEaseOutTimeMs);
		DUMP_FIELD(doubleJump_alwaysDipView);
		DUMP_STRING(doubleJump_sound);
		DUMP_STRING(doubleJump_soundPlayer);
		DUMP_STRING(doubleJump_releaseSound);
		DUMP_STRING(doubleJump_releaseSoundPlayer);
		DUMP_FIELD(slide_sprint_penalty_ms);
		DUMP_FIELD(slide_allow_firing);
		DUMP_FIELD(slide_allow_ads);
		DUMP_FIELD(slide_allow_weapon_switch);
		DUMP_FIELD(slide_jump_speed_scale);
		DUMP_FIELD(slide_energy_cost_ratio);
		DUMP_FIELD(slide_energy_unknown01);
		DUMP_FIELD(slide_energy_unknown02);
		DUMP_FIELD(slide_energy_unknown03);
		DUMP_FIELD(slide_max_time_ms);
		DUMP_FIELD(slide_max_time_reduced_ms);
		DUMP_FIELD(slide_max_time_base_ms);
		DUMP_FIELD(wallRun_minZVel);
		DUMP_FIELD(wallRun_minTriggerSpeed);
		DUMP_FIELD(wallRun_minMaintainSpeed);
		DUMP_FIELD(wallRun_delayPeriodMs);
		DUMP_FIELD(wallRun_minJumpHeight);
		DUMP_FIELD(wallRun_maxTimeMs);
		DUMP_FIELD(wallRun_fallStageTimeMs);
		DUMP_FIELD(wallRun_maxHeight);
		DUMP_FIELD(wallRun_jumpHeight);
		DUMP_FIELD(wallRun_jumpVelocity);
		DUMP_FIELD(wallRun_frictionScale);
		DUMP_FIELD(wallRun_speedScale);
		DUMP_FIELD(wallRun_speedScaleADS);
		DUMP_FIELD(wallRun_energyInitialCost);
		DUMP_FIELD(wallRun_energyChangePerSecond);
		DUMP_FIELD(wallRun_energyUnknown01);
		DUMP_FIELD(wallRun_energyUnknown02);
		DUMP_FIELD(suitAnimType);
		DUMP_ASSET(animPackage);
		DUMP_ASSET(animPackageL);
		DUMP_ASSET(animPackageR);
		DUMP_ASSET(animPackageRelaxed);
		DUMP_ASSET(animPackageSafe);
		DUMP_ASSET(animPackageUnk);
		DUMP_ASSET(scriptableDef);
		DUMP_FIELD(viewheight_stand);
		DUMP_FIELD(viewheight_crouch);
		DUMP_FIELD(viewheight_prone);
		DUMP_FIELD(viewheight_laststand);
		DUMP_FIELD(viewheight_dead);
		DUMP_FIELD(viewheight_swim);
		DUMP_FIELD(viewheight_slide);
		DUMP_FIELD(bounds_radius);
		DUMP_FIELD(bounds_height_stand);
		DUMP_FIELD(bounds_height_crouch);
		DUMP_FIELD(bounds_height_prone);
		DUMP_FIELD_ARR(bounds_stand.midPoint, 3);
		DUMP_FIELD_ARR(bounds_stand.halfSize, 3);
		DUMP_FIELD_ARR(bounds_crouch.midPoint, 3);
		DUMP_FIELD_ARR(bounds_crouch.halfSize, 3);
		DUMP_FIELD_ARR(bounds_prone.midPoint, 3);
		DUMP_FIELD_ARR(bounds_prone.halfSize, 3);
		DUMP_FIELD(radialMotionBlur_interpTimeIn);
		DUMP_FIELD(radialMotionBlur_interpTimeOut);
		DUMP_FIELD(radialMotionBlur_sprintMinRadius);
		DUMP_FIELD(radialMotionBlur_sprintMaxRadius);
		DUMP_FIELD(radialMotionBlur_sprintMinStrength);
		DUMP_FIELD(radialMotionBlur_sprintMaxStrength);
		DUMP_FIELD(radialMotionBlur_slideMinRadius);
		DUMP_FIELD(radialMotionBlur_slideMaxRadius);
		DUMP_FIELD(radialMotionBlur_slideMinStrength);
		DUMP_FIELD(radialMotionBlur_slideMaxStrength);
		DUMP_FIELD(radialMotionBlur_doubleJumpMinRadius);
		DUMP_FIELD(radialMotionBlur_doubleJumpMaxRadius);
		DUMP_FIELD(radialMotionBlur_doubleJumpMinStrength);
		DUMP_FIELD(radialMotionBlur_doubleJumpMaxStrength);
		DUMP_FIELD(radialMotionBlur_wallRunMinRadius);
		DUMP_FIELD(radialMotionBlur_wallRunMaxRadius);
		DUMP_FIELD(radialMotionBlur_wallRunMinStrength);
		DUMP_FIELD(radialMotionBlur_wallRunMaxStrength);
		DUMP_FIELD(radialMotionBlur_groundPoundMinRadius);
		DUMP_FIELD(radialMotionBlur_groundPoundMaxRadius);
		DUMP_FIELD(radialMotionBlur_groundPoundMinStrength);
		DUMP_FIELD(radialMotionBlur_groundPoundMaxStrength);
		DUMP_FIELD(radialMotionBlur_rewindMinRadius);
		DUMP_FIELD(radialMotionBlur_rewindMaxRadius);
		DUMP_FIELD(radialMotionBlur_rewindMinStrength);
		DUMP_FIELD(radialMotionBlur_rewindMaxStrength);
		DUMP_FIELD(radialMotionBlur_dodgeMinRadius);
		DUMP_FIELD(radialMotionBlur_dodgeMaxRadius);
		DUMP_FIELD(radialMotionBlur_dodgeMinStrength);
		DUMP_FIELD(radialMotionBlur_dodgeMaxStrength);
		DUMP_ASSET(groundPound_activationRumble);
		DUMP_STRING(groundPound_activationSound);
		DUMP_STRING(groundPound_activationSoundPlayer);
		DUMP_ASSET(groundPound_landingRumble);
		DUMP_STRING(groundPound_landingSound);
		DUMP_STRING(groundPound_landingSoundPlayer);
		DUMP_ASSET(landing_rumbleLowHeight);
		DUMP_ASSET(landing_rumbleMediumHeight);
		DUMP_ASSET(landing_rumbleHighHeight);
		DUMP_ASSET(landing_rumbleExtremeHeight);
		DUMP_FIELD(landing_speedScale);
		DUMP_FIELD(footstep_shakeBroadcastRadiusInches);
		DUMP_FIELD(footstep_shakeDurationMs);
		DUMP_FIELD(footstep_shakeAmplitude);
		DUMP_ASSET(footstep_rumble);

		auto str = data.dump(4);
		data.clear();
		file.write(str);
		file.close();
	}
}
