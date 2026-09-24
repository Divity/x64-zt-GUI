#include "std_include.hpp"
#include "animclass.hpp"

namespace zonetool::iw7
{
	void anim_class::add_script_string(scr_string_t* ptr, const char* str)
	{
		for (std::uint32_t i = 0; i < this->script_strings.size(); i++)
		{
			if (this->script_strings[i].first == ptr)
			{
				return;
			}
		}
		this->script_strings.push_back(std::pair<scr_string_t*, const char*>(ptr, str));
	}

	const char* anim_class::get_script_string(scr_string_t* ptr)
	{
		for (std::uint32_t i = 0; i < this->script_strings.size(); i++)
		{
			if (this->script_strings[i].first == ptr)
			{
				return this->script_strings[i].second;
			}
		}
		return nullptr;
	}

	namespace
	{
		std::uint8_t u8(char v)
		{
			return static_cast<std::uint8_t>(v);
		}
	}

	AnimationClass* anim_class::parse(const std::string& name, zone_memory* mem)
	{
		assetmanager::reader read(mem);

		const auto path = "animclass\\"s + name;
		if (!read.open(path))
		{
			return nullptr;
		}

		ZONETOOL_INFO("Parsing animclass \"%s\"...", name.data());

		const auto asset = read.read_single<AnimationClass>();
		asset->name = read.read_string();

		if (asset->stateMachine)
		{
			auto* sm = read.read_single<AnimationStateMachine>();
			asset->stateMachine = sm;
			this->add_script_string(&sm->name, read.read_string());

			if (sm->states)
			{
				sm->states = read.read_array<AnimationState>();
				for (auto i = 0; i < sm->stateCount; i++)
				{
					auto* state = &sm->states[i];
					this->add_script_string(&state->name, read.read_string());
					this->add_script_string(&state->notify, read.read_string());

					if (state->animEntries)
					{
						state->animEntries = read.read_array<AnimationEntry>();
						for (auto e = 0; e < u8(state->entryCount); e++)
						{
							this->add_script_string(&state->animEntries[e].animName, read.read_string());
						}
					}

					state->animIndices = nullptr;

					if (state->aliasList)
					{
						state->aliasList = read.read_array<AnimAlias>();
						for (auto a = 0; a < u8(state->aliasCount); a++)
						{
							this->add_script_string(&state->aliasList[a].aliasName, read.read_string());
							if (state->aliasList[a].aliasInfo)
							{
								state->aliasList[a].aliasInfo = read.read_array<AnimAliasInfo>();
							}
						}
					}
				}
			}

			if (sm->aimSets)
			{
				sm->aimSets = read.read_array<AnimationAimSet>();
				for (auto i = 0; i < sm->aimSetCount; i++)
				{
					auto* aim = &sm->aimSets[i];
					this->add_script_string(&aim->name, read.read_string());
					this->add_script_string(&aim->rootName, read.read_string());

					if (aim->animName)
					{
						aim->animName = read.read_array<scr_string_t>();
						for (auto j = 0; j < aim->animCount; j++)
						{
							this->add_script_string(&aim->animName[j], read.read_string());
						}
					}

					aim->animIndices = nullptr;
					aim->aimNodeIndices = nullptr;
				}
			}
		}

		this->add_script_string(&asset->animTree, read.read_string());
		asset->scriptable = read.read_asset<ScriptableDef>();

		const auto read_strings = [&](scr_string_t*& arr, int count)
		{
			if (arr)
			{
				arr = read.read_array<scr_string_t>();
				for (auto i = 0; i < count; i++)
				{
					this->add_script_string(&arr[i], read.read_string());
				}
			}
		};

		read_strings(asset->soundNotes, asset->soundCount);
		read_strings(asset->soundNames, asset->soundCount);
		read_strings(asset->soundOptions, asset->soundCount);
		read_strings(asset->effectNotes, asset->effectCount);

		if (asset->effectDefs)
		{
			asset->effectDefs = read.read_array<FxCombinedDef>();
			for (auto i = 0; i < asset->effectCount; i++)
			{
				auto& def = asset->effectDefs[i];
				if (def.type == FX_COMBINED_VFX)
				{
					def.u.vfx = read.read_asset<ParticleSystemDef>();
				}
				else
				{
					def.u.fx = read.read_asset<FxEffectDef>();
				}
			}
		}

		read_strings(asset->effectTags, asset->effectCount);

		return asset;
	}

	void anim_class::init(const std::string& name, zone_memory* mem)
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
			ZONETOOL_FATAL("animclass %s is missing...", name.data());
		}
	}

	void anim_class::prepare(zone_buffer* buf, zone_memory* mem)
	{
		auto* asset = this->asset_;

		const auto convert = [&](scr_string_t* ptr)
		{
			*ptr = static_cast<scr_string_t>(buf->write_scriptstring(this->get_script_string(ptr)));
		};

		if (auto* sm = asset->stateMachine)
		{
			convert(&sm->name);

			for (auto i = 0; sm->states && i < sm->stateCount; i++)
			{
				auto* state = &sm->states[i];
				convert(&state->name);
				convert(&state->notify);

				for (auto e = 0; state->animEntries && e < u8(state->entryCount); e++)
				{
					convert(&state->animEntries[e].animName);
				}

				for (auto a = 0; state->aliasList && a < u8(state->aliasCount); a++)
				{
					convert(&state->aliasList[a].aliasName);
				}
			}

			for (auto i = 0; sm->aimSets && i < sm->aimSetCount; i++)
			{
				auto* aim = &sm->aimSets[i];
				convert(&aim->name);
				convert(&aim->rootName);

				for (auto j = 0; aim->animName && j < aim->animCount; j++)
				{
					convert(&aim->animName[j]);
				}
			}
		}

		convert(&asset->animTree);

		const auto convert_array = [&](scr_string_t* arr, int count)
		{
			for (auto i = 0; arr && i < count; i++)
			{
				convert(&arr[i]);
			}
		};

		convert_array(asset->soundNotes, asset->soundCount);
		convert_array(asset->soundNames, asset->soundCount);
		convert_array(asset->soundOptions, asset->soundCount);
		convert_array(asset->effectNotes, asset->effectCount);
		convert_array(asset->effectTags, asset->effectCount);
	}

	void anim_class::load_depending(zone_base* zone)
	{
		auto* asset = this->asset_;

		if (asset->scriptable)
		{
			zone->add_asset_of_type(ASSET_TYPE_SCRIPTABLE, asset->scriptable->name);
		}

		for (auto i = 0; asset->effectDefs && i < asset->effectCount; i++)
		{
			const auto& def = asset->effectDefs[i];
			if (!def.u.data)
			{
				continue;
			}

			if (def.type == FX_COMBINED_VFX)
			{
				zone->add_asset_of_type(ASSET_TYPE_VFX, def.u.vfx->name);
			}
			else
			{
				zone->add_asset_of_type(ASSET_TYPE_FX, def.u.fx->name);
			}
		}
	}

	std::string anim_class::name()
	{
		return this->name_;
	}

	std::int32_t anim_class::type()
	{
		return ASSET_TYPE_ANIMCLASS;
	}

	void anim_class::write(zone_base* zone, zone_buffer* buf)
	{
		auto* data = this->asset_;
		auto* dest = buf->write(data);

		buf->push_stream(XFILE_BLOCK_VIRTUAL);

		dest->name = buf->write_str(this->name());

		if (data->stateMachine)
		{
			buf->align(7);
			auto* data_sm = data->stateMachine;
			auto* dest_sm = buf->write(data_sm);

			if (data_sm->states)
			{
				buf->align(7);
				auto* dest_states = buf->write(data_sm->states, data_sm->stateCount);
				for (auto i = 0; i < data_sm->stateCount; i++)
				{
					const auto* state = &data_sm->states[i];

					if (state->animEntries)
					{
						buf->align(3);
						buf->write(state->animEntries, u8(state->entryCount));
						buf->clear_pointer(&dest_states[i].animEntries);
					}

					buf->push_stream(XFILE_BLOCK_RUNTIME);
					if (u8(state->entryCount))
					{
						buf->align(3);
						buf->write(static_cast<std::uint64_t*>(nullptr), u8(state->entryCount));
						buf->clear_pointer(&dest_states[i].animIndices);
					}
					else
					{
						dest_states[i].animIndices = nullptr;
					}
					buf->pop_stream();

					if (state->aliasList)
					{
						buf->align(7);
						auto* dest_aliases = buf->write(state->aliasList, u8(state->aliasCount));
						for (auto a = 0; a < u8(state->aliasCount); a++)
						{
							if (state->aliasList[a].aliasInfo)
							{
								buf->align(3);
								buf->write(state->aliasList[a].aliasInfo, u8(state->aliasList[a].animCount));
								buf->clear_pointer(&dest_aliases[a].aliasInfo);
							}
						}
						buf->clear_pointer(&dest_states[i].aliasList);
					}
				}
				buf->clear_pointer(&dest_sm->states);
			}

			if (data_sm->aimSets)
			{
				buf->align(7);
				auto* dest_aims = buf->write(data_sm->aimSets, data_sm->aimSetCount);
				for (auto i = 0; i < data_sm->aimSetCount; i++)
				{
					const auto* aim = &data_sm->aimSets[i];

					if (aim->animName)
					{
						buf->align(3);
						buf->write(aim->animName, aim->animCount);
						buf->clear_pointer(&dest_aims[i].animName);
					}

					for (auto* field : { &dest_aims[i].animIndices, &dest_aims[i].aimNodeIndices })
					{
						buf->push_stream(XFILE_BLOCK_RUNTIME);
						if (aim->animCount)
						{
							buf->align(3);
							buf->write(static_cast<std::uint64_t*>(nullptr), aim->animCount);
							buf->clear_pointer(field);
						}
						else
						{
							*field = nullptr;
						}
						buf->pop_stream();
					}
				}
				buf->clear_pointer(&dest_sm->aimSets);
			}

			buf->clear_pointer(&dest->stateMachine);
		}

		if (data->scriptable)
		{
			dest->scriptable = reinterpret_cast<ScriptableDef*>(zone->get_asset_pointer(
				ASSET_TYPE_SCRIPTABLE, data->scriptable->name));
		}

		const auto write_strings = [&](scr_string_t* src, scr_string_t** dst_field, int count)
		{
			if (src)
			{
				buf->align(3);
				buf->write(src, count);
				buf->clear_pointer(dst_field);
			}
		};

		write_strings(data->soundNotes, &dest->soundNotes, data->soundCount);
		write_strings(data->soundNames, &dest->soundNames, data->soundCount);
		write_strings(data->soundOptions, &dest->soundOptions, data->soundCount);
		write_strings(data->effectNotes, &dest->effectNotes, data->effectCount);

		if (data->effectDefs)
		{
			buf->align(7);
			auto* dest_defs = buf->write(data->effectDefs, data->effectCount);
			for (auto i = 0; i < data->effectCount; i++)
			{
				const auto& def = data->effectDefs[i];
				if (!def.u.data)
				{
					continue;
				}

				if (def.type == FX_COMBINED_VFX)
				{
					dest_defs[i].u.vfx = reinterpret_cast<ParticleSystemDef*>(zone->get_asset_pointer(
						ASSET_TYPE_VFX, def.u.vfx->name));
				}
				else
				{
					dest_defs[i].u.fx = reinterpret_cast<FxEffectDef*>(zone->get_asset_pointer(
						ASSET_TYPE_FX, def.u.fx->name));
				}
			}
			buf->clear_pointer(&dest->effectDefs);
		}

		write_strings(data->effectTags, &dest->effectTags, data->effectCount);

		buf->pop_stream();
	}

	void anim_class::dump(AnimationClass* asset)
	{
		const auto path = "animclass\\"s + asset->name;

		assetmanager::dumper dumper;
		if (!dumper.open(path))
		{
			return;
		}

		dumper.dump_single(asset);
		dumper.dump_string(asset->name);

		if (auto* sm = asset->stateMachine)
		{
			dumper.dump_single(sm);
			dumper.dump_string(SL_ConvertToString(sm->name));

			if (sm->states)
			{
				dumper.dump_array(sm->states, sm->stateCount);
				for (auto i = 0; i < sm->stateCount; i++)
				{
					const auto* state = &sm->states[i];
					dumper.dump_string(SL_ConvertToString(state->name));
					dumper.dump_string(SL_ConvertToString(state->notify));

					if (state->animEntries)
					{
						dumper.dump_array(state->animEntries, u8(state->entryCount));
						for (auto e = 0; e < u8(state->entryCount); e++)
						{
							dumper.dump_string(SL_ConvertToString(state->animEntries[e].animName));
						}
					}

					if (state->aliasList)
					{
						dumper.dump_array(state->aliasList, u8(state->aliasCount));
						for (auto a = 0; a < u8(state->aliasCount); a++)
						{
							dumper.dump_string(SL_ConvertToString(state->aliasList[a].aliasName));
							if (state->aliasList[a].aliasInfo)
							{
								dumper.dump_array(state->aliasList[a].aliasInfo, u8(state->aliasList[a].animCount));
							}
						}
					}
				}
			}

			if (sm->aimSets)
			{
				dumper.dump_array(sm->aimSets, sm->aimSetCount);
				for (auto i = 0; i < sm->aimSetCount; i++)
				{
					const auto* aim = &sm->aimSets[i];
					dumper.dump_string(SL_ConvertToString(aim->name));
					dumper.dump_string(SL_ConvertToString(aim->rootName));

					if (aim->animName)
					{
						dumper.dump_array(aim->animName, aim->animCount);
						for (auto j = 0; j < aim->animCount; j++)
						{
							dumper.dump_string(SL_ConvertToString(aim->animName[j]));
						}
					}
				}
			}
		}

		dumper.dump_string(SL_ConvertToString(asset->animTree));
		dumper.dump_asset(asset->scriptable);

		const auto dump_strings = [&](scr_string_t* arr, int count)
		{
			if (arr)
			{
				dumper.dump_array(arr, count);
				for (auto i = 0; i < count; i++)
				{
					dumper.dump_string(SL_ConvertToString(arr[i]));
				}
			}
		};

		dump_strings(asset->soundNotes, asset->soundCount);
		dump_strings(asset->soundNames, asset->soundCount);
		dump_strings(asset->soundOptions, asset->soundCount);
		dump_strings(asset->effectNotes, asset->effectCount);

		if (asset->effectDefs)
		{
			dumper.dump_array(asset->effectDefs, asset->effectCount);
			for (auto i = 0; i < asset->effectCount; i++)
			{
				const auto& def = asset->effectDefs[i];
				if (def.type == FX_COMBINED_VFX)
				{
					dumper.dump_asset(def.u.vfx);
				}
				else
				{
					dumper.dump_asset(def.u.fx);
				}
			}
		}

		dump_strings(asset->effectTags, asset->effectCount);

		dumper.close();
	}
}
