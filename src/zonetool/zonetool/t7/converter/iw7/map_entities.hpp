#pragma once

namespace zonetool::t7
{
	namespace converter::iw7::map_entities
	{
		struct key_value
		{
			std::string key;
			std::string value;
			bool quoted = true; // key IDs (the physics asset's 51961) are written bare, names quoted
		};

		struct entity
		{
			std::vector<key_value> keys;

			const std::string* get(const std::string& key) const
			{
				for (const auto& kv : this->keys)
				{
					if (kv.key == key)
					{
						return &kv.value;
					}
				}
				return nullptr;
			}

			bool is(const std::string& key, const std::string& value) const
			{
				const auto* v = this->get(key);
				return v && *v == value;
			}

			void set(const std::string& key, const std::string& value)
			{
				for (auto& kv : this->keys)
				{
					if (kv.key == key)
					{
						kv.value = value;
						return;
					}
				}
				this->keys.push_back({ key, value, true });
			}

			void erase(const std::string& key)
			{
				std::erase_if(this->keys, [&](const auto& kv) { return kv.key == key; });
			}
		};

		// one pair per line: "key" "value", or key "value" for a key ID
		std::vector<entity> parse(const std::string& text);
		std::string write(const std::vector<entity>& ents);

		// the path node entities of the converted aipaths, in its fixed node order (navmesh::convert)
		void set_path_nodes(std::vector<entity> nodes);

		// script_models the GfxWorld conversion places for static models IW7 cannot draw as static models (gfxworld.cpp)
		void set_script_models(std::vector<entity> models);

		// light entities for scriptable primary lights (comworld.cpp: the flickering lights)
		void set_light_entities(std::vector<entity> lights);

		// keeps what an IW7 MP / CP server can spawn: BO3 path nodes, client effects and render / tool entities go,
		// other BO3-only entities become script_structs (the BO3 classname in script_type); the converted path nodes
		// are added at the end
		void keep_spawnable(std::vector<entity>& ents);
	}
}
