#include <std_include.hpp>
#include "zonetool/t7/converter/iw7/include.hpp"
#include "model_offset.hpp"

#include <utils/io.hpp>

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace model_offset
		{
			namespace
			{
				const offset_t none{};
				std::unordered_map<std::string, offset_t> offsets;
				bool loaded = false;

				void quat_to_mat(const float* q, float m[3][3])
				{
					const float x = q[0], y = q[1], z = q[2], w = q[3];
					const float n = std::sqrt(x * x + y * y + z * z + w * w);
					const float s = (n > 0.0f) ? (1.0f / n) : 1.0f;
					const float qx = x * s, qy = y * s, qz = z * s, qw = w * s;

					m[0][0] = 1.0f - 2.0f * (qy * qy + qz * qz);
					m[0][1] = 2.0f * (qx * qy - qz * qw);
					m[0][2] = 2.0f * (qx * qz + qy * qw);
					m[1][0] = 2.0f * (qx * qy + qz * qw);
					m[1][1] = 1.0f - 2.0f * (qx * qx + qz * qz);
					m[1][2] = 2.0f * (qy * qz - qx * qw);
					m[2][0] = 2.0f * (qx * qz - qy * qw);
					m[2][1] = 2.0f * (qy * qz + qx * qw);
					m[2][2] = 1.0f - 2.0f * (qx * qx + qy * qy);
				}

				void load()
				{
					loaded = true;

					const std::string path = "t7_to_iw7_model_offsets.json";
					if (!utils::io::file_exists(path))
					{
						return;
					}

					try
					{
						const auto data = nlohmann::json::parse(utils::io::read_file(path));
						for (auto& [key, value] : data.items())
						{
							offset_t off{};

							// a model may need only a rename, only a transform, or both
							if (value.contains("quat") && value.contains("trans"))
							{
								const auto& q = value["quat"];
								const auto& t = value["trans"];

								float quat[4] = { q[0].get<float>(), q[1].get<float>(),
												  q[2].get<float>(), q[3].get<float>() };
								quat_to_mat(quat, off.rot);
								off.trans[0] = t[0].get<float>();
								off.trans[1] = t[1].get<float>();
								off.trans[2] = t[2].get<float>();
								off.valid = true;
							}

							if (value.contains("bones"))
							{
								for (auto& [from, to] : value["bones"].items())
								{
									off.bones[from] = to.get<std::string>();
								}
							}

							offsets[key] = off;
							ZONETOOL_INFO("model offset loaded for \"%s\" (transform %s, %llu bone rename(s))",
								key.data(), off.valid ? "yes" : "no", off.bones.size());
						}
					}
					catch (const std::exception& e)
					{
						ZONETOOL_ERROR("failed to parse %s: %s", path.data(), e.what());
						offsets.clear();
					}
				}
			}

			const offset_t& get(const std::string& name)
			{
				if (!loaded)
				{
					load();
				}

				for (const auto& [key, off] : offsets)
				{
					if (name.compare(0, key.size(), key) == 0)
					{
						return offsets.at(key);
					}
				}

				return none;
			}

			void apply_point(const offset_t& off, float* xyz)
			{
				if (!off.valid) return;

				const float x = xyz[0], y = xyz[1], z = xyz[2];
				xyz[0] = off.rot[0][0] * x + off.rot[0][1] * y + off.rot[0][2] * z + off.trans[0];
				xyz[1] = off.rot[1][0] * x + off.rot[1][1] * y + off.rot[1][2] * z + off.trans[1];
				xyz[2] = off.rot[2][0] * x + off.rot[2][1] * y + off.rot[2][2] * z + off.trans[2];
			}

			void apply_dir(const offset_t& off, float* xyz)
			{
				if (!off.valid) return;

				const float x = xyz[0], y = xyz[1], z = xyz[2];
				xyz[0] = off.rot[0][0] * x + off.rot[0][1] * y + off.rot[0][2] * z;
				xyz[1] = off.rot[1][0] * x + off.rot[1][1] * y + off.rot[1][2] * z;
				xyz[2] = off.rot[2][0] * x + off.rot[2][1] * y + off.rot[2][2] * z;
			}

			void apply_quat(const offset_t& off, float* quat)
			{
				if (!off.valid) return;

				// rebuild the bone's rotation as off.rot * bone_rot, via matrices
				float b[3][3];
				quat_to_mat(quat, b);

				float r[3][3];
				for (auto i = 0; i < 3; i++)
				{
					for (auto j = 0; j < 3; j++)
					{
						r[i][j] = off.rot[i][0] * b[0][j] + off.rot[i][1] * b[1][j] + off.rot[i][2] * b[2][j];
					}
				}

				const float tr = r[0][0] + r[1][1] + r[2][2];
				if (tr > 0.0f)
				{
					const float s = 0.5f / std::sqrt(tr + 1.0f);
					quat[0] = (r[2][1] - r[1][2]) * s;
					quat[1] = (r[0][2] - r[2][0]) * s;
					quat[2] = (r[1][0] - r[0][1]) * s;
					quat[3] = 0.25f / s;
				}
				else if (r[0][0] > r[1][1] && r[0][0] > r[2][2])
				{
					const float s = 2.0f * std::sqrt(1.0f + r[0][0] - r[1][1] - r[2][2]);
					quat[0] = 0.25f * s;
					quat[1] = (r[0][1] + r[1][0]) / s;
					quat[2] = (r[0][2] + r[2][0]) / s;
					quat[3] = (r[2][1] - r[1][2]) / s;
				}
				else if (r[1][1] > r[2][2])
				{
					const float s = 2.0f * std::sqrt(1.0f + r[1][1] - r[0][0] - r[2][2]);
					quat[0] = (r[0][1] + r[1][0]) / s;
					quat[1] = 0.25f * s;
					quat[2] = (r[1][2] + r[2][1]) / s;
					quat[3] = (r[0][2] - r[2][0]) / s;
				}
				else
				{
					const float s = 2.0f * std::sqrt(1.0f + r[2][2] - r[0][0] - r[1][1]);
					quat[0] = (r[0][2] + r[2][0]) / s;
					quat[1] = (r[1][2] + r[2][1]) / s;
					quat[2] = 0.25f * s;
					quat[3] = (r[1][0] - r[0][1]) / s;
				}
			}
		}
	}
}
