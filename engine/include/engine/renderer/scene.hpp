#pragma once

#include <engine/ecs/registry.hpp>
#include <engine/renderer/mesh.hpp>

#include <glm/glm.hpp>

#include <memory>
#include <vector>

namespace engine {

class VulkanDevice;

struct CameraBounds {
    float min_x = -1e9f, max_x = 1e9f;
    float min_y = -1e9f, max_y = 1e9f;
    float max_z = 1e9f;
};

class Scene {
public:
    virtual ~Scene() = default;
    virtual void init(VulkanDevice& device, VkCommandPool pool) = 0;
    virtual const char* name() const = 0;

    Registry registry;
    std::vector<std::unique_ptr<Mesh>> meshes;

    glm::vec3 camera_start{2, 2, 2};
    float camera_yaw = -135.0f;
    float camera_pitch = -35.26f;
    glm::vec3 light_pos{5, 5, 5};
    glm::vec3 light_color{1, 1, 1};
    glm::vec3 fog_color{0.02f, 0.02f, 0.03f};
    float fog_density = 0.15f;
    bool skybox_enabled = true;
    CameraBounds bounds;

    glm::vec3 light_dir{0, 0, -1};
    float light_cone_angle = 0.0f;  // degrees; 0 = point light (no cone)

    float shadow_ortho_size = 12.0f;
    float shadow_far = 20.0f;
    glm::vec3 shadow_up{0, 0, 1};

    float bloom_threshold = 1.0f;
    float bloom_intensity = 0.0f;
};

class OutdoorScene : public Scene {
public:
    void init(VulkanDevice& device, VkCommandPool pool) override;
    const char* name() const override { return "Outdoor"; }
};

class IndoorScene : public Scene {
public:
    IndoorScene();
    void init(VulkanDevice& device, VkCommandPool pool) override;
    const char* name() const override { return "Indoor"; }
};

}  // namespace engine
