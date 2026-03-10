#include <engine/ecs/components.hpp>
#include <engine/core/log.hpp>
#include <engine/renderer/scene.hpp>
#include <engine/renderer/vk_buffer.hpp>
#include <engine/renderer/vk_device.hpp>

#include <cmath>
#include <string>

namespace engine {

// OutdoorScene

void OutdoorScene::init(VulkanDevice& device, VkCommandPool pool) {
    meshes.push_back(
        Mesh::loadFromOBJ(device, pool, std::string(ASSETS_DIR) + "cube.obj"));

    std::vector<Vertex> ground_verts = {
        {{-100.0f, -100.0f, -0.5f}, {0.7f, 0.7f, 0.7f}, {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{ 100.0f, -100.0f, -0.5f}, {0.7f, 0.7f, 0.7f}, {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{ 100.0f,  100.0f, -0.5f}, {0.7f, 0.7f, 0.7f}, {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{-100.0f,  100.0f, -0.5f}, {0.7f, 0.7f, 0.7f}, {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    };
    std::vector<uint32_t> ground_indices = {0, 1, 2, 2, 3, 0};
    meshes.push_back(std::make_unique<Mesh>(device, pool, ground_verts, ground_indices));

    // Cube entity: rotating
    Entity cube = registry.create();
    registry.emplace<Transform>(cube);
    registry.emplace<MeshRenderer>(cube).mesh = meshes[0].get();
    registry.emplace<Rotator>(cube);

    // Ground entity: static
    Entity ground = registry.create();
    registry.emplace<Transform>(ground);
    registry.emplace<MeshRenderer>(ground).mesh = meshes[1].get();

    logInfo("OutdoorScene initialized ({} entities, {} meshes)", 2, meshes.size());
}

// IndoorScene

IndoorScene::IndoorScene() {
    camera_start = {3, 3, 1.2f};
    camera_yaw = -135.0f;
    camera_pitch = -15.0f;
    light_pos = {0, 0, 6.0f};
    light_color = {1, 1, 1};
    light_dir = {0, 0, -1};
    light_cone_angle = 50.0f;
    fog_density = 0.04f;
    fog_color = {0.08f, 0.08f, 0.09f};
    skybox_enabled = false;
    bounds = {-4.5f, 4.5f, -4.5f, 4.5f, 7.2f};
    shadow_ortho_size = 6.0f;
    shadow_far = 15.0f;
    shadow_up = {0, 1, 0};
}

static void createRoomMesh(VulkanDevice& device, VkCommandPool pool,
                           std::vector<std::unique_ptr<Mesh>>& meshes) {
    // Inverted box: X[-5,5], Y[-5,5], Z[-0.5, 7.5] → normals inward
    constexpr float xn = -5.0f, xp = 5.0f;
    constexpr float yn = -5.0f, yp = 5.0f;
    constexpr float zn = -0.5f, zp = 7.5f;
    constexpr float tw = 4.0f;                   // tex tile count for 10-unit walls
    constexpr float th = tw * (8.0f / 10.0f);    // keep cells square (wall is 8 tall)

    glm::vec3 white{1, 1, 1};

    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;

    auto quad = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d,
                    glm::vec3 normal, glm::vec2 uv_a, glm::vec2 uv_b,
                    glm::vec2 uv_c, glm::vec2 uv_d) {
        uint32_t base = static_cast<uint32_t>(verts.size());
        verts.push_back({a, white, uv_a, normal});
        verts.push_back({b, white, uv_b, normal});
        verts.push_back({c, white, uv_c, normal});
        verts.push_back({d, white, uv_d, normal});
        // CCW from inside
        indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    };

    // Floor (Z = zn, normal +Z)
    quad({xn, yn, zn}, {xp, yn, zn}, {xp, yp, zn}, {xn, yp, zn},
         {0, 0, 1}, {0, 0}, {tw, 0}, {tw, tw}, {0, tw});

    // Ceiling (Z = zp, normal -Z)
    quad({xn, yp, zp}, {xp, yp, zp}, {xp, yn, zp}, {xn, yn, zp},
         {0, 0, -1}, {0, 0}, {tw, 0}, {tw, tw}, {0, tw});

    // Wall -Y (normal +Y) — CCW from inside
    quad({xn, yn, zn}, {xn, yn, zp}, {xp, yn, zp}, {xp, yn, zn},
         {0, 1, 0}, {0, 0}, {0, th}, {tw, th}, {tw, 0});

    // Wall +Y (normal -Y) — CCW from inside
    quad({xn, yp, zn}, {xp, yp, zn}, {xp, yp, zp}, {xn, yp, zp},
         {0, -1, 0}, {0, 0}, {tw, 0}, {tw, th}, {0, th});

    // Wall -X (normal +X) — CCW from inside
    quad({xn, yn, zn}, {xn, yp, zn}, {xn, yp, zp}, {xn, yn, zp},
         {1, 0, 0}, {0, 0}, {tw, 0}, {tw, th}, {0, th});

    // Wall +X (normal -X) — CCW from inside
    quad({xp, yn, zn}, {xp, yn, zp}, {xp, yp, zp}, {xp, yp, zn},
         {-1, 0, 0}, {0, 0}, {0, th}, {tw, th}, {tw, 0});

    meshes.push_back(std::make_unique<Mesh>(device, pool, verts, indices));
}

static void createBoxMesh(VulkanDevice& device, VkCommandPool pool,
                          std::vector<std::unique_ptr<Mesh>>& meshes,
                          glm::vec3 half, glm::vec3 center, glm::vec3 color,
                          glm::vec2 uv) {
    // Outward-facing box
    float xn = center.x - half.x, xp = center.x + half.x;
    float yn = center.y - half.y, yp = center.y + half.y;
    float zn = center.z - half.z, zp = center.z + half.z;

    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;

    auto quad = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 n) {
        uint32_t base = static_cast<uint32_t>(verts.size());
        verts.push_back({a, color, uv, n});
        verts.push_back({b, color, uv, n});
        verts.push_back({c, color, uv, n});
        verts.push_back({d, color, uv, n});
        indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    };

    quad({xn, yn, zp}, {xp, yn, zp}, {xp, yp, zp}, {xn, yp, zp}, {0, 0, 1});   // +Z
    quad({xn, yp, zn}, {xp, yp, zn}, {xp, yn, zn}, {xn, yn, zn}, {0, 0, -1});   // -Z
    quad({xn, yn, zn}, {xp, yn, zn}, {xp, yn, zp}, {xn, yn, zp}, {0, -1, 0});   // -Y
    quad({xp, yp, zn}, {xn, yp, zn}, {xn, yp, zp}, {xp, yp, zp}, {0, 1, 0});    // +Y
    quad({xn, yp, zn}, {xn, yn, zn}, {xn, yn, zp}, {xn, yp, zp}, {-1, 0, 0});   // -X
    quad({xp, yn, zn}, {xp, yp, zn}, {xp, yp, zp}, {xp, yn, zp}, {1, 0, 0});    // +X

    meshes.push_back(std::make_unique<Mesh>(device, pool, verts, indices));
}

void IndoorScene::init(VulkanDevice& device, VkCommandPool pool) {
    // [0] Cube
    meshes.push_back(
        Mesh::loadFromOBJ(device, pool, std::string(ASSETS_DIR) + "cube.obj"));

    // [1] Room
    createRoomMesh(device, pool, meshes);

    // [2] Lamp body
    glm::vec3 lamp_color{5.0f, 4.5f, 3.0f};
    createBoxMesh(device, pool, meshes,
                  {0.2f, 0.2f, 0.15f}, {0, 0, 6.15f}, lamp_color, {0.5f, 0.5f});

    // [3] Lamp cord
    createBoxMesh(device, pool, meshes,
                  {0.01f, 0.01f, 0.6f}, {0, 0, 6.9f}, lamp_color, {0.5f, 0.5f});

    // Cube entity
    Entity cube = registry.create();
    registry.emplace<Transform>(cube);
    registry.emplace<MeshRenderer>(cube).mesh = meshes[0].get();
    registry.emplace<Rotator>(cube);

    // Room entity
    Entity room = registry.create();
    registry.emplace<Transform>(room);
    registry.emplace<MeshRenderer>(room).mesh = meshes[1].get();

    // Lamp body entity
    Entity lamp = registry.create();
    registry.emplace<Transform>(lamp);
    registry.emplace<MeshRenderer>(lamp).mesh = meshes[2].get();

    // Lamp cord entity
    Entity cord = registry.create();
    registry.emplace<Transform>(cord);
    registry.emplace<MeshRenderer>(cord).mesh = meshes[3].get();

    // Point light entity
    Entity light_e = registry.create();
    auto& lt = registry.emplace<Transform>(light_e);
    lt.position = {0, 0, 6.0f};
    registry.emplace<PointLight>(light_e);

    // [4] Volumetric cone mesh — truncated cone with outward normals + height gradient
    // Rendered two-sided with Fresnel-fade shader: bright when viewed edge-on.
    // No overlapping geometry = no center line artifact.
    {
        constexpr int segments = 24;
        constexpr int rings = 6;
        float cone_length = light_pos.z + 0.5f;  // light to floor
        float tan_angle = std::tan(glm::radians(light_cone_angle));
        float top_radius = 0.15f;  // small opening at lamp
        glm::vec3 cone_color{1.0f, 0.95f, 0.8f};
        // Cone surface normal: for a cone with half-angle alpha, the normal
        // points outward at angle (90-alpha) from the axis
        float cos_half = std::cos(glm::radians(light_cone_angle));
        float sin_half = std::sin(glm::radians(light_cone_angle));

        std::vector<Vertex> cverts;
        std::vector<uint32_t> cindices;

        for (int r = 0; r <= rings; r++) {
            float t = static_cast<float>(r) / static_cast<float>(rings);
            float radius = top_radius + t * (cone_length * tan_angle - top_radius);
            float z = light_pos.z - t * cone_length;

            for (int s = 0; s < segments; s++) {
                float a = 2.0f * glm::pi<float>() * static_cast<float>(s) /
                          static_cast<float>(segments);
                float cx = std::cos(a);
                float cy = std::sin(a);

                // Outward-pointing cone surface normal
                glm::vec3 normal = glm::normalize(glm::vec3(cx * cos_half, cy * cos_half, sin_half));

                // Encode height gradient in vertex color alpha channel via the color field:
                // color.r = base color, color.g = base color, color.b = height (0=bottom, 1=top)
                // We'll use texCoord.y for the height gradient instead
                cverts.push_back({{cx * radius, cy * radius, z},
                                  cone_color,
                                  {static_cast<float>(s) / segments, 1.0f - t},
                                  normal});
            }
        }

        // Quads between rings
        for (int r = 0; r < rings; r++) {
            uint32_t ring_a = static_cast<uint32_t>(r * segments);
            uint32_t ring_b = ring_a + static_cast<uint32_t>(segments);
            for (int s = 0; s < segments; s++) {
                uint32_t curr = ring_a + static_cast<uint32_t>(s);
                uint32_t next = ring_a + static_cast<uint32_t>((s + 1) % segments);
                uint32_t curr_b = ring_b + static_cast<uint32_t>(s);
                uint32_t next_b = ring_b + static_cast<uint32_t>((s + 1) % segments);
                cindices.insert(cindices.end(), {curr, next_b, next, curr, curr_b, next_b});
            }
        }

        logInfo("Volumetric cone: {} verts, {} indices (closed cone mesh)",
                cverts.size(), cindices.size());
        meshes.push_back(std::make_unique<Mesh>(device, pool, cverts, cindices));
    }

    Entity cone = registry.create();
    registry.emplace<Transform>(cone);
    registry.emplace<MeshRenderer>(cone).mesh = meshes[4].get();
    registry.emplace<VolumetricCone>(cone);

    logInfo("IndoorScene initialized ({} entities, {} meshes)", 7, meshes.size());
}

}  // namespace engine
