#pragma once

#include <engine/renderer/mesh.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace engine {

struct Transform {
    glm::vec3 position{0};
    glm::vec3 rotation{0};  // euler degrees
    glm::vec3 scale{1};

    glm::mat4 matrix() const {
        glm::mat4 m = glm::translate(glm::mat4(1.0f), position);
        m = glm::rotate(m, glm::radians(rotation.x), glm::vec3(1, 0, 0));
        m = glm::rotate(m, glm::radians(rotation.y), glm::vec3(0, 1, 0));
        m = glm::rotate(m, glm::radians(rotation.z), glm::vec3(0, 0, 1));
        m = glm::scale(m, scale);
        return m;
    }
};

struct MeshRenderer {
    Mesh* mesh = nullptr;
};

struct Rotator {
    glm::vec3 axis{0, 0, 1};
    float speed = 90.0f;  // degrees/sec
};

struct PointLight {
    glm::vec3 color{1, 1, 1};
};

struct VolumetricCone {};  // tag: rendered with additive blend pipeline
struct Emissive {};        // tag: skip lighting, output vertex color directly

}  // namespace engine
