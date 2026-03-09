#pragma once

#include <glm/glm.hpp>

namespace engine {

class Window;

class Camera {
public:
    Camera(glm::vec3 position = glm::vec3(2.0f, 2.0f, 2.0f),
           float yaw = -135.0f, float pitch = -35.26f);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspect) const;
    glm::vec3 getPosition() const { return position_; }

    void processKeyboard(Window& window, float delta_time);
    void processMouse(float x_offset, float y_offset);
    void processScroll(float y_offset);

    bool didJump() { bool v = just_jumped_; just_jumped_ = false; return v; }
    bool didLand() { bool v = just_landed_; just_landed_ = false; return v; }

private:
    void updateVectors();

    glm::vec3 position_;
    glm::vec3 front_;
    glm::vec3 up_;
    glm::vec3 right_;

    float yaw_;
    float pitch_;
    float speed_ = 2.5f;
    float sensitivity_ = 0.1f;
    float fov_ = 45.0f;

    float vertical_velocity_ = 0.0f;
    bool grounded_ = false;
    bool just_jumped_ = false;
    bool just_landed_ = false;

    static constexpr float kGravity = 20.0f;
    static constexpr float kJumpImpulse = 7.0f;
    static constexpr float kEyeHeight = 1.7f;
    static constexpr float kGroundZ = -0.5f;
    static constexpr float kMinZ = kGroundZ + kEyeHeight;
};

}  // namespace engine
