#pragma once

#include <glm/glm.hpp>

namespace engine {

class Window;
struct CameraBounds;

class Camera {
public:
    Camera(glm::vec3 position = glm::vec3(2.0f, 2.0f, 2.0f),
           float yaw = -135.0f, float pitch = -35.26f);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspect) const;
    glm::vec3 getPosition() const { return position_; }
    glm::vec3 getFront() const { return front_; }
    glm::vec3 getUp() const { return up_; }

    void processKeyboard(Window& window, float delta_time);
    void processMouse(float x_offset, float y_offset);
    void processScroll(float y_offset);

    void reset(glm::vec3 pos, float yaw, float pitch);
    void setBounds(float min_x, float max_x, float min_y, float max_y, float max_z);

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

    static constexpr float kSprintSpeedMultiplier = 2.0f;
    static constexpr float kSprintFovBonus = 10.0f;
    static constexpr float kSprintFovLerpSpeed = 8.0f;

    bool sprinting_ = false;
    float sprint_fov_offset_ = 0.0f;

    float bounds_min_x_ = -1e9f, bounds_max_x_ = 1e9f;
    float bounds_min_y_ = -1e9f, bounds_max_y_ = 1e9f;
    float bounds_max_z_ = 1e9f;
};

}  // namespace engine
