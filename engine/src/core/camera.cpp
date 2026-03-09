#include <engine/core/camera.hpp>
#include <engine/core/window.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace engine {

static constexpr glm::vec3 kWorldUp = glm::vec3(0.0f, 0.0f, 1.0f);

Camera::Camera(glm::vec3 position, float yaw, float pitch)
    : position_(position), yaw_(yaw), pitch_(pitch) {
    updateVectors();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position_, position_ + front_, up_);
}

glm::mat4 Camera::getProjectionMatrix(float aspect) const {
    auto proj = glm::perspective(glm::radians(fov_), aspect, 0.1f, 100.0f);
    proj[1][1] *= -1;
    return proj;
}

void Camera::processKeyboard(Window& window, float delta_time) {
    float velocity = speed_ * delta_time;

    glm::vec3 forward = glm::normalize(glm::vec3(front_.x, front_.y, 0.0f));
    glm::vec3 strafe = glm::normalize(glm::cross(forward, kWorldUp));

    if (window.isKeyPressed(GLFW_KEY_W)) position_ += forward * velocity;
    if (window.isKeyPressed(GLFW_KEY_S)) position_ -= forward * velocity;
    if (window.isKeyPressed(GLFW_KEY_A)) position_ -= strafe * velocity;
    if (window.isKeyPressed(GLFW_KEY_D)) position_ += strafe * velocity;
    if (window.isKeyPressed(GLFW_KEY_SPACE) && grounded_) {
        vertical_velocity_ = kJumpImpulse;
        grounded_ = false;
    }

    vertical_velocity_ -= kGravity * delta_time;
    position_.z += vertical_velocity_ * delta_time;

    if (position_.z <= kMinZ) {
        position_.z = kMinZ;
        vertical_velocity_ = 0.0f;
        grounded_ = true;
    }
}

void Camera::processMouse(float x_offset, float y_offset) {
    yaw_ += x_offset * sensitivity_;
    pitch_ += y_offset * sensitivity_;
    pitch_ = std::clamp(pitch_, -89.0f, 89.0f);
    updateVectors();
}

void Camera::processScroll(float y_offset) {
    fov_ = std::clamp(fov_ - y_offset, 1.0f, 120.0f);
}

void Camera::updateVectors() {
    float yaw_rad = glm::radians(yaw_);
    float pitch_rad = glm::radians(pitch_);

    front_.x = std::cos(pitch_rad) * std::cos(yaw_rad);
    front_.y = std::cos(pitch_rad) * std::sin(yaw_rad);
    front_.z = std::sin(pitch_rad);
    front_ = glm::normalize(front_);

    right_ = glm::normalize(glm::cross(front_, kWorldUp));
    up_ = glm::normalize(glm::cross(right_, front_));
}

}  // namespace engine
