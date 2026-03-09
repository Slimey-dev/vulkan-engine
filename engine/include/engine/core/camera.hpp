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

    void processKeyboard(Window& window, float delta_time);
    void processMouse(float x_offset, float y_offset);
    void processScroll(float y_offset);

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
};

}  // namespace engine
