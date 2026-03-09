#pragma once

#include <GLFW/glfw3.h>

#include <cstdint>
#include <string>

namespace engine {

class Window {
public:
    Window(uint32_t width, uint32_t height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() const;
    void pollEvents();
    void close();

    GLFWwindow* getHandle() const { return window_; }
    VkExtent2D getExtent() const;
    bool wasResized() const { return framebuffer_resized_; }
    void resetResizedFlag() { framebuffer_resized_ = false; }

    bool isKeyPressed(int key) const;
    void getMouseDelta(float& dx, float& dy);
    float getScrollDelta();
    void setCursorCaptured(bool captured);
    bool isCursorCaptured() const { return cursor_captured_; }

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    GLFWwindow* window_ = nullptr;
    uint32_t width_;
    uint32_t height_;
    bool framebuffer_resized_ = false;

    double last_mouse_x_ = 0.0;
    double last_mouse_y_ = 0.0;
    float mouse_dx_ = 0.0f;
    float mouse_dy_ = 0.0f;
    float scroll_dy_ = 0.0f;
    bool cursor_captured_ = false;
    bool first_mouse_ = true;
};

}  // namespace engine
