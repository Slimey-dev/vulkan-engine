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

    GLFWwindow* getHandle() const { return window_; }
    VkExtent2D getExtent() const;
    bool wasResized() const { return framebuffer_resized_; }
    void resetResizedFlag() { framebuffer_resized_ = false; }

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    GLFWwindow* window_ = nullptr;
    uint32_t width_;
    uint32_t height_;
    bool framebuffer_resized_ = false;
};

}  // namespace engine
