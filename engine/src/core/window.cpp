#include <engine/core/log.hpp>
#include <engine/core/window.hpp>

#include <stdexcept>

namespace engine {

Window::Window(uint32_t width, uint32_t height, const std::string& title)
    : width_(width), height_(height) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window_ = glfwCreateWindow(width_, height_, title.c_str(), nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);

    logInfo("Window created: {}x{}", width_, height_);
}

Window::~Window() {
    glfwDestroyWindow(window_);
    glfwTerminate();
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(window_);
}

void Window::pollEvents() {
    glfwPollEvents();
}

VkExtent2D Window::getExtent() const {
    int w, h;
    glfwGetFramebufferSize(window_, &w, &h);
    return {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
}

void Window::framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    self->framebuffer_resized_ = true;
}

}  // namespace engine
