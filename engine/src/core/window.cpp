#include <engine/core/log.hpp>
#include <engine/core/window.hpp>
#ifdef __APPLE__
#include <engine/core/macos_fullscreen.hpp>
#endif

#include <stdexcept>

namespace engine {

Window::Window(const std::string& title) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    width_ = mode->width;
    height_ = mode->height;

#ifdef __APPLE__
    // Create windowed, then enter macOS native fullscreen (no flicker)
    window_ = glfwCreateWindow(width_, height_, title.c_str(), nullptr, nullptr);
#else
    window_ = glfwCreateWindow(width_, height_, title.c_str(), monitor, nullptr);
#endif
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

#ifdef __APPLE__
    enableNativeFullscreen(window_);
    toggleNativeFullscreen(window_);
#endif

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
    glfwSetCursorPosCallback(window_, cursorPosCallback);
    glfwSetScrollCallback(window_, scrollCallback);
    glfwSetKeyCallback(window_, keyCallback);

    int fb_w, fb_h;
    glfwGetFramebufferSize(window_, &fb_w, &fb_h);
    logInfo("Window created: {}x{} (framebuffer: {}x{})", width_, height_, fb_w, fb_h);
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

void Window::close() {
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
}

VkExtent2D Window::getExtent() const {
    int w, h;
    glfwGetFramebufferSize(window_, &w, &h);
    return {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
}

bool Window::isKeyPressed(int key) const {
    return glfwGetKey(window_, key) == GLFW_PRESS;
}

void Window::getMouseDelta(float& dx, float& dy) {
    dx = mouse_dx_;
    dy = mouse_dy_;
    mouse_dx_ = 0.0f;
    mouse_dy_ = 0.0f;
}

float Window::getScrollDelta() {
    float delta = scroll_dy_;
    scroll_dy_ = 0.0f;
    return delta;
}

void Window::setCursorCaptured(bool captured) {
    cursor_captured_ = captured;
    glfwSetInputMode(window_, GLFW_CURSOR,
                     captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    if (captured) {
        first_mouse_ = true;
    }
}

void Window::framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    self->framebuffer_resized_ = true;
}

void Window::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (!self->cursor_captured_) return;

    if (self->first_mouse_) {
        self->last_mouse_x_ = xpos;
        self->last_mouse_y_ = ypos;
        self->first_mouse_ = false;
        return;
    }

    self->mouse_dx_ += static_cast<float>(self->last_mouse_x_ - xpos);
    self->mouse_dy_ += static_cast<float>(self->last_mouse_y_ - ypos);
    self->last_mouse_x_ = xpos;
    self->last_mouse_y_ = ypos;
}

void Window::scrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    self->scroll_dy_ += static_cast<float>(yoffset);
}

void Window::keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        self->toggleUIMode();
    }
}

void Window::toggleUIMode() {
    ui_mode_ = !ui_mode_;
    setCursorCaptured(!ui_mode_);
}

}  // namespace engine
