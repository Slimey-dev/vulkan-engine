#include <engine/core/macos_fullscreen.hpp>

#ifdef __APPLE__
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#import <Cocoa/Cocoa.h>

namespace engine {

void enableNativeFullscreen(GLFWwindow* window) {
    NSWindow* nsWindow = glfwGetCocoaWindow(window);
    [nsWindow setCollectionBehavior:[nsWindow collectionBehavior] |
                                    NSWindowCollectionBehaviorFullScreenPrimary];
}

void toggleNativeFullscreen(GLFWwindow* window) {
    NSWindow* nsWindow = glfwGetCocoaWindow(window);
    [nsWindow toggleFullScreen:nil];
}

}  // namespace engine
#endif
