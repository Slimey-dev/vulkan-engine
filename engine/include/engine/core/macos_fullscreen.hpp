#pragma once

struct GLFWwindow;

namespace engine {

#ifdef __APPLE__
void enableNativeFullscreen(GLFWwindow* window);
void toggleNativeFullscreen(GLFWwindow* window);
#endif

}  // namespace engine
