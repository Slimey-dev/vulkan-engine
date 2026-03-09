#include <engine/core/log.hpp>
#include <engine/core/window.hpp>
#include <engine/renderer/renderer.hpp>

#include <cstdlib>
#include <exception>

int main() {
    try {
        engine::Window window(1280, 720, "Vulkan Engine");
        engine::Renderer renderer(window);

        while (!window.shouldClose()) {
            window.pollEvents();
            renderer.drawFrame();
        }
    } catch (const std::exception& e) {
        engine::logError("Fatal: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
