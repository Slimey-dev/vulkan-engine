# Vulkan Engine

Custom Vulkan 1.3+ game engine in C++20. Currently renders a rotating 3D cube with per-face colors, perspective camera, and depth testing.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
./build/app/triangle
```

Release: `cmake -B build-release -DCMAKE_BUILD_TYPE=Release && cmake --build build-release -j$(nproc)`

## Project Structure

- `engine/` — Static library (`VulkanEngine`). Headers in `include/engine/`, sources in `src/`.
  - `core/` — Window (GLFW), logging
  - `renderer/` — Vulkan wrappers (instance, device, swapchain, pipeline, buffer, descriptors), renderer orchestrator
- `app/` — Executable (`triangle`). Links against engine.
- `shaders/` — GLSL sources. Compiled to SPIR-V at build time via `cmake/CompileShaders.cmake`.
- `cmake/` — Reusable CMake modules.

## Conventions

- **Namespace**: `engine::`. Buffer utilities in `engine::vk_buffer::`.
- **Headers**: `#include <engine/subsystem/file.hpp>` (angle brackets, namespaced path).
- **Vulkan wrappers**: `Vulkan*` prefix classes (VulkanInstance, VulkanDevice, VulkanSwapchain, VulkanPipeline, VulkanDescriptors). RAII, non-copyable.
- **Ownership**: Renderer owns all Vulkan objects via `std::unique_ptr`. VkSurfaceKHR is a raw handle owned by Renderer.
- **Logging**: `engine::logInfo/logError/logWarn` — header-only templates using `std::format`.
- **Style**: `.clang-format` (Google base, 4-space indent, 100 col). Trailing underscores for members (`device_`).
- **Shaders**: GLSL `#version 450`. Compiled to `${CMAKE_BINARY_DIR}/shaders/`. Loaded at runtime via `SHADER_DIR` compile definition.
- **Validation**: Layers enabled in Debug (`#ifndef NDEBUG`), stripped in Release.

## Dependencies (all system-installed)

Vulkan SDK 1.4, GLFW 3.4, GLM 1.0, glslc, CMake 3.24+

## Key Design Decisions

- render_finished semaphores are per-swapchain-image (not per-frame-in-flight) to avoid semaphore reuse errors
- Depth format chosen at runtime via `findSupportedFormat()` (prefers D32_SFLOAT)
- Uniform buffers are persistently mapped (one per frame-in-flight, HOST_VISIBLE + HOST_COHERENT)
- Vertex/index buffers use staging transfer to device-local memory
- Front face winding is COUNTER_CLOCKWISE (GLM right-handed convention)
- `ubo.proj[1][1] *= -1` to flip Y for Vulkan coordinate system
