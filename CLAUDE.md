# Vulkan Engine

Custom Vulkan 1.3+ game engine in C++20. Renders a textured rotating 3D cube loaded from OBJ, with interactive FPS camera, depth testing, and anisotropic texture filtering.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
./build/app/triangle
```

Release: `cmake -B build-release -DCMAKE_BUILD_TYPE=Release && cmake --build build-release -j$(nproc)`

## Project Structure

- `engine/` — Static library (`VulkanEngine`). Headers in `include/engine/`, sources in `src/`.
  - `core/` — Window (GLFW), Camera (FPS), logging
  - `renderer/` — Vulkan wrappers (instance, device, swapchain, pipeline, buffer, texture, descriptors), mesh loading, renderer orchestrator
- `app/` — Executable (`triangle`). Links against engine.
- `assets/` — Runtime assets (OBJ models, textures). Referenced via `ASSETS_DIR` compile definition.
- `shaders/` — GLSL sources. Compiled to SPIR-V at build time via `cmake/CompileShaders.cmake`.
- `cmake/` — Reusable CMake modules.
- `third_party/` — Vendored headers: `stb/stb_image.h`, `tinyobj/tiny_obj_loader.h`.

## Conventions

- **Namespace**: `engine::`. Buffer utilities in `engine::vk_buffer::`.
- **Headers**: `#include <engine/subsystem/file.hpp>` (angle brackets, namespaced path).
- **Vulkan wrappers**: `Vulkan*` prefix classes (VulkanInstance, VulkanDevice, VulkanSwapchain, VulkanPipeline, VulkanTexture, VulkanDescriptors). RAII, non-copyable. `Mesh` class owns vertex/index buffers.
- **Ownership**: Renderer owns all Vulkan objects via `std::unique_ptr`. VkSurfaceKHR is a raw handle owned by Renderer.
- **Logging**: `engine::logInfo/logError/logWarn` — header-only templates using `std::format`.
- **Style**: `.clang-format` (Google base, 4-space indent, 100 col). Trailing underscores for members (`device_`).
- **Shaders**: GLSL `#version 450`. Compiled to `${CMAKE_BINARY_DIR}/shaders/`. Loaded at runtime via `SHADER_DIR` compile definition.
- **Validation**: Layers enabled in Debug (`#ifndef NDEBUG`), stripped in Release.

## Dependencies

System-installed: Vulkan SDK 1.4, GLFW 3.4, GLM 1.0, glslc, CMake 3.24+
Vendored (third_party/): stb_image.h, tiny_obj_loader.h

## Key Design Decisions

- render_finished semaphores are per-swapchain-image (not per-frame-in-flight) to avoid semaphore reuse errors
- Depth format chosen at runtime via `findSupportedFormat()` (prefers D32_SFLOAT)
- Uniform buffers are persistently mapped (one per frame-in-flight, HOST_VISIBLE + HOST_COHERENT)
- Vertex/index buffers use staging transfer to device-local memory
- Front face winding is COUNTER_CLOCKWISE (GLM right-handed convention)
- `ubo.proj[1][1] *= -1` to flip Y for Vulkan coordinate system (now in Camera::getProjectionMatrix)
- Textures use R8G8B8A8_SRGB format, anisotropic filtering, REPEAT addressing
- Descriptor set: binding 0 = UBO (vertex), binding 1 = combined image sampler (fragment)
- OBJ loading flips V texcoord (`1.0 - v`) for Vulkan top-left UV origin
- Mesh uses uint32_t indices (VK_INDEX_TYPE_UINT32)
- Camera uses Z-up world, mouse X delta is negated for correct yaw direction
