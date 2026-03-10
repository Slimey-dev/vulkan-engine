add_library(VulkanEngine STATIC
    src/core/audio.cpp
    src/core/camera.cpp
    src/core/window.cpp
    src/renderer/vk_instance.cpp
    src/renderer/vk_device.cpp
    src/renderer/vk_swapchain.cpp
    src/renderer/vk_pipeline.cpp
    src/renderer/vk_buffer.cpp
    src/renderer/vk_texture.cpp
    src/renderer/mesh.cpp
    src/renderer/vk_descriptors.cpp
    src/renderer/renderer.cpp
    src/renderer/scene.cpp
    ${CMAKE_SOURCE_DIR}/third_party/imgui/imgui.cpp
    ${CMAKE_SOURCE_DIR}/third_party/imgui/imgui_draw.cpp
    ${CMAKE_SOURCE_DIR}/third_party/imgui/imgui_tables.cpp
    ${CMAKE_SOURCE_DIR}/third_party/imgui/imgui_widgets.cpp
    ${CMAKE_SOURCE_DIR}/third_party/imgui/imgui_demo.cpp
    ${CMAKE_SOURCE_DIR}/third_party/imgui/imgui_impl_glfw.cpp
    ${CMAKE_SOURCE_DIR}/third_party/imgui/imgui_impl_vulkan.cpp
)

target_include_directories(VulkanEngine
    PUBLIC  include
    PRIVATE src
            ${CMAKE_SOURCE_DIR}/third_party/stb
            ${CMAKE_SOURCE_DIR}/third_party/tinyobj
            ${CMAKE_SOURCE_DIR}/third_party/imgui
            ${CMAKE_SOURCE_DIR}/third_party/miniaudio
)

target_link_libraries(VulkanEngine
    PUBLIC  Vulkan::Vulkan
            glfw
            glm::glm-header-only
    PRIVATE dl pthread m
)

if(APPLE)
    target_sources(VulkanEngine PRIVATE src/core/macos_fullscreen.mm)
    target_link_libraries(VulkanEngine PRIVATE "-framework Cocoa")
endif()

target_compile_definitions(VulkanEngine
    PUBLIC  GLFW_INCLUDE_VULKAN
            GLM_FORCE_RADIANS
            GLM_FORCE_DEPTH_ZERO_TO_ONE
    PRIVATE SHADER_DIR="${CMAKE_BINARY_DIR}/shaders/"
            ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets/"
)
