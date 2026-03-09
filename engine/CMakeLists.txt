add_library(VulkanEngine STATIC
    src/core/window.cpp
    src/renderer/vk_instance.cpp
    src/renderer/vk_device.cpp
    src/renderer/vk_swapchain.cpp
    src/renderer/vk_pipeline.cpp
    src/renderer/vk_buffer.cpp
    src/renderer/vk_descriptors.cpp
    src/renderer/renderer.cpp
)

target_include_directories(VulkanEngine
    PUBLIC  include
    PRIVATE src
)

target_link_libraries(VulkanEngine
    PUBLIC  Vulkan::Vulkan
            glfw
            glm::glm-header-only
)

target_compile_definitions(VulkanEngine
    PUBLIC  GLFW_INCLUDE_VULKAN
            GLM_FORCE_RADIANS
            GLM_FORCE_DEPTH_ZERO_TO_ONE
    PRIVATE SHADER_DIR="${CMAKE_BINARY_DIR}/shaders/"
)
