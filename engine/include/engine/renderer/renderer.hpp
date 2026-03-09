#pragma once

#include <engine/core/camera.hpp>
#include <engine/core/window.hpp>
#include <engine/renderer/vk_device.hpp>
#include <engine/renderer/vk_instance.hpp>
#include <engine/renderer/vk_descriptors.hpp>
#include <engine/renderer/mesh.hpp>
#include <engine/renderer/vk_pipeline.hpp>
#include <engine/renderer/vk_swapchain.hpp>
#include <engine/renderer/vk_texture.hpp>

#include <glm/glm.hpp>

#include <memory>
#include <vector>

namespace engine {

class Renderer {
public:
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    Renderer(Window& window);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void drawFrame();

private:
    void createSurface();
    void createCommandPool();
    void createTexture();
    void createShadowResources();
    void createSkyboxCubemap();
    void createSkyboxMesh();
    void createPixelResources();
    void cleanupPixelResources();
    void initImGui();
    void shutdownImGui();
    void loadMesh();
    void createCommandBuffers();
    void createSyncObjects();
    void updateUBO();
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t image_index);
    void recordShadowPass(VkCommandBuffer cmd);
    void recreateSwapchain();

    Window& window_;

    std::unique_ptr<VulkanInstance> instance_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    std::unique_ptr<VulkanDevice> device_;
    std::unique_ptr<VulkanSwapchain> swapchain_;
    std::unique_ptr<VulkanTexture> texture_;
    std::unique_ptr<VulkanDescriptors> descriptors_;
    std::unique_ptr<VulkanPipeline> pipeline_;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<Mesh> ground_mesh_;
    std::vector<VkCommandBuffer> command_buffers_;

    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence> in_flight_fences_;
    uint32_t current_frame_ = 0;

    Camera camera_;
    float last_frame_time_ = 0.0f;
    glm::mat4 cube_model_{1.0f};
    glm::vec3 light_pos_{5.0f, 5.0f, 5.0f};
    glm::vec3 light_color_{1.0f, 1.0f, 1.0f};

    // Shadow map
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048;
    VkImage shadow_image_ = VK_NULL_HANDLE;
    VkDeviceMemory shadow_image_memory_ = VK_NULL_HANDLE;
    VkImageView shadow_image_view_ = VK_NULL_HANDLE;
    VkSampler shadow_sampler_ = VK_NULL_HANDLE;
    VkRenderPass shadow_render_pass_ = VK_NULL_HANDLE;
    VkFramebuffer shadow_framebuffer_ = VK_NULL_HANDLE;
    std::unique_ptr<VulkanPipeline> shadow_pipeline_;

    // Skybox
    VkImage skybox_image_ = VK_NULL_HANDLE;
    VkDeviceMemory skybox_image_memory_ = VK_NULL_HANDLE;
    VkImageView skybox_image_view_ = VK_NULL_HANDLE;
    VkSampler skybox_sampler_ = VK_NULL_HANDLE;
    std::unique_ptr<VulkanPipeline> skybox_pipeline_;
    VkBuffer skybox_vertex_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory skybox_vertex_memory_ = VK_NULL_HANDLE;
    VkBuffer skybox_index_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory skybox_index_memory_ = VK_NULL_HANDLE;
    uint32_t skybox_index_count_ = 0;

    // Pixelation (offscreen render target)
    static constexpr uint32_t PIXEL_SCALE = 8;
    VkImage pixel_color_image_ = VK_NULL_HANDLE;
    VkDeviceMemory pixel_color_memory_ = VK_NULL_HANDLE;
    VkImageView pixel_color_view_ = VK_NULL_HANDLE;
    VkImage pixel_depth_image_ = VK_NULL_HANDLE;
    VkDeviceMemory pixel_depth_memory_ = VK_NULL_HANDLE;
    VkImageView pixel_depth_view_ = VK_NULL_HANDLE;
    VkRenderPass pixel_render_pass_ = VK_NULL_HANDLE;
    VkFramebuffer pixel_framebuffer_ = VK_NULL_HANDLE;
    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;

    // UI overlay (full-res render pass over blitted swapchain)
    VkRenderPass ui_render_pass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> ui_framebuffers_;

    // ImGui
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;
};

}  // namespace engine
