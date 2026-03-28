#pragma once

#include <engine/core/audio.hpp>
#include <engine/core/camera.hpp>
#include <engine/core/spatial_audio.hpp>
#include <engine/core/window.hpp>
#include <engine/renderer/render_graph.hpp>
#include <engine/renderer/vk_device.hpp>
#include <engine/renderer/vk_instance.hpp>
#include <engine/renderer/vk_descriptors.hpp>
#include <engine/renderer/mesh.hpp>
#include <engine/renderer/scene.hpp>
#include <engine/renderer/vk_pipeline.hpp>
#include <engine/renderer/vk_swapchain.hpp>
#include <engine/renderer/vk_texture.hpp>

#include <glm/glm.hpp>
#include <glm/vec2.hpp>

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
    void buildRenderGraph();
    void createBloomDescriptors();
    void cleanupBloomDescriptors();
    void createPipelines();
    void initImGui();
    void shutdownImGui();
    void loadScene(int index);
    void createCommandBuffers();
    void createSyncObjects();
    void updateUBO();
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t image_index);
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
    std::unique_ptr<Scene> scene_;
    int current_scene_index_ = 0;
    std::vector<VkCommandBuffer> command_buffers_;

    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence> in_flight_fences_;
    uint32_t current_frame_ = 0;

    SpatialAudio spatial_audio_;
    Audio audio_;
    Camera camera_;
    float last_frame_time_ = 0.0f;
    glm::vec3 light_pos_{5.0f, 5.0f, 5.0f};
    glm::vec3 light_color_{1.0f, 1.0f, 1.0f};
    glm::vec3 fog_color_{0.02f, 0.02f, 0.03f};
    float fog_density_ = 0.15f;
    glm::vec3 light_dir_{0, 0, -1};
    float light_cone_angle_ = 0.0f;

    // Shadow map
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048;
    VkImage shadow_image_ = VK_NULL_HANDLE;
    VkDeviceMemory shadow_image_memory_ = VK_NULL_HANDLE;
    VkImageView shadow_image_view_ = VK_NULL_HANDLE;
    VkSampler shadow_sampler_ = VK_NULL_HANDLE;
    std::unique_ptr<VulkanPipeline> shadow_pipeline_;
    std::unique_ptr<VulkanPipeline> volumetric_pipeline_;
    std::unique_ptr<VulkanPipeline> bloom_extract_pipeline_;
    std::unique_ptr<VulkanPipeline> bloom_blur_pipeline_;
    std::unique_ptr<VulkanPipeline> bloom_composite_pipeline_;

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

    // Render graph
    static constexpr uint32_t PIXEL_SCALE = 8;
    std::unique_ptr<RenderGraph> render_graph_;
    ResourceId shadow_map_id_{};
    ResourceId swapchain_id_{};
    ResourceId pixel_color_id_{};
    ResourceId pixel_depth_id_{};
    ResourceId bloom_extract_id_{};
    ResourceId bloom_blur_h_id_{};
    ResourceId bloom_blurred_id_{};

    // Raw pointer aliases for pass bindings (point into unique_ptrs, updated by createPipelines)
    VulkanPipeline* pipeline_ptr_ = nullptr;
    VulkanPipeline* shadow_pipeline_ptr_ = nullptr;
    VulkanPipeline* skybox_pipeline_ptr_ = nullptr;
    VulkanPipeline* volumetric_pipeline_ptr_ = nullptr;
    VulkanPipeline* bloom_extract_pipeline_ptr_ = nullptr;
    VulkanPipeline* bloom_blur_pipeline_ptr_ = nullptr;
    VulkanPipeline* bloom_composite_pipeline_ptr_ = nullptr;
    VulkanDescriptors* descriptors_ptr_ = nullptr;
    Scene* scene_ptr_ = nullptr;

    // Bloom
    VkDescriptorSetLayout bloom_desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout bloom_composite_desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool bloom_desc_pool_ = VK_NULL_HANDLE;
    VkSampler bloom_sampler_ = VK_NULL_HANDLE;
    VkSampler bloom_nearest_sampler_ = VK_NULL_HANDLE;
    VkDescriptorSet bloom_extract_set_ = VK_NULL_HANDLE;
    VkDescriptorSet bloom_blur_h_set_ = VK_NULL_HANDLE;
    VkDescriptorSet bloom_blur_v_set_ = VK_NULL_HANDLE;
    VkDescriptorSet bloom_composite_set_ = VK_NULL_HANDLE;
    float bloom_threshold_ = 1.0f;
    float bloom_intensity_ = 0.0f;
    glm::vec2 bloom_blur_h_dir_{};
    glm::vec2 bloom_blur_v_dir_{};

    // Debug ray visualization
    std::unique_ptr<VulkanPipeline> debug_line_pipeline_;
    VulkanPipeline* debug_line_pipeline_ptr_ = nullptr;
    VkBuffer debug_line_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory debug_line_buffer_memory_ = VK_NULL_HANDLE;
    uint32_t debug_line_vertex_count_ = 0;
    bool show_debug_rays_ = false;
    int debug_ray_count_ = 5;
    int debug_ray_bounces_ = 4;
    void updateDebugRays();

    // ImGui
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;
};

}  // namespace engine
