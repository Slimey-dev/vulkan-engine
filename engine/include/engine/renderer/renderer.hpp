#pragma once

#include <engine/core/window.hpp>
#include <engine/renderer/vk_device.hpp>
#include <engine/renderer/vk_instance.hpp>
#include <engine/renderer/vk_descriptors.hpp>
#include <engine/renderer/vk_pipeline.hpp>
#include <engine/renderer/vk_swapchain.hpp>

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
    void createVertexBuffer();
    void createIndexBuffer();
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
    std::unique_ptr<VulkanDescriptors> descriptors_;
    std::unique_ptr<VulkanPipeline> pipeline_;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkBuffer vertex_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertex_buffer_memory_ = VK_NULL_HANDLE;
    VkBuffer index_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory index_buffer_memory_ = VK_NULL_HANDLE;
    uint32_t index_count_ = 0;
    std::vector<VkCommandBuffer> command_buffers_;

    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence> in_flight_fences_;
    uint32_t current_frame_ = 0;
};

}  // namespace engine
