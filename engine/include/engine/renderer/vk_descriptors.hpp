#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <vector>

namespace engine {

class VulkanDevice;

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

class VulkanDescriptors {
public:
    VulkanDescriptors(VulkanDevice& device, uint32_t frames_in_flight,
                      VkImageView texture_view, VkSampler texture_sampler);
    ~VulkanDescriptors();

    VulkanDescriptors(const VulkanDescriptors&) = delete;
    VulkanDescriptors& operator=(const VulkanDescriptors&) = delete;

    VkDescriptorSetLayout getLayout() const { return layout_; }
    VkDescriptorSet getSet(uint32_t frame_index) const { return sets_[frame_index]; }

    void updateUniformBuffer(uint32_t frame_index, const UniformBufferObject& ubo);

private:
    void createLayout();
    void createPool(uint32_t frames_in_flight);
    void createUniformBuffers(uint32_t frames_in_flight);
    void createSets(uint32_t frames_in_flight);

    VulkanDevice& device_;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;

    std::vector<VkBuffer> uniform_buffers_;
    std::vector<VkDeviceMemory> uniform_buffers_memory_;
    std::vector<void*> uniform_buffers_mapped_;

    VkImageView texture_view_ = VK_NULL_HANDLE;
    VkSampler texture_sampler_ = VK_NULL_HANDLE;
};

}  // namespace engine
