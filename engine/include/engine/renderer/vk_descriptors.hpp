#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <vector>

namespace engine {

class VulkanDevice;

struct UniformBufferObject {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 light_space;
    glm::vec4 light_pos;
    glm::vec4 view_pos;
    glm::vec4 light_color;
    glm::vec4 fog_color;
    glm::vec4 fog_params;   // x = density
    glm::vec4 light_dir;    // xyz = spotlight dir, w = cos(outer cutoff); w==0 → point light
};

struct PushConstants {
    glm::mat4 model;
};

class VulkanDescriptors {
public:
    VulkanDescriptors(VulkanDevice& device, uint32_t frames_in_flight,
                      VkImageView texture_view, VkSampler texture_sampler,
                      VkImageView shadow_view, VkSampler shadow_sampler,
                      VkImageView cubemap_view, VkSampler cubemap_sampler);
    ~VulkanDescriptors();

    VulkanDescriptors(const VulkanDescriptors&) = delete;
    VulkanDescriptors& operator=(const VulkanDescriptors&) = delete;

    VkDescriptorSetLayout getLayout() const { return layout_; }
    VkDescriptorSet getSet(uint32_t frame_index) const { return sets_[frame_index]; }

    VkDescriptorSetLayout getSkyboxLayout() const { return skybox_layout_; }
    VkDescriptorSet getSkyboxSet(uint32_t frame_index) const { return skybox_sets_[frame_index]; }

    void updateUniformBuffer(uint32_t frame_index, const UniformBufferObject& ubo);

private:
    void createLayout();
    void createSkyboxLayout();
    void createPool(uint32_t frames_in_flight);
    void createUniformBuffers(uint32_t frames_in_flight);
    void createSets(uint32_t frames_in_flight);
    void createSkyboxSets(uint32_t frames_in_flight);

    VulkanDevice& device_;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout skybox_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;
    std::vector<VkDescriptorSet> skybox_sets_;

    std::vector<VkBuffer> uniform_buffers_;
    std::vector<VkDeviceMemory> uniform_buffers_memory_;
    std::vector<void*> uniform_buffers_mapped_;

    VkImageView texture_view_ = VK_NULL_HANDLE;
    VkSampler texture_sampler_ = VK_NULL_HANDLE;
    VkImageView shadow_view_ = VK_NULL_HANDLE;
    VkSampler shadow_sampler_ = VK_NULL_HANDLE;
    VkImageView cubemap_view_ = VK_NULL_HANDLE;
    VkSampler cubemap_sampler_ = VK_NULL_HANDLE;
};

}  // namespace engine
