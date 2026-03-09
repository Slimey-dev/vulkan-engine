#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

namespace engine {

class VulkanDevice;

class VulkanTexture {
public:
    VulkanTexture(VulkanDevice& device, VkCommandPool command_pool,
                  const uint8_t* pixels, uint32_t width, uint32_t height);
    VulkanTexture(VulkanDevice& device, VkCommandPool command_pool,
                  const std::string& filepath);
    ~VulkanTexture();

    VulkanTexture(const VulkanTexture&) = delete;
    VulkanTexture& operator=(const VulkanTexture&) = delete;

    VkImageView getImageView() const { return image_view_; }
    VkSampler getSampler() const { return sampler_; }

private:
    void create(VkCommandPool command_pool, const uint8_t* pixels,
                uint32_t width, uint32_t height);

    VulkanDevice& device_;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView image_view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

}  // namespace engine
