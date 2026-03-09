#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <engine/core/log.hpp>
#include <engine/renderer/vk_buffer.hpp>
#include <engine/renderer/vk_device.hpp>
#include <engine/renderer/vk_texture.hpp>

#include <cstring>
#include <stdexcept>

namespace engine {

VulkanTexture::VulkanTexture(VulkanDevice& device, VkCommandPool command_pool,
                             const uint8_t* pixels, uint32_t width, uint32_t height)
    : device_(device) {
    create(command_pool, pixels, width, height);
}

VulkanTexture::VulkanTexture(VulkanDevice& device, VkCommandPool command_pool,
                             const std::string& filepath)
    : device_(device) {
    int width, height, channels;
    stbi_uc* pixels = stbi_load(filepath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load texture: " + filepath);
    }

    create(command_pool, pixels, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    stbi_image_free(pixels);

    logInfo("Texture loaded: {} ({}x{})", filepath, width, height);
}

VulkanTexture::~VulkanTexture() {
    vkDestroySampler(device_.getHandle(), sampler_, nullptr);
    vkDestroyImageView(device_.getHandle(), image_view_, nullptr);
    vkDestroyImage(device_.getHandle(), image_, nullptr);
    vkFreeMemory(device_.getHandle(), memory_, nullptr);
}

void VulkanTexture::create(VkCommandPool command_pool, const uint8_t* pixels,
                           uint32_t width, uint32_t height) {
    VkDeviceSize image_size = width * height * 4;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    vk_buffer::createBuffer(device_, image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            staging_buffer, staging_memory);

    void* data;
    vkMapMemory(device_.getHandle(), staging_memory, 0, image_size, 0, &data);
    std::memcpy(data, pixels, image_size);
    vkUnmapMemory(device_.getHandle(), staging_memory);

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent = {width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = VK_FORMAT_R8G8B8A8_SRGB;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device_.getHandle(), &image_info, nullptr, &image_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture image");
    }

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(device_.getHandle(), image_, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = vk_buffer::findMemoryType(
        device_.getPhysicalDevice(), mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device_.getHandle(), &alloc_info, nullptr, &memory_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate texture image memory");
    }
    vkBindImageMemory(device_.getHandle(), image_, memory_, 0);

    vk_buffer::transitionImageLayout(device_.getHandle(), command_pool,
                                     device_.getGraphicsQueue(), image_,
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vk_buffer::copyBufferToImage(device_.getHandle(), command_pool, device_.getGraphicsQueue(),
                                 staging_buffer, image_, width, height);
    vk_buffer::transitionImageLayout(device_.getHandle(), command_pool,
                                     device_.getGraphicsQueue(), image_,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device_.getHandle(), staging_buffer, nullptr);
    vkFreeMemory(device_.getHandle(), staging_memory, nullptr);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image_;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device_.getHandle(), &view_info, nullptr, &image_view_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture image view");
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device_.getPhysicalDevice(), &props);

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = props.limits.maxSamplerAnisotropy;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device_.getHandle(), &sampler_info, nullptr, &sampler_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture sampler");
    }
}

}  // namespace engine
