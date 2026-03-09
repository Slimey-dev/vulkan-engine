#include <engine/renderer/vk_buffer.hpp>
#include <engine/renderer/vk_descriptors.hpp>
#include <engine/renderer/vk_device.hpp>

#include <array>
#include <cstring>
#include <stdexcept>

namespace engine {

VulkanDescriptors::VulkanDescriptors(VulkanDevice& device, uint32_t frames_in_flight,
                                     VkImageView texture_view, VkSampler texture_sampler)
    : device_(device), texture_view_(texture_view), texture_sampler_(texture_sampler) {
    createLayout();
    createPool(frames_in_flight);
    createUniformBuffers(frames_in_flight);
    createSets(frames_in_flight);
}

VulkanDescriptors::~VulkanDescriptors() {
    for (size_t i = 0; i < uniform_buffers_.size(); i++) {
        vkUnmapMemory(device_.getHandle(), uniform_buffers_memory_[i]);
        vkDestroyBuffer(device_.getHandle(), uniform_buffers_[i], nullptr);
        vkFreeMemory(device_.getHandle(), uniform_buffers_memory_[i], nullptr);
    }
    vkDestroyDescriptorPool(device_.getHandle(), pool_, nullptr);
    vkDestroyDescriptorSetLayout(device_.getHandle(), layout_, nullptr);
}

void VulkanDescriptors::updateUniformBuffer(uint32_t frame_index,
                                            const UniformBufferObject& ubo) {
    std::memcpy(uniform_buffers_mapped_[frame_index], &ubo, sizeof(ubo));
}

void VulkanDescriptors::createLayout() {
    VkDescriptorSetLayoutBinding ubo_binding{};
    ubo_binding.binding = 0;
    ubo_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo_binding.descriptorCount = 1;
    ubo_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding sampler_binding{};
    sampler_binding.binding = 1;
    sampler_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sampler_binding.descriptorCount = 1;
    sampler_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {ubo_binding, sampler_binding};

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = static_cast<uint32_t>(bindings.size());
    info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device_.getHandle(), &info, nullptr, &layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

void VulkanDescriptors::createPool(uint32_t frames_in_flight) {
    std::array<VkDescriptorPoolSize, 2> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = frames_in_flight;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = frames_in_flight;

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    info.pPoolSizes = pool_sizes.data();
    info.maxSets = frames_in_flight;

    if (vkCreateDescriptorPool(device_.getHandle(), &info, nullptr, &pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

void VulkanDescriptors::createUniformBuffers(uint32_t frames_in_flight) {
    VkDeviceSize size = sizeof(UniformBufferObject);

    uniform_buffers_.resize(frames_in_flight);
    uniform_buffers_memory_.resize(frames_in_flight);
    uniform_buffers_mapped_.resize(frames_in_flight);

    for (uint32_t i = 0; i < frames_in_flight; i++) {
        vk_buffer::createBuffer(device_, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                uniform_buffers_[i], uniform_buffers_memory_[i]);

        vkMapMemory(device_.getHandle(), uniform_buffers_memory_[i], 0, size, 0,
                    &uniform_buffers_mapped_[i]);
    }
}

void VulkanDescriptors::createSets(uint32_t frames_in_flight) {
    std::vector<VkDescriptorSetLayout> layouts(frames_in_flight, layout_);

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool_;
    alloc_info.descriptorSetCount = frames_in_flight;
    alloc_info.pSetLayouts = layouts.data();

    sets_.resize(frames_in_flight);
    if (vkAllocateDescriptorSets(device_.getHandle(), &alloc_info, sets_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets");
    }

    for (uint32_t i = 0; i < frames_in_flight; i++) {
        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = uniform_buffers_[i];
        buffer_info.offset = 0;
        buffer_info.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo image_info{};
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_info.imageView = texture_view_;
        image_info.sampler = texture_sampler_;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = sets_[i];
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &buffer_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = sets_[i];
        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &image_info;

        vkUpdateDescriptorSets(device_.getHandle(), static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

}  // namespace engine
