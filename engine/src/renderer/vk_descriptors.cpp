#include <engine/renderer/vk_buffer.hpp>
#include <engine/renderer/vk_descriptors.hpp>
#include <engine/renderer/vk_device.hpp>

#include <cstring>
#include <stdexcept>

namespace engine {

VulkanDescriptors::VulkanDescriptors(VulkanDevice& device, uint32_t frames_in_flight)
    : device_(device) {
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
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = 1;
    info.pBindings = &binding;

    if (vkCreateDescriptorSetLayout(device_.getHandle(), &info, nullptr, &layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

void VulkanDescriptors::createPool(uint32_t frames_in_flight) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount = frames_in_flight;

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = 1;
    info.pPoolSizes = &pool_size;
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

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = sets_[i];
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &buffer_info;

        vkUpdateDescriptorSets(device_.getHandle(), 1, &write, 0, nullptr);
    }
}

}  // namespace engine
