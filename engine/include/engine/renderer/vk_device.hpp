#pragma once

#include <vulkan/vulkan.h>

#include <optional>

namespace engine {

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;

    bool isComplete() const { return graphics.has_value() && present.has_value(); }
};

class VulkanDevice {
public:
    VulkanDevice(VkInstance instance, VkSurfaceKHR surface);
    ~VulkanDevice();

    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

    VkDevice getHandle() const { return device_; }
    VkPhysicalDevice getPhysicalDevice() const { return physical_device_; }
    VkQueue getGraphicsQueue() const { return graphics_queue_; }
    VkQueue getPresentQueue() const { return present_queue_; }
    QueueFamilyIndices getQueueFamilies() const { return queue_families_; }

private:
    void pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);
    void createLogicalDevice();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);
    bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface);

    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue present_queue_ = VK_NULL_HANDLE;
    QueueFamilyIndices queue_families_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
};

}  // namespace engine
