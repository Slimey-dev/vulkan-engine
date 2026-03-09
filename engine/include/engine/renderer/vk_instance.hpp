#pragma once

#include <vulkan/vulkan.h>

namespace engine {

class VulkanInstance {
public:
    VulkanInstance();
    ~VulkanInstance();

    VulkanInstance(const VulkanInstance&) = delete;
    VulkanInstance& operator=(const VulkanInstance&) = delete;

    VkInstance getHandle() const { return instance_; }

private:
    bool checkValidationLayerSupport();
    void setupDebugMessenger();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool validation_enabled_ = false;
};

}  // namespace engine
