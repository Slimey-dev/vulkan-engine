#include <engine/core/log.hpp>
#include <engine/renderer/vk_device.hpp>

#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {

static const std::vector<const char*> kDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
static constexpr const char* kPortabilitySubsetExt = "VK_KHR_portability_subset";

VulkanDevice::VulkanDevice(VkInstance instance, VkSurfaceKHR surface) : surface_(surface) {
    pickPhysicalDevice(instance, surface);
    createLogicalDevice();
}

VulkanDevice::~VulkanDevice() {
    vkDestroyDevice(device_, nullptr);
}

void VulkanDevice::pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    // Prefer discrete GPU
    for (const auto& dev : devices) {
        if (!isDeviceSuitable(dev, surface)) continue;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = dev;
            logInfo("Selected GPU: {}", props.deviceName);
            break;
        }
    }

    // Fall back to any suitable
    if (physical_device_ == VK_NULL_HANDLE) {
        for (const auto& dev : devices) {
            if (isDeviceSuitable(dev, surface)) {
                physical_device_ = dev;
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(dev, &props);
                logInfo("Selected GPU (fallback): {}", props.deviceName);
                break;
            }
        }
    }

    if (physical_device_ == VK_NULL_HANDLE) {
        throw std::runtime_error("No suitable GPU found");
    }

    queue_families_ = findQueueFamilies(physical_device_, surface);
}

void VulkanDevice::createLogicalDevice() {
    std::set<uint32_t> unique_families = {queue_families_.graphics.value(),
                                          queue_families_.present.value()};

    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    float priority = 1.0f;
    for (uint32_t family : unique_families) {
        VkDeviceQueueCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        info.queueFamilyIndex = family;
        info.queueCount = 1;
        info.pQueuePriorities = &priority;
        queue_infos.push_back(info);
    }

    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy = VK_TRUE;

    std::vector<const char*> device_extensions(kDeviceExtensions.begin(), kDeviceExtensions.end());
    if (portability_subset_) {
        device_extensions.push_back(kPortabilitySubsetExt);
    }

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
    create_info.pQueueCreateInfos = queue_infos.data();
    create_info.pEnabledFeatures = &features;
    create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();

    if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device");
    }

    vkGetDeviceQueue(device_, queue_families_.graphics.value(), 0, &graphics_queue_);
    vkGetDeviceQueue(device_, queue_families_.present.value(), 0, &present_queue_);

    logInfo("Logical device created");
}

QueueFamilyIndices VulkanDevice::findQueueFamilies(VkPhysicalDevice device,
                                                   VkSurfaceKHR surface) {
    QueueFamilyIndices indices;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphics = i;
        }
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present);
        if (present) {
            indices.present = i;
        }
        if (indices.isComplete()) break;
    }

    return indices;
}

bool VulkanDevice::isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
    auto indices = findQueueFamilies(device, surface);
    if (!indices.isComplete()) return false;

    // Check extension support
    uint32_t ext_count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> available(ext_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, available.data());

    std::set<std::string> required(kDeviceExtensions.begin(), kDeviceExtensions.end());
    bool has_portability = false;
    for (const auto& ext : available) {
        required.erase(ext.extensionName);
        if (std::strcmp(ext.extensionName, kPortabilitySubsetExt) == 0) {
            has_portability = true;
        }
    }
    if (!required.empty()) return false;
    portability_subset_ = has_portability;

    // Check swapchain support
    uint32_t format_count, mode_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, nullptr);

    return format_count > 0 && mode_count > 0;
}

}  // namespace engine
