#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace engine {

class VulkanDevice;

class VulkanSwapchain {
public:
    VulkanSwapchain(VulkanDevice& device, VkSurfaceKHR surface, VkExtent2D window_extent);
    ~VulkanSwapchain();

    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;

    VkSwapchainKHR getHandle() const { return swapchain_; }
    VkRenderPass getRenderPass() const { return render_pass_; }
    VkFramebuffer getFramebuffer(uint32_t index) const { return framebuffers_[index]; }
    VkExtent2D getExtent() const { return extent_; }
    VkFormat getImageFormat() const { return image_format_; }
    uint32_t getImageCount() const { return static_cast<uint32_t>(images_.size()); }

    void recreate(VkExtent2D new_extent);

private:
    void create(VkExtent2D window_extent);
    void createImageViews();
    void createDepthResources();
    void createRenderPass();
    void createFramebuffers();
    void cleanupDepthResources();
    void cleanup();

    VkFormat findDepthFormat();
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling,
                                VkFormatFeatureFlags features);
    VkSurfaceFormatKHR chooseSurfaceFormat();
    VkPresentModeKHR choosePresentMode();
    VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                            VkExtent2D window_extent);

    VulkanDevice& device_;
    VkSurfaceKHR surface_;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat image_format_;
    VkFormat depth_format_;
    VkExtent2D extent_;

    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;
    VkImage depth_image_ = VK_NULL_HANDLE;
    VkDeviceMemory depth_image_memory_ = VK_NULL_HANDLE;
    VkImageView depth_image_view_ = VK_NULL_HANDLE;
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;
};

}  // namespace engine
