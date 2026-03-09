#include <engine/core/log.hpp>
#include <engine/renderer/vk_device.hpp>
#include <engine/renderer/vk_swapchain.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace engine {

VulkanSwapchain::VulkanSwapchain(VulkanDevice& device, VkSurfaceKHR surface,
                                 VkExtent2D window_extent)
    : device_(device), surface_(surface) {
    create(window_extent);
    createImageViews();
    createRenderPass();
    createFramebuffers();
}

VulkanSwapchain::~VulkanSwapchain() {
    cleanup();
    vkDestroyRenderPass(device_.getHandle(), render_pass_, nullptr);
}

void VulkanSwapchain::recreate(VkExtent2D new_extent) {
    vkDeviceWaitIdle(device_.getHandle());

    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_.getHandle(), fb, nullptr);
    for (auto iv : image_views_) vkDestroyImageView(device_.getHandle(), iv, nullptr);
    vkDestroySwapchainKHR(device_.getHandle(), swapchain_, nullptr);

    framebuffers_.clear();
    image_views_.clear();
    images_.clear();

    create(new_extent);
    createImageViews();
    createFramebuffers();

    logInfo("Swapchain recreated: {}x{}", extent_.width, extent_.height);
}

void VulkanSwapchain::create(VkExtent2D window_extent) {
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_.getPhysicalDevice(), surface_, &capabilities);

    auto format = chooseSurfaceFormat();
    auto mode = choosePresentMode();
    auto extent = chooseExtent(capabilities, window_extent);

    uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = surface_;
    info.minImageCount = image_count;
    info.imageFormat = format.format;
    info.imageColorSpace = format.colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto indices = device_.getQueueFamilies();
    uint32_t family_indices[] = {indices.graphics.value(), indices.present.value()};
    if (indices.graphics != indices.present) {
        info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = 2;
        info.pQueueFamilyIndices = family_indices;
    } else {
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    info.preTransform = capabilities.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = mode;
    info.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device_.getHandle(), &info, nullptr, &swapchain_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swapchain");
    }

    vkGetSwapchainImagesKHR(device_.getHandle(), swapchain_, &image_count, nullptr);
    images_.resize(image_count);
    vkGetSwapchainImagesKHR(device_.getHandle(), swapchain_, &image_count, images_.data());

    image_format_ = format.format;
    extent_ = extent;

    logInfo("Swapchain created: {}x{}, {} images", extent_.width, extent_.height, image_count);
}

void VulkanSwapchain::createImageViews() {
    image_views_.resize(images_.size());
    for (size_t i = 0; i < images_.size(); i++) {
        VkImageViewCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image = images_[i];
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = image_format_;
        info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.baseMipLevel = 0;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.baseArrayLayer = 0;
        info.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device_.getHandle(), &info, nullptr, &image_views_[i]) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view");
        }
    }
}

void VulkanSwapchain::createRenderPass() {
    VkAttachmentDescription color_attachment{};
    color_attachment.format = image_format_;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &color_attachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dependency;

    if (vkCreateRenderPass(device_.getHandle(), &info, nullptr, &render_pass_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass");
    }
}

void VulkanSwapchain::createFramebuffers() {
    framebuffers_.resize(image_views_.size());
    for (size_t i = 0; i < image_views_.size(); i++) {
        VkFramebufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass = render_pass_;
        info.attachmentCount = 1;
        info.pAttachments = &image_views_[i];
        info.width = extent_.width;
        info.height = extent_.height;
        info.layers = 1;

        if (vkCreateFramebuffer(device_.getHandle(), &info, nullptr, &framebuffers_[i]) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create framebuffer");
        }
    }
}

void VulkanSwapchain::cleanup() {
    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_.getHandle(), fb, nullptr);
    for (auto iv : image_views_) vkDestroyImageView(device_.getHandle(), iv, nullptr);
    vkDestroySwapchainKHR(device_.getHandle(), swapchain_, nullptr);
}

VkSurfaceFormatKHR VulkanSwapchain::chooseSurfaceFormat() {
    uint32_t count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device_.getPhysicalDevice(), surface_, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device_.getPhysicalDevice(), surface_, &count,
                                         formats.data());

    for (const auto& fmt : formats) {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
            fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return fmt;
        }
    }
    return formats[0];
}

VkPresentModeKHR VulkanSwapchain::choosePresentMode() {
    uint32_t count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device_.getPhysicalDevice(), surface_, &count,
                                              nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device_.getPhysicalDevice(), surface_, &count,
                                              modes.data());

    for (auto mode : modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanSwapchain::chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                         VkExtent2D window_extent) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    VkExtent2D extent = window_extent;
    extent.width = std::clamp(extent.width, capabilities.minImageExtent.width,
                              capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height);
    return extent;
}

}  // namespace engine
