#include <engine/core/log.hpp>
#include <engine/renderer/renderer.hpp>

#include <stdexcept>

namespace engine {

Renderer::Renderer(Window& window) : window_(window) {
    instance_ = std::make_unique<VulkanInstance>();
    createSurface();
    device_ = std::make_unique<VulkanDevice>(instance_->getHandle(), surface_);
    swapchain_ = std::make_unique<VulkanSwapchain>(*device_, surface_, window_.getExtent());
    pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), swapchain_->getRenderPass(), swapchain_->getExtent(),
        std::string(SHADER_DIR) + "triangle.vert.spv",
        std::string(SHADER_DIR) + "triangle.frag.spv");

    createCommandPool();
    createCommandBuffers();
    createSyncObjects();

    logInfo("Renderer initialized");
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(device_->getHandle());

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device_->getHandle(), image_available_semaphores_[i], nullptr);
        vkDestroySemaphore(device_->getHandle(), render_finished_semaphores_[i], nullptr);
        vkDestroyFence(device_->getHandle(), in_flight_fences_[i], nullptr);
    }

    vkDestroyCommandPool(device_->getHandle(), command_pool_, nullptr);

    pipeline_.reset();
    swapchain_.reset();
    device_.reset();

    vkDestroySurfaceKHR(instance_->getHandle(), surface_, nullptr);
    instance_.reset();
}

void Renderer::drawFrame() {
    vkWaitForFences(device_->getHandle(), 1, &in_flight_fences_[current_frame_], VK_TRUE,
                    UINT64_MAX);

    uint32_t image_index;
    VkResult result = vkAcquireNextImageKHR(device_->getHandle(), swapchain_->getHandle(),
                                            UINT64_MAX,
                                            image_available_semaphores_[current_frame_],
                                            VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    vkResetFences(device_->getHandle(), 1, &in_flight_fences_[current_frame_]);
    vkResetCommandBuffer(command_buffers_[current_frame_], 0);
    recordCommandBuffer(command_buffers_[current_frame_], image_index);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore wait_semaphores[] = {image_available_semaphores_[current_frame_]};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffers_[current_frame_];

    VkSemaphore signal_semaphores[] = {render_finished_semaphores_[current_frame_]};
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    if (vkQueueSubmit(device_->getGraphicsQueue(), 1, &submit_info,
                      in_flight_fences_[current_frame_]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer");
    }

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;

    VkSwapchainKHR swapchains[] = {swapchain_->getHandle()};
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapchains;
    present_info.pImageIndices = &image_index;

    result = vkQueuePresentKHR(device_->getPresentQueue(), &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
        window_.wasResized()) {
        window_.resetResizedFlag();
        recreateSwapchain();
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swapchain image");
    }

    current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Renderer::createSurface() {
    if (glfwCreateWindowSurface(instance_->getHandle(), window_.getHandle(), nullptr, &surface_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
}

void Renderer::createCommandPool() {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = device_->getQueueFamilies().graphics.value();

    if (vkCreateCommandPool(device_->getHandle(), &info, nullptr, &command_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

void Renderer::createCommandBuffers() {
    command_buffers_.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = command_pool_;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = static_cast<uint32_t>(command_buffers_.size());

    if (vkAllocateCommandBuffers(device_->getHandle(), &info, command_buffers_.data()) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }
}

void Renderer::createSyncObjects() {
    image_available_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    render_finished_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device_->getHandle(), &sem_info, nullptr,
                              &image_available_semaphores_[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device_->getHandle(), &sem_info, nullptr,
                              &render_finished_semaphores_[i]) != VK_SUCCESS ||
            vkCreateFence(device_->getHandle(), &fence_info, nullptr, &in_flight_fences_[i]) !=
                VK_SUCCESS) {
            throw std::runtime_error("Failed to create sync objects");
        }
    }
}

void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t image_index) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer");
    }

    VkRenderPassBeginInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_info.renderPass = swapchain_->getRenderPass();
    rp_info.framebuffer = swapchain_->getFramebuffer(image_index);
    rp_info.renderArea.offset = {0, 0};
    rp_info.renderArea.extent = swapchain_->getExtent();

    VkClearValue clear_color = {{{0.01f, 0.01f, 0.02f, 1.0f}}};
    rp_info.clearValueCount = 1;
    rp_info.pClearValues = &clear_color;

    vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getHandle());

    VkViewport viewport{};
    viewport.width = static_cast<float>(swapchain_->getExtent().width);
    viewport.height = static_cast<float>(swapchain_->getExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = swapchain_->getExtent();
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer");
    }
}

void Renderer::recreateSwapchain() {
    VkExtent2D extent = window_.getExtent();
    while (extent.width == 0 || extent.height == 0) {
        extent = window_.getExtent();
        glfwWaitEvents();
    }

    swapchain_->recreate(extent);
}

}  // namespace engine
