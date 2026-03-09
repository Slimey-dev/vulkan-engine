#include <engine/core/log.hpp>
#include <engine/renderer/renderer.hpp>
#include <engine/renderer/vk_buffer.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace engine {

Renderer::Renderer(Window& window) : window_(window) {
    instance_ = std::make_unique<VulkanInstance>();
    createSurface();
    device_ = std::make_unique<VulkanDevice>(instance_->getHandle(), surface_);
    swapchain_ = std::make_unique<VulkanSwapchain>(*device_, surface_, window_.getExtent());
    createCommandPool();
    createPixelResources();
    createTexture();
    createShadowResources();
    createSkyboxCubemap();
    descriptors_ = std::make_unique<VulkanDescriptors>(
        *device_, MAX_FRAMES_IN_FLIGHT, texture_->getImageView(), texture_->getSampler(),
        shadow_image_view_, shadow_sampler_, skybox_image_view_, skybox_sampler_);

    auto binding = Vertex::getBindingDescription();
    auto attributes = Vertex::getAttributeDescriptions();

    PipelineConfig shadow_config{};
    shadow_config.depth_bias = true;
    shadow_config.depth_bias_constant = 1.25f;
    shadow_config.depth_bias_slope = 1.75f;
    shadow_config.has_color_attachment = false;

    shadow_pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), shadow_render_pass_,
        std::string(SHADER_DIR) + "shadow.vert.spv",
        std::string(SHADER_DIR) + "shadow.frag.spv",
        std::vector{binding},
        std::vector<VkVertexInputAttributeDescription>(attributes.begin(), attributes.end()),
        descriptors_->getLayout(), shadow_config);

    pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), pixel_render_pass_,
        std::string(SHADER_DIR) + "triangle.vert.spv",
        std::string(SHADER_DIR) + "triangle.frag.spv",
        std::vector{binding},
        std::vector<VkVertexInputAttributeDescription>(attributes.begin(), attributes.end()),
        descriptors_->getLayout());

    // Skybox pipeline (position-only vertex input)
    VkVertexInputBindingDescription skybox_binding{};
    skybox_binding.binding = 0;
    skybox_binding.stride = sizeof(glm::vec3);
    skybox_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription skybox_attr{};
    skybox_attr.binding = 0;
    skybox_attr.location = 0;
    skybox_attr.format = VK_FORMAT_R32G32B32_SFLOAT;
    skybox_attr.offset = 0;

    PipelineConfig skybox_config{};
    skybox_config.depth_write = false;
    skybox_config.depth_compare_op = VK_COMPARE_OP_LESS_OR_EQUAL;
    skybox_config.cull_mode = VK_CULL_MODE_FRONT_BIT;
    skybox_config.has_push_constants = false;

    skybox_pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), pixel_render_pass_,
        std::string(SHADER_DIR) + "skybox.vert.spv",
        std::string(SHADER_DIR) + "skybox.frag.spv",
        std::vector{skybox_binding}, std::vector{skybox_attr},
        descriptors_->getSkyboxLayout(), skybox_config);

    loadMesh();
    createSkyboxMesh();
    createCommandBuffers();
    createSyncObjects();
    initImGui();

    window_.setCursorCaptured(true);
    logInfo("Renderer initialized");
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(device_->getHandle());
    shutdownImGui();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device_->getHandle(), image_available_semaphores_[i], nullptr);
        vkDestroyFence(device_->getHandle(), in_flight_fences_[i], nullptr);
    }
    for (auto sem : render_finished_semaphores_) {
        vkDestroySemaphore(device_->getHandle(), sem, nullptr);
    }

    mesh_.reset();
    ground_mesh_.reset();
    vkDestroyCommandPool(device_->getHandle(), command_pool_, nullptr);

    pipeline_.reset();
    shadow_pipeline_.reset();
    skybox_pipeline_.reset();
    descriptors_.reset();
    texture_.reset();

    cleanupPixelResources();
    vkDestroyRenderPass(device_->getHandle(), pixel_render_pass_, nullptr);
    vkDestroyRenderPass(device_->getHandle(), ui_render_pass_, nullptr);

    vkDestroyFramebuffer(device_->getHandle(), shadow_framebuffer_, nullptr);
    vkDestroyRenderPass(device_->getHandle(), shadow_render_pass_, nullptr);
    vkDestroySampler(device_->getHandle(), shadow_sampler_, nullptr);
    vkDestroyImageView(device_->getHandle(), shadow_image_view_, nullptr);
    vkDestroyImage(device_->getHandle(), shadow_image_, nullptr);
    vkFreeMemory(device_->getHandle(), shadow_image_memory_, nullptr);

    vkDestroySampler(device_->getHandle(), skybox_sampler_, nullptr);
    vkDestroyImageView(device_->getHandle(), skybox_image_view_, nullptr);
    vkDestroyImage(device_->getHandle(), skybox_image_, nullptr);
    vkFreeMemory(device_->getHandle(), skybox_image_memory_, nullptr);
    vkDestroyBuffer(device_->getHandle(), skybox_index_buffer_, nullptr);
    vkFreeMemory(device_->getHandle(), skybox_index_memory_, nullptr);
    vkDestroyBuffer(device_->getHandle(), skybox_vertex_buffer_, nullptr);
    vkFreeMemory(device_->getHandle(), skybox_vertex_memory_, nullptr);

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
    updateUBO();
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

    VkSemaphore signal_semaphores[] = {render_finished_semaphores_[image_index]};
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

void Renderer::updateUBO() {
    static auto start = std::chrono::high_resolution_clock::now();
    float current_time =
        std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();
    float delta_time = current_time - last_frame_time_;
    last_frame_time_ = current_time;

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (window_.isKeyPressed(GLFW_KEY_ESCAPE)) {
        if (window_.isUIMode()) {
            window_.toggleUIMode();
        } else {
            window_.close();
        }
    }

    if (!window_.isUIMode()) {
        camera_.processKeyboard(window_, delta_time);
        float mouse_dx, mouse_dy;
        window_.getMouseDelta(mouse_dx, mouse_dy);
        camera_.processMouse(mouse_dx, mouse_dy);
        camera_.processScroll(window_.getScrollDelta());
    } else {
        float mouse_dx, mouse_dy;
        window_.getMouseDelta(mouse_dx, mouse_dy);
        window_.getScrollDelta();
    }

    if (camera_.didJump()) audio_.playJump();
    if (camera_.didLand()) audio_.playLand();

    // ImGui debug panel
    {
        ImGui::Begin("Engine Debug");
        ImGui::Text("FPS: %.1f (%.3f ms)", 1.0f / delta_time, delta_time * 1000.0f);

        glm::vec3 cam_pos = camera_.getPosition();
        ImGui::Text("Camera: (%.2f, %.2f, %.2f)", cam_pos.x, cam_pos.y, cam_pos.z);

        ImGui::DragFloat3("Light Position", &light_pos_.x, 0.1f, -20.0f, 20.0f);
        ImGui::ColorEdit3("Light Color", &light_color_.x);

        ImGui::Separator();
        ImGui::Text("Fog");
        ImGui::SliderFloat("Fog Density", &fog_density_, 0.0f, 1.0f, "%.3f");
        ImGui::ColorEdit3("Fog Color", &fog_color_.x);

        ImGui::Separator();
        ImGui::Text("Press [Tab] to toggle UI/FPS mode");
        ImGui::End();
    }

    cube_model_ = glm::rotate(glm::mat4(1.0f), current_time * glm::radians(90.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f));

    auto extent = swapchain_->getExtent();
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);

    glm::mat4 light_view =
        glm::lookAt(light_pos_, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 light_proj = glm::ortho(-12.0f, 12.0f, -12.0f, 12.0f, 0.1f, 20.0f);
    light_proj[1][1] *= -1;  // Vulkan Y-flip

    UniformBufferObject ubo{};
    ubo.view = camera_.getViewMatrix();
    ubo.proj = camera_.getProjectionMatrix(aspect);
    ubo.light_space = light_proj * light_view;
    ubo.light_pos = glm::vec4(light_pos_, 0.0f);
    ubo.view_pos = glm::vec4(camera_.getPosition(), 0.0f);
    ubo.light_color = glm::vec4(light_color_, 0.0f);
    ubo.fog_color = glm::vec4(fog_color_, 1.0f);
    ubo.fog_params = glm::vec4(fog_density_, 0.0f, 0.0f, 0.0f);

    descriptors_->updateUniformBuffer(current_frame_, ubo);
}

void Renderer::createTexture() {
    constexpr uint32_t size = 256;
    constexpr uint32_t cell = size / 8;
    std::vector<uint8_t> pixels(size * size * 4);

    for (uint32_t y = 0; y < size; y++) {
        for (uint32_t x = 0; x < size; x++) {
            bool white = ((x / cell) + (y / cell)) % 2 == 0;
            uint32_t idx = (y * size + x) * 4;
            pixels[idx + 0] = white ? 255 : 40;
            pixels[idx + 1] = white ? 255 : 40;
            pixels[idx + 2] = white ? 255 : 40;
            pixels[idx + 3] = 255;
        }
    }

    texture_ = std::make_unique<VulkanTexture>(*device_, command_pool_, pixels.data(), size, size);
    logInfo("Checkerboard texture generated ({}x{})", size, size);
}

void Renderer::createShadowResources() {
    VkFormat depth_format = swapchain_->getDepthFormat();

    // Create depth image
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = depth_format;
    image_info.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage =
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    if (vkCreateImage(device_->getHandle(), &image_info, nullptr, &shadow_image_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shadow map image");
    }

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(device_->getHandle(), shadow_image_, &mem_req);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = vk_buffer::findMemoryType(
        device_->getPhysicalDevice(), mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device_->getHandle(), &alloc_info, nullptr, &shadow_image_memory_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate shadow map memory");
    }
    vkBindImageMemory(device_->getHandle(), shadow_image_, shadow_image_memory_, 0);

    // Create image view
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = shadow_image_;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = depth_format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device_->getHandle(), &view_info, nullptr, &shadow_image_view_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create shadow map image view");
    }

    // Create sampler with depth comparison
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sampler_info.compareEnable = VK_TRUE;
    sampler_info.compareOp = VK_COMPARE_OP_LESS;

    if (vkCreateSampler(device_->getHandle(), &sampler_info, nullptr, &shadow_sampler_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create shadow sampler");
    }

    // Create shadow render pass (depth-only)
    VkAttachmentDescription depth_attachment{};
    depth_attachment.format = depth_format;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depth_ref{};
    depth_ref.attachment = 0;
    depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &depth_ref;

    VkSubpassDependency dep{};
    dep.srcSubpass = 0;
    dep.dstSubpass = VK_SUBPASS_EXTERNAL;
    dep.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 1;
    rp_info.pAttachments = &depth_attachment;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 1;
    rp_info.pDependencies = &dep;

    if (vkCreateRenderPass(device_->getHandle(), &rp_info, nullptr, &shadow_render_pass_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create shadow render pass");
    }

    // Create framebuffer
    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = shadow_render_pass_;
    fb_info.attachmentCount = 1;
    fb_info.pAttachments = &shadow_image_view_;
    fb_info.width = SHADOW_MAP_SIZE;
    fb_info.height = SHADOW_MAP_SIZE;
    fb_info.layers = 1;

    if (vkCreateFramebuffer(device_->getHandle(), &fb_info, nullptr, &shadow_framebuffer_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create shadow framebuffer");
    }

    logInfo("Shadow map created ({}x{})", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
}

void Renderer::createPixelResources() {
    auto extent = swapchain_->getExtent();
    pixel_width_ = std::max(extent.width / PIXEL_SCALE, 1u);
    pixel_height_ = std::max(extent.height / PIXEL_SCALE, 1u);

    VkFormat color_format = swapchain_->getImageFormat();
    VkFormat depth_format = swapchain_->getDepthFormat();

    // Color image
    VkImageCreateInfo color_info{};
    color_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    color_info.imageType = VK_IMAGE_TYPE_2D;
    color_info.format = color_format;
    color_info.extent = {pixel_width_, pixel_height_, 1};
    color_info.mipLevels = 1;
    color_info.arrayLayers = 1;
    color_info.samples = VK_SAMPLE_COUNT_1_BIT;
    color_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    color_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    if (vkCreateImage(device_->getHandle(), &color_info, nullptr, &pixel_color_image_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create pixel color image");
    }

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(device_->getHandle(), pixel_color_image_, &mem_req);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = vk_buffer::findMemoryType(
        device_->getPhysicalDevice(), mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device_->getHandle(), &alloc_info, nullptr, &pixel_color_memory_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate pixel color memory");
    }
    vkBindImageMemory(device_->getHandle(), pixel_color_image_, pixel_color_memory_, 0);

    VkImageViewCreateInfo color_view_info{};
    color_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    color_view_info.image = pixel_color_image_;
    color_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    color_view_info.format = color_format;
    color_view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(device_->getHandle(), &color_view_info, nullptr, &pixel_color_view_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create pixel color image view");
    }

    // Depth image
    VkImageCreateInfo depth_info{};
    depth_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depth_info.imageType = VK_IMAGE_TYPE_2D;
    depth_info.format = depth_format;
    depth_info.extent = {pixel_width_, pixel_height_, 1};
    depth_info.mipLevels = 1;
    depth_info.arrayLayers = 1;
    depth_info.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    if (vkCreateImage(device_->getHandle(), &depth_info, nullptr, &pixel_depth_image_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create pixel depth image");
    }

    vkGetImageMemoryRequirements(device_->getHandle(), pixel_depth_image_, &mem_req);
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = vk_buffer::findMemoryType(
        device_->getPhysicalDevice(), mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device_->getHandle(), &alloc_info, nullptr, &pixel_depth_memory_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate pixel depth memory");
    }
    vkBindImageMemory(device_->getHandle(), pixel_depth_image_, pixel_depth_memory_, 0);

    VkImageViewCreateInfo depth_view_info{};
    depth_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depth_view_info.image = pixel_depth_image_;
    depth_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depth_view_info.format = depth_format;
    depth_view_info.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(device_->getHandle(), &depth_view_info, nullptr, &pixel_depth_view_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create pixel depth image view");
    }

    // Render pass (color → TRANSFER_SRC for blit)
    if (pixel_render_pass_ == VK_NULL_HANDLE) {
        VkAttachmentDescription color_att{};
        color_att.format = color_format;
        color_att.samples = VK_SAMPLE_COUNT_1_BIT;
        color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        VkAttachmentDescription depth_att{};
        depth_att.format = depth_format;
        depth_att.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_att.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_ref;
        subpass.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency dep{};
        dep.srcSubpass = 0;
        dep.dstSubpass = VK_SUBPASS_EXTERNAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dep.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {color_att, depth_att};

        VkRenderPassCreateInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.attachmentCount = static_cast<uint32_t>(attachments.size());
        rp_info.pAttachments = attachments.data();
        rp_info.subpassCount = 1;
        rp_info.pSubpasses = &subpass;
        rp_info.dependencyCount = 1;
        rp_info.pDependencies = &dep;

        if (vkCreateRenderPass(device_->getHandle(), &rp_info, nullptr, &pixel_render_pass_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create pixel render pass");
        }
    }

    // Framebuffer
    std::array<VkImageView, 2> fb_attachments = {pixel_color_view_, pixel_depth_view_};

    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = pixel_render_pass_;
    fb_info.attachmentCount = static_cast<uint32_t>(fb_attachments.size());
    fb_info.pAttachments = fb_attachments.data();
    fb_info.width = pixel_width_;
    fb_info.height = pixel_height_;
    fb_info.layers = 1;

    if (vkCreateFramebuffer(device_->getHandle(), &fb_info, nullptr, &pixel_framebuffer_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create pixel framebuffer");
    }

    // UI overlay render pass (loads blitted swapchain content, renders ImGui at full res)
    if (ui_render_pass_ == VK_NULL_HANDLE) {
        VkAttachmentDescription color_att{};
        color_att.format = color_format;
        color_att.samples = VK_SAMPLE_COUNT_1_BIT;
        color_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_att.initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        color_att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_ref;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dep.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.attachmentCount = 1;
        rp_info.pAttachments = &color_att;
        rp_info.subpassCount = 1;
        rp_info.pSubpasses = &subpass;
        rp_info.dependencyCount = 1;
        rp_info.pDependencies = &dep;

        if (vkCreateRenderPass(device_->getHandle(), &rp_info, nullptr, &ui_render_pass_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create UI overlay render pass");
        }
    }

    // UI framebuffers (one per swapchain image)
    ui_framebuffers_.resize(swapchain_->getImageCount());
    for (uint32_t i = 0; i < swapchain_->getImageCount(); i++) {
        VkImageView view = swapchain_->getImageView(i);

        VkFramebufferCreateInfo ui_fb_info{};
        ui_fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ui_fb_info.renderPass = ui_render_pass_;
        ui_fb_info.attachmentCount = 1;
        ui_fb_info.pAttachments = &view;
        ui_fb_info.width = swapchain_->getExtent().width;
        ui_fb_info.height = swapchain_->getExtent().height;
        ui_fb_info.layers = 1;

        if (vkCreateFramebuffer(device_->getHandle(), &ui_fb_info, nullptr,
                                &ui_framebuffers_[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create UI overlay framebuffer");
        }
    }

    logInfo("Pixel resources created ({}x{}, scale 1/{})", pixel_width_, pixel_height_,
            PIXEL_SCALE);
}

void Renderer::cleanupPixelResources() {
    for (auto fb : ui_framebuffers_) {
        vkDestroyFramebuffer(device_->getHandle(), fb, nullptr);
    }
    ui_framebuffers_.clear();
    vkDestroyFramebuffer(device_->getHandle(), pixel_framebuffer_, nullptr);
    pixel_framebuffer_ = VK_NULL_HANDLE;
    vkDestroyImageView(device_->getHandle(), pixel_color_view_, nullptr);
    pixel_color_view_ = VK_NULL_HANDLE;
    vkDestroyImage(device_->getHandle(), pixel_color_image_, nullptr);
    pixel_color_image_ = VK_NULL_HANDLE;
    vkFreeMemory(device_->getHandle(), pixel_color_memory_, nullptr);
    pixel_color_memory_ = VK_NULL_HANDLE;
    vkDestroyImageView(device_->getHandle(), pixel_depth_view_, nullptr);
    pixel_depth_view_ = VK_NULL_HANDLE;
    vkDestroyImage(device_->getHandle(), pixel_depth_image_, nullptr);
    pixel_depth_image_ = VK_NULL_HANDLE;
    vkFreeMemory(device_->getHandle(), pixel_depth_memory_, nullptr);
    pixel_depth_memory_ = VK_NULL_HANDLE;
}

void Renderer::createSkyboxCubemap() {
    constexpr uint32_t size = 512;
    constexpr uint32_t face_count = 6;
    constexpr VkDeviceSize face_size = size * size * 4;
    constexpr VkDeviceSize total_size = face_size * face_count;

    std::vector<uint8_t> pixels(total_size);

    glm::vec3 zenith(0.15f, 0.35f, 0.85f);
    glm::vec3 horizon(0.6f, 0.75f, 0.95f);
    glm::vec3 ground(0.35f, 0.35f, 0.35f);

    glm::vec3 sun_dir = glm::normalize(light_pos_);
    glm::vec3 sun_color(1.0f, 0.95f, 0.8f);
    constexpr float sun_radius = 0.07f;
    constexpr float glow_radius = 0.2f;

    for (uint32_t face = 0; face < face_count; face++) {
        for (uint32_t y = 0; y < size; y++) {
            for (uint32_t x = 0; x < size; x++) {
                float u = 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(size) - 1.0f;
                float v = 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(size) - 1.0f;

                glm::vec3 dir;
                switch (face) {
                    case 0: dir = glm::vec3( 1.0f, -v, -u); break;  // +X
                    case 1: dir = glm::vec3(-1.0f, -v,  u); break;  // -X
                    case 2: dir = glm::vec3( u,  1.0f,  v); break;  // +Y
                    case 3: dir = glm::vec3( u, -1.0f, -v); break;  // -Y
                    case 4: dir = glm::vec3( u, -v,  1.0f); break;  // +Z
                    case 5: dir = glm::vec3(-u, -v, -1.0f); break;  // -Z
                    default: dir = glm::vec3(0.0f); break;
                }

                dir = glm::normalize(dir);
                float elevation = dir.z;

                glm::vec3 color;
                if (elevation > 0.0f) {
                    color = glm::mix(horizon, zenith, elevation);
                } else {
                    color = glm::mix(horizon, ground, -elevation);
                }

                // Sun disc + glow
                float cos_angle = glm::dot(dir, sun_dir);
                float angle = std::acos(glm::clamp(cos_angle, -1.0f, 1.0f));
                if (angle < sun_radius) {
                    color = sun_color;
                } else if (angle < glow_radius) {
                    float t = 1.0f - (angle - sun_radius) / (glow_radius - sun_radius);
                    t = t * t;  // quadratic falloff
                    color = glm::mix(color, sun_color, t * 0.6f);
                }

                uint32_t idx = (face * size * size + y * size + x) * 4;
                pixels[idx + 0] = static_cast<uint8_t>(glm::clamp(color.r, 0.0f, 1.0f) * 255.0f);
                pixels[idx + 1] = static_cast<uint8_t>(glm::clamp(color.g, 0.0f, 1.0f) * 255.0f);
                pixels[idx + 2] = static_cast<uint8_t>(glm::clamp(color.b, 0.0f, 1.0f) * 255.0f);
                pixels[idx + 3] = 255;
            }
        }
    }

    // Create cubemap image
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R8G8B8A8_SRGB;
    image_info.extent = {size, size, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = face_count;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

    if (vkCreateImage(device_->getHandle(), &image_info, nullptr, &skybox_image_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create skybox cubemap image");
    }

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(device_->getHandle(), skybox_image_, &mem_req);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = vk_buffer::findMemoryType(
        device_->getPhysicalDevice(), mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device_->getHandle(), &alloc_info, nullptr, &skybox_image_memory_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate skybox cubemap memory");
    }
    vkBindImageMemory(device_->getHandle(), skybox_image_, skybox_image_memory_, 0);

    // Staging upload
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    vk_buffer::createBuffer(*device_, total_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            staging_buffer, staging_memory);

    void* data;
    vkMapMemory(device_->getHandle(), staging_memory, 0, total_size, 0, &data);
    std::memcpy(data, pixels.data(), total_size);
    vkUnmapMemory(device_->getHandle(), staging_memory);

    vk_buffer::transitionImageLayout(device_->getHandle(), command_pool_,
                                     device_->getGraphicsQueue(), skybox_image_,
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, face_count);

    vk_buffer::copyBufferToImage(device_->getHandle(), command_pool_,
                                 device_->getGraphicsQueue(), staging_buffer, skybox_image_, size,
                                 size, face_count, face_size);

    vk_buffer::transitionImageLayout(device_->getHandle(), command_pool_,
                                     device_->getGraphicsQueue(), skybox_image_,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, face_count);

    vkDestroyBuffer(device_->getHandle(), staging_buffer, nullptr);
    vkFreeMemory(device_->getHandle(), staging_memory, nullptr);

    // Create image view (cube type)
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = skybox_image_;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = face_count;

    if (vkCreateImageView(device_->getHandle(), &view_info, nullptr, &skybox_image_view_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create skybox cubemap image view");
    }

    // Create sampler
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    if (vkCreateSampler(device_->getHandle(), &sampler_info, nullptr, &skybox_sampler_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create skybox sampler");
    }

    logInfo("Skybox cubemap created ({}x{} per face, {} KB)", size, size, total_size / 1024);
}

void Renderer::createSkyboxMesh() {
    std::vector<glm::vec3> vertices = {
        {-1.0f, -1.0f, -1.0f}, { 1.0f, -1.0f, -1.0f},
        { 1.0f,  1.0f, -1.0f}, {-1.0f,  1.0f, -1.0f},
        {-1.0f, -1.0f,  1.0f}, { 1.0f, -1.0f,  1.0f},
        { 1.0f,  1.0f,  1.0f}, {-1.0f,  1.0f,  1.0f},
    };

    std::vector<uint32_t> indices = {
        0, 2, 1, 0, 3, 2,  // -Z
        4, 5, 6, 4, 6, 7,  // +Z
        0, 4, 7, 0, 7, 3,  // -X
        1, 2, 6, 1, 6, 5,  // +X
        0, 1, 5, 0, 5, 4,  // -Y
        3, 7, 6, 3, 6, 2,  // +Y
    };

    skybox_index_count_ = static_cast<uint32_t>(indices.size());

    VkDeviceSize vertex_size = sizeof(glm::vec3) * vertices.size();
    VkDeviceSize index_size = sizeof(uint32_t) * indices.size();

    // Vertex buffer via staging
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    vk_buffer::createBuffer(*device_, vertex_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            staging_buffer, staging_memory);

    void* data;
    vkMapMemory(device_->getHandle(), staging_memory, 0, vertex_size, 0, &data);
    std::memcpy(data, vertices.data(), vertex_size);
    vkUnmapMemory(device_->getHandle(), staging_memory);

    vk_buffer::createBuffer(*device_, vertex_size,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skybox_vertex_buffer_,
                            skybox_vertex_memory_);
    vk_buffer::copyBuffer(*device_, command_pool_, staging_buffer, skybox_vertex_buffer_,
                           vertex_size);

    vkDestroyBuffer(device_->getHandle(), staging_buffer, nullptr);
    vkFreeMemory(device_->getHandle(), staging_memory, nullptr);

    // Index buffer via staging
    vk_buffer::createBuffer(*device_, index_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            staging_buffer, staging_memory);

    vkMapMemory(device_->getHandle(), staging_memory, 0, index_size, 0, &data);
    std::memcpy(data, indices.data(), index_size);
    vkUnmapMemory(device_->getHandle(), staging_memory);

    vk_buffer::createBuffer(*device_, index_size,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skybox_index_buffer_,
                            skybox_index_memory_);
    vk_buffer::copyBuffer(*device_, command_pool_, staging_buffer, skybox_index_buffer_, index_size);

    vkDestroyBuffer(device_->getHandle(), staging_buffer, nullptr);
    vkFreeMemory(device_->getHandle(), staging_memory, nullptr);

    logInfo("Skybox mesh created");
}

void Renderer::loadMesh() {
    mesh_ = Mesh::loadFromOBJ(*device_, command_pool_, std::string(ASSETS_DIR) + "cube.obj");

    // Ground plane at Z = -0.5 (base of the cube)
    // Use tex coords 0,0 so it samples a single texel, vertex color controls the tint
    std::vector<Vertex> ground_verts = {
        {{-100.0f, -100.0f, -0.5f}, {0.7f, 0.7f, 0.7f}, {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{ 100.0f, -100.0f, -0.5f}, {0.7f, 0.7f, 0.7f}, {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{ 100.0f,  100.0f, -0.5f}, {0.7f, 0.7f, 0.7f}, {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{-100.0f,  100.0f, -0.5f}, {0.7f, 0.7f, 0.7f}, {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    };
    std::vector<uint32_t> ground_indices = {0, 1, 2, 2, 3, 0};
    ground_mesh_ = std::make_unique<Mesh>(*device_, command_pool_, ground_verts, ground_indices);
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
    render_finished_semaphores_.resize(swapchain_->getImageCount());
    in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device_->getHandle(), &sem_info, nullptr,
                              &image_available_semaphores_[i]) != VK_SUCCESS ||
            vkCreateFence(device_->getHandle(), &fence_info, nullptr, &in_flight_fences_[i]) !=
                VK_SUCCESS) {
            throw std::runtime_error("Failed to create sync objects");
        }
    }
    for (size_t i = 0; i < swapchain_->getImageCount(); i++) {
        if (vkCreateSemaphore(device_->getHandle(), &sem_info, nullptr,
                              &render_finished_semaphores_[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create sync objects");
        }
    }
}

void Renderer::recordShadowPass(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_info.renderPass = shadow_render_pass_;
    rp_info.framebuffer = shadow_framebuffer_;
    rp_info.renderArea.offset = {0, 0};
    rp_info.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};

    VkClearValue clear_value{};
    clear_value.depthStencil = {1.0f, 0};
    rp_info.clearValueCount = 1;
    rp_info.pClearValues = &clear_value;

    vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_->getHandle());

    VkDescriptorSet desc_set = descriptors_->getSet(current_frame_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_->getLayout(),
                            0, 1, &desc_set, 0, nullptr);

    VkViewport viewport{};
    viewport.width = static_cast<float>(SHADOW_MAP_SIZE);
    viewport.height = static_cast<float>(SHADOW_MAP_SIZE);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Draw cube
    PushConstants pc{};
    pc.model = cube_model_;
    vkCmdPushConstants(cmd, shadow_pipeline_->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(PushConstants), &pc);
    mesh_->bind(cmd);
    mesh_->draw(cmd);

    // Draw ground
    pc.model = glm::mat4(1.0f);
    vkCmdPushConstants(cmd, shadow_pipeline_->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(PushConstants), &pc);
    ground_mesh_->bind(cmd);
    ground_mesh_->draw(cmd);

    vkCmdEndRenderPass(cmd);
}

void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t image_index) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer");
    }

    // Shadow pass
    recordShadowPass(cmd);

    // Offscreen pass (low-res pixel render target)
    VkRenderPassBeginInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_info.renderPass = pixel_render_pass_;
    rp_info.framebuffer = pixel_framebuffer_;
    rp_info.renderArea.offset = {0, 0};
    rp_info.renderArea.extent = {pixel_width_, pixel_height_};

    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color = {{0.01f, 0.01f, 0.02f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};
    rp_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
    rp_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.width = static_cast<float>(pixel_width_);
    viewport.height = static_cast<float>(pixel_height_);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = {pixel_width_, pixel_height_};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Skybox (drawn first; pos.xyww places it at max depth)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox_pipeline_->getHandle());
    VkDescriptorSet skybox_set = descriptors_->getSkyboxSet(current_frame_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox_pipeline_->getLayout(),
                            0, 1, &skybox_set, 0, nullptr);
    VkBuffer skybox_buffers[] = {skybox_vertex_buffer_};
    VkDeviceSize skybox_offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, skybox_buffers, skybox_offsets);
    vkCmdBindIndexBuffer(cmd, skybox_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, skybox_index_count_, 1, 0, 0, 0);

    // Scene
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getHandle());
    VkDescriptorSet desc_set = descriptors_->getSet(current_frame_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getLayout(), 0, 1,
                            &desc_set, 0, nullptr);

    // Draw cube
    PushConstants pc{};
    pc.model = cube_model_;
    vkCmdPushConstants(cmd, pipeline_->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(PushConstants), &pc);
    mesh_->bind(cmd);
    mesh_->draw(cmd);

    // Draw ground
    pc.model = glm::mat4(1.0f);
    vkCmdPushConstants(cmd, pipeline_->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(PushConstants), &pc);
    ground_mesh_->bind(cmd);
    ground_mesh_->draw(cmd);

    vkCmdEndRenderPass(cmd);

    // Blit offscreen → swapchain with nearest filtering (pixelated upscale)
    VkImage swapchain_image = swapchain_->getImage(image_index);

    VkImageMemoryBarrier pre_blit{};
    pre_blit.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    pre_blit.srcAccessMask = 0;
    pre_blit.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre_blit.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    pre_blit.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre_blit.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_blit.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_blit.image = swapchain_image;
    pre_blit.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &pre_blit);

    VkImageBlit blit_region{};
    blit_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit_region.srcOffsets[0] = {0, 0, 0};
    blit_region.srcOffsets[1] = {static_cast<int32_t>(pixel_width_),
                                 static_cast<int32_t>(pixel_height_), 1};
    blit_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit_region.dstOffsets[0] = {0, 0, 0};
    blit_region.dstOffsets[1] = {static_cast<int32_t>(swapchain_->getExtent().width),
                                 static_cast<int32_t>(swapchain_->getExtent().height), 1};

    vkCmdBlitImage(cmd, pixel_color_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_region, VK_FILTER_NEAREST);

    VkImageMemoryBarrier post_blit{};
    post_blit.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    post_blit.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    post_blit.dstAccessMask = 0;
    post_blit.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    post_blit.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    post_blit.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post_blit.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post_blit.image = swapchain_image;
    post_blit.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr,
                         1, &post_blit);

    // UI overlay pass (ImGui at full resolution)
    VkRenderPassBeginInfo ui_rp_info{};
    ui_rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    ui_rp_info.renderPass = ui_render_pass_;
    ui_rp_info.framebuffer = ui_framebuffers_[image_index];
    ui_rp_info.renderArea.offset = {0, 0};
    ui_rp_info.renderArea.extent = swapchain_->getExtent();

    vkCmdBeginRenderPass(cmd, &ui_rp_info, VK_SUBPASS_CONTENTS_INLINE);

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRenderPass(cmd);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer");
    }
}

void Renderer::initImGui() {
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    };

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = pool_sizes;

    if (vkCreateDescriptorPool(device_->getHandle(), &pool_info, nullptr, &imgui_pool_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create ImGui descriptor pool");
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window_.getHandle(), true);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.ApiVersion = VK_API_VERSION_1_3;
    init_info.Instance = instance_->getHandle();
    init_info.PhysicalDevice = device_->getPhysicalDevice();
    init_info.Device = device_->getHandle();
    init_info.QueueFamily = device_->getQueueFamilies().graphics.value();
    init_info.Queue = device_->getGraphicsQueue();
    init_info.DescriptorPool = imgui_pool_;
    init_info.MinImageCount = 2;
    init_info.ImageCount = swapchain_->getImageCount();
    init_info.PipelineInfoMain.RenderPass = ui_render_pass_;
    init_info.PipelineInfoMain.Subpass = 0;

    ImGui_ImplVulkan_Init(&init_info);
    logInfo("ImGui initialized");
}

void Renderer::shutdownImGui() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(device_->getHandle(), imgui_pool_, nullptr);
}

void Renderer::recreateSwapchain() {
    VkExtent2D extent = window_.getExtent();
    while (extent.width == 0 || extent.height == 0) {
        extent = window_.getExtent();
        glfwWaitEvents();
    }

    swapchain_->recreate(extent);
    cleanupPixelResources();
    createPixelResources();
}

}  // namespace engine
