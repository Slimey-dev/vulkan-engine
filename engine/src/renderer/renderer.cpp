#include <engine/core/log.hpp>
#include <engine/ecs/components.hpp>
#include <engine/renderer/passes/blit_pass.hpp>
#include <engine/renderer/passes/bloom_blur_pass.hpp>
#include <engine/renderer/passes/bloom_composite_pass.hpp>
#include <engine/renderer/passes/bloom_extract_pass.hpp>
#include <engine/renderer/passes/scene_pass.hpp>
#include <engine/renderer/passes/shadow_pass.hpp>
#include <engine/renderer/passes/ui_pass.hpp>
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
    createTexture();
    createShadowResources();
    createSkyboxCubemap();
    descriptors_ = std::make_unique<VulkanDescriptors>(
        *device_, MAX_FRAMES_IN_FLIGHT, texture_->getImageView(), texture_->getSampler(),
        shadow_image_view_, shadow_sampler_, skybox_image_view_, skybox_sampler_);

    audio_.setSpatialAudio(&spatial_audio_);
    loadScene(0);
    createSkyboxMesh();
    buildRenderGraph();
    createBloomDescriptors();
    createPipelines();
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

    scene_.reset();
    vkDestroyCommandPool(device_->getHandle(), command_pool_, nullptr);

    pipeline_.reset();
    shadow_pipeline_.reset();
    skybox_pipeline_.reset();
    volumetric_pipeline_.reset();
    bloom_extract_pipeline_.reset();
    bloom_blur_pipeline_.reset();
    bloom_composite_pipeline_.reset();
    debug_line_pipeline_.reset();
    if (debug_line_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_->getHandle(), debug_line_buffer_, nullptr);
        vkFreeMemory(device_->getHandle(), debug_line_buffer_memory_, nullptr);
    }
    cleanupBloomDescriptors();
    descriptors_.reset();
    texture_.reset();

    render_graph_.reset();

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

    if (camera_.didJump()) audio_.playJump(camera_.getPosition());
    if (camera_.didLand()) audio_.playLand(camera_.getPosition());

    spatial_audio_.setListener(camera_.getPosition(), camera_.getFront(), camera_.getUp());
    spatial_audio_.simulate();

    // Scene switching
    if (window_.isKeyPressed(GLFW_KEY_1) && current_scene_index_ != 0) loadScene(0);
    if (window_.isKeyPressed(GLFW_KEY_2) && current_scene_index_ != 1) loadScene(1);
    if (window_.isKeyPressed(GLFW_KEY_3) && current_scene_index_ != 2) loadScene(2);

    // Update Rotator system
    scene_->registry.each<Transform, Rotator>(
        [&](Entity, Transform& t, Rotator& r) { t.rotation += r.axis * r.speed * delta_time; });

    // ImGui debug panel
    {
        ImGui::Begin("Engine Debug");
        ImGui::Text("Scene: %s [1/2/3]", scene_->name());
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
        ImGui::Text("Bloom");
        ImGui::SliderFloat("Bloom Threshold", &bloom_threshold_, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Bloom Intensity", &bloom_intensity_, 0.0f, 3.0f, "%.2f");

        ImGui::Separator();
        ImGui::Text("Spatial Audio");
        ImGui::Checkbox("Enabled##spatial", &spatial_audio_.enabled);
        ImGui::Checkbox("HRTF", &spatial_audio_.hrtf_enabled);
        ImGui::SliderFloat("Reverb Mix", &spatial_audio_.reverb_mix, 0.0f, 2.0f, "%.2f");
        ImGui::Checkbox("Show Debug Rays", &show_debug_rays_);
        if (show_debug_rays_) {
            ImGui::SliderInt("Ray Count", &debug_ray_count_, 1, 12);
            ImGui::SliderInt("Bounces", &debug_ray_bounces_, 1, 8);
        }

        ImGui::Separator();
        ImGui::Text("Press [Tab] to toggle UI/FPS mode");
        ImGui::End();
    }

    if (show_debug_rays_) {
        updateDebugRays();
    }

    auto extent = swapchain_->getExtent();
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);

    float osz = scene_->shadow_ortho_size;
    glm::mat4 light_view =
        glm::lookAt(light_pos_, glm::vec3(0.0f), scene_->shadow_up);
    glm::mat4 light_proj = glm::ortho(-osz, osz, -osz, osz, 0.1f, scene_->shadow_far);
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
    float cos_outer = light_cone_angle_ > 0.0f
                          ? std::cos(glm::radians(light_cone_angle_))
                          : 0.0f;
    ubo.light_dir = glm::vec4(glm::normalize(light_dir_), cos_outer);

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

    logInfo("Shadow map created ({}x{})", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
}

void Renderer::buildRenderGraph() {
    RenderGraphBuilder builder(*device_);

    VkFormat depth_format = swapchain_->getDepthFormat();

    shadow_map_id_ = builder.importImage("shadow_map", shadow_image_, shadow_image_view_,
                                         depth_format, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE},
                                         VK_IMAGE_ASPECT_DEPTH_BIT);
    swapchain_id_ = builder.importSwapchain("swapchain", *swapchain_);

    float scale = 1.0f / static_cast<float>(PIXEL_SCALE);
    VkFormat hdr_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    pixel_color_id_ = builder.createImage("pixel_color",
                                          {.width_scale = scale,
                                           .height_scale = scale,
                                           .format = hdr_format});
    pixel_depth_id_ = builder.createImage("pixel_depth",
                                          {.width_scale = scale,
                                           .height_scale = scale,
                                           .format = depth_format,
                                           .aspect = VK_IMAGE_ASPECT_DEPTH_BIT});
    bloom_extract_id_ = builder.createImage("bloom_extract",
                                            {.width_scale = scale,
                                             .height_scale = scale,
                                             .format = hdr_format});
    bloom_blur_h_id_ = builder.createImage("bloom_blur_h",
                                           {.width_scale = scale,
                                            .height_scale = scale,
                                            .format = hdr_format});
    bloom_blurred_id_ = builder.createImage("bloom_blurred",
                                            {.width_scale = scale,
                                             .height_scale = scale,
                                             .format = hdr_format});

    auto* shadow = builder.addPass<ShadowPass>("shadow", shadow_pipeline_ptr_, descriptors_ptr_,
                                               scene_ptr_, current_frame_);
    shadow->extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
    VkClearValue shadow_clear{};
    shadow_clear.depthStencil = {1.0f, 0};
    shadow->clear_values = {shadow_clear};
    builder.passWrites(shadow, shadow_map_id_, ResourceUsage::DepthAttachmentWrite);

    auto* scene_pass = builder.addPass<ScenePass>(
        "scene",
        ScenePassContext{pipeline_ptr_, skybox_pipeline_ptr_, volumetric_pipeline_ptr_,
                         debug_line_pipeline_ptr_, descriptors_ptr_, scene_ptr_, current_frame_,
                         skybox_vertex_buffer_, skybox_index_buffer_, skybox_index_count_,
                         debug_line_buffer_, debug_line_vertex_count_, show_debug_rays_});
    VkClearValue color_clear{};
    color_clear.color = {{fog_color_.x, fog_color_.y, fog_color_.z, 1.0f}};
    VkClearValue depth_clear{};
    depth_clear.depthStencil = {1.0f, 0};
    scene_pass->clear_values = {color_clear, depth_clear};
    builder.passReads(scene_pass, shadow_map_id_, ResourceUsage::ShaderReadOnly);
    builder.passWrites(scene_pass, pixel_color_id_, ResourceUsage::ColorAttachmentWrite);
    builder.passWrites(scene_pass, pixel_depth_id_, ResourceUsage::DepthAttachmentWrite);

    VkClearValue black{};
    black.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    auto* bloom_extract = builder.addPass<BloomExtractPass>(
        "bloom_extract", bloom_extract_pipeline_ptr_, bloom_extract_set_, bloom_threshold_);
    bloom_extract->clear_values = {black};
    builder.passReads(bloom_extract, pixel_color_id_, ResourceUsage::ShaderReadOnly);
    builder.passWrites(bloom_extract, bloom_extract_id_, ResourceUsage::ColorAttachmentWrite);

    auto* bloom_blur_h = builder.addPass<BloomBlurPass>(
        "bloom_blur_h", bloom_blur_pipeline_ptr_, bloom_blur_h_set_, bloom_blur_h_dir_);
    bloom_blur_h->clear_values = {black};
    builder.passReads(bloom_blur_h, bloom_extract_id_, ResourceUsage::ShaderReadOnly);
    builder.passWrites(bloom_blur_h, bloom_blur_h_id_, ResourceUsage::ColorAttachmentWrite);

    auto* bloom_blur_v = builder.addPass<BloomBlurPass>(
        "bloom_blur_v", bloom_blur_pipeline_ptr_, bloom_blur_v_set_, bloom_blur_v_dir_);
    bloom_blur_v->clear_values = {black};
    builder.passReads(bloom_blur_v, bloom_blur_h_id_, ResourceUsage::ShaderReadOnly);
    builder.passWrites(bloom_blur_v, bloom_blurred_id_, ResourceUsage::ColorAttachmentWrite);

    auto* composite = builder.addPass<BloomCompositePass>(
        "composite", bloom_composite_pipeline_ptr_, bloom_composite_set_, bloom_intensity_);
    composite->clear_values = {black};
    builder.passReads(composite, pixel_color_id_, ResourceUsage::ShaderReadOnly);
    builder.passReads(composite, bloom_blurred_id_, ResourceUsage::ShaderReadOnly);
    builder.passWrites(composite, swapchain_id_, ResourceUsage::ColorAttachmentWrite);

    auto* ui = builder.addPass<UIPass>("ui");
    builder.passWrites(ui, swapchain_id_, ResourceUsage::ColorAttachmentWrite);

    render_graph_ = builder.build(swapchain_->getExtent());
    logInfo("Render graph built");
}

void Renderer::createPipelines() {
    auto binding = Vertex::getBindingDescription();
    auto attributes = Vertex::getAttributeDescriptions();

    uint32_t pc_size = sizeof(PushConstants);

    PipelineConfig shadow_config{};
    shadow_config.depth_bias = true;
    shadow_config.depth_bias_constant = 1.25f;
    shadow_config.depth_bias_slope = 1.75f;
    shadow_config.has_color_attachment = false;
    shadow_config.push_constant_size = pc_size;

    shadow_pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), render_graph_->getRenderPass("shadow"),
        std::string(SHADER_DIR) + "shadow.vert.spv",
        std::string(SHADER_DIR) + "shadow.frag.spv",
        std::vector{binding},
        std::vector<VkVertexInputAttributeDescription>(attributes.begin(), attributes.end()),
        descriptors_->getLayout(), shadow_config);

    PipelineConfig scene_config{};
    scene_config.push_constant_size = pc_size;
    scene_config.push_constant_stages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), render_graph_->getRenderPass("scene"),
        std::string(SHADER_DIR) + "triangle.vert.spv",
        std::string(SHADER_DIR) + "triangle.frag.spv",
        std::vector{binding},
        std::vector<VkVertexInputAttributeDescription>(attributes.begin(), attributes.end()),
        descriptors_->getLayout(), scene_config);

    PipelineConfig vol_config{};
    vol_config.depth_write = false;
    vol_config.cull_mode = VK_CULL_MODE_NONE;
    vol_config.additive_blend = true;
    vol_config.push_constant_size = pc_size;
    volumetric_pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), render_graph_->getRenderPass("scene"),
        std::string(SHADER_DIR) + "volumetric.vert.spv",
        std::string(SHADER_DIR) + "volumetric.frag.spv",
        std::vector{binding},
        std::vector<VkVertexInputAttributeDescription>(attributes.begin(), attributes.end()),
        descriptors_->getLayout(), vol_config);

    // Debug line pipeline
    PipelineConfig debug_line_config{};
    debug_line_config.depth_write = false;
    debug_line_config.cull_mode = VK_CULL_MODE_NONE;
    debug_line_config.line_topology = true;
    debug_line_config.has_push_constants = false;

    debug_line_pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), render_graph_->getRenderPass("scene"),
        std::string(SHADER_DIR) + "debug_line.vert.spv",
        std::string(SHADER_DIR) + "debug_line.frag.spv",
        std::vector{binding},
        std::vector<VkVertexInputAttributeDescription>(attributes.begin(), attributes.end()),
        descriptors_->getLayout(), debug_line_config);

    // Skybox pipeline
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
        device_->getHandle(), render_graph_->getRenderPass("scene"),
        std::string(SHADER_DIR) + "skybox.vert.spv",
        std::string(SHADER_DIR) + "skybox.frag.spv",
        std::vector{skybox_binding}, std::vector{skybox_attr},
        descriptors_->getSkyboxLayout(), skybox_config);

    // Bloom extract pipeline
    PipelineConfig bloom_extract_config{};
    bloom_extract_config.depth_test = false;
    bloom_extract_config.depth_write = false;
    bloom_extract_config.cull_mode = VK_CULL_MODE_NONE;
    bloom_extract_config.push_constant_size = sizeof(float);
    bloom_extract_config.push_constant_stages = VK_SHADER_STAGE_FRAGMENT_BIT;
    bloom_extract_pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), render_graph_->getRenderPass("bloom_extract"),
        std::string(SHADER_DIR) + "fullscreen.vert.spv",
        std::string(SHADER_DIR) + "bloom_extract.frag.spv",
        std::vector<VkVertexInputBindingDescription>{},
        std::vector<VkVertexInputAttributeDescription>{}, bloom_desc_layout_,
        bloom_extract_config);

    // Bloom blur pipeline (shared for H and V)
    PipelineConfig bloom_blur_config{};
    bloom_blur_config.depth_test = false;
    bloom_blur_config.depth_write = false;
    bloom_blur_config.cull_mode = VK_CULL_MODE_NONE;
    bloom_blur_config.push_constant_size = sizeof(float) * 2;
    bloom_blur_config.push_constant_stages = VK_SHADER_STAGE_FRAGMENT_BIT;
    bloom_blur_pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), render_graph_->getRenderPass("bloom_blur_h"),
        std::string(SHADER_DIR) + "fullscreen.vert.spv",
        std::string(SHADER_DIR) + "bloom_blur.frag.spv",
        std::vector<VkVertexInputBindingDescription>{},
        std::vector<VkVertexInputAttributeDescription>{}, bloom_desc_layout_, bloom_blur_config);

    // Bloom composite pipeline
    PipelineConfig bloom_composite_config{};
    bloom_composite_config.depth_test = false;
    bloom_composite_config.depth_write = false;
    bloom_composite_config.cull_mode = VK_CULL_MODE_NONE;
    bloom_composite_config.push_constant_size = sizeof(float);
    bloom_composite_config.push_constant_stages = VK_SHADER_STAGE_FRAGMENT_BIT;
    bloom_composite_pipeline_ = std::make_unique<VulkanPipeline>(
        device_->getHandle(), render_graph_->getRenderPass("composite"),
        std::string(SHADER_DIR) + "fullscreen.vert.spv",
        std::string(SHADER_DIR) + "bloom_composite.frag.spv",
        std::vector<VkVertexInputBindingDescription>{},
        std::vector<VkVertexInputAttributeDescription>{}, bloom_composite_desc_layout_,
        bloom_composite_config);

    // Update pointer aliases for pass bindings
    pipeline_ptr_ = pipeline_.get();
    shadow_pipeline_ptr_ = shadow_pipeline_.get();
    skybox_pipeline_ptr_ = skybox_pipeline_.get();
    volumetric_pipeline_ptr_ = volumetric_pipeline_.get();
    bloom_extract_pipeline_ptr_ = bloom_extract_pipeline_.get();
    bloom_blur_pipeline_ptr_ = bloom_blur_pipeline_.get();
    bloom_composite_pipeline_ptr_ = bloom_composite_pipeline_.get();
    debug_line_pipeline_ptr_ = debug_line_pipeline_.get();
    descriptors_ptr_ = descriptors_.get();
    scene_ptr_ = scene_.get();
}

void Renderer::createBloomDescriptors() {
    VkDevice dev = device_->getHandle();

    // Create descriptor set layouts
    VkDescriptorSetLayoutBinding single_binding{};
    single_binding.binding = 0;
    single_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    single_binding.descriptorCount = 1;
    single_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &single_binding;
    if (vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &bloom_desc_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create bloom descriptor set layout");
    }

    VkDescriptorSetLayoutBinding composite_bindings[2]{};
    composite_bindings[0].binding = 0;
    composite_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    composite_bindings[0].descriptorCount = 1;
    composite_bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    composite_bindings[1].binding = 1;
    composite_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    composite_bindings[1].descriptorCount = 1;
    composite_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    layout_info.bindingCount = 2;
    layout_info.pBindings = composite_bindings;
    if (vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &bloom_composite_desc_layout_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create bloom composite descriptor set layout");
    }

    // Create descriptor pool
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = 5;  // 3 single + 2 composite

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 4;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(dev, &pool_info, nullptr, &bloom_desc_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create bloom descriptor pool");
    }

    // Create samplers
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(dev, &sampler_info, nullptr, &bloom_sampler_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create bloom sampler");
    }

    sampler_info.magFilter = VK_FILTER_NEAREST;
    sampler_info.minFilter = VK_FILTER_NEAREST;
    if (vkCreateSampler(dev, &sampler_info, nullptr, &bloom_nearest_sampler_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create bloom nearest sampler");
    }

    // Allocate descriptor sets
    VkDescriptorSetLayout single_layouts[3] = {bloom_desc_layout_, bloom_desc_layout_,
                                               bloom_desc_layout_};
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = bloom_desc_pool_;
    alloc_info.descriptorSetCount = 3;
    alloc_info.pSetLayouts = single_layouts;

    VkDescriptorSet single_sets[3];
    if (vkAllocateDescriptorSets(dev, &alloc_info, single_sets) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate bloom descriptor sets");
    }
    bloom_extract_set_ = single_sets[0];
    bloom_blur_h_set_ = single_sets[1];
    bloom_blur_v_set_ = single_sets[2];

    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &bloom_composite_desc_layout_;
    if (vkAllocateDescriptorSets(dev, &alloc_info, &bloom_composite_set_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate bloom composite descriptor set");
    }

    // Write descriptor sets with image views from render graph
    auto& pixel_color_res = render_graph_->getResource(pixel_color_id_);
    auto& bloom_extract_res = render_graph_->getResource(bloom_extract_id_);
    auto& bloom_blur_h_res = render_graph_->getResource(bloom_blur_h_id_);
    auto& bloom_blurred_res = render_graph_->getResource(bloom_blurred_id_);

    auto writeSet = [&](VkDescriptorSet set, uint32_t binding, VkImageView view,
                        VkSampler sampler) {
        VkDescriptorImageInfo img_info{};
        img_info.sampler = sampler;
        img_info.imageView = view;
        img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &img_info;
        vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
    };

    // Extract reads pixel_color
    writeSet(bloom_extract_set_, 0, pixel_color_res.view, bloom_sampler_);
    // Blur H reads bloom_extract
    writeSet(bloom_blur_h_set_, 0, bloom_extract_res.view, bloom_sampler_);
    // Blur V reads bloom_blur_h
    writeSet(bloom_blur_v_set_, 0, bloom_blur_h_res.view, bloom_sampler_);
    // Composite reads pixel_color (nearest) + bloom_blurred (linear)
    writeSet(bloom_composite_set_, 0, pixel_color_res.view, bloom_nearest_sampler_);
    writeSet(bloom_composite_set_, 1, bloom_blurred_res.view, bloom_sampler_);

    // Compute blur directions
    bloom_blur_h_dir_ = {1.0f / static_cast<float>(bloom_extract_res.extent.width), 0.0f};
    bloom_blur_v_dir_ = {0.0f, 1.0f / static_cast<float>(bloom_extract_res.extent.height)};

    logInfo("Bloom descriptors created");
}

void Renderer::cleanupBloomDescriptors() {
    VkDevice dev = device_->getHandle();
    if (bloom_desc_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, bloom_desc_pool_, nullptr);
        bloom_desc_pool_ = VK_NULL_HANDLE;
    }
    if (bloom_desc_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, bloom_desc_layout_, nullptr);
        bloom_desc_layout_ = VK_NULL_HANDLE;
    }
    if (bloom_composite_desc_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, bloom_composite_desc_layout_, nullptr);
        bloom_composite_desc_layout_ = VK_NULL_HANDLE;
    }
    if (bloom_sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(dev, bloom_sampler_, nullptr);
        bloom_sampler_ = VK_NULL_HANDLE;
    }
    if (bloom_nearest_sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(dev, bloom_nearest_sampler_, nullptr);
        bloom_nearest_sampler_ = VK_NULL_HANDLE;
    }
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

// Möller–Trumbore ray-triangle intersection
static bool rayTriangle(glm::vec3 origin, glm::vec3 dir, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
                        float& t_out, glm::vec3& normal_out) {
    glm::vec3 e1 = v1 - v0, e2 = v2 - v0;
    glm::vec3 h = glm::cross(dir, e2);
    float a = glm::dot(e1, h);
    if (std::abs(a) < 1e-6f) return false;
    float f = 1.0f / a;
    glm::vec3 s = origin - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    glm::vec3 q = glm::cross(s, e1);
    float v = f * glm::dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = f * glm::dot(e2, q);
    if (t < 0.001f) return false;
    t_out = t;
    normal_out = glm::normalize(glm::cross(e1, e2));
    return true;
}

static glm::vec3 getBounceColor(int bounce) {
    switch (bounce) {
        case 0: return {1.0f, 0.0f, 0.0f};   // red
        case 1: return {1.0f, 0.6f, 0.0f};   // orange
        case 2: return {1.0f, 1.0f, 0.0f};   // yellow
        case 3: return {0.0f, 1.0f, 0.0f};   // green
        case 4: return {0.0f, 0.6f, 1.0f};   // cyan
        case 5: return {0.2f, 0.2f, 1.0f};   // blue
        case 6: return {0.7f, 0.0f, 1.0f};   // purple
        default: return {1.0f, 0.0f, 0.6f};  // pink
    }
}

// Max verts: 12 rays * 9 bounces * 2 verts per line = 216
static constexpr uint32_t kMaxDebugLineVerts = 256;

void Renderer::updateDebugRays() {
    auto acoustic_meshes = scene_->getAcousticMeshes();
    if (acoustic_meshes.empty()) {
        debug_line_vertex_count_ = 0;
        return;
    }

    // Allocate persistent buffer once
    if (debug_line_buffer_ == VK_NULL_HANDLE) {
        VkDeviceSize buf_size = sizeof(Vertex) * kMaxDebugLineVerts;
        vk_buffer::createBuffer(*device_, buf_size,
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                debug_line_buffer_, debug_line_buffer_memory_);
    }

    // Gather all triangles
    struct Tri { glm::vec3 v0, v1, v2; };
    std::vector<Tri> tris;
    for (const auto* mesh : acoustic_meshes) {
        if (!mesh) continue;
        for (size_t i = 0; i + 2 < mesh->indices.size(); i += 3) {
            tris.push_back({mesh->positions[mesh->indices[i]],
                            mesh->positions[mesh->indices[i + 1]],
                            mesh->positions[mesh->indices[i + 2]]});
        }
    }

    std::vector<Vertex> line_verts;
    glm::vec3 cam_pos = scene_->has_debug_ray_origin ? scene_->debug_ray_origin
                                                      : camera_.getPosition();

    constexpr float golden_angle = 2.399963f;

    for (int r = 0; r < debug_ray_count_; r++) {
        // Full sphere Fibonacci sampling: z ranges from +1 to -1
        float z = 1.0f - (2.0f * r + 1.0f) / static_cast<float>(debug_ray_count_);
        float rad = std::sqrt(1.0f - z * z);
        float theta = golden_angle * r;
        glm::vec3 dir = glm::normalize(
            glm::vec3(rad * std::cos(theta), rad * std::sin(theta), z));

        glm::vec3 origin = cam_pos;

        for (int bounce = 0; bounce <= debug_ray_bounces_; bounce++) {
            float closest_t = 1e9f;
            glm::vec3 hit_normal{0};
            bool hit = false;

            for (const auto& tri : tris) {
                float t;
                glm::vec3 n;
                if (rayTriangle(origin, dir, tri.v0, tri.v1, tri.v2, t, n) && t < closest_t) {
                    closest_t = t;
                    hit_normal = n;
                    hit = true;
                }
            }

            glm::vec3 color = getBounceColor(bounce);

            if (!hit) {
                glm::vec3 end = origin + dir * 3.0f;
                line_verts.push_back({origin, color, {0, 0}, {0, 0, 0}});
                line_verts.push_back({end, color * 0.3f, {0, 0}, {0, 0, 0}});
                break;
            }

            glm::vec3 hit_point = origin + dir * closest_t;
            line_verts.push_back({origin, color, {0, 0}, {0, 0, 0}});
            line_verts.push_back({hit_point, color, {0, 0}, {0, 0, 0}});

            if (glm::dot(dir, hit_normal) > 0) hit_normal = -hit_normal;
            dir = glm::reflect(dir, hit_normal);
            origin = hit_point + hit_normal * 0.01f;

            if (line_verts.size() >= kMaxDebugLineVerts - 2) break;
        }
        if (line_verts.size() >= kMaxDebugLineVerts - 2) break;
    }

    debug_line_vertex_count_ = static_cast<uint32_t>(line_verts.size());

    if (debug_line_vertex_count_ > 0) {
        void* data;
        vkMapMemory(device_->getHandle(), debug_line_buffer_memory_, 0,
                     sizeof(Vertex) * debug_line_vertex_count_, 0, &data);
        std::memcpy(data, line_verts.data(), sizeof(Vertex) * debug_line_vertex_count_);
        vkUnmapMemory(device_->getHandle(), debug_line_buffer_memory_);
    }
}

void Renderer::loadScene(int index) {
    if (scene_) {
        vkDeviceWaitIdle(device_->getHandle());
        scene_.reset();
    }

    if (index == 1) {
        scene_ = std::make_unique<IndoorScene>();
    } else if (index == 2) {
        scene_ = std::make_unique<DebugViewScene>();
    } else {
        scene_ = std::make_unique<OutdoorScene>();
    }
    scene_->init(*device_, command_pool_);
    scene_ptr_ = scene_.get();
    current_scene_index_ = index;

    spatial_audio_.clearScene();
    auto acoustic_meshes = scene_->getAcousticMeshes();
    if (!acoustic_meshes.empty()) {
        spatial_audio_.buildScene(acoustic_meshes);
    }

    camera_.reset(scene_->camera_start, scene_->camera_yaw, scene_->camera_pitch);
    camera_.setBounds(scene_->bounds.min_x, scene_->bounds.max_x,
                      scene_->bounds.min_y, scene_->bounds.max_y,
                      scene_->bounds.max_z);

    light_pos_ = scene_->light_pos;
    light_color_ = scene_->light_color;
    fog_color_ = scene_->fog_color;
    fog_density_ = scene_->fog_density;
    light_dir_ = scene_->light_dir;
    light_cone_angle_ = scene_->light_cone_angle;
    bloom_threshold_ = scene_->bloom_threshold;
    bloom_intensity_ = scene_->bloom_intensity;
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

void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t image_index) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer");
    }

    render_graph_->execute(cmd, image_index, current_frame_);

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
    init_info.PipelineInfoMain.RenderPass = render_graph_->getRenderPass("ui");
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

    vkDeviceWaitIdle(device_->getHandle());
    swapchain_->recreate(extent);

    // Rebuild graph (recreates transient resources, render passes, framebuffers)
    cleanupBloomDescriptors();
    render_graph_.reset();
    buildRenderGraph();
    createBloomDescriptors();

    // Recreate pipelines with new render passes
    pipeline_.reset();
    shadow_pipeline_.reset();
    skybox_pipeline_.reset();
    volumetric_pipeline_.reset();
    bloom_extract_pipeline_.reset();
    bloom_blur_pipeline_.reset();
    bloom_composite_pipeline_.reset();
    createPipelines();

    // Recreate ImGui with new render pass
    shutdownImGui();
    initImGui();
}

}  // namespace engine
