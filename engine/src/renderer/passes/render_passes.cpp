#include <engine/ecs/components.hpp>
#include <engine/renderer/passes/blit_pass.hpp>
#include <engine/renderer/passes/bloom_blur_pass.hpp>
#include <engine/renderer/passes/bloom_composite_pass.hpp>
#include <engine/renderer/passes/bloom_extract_pass.hpp>
#include <engine/renderer/passes/scene_pass.hpp>
#include <engine/renderer/passes/shadow_pass.hpp>
#include <engine/renderer/passes/ui_pass.hpp>
#include <engine/renderer/render_graph.hpp>
#include <engine/renderer/scene.hpp>
#include <engine/renderer/vk_descriptors.hpp>
#include <engine/renderer/vk_pipeline.hpp>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

namespace engine {

// --- ShadowPass ---

ShadowPass::ShadowPass(VulkanPipeline*& pipeline, VulkanDescriptors*& descriptors, Scene*& scene,
                       uint32_t& current_frame)
    : pipeline_(pipeline), descriptors_(descriptors), scene_(scene), current_frame_(current_frame) {
}

void ShadowPass::record(VkCommandBuffer cmd, const RenderGraph& graph) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getHandle());

    VkDescriptorSet desc_set = descriptors_->getSet(current_frame_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getLayout(), 0, 1,
                            &desc_set, 0, nullptr);

    setViewportAndScissor(cmd, extent);

    scene_->registry.each<Transform, MeshRenderer>(
        [&](Entity e, Transform& t, MeshRenderer& mr) {
            if (scene_->registry.has<VolumetricCone>(e)) return;
            PushConstants pc{};
            pc.model = t.matrix();
            vkCmdPushConstants(cmd, pipeline_->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(PushConstants), &pc);
            mr.mesh->bind(cmd);
            mr.mesh->draw(cmd);
        });
}

// --- ScenePass ---

ScenePass::ScenePass(ScenePassContext ctx) : ctx_(ctx) {}

void ScenePass::record(VkCommandBuffer cmd, const RenderGraph& graph) {
    setViewportAndScissor(cmd, extent);

    if (ctx_.scene->skybox_enabled) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx_.skybox_pipeline->getHandle());
        VkDescriptorSet skybox_set = ctx_.descriptors->getSkyboxSet(ctx_.current_frame);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                ctx_.skybox_pipeline->getLayout(), 0, 1, &skybox_set, 0, nullptr);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &ctx_.skybox_vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, ctx_.skybox_index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, ctx_.skybox_index_count, 1, 0, 0, 0);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx_.pipeline->getHandle());
    VkDescriptorSet desc_set = ctx_.descriptors->getSet(ctx_.current_frame);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx_.pipeline->getLayout(), 0, 1,
                            &desc_set, 0, nullptr);

    ctx_.scene->registry.each<Transform, MeshRenderer>(
        [&](Entity e, Transform& t, MeshRenderer& mr) {
            if (ctx_.scene->registry.has<VolumetricCone>(e)) return;
            PushConstants pc{};
            pc.model = t.matrix();
            pc.emissive = ctx_.scene->registry.has<Emissive>(e) ? 1.0f : 0.0f;
            vkCmdPushConstants(cmd, ctx_.pipeline->getLayout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(PushConstants), &pc);
            mr.mesh->bind(cmd);
            mr.mesh->draw(cmd);
        });

    ctx_.scene->registry.each<Transform, MeshRenderer, VolumetricCone>(
        [&](Entity, Transform& t, MeshRenderer& mr, VolumetricCone&) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              ctx_.volumetric_pipeline->getHandle());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    ctx_.volumetric_pipeline->getLayout(), 0, 1, &desc_set, 0,
                                    nullptr);
            PushConstants pc{};
            pc.model = t.matrix();
            vkCmdPushConstants(cmd, ctx_.volumetric_pipeline->getLayout(),
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pc);
            mr.mesh->bind(cmd);
            mr.mesh->draw(cmd);
        });

    if (ctx_.show_debug_rays && ctx_.debug_line_buffer != VK_NULL_HANDLE &&
        ctx_.debug_line_vertex_count > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          ctx_.debug_line_pipeline->getHandle());
        VkDescriptorSet desc = ctx_.descriptors->getSet(ctx_.current_frame);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                ctx_.debug_line_pipeline->getLayout(), 0, 1, &desc, 0, nullptr);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &ctx_.debug_line_buffer, &offset);
        vkCmdDraw(cmd, ctx_.debug_line_vertex_count, 1, 0, 0);
    }
}

// --- BlitPass ---

BlitPass::BlitPass(ResourceId src, ResourceId dst) : src_(src), dst_(dst) {
    is_transfer_pass = true;
}

void BlitPass::record(VkCommandBuffer cmd, const RenderGraph& graph) {
    const auto& src_res = graph.getResource(src_);
    const auto& dst_res = graph.getResource(dst_);

    VkImageBlit blit_region{};
    blit_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit_region.srcOffsets[0] = {0, 0, 0};
    blit_region.srcOffsets[1] = {static_cast<int32_t>(src_res.extent.width),
                                 static_cast<int32_t>(src_res.extent.height), 1};
    blit_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit_region.dstOffsets[0] = {0, 0, 0};
    blit_region.dstOffsets[1] = {static_cast<int32_t>(dst_res.extent.width),
                                 static_cast<int32_t>(dst_res.extent.height), 1};

    vkCmdBlitImage(cmd, src_res.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_res.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_region, VK_FILTER_NEAREST);
}

// --- UIPass ---

void UIPass::record(VkCommandBuffer cmd, const RenderGraph& graph) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

// --- BloomExtractPass ---

BloomExtractPass::BloomExtractPass(VulkanPipeline*& pipeline, VkDescriptorSet& desc_set,
                                   float& threshold)
    : pipeline_(pipeline), desc_set_(desc_set), threshold_(threshold) {}

void BloomExtractPass::record(VkCommandBuffer cmd, const RenderGraph& graph) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getHandle());
    setViewportAndScissor(cmd, extent);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getLayout(), 0, 1,
                            &desc_set_, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline_->getLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float),
                       &threshold_);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

// --- BloomBlurPass ---

BloomBlurPass::BloomBlurPass(VulkanPipeline*& pipeline, VkDescriptorSet& desc_set,
                             glm::vec2& direction)
    : pipeline_(pipeline), desc_set_(desc_set), direction_(direction) {}

void BloomBlurPass::record(VkCommandBuffer cmd, const RenderGraph& graph) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getHandle());
    setViewportAndScissor(cmd, extent);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getLayout(), 0, 1,
                            &desc_set_, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline_->getLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(glm::vec2), &direction_);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

// --- BloomCompositePass ---

BloomCompositePass::BloomCompositePass(VulkanPipeline*& pipeline, VkDescriptorSet& desc_set,
                                       float& intensity)
    : pipeline_(pipeline), desc_set_(desc_set), intensity_(intensity) {}

void BloomCompositePass::record(VkCommandBuffer cmd, const RenderGraph& graph) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getHandle());
    setViewportAndScissor(cmd, extent);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getLayout(), 0, 1,
                            &desc_set_, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline_->getLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float),
                       &intensity_);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

}  // namespace engine
