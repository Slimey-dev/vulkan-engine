#include <engine/ecs/components.hpp>
#include <engine/renderer/passes/blit_pass.hpp>
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

ScenePass::ScenePass(VulkanPipeline*& pipeline, VulkanPipeline*& skybox_pipeline,
                     VulkanPipeline*& volumetric_pipeline, VulkanDescriptors*& descriptors,
                     Scene*& scene, uint32_t& current_frame, VkBuffer& skybox_vertex_buffer,
                     VkBuffer& skybox_index_buffer, uint32_t& skybox_index_count)
    : pipeline_(pipeline),
      skybox_pipeline_(skybox_pipeline),
      volumetric_pipeline_(volumetric_pipeline),
      descriptors_(descriptors),
      scene_(scene),
      current_frame_(current_frame),
      skybox_vertex_buffer_(skybox_vertex_buffer),
      skybox_index_buffer_(skybox_index_buffer),
      skybox_index_count_(skybox_index_count) {}

void ScenePass::record(VkCommandBuffer cmd, const RenderGraph& graph) {
    setViewportAndScissor(cmd, extent);

    // Skybox
    if (scene_->skybox_enabled) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox_pipeline_->getHandle());
        VkDescriptorSet skybox_set = descriptors_->getSkyboxSet(current_frame_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                skybox_pipeline_->getLayout(), 0, 1, &skybox_set, 0, nullptr);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &skybox_vertex_buffer_, &offset);
        vkCmdBindIndexBuffer(cmd, skybox_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, skybox_index_count_, 1, 0, 0, 0);
    }

    // Opaque entities
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getHandle());
    VkDescriptorSet desc_set = descriptors_->getSet(current_frame_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->getLayout(), 0, 1,
                            &desc_set, 0, nullptr);

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

    // Volumetric cone (additive blend)
    scene_->registry.each<Transform, MeshRenderer, VolumetricCone>(
        [&](Entity, Transform& t, MeshRenderer& mr, VolumetricCone&) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              volumetric_pipeline_->getHandle());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    volumetric_pipeline_->getLayout(), 0, 1, &desc_set, 0,
                                    nullptr);
            PushConstants pc{};
            pc.model = t.matrix();
            vkCmdPushConstants(cmd, volumetric_pipeline_->getLayout(), VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(PushConstants), &pc);
            mr.mesh->bind(cmd);
            mr.mesh->draw(cmd);
        });
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

}  // namespace engine
