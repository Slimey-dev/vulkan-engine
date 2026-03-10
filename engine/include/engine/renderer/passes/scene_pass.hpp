#pragma once

#include <engine/renderer/render_graph.hpp>

#include <vulkan/vulkan.h>

namespace engine {

class VulkanDescriptors;
class VulkanPipeline;
class Scene;

class ScenePass : public RenderPassNode {
public:
    ScenePass(VulkanPipeline*& pipeline, VulkanPipeline*& skybox_pipeline,
              VulkanPipeline*& volumetric_pipeline, VulkanDescriptors*& descriptors, Scene*& scene,
              uint32_t& current_frame, VkBuffer& skybox_vertex_buffer,
              VkBuffer& skybox_index_buffer, uint32_t& skybox_index_count);
    void record(VkCommandBuffer cmd, const RenderGraph& graph) override;

private:
    VulkanPipeline*& pipeline_;
    VulkanPipeline*& skybox_pipeline_;
    VulkanPipeline*& volumetric_pipeline_;
    VulkanDescriptors*& descriptors_;
    Scene*& scene_;
    uint32_t& current_frame_;
    VkBuffer& skybox_vertex_buffer_;
    VkBuffer& skybox_index_buffer_;
    uint32_t& skybox_index_count_;
};

}  // namespace engine
