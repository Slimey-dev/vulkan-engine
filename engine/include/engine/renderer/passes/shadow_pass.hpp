#pragma once

#include <engine/renderer/render_graph.hpp>

namespace engine {

class VulkanDescriptors;
class VulkanPipeline;
class Scene;

class ShadowPass : public RenderPassNode {
public:
    ShadowPass(VulkanPipeline*& pipeline, VulkanDescriptors*& descriptors, Scene*& scene,
               uint32_t& current_frame);
    void record(VkCommandBuffer cmd, const RenderGraph& graph) override;

private:
    VulkanPipeline*& pipeline_;
    VulkanDescriptors*& descriptors_;
    Scene*& scene_;
    uint32_t& current_frame_;
};

}  // namespace engine
