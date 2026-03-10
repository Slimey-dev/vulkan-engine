#pragma once

#include <engine/renderer/render_graph.hpp>

namespace engine {

class VulkanPipeline;

class BloomCompositePass : public RenderPassNode {
public:
    BloomCompositePass(VulkanPipeline*& pipeline, VkDescriptorSet& desc_set, float& intensity);
    void record(VkCommandBuffer cmd, const RenderGraph& graph) override;

private:
    VulkanPipeline*& pipeline_;
    VkDescriptorSet& desc_set_;
    float& intensity_;
};

}  // namespace engine
