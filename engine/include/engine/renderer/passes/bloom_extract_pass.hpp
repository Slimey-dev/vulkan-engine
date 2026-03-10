#pragma once

#include <engine/renderer/render_graph.hpp>

namespace engine {

class VulkanPipeline;

class BloomExtractPass : public RenderPassNode {
public:
    BloomExtractPass(VulkanPipeline*& pipeline, VkDescriptorSet& desc_set, float& threshold);
    void record(VkCommandBuffer cmd, const RenderGraph& graph) override;

private:
    VulkanPipeline*& pipeline_;
    VkDescriptorSet& desc_set_;
    float& threshold_;
};

}  // namespace engine
