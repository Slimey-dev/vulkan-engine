#pragma once

#include <engine/renderer/render_graph.hpp>

#include <glm/glm.hpp>

namespace engine {

class VulkanPipeline;

class BloomBlurPass : public RenderPassNode {
public:
    BloomBlurPass(VulkanPipeline*& pipeline, VkDescriptorSet& desc_set, glm::vec2& direction);
    void record(VkCommandBuffer cmd, const RenderGraph& graph) override;

private:
    VulkanPipeline*& pipeline_;
    VkDescriptorSet& desc_set_;
    glm::vec2& direction_;
};

}  // namespace engine
