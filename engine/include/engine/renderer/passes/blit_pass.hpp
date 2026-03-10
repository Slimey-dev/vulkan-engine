#pragma once

#include <engine/renderer/render_graph.hpp>

namespace engine {

class BlitPass : public RenderPassNode {
public:
    BlitPass(ResourceId src, ResourceId dst);
    void record(VkCommandBuffer cmd, const RenderGraph& graph) override;

private:
    ResourceId src_;
    ResourceId dst_;
};

}  // namespace engine
