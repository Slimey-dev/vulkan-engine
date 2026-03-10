#pragma once

#include <engine/renderer/render_graph.hpp>

namespace engine {

class UIPass : public RenderPassNode {
public:
    UIPass() = default;
    void record(VkCommandBuffer cmd, const RenderGraph& graph) override;
};

}  // namespace engine
