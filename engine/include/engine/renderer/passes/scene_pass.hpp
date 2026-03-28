#pragma once

#include <engine/renderer/render_graph.hpp>

#include <vulkan/vulkan.h>

namespace engine {

class VulkanDescriptors;
class VulkanPipeline;
class Scene;

struct ScenePassContext {
    VulkanPipeline*& pipeline;
    VulkanPipeline*& skybox_pipeline;
    VulkanPipeline*& volumetric_pipeline;
    VulkanPipeline*& debug_line_pipeline;
    VulkanDescriptors*& descriptors;
    Scene*& scene;
    uint32_t& current_frame;
    VkBuffer& skybox_vertex_buffer;
    VkBuffer& skybox_index_buffer;
    uint32_t& skybox_index_count;
    VkBuffer& debug_line_buffer;
    uint32_t& debug_line_vertex_count;
    bool& show_debug_rays;
};

class ScenePass : public RenderPassNode {
public:
    explicit ScenePass(ScenePassContext ctx);
    void record(VkCommandBuffer cmd, const RenderGraph& graph) override;

private:
    ScenePassContext ctx_;
};

}  // namespace engine
