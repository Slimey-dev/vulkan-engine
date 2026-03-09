#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace engine {

struct PipelineConfig {
    bool depth_write = true;
    VkCompareOp depth_compare_op = VK_COMPARE_OP_LESS;
    VkCullModeFlags cull_mode = VK_CULL_MODE_BACK_BIT;
    bool has_push_constants = true;
    bool depth_bias = false;
    float depth_bias_constant = 0.0f;
    float depth_bias_slope = 0.0f;
    bool has_color_attachment = true;
};

class VulkanPipeline {
public:
    VulkanPipeline(VkDevice device, VkRenderPass render_pass,
                   const std::string& vert_path, const std::string& frag_path,
                   const std::vector<VkVertexInputBindingDescription>& bindings,
                   const std::vector<VkVertexInputAttributeDescription>& attributes,
                   VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE,
                   bool depth_only = false);

    VulkanPipeline(VkDevice device, VkRenderPass render_pass,
                   const std::string& vert_path, const std::string& frag_path,
                   const std::vector<VkVertexInputBindingDescription>& bindings,
                   const std::vector<VkVertexInputAttributeDescription>& attributes,
                   VkDescriptorSetLayout descriptor_set_layout,
                   const PipelineConfig& config);
    ~VulkanPipeline();

    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;

    VkPipeline getHandle() const { return pipeline_; }
    VkPipelineLayout getLayout() const { return layout_; }

private:
    static std::vector<char> readFile(const std::string& filepath);
    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace engine
