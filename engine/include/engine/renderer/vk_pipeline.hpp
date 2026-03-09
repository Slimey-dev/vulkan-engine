#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace engine {

class VulkanPipeline {
public:
    VulkanPipeline(VkDevice device, VkRenderPass render_pass, VkExtent2D extent,
                   const std::string& vert_path, const std::string& frag_path);
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
