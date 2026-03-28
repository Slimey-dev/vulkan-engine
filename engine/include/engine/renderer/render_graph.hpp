#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

class GpuProfiler;
class VulkanDevice;
class VulkanSwapchain;

enum class ResourceId : uint32_t {};

enum class ResourceUsage {
    ColorAttachmentWrite,
    DepthAttachmentWrite,
    ShaderReadOnly,
    TransferSrc,
    TransferDst,
    Present,
};

struct ImageDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    float width_scale = 1.0f;
    float height_scale = 1.0f;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags extra_usage = 0;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
};

struct PhysicalResource {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    bool owned = false;
};

struct ResourceAccess {
    ResourceId id;
    ResourceUsage usage;
};

class RenderGraph;

inline void setViewportAndScissor(VkCommandBuffer cmd, VkExtent2D extent) {
    VkViewport vp{0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height),
                  0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

class RenderPassNode {
public:
    virtual ~RenderPassNode() = default;
    virtual void record(VkCommandBuffer cmd, const RenderGraph& graph) = 0;

    std::string name;
    std::vector<ResourceAccess> inputs;
    std::vector<ResourceAccess> outputs;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkExtent2D extent{};
    std::vector<VkClearValue> clear_values;
    bool is_transfer_pass = false;

    // Final layout for each output attachment (parallel to outputs)
    std::vector<VkImageLayout> output_final_layouts;

    // For swapchain-indexed framebuffers (e.g. UI pass)
    std::vector<VkFramebuffer> per_image_framebuffers;
};

struct ImageBarrier {
    ResourceId resource;
    VkImageLayout old_layout;
    VkImageLayout new_layout;
    VkAccessFlags src_access;
    VkAccessFlags dst_access;
    VkPipelineStageFlags src_stage;
    VkPipelineStageFlags dst_stage;
};

struct ExecutionStep {
    uint32_t pass_index;
    std::vector<ImageBarrier> pre_barriers;
};

class RenderGraph {
public:
    RenderGraph(VkDevice device);
    ~RenderGraph();

    RenderGraph(const RenderGraph&) = delete;
    RenderGraph& operator=(const RenderGraph&) = delete;

    void execute(VkCommandBuffer cmd, uint32_t swapchain_image_index, uint32_t current_frame);
    void setProfiler(GpuProfiler* profiler) { profiler_ = profiler; }

    const PhysicalResource& getResource(ResourceId id) const;
    VkRenderPass getRenderPass(const std::string& pass_name) const;
    VkExtent2D getPassExtent(const std::string& pass_name) const;

    void setSwapchainImage(ResourceId id, uint32_t image_index, VkImage image, VkImageView view);

private:
    friend class RenderGraphBuilder;

    VkDevice device_;
    GpuProfiler* profiler_ = nullptr;
    std::vector<std::unique_ptr<RenderPassNode>> passes_;
    std::vector<PhysicalResource> resources_;
    std::vector<ExecutionStep> steps_;
    std::unordered_map<std::string, uint32_t> pass_name_map_;

    // Swapchain resource tracking
    struct SwapchainBinding {
        ResourceId id;
        VulkanSwapchain* swapchain;
    };
    std::vector<SwapchainBinding> swapchain_bindings_;
};

struct ResourceInfo {
    std::string name;
    ImageDesc desc;
    bool is_imported = false;
    bool is_swapchain = false;
    VulkanSwapchain* swapchain = nullptr;
    // Pre-filled for imported resources
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
};

class RenderGraphBuilder {
public:
    RenderGraphBuilder(VulkanDevice& device);

    ResourceId createImage(const std::string& name, const ImageDesc& desc);
    ResourceId importImage(const std::string& name, VkImage image, VkImageView view,
                           VkFormat format, VkExtent2D extent,
                           VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
    ResourceId importSwapchain(const std::string& name, VulkanSwapchain& swapchain);

    template <typename T, typename... Args>
    T* addPass(const std::string& name, Args&&... args) {
        auto pass = std::make_unique<T>(std::forward<Args>(args)...);
        pass->name = name;
        T* ptr = pass.get();
        passes_.push_back(std::move(pass));
        return ptr;
    }

    void passReads(RenderPassNode* pass, ResourceId resource, ResourceUsage usage);
    void passWrites(RenderPassNode* pass, ResourceId resource, ResourceUsage usage);

    std::unique_ptr<RenderGraph> build(VkExtent2D swapchain_extent);

private:
    void topologicalSort();
    void allocateTransientResources(VkExtent2D swapchain_extent);
    void createRenderPasses();
    void createFramebuffers();
    void computeBarriers();

    VulkanDevice& device_;
    std::vector<ResourceInfo> resources_;
    std::vector<std::unique_ptr<RenderPassNode>> passes_;
    std::vector<uint32_t> sorted_order_;

    // Built outputs
    std::vector<PhysicalResource> physical_resources_;
    std::vector<ExecutionStep> steps_;
};

}  // namespace engine
