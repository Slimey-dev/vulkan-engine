#include <engine/core/log.hpp>
#include <engine/renderer/render_graph.hpp>
#include <engine/renderer/vk_buffer.hpp>
#include <engine/renderer/vk_device.hpp>
#include <engine/renderer/vk_swapchain.hpp>

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace engine {

namespace {

struct UsageInfo {
    VkImageLayout layout;
    VkPipelineStageFlags stage;
    VkAccessFlags access;
};

UsageInfo getUsageInfo(ResourceUsage usage, bool is_depth = false) {
    switch (usage) {
        case ResourceUsage::ColorAttachmentWrite:
            return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};
        case ResourceUsage::DepthAttachmentWrite:
            return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
        case ResourceUsage::ShaderReadOnly:
            return {is_depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                             : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT};
        case ResourceUsage::TransferSrc:
            return {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT};
        case ResourceUsage::TransferDst:
            return {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT};
        case ResourceUsage::Present:
            return {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0};
    }
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0};
}

bool isDepthFormat(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM;
}

VkImageUsageFlags usageToVkUsage(ResourceUsage usage) {
    switch (usage) {
        case ResourceUsage::ColorAttachmentWrite:
            return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        case ResourceUsage::DepthAttachmentWrite:
            return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        case ResourceUsage::ShaderReadOnly:
            return VK_IMAGE_USAGE_SAMPLED_BIT;
        case ResourceUsage::TransferSrc:
            return VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        case ResourceUsage::TransferDst:
            return VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        case ResourceUsage::Present:
            return 0;
    }
    return 0;
}

}  // namespace

// --- RenderGraph ---

RenderGraph::RenderGraph(VkDevice device) : device_(device) {}

RenderGraph::~RenderGraph() {
    for (auto& pass : passes_) {
        if (pass->render_pass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device_, pass->render_pass, nullptr);
        }
        if (pass->framebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_, pass->framebuffer, nullptr);
        }
        for (auto fb : pass->per_image_framebuffers) {
            vkDestroyFramebuffer(device_, fb, nullptr);
        }
    }
    for (auto& res : resources_) {
        if (res.owned) {
            vkDestroyImageView(device_, res.view, nullptr);
            vkDestroyImage(device_, res.image, nullptr);
            vkFreeMemory(device_, res.memory, nullptr);
        }
    }
}

void RenderGraph::execute(VkCommandBuffer cmd, uint32_t swapchain_image_index,
                          uint32_t current_frame) {
    // Update swapchain images for this frame
    for (auto& binding : swapchain_bindings_) {
        auto idx = static_cast<uint32_t>(binding.id);
        resources_[idx].image = binding.swapchain->getImage(swapchain_image_index);
        resources_[idx].view = binding.swapchain->getImageView(swapchain_image_index);
        resources_[idx].current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    for (auto& step : steps_) {
        auto& pass = passes_[step.pass_index];

        // Insert pre-barriers
        for (auto& barrier : step.pre_barriers) {
            auto res_idx = static_cast<uint32_t>(barrier.resource);
            auto& res = resources_[res_idx];

            VkImageMemoryBarrier img_barrier{};
            img_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            img_barrier.srcAccessMask = barrier.src_access;
            img_barrier.dstAccessMask = barrier.dst_access;
            img_barrier.oldLayout = res.current_layout;
            img_barrier.newLayout = barrier.new_layout;
            img_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            img_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            img_barrier.image = res.image;
            img_barrier.subresourceRange = {res.aspect, 0, 1, 0, 1};

            vkCmdPipelineBarrier(cmd, barrier.src_stage, barrier.dst_stage, 0, 0, nullptr, 0,
                                 nullptr, 1, &img_barrier);

            res.current_layout = barrier.new_layout;
        }

        if (!pass->is_transfer_pass) {
            VkFramebuffer fb = pass->framebuffer;
            if (!pass->per_image_framebuffers.empty()) {
                fb = pass->per_image_framebuffers[swapchain_image_index];
            }

            VkRenderPassBeginInfo rp_info{};
            rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp_info.renderPass = pass->render_pass;
            rp_info.framebuffer = fb;
            rp_info.renderArea.offset = {0, 0};
            rp_info.renderArea.extent = pass->extent;
            rp_info.clearValueCount = static_cast<uint32_t>(pass->clear_values.size());
            rp_info.pClearValues = pass->clear_values.data();

            vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
            pass->record(cmd, *this);
            vkCmdEndRenderPass(cmd);
        } else {
            pass->record(cmd, *this);
        }
    }
}

const PhysicalResource& RenderGraph::getResource(ResourceId id) const {
    return resources_[static_cast<uint32_t>(id)];
}

VkRenderPass RenderGraph::getRenderPass(const std::string& pass_name) const {
    auto it = pass_name_map_.find(pass_name);
    if (it == pass_name_map_.end()) {
        throw std::runtime_error("Render pass not found: " + pass_name);
    }
    return passes_[it->second]->render_pass;
}

VkExtent2D RenderGraph::getPassExtent(const std::string& pass_name) const {
    auto it = pass_name_map_.find(pass_name);
    if (it == pass_name_map_.end()) {
        throw std::runtime_error("Render pass not found: " + pass_name);
    }
    return passes_[it->second]->extent;
}

void RenderGraph::setSwapchainImage(ResourceId id, uint32_t image_index, VkImage image,
                                    VkImageView view) {
    auto idx = static_cast<uint32_t>(id);
    resources_[idx].image = image;
    resources_[idx].view = view;
}

// --- RenderGraphBuilder ---

RenderGraphBuilder::RenderGraphBuilder(VulkanDevice& device) : device_(device) {}

ResourceId RenderGraphBuilder::createImage(const std::string& name, const ImageDesc& desc) {
    auto id = static_cast<ResourceId>(resources_.size());
    ResourceInfo info{};
    info.name = name;
    info.desc = desc;
    info.is_imported = false;
    info.aspect = desc.aspect;
    resources_.push_back(std::move(info));
    return id;
}

ResourceId RenderGraphBuilder::importImage(const std::string& name, VkImage image,
                                           VkImageView view, VkFormat format, VkExtent2D extent,
                                           VkImageAspectFlags aspect) {
    auto id = static_cast<ResourceId>(resources_.size());
    ResourceInfo info{};
    info.name = name;
    info.is_imported = true;
    info.image = image;
    info.view = view;
    info.format = format;
    info.extent = extent;
    info.aspect = aspect;
    resources_.push_back(std::move(info));
    return id;
}

ResourceId RenderGraphBuilder::importSwapchain(const std::string& name,
                                               VulkanSwapchain& swapchain) {
    auto id = static_cast<ResourceId>(resources_.size());
    ResourceInfo info{};
    info.name = name;
    info.is_imported = true;
    info.is_swapchain = true;
    info.swapchain = &swapchain;
    info.format = swapchain.getImageFormat();
    info.extent = swapchain.getExtent();
    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    resources_.push_back(std::move(info));
    return id;
}

void RenderGraphBuilder::passReads(RenderPassNode* pass, ResourceId resource,
                                   ResourceUsage usage) {
    pass->inputs.push_back({resource, usage});
}

void RenderGraphBuilder::passWrites(RenderPassNode* pass, ResourceId resource,
                                    ResourceUsage usage) {
    pass->outputs.push_back({resource, usage});
}

std::unique_ptr<RenderGraph> RenderGraphBuilder::build(VkExtent2D swapchain_extent) {
    topologicalSort();
    allocateTransientResources(swapchain_extent);
    createRenderPasses();
    createFramebuffers();
    computeBarriers();

    auto graph = std::make_unique<RenderGraph>(device_.getHandle());

    // Transfer physical resources
    graph->resources_ = std::move(physical_resources_);

    // Reorder passes according to sorted order
    std::vector<std::unique_ptr<RenderPassNode>> sorted_passes;
    sorted_passes.reserve(sorted_order_.size());
    for (uint32_t idx : sorted_order_) {
        graph->pass_name_map_[passes_[idx]->name] = static_cast<uint32_t>(sorted_passes.size());
        sorted_passes.push_back(std::move(passes_[idx]));
    }
    graph->passes_ = std::move(sorted_passes);

    graph->steps_ = std::move(steps_);

    // Register swapchain bindings
    for (size_t i = 0; i < resources_.size(); i++) {
        if (resources_[i].is_swapchain) {
            graph->swapchain_bindings_.push_back(
                {static_cast<ResourceId>(i), resources_[i].swapchain});
        }
    }

    return graph;
}

void RenderGraphBuilder::topologicalSort() {
    uint32_t n = static_cast<uint32_t>(passes_.size());

    // Build adjacency: if pass A writes resource R and pass B reads R, A -> B
    std::vector<std::vector<uint32_t>> adj(n);
    std::vector<uint32_t> in_degree(n, 0);

    // Map resource -> writer pass index
    std::unordered_map<uint32_t, uint32_t> resource_writer;
    for (uint32_t i = 0; i < n; i++) {
        for (auto& out : passes_[i]->outputs) {
            resource_writer[static_cast<uint32_t>(out.id)] = i;
        }
    }

    for (uint32_t i = 0; i < n; i++) {
        for (auto& in : passes_[i]->inputs) {
            auto it = resource_writer.find(static_cast<uint32_t>(in.id));
            if (it != resource_writer.end() && it->second != i) {
                adj[it->second].push_back(i);
                in_degree[i]++;
            }
        }
    }

    // Also handle write-after-write ordering for resources written by multiple passes
    // (e.g. swapchain written by blit then UI)
    std::unordered_map<uint32_t, std::vector<uint32_t>> resource_writers;
    for (uint32_t i = 0; i < n; i++) {
        for (auto& out : passes_[i]->outputs) {
            resource_writers[static_cast<uint32_t>(out.id)].push_back(i);
        }
    }
    for (auto& [res, writers] : resource_writers) {
        for (size_t i = 0; i + 1 < writers.size(); i++) {
            // Check if edge already exists
            bool found = false;
            for (auto dest : adj[writers[i]]) {
                if (dest == writers[i + 1]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                adj[writers[i]].push_back(writers[i + 1]);
                in_degree[writers[i + 1]]++;
            }
        }
    }

    // Kahn's algorithm
    std::queue<uint32_t> queue;
    for (uint32_t i = 0; i < n; i++) {
        if (in_degree[i] == 0) queue.push(i);
    }

    sorted_order_.clear();
    while (!queue.empty()) {
        uint32_t u = queue.front();
        queue.pop();
        sorted_order_.push_back(u);
        for (uint32_t v : adj[u]) {
            if (--in_degree[v] == 0) queue.push(v);
        }
    }

    if (sorted_order_.size() != n) {
        throw std::runtime_error("Render graph has cycles");
    }
}

void RenderGraphBuilder::allocateTransientResources(VkExtent2D swapchain_extent) {
    physical_resources_.resize(resources_.size());

    // Cache swapchain format for transient resources with UNDEFINED format
    VkFormat swapchain_format = VK_FORMAT_UNDEFINED;
    for (auto& r : resources_) {
        if (r.is_swapchain) {
            swapchain_format = r.format;
            break;
        }
    }

    // Gather all usages for each resource to determine VkImageUsageFlags
    std::vector<VkImageUsageFlags> resource_usage_flags(resources_.size(), 0);
    for (auto& pass : passes_) {
        for (auto& in : pass->inputs) {
            resource_usage_flags[static_cast<uint32_t>(in.id)] |= usageToVkUsage(in.usage);
        }
        for (auto& out : pass->outputs) {
            resource_usage_flags[static_cast<uint32_t>(out.id)] |= usageToVkUsage(out.usage);
        }
    }

    for (size_t i = 0; i < resources_.size(); i++) {
        auto& info = resources_[i];
        auto& phys = physical_resources_[i];

        if (info.is_imported) {
            phys.image = info.image;
            phys.view = info.view;
            phys.format = info.format;
            phys.extent = info.extent;
            phys.aspect = info.aspect;
            phys.owned = false;
            phys.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            continue;
        }

        // Resolve dimensions
        uint32_t w = info.desc.width;
        uint32_t h = info.desc.height;
        if (w == 0) w = std::max(static_cast<uint32_t>(swapchain_extent.width * info.desc.width_scale), 1u);
        if (h == 0) h = std::max(static_cast<uint32_t>(swapchain_extent.height * info.desc.height_scale), 1u);

        VkFormat format = info.desc.format;
        if (format == VK_FORMAT_UNDEFINED) {
            format = swapchain_format;
        }

        VkImageAspectFlags aspect = info.desc.aspect;
        if (isDepthFormat(format)) {
            aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        }

        VkImageUsageFlags usage = resource_usage_flags[i] | info.desc.extra_usage;

        // Create image
        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = format;
        image_info.extent = {w, h, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = usage;

        if (vkCreateImage(device_.getHandle(), &image_info, nullptr, &phys.image) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create transient image: " + info.name);
        }

        VkMemoryRequirements mem_req;
        vkGetImageMemoryRequirements(device_.getHandle(), phys.image, &mem_req);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_req.size;
        alloc_info.memoryTypeIndex = vk_buffer::findMemoryType(
            device_.getPhysicalDevice(), mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device_.getHandle(), &alloc_info, nullptr, &phys.memory) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate transient image memory: " + info.name);
        }
        vkBindImageMemory(device_.getHandle(), phys.image, phys.memory, 0);

        // Create image view
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = phys.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format;
        view_info.subresourceRange = {aspect, 0, 1, 0, 1};

        if (vkCreateImageView(device_.getHandle(), &view_info, nullptr, &phys.view) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create transient image view: " + info.name);
        }

        phys.format = format;
        phys.extent = {w, h};
        phys.aspect = aspect;
        phys.owned = true;
        phys.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;

        info.format = format;
        info.extent = {w, h};
        info.aspect = aspect;
    }
}

void RenderGraphBuilder::createRenderPasses() {
    // Build set of resources written by earlier passes (for detecting LOAD vs CLEAR)
    std::unordered_set<uint32_t> previously_written;

    for (uint32_t idx : sorted_order_) {
        auto& pass = passes_[idx];
        if (pass->is_transfer_pass) {
            for (auto& out : pass->outputs) {
                previously_written.insert(static_cast<uint32_t>(out.id));
            }
            continue;
        }

        std::vector<VkAttachmentDescription> attachments;
        std::vector<VkAttachmentReference> color_refs;
        VkAttachmentReference depth_ref{};
        bool has_depth = false;

        // Find position of this pass in sorted order
        size_t my_pos = 0;
        for (size_t si = 0; si < sorted_order_.size(); si++) {
            if (sorted_order_[si] == idx) {
                my_pos = si;
                break;
            }
        }

        for (auto& out : pass->outputs) {
            auto res_idx = static_cast<uint32_t>(out.id);
            auto& info = resources_[res_idx];
            VkFormat format = info.format;
            if (format == VK_FORMAT_UNDEFINED && info.is_swapchain) {
                format = info.swapchain->getImageFormat();
            }

            VkAttachmentDescription att{};
            att.format = format;
            att.samples = VK_SAMPLE_COUNT_1_BIT;
            att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

            if (out.usage == ResourceUsage::DepthAttachmentWrite) {
                att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                // Check if any later pass reads this as ShaderReadOnly
                bool sampled_later = false;
                for (size_t sj = my_pos + 1; sj < sorted_order_.size(); sj++) {
                    for (auto& in : passes_[sorted_order_[sj]]->inputs) {
                        if (static_cast<uint32_t>(in.id) == res_idx &&
                            in.usage == ResourceUsage::ShaderReadOnly) {
                            sampled_later = true;
                        }
                    }
                }
                att.finalLayout = sampled_later ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

                depth_ref.attachment = static_cast<uint32_t>(attachments.size());
                depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                has_depth = true;
            } else {
                // Color attachment — load if previously written by an earlier pass
                bool load_existing = previously_written.count(res_idx) > 0;

                att.loadOp = load_existing ? VK_ATTACHMENT_LOAD_OP_LOAD
                                           : VK_ATTACHMENT_LOAD_OP_CLEAR;
                att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                // Determine final layout
                // Check if any later pass uses this resource
                bool is_last_writer = true;
                VkImageLayout final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                for (size_t sj = my_pos + 1; sj < sorted_order_.size(); sj++) {
                    auto& later_pass = passes_[sorted_order_[sj]];
                    for (auto& in : later_pass->inputs) {
                        if (static_cast<uint32_t>(in.id) == res_idx) {
                            bool is_depth_res = isDepthFormat(info.format);
                            final_layout = getUsageInfo(in.usage, is_depth_res).layout;
                        }
                    }
                    for (auto& out2 : later_pass->outputs) {
                        if (static_cast<uint32_t>(out2.id) == res_idx) {
                            is_last_writer = false;
                        }
                    }
                }

                // If this is the last writer to a swapchain resource, end at PRESENT_SRC_KHR
                if (is_last_writer && info.is_swapchain) {
                    final_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                }

                if (load_existing) {
                    // The barrier will transition to COLOR_ATTACHMENT_OPTIMAL before we begin
                    att.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                } else {
                    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                }
                att.finalLayout = final_layout;

                VkAttachmentReference ref{};
                ref.attachment = static_cast<uint32_t>(attachments.size());
                ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                color_refs.push_back(ref);
            }

            attachments.push_back(att);
        }

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = static_cast<uint32_t>(color_refs.size());
        subpass.pColorAttachments = color_refs.empty() ? nullptr : color_refs.data();
        subpass.pDepthStencilAttachment = has_depth ? &depth_ref : nullptr;

        // Subpass dependency
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dep.srcAccessMask = 0;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.attachmentCount = static_cast<uint32_t>(attachments.size());
        rp_info.pAttachments = attachments.data();
        rp_info.subpassCount = 1;
        rp_info.pSubpasses = &subpass;
        rp_info.dependencyCount = 1;
        rp_info.pDependencies = &dep;

        if (vkCreateRenderPass(device_.getHandle(), &rp_info, nullptr, &pass->render_pass) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass: " + pass->name);
        }

        // Track this pass's outputs
        for (auto& out : pass->outputs) {
            previously_written.insert(static_cast<uint32_t>(out.id));
        }
    }
}

void RenderGraphBuilder::createFramebuffers() {
    for (uint32_t idx : sorted_order_) {
        auto& pass = passes_[idx];
        if (pass->is_transfer_pass) continue;

        // Collect attachment views
        std::vector<VkImageView> views;
        bool has_swapchain_attachment = false;
        uint32_t swapchain_image_count = 0;

        for (auto& out : pass->outputs) {
            auto res_idx = static_cast<uint32_t>(out.id);
            auto& info = resources_[res_idx];
            if (info.is_swapchain) {
                has_swapchain_attachment = true;
                swapchain_image_count = info.swapchain->getImageCount();
                views.push_back(VK_NULL_HANDLE);  // placeholder
            } else {
                views.push_back(physical_resources_[res_idx].view);
            }
        }

        // Determine extent from first output
        VkExtent2D extent{};
        if (!pass->outputs.empty()) {
            auto res_idx = static_cast<uint32_t>(pass->outputs[0].id);
            auto& info = resources_[res_idx];
            if (info.is_swapchain) {
                extent = info.swapchain->getExtent();
            } else {
                extent = physical_resources_[res_idx].extent;
            }
        }
        pass->extent = extent;

        if (has_swapchain_attachment) {
            // Create per-swapchain-image framebuffers
            pass->per_image_framebuffers.resize(swapchain_image_count);
            for (uint32_t i = 0; i < swapchain_image_count; i++) {
                // Fill in swapchain view
                auto fb_views = views;
                for (size_t v = 0; v < pass->outputs.size(); v++) {
                    auto res_idx = static_cast<uint32_t>(pass->outputs[v].id);
                    if (resources_[res_idx].is_swapchain) {
                        fb_views[v] = resources_[res_idx].swapchain->getImageView(i);
                    }
                }

                VkFramebufferCreateInfo fb_info{};
                fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                fb_info.renderPass = pass->render_pass;
                fb_info.attachmentCount = static_cast<uint32_t>(fb_views.size());
                fb_info.pAttachments = fb_views.data();
                fb_info.width = extent.width;
                fb_info.height = extent.height;
                fb_info.layers = 1;

                if (vkCreateFramebuffer(device_.getHandle(), &fb_info, nullptr,
                                        &pass->per_image_framebuffers[i]) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create framebuffer: " + pass->name);
                }
            }
        } else {
            VkFramebufferCreateInfo fb_info{};
            fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb_info.renderPass = pass->render_pass;
            fb_info.attachmentCount = static_cast<uint32_t>(views.size());
            fb_info.pAttachments = views.data();
            fb_info.width = extent.width;
            fb_info.height = extent.height;
            fb_info.layers = 1;

            if (vkCreateFramebuffer(device_.getHandle(), &fb_info, nullptr, &pass->framebuffer) !=
                VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer: " + pass->name);
            }
        }
    }
}

void RenderGraphBuilder::computeBarriers() {
    struct LastUsage {
        ResourceUsage usage;
        bool valid = false;
    };
    std::vector<LastUsage> last_usage(resources_.size());
    std::unordered_set<uint32_t> previously_written;

    // Helper: build a barrier transitioning from last known usage to a target
    auto makeBarrier = [&](ResourceId id, VkImageLayout new_layout, VkAccessFlags dst_access,
                           VkPipelineStageFlags dst_stage) -> ImageBarrier {
        auto res_idx = static_cast<uint32_t>(id);
        bool depth = isDepthFormat(resources_[res_idx].format);
        ImageBarrier b{};
        b.resource = id;
        b.new_layout = new_layout;
        b.dst_access = dst_access;
        b.dst_stage = dst_stage;
        if (last_usage[res_idx].valid) {
            auto src = getUsageInfo(last_usage[res_idx].usage, depth);
            b.src_access = src.access;
            b.src_stage = src.stage;
            b.old_layout = src.layout;
        } else {
            b.src_access = 0;
            b.src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            b.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
        return b;
    };

    auto makeUsageBarrier = [&](ResourceId id, ResourceUsage usage) -> ImageBarrier {
        auto res_idx = static_cast<uint32_t>(id);
        bool depth = isDepthFormat(resources_[res_idx].format);
        auto target = getUsageInfo(usage, depth);
        return makeBarrier(id, target.layout, target.access, target.stage);
    };

    steps_.clear();
    for (size_t step_idx = 0; step_idx < sorted_order_.size(); step_idx++) {
        uint32_t pass_idx = sorted_order_[step_idx];
        auto& pass = passes_[pass_idx];

        ExecutionStep step{};
        step.pass_index = static_cast<uint32_t>(step_idx);

        // Barriers for inputs
        for (auto& in : pass->inputs) {
            step.pre_barriers.push_back(makeUsageBarrier(in.id, in.usage));
            last_usage[static_cast<uint32_t>(in.id)] = {in.usage, true};
        }

        // For render passes that load existing content, transition to COLOR_ATTACHMENT_OPTIMAL
        if (!pass->is_transfer_pass) {
            for (auto& out : pass->outputs) {
                auto res_idx = static_cast<uint32_t>(out.id);
                if (out.usage == ResourceUsage::ColorAttachmentWrite &&
                    previously_written.count(res_idx) > 0) {
                    step.pre_barriers.push_back(makeBarrier(
                        out.id, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT));
                }
            }
        }

        if (pass->is_transfer_pass) {
            for (auto& out : pass->outputs) {
                step.pre_barriers.push_back(makeUsageBarrier(out.id, out.usage));
                auto res_idx = static_cast<uint32_t>(out.id);
                last_usage[res_idx] = {out.usage, true};
                previously_written.insert(res_idx);
            }
        } else {
            for (auto& out : pass->outputs) {
                auto res_idx = static_cast<uint32_t>(out.id);
                last_usage[res_idx] = {out.usage, true};
                previously_written.insert(res_idx);
            }
        }

        steps_.push_back(std::move(step));
    }

    // Final barrier: transition swapchain to PRESENT_SRC_KHR if needed
    // Check if the last usage of any swapchain resource is not Present
    for (size_t i = 0; i < resources_.size(); i++) {
        if (!resources_[i].is_swapchain) continue;
        if (last_usage[i].valid && last_usage[i].usage != ResourceUsage::Present) {
            // The UI render pass should already transition to PRESENT_SRC_KHR via finalLayout.
            // If the last write is ColorAttachmentWrite via a render pass, the render pass
            // handles it. So we typically don't need an explicit barrier here.
        }
    }
}

}  // namespace engine
