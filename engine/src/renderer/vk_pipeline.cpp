#include <engine/core/log.hpp>
#include <engine/renderer/vk_pipeline.hpp>

#include <fstream>
#include <stdexcept>

namespace engine {

VulkanPipeline::~VulkanPipeline() {
    vkDestroyPipeline(device_, pipeline_, nullptr);
    vkDestroyPipelineLayout(device_, layout_, nullptr);
}

VulkanPipeline::VulkanPipeline(
    VkDevice device, VkRenderPass render_pass, const std::string& vert_path,
    const std::string& frag_path,
    const std::vector<VkVertexInputBindingDescription>& bindings,
    const std::vector<VkVertexInputAttributeDescription>& attributes,
    VkDescriptorSetLayout descriptor_set_layout, const PipelineConfig& config)
    : device_(device) {
    auto vert_code = readFile(vert_path);
    auto frag_code = readFile(frag_path);

    VkShaderModule vert_module = createShaderModule(vert_code);
    VkShaderModule frag_module = createShaderModule(frag_code);

    VkPipelineShaderStageCreateInfo vert_stage{};
    vert_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage.module = vert_module;
    vert_stage.pName = "main";

    VkPipelineShaderStageCreateInfo frag_stage{};
    frag_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage.module = frag_module;
    frag_stage.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vert_stage, frag_stage};

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
    vertex_input.pVertexBindingDescriptions = bindings.data();
    vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = config.cull_mode;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    if (config.depth_bias) {
        rasterizer.depthBiasEnable = VK_TRUE;
        rasterizer.depthBiasConstantFactor = config.depth_bias_constant;
        rasterizer.depthBiasSlopeFactor = config.depth_bias_slope;
    }

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    if (config.has_color_attachment) {
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;
    }

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (descriptor_set_layout != VK_NULL_HANDLE) {
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &descriptor_set_layout;
    }

    VkPushConstantRange push_range{};
    if (config.has_push_constants) {
        push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(float) * 16;  // mat4
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_range;
    }

    if (vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, vert_module, nullptr);
        vkDestroyShaderModule(device_, frag_module, nullptr);
        throw std::runtime_error("Failed to create pipeline layout");
    }

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = config.depth_write ? VK_TRUE : VK_FALSE;
    depth_stencil.depthCompareOp = config.depth_compare_op;

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = layout_;
    pipeline_info.renderPass = render_pass;
    pipeline_info.subpass = 0;

    if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                  &pipeline_) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, vert_module, nullptr);
        vkDestroyShaderModule(device_, frag_module, nullptr);
        throw std::runtime_error("Failed to create graphics pipeline");
    }

    vkDestroyShaderModule(device_, vert_module, nullptr);
    vkDestroyShaderModule(device_, frag_module, nullptr);

    logInfo("Graphics pipeline created");
}

std::vector<char> VulkanPipeline::readFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filepath);
    }
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    return buffer;
}

VkShaderModule VulkanPipeline::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module;
    if (vkCreateShaderModule(device_, &info, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    return module;
}

}  // namespace engine
