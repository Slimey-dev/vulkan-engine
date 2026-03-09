#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine {

class VulkanDevice;
struct Vertex;

class Mesh {
public:
    Mesh(VulkanDevice& device, VkCommandPool command_pool,
         const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);
    ~Mesh();

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    void bind(VkCommandBuffer cmd) const;
    void draw(VkCommandBuffer cmd) const;

    static std::unique_ptr<Mesh> loadFromOBJ(VulkanDevice& device, VkCommandPool command_pool,
                                             const std::string& filepath);

private:
    VulkanDevice& device_;
    VkBuffer vertex_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertex_buffer_memory_ = VK_NULL_HANDLE;
    VkBuffer index_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory index_buffer_memory_ = VK_NULL_HANDLE;
    uint32_t index_count_ = 0;
};

}  // namespace engine
