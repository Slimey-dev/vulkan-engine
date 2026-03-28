#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <engine/core/log.hpp>
#include <engine/renderer/mesh.hpp>
#include <engine/renderer/vk_buffer.hpp>
#include <engine/renderer/vk_device.hpp>

#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace engine {

Mesh::Mesh(VulkanDevice& device, VkCommandPool command_pool,
           const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
           bool retain_acoustic_data)
    : device_(device), index_count_(static_cast<uint32_t>(indices.size())) {
    if (retain_acoustic_data) {
        acoustic_mesh_ = std::make_unique<AcousticMesh>();
        acoustic_mesh_->positions.reserve(vertices.size());
        for (const auto& v : vertices) {
            acoustic_mesh_->positions.push_back(v.position);
        }
        acoustic_mesh_->indices = indices;
    }

    // Vertex buffer via staging
    VkDeviceSize vb_size = sizeof(Vertex) * vertices.size();
    VkBuffer staging;
    VkDeviceMemory staging_mem;

    vk_buffer::createBuffer(device_, vb_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            staging, staging_mem);

    void* data;
    vkMapMemory(device_.getHandle(), staging_mem, 0, vb_size, 0, &data);
    std::memcpy(data, vertices.data(), vb_size);
    vkUnmapMemory(device_.getHandle(), staging_mem);

    vk_buffer::createBuffer(device_, vb_size,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer_,
                            vertex_buffer_memory_);
    vk_buffer::copyBuffer(device_, command_pool, staging, vertex_buffer_, vb_size);

    vkDestroyBuffer(device_.getHandle(), staging, nullptr);
    vkFreeMemory(device_.getHandle(), staging_mem, nullptr);

    // Index buffer via staging
    VkDeviceSize ib_size = sizeof(uint32_t) * indices.size();

    vk_buffer::createBuffer(device_, ib_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            staging, staging_mem);

    vkMapMemory(device_.getHandle(), staging_mem, 0, ib_size, 0, &data);
    std::memcpy(data, indices.data(), ib_size);
    vkUnmapMemory(device_.getHandle(), staging_mem);

    vk_buffer::createBuffer(device_, ib_size,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer_,
                            index_buffer_memory_);
    vk_buffer::copyBuffer(device_, command_pool, staging, index_buffer_, ib_size);

    vkDestroyBuffer(device_.getHandle(), staging, nullptr);
    vkFreeMemory(device_.getHandle(), staging_mem, nullptr);
}

Mesh::~Mesh() {
    vkDestroyBuffer(device_.getHandle(), index_buffer_, nullptr);
    vkFreeMemory(device_.getHandle(), index_buffer_memory_, nullptr);
    vkDestroyBuffer(device_.getHandle(), vertex_buffer_, nullptr);
    vkFreeMemory(device_.getHandle(), vertex_buffer_memory_, nullptr);
}

void Mesh::bind(VkCommandBuffer cmd) const {
    VkBuffer buffers[] = {vertex_buffer_};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
    vkCmdBindIndexBuffer(cmd, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
}

void Mesh::draw(VkCommandBuffer cmd) const {
    vkCmdDrawIndexed(cmd, index_count_, 1, 0, 0, 0);
}

struct IndexKey {
    int vertex_index;
    int normal_index;
    int texcoord_index;
    bool operator==(const IndexKey&) const = default;
};

struct IndexKeyHash {
    size_t operator()(const IndexKey& k) const {
        size_t h = std::hash<int>{}(k.vertex_index);
        h ^= std::hash<int>{}(k.normal_index) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.texcoord_index) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

std::unique_ptr<Mesh> Mesh::loadFromOBJ(VulkanDevice& device, VkCommandPool command_pool,
                                        const std::string& filepath,
                                        bool retain_acoustic_data) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str())) {
        throw std::runtime_error("Failed to load OBJ: " + filepath + " " + err);
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::unordered_map<IndexKey, uint32_t, IndexKeyHash> unique_vertices;

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            IndexKey key{index.vertex_index, index.normal_index, index.texcoord_index};

            if (auto it = unique_vertices.find(key); it != unique_vertices.end()) {
                indices.push_back(it->second);
            } else {
                Vertex vertex{};

                vertex.position = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2],
                };

                if (index.texcoord_index >= 0) {
                    vertex.tex_coord = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
                    };
                }

                if (index.normal_index >= 0) {
                    vertex.normal = {
                        attrib.normals[3 * index.normal_index + 0],
                        attrib.normals[3 * index.normal_index + 1],
                        attrib.normals[3 * index.normal_index + 2],
                    };
                }

                if (!attrib.colors.empty()) {
                    vertex.color = {
                        attrib.colors[3 * index.vertex_index + 0],
                        attrib.colors[3 * index.vertex_index + 1],
                        attrib.colors[3 * index.vertex_index + 2],
                    };
                } else {
                    vertex.color = {1.0f, 1.0f, 1.0f};
                }

                auto new_index = static_cast<uint32_t>(vertices.size());
                unique_vertices[key] = new_index;
                vertices.push_back(vertex);
                indices.push_back(new_index);
            }
        }
    }

    logInfo("Loaded OBJ: {} ({} vertices, {} indices)", filepath, vertices.size(), indices.size());

    return std::make_unique<Mesh>(device, command_pool, vertices, indices, retain_acoustic_data);
}

}  // namespace engine
