#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace engine {

class VulkanDevice;

struct PassTiming {
    std::string name;
    float ms = 0.0f;
    float percentage = 0.0f;
};

class GpuProfiler {
public:
    static constexpr uint32_t MAX_PASSES = 16;
    static constexpr uint32_t QUERIES_PER_PASS = 2;
    static constexpr uint32_t MAX_QUERIES_PER_FRAME = MAX_PASSES * QUERIES_PER_PASS;
    static constexpr uint32_t MAX_FRAMES = 2;
    static constexpr size_t HISTORY_SIZE = 200;

    GpuProfiler(VulkanDevice& device);
    ~GpuProfiler();

    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;

    void beginFrame(uint32_t frame_index);
    void beginPass(VkCommandBuffer cmd, uint32_t frame_index, uint32_t pass_index,
                   const std::string& name);
    void endPass(VkCommandBuffer cmd, uint32_t frame_index, uint32_t pass_index);

    const std::vector<PassTiming>& getPassTimings() const { return pass_timings_; }
    float getTotalGpuTimeMs() const { return total_gpu_time_ms_; }
    const std::array<float, HISTORY_SIZE>& getHistoryMs() const { return history_; }
    size_t getHistoryOffset() const { return history_write_ % HISTORY_SIZE; }

private:
    VkDevice device_;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    float ns_per_tick_ = 0.0f;
    uint64_t timestamp_mask_ = 0;

    std::vector<PassTiming> pass_timings_;
    float total_gpu_time_ms_ = 0.0f;

    std::array<float, HISTORY_SIZE> history_{};
    size_t history_write_ = 0;

    std::array<uint64_t, MAX_QUERIES_PER_FRAME> timestamp_readback_{};
    std::array<uint32_t, MAX_FRAMES> pass_count_{};
    std::array<std::vector<std::string>, MAX_FRAMES> pass_names_;
    uint32_t frames_seen_ = 0;
};

}  // namespace engine
