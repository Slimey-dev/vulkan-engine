#include <engine/renderer/gpu_profiler.hpp>
#include <engine/renderer/vk_device.hpp>

#include <stdexcept>

namespace engine {

GpuProfiler::GpuProfiler(VulkanDevice& device) : device_(device.getHandle()) {
    ns_per_tick_ = device.getTimestampPeriod();

    uint32_t valid_bits = device.getTimestampValidBits();
    timestamp_mask_ =
        (valid_bits >= 64) ? UINT64_MAX : ((1ULL << valid_bits) - 1);

    VkQueryPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = MAX_QUERIES_PER_FRAME * MAX_FRAMES;

    if (vkCreateQueryPool(device_, &ci, nullptr, &query_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create timestamp query pool");
    }

    vkResetQueryPool(device_, query_pool_, 0, MAX_QUERIES_PER_FRAME * MAX_FRAMES);
}

GpuProfiler::~GpuProfiler() {
    if (query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_, query_pool_, nullptr);
    }
}

void GpuProfiler::beginFrame(uint32_t frame_index) {
    uint32_t count = pass_count_[frame_index];
    uint32_t base = frame_index * MAX_QUERIES_PER_FRAME;
    uint32_t query_count = count * QUERIES_PER_PASS;

    if (count > 0 && frames_seen_ >= MAX_FRAMES) {
        VkResult res = vkGetQueryPoolResults(
            device_, query_pool_, base, query_count,
            query_count * sizeof(uint64_t), timestamp_readback_.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);

        if (res == VK_SUCCESS) {
            pass_timings_.clear();
            pass_timings_.reserve(count);
            total_gpu_time_ms_ = 0.0f;

            for (uint32_t i = 0; i < count; i++) {
                uint64_t begin_tick = timestamp_readback_[i * 2];
                uint64_t end_tick = timestamp_readback_[i * 2 + 1];
                uint64_t delta = (end_tick - begin_tick) & timestamp_mask_;
                float ms = static_cast<float>(delta) * ns_per_tick_ / 1e6f;

                PassTiming timing;
                timing.name = pass_names_[frame_index][i];
                timing.ms = ms;
                pass_timings_.push_back(timing);
                total_gpu_time_ms_ += ms;
            }

            for (auto& t : pass_timings_) {
                t.percentage =
                    (total_gpu_time_ms_ > 0.0f) ? (t.ms / total_gpu_time_ms_) * 100.0f : 0.0f;
            }

            history_[history_write_ % HISTORY_SIZE] = total_gpu_time_ms_;
            history_write_++;
        }
    }

    if (query_count > 0) {
        vkResetQueryPool(device_, query_pool_, base, query_count);
    }

    pass_count_[frame_index] = 0;
    pass_names_[frame_index].clear();
    frames_seen_++;
}

void GpuProfiler::beginPass(VkCommandBuffer cmd, uint32_t frame_index, uint32_t pass_index,
                             const std::string& name) {
    uint32_t query = frame_index * MAX_QUERIES_PER_FRAME + pass_index * QUERIES_PER_PASS;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_NONE, query_pool_, query);

    if (pass_index >= pass_names_[frame_index].size()) {
        pass_names_[frame_index].resize(pass_index + 1);
    }
    pass_names_[frame_index][pass_index] = name;
    pass_count_[frame_index] = pass_index + 1;
}

void GpuProfiler::endPass(VkCommandBuffer cmd, uint32_t frame_index, uint32_t pass_index) {
    uint32_t query = frame_index * MAX_QUERIES_PER_FRAME + pass_index * QUERIES_PER_PASS + 1;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, query_pool_, query);
}

}  // namespace engine
