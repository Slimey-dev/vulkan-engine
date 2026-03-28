#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace engine {

struct AcousticMesh;

class SpatialAudio {
public:
    SpatialAudio();
    ~SpatialAudio();

    SpatialAudio(const SpatialAudio&) = delete;
    SpatialAudio& operator=(const SpatialAudio&) = delete;

    bool isAvailable() const;

    void buildScene(std::span<const AcousticMesh*> meshes);
    void clearScene();

    void setListener(glm::vec3 position, glm::vec3 forward, glm::vec3 up);

    uint32_t createSource(glm::vec3 position);
    void updateSource(uint32_t id, glm::vec3 position);
    void removeSource(uint32_t id);

    // Process mono input through HRTF + reverb → stereo output
    void processBinaural(uint32_t source_id, const float* in_mono, float* out_stereo,
                         uint32_t frame_count);

    // Get the persistent listener source (always has a valid IR for reverb)
    uint32_t getListenerSourceId() const;

    // Run reflection simulation (call once per frame after updating sources)
    void simulate();

    // Debug controls
    bool enabled = true;
    float reverb_mix = 1.0f;
    bool hrtf_enabled = true;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine
