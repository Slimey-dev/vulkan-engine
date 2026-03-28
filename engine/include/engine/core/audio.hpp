#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>

namespace engine {

class SpatialAudio;

class Audio {
public:
    Audio();
    ~Audio();

    Audio(const Audio&) = delete;
    Audio& operator=(const Audio&) = delete;

    void setSpatialAudio(SpatialAudio* spatial);

    void playJump(glm::vec3 position);
    void playLand(glm::vec3 position);

    uint32_t playSound(int slot, glm::vec3 position);
    void updateSoundPosition(uint32_t handle, glm::vec3 position);

private:
    void playSoundAtPosition(int slot, glm::vec3 position);
    struct SoundParams {
        float freq_start;
        float freq_end;
        float duration;
        float noise_amount;
        float square_amplitude;
        uint32_t noise_seed;
        enum class Envelope { Linear, Cubic } envelope;
    };

    void generateSound(const SoundParams& params, int slot);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine
