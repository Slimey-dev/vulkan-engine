#pragma once

#include <memory>

namespace engine {

class Audio {
public:
    Audio();
    ~Audio();

    Audio(const Audio&) = delete;
    Audio& operator=(const Audio&) = delete;

    void playJump();
    void playLand();

private:
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
