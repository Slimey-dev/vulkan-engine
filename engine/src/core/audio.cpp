#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <engine/core/audio.hpp>
#include <engine/core/log.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace engine {

static constexpr uint32_t kSampleRate = 44100;
static constexpr float kPi = 3.14159265358979f;
static constexpr int kJumpSlot = 0;
static constexpr int kLandSlot = 1;

struct Audio::Impl {
    ma_engine engine;
    ma_audio_buffer buffers[2];
    ma_sound sounds[2];
    bool ready[2] = {false, false};
    std::vector<float> samples[2];
};

Audio::Audio() : impl_(std::make_unique<Impl>()) {
    ma_engine_config config = ma_engine_config_init();
    config.channels = 1;
    config.sampleRate = kSampleRate;

    if (ma_engine_init(&config, &impl_->engine) != MA_SUCCESS) {
        impl_.reset();
        throw std::runtime_error("Failed to initialize audio engine");
    }

    generateSound(
        {.freq_start = 40.0f,
         .freq_end = 100.0f,
         .duration = 0.15f,
         .noise_amount = 0.3f,
         .square_amplitude = 0.4f,
         .noise_seed = 12345,
         .envelope = SoundParams::Envelope::Linear},
        kJumpSlot);

    generateSound(
        {.freq_start = 80.0f,
         .freq_end = 30.0f,
         .duration = 0.12f,
         .noise_amount = 0.5f,
         .square_amplitude = 0.5f,
         .noise_seed = 67890,
         .envelope = SoundParams::Envelope::Cubic},
        kLandSlot);

    logInfo("Audio initialized");
}

Audio::~Audio() {
    if (!impl_) return;
    for (int i = 0; i < 2; i++) {
        if (impl_->ready[i]) {
            ma_sound_uninit(&impl_->sounds[i]);
            ma_audio_buffer_uninit(&impl_->buffers[i]);
        }
    }
    ma_engine_uninit(&impl_->engine);
}

void Audio::playJump() {
    if (!impl_ || !impl_->ready[kJumpSlot]) return;
    ma_sound_seek_to_pcm_frame(&impl_->sounds[kJumpSlot], 0);
    ma_sound_start(&impl_->sounds[kJumpSlot]);
}

void Audio::playLand() {
    if (!impl_ || !impl_->ready[kLandSlot]) return;
    ma_sound_seek_to_pcm_frame(&impl_->sounds[kLandSlot], 0);
    ma_sound_start(&impl_->sounds[kLandSlot]);
}

void Audio::generateSound(const SoundParams& params, int slot) {
    uint32_t num_samples = static_cast<uint32_t>(kSampleRate * params.duration);
    auto& samples = impl_->samples[slot];
    samples.resize(num_samples);

    float phase = 0.0f;
    uint32_t noise_state = params.noise_seed;

    for (uint32_t i = 0; i < num_samples; i++) {
        float t = static_cast<float>(i) / static_cast<float>(num_samples);
        float freq = params.freq_start + (params.freq_end - params.freq_start) * t;

        float square =
            std::sin(2.0f * kPi * phase) > 0.0f ? params.square_amplitude : -params.square_amplitude;
        noise_state = noise_state * 1103515245 + 12345;
        float noise =
            (static_cast<float>(noise_state >> 16) / 32768.0f - 1.0f) * params.noise_amount;
        float sample = square + noise * (1.0f - t * 0.5f);

        float envelope;
        if (params.envelope == SoundParams::Envelope::Cubic) {
            envelope = (1.0f - t) * (1.0f - t) * (1.0f - t);
        } else {
            envelope = 1.0f - t * 0.6f;
        }
        samples[i] = sample * envelope;

        phase += freq / static_cast<float>(kSampleRate);
        if (phase >= 1.0f) phase -= 1.0f;
    }

    ma_audio_buffer_config buf_config =
        ma_audio_buffer_config_init(ma_format_f32, 1, num_samples, samples.data(), nullptr);
    buf_config.sampleRate = kSampleRate;

    if (ma_audio_buffer_init(&buf_config, &impl_->buffers[slot]) == MA_SUCCESS) {
        if (ma_sound_init_from_data_source(
                &impl_->engine, &impl_->buffers[slot],
                MA_SOUND_FLAG_NO_PITCH | MA_SOUND_FLAG_NO_SPATIALIZATION, nullptr,
                &impl_->sounds[slot]) == MA_SUCCESS) {
            ma_sound_set_volume(&impl_->sounds[slot], 0.15f);
            impl_->ready[slot] = true;
        }
    }
}

}  // namespace engine
