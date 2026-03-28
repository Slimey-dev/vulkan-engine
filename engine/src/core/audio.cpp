#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <engine/core/audio.hpp>
#include <engine/core/log.hpp>
#include <engine/core/spatial_audio.hpp>

#include <glm/gtc/constants.hpp>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace engine {

static constexpr uint32_t kSampleRate = 44100;
static constexpr uint32_t kFrameSize = 1024;
static constexpr int kJumpSlot = 0;
static constexpr int kLandSlot = 1;
static constexpr int kNumSlots = 2;

// Custom node that pipes mono audio through Steam Audio's binaural processing
struct SteamAudioNode {
    ma_node_base base;
    SpatialAudio* spatial;
    uint32_t source_id;
};

static ma_node_vtable steam_audio_node_vtable;
static bool vtable_initialized = false;

static void steam_audio_node_process(ma_node* pNode, const float** ppFramesIn,
                                     ma_uint32* pFrameCountIn, float** ppFramesOut,
                                     ma_uint32* pFrameCountOut) {
    (void)pFrameCountIn;
    auto* node = reinterpret_cast<SteamAudioNode*>(pNode);
    uint32_t frames = *pFrameCountOut;

    if (node->spatial && node->source_id != 0) {
        // Input is mono (interleaved, 1 channel), output is stereo (interleaved, 2 channels)
        // Steam Audio processBinaural expects mono in, interleaved stereo out
        node->spatial->processBinaural(node->source_id, ppFramesIn[0], ppFramesOut[0], frames);
    } else {
        // Passthrough: mono → stereo
        for (uint32_t i = 0; i < frames; i++) {
            ppFramesOut[0][i * 2] = ppFramesIn[0][i];
            ppFramesOut[0][i * 2 + 1] = ppFramesIn[0][i];
        }
    }
}

static void initVtable() {
    if (vtable_initialized) return;
    steam_audio_node_vtable.onProcess = steam_audio_node_process;
    steam_audio_node_vtable.inputBusCount = 1;
    steam_audio_node_vtable.outputBusCount = 1;
    steam_audio_node_vtable.flags = 0;
    vtable_initialized = true;
}

// Reverb tail needs silence after the dry sound to decay naturally
static constexpr float kReverbTailSeconds = 2.0f;
static constexpr uint32_t kReverbTailSamples =
    static_cast<uint32_t>(kReverbTailSeconds * kSampleRate);

struct SpatializedSound {
    SteamAudioNode node;
    ma_audio_buffer buffer;
    ma_sound sound;
    std::vector<float> padded_samples;  // sound + silence for reverb tail
    uint32_t spatial_source_id = 0;
    bool active = false;
};

struct Audio::Impl {
    ma_engine engine;
    ma_audio_buffer buffers[kNumSlots];
    ma_sound sounds[kNumSlots];
    bool ready[kNumSlots] = {};
    std::vector<float> samples[kNumSlots];
    SpatialAudio* spatial = nullptr;

    // Pool of spatialized sounds for one-shot effects
    static constexpr int kMaxSpatialSounds = 8;
    SpatializedSound spatialized[kMaxSpatialSounds] = {};
};

Audio::Audio() : impl_(std::make_unique<Impl>()) {
    initVtable();

    ma_engine_config config = ma_engine_config_init();
    config.channels = 2;  // stereo output for HRTF
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

    logInfo("Audio initialized (stereo output)");
}

Audio::~Audio() {
    if (!impl_) return;

    // Clean up spatialized sounds (they don't own the Steam Audio source)
    for (int i = 0; i < Impl::kMaxSpatialSounds; i++) {
        auto& sp = impl_->spatialized[i];
        if (sp.active) {
            ma_sound_uninit(&sp.sound);
            ma_node_uninit(&sp.node.base, nullptr);
            ma_audio_buffer_uninit(&sp.buffer);
            sp.active = false;
        }
    }

    for (int i = 0; i < kNumSlots; i++) {
        if (impl_->ready[i]) {
            ma_sound_uninit(&impl_->sounds[i]);
            ma_audio_buffer_uninit(&impl_->buffers[i]);
        }
    }
    ma_engine_uninit(&impl_->engine);
}

void Audio::setSpatialAudio(SpatialAudio* spatial) {
    if (impl_) impl_->spatial = spatial;
}

void Audio::playSoundAtPosition(int slot, glm::vec3 position) {
    if (!impl_ || slot < 0 || slot >= kNumSlots || !impl_->ready[slot]) return;

    if (impl_->spatial && impl_->spatial->isAvailable() &&
        impl_->spatial->getListenerSourceId() != 0) {
        playSound(slot, position);
    } else {
        ma_sound_seek_to_pcm_frame(&impl_->sounds[slot], 0);
        ma_sound_start(&impl_->sounds[slot]);
    }
}

void Audio::playJump(glm::vec3 position) { playSoundAtPosition(kJumpSlot, position); }
void Audio::playLand(glm::vec3 position) { playSoundAtPosition(kLandSlot, position); }

uint32_t Audio::playSound(int slot, glm::vec3 position) {
    if (!impl_ || slot < 0 || slot >= kNumSlots || !impl_->ready[slot]) return 0;
    if (!impl_->spatial || !impl_->spatial->isAvailable()) return 0;

    // Use the persistent listener source — it always has a pre-computed IR
    uint32_t source_id = impl_->spatial->getListenerSourceId();
    if (source_id == 0) return 0;

    // Update source position to where the sound is emitted
    impl_->spatial->updateSource(source_id, position);

    // Find a free spatialized sound slot (or reuse one that's done playing)
    int free_idx = -1;
    for (int i = 0; i < Impl::kMaxSpatialSounds; i++) {
        auto& sp = impl_->spatialized[i];
        if (!sp.active) {
            free_idx = i;
            break;
        }
        if (!ma_sound_is_playing(&sp.sound)) {
            ma_sound_uninit(&sp.sound);
            ma_node_uninit(&sp.node.base, nullptr);
            ma_audio_buffer_uninit(&sp.buffer);
            sp.active = false;
            free_idx = i;
            break;
        }
    }

    if (free_idx < 0) return 0;

    auto& sp = impl_->spatialized[free_idx];
    sp.spatial_source_id = source_id;

    uint32_t num_samples = static_cast<uint32_t>(impl_->samples[slot].size());
    uint32_t padded_size = num_samples + kReverbTailSamples;
    sp.padded_samples.resize(padded_size);
    std::memcpy(sp.padded_samples.data(), impl_->samples[slot].data(),
                num_samples * sizeof(float));
    std::memset(sp.padded_samples.data() + num_samples, 0, kReverbTailSamples * sizeof(float));

    ma_audio_buffer_config buf_config =
        ma_audio_buffer_config_init(ma_format_f32, 1, padded_size, sp.padded_samples.data(),
                                    nullptr);
    buf_config.sampleRate = kSampleRate;

    if (ma_audio_buffer_init(&buf_config, &sp.buffer) != MA_SUCCESS) return 0;

    // Create custom node: 1ch mono input → 2ch stereo output
    ma_node_graph* graph = ma_engine_get_node_graph(&impl_->engine);

    ma_uint32 input_channels[] = {1};
    ma_uint32 output_channels[] = {2};
    ma_node_config node_config = ma_node_config_init();
    node_config.vtable = &steam_audio_node_vtable;
    node_config.pInputChannels = input_channels;
    node_config.pOutputChannels = output_channels;
    node_config.inputBusCount = 1;
    node_config.outputBusCount = 1;

    if (ma_node_init(graph, &node_config, nullptr, &sp.node.base) != MA_SUCCESS) {
        ma_audio_buffer_uninit(&sp.buffer);
        return 0;
    }

    sp.node.spatial = impl_->spatial;
    sp.node.source_id = source_id;

    // Connect node output to engine endpoint
    ma_node* endpoint = ma_engine_get_endpoint(&impl_->engine);
    ma_node_attach_output_bus(&sp.node.base, 0, endpoint, 0);

    // Create sound routed through our custom node
    if (ma_sound_init_from_data_source(&impl_->engine, &sp.buffer,
                                       MA_SOUND_FLAG_NO_PITCH | MA_SOUND_FLAG_NO_SPATIALIZATION,
                                       nullptr, &sp.sound) != MA_SUCCESS) {
        ma_node_uninit(&sp.node.base, nullptr);
        ma_audio_buffer_uninit(&sp.buffer);
        return 0;
    }

    ma_node_attach_output_bus(&sp.sound, 0, &sp.node.base, 0);
    ma_sound_set_volume(&sp.sound, 0.15f);
    sp.active = true;

    ma_sound_start(&sp.sound);
    return source_id;
}

void Audio::updateSoundPosition(uint32_t handle, glm::vec3 position) {
    if (!impl_ || !impl_->spatial || handle == 0) return;
    impl_->spatial->updateSource(handle, position);
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
            std::sin(2.0f * glm::pi<float>() * phase) > 0.0f ? params.square_amplitude : -params.square_amplitude;
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
