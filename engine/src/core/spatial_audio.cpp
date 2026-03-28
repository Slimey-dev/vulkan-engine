#include <engine/core/spatial_audio.hpp>
#include <engine/core/log.hpp>
#include <engine/renderer/mesh.hpp>

#ifdef ENGINE_STEAM_AUDIO
#include <phonon.h>
#endif

#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace engine {

static constexpr int kSampleRate = 44100;
static constexpr int kFrameSize = 1024;
static constexpr int kMaxSources = 8;
static constexpr int kMaxReflectionOrder = 1;
static constexpr float kMaxDuration = 2.0f;
static constexpr int kNumRays = 4096;
static constexpr int kNumBounces = 8;
static constexpr int kNumDiffuseSamples = 32;

#ifdef ENGINE_STEAM_AUDIO

// Engine Z-up → Steam Audio Y-up coordinate transform
static IPLVector3 toIPL(glm::vec3 v) { return {v.x, v.z, v.y}; }

struct SourceData {
    IPLSource source = nullptr;
    IPLBinauralEffect binaural = nullptr;
    IPLDirectEffect direct = nullptr;
    IPLReflectionEffect reflection = nullptr;
    IPLAmbisonicsRotationEffect ambi_rotation = nullptr;
    IPLAmbisonicsBinauralEffect ambi_binaural = nullptr;
    glm::vec3 position{0};
};

struct SpatialAudio::Impl {
    IPLContext context = nullptr;
    IPLHRTF hrtf = nullptr;
    IPLSimulator simulator = nullptr;
    IPLScene scene = nullptr;
    IPLStaticMesh static_mesh = nullptr;

    IPLAudioSettings audio_settings{};

    std::unordered_map<uint32_t, SourceData> sources;
    uint32_t next_source_id = 1;
    uint32_t listener_source_id = 0;  // persistent source at listener position

    glm::vec3 listener_pos{0};
    glm::vec3 listener_forward{0, 1, 0};
    glm::vec3 listener_up{0, 0, 1};

    bool scene_built = false;

    // Background reflection thread
    std::thread reflection_thread;
    std::atomic<bool> reflection_running{false};
    std::atomic<bool> shutdown{false};
    std::mutex sim_mutex;  // protects simulator during source add/remove
    glm::vec3 last_reflection_pos{0};
    static constexpr float kReflectionUpdateDistance = 0.3f;

    // Pre-allocated audio buffers for processing
    IPLAudioBuffer mono_buffer{};
    IPLAudioBuffer stereo_buffer{};
    IPLAudioBuffer reflection_buffer{};
    std::vector<float> mono_data;
    std::vector<float> stereo_left;
    std::vector<float> stereo_right;
    static constexpr int kAmbiChannels = (kMaxReflectionOrder + 1) * (kMaxReflectionOrder + 1);
    std::vector<float> reflection_channels[kAmbiChannels];
    float* reflection_channel_ptrs[kAmbiChannels]{};
    std::vector<float> rotated_channels[kAmbiChannels];
    float* rotated_channel_ptrs[kAmbiChannels]{};
    std::vector<float> reverb_stereo_left;
    std::vector<float> reverb_stereo_right;

    void allocateBuffers() {
        mono_data.resize(kFrameSize, 0.0f);
        stereo_left.resize(kFrameSize, 0.0f);
        stereo_right.resize(kFrameSize, 0.0f);
        reverb_stereo_left.resize(kFrameSize, 0.0f);
        reverb_stereo_right.resize(kFrameSize, 0.0f);
        for (int i = 0; i < kAmbiChannels; i++) {
            reflection_channels[i].resize(kFrameSize, 0.0f);
            reflection_channel_ptrs[i] = reflection_channels[i].data();
            rotated_channels[i].resize(kFrameSize, 0.0f);
            rotated_channel_ptrs[i] = rotated_channels[i].data();
        }
    }
};

SpatialAudio::SpatialAudio() : impl_(std::make_unique<Impl>()) {
    IPLContextSettings ctx_settings{};
    ctx_settings.version = STEAMAUDIO_VERSION;

    if (iplContextCreate(&ctx_settings, &impl_->context) != IPL_STATUS_SUCCESS) {
        logError("Failed to create Steam Audio context");
        impl_.reset();
        return;
    }

    impl_->audio_settings.samplingRate = kSampleRate;
    impl_->audio_settings.frameSize = kFrameSize;

    IPLHRTFSettings hrtf_settings{};
    hrtf_settings.type = IPL_HRTFTYPE_DEFAULT;
    hrtf_settings.volume = 1.0f;

    if (iplHRTFCreate(impl_->context, &impl_->audio_settings, &hrtf_settings, &impl_->hrtf) !=
        IPL_STATUS_SUCCESS) {
        logError("Failed to create HRTF");
        iplContextRelease(&impl_->context);
        impl_.reset();
        return;
    }

    // Create simulator
    IPLSimulationSettings sim_settings{};
    sim_settings.flags =
        static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS);
    sim_settings.sceneType = IPL_SCENETYPE_DEFAULT;
    sim_settings.reflectionType = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
    sim_settings.maxNumOcclusionSamples = 16;
    sim_settings.maxNumRays = kNumRays;
    sim_settings.numDiffuseSamples = kNumDiffuseSamples;
    sim_settings.maxDuration = kMaxDuration;
    sim_settings.maxOrder = kMaxReflectionOrder;
    sim_settings.maxNumSources = kMaxSources;
    sim_settings.numThreads = 1;
    sim_settings.samplingRate = kSampleRate;
    sim_settings.frameSize = kFrameSize;

    if (iplSimulatorCreate(impl_->context, &sim_settings, &impl_->simulator) !=
        IPL_STATUS_SUCCESS) {
        logError("Failed to create Steam Audio simulator");
        iplHRTFRelease(&impl_->hrtf);
        iplContextRelease(&impl_->context);
        impl_.reset();
        return;
    }

    impl_->allocateBuffers();
    logInfo("Steam Audio initialized ({}Hz, {} frame size)", kSampleRate, kFrameSize);
}

SpatialAudio::~SpatialAudio() {
    if (!impl_) return;

    impl_->shutdown.store(true);
    if (impl_->reflection_thread.joinable()) {
        impl_->reflection_thread.join();
    }

    for (auto& [id, src] : impl_->sources) {
        if (src.ambi_binaural) iplAmbisonicsBinauralEffectRelease(&src.ambi_binaural);
        if (src.ambi_rotation) iplAmbisonicsRotationEffectRelease(&src.ambi_rotation);
        if (src.reflection) iplReflectionEffectRelease(&src.reflection);
        if (src.direct) iplDirectEffectRelease(&src.direct);
        if (src.binaural) iplBinauralEffectRelease(&src.binaural);
        if (src.source) {
            iplSourceRemove(src.source, impl_->simulator);
            iplSourceRelease(&src.source);
        }
    }

    if (impl_->static_mesh) iplStaticMeshRelease(&impl_->static_mesh);
    if (impl_->scene) iplSceneRelease(&impl_->scene);
    if (impl_->simulator) iplSimulatorRelease(&impl_->simulator);
    if (impl_->hrtf) iplHRTFRelease(&impl_->hrtf);
    if (impl_->context) iplContextRelease(&impl_->context);
}

bool SpatialAudio::isAvailable() const { return impl_ != nullptr; }

uint32_t SpatialAudio::getListenerSourceId() const {
    return impl_ ? impl_->listener_source_id : 0;
}

void SpatialAudio::buildScene(std::span<const AcousticMesh*> meshes) {
    if (!impl_) return;

    // Wait for any in-flight reflection simulation
    if (impl_->reflection_thread.joinable()) {
        impl_->reflection_thread.join();
    }

    clearScene();

    // Count total vertices and triangles
    int total_verts = 0;
    int total_tris = 0;
    for (const auto* mesh : meshes) {
        if (!mesh) continue;
        total_verts += static_cast<int>(mesh->positions.size());
        total_tris += static_cast<int>(mesh->indices.size()) / 3;
    }

    if (total_verts == 0 || total_tris == 0) return;

    // Create scene
    IPLSceneSettings scene_settings{};
    scene_settings.type = IPL_SCENETYPE_DEFAULT;

    if (iplSceneCreate(impl_->context, &scene_settings, &impl_->scene) != IPL_STATUS_SUCCESS) {
        logError("Failed to create acoustic scene");
        return;
    }

    // Build vertex and triangle arrays
    std::vector<IPLVector3> ipl_verts;
    std::vector<IPLTriangle> ipl_tris;
    std::vector<IPLint32> material_indices;
    ipl_verts.reserve(total_verts);
    ipl_tris.reserve(total_tris);
    material_indices.reserve(total_tris);

    int vertex_offset = 0;
    for (const auto* mesh : meshes) {
        if (!mesh) continue;

        for (const auto& pos : mesh->positions) {
            ipl_verts.push_back(toIPL(pos));
        }

        for (size_t i = 0; i + 2 < mesh->indices.size(); i += 3) {
            IPLTriangle tri;
            tri.indices[0] = static_cast<IPLint32>(mesh->indices[i] + vertex_offset);
            tri.indices[1] = static_cast<IPLint32>(mesh->indices[i + 1] + vertex_offset);
            tri.indices[2] = static_cast<IPLint32>(mesh->indices[i + 2] + vertex_offset);
            ipl_tris.push_back(tri);
            material_indices.push_back(0);
        }

        vertex_offset += static_cast<int>(mesh->positions.size());
    }

    // Concrete-like material for indoor walls
    IPLMaterial wall_material{};
    wall_material.absorption[0] = 0.05f;  // low freq
    wall_material.absorption[1] = 0.07f;  // mid freq
    wall_material.absorption[2] = 0.08f;  // high freq
    wall_material.scattering = 0.05f;
    wall_material.transmission[0] = 0.015f;
    wall_material.transmission[1] = 0.002f;
    wall_material.transmission[2] = 0.001f;

    IPLStaticMeshSettings mesh_settings{};
    mesh_settings.numVertices = static_cast<IPLint32>(ipl_verts.size());
    mesh_settings.numTriangles = static_cast<IPLint32>(ipl_tris.size());
    mesh_settings.numMaterials = 1;
    mesh_settings.vertices = ipl_verts.data();
    mesh_settings.triangles = ipl_tris.data();
    mesh_settings.materialIndices = material_indices.data();
    mesh_settings.materials = &wall_material;

    if (iplStaticMeshCreate(impl_->scene, &mesh_settings, &impl_->static_mesh) !=
        IPL_STATUS_SUCCESS) {
        logError("Failed to create static mesh for acoustic scene");
        iplSceneRelease(&impl_->scene);
        impl_->scene = nullptr;
        return;
    }

    iplStaticMeshAdd(impl_->static_mesh, impl_->scene);
    iplSceneCommit(impl_->scene);

    iplSimulatorSetScene(impl_->simulator, impl_->scene);
    iplSimulatorCommit(impl_->simulator);

    impl_->scene_built = true;

    // Create a persistent listener source so its IR is always pre-computed
    impl_->listener_source_id = createSource(impl_->listener_pos);

    logInfo("Acoustic scene built: {} vertices, {} triangles", total_verts, total_tris);
}

void SpatialAudio::clearScene() {
    if (!impl_) return;

    if (impl_->reflection_thread.joinable()) {
        impl_->reflection_thread.join();
    }

    // Remove persistent listener source
    if (impl_->listener_source_id != 0) {
        removeSource(impl_->listener_source_id);
        impl_->listener_source_id = 0;
    }

    if (impl_->static_mesh) {
        iplStaticMeshRemove(impl_->static_mesh, impl_->scene);
        iplStaticMeshRelease(&impl_->static_mesh);
        impl_->static_mesh = nullptr;
    }
    if (impl_->scene) {
        iplSceneRelease(&impl_->scene);
        impl_->scene = nullptr;
    }
    impl_->scene_built = false;
    impl_->last_reflection_pos = glm::vec3(0);
}

void SpatialAudio::setListener(glm::vec3 position, glm::vec3 forward, glm::vec3 up) {
    if (!impl_) return;
    impl_->listener_pos = position;
    impl_->listener_forward = forward;
    impl_->listener_up = up;

    // Keep the persistent listener source at the listener position
    if (impl_->listener_source_id != 0) {
        updateSource(impl_->listener_source_id, position);
    }
}

uint32_t SpatialAudio::createSource(glm::vec3 position) {
    if (!impl_) return 0;

    uint32_t id = impl_->next_source_id++;
    SourceData data;
    data.position = position;

    // Create simulation source
    IPLSourceSettings src_settings{};
    src_settings.flags =
        static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS);

    if (iplSourceCreate(impl_->simulator, &src_settings, &data.source) != IPL_STATUS_SUCCESS) {
        logError("Failed to create Steam Audio source");
        return 0;
    }
    iplSourceAdd(data.source, impl_->simulator);

    // Create binaural effect
    IPLBinauralEffectSettings binaural_settings{};
    binaural_settings.hrtf = impl_->hrtf;

    if (iplBinauralEffectCreate(impl_->context, &impl_->audio_settings, &binaural_settings,
                                &data.binaural) != IPL_STATUS_SUCCESS) {
        logError("Failed to create binaural effect");
        iplSourceRemove(data.source, impl_->simulator);
        iplSourceRelease(&data.source);
        return 0;
    }

    // Create direct effect
    IPLDirectEffectSettings direct_settings{};
    direct_settings.numChannels = 1;

    if (iplDirectEffectCreate(impl_->context, &impl_->audio_settings, &direct_settings,
                              &data.direct) != IPL_STATUS_SUCCESS) {
        logError("Failed to create direct effect");
        iplBinauralEffectRelease(&data.binaural);
        iplSourceRemove(data.source, impl_->simulator);
        iplSourceRelease(&data.source);
        return 0;
    }

    // Create reflection effect (convolution)
    IPLReflectionEffectSettings refl_settings{};
    refl_settings.type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
    refl_settings.irSize = static_cast<IPLint32>(kMaxDuration * kSampleRate);
    refl_settings.numChannels = Impl::kAmbiChannels;

    if (iplReflectionEffectCreate(impl_->context, &impl_->audio_settings, &refl_settings,
                                  &data.reflection) != IPL_STATUS_SUCCESS) {
        logError("Failed to create reflection effect");
        iplDirectEffectRelease(&data.direct);
        iplBinauralEffectRelease(&data.binaural);
        iplSourceRemove(data.source, impl_->simulator);
        iplSourceRelease(&data.source);
        return 0;
    }

    // Create ambisonics rotation effect (world space → listener space)
    IPLAmbisonicsRotationEffectSettings rot_settings{};
    rot_settings.maxOrder = kMaxReflectionOrder;

    if (iplAmbisonicsRotationEffectCreate(impl_->context, &impl_->audio_settings, &rot_settings,
                                           &data.ambi_rotation) != IPL_STATUS_SUCCESS) {
        logError("Failed to create ambisonics rotation effect");
        iplReflectionEffectRelease(&data.reflection);
        iplDirectEffectRelease(&data.direct);
        iplBinauralEffectRelease(&data.binaural);
        iplSourceRemove(data.source, impl_->simulator);
        iplSourceRelease(&data.source);
        return 0;
    }

    // Create ambisonics binaural effect (ambisonics → binaural stereo via HRTF)
    IPLAmbisonicsBinauralEffectSettings ambi_bin_settings{};
    ambi_bin_settings.hrtf = impl_->hrtf;
    ambi_bin_settings.maxOrder = kMaxReflectionOrder;

    if (iplAmbisonicsBinauralEffectCreate(impl_->context, &impl_->audio_settings,
                                           &ambi_bin_settings, &data.ambi_binaural) !=
        IPL_STATUS_SUCCESS) {
        logError("Failed to create ambisonics binaural effect");
        iplAmbisonicsRotationEffectRelease(&data.ambi_rotation);
        iplReflectionEffectRelease(&data.reflection);
        iplDirectEffectRelease(&data.direct);
        iplBinauralEffectRelease(&data.binaural);
        iplSourceRemove(data.source, impl_->simulator);
        iplSourceRelease(&data.source);
        return 0;
    }

    impl_->sources[id] = data;
    {
        std::lock_guard lock(impl_->sim_mutex);
        iplSimulatorCommit(impl_->simulator);
    }
    return id;
}

void SpatialAudio::updateSource(uint32_t id, glm::vec3 position) {
    if (!impl_) return;
    auto it = impl_->sources.find(id);
    if (it == impl_->sources.end()) return;
    it->second.position = position;
}

void SpatialAudio::removeSource(uint32_t id) {
    if (!impl_) return;
    auto it = impl_->sources.find(id);
    if (it == impl_->sources.end()) return;

    auto& src = it->second;
    if (src.ambi_binaural) iplAmbisonicsBinauralEffectRelease(&src.ambi_binaural);
    if (src.ambi_rotation) iplAmbisonicsRotationEffectRelease(&src.ambi_rotation);
    if (src.reflection) iplReflectionEffectRelease(&src.reflection);
    if (src.direct) iplDirectEffectRelease(&src.direct);
    if (src.binaural) iplBinauralEffectRelease(&src.binaural);
    if (src.source) {
        iplSourceRemove(src.source, impl_->simulator);
        iplSourceRelease(&src.source);
    }
    impl_->sources.erase(it);
    {
        std::lock_guard lock(impl_->sim_mutex);
        iplSimulatorCommit(impl_->simulator);
    }
}

void SpatialAudio::processBinaural(uint32_t source_id, const float* in_mono, float* out_stereo,
                                    uint32_t frame_count) {
    if (!impl_ || !enabled) {
        // Passthrough: copy mono to both stereo channels
        for (uint32_t i = 0; i < frame_count; i++) {
            out_stereo[i * 2] = in_mono[i];
            out_stereo[i * 2 + 1] = in_mono[i];
        }
        return;
    }

    auto it = impl_->sources.find(source_id);
    if (it == impl_->sources.end()) {
        for (uint32_t i = 0; i < frame_count; i++) {
            out_stereo[i * 2] = in_mono[i];
            out_stereo[i * 2 + 1] = in_mono[i];
        }
        return;
    }

    auto& src = it->second;
    uint32_t process_frames = std::min(frame_count, static_cast<uint32_t>(kFrameSize));

    // Set up deinterleaved mono input
    std::memcpy(impl_->mono_data.data(), in_mono, process_frames * sizeof(float));
    float* mono_ptr = impl_->mono_data.data();
    IPLAudioBuffer mono_buf{};
    mono_buf.numChannels = 1;
    mono_buf.numSamples = static_cast<IPLint32>(process_frames);
    mono_buf.data = &mono_ptr;

    // Apply direct effect (distance attenuation + occlusion)
    IPLSimulationOutputs outputs{};
    iplSourceGetOutputs(src.source,
                        static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_DIRECT |
                                                         IPL_SIMULATIONFLAGS_REFLECTIONS),
                        &outputs);

    IPLAudioBuffer direct_out{};
    float* direct_ptr = impl_->mono_data.data();  // in-place is OK for direct effect
    direct_out.numChannels = 1;
    direct_out.numSamples = static_cast<IPLint32>(process_frames);
    direct_out.data = &direct_ptr;

    iplDirectEffectApply(src.direct, &outputs.direct, &mono_buf, &direct_out);

    // Apply binaural HRTF
    float* stereo_ptrs[2] = {impl_->stereo_left.data(), impl_->stereo_right.data()};
    IPLAudioBuffer stereo_buf{};
    stereo_buf.numChannels = 2;
    stereo_buf.numSamples = static_cast<IPLint32>(process_frames);
    stereo_buf.data = stereo_ptrs;

    if (hrtf_enabled) {
        IPLVector3 dir = iplCalculateRelativeDirection(
            impl_->context, toIPL(src.position), toIPL(impl_->listener_pos),
            toIPL(impl_->listener_forward), toIPL(impl_->listener_up));

        IPLBinauralEffectParams binaural_params{};
        binaural_params.direction = dir;
        binaural_params.interpolation = IPL_HRTFINTERPOLATION_NEAREST;
        binaural_params.spatialBlend = 1.0f;
        binaural_params.hrtf = impl_->hrtf;

        iplBinauralEffectApply(src.binaural, &binaural_params, &direct_out, &stereo_buf);
    } else {
        // No HRTF: just duplicate mono to stereo
        std::memcpy(stereo_ptrs[0], direct_ptr, process_frames * sizeof(float));
        std::memcpy(stereo_ptrs[1], direct_ptr, process_frames * sizeof(float));
    }

    // Apply reflection/reverb if scene is built
    if (impl_->scene_built && reverb_mix > 0.0f && src.reflection && src.ambi_rotation &&
        src.ambi_binaural) {
        // Step 1: convolution reverb → ambisonics (world space)
        IPLAudioBuffer refl_out{};
        refl_out.numChannels = Impl::kAmbiChannels;
        refl_out.numSamples = static_cast<IPLint32>(process_frames);
        refl_out.data = impl_->reflection_channel_ptrs;

        for (int ch = 0; ch < Impl::kAmbiChannels; ch++) {
            std::memset(impl_->reflection_channels[ch].data(), 0, process_frames * sizeof(float));
        }

        iplReflectionEffectApply(src.reflection, &outputs.reflections, &mono_buf, &refl_out,
                                 nullptr);

        // Step 2: rotate ambisonics from world space → listener space
        IPLAudioBuffer rotated_out{};
        rotated_out.numChannels = Impl::kAmbiChannels;
        rotated_out.numSamples = static_cast<IPLint32>(process_frames);
        rotated_out.data = impl_->rotated_channel_ptrs;

        for (int ch = 0; ch < Impl::kAmbiChannels; ch++) {
            std::memset(impl_->rotated_channels[ch].data(), 0, process_frames * sizeof(float));
        }

        IPLCoordinateSpace3 listener_orient{};
        listener_orient.origin = toIPL(impl_->listener_pos);
        listener_orient.ahead = toIPL(impl_->listener_forward);
        listener_orient.up = toIPL(impl_->listener_up);
        listener_orient.right = {
            listener_orient.ahead.y * listener_orient.up.z -
                listener_orient.ahead.z * listener_orient.up.y,
            listener_orient.ahead.z * listener_orient.up.x -
                listener_orient.ahead.x * listener_orient.up.z,
            listener_orient.ahead.x * listener_orient.up.y -
                listener_orient.ahead.y * listener_orient.up.x};

        IPLAmbisonicsRotationEffectParams rot_params{};
        rot_params.orientation = listener_orient;
        rot_params.order = kMaxReflectionOrder;

        iplAmbisonicsRotationEffectApply(src.ambi_rotation, &rot_params, &refl_out, &rotated_out);

        // Step 3: decode rotated ambisonics → binaural stereo via HRTF
        float* reverb_stereo_ptrs[2] = {impl_->reverb_stereo_left.data(),
                                         impl_->reverb_stereo_right.data()};
        IPLAudioBuffer reverb_stereo_buf{};
        reverb_stereo_buf.numChannels = 2;
        reverb_stereo_buf.numSamples = static_cast<IPLint32>(process_frames);
        reverb_stereo_buf.data = reverb_stereo_ptrs;

        std::memset(reverb_stereo_ptrs[0], 0, process_frames * sizeof(float));
        std::memset(reverb_stereo_ptrs[1], 0, process_frames * sizeof(float));

        IPLAmbisonicsBinauralEffectParams ambi_params{};
        ambi_params.hrtf = impl_->hrtf;
        ambi_params.order = kMaxReflectionOrder;

        iplAmbisonicsBinauralEffectApply(src.ambi_binaural, &ambi_params, &rotated_out,
                                          &reverb_stereo_buf);

        // Mix reverb into output
        float wet = reverb_mix;
        for (uint32_t i = 0; i < process_frames; i++) {
            stereo_ptrs[0][i] += reverb_stereo_ptrs[0][i] * wet;
            stereo_ptrs[1][i] += reverb_stereo_ptrs[1][i] * wet;
        }
    }

    // Interleave to output
    for (uint32_t i = 0; i < process_frames; i++) {
        out_stereo[i * 2] = stereo_ptrs[0][i];
        out_stereo[i * 2 + 1] = stereo_ptrs[1][i];
    }
    // Zero any remaining frames
    for (uint32_t i = process_frames; i < frame_count; i++) {
        out_stereo[i * 2] = 0.0f;
        out_stereo[i * 2 + 1] = 0.0f;
    }
}

void SpatialAudio::simulate() {
    if (!impl_ || !enabled) return;

    // Set listener
    IPLCoordinateSpace3 listener{};
    listener.origin = toIPL(impl_->listener_pos);
    listener.ahead = toIPL(impl_->listener_forward);
    listener.up = toIPL(impl_->listener_up);
    // Right = ahead × up (in Steam Audio's Y-up system)
    listener.right = {
        listener.ahead.y * listener.up.z - listener.ahead.z * listener.up.y,
        listener.ahead.z * listener.up.x - listener.ahead.x * listener.up.z,
        listener.ahead.x * listener.up.y - listener.ahead.y * listener.up.x};

    // Set shared inputs
    IPLSimulationSharedInputs shared{};
    shared.listener = listener;
    shared.numRays = kNumRays;
    shared.numBounces = kNumBounces;
    shared.duration = kMaxDuration;
    shared.order = kMaxReflectionOrder;
    shared.irradianceMinDistance = 0.1f;

    iplSimulatorSetSharedInputs(
        impl_->simulator,
        static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS),
        &shared);

    // Set per-source inputs
    for (auto& [id, src] : impl_->sources) {
        IPLCoordinateSpace3 source_coords{};
        source_coords.origin = toIPL(src.position);
        source_coords.ahead = {0, 0, -1};
        source_coords.up = {0, 1, 0};
        source_coords.right = {1, 0, 0};

        IPLSimulationInputs inputs{};
        inputs.flags = static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_DIRECT |
                                                        IPL_SIMULATIONFLAGS_REFLECTIONS);
        inputs.directFlags = static_cast<IPLDirectSimulationFlags>(
            IPL_DIRECTSIMULATIONFLAGS_DISTANCEATTENUATION | IPL_DIRECTSIMULATIONFLAGS_OCCLUSION |
            IPL_DIRECTSIMULATIONFLAGS_AIRABSORPTION);
        inputs.source = source_coords;
        inputs.distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_DEFAULT;
        inputs.airAbsorptionModel.type = IPL_AIRABSORPTIONTYPE_DEFAULT;
        inputs.occlusionType = IPL_OCCLUSIONTYPE_RAYCAST;
        inputs.reverbScale[0] = 1.0f;
        inputs.reverbScale[1] = 1.0f;
        inputs.reverbScale[2] = 1.0f;

        iplSourceSetInputs(src.source,
                           static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_DIRECT |
                                                            IPL_SIMULATIONFLAGS_REFLECTIONS),
                           &inputs);
    }

    // Direct simulation is cheap — run every frame
    iplSimulatorRunDirect(impl_->simulator);

    // Reflections are expensive — run on background thread, only when listener moves
    if (impl_->scene_built && !impl_->reflection_running.load()) {
        float dist = glm::length(impl_->listener_pos - impl_->last_reflection_pos);
        if (dist > Impl::kReflectionUpdateDistance || impl_->last_reflection_pos == glm::vec3(0)) {
            impl_->last_reflection_pos = impl_->listener_pos;
            impl_->reflection_running.store(true);

            // Join previous thread if still joinable
            if (impl_->reflection_thread.joinable()) {
                impl_->reflection_thread.join();
            }

            impl_->reflection_thread = std::thread([this]() {
                std::lock_guard lock(impl_->sim_mutex);
                if (!impl_->shutdown.load()) {
                    iplSimulatorRunReflections(impl_->simulator);
                }
                impl_->reflection_running.store(false);
            });
        }
    }
}

#else  // !ENGINE_STEAM_AUDIO — stub implementation

struct SpatialAudio::Impl {};

SpatialAudio::SpatialAudio() : impl_(std::make_unique<Impl>()) {
    logInfo("Steam Audio not available (compiled without ENGINE_STEAM_AUDIO)");
}
SpatialAudio::~SpatialAudio() = default;
bool SpatialAudio::isAvailable() const { return false; }
uint32_t SpatialAudio::getListenerSourceId() const { return 0; }
void SpatialAudio::buildScene(std::span<const AcousticMesh*>) {}
void SpatialAudio::clearScene() {}
void SpatialAudio::setListener(glm::vec3, glm::vec3, glm::vec3) {}
uint32_t SpatialAudio::createSource(glm::vec3) { return 0; }
void SpatialAudio::updateSource(uint32_t, glm::vec3) {}
void SpatialAudio::removeSource(uint32_t) {}
void SpatialAudio::processBinaural(uint32_t, const float* in_mono, float* out_stereo,
                                    uint32_t frame_count) {
    for (uint32_t i = 0; i < frame_count; i++) {
        out_stereo[i * 2] = in_mono[i];
        out_stereo[i * 2 + 1] = in_mono[i];
    }
}
void SpatialAudio::simulate() {}

#endif  // ENGINE_STEAM_AUDIO

}  // namespace engine
