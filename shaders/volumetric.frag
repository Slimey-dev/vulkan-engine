#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragNormal;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpace;
    vec4 lightPos;
    vec4 viewPos;
    vec4 lightColor;
    vec4 fogColor;
    vec4 fogParams;
    vec4 lightDir;
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(fragWorldPos - ubo.viewPos.xyz);

    // Fresnel fade: bright at shallow angles, but fade out at the very edge
    // (silhouette) so the cone boundary is soft instead of a hard line
    float NdotV = abs(dot(normal, viewDir));
    float fresnel = pow(1.0 - NdotV, 1.2) * smoothstep(0.0, 0.5, NdotV);

    // Height gradient: fragTexCoord.y = 1 at top (lamp), 0 at bottom (floor)
    float heightFade = fragTexCoord.y;

    // Soft fade near room boundaries so the cone/wall intersection is blurred
    // Room geometry: X[-5,5], Y[-5,5], Z[-0.5,7.5] — must match scene.cpp createRoomMesh()
    float fadeWidth = 1.5;
    float wallFade = 1.0;
    wallFade *= smoothstep(0.0, fadeWidth, fragWorldPos.x - (-5.0));
    wallFade *= smoothstep(0.0, fadeWidth, 5.0 - fragWorldPos.x);
    wallFade *= smoothstep(0.0, fadeWidth, fragWorldPos.y - (-5.0));
    wallFade *= smoothstep(0.0, fadeWidth, 5.0 - fragWorldPos.y);
    wallFade *= smoothstep(0.0, fadeWidth, fragWorldPos.z - (-0.5));

    // Combine: bright near lamp, fading toward floor, soft near walls
    float intensity = fresnel * heightFade * wallFade * 0.15;

    // Bloom glow near the lamp: use a steep power curve so only the top
    // of the cone (heightFade close to 1) produces values above the bloom threshold
    float bloomGlow = pow(heightFade, 6.0) * fresnel * wallFade * 1.2;

    outColor = vec4(fragColor * (intensity + bloomGlow), 1.0);
}
