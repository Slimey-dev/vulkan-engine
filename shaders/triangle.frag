#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in vec3 fragNormal;
layout(location = 4) in vec4 fragLightSpacePos;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpace;
    vec4 lightPos;
    vec4 viewPos;
    vec4 lightColor;
    vec4 fogColor;
    vec4 fogParams;   // x = density
    vec4 lightDir;    // xyz = spotlight dir, w = cos(outer cutoff); w==0 → point light
} ubo;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2DShadow shadowMap;

layout(push_constant) uniform PushConstants {
    mat4 model;
    float emissive;
} pc;

layout(location = 0) out vec4 outColor;

float calcShadow(vec4 lightSpacePos) {
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    // Vulkan depth is already [0,1], just remap XY from [-1,1] to [0,1]
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z > 1.0) {
        return 0.0;
    }

    // Fade shadow near the edges of the shadow map to avoid hard cutoff
    float edgeFade = 1.0;
    float fadeWidth = 0.15;
    edgeFade *= smoothstep(0.0, fadeWidth, projCoords.x);
    edgeFade *= smoothstep(0.0, fadeWidth, 1.0 - projCoords.x);
    edgeFade *= smoothstep(0.0, fadeWidth, projCoords.y);
    edgeFade *= smoothstep(0.0, fadeWidth, 1.0 - projCoords.y);

    // 3x3 PCF for soft shadow edges
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    float shadow = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(x, y) * texelSize;
            shadow += texture(shadowMap, vec3(projCoords.xy + offset, projCoords.z));
        }
    }
    shadow /= 9.0;
    return (1.0 - shadow) * edgeFade;
}

void main() {
    vec3 texColor = texture(texSampler, fragTexCoord).rgb * fragColor;

    if (pc.emissive > 0.0) {
        outColor = vec4(texColor * pc.emissive, 1.0);
        return;
    }

    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(ubo.lightPos.xyz - fragWorldPos);
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragWorldPos);

    // Ambient
    float ambientStrength = 0.15;
    vec3 ambient = ambientStrength * ubo.lightColor.xyz;

    // Shadow
    float shadow = calcShadow(fragLightSpacePos);

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * ubo.lightColor.xyz;

    // Specular (Blinn-Phong)
    float specularStrength = 0.5;
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * ubo.lightColor.xyz;

    // Spotlight cone attenuation with soft outer edge
    float spotAtten = 1.0;
    if (ubo.lightDir.w > 0.0) {
        float cosOuter = ubo.lightDir.w;
        float theta = dot(normalize(fragWorldPos - ubo.lightPos.xyz), normalize(ubo.lightDir.xyz));
        // Soft falloff: fade from cosOuter-0.15 to cosOuter
        spotAtten = smoothstep(cosOuter - 0.15, cosOuter, theta);
    }

    vec3 result = (ambient + spotAtten * (1.0 - shadow) * (diffuse + specular)) * texColor;

    // Distance fog
    float dist = length(fragWorldPos - ubo.viewPos.xyz);
    float fogFactor = exp(-ubo.fogParams.x * dist);
    result = mix(ubo.fogColor.rgb, result, fogFactor);

    outColor = vec4(result, 1.0);
}
