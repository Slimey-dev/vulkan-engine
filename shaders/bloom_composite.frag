#version 450

layout(location = 0) in vec2 texCoord;

layout(binding = 0) uniform sampler2D sceneImage;
layout(binding = 1) uniform sampler2D bloomImage;

layout(push_constant) uniform PushConstants {
    float intensity;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 scene = texture(sceneImage, texCoord).rgb;
    vec3 bloom = texture(bloomImage, texCoord).rgb;
    outColor = vec4(scene + bloom * pc.intensity, 1.0);
}
