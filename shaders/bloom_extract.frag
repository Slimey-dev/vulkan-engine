#version 450

layout(location = 0) in vec2 texCoord;

layout(binding = 0) uniform sampler2D inputImage;

layout(push_constant) uniform PushConstants {
    float threshold;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = texture(inputImage, texCoord).rgb;
    vec3 excess = max(color - vec3(pc.threshold), vec3(0.0));
    outColor = vec4(excess, 1.0);
}
