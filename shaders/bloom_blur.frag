#version 450

layout(location = 0) in vec2 texCoord;

layout(binding = 0) uniform sampler2D inputImage;

layout(push_constant) uniform PushConstants {
    vec2 direction;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    vec3 result = texture(inputImage, texCoord).rgb * weights[0];
    for (int i = 1; i < 5; i++) {
        vec2 offset = pc.direction * float(i);
        result += texture(inputImage, texCoord + offset).rgb * weights[i];
        result += texture(inputImage, texCoord - offset).rgb * weights[i];
    }
    outColor = vec4(result, 1.0);
}
