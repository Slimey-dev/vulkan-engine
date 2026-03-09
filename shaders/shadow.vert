#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpace;
    vec4 lightPos;
    vec4 viewPos;
    vec4 lightColor;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pc;

void main() {
    gl_Position = ubo.lightSpace * pc.model * vec4(inPosition, 1.0);
}
