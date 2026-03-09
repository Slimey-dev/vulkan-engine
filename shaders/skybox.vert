#version 450

layout(location = 0) in vec3 inPosition;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpace;
    vec4 lightPos;
    vec4 viewPos;
    vec4 lightColor;
} ubo;

layout(location = 0) out vec3 fragTexDir;

void main() {
    fragTexDir = inPosition;

    // Strip translation from view matrix
    mat4 rotView = mat4(mat3(ubo.view));
    vec4 pos = ubo.proj * rotView * vec4(inPosition, 1.0);

    // z = w so depth is always 1.0 after perspective divide
    gl_Position = pos.xyww;
}
