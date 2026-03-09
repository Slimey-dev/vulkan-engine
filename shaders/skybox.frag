#version 450

layout(location = 0) in vec3 fragTexDir;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpace;
    vec4 lightPos;
    vec4 viewPos;
    vec4 lightColor;
    vec4 fogColor;
    vec4 fogParams;  // x = density
} ubo;

layout(binding = 1) uniform samplerCube skybox;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = texture(skybox, fragTexDir).rgb;

    // Exponential fog based on elevation — horizon is fully fogged, zenith partially clear
    vec3 dir = normalize(fragTexDir);
    float elevation = abs(dir.z);
    float fogFactor = 1.0 - exp(-ubo.fogParams.x * 40.0 * (1.0 - elevation));
    color = mix(color, ubo.fogColor.rgb, fogFactor);

    outColor = vec4(color, 1.0);
}
