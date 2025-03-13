#version 440

layout(location = 0) in vec3 v_normal;

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 u_mvp;
    mat3 u_normal;
    vec3 u_lightDirection;
};

void main()
{
    vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

    float p = max(dot(v_normal, normalize(u_lightDirection)), 0.0);

    vec3 rgb = color.rgb * (0.2 + p * 0.8);
    fragColor = vec4(rgb, color.a);
}
