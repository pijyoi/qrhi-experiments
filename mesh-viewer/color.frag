#version 440

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 fragColor;

layout(binding = 0, std140) uniform buf {
    mat4 u_mvp;
    mat3 u_normal;
    vec3 u_lightDirection;
};

void main()
{
    float p = max(dot(v_normal, normalize(u_lightDirection)), 0.0);

    vec3 rgb = v_color.rgb * (0.2 + p * 0.8);
    fragColor = vec4(rgb, v_color.a);
}
