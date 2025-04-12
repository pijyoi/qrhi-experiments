#version 440

layout(location = 0) in vec4 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec4 v_color;

layout(binding = 0, std140) uniform buf {
    mat4 u_mvp;
    mat3 u_normal;
    vec3 u_lightDirection;
};

void main()
{
    v_normal = normalize(u_normal * a_normal);
    v_color = a_color;
    gl_Position = u_mvp * a_pos;
}
