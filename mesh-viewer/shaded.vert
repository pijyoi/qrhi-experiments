#version 440

layout(location = 0) in vec4 a_pos;
layout(location = 1) in vec3 a_normal;

layout(location = 0) out vec3 v_normal;

layout(std140, binding = 0) uniform buf {
    mat4 u_mvp;
    mat3 u_normal;
    vec3 u_lightDirection;
};

void main()
{
    v_normal = normalize(u_normal * a_normal);
    gl_Position = u_mvp * a_pos;
}
