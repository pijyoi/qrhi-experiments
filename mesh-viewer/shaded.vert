#version 440

layout(location = 0) in vec4 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_tex;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_tex;

layout(binding = 0, std140) uniform buf {
    mat4 u_mvp;
    mat3 u_normal;
    vec3 u_lightDirection;
};

void main()
{
    v_normal = normalize(u_normal * a_normal);
    v_tex = a_tex;
    gl_Position = u_mvp * a_pos;
}
