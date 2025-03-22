#version 440

layout(location = 0) in vec4 a_pos;
layout(location = 1) in vec3 a_normal;

layout(location = 0) out vec4 v_color;

layout(std140, binding = 0) uniform buf {
    mat4 u_mvp;
    mat3 u_normal;
    float u_scale;
};

void main()
{
    gl_Position = u_mvp * a_pos;
    gl_PointSize = u_scale == 0.0 ? 1.0 : u_scale / gl_Position.w;

    vec3 normal = normalize(u_normal * a_normal);
    vec3 rgb = (normal + 1.0) * 0.5;
    v_color = vec4(rgb, 1.0);
}
