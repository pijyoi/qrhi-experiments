#version 440

layout(location = 0) out vec2 v_texcoord;

layout(std140, binding = 0) uniform buf {
    mat4 u_view;
    vec2 u_minmax;
};

void main()
{
    v_texcoord = vec2(gl_VertexIndex & 1, (gl_VertexIndex & 2) >> 1);
    vec4 pos = vec4(v_texcoord * 2.0 - 1.0, 0.0, 1.0);
    gl_Position = u_view * pos;
}
