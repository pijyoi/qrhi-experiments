#version 440

layout(location = 0) out vec2 v_texcoord;

void main()
{
    v_texcoord = vec2(gl_VertexIndex & 1, (gl_VertexIndex & 2) >> 1);
    vec2 pos = v_texcoord * 2.0 - 1.0;
    gl_Position = vec4(pos.x, -pos.y, 0.0, 1.0);
}
