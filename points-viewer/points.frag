#version 440

layout(location = 0) in vec4 v_color;

layout(location = 0) out vec4 fragColor;

void main()
{
    vec2 xy = (gl_PointCoord - 0.5) * 2.0;
    if (dot(xy, xy) <= 1.0) fragColor = v_color;
    else discard;
}
