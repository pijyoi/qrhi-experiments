#version 440

layout(location = 0) in vec2 v_texcoord;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    vec2 u_minmax;
    float u_yscale;
};

layout(binding = 1) uniform sampler2D tex_lut;
layout(binding = 2) uniform sampler2D tex_img;

void main()
{
    float ivalue = texture(tex_img, v_texcoord).r;

    ivalue = (ivalue - u_minmax.x) / (u_minmax.y - u_minmax.x);
    ivalue = clamp(ivalue, 0.0, 1.0);

    fragColor = texture(tex_lut, vec2(ivalue, 0));
}
