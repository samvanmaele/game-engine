#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

in vec2 ftex;
in float dis;
layout(location = 13) uniform sampler2D material;

out vec4 color;

void main()
{
    color = texture(material, vec2(ftex.y, ftex.x));
    if (color.a <= 0.0) discard;
    color.r = max(color.r, dis * 0.1);
    color.b = max(color.b, dis);
    color.a = min(color.a, 1 - 5*dis);
    color = pow(color, vec4(0.45));
}