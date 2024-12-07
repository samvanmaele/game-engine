#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

in vec2 fTex;
in float materialId;
in float dis;

layout(location = 14) uniform sampler2D material[2];
const vec4 farColor = vec4(0, 0, 0.5, 1);

out vec4 color;

void main()
{
    int texId = clamp(int(materialId), 0, 1);
    vec4 tex = (texture(material[texId], 500 * fTex) + texture(material[texId], fTex)) * 0.5;
    color = mix(tex, farColor + tex, dis);
    color = pow(color, vec4(0.45));
}