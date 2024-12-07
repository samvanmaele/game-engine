#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

in vec3 TexCoords;
uniform samplerCube Texture;

out vec4 color;

void main()
{
    color = texture(Texture, TexCoords);
    color = pow(color, vec4(0.45));
}