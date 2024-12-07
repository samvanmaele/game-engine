#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vpos;

void main()
{
    gl_Position = vec4(vpos, 1);
}