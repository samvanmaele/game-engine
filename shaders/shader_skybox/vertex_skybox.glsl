#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vpos;
layout(location = 1) uniform mat4 projection;
layout(location = 5) uniform mat4 view;

out vec3 TexCoords;

void main()
{
    TexCoords = vpos;
    vec4 pos = projection * mat4(mat3(view)) * vec4(vpos, 1.0);
    gl_Position = pos.xyww;
} 