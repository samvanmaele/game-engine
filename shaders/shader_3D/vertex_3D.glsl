#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vpos;
layout(location = 1) in vec3 vnorm;
layout(location = 2) in vec2 vtex;

layout(location = 3) uniform mat4 projection;
layout(location = 7) uniform mat4 view;
layout(location = 11) uniform mat4 model;

out vec2 TexCoords;
out vec3 fragPos;
out vec3 fragNorm;

void main()
{
    vec4 position = vec4(vpos, 1.0);
    vec4 normal = vec4(vnorm, 0.0);

    gl_Position = projection * view * model * position;

	TexCoords = vtex;
    fragPos = (model * position).xyz;
    fragNorm = (model * normal).xyz;
}