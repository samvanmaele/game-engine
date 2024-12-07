#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vpos;
layout(location = 1) in vec3 vnorm;
layout(location = 2) in vec2 vtex;

layout(location = 3) in ivec4 vboneIds; 
layout(location = 4) in vec4 vweights;

layout(location = 5) uniform mat4 projection;
layout(location = 9) uniform mat4 view;
layout(location = 13) uniform mat4 model;

layout(location = 18) uniform mat4 finalBonesMatrices[50];

out vec2 TexCoords;
out vec3 fragPos;
out vec3 fragNorm;

vec4 applyBone(vec4 p)
{
    vec4 result = vec4(0.0);
    for(int i = 0; i < 4; ++i)
    {
        if(vboneIds[i] >= 50) 
        {
            result = p;
            break;
        }
        result += vweights[i] * (finalBonesMatrices[vboneIds[i]] * p);
    }
    return result;
}

void main()
{
    vec4 position = applyBone(vec4(vpos, 1.0));
    vec4 normal = normalize(applyBone(vec4(vnorm, 0.0)));

    gl_Position = projection * view * model * position;

	TexCoords = vtex;
    fragPos = (model * position).xyz;
    fragNorm = (model * normal).xyz;
}