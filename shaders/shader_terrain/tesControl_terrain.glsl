#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

layout(vertices = 3) out;
layout(location = 9) uniform mat4 yawMat;

const int minTessLevel = 1;
const int maxTessLevel = 24;
const int minDistance = 64;
const int maxDistance = 128;

void main()
{
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    if (gl_InvocationID == 0)
    {
        vec4 eyePos0 = yawMat * (gl_in[0].gl_Position + (gl_in[1].gl_Position - gl_in[0].gl_Position)/2);
        vec4 eyePos1 = yawMat * (gl_in[0].gl_Position + (gl_in[2].gl_Position - gl_in[0].gl_Position)/2);
        vec4 eyePos2 = yawMat * (gl_in[1].gl_Position + (gl_in[2].gl_Position - gl_in[1].gl_Position)/2);

        float dis0 = length(vec2(eyePos0.x, eyePos0.z));
        float dis1 = length(vec2(eyePos1.x, eyePos1.z));
        float dis2 = length(vec2(eyePos2.x, eyePos2.z));

        float tes0 = clamp((dis0 - minDistance) / (maxDistance - minDistance), 0, 1);
        float tes1 = clamp((dis1 - minDistance) / (maxDistance - minDistance), 0, 1);
        float tes2 = clamp((dis2 - minDistance) / (maxDistance - minDistance), 0, 1);

        float tesLevel0 = mix(maxTessLevel, minTessLevel, tes0);
        float tesLevel1 = mix(maxTessLevel, minTessLevel, tes1);
        float tesLevel2 = mix(maxTessLevel, minTessLevel, tes2);

        gl_TessLevelOuter[0] = tesLevel2;
        gl_TessLevelOuter[1] = tesLevel1;
        gl_TessLevelOuter[2] = tesLevel0;
        
        gl_TessLevelInner[0] = max(max(tesLevel0, tesLevel1), tesLevel2);
    }
}