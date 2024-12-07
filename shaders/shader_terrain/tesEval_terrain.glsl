#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

layout(triangles, equal_spacing, ccw) in;
layout(location = 1) uniform mat4 projection;
layout(location = 5) uniform mat4 view;
layout(location = 13) uniform sampler2D heightMap;


out vec2 fTex;
out float materialId;
out float dis;

void main()
{
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    float w = gl_TessCoord.z;

    vec4 pos0 = gl_in[0].gl_Position;
    vec4 pos1 = gl_in[1].gl_Position;
    vec4 pos2 = gl_in[2].gl_Position;

    vec4 pos = u * pos0 + v * pos1 + w * pos2;
    vec2 texCoord = pos.xz/2000 + 0.5;

    float height = texture(heightMap, texCoord).x;
    pos.y = height * 1000;
    
    vec4 worldPos = projection * view * pos;

    gl_Position = worldPos;
    fTex = texCoord;
    materialId = height + 0.999;
    dis = clamp(length(vec2(worldPos.x, worldPos.z)) * 0.005, 0, 1);
}