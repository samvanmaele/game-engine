#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vpos;
layout(location = 2) in vec2 vtex;
layout(location = 3) uniform mat4 projection;
layout(location = 7) uniform mat4 view;
layout(location = 11) uniform vec3 cameraPos;
layout(location = 1) uniform float time;
layout(location = 12) uniform sampler2D heightMap;

out vec2 ftex;
out float dis;

mat3 rotate(float angle)
{
    return mat3(cos(angle), 0, -sin(angle),
                0, 1, 0,
                sin(angle), 0, cos(angle));
}

void main()
{
    vec2 tex = vec2(mod(gl_InstanceID, 1000), floor(gl_InstanceID/1000)) / 5 - 100;
    tex += floor(vec2(cameraPos.x, cameraPos.z));

    float height = texture(heightMap, tex / 2000 + 0.5).x * 1000;
    vec3 pos = vec3(tex.x, height, tex.y);
    float offset = cos(12 * tex.x + 16 * cos(tex.y));

    pos.x += offset / 25 + (1 - cos(vpos.y)) * (cos(time + gl_InstanceID * 0.1)+1.2);
    pos.z += offset / 15;
    pos += rotate(offset) * vpos;

    vec4 worldPos = view * vec4(pos, 1);

    gl_Position = projection * worldPos;
    ftex = vtex;
    dis = clamp(length(vec2(worldPos.x, worldPos.z)) * 0.005, 0, 1);
}