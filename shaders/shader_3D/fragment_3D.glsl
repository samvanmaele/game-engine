#version 410 core
#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

struct PointLight {
    vec3 position;
    vec3 color;
    float strength;
};

in vec2 TexCoords;
in vec3 fragPos;
in vec3 fragNorm;

uniform sampler2D imageTexture;
layout(location = 15) uniform vec3 cameraPosition;
uniform PointLight Lights[10];

out vec4 color;

vec3 calculatePointLight(PointLight light, vec3 fragPos, vec3 fragNorm);

void main()
{
    vec4 baseTexture = texture(imageTexture, TexCoords);
    //ambient
    vec3 temp = 0.2 * baseTexture.rgb;

    for (int i = 0; i < 100; i++) {
        temp += calculatePointLight(Lights[i], fragPos, fragNorm);
    }

    color = vec4(temp, baseTexture.a);
    color = pow(color, vec4(0.45));
}

vec3 calculatePointLight(PointLight light, vec3 fragPos, vec3 fragNorm)
{
    vec3 baseTexture = texture(imageTexture, TexCoords).rgb;
    vec3 result = vec3(0);

    //geometric data
    vec3 fragLight = light.position - fragPos;
    float distance = length(fragLight);
    fragLight = normalize(fragLight);
    vec3 fragCamera = normalize(cameraPosition - fragPos);
    vec3 halfVec = normalize(fragLight + fragCamera);

    //diffuse
    result += light.color * light.strength * max(0.0, dot(fragNorm, fragLight)) / (distance * distance) * baseTexture;

    //specular
    result += 0.4 * light.color * light.strength * pow(max(0.0, dot(fragNorm, halfVec)),32) / (distance * distance);

    return result;
}