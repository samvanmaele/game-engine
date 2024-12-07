#version 410 core

in vec2 fragmentTexCoord;
out vec4 color;
uniform sampler2D material;

void main()
{
    color = texture(material, fragmentTexCoord);
    color = pow(color, vec4(0.45));
}