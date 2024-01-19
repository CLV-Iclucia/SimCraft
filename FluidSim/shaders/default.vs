#version 330 core

layout(location = 0) in vec2 aPos;

out vec2 screenPos;
void main() {
    screenPos = aPos;
    gl_Position = vec4(aPos, 0.f, 1.f);
}