#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
layout (location = 0) in vec3 aWorldPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 worldPos;
out vec3 normal;

void main() {
    gl_Position = projection * view * model * vec4(aWorldPos, 1.0f);
    worldPos = aWorldPos;
    normal = aNormal;
}