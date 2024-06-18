// perspective projection vertex shader
#version 330 core
layout(location = 0) in vec3 aColor;
layout(location = 1) in vec3 aNormal;
out vec3 worldNormal;
out vec3 worldPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    worldNormal = normalize(vec3(view * model * vec4(aNormal, 0.0)));
    worldPos = vec3(view * model * vec4(aPos, 1.0));
}