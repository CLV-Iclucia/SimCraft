#version 330 core

layout(location = 0) in highp_vec3 aPos;
layout(location = 1) in highp_vec3 aNormal;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;
out highp_vec3 FragPos;
out highp_vec3 Normal;

void main() {
	FragPos = vec3(proj * view * model * vec4(aPos, 1.0));
	Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
}