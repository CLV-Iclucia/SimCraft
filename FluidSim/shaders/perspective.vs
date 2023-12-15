#version 330 core

layout(location = 0) in highp_vec3 aPos;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

out highp_vec3 FragPos;
void main() {
	FragPos = vec3(proj * view * model * vec4(aPos, 1.0));
	gl_PointSize = 10.f;
}