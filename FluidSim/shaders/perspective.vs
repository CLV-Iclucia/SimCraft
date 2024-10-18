#version 330 core
layout(location=0) in highp vec3 aPos;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

void main() {
	gl_Position = proj * view * model * vec4(aPos, 1.0);
	gl_PointSize = 5.f;
}