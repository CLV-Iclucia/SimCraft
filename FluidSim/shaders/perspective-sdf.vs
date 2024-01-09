#version 330 core

layout(location = 0) in highp vec3 aPos;
layout(location = 1) in highp vec3 aColor;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

out highp vec3 myColor;
void main() {
	gl_Position = proj * view * model * vec4(aPos, 1.0);
	myColor = aColor;
	gl_PointSize = 5.0f;
}