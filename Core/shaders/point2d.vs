#version 330 core

// Input vertex data, directly pass to fragment shader
layout(location = 0) in highp_vec2 aPos;

void main() {
    gl_Position = highp_vec4(aPos.x - 0.5, aPos.y - 0.5, 0.0, 1.0);
    gl_PointSize = 6.0;
}