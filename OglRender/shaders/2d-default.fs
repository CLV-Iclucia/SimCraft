#version 330 core

in vec3 myColor;

void main() {
  gl_FragColor = vec4(myColor, 1.0f);
}