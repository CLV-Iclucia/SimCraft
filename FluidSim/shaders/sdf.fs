#version 330 core

in highp vec3 myColor;
out highp vec4 FragColor;
void main() {
//     if (myColor.x < -0.1 || myColor.x > 0.1)
//        discard;
    if (myColor.x > 0.0) discard;
    FragColor = vec4(myColor, 1.0);
}