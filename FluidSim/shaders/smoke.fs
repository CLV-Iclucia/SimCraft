#version 330 core
in vec2 screenPos;

uniform sampler3D smokeTex;
const vec3 cameraPos = vec3(-0.5, 0.5, 0.5); // looks to the positive x-axis
const float nearPlane = 0.5;
const float stepSize = 0.04f;

struct Ray {
    vec3 orig;
    vec3 dir;
};

bool insideBox(vec3 pos) {
    return pos.y >= 0.0 && pos.y <= 1.0 && pos.z >= 0.0 && pos.z <= 1.0 && pos.x >= 0.0 && pos.x <= 1.0;
}

void main() {
    float screen_x = cameraPos.x + nearPlane;
    float screen_y = screenPos.y * 0.5 + 0.5;
    float screen_z = screenPos.x * 0.5 + 0.5;
    vec3 dir = normalize(vec3(screen_x, screen_y, screen_z) - cameraPos);
    float color = 0.0;
    float T = 1.0;
    // compute the first intersection with the box
    // since camera faces the positive x-axis, the first intersection is always on the x = 0 plane
    // so t = -cameraPos.x / dir.x
    float t1 = -cameraPos.x / dir.x;
    vec3 p1 = cameraPos + t1 * dir;
    // if p1 outside the box, then discard
    if (p1.y < 0.0 || p1.y > 1.0 || p1.z < 0.0 || p1.z > 1.0)
        discard;
    vec3 pos = p1;
    for (int i = 0; i < 100; i++) {
        float density = texture(smokeTex, pos).r;
        color += density * stepSize;
        pos += stepSize * dir;
        if (!insideBox(pos))
            break;
    }
    gl_FragColor = vec4(color, 0.f, color, 1.0);
}