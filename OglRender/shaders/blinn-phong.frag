#version 330 core
in vec3 worldNormal;
in vec3 worldPos;

out vec4 FragColor;

struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform PointLight pointLight;
void main() {
    vec3 norm = normalize(worldNormal);
    vec3 lightDir = normalize(pointLight.position - worldPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * pointLight.diffuse;
    vec3 ambient = pointLight.ambient;
    vec3 result = (ambient + diffuse) * vec3(1.0);
    FragColor = vec4(result, 1.0);
}