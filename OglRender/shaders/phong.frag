#version 330 core

struct DirectionalLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform DirectionalLight light;

struct PhongMaterial {
    vec3 baseColor;
    float shininess;
};

uniform PhongMaterial phong;

in vec3 worldPos;
in vec3 normal;
out vec4 FragColor;

void main() {
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(normal, lightDir), 0.0f);
    vec3 diffuse = light.diffuse * diff;

    vec3 viewDir = normalize(-worldPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0f), phong.shininess);
    vec3 specular = light.specular * spec;

    vec3 result = (light.ambient + diffuse + specular) * phong.baseColor;
    FragColor = vec4(result, 1.f);
}