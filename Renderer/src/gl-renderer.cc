#include "gl-renderer.h"
#include <glad/glad.h>      // Must come before GLFW
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cmath>

namespace sim::renderer {

// Shader source code
static const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec3 aColor;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragPos;
    out vec3 Normal;
    out vec3 VertexColor;

    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        VertexColor = aColor;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

static const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;
    in vec3 Normal;
    in vec3 VertexColor;

    uniform vec3 lightPos;
    uniform vec3 viewPos;

    void main() {
        // Ambient
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * VertexColor;

        // Diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * VertexColor;

        // Specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);

        vec3 result = ambient + diffuse + specular;
        FragColor = vec4(result, 1.0);
    }
)";

// Wireframe shader
static const char* wireVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 view;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * view * vec4(aPos, 1.0);
    }
)";

static const char* wireFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 wireColor;
    void main() {
        FragColor = vec4(wireColor, 1.0);
    }
)";

// Helper function to compile shader
static unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation error:\n" << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Helper function to create shader program
static unsigned int createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader program linking error:\n" << infoLog << std::endl;
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

void GLRenderer::initialize(const RendererConfig& config) {
    m_config = config;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (config.headless) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }

    m_window = glfwCreateWindow(config.windowWidth, config.windowHeight,
                                config.windowTitle.c_str(), nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(m_window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    if (config.vsync) {
        glfwSwapInterval(1);
    } else {
        glfwSwapInterval(0);
    }

    // Compile shaders
    m_meshShader = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    m_wireShader = createShaderProgram(wireVertexShaderSource, wireFragmentShaderSource);

    // Set up OpenGL state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    // Set up input callbacks
    setupInputCallbacks();

    std::cout << "GLRenderer initialized successfully" << std::endl;
}

void GLRenderer::drawFrame(const SceneProxy& scene) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Use mesh shader
    glUseProgram(m_meshShader);

    // Set up lighting
    glm::vec3 lightPos(5.0f, 5.0f, 5.0f);
    glUniform3fv(glGetUniformLocation(m_meshShader, "lightPos"), 1, glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(m_meshShader, "viewPos"), 1, glm::value_ptr(m_camera.position));

    // Set up matrices
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = computeViewMatrix();
    glm::mat4 projection = computeProjectionMatrix();

    glUniformMatrix4fv(glGetUniformLocation(m_meshShader, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(m_meshShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_meshShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // Draw meshes
    for (const auto& mesh : scene.meshes) {
        drawMesh(mesh);
    }

    // Draw wireframes
    if (!scene.wireframes.empty()) {
        glUseProgram(m_wireShader);
        glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        for (const auto& wire : scene.wireframes) {
            drawWireframe(wire);
        }
    }

    // Draw particles
    for (const auto& particle : scene.particles) {
        drawParticles(particle);
    }
}

void GLRenderer::uploadMesh(const MeshProxy& mesh) {
    auto it = m_meshCache.find(mesh.name);
    bool needCreate = (it == m_meshCache.end());

    GLMeshState state;
    if (!needCreate) {
        state = it->second;
        // Check if we need to reallocate
        if (state.vertexCount != mesh.positions.size()) {
            // Delete old buffers
            glDeleteVertexArrays(1, &state.vao);
            glDeleteBuffers(1, &state.vbo);
            glDeleteBuffers(1, &state.ebo);
            if (state.nbo) glDeleteBuffers(1, &state.nbo);
            needCreate = true;
        }
    }

    if (needCreate) {
        // Create new buffers
        glGenVertexArrays(1, &state.vao);
        glGenBuffers(1, &state.vbo);
        glGenBuffers(1, &state.ebo);
        glGenBuffers(1, &state.nbo);
        state.vertexCount = mesh.positions.size();
    }

    state.indexCount = static_cast<int>(mesh.triangles.size() * 3);

    // Upload data
    glBindVertexArray(state.vao);

    // Positions
    glBindBuffer(GL_ARRAY_BUFFER, state.vbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.positions.size() * sizeof(core::Vec3f),
                 mesh.positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f), (void*)0);
    glEnableVertexAttribArray(0);

    // Normals
    if (!mesh.normals.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, state.nbo);
        glBufferData(GL_ARRAY_BUFFER, mesh.normals.size() * sizeof(core::Vec3f),
                     mesh.normals.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f), (void*)0);
        glEnableVertexAttribArray(1);
    } else {
        // TODO: compute face normals or skip
        glDisableVertexAttribArray(1);
    }

    // Colors (use default if not provided)
    if (!mesh.colors.empty()) {
        // Would need another VBO for colors
    }

    // Indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, state.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.triangles.size() * sizeof(core::Vec3u),
                 mesh.triangles.data(), GL_DYNAMIC_DRAW);

    glBindVertexArray(0);

    m_meshCache[mesh.name] = state;
}

void GLRenderer::drawMesh(const MeshProxy& mesh) {
    uploadMesh(mesh);

    auto it = m_meshCache.find(mesh.name);
    if (it == m_meshCache.end()) return;

    const auto& state = it->second;

    glUseProgram(m_meshShader);
    glBindVertexArray(state.vao);
    glDrawElements(GL_TRIANGLES, state.indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void GLRenderer::drawWireframe(const WireframeProxy& wf) {
    // Simple wireframe drawing - would need implementation
    // For now, just draw lines
}

void GLRenderer::drawParticles(const ParticleProxy& particles) {
    // Simple particle rendering - would use point sprites or instanced spheres
    // For now, just placeholder
}

glm::mat4 GLRenderer::computeViewMatrix() const {
    return glm::lookAt(m_camera.position, m_camera.target, m_camera.up);
}

glm::mat4 GLRenderer::computeProjectionMatrix() const {
    int width, height;
    glfwGetWindowSize(m_window, &width, &height);
    float aspect = (float)width / (float)height;
    return glm::perspective(glm::radians(m_camera.fov), aspect, m_camera.nearPlane, m_camera.farPlane);
}

bool GLRenderer::pollAndSwap() {
    if (!m_window) return false;

    glfwPollEvents();

    if (glfwWindowShouldClose(m_window)) {
        return false;
    }

    updateCameraFromInput();
    glfwSwapBuffers(m_window);
    return true;
}

void GLRenderer::cleanup() {
    for (auto& [name, state] : m_meshCache) {
        glDeleteVertexArrays(1, &state.vao);
        glDeleteBuffers(1, &state.vbo);
        glDeleteBuffers(1, &state.ebo);
        if (state.nbo) glDeleteBuffers(1, &state.nbo);
    }
    m_meshCache.clear();

    if (m_meshShader) glDeleteProgram(m_meshShader);
    if (m_wireShader) glDeleteProgram(m_wireShader);
    if (m_particleShader) glDeleteProgram(m_particleShader);

    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }

    glfwTerminate();
    std::cout << "GLRenderer cleaned up" << std::endl;
}

void GLRenderer::setupInputCallbacks() {
    if (!m_window) return;

    glfwSetWindowUserPointer(m_window, this);

    glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int button, int action, int mods) {
        auto* renderer = static_cast<GLRenderer*>(glfwGetWindowUserPointer(window));
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            renderer->m_mousePressed = (action == GLFW_PRESS);
            if (renderer->m_mousePressed) {
                glfwGetCursorPos(window, &renderer->m_lastMouseX, &renderer->m_lastMouseY);
            }
        }
    });

    glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double xpos, double ypos) {
        auto* renderer = static_cast<GLRenderer*>(glfwGetWindowUserPointer(window));
        if (renderer->m_mousePressed) {
            double dx = xpos - renderer->m_lastMouseX;
            double dy = ypos - renderer->m_lastMouseY;

            renderer->m_yaw -= (float)dx * 0.5f;
            renderer->m_pitch -= (float)dy * 0.5f;

            // Clamp pitch
            renderer->m_pitch = std::max(-89.0f, std::min(89.0f, renderer->m_pitch));

            renderer->m_lastMouseX = xpos;
            renderer->m_lastMouseY = ypos;
        }
    });

    glfwSetScrollCallback(m_window, [](GLFWwindow* window, double xoffset, double yoffset) {
        auto* renderer = static_cast<GLRenderer*>(glfwGetWindowUserPointer(window));
        renderer->m_distance -= (float)yoffset * 0.5f;
        renderer->m_distance = std::max(0.1f, renderer->m_distance);
    });
}

void GLRenderer::updateCameraFromInput() {
    // Update camera position based on yaw, pitch, distance
    float yawRad = glm::radians(m_yaw);
    float pitchRad = glm::radians(m_pitch);

    m_camera.position.x = m_camera.target.x + m_distance * cos(pitchRad) * cos(yawRad);
    m_camera.position.y = m_camera.target.y + m_distance * sin(pitchRad);
    m_camera.position.z = m_camera.target.z + m_distance * cos(pitchRad) * sin(yawRad);

    // Call user callback if set
    if (m_inputCallback) {
        m_inputCallback(m_camera);
    }
}

// Factory function
std::unique_ptr<Renderer> createRenderer(const RendererConfig& config) {
    return std::make_unique<GLRenderer>();
}

} // namespace sim::renderer
