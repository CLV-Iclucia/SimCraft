//
// Created by creeper on 6/17/24.
//
#include <ogl-render/resource-handles.h>
#include <ogl-render/shader-prog.h>
#include <ogl-render/ogl-gui.h>
#include <ogl-render/platform.h>
#include <ogl-render/camera.h>
#include <ogl-render/camera-controller.h>
#include <ogl-render/drawable-mesh.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

using namespace opengl;
ImGuiIO &initImGui(GLFWwindow *window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void) io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");
  return io;
}

int main() {
  GuiOption option{1024, 1024, "Mesh View"};
  std::unique_ptr<OpenglGui> gui = std::make_unique<OpenglGui>(option);
  auto &io = initImGui(gui->window->window());
  ShaderProgramConfig config{
      .vertex_shader_path = BUILTIN_SHADER_DIR"/perspective.vert",
      .fragment_shader_path = BUILTIN_SHADER_DIR"/blinn-phong.frag"
  };
  ShaderProgram shader(config);
  Camera camera(defaultCameraParams());
  FpsCameraController controller(camera, *gui->window, 0.1, 0.1);
  controller.registerInputListeners();
  DrawableMesh mesh(shader.attributeLayout());
  gui->render([&]() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // ImGui code goes here
    ImGui::Begin("FPS");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / io.Framerate, io.Framerate);
    ImGui::End();
    ImGui::Render();
    glCheckError(glClearColor(1.0f, 1.0f, 1.0f, 1.0f));
    glCheckError(glClear(GL_COLOR_BUFFER_BIT));
    glCheckError(glViewport(0, 0, gui->window->width(), gui->window->height()));
    shader.use();
    shader.setUniform("color", 1.0f, 0.1f, 0.1f);
    glCheckError(glDrawArrays(GL_TRIANGLES, 0, 3));
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  });
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}