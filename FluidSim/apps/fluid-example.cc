//
// Created by creeper on 23-9-1.
//
#include <Core/animation.h>
#include <Core/render/shader-prog.h>
#include <Core/render/buffer-obj.h>
#include <FluidSim/common/fluid-simulator.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <iostream>
#include <memory>
bool show_demo_window = true;
bool show_another_window = false;
ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
bool initGLFW(GLFWwindow *&window) {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window = glfwCreateWindow(1280, 720, "FLUID-GL", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  return true;
}

ImGuiIO &initImGui(GLFWwindow *window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");
  return io;
}
struct Options {
  // size and resolution of the grid
  // number of particles
  // 2D or 3D
  int nParticles = 1024;
  int dim = 2;
  core::Vec2f size = core::Vec2f(1.0f, 1.0f);
  core::Vec2i resolution = core::Vec2i(64, 64);
} opt;
int main(int argc, char **argv) {
  // parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-n") {
      opt.nParticles = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "-d") {
      opt.dim = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "-s") {
      opt.size = core::Vec2f(std::stof(argv[++i]), std::stof(argv[++i]));
    } else if (std::string(argv[i]) == "-r") {
      opt.resolution = core::Vec2i(std::stoi(argv[++i]), std::stoi(argv[++i]));
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;
      return -1;
    }
  }
  GLFWwindow *window;
  if (!initGLFW(window))
    return -1;
  ImGuiIO &io = initImGui(window);
  std::unique_ptr<fluid::ApicSimulator2D> simulator = std::make_unique<fluid::ApicSimulator2D>();
  simulator->init(opt.nParticles, opt.size, opt.resolution);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }
  // create shader program
  Core::ShaderProg shader("../FluidSim/shaders/vertex.glsl",
                              "../FluidSim/shaders/fragment.glsl");
  // create vertex array object
  core::VertexArrayObj vao;
  // create vertex buffer object
  core::VertexBufferObj vbo;
  // since we only draw points there is no need for element buffer object
  // pass data to vbo
  vbo.passData(simulator->positions());
  // bind vao
  vao.bind();
  // bind vbo
  vbo.bind();
  // set vertex attribute
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  shader.use();
  core::Frame frame;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (show_demo_window)
      ImGui::ShowDemoWindow(&show_demo_window);
    {
      static float f = 0.0f;
      static int counter = 0;
      ImGui::Begin("Hello, world!");
      ImGui::Text("This is some useful text.");
      ImGui::Checkbox("Demo Window", &show_demo_window);
      ImGui::Checkbox("Another Window", &show_another_window);
      ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
      ImGui::ColorEdit3("clear color", (float *)&clear_color);
      if (ImGui::Button("Button"))
        counter++;
      ImGui::SameLine();
      ImGui::Text("counter = %d", counter);
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / io.Framerate, io.Framerate);
      ImGui::End();
    }

    if (show_another_window) {
      ImGui::Begin("Another Window", &show_another_window);
      ImGui::Text("Hello from another window!");
      if (ImGui::Button("Close Me"))
        show_another_window = false;
      ImGui::End();
    }
    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glDrawArrays(GL_POINT, 0, opt.nParticles);
    glfwSwapBuffers(window);
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}