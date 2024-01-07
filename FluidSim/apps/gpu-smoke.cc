#include <Core/core.h>
#include <Core/animation.h>
#include <Core/mesh.h>
#include <ogl-render/ogl-ctx.h>
#include <ogl-render/camera.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <iostream>
#include <memory>

#include "..\include\FluidSim\gpu\smoke-simulator.cuh"
using namespace opengl;
ImVec4 clear_color = ImVec4(0.0f, 1.0f, 1.0f, 1.00f);
bool initGLFW(GLFWwindow*& window) {
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

ImGuiIO& initImGui(GLFWwindow* window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");
  return io;
}

struct Options {
  int nParticles = 0;
  core::Real density = 1e3;
  core::Vec3d size = core::Vec3d(1.0);
  core::Vec3i resolution = core::Vec3i(64);
  std::string colliderPath;
} opt;

void processWindowInput(GLFWwindow* window, FpsCamera& camera) {
  static float lastFrame = 0.f;
  float currentFrame = glfwGetTime();
  float deltaTime = currentFrame - lastFrame;
  lastFrame = currentFrame;
  camera.processKeyBoard(window, deltaTime);
}

Options parseOptions(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-n") {
      opt.nParticles = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "-r") {
      opt.resolution = core::Vec3i(std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "-s") {
      opt.size = core::Vec3d(std::stod(argv[++i]));
    } else if (std::string(argv[i]) == "-c") {
      opt.colliderPath = std::string(argv[++i]);
    } else if (std::string(argv[i]) == "-d") {
      opt.density = std::stod(argv[++i]);
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;
      return opt;
    }
  }
  if (opt.nParticles == 0)
    opt.nParticles = opt.resolution.x * opt.resolution.y * opt.resolution.z;
  if (opt.colliderPath.empty())
    opt.colliderPath = SIMCRAFT_ASSETS_DIR"/spot.obj";
  return opt;
}

int main(int argc, char** argv) {
  Options opt = parseOptions(argc, argv);
  GLFWwindow* window;
  if (!initGLFW(window)) return -1;
  ImGuiIO& io = initImGui(window);

  std::unique_ptr<fluid::GpuSmokeSimulator> simulator =
      std::make_unique<fluid::GpuSmokeSimulator>(
          opt.nParticles, opt.size, opt.resolution, opt.density);

  core::Mesh colliderMesh;

  if (!myLoadObj(opt.colliderPath, &colliderMesh)) {
    std::cerr << "Failed to load mesh: " << opt.colliderPath << std::endl;
    return -1;
  }

  simulator->buildCollider(colliderMesh);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  ShaderProg fluidShader(FLUIDSIM_SHADER_DIR"/perspective.vs",
                         FLUIDSIM_SHADER_DIR"/point.fs");
  fluidShader.initAttributeHandles();
  fluidShader.initUniformHandles();
  ShaderProg colliderShader(FLUIDSIM_SHADER_DIR"/perspective-mesh.vs",
                            FLUIDSIM_SHADER_DIR"/normal.fs");
  colliderShader.initAttributeHandles();
  colliderShader.initUniformHandles();

  OpenGLContext fluidCtx, colliderCtx;
  fluidCtx.vao.bind();
  fluidCtx.newAttribute("aPos", simulator->positions(), 3, 3 * sizeof(double),
                        GL_DOUBLE);
  colliderCtx.vao.bind();

  // colliderCtx.newAttribute("aPos", simulator->colliderPositions(), 3,
  //                          3 * sizeof(double), GL_DOUBLE);
  core::Frame frame;
  FpsCamera camera;
  core::Mat4f model;
  model[0][0] = model[1][1] = model[2][2] = model[3][3] = 1.f;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame(); {
      ImGui::Begin("Hello, fluid simulation!");
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / io.Framerate, io.Framerate);
      ImGui::End();
    }
    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    simulator->step(frame);
    fluidCtx.vao.bind();
    fluidCtx.vbo[fluidCtx.attribute("aPos")].passData(simulator->positions());
    fluidShader.use();
    glEnable(GL_PROGRAM_POINT_SIZE);
    fluidShader.setMat4f("view", camera.getViewMatrix());
    fluidShader.setMat4f("model", model);
    fluidShader.setMat4f(
        "proj", camera.getProjectionMatrix(display_w, display_h));
    glDrawArrays(GL_POINTS, 0, opt.nParticles);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}