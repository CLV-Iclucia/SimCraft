#include <Core/core.h>
#include <Core/animation.h>
#include <Core/mesh.h>
#include <ogl-render/ogl-ctx.h>
#include <ogl-render/camera.h>
#include <FluidSim/cpu/advect-solver.h>
#include <FluidSim/cpu/project-solver.h>
#include <FluidSim/cpu/fluid-simulator.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <iostream>
#include <memory>
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
  core::Vec3d size = core::Vec3d(1.0);
  core::Vec3i resolution = core::Vec3i(64);
  std::string colliderPath = std::format("{}/spot.obj", SIMCRAFT_ASSETS_DIR);
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
    } else {
      std::cerr << std::format("Unknown option: {}", argv[i]) << std::endl;
      return opt;
    }
  }
  return opt;
}

int main(int argc, char** argv) {
  auto [nParticles, size, resolution, colliderPath] = parseOptions(argc, argv);
  GLFWwindow* window;
  if (!initGLFW(window)) return -1;
  ImGuiIO& io = initImGui(window);

  std::unique_ptr<fluid::HybridFluidSimulator3D> simulator =
      std::make_unique<fluid::HybridFluidSimulator3D>(
          nParticles, size, resolution);

  core::Mesh colliderMesh;

  if (!myLoadObj(colliderPath, &colliderMesh)) {
    std::cerr << "Failed to load mesh: " << colliderPath << std::endl;
    return -1;
  }

  simulator->buildCollider(colliderMesh);
  std::unique_ptr<fluid::HybridAdvectionSolver3D> advector = std::make_unique<
    fluid::PicAdvector3D>(nParticles, resolution.x, resolution.y,
                          resolution.z);
  std::unique_ptr<fluid::FvmSolver3D> projector = std::make_unique<
    fluid::FvmSolver3D>(resolution.x, resolution.y, resolution.z);
  std::unique_ptr<fluid::CgSolver3D> micpcg = std::make_unique<
    fluid::CgSolver3D>(resolution.x, resolution.y, resolution.z);
  std::unique_ptr<fluid::Preconditioner3D> preconder = std::make_unique<
    fluid::ModifiedIncompleteCholesky3D>(resolution.x, resolution.y,
                                         resolution.z);
  std::unique_ptr<fluid::ParticleSystemReconstructor<core::Real, 3>>
      reconstructor = std::make_unique<fluid::NaiveReconstructor<
        core::Real, 3>>(nParticles, resolution.x, resolution.y, resolution.z,
                        size);

  micpcg->setPreconditioner(preconder.get());
  projector->setCompressedSolver(micpcg.get());
  simulator->setAdvector(advector.get());
  simulator->setProjector(projector.get());
  simulator->setReconstructor(reconstructor.get());

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }
  ShaderProg fluidShader(
      std::format("{}/perspective.vs", FLUIDSIM_SHADER_DIR).c_str(),
      std::format("{}/point.fs", FLUIDSIM_SHADER_DIR).c_str());
  fluidShader.initAttributeHandles();
  fluidShader.initUniformHandles();
  ShaderProg colliderShader(
      std::format("{}/perspective.vs", FLUIDSIM_SHADER_DIR).c_str(),
      std::format("{}/normal.fs", FLUIDSIM_SHADER_DIR).c_str());
  colliderShader.initAttributeHandles();
  colliderShader.initUniformHandles();

  OpenGLContext fluidCtx, colliderCtx;
  fluidCtx.vao.bind();
  fluidCtx.newAttribute("aPos", simulator->positions(), 3, 3 * sizeof(double),
                        GL_DOUBLE);
  colliderCtx.vao.bind();
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
    glDrawArrays(GL_POINTS, 0, nParticles);
    colliderCtx.vao.bind();
    colliderShader.use();
    colliderShader.setMat4f("view", camera.getViewMatrix());
    colliderShader.setMat4f("model", model);
    colliderShader.setMat4f(
        "proj", camera.getProjectionMatrix(display_w, display_h));
    opengl::OpenGLContext::draw(GL_TRIANGLES, colliderMesh.indices.size());
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}