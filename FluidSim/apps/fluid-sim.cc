#include <Core/core.h>
#include <Core/animation.h>
#include <Core/mesh.h>
#include <ogl-render/resource-handles.h>
#include <ogl-render/camera.h>
#include <FluidSim/fluid-simulator.h>
#include <FluidSim/rebuild-surface.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <iostream>
#include <memory>
using namespace opengl;
using namespace core;
Vec4d kClearColor = Vec4d(1.0f, 1.0f, 1.0f, 1.00f);

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

struct Options {
  int nParticles = 65536;
  core::Vec3d size = core::Vec3d(1.0);
  core::Vec3i resolution = core::Vec3i(64);
  std::string colliderPath = std::format("{}/complex_bunny.obj",
                                         SIMCRAFT_ASSETS_DIR);
} opt;

void processWindowInput(GLFWwindow* window, TargetCamera& camera) {
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

void relocateMesh(core::Mesh& mesh, const core::Vec3d& size) {
  core::Real max_coord = -1e9, min_coord = 1e9;
  for (const auto& v : mesh.vertices) {
    max_coord = std::max(max_coord, std::max(std::max(v.x, v.y), v.z));
    min_coord = std::min(min_coord, std::min(std::min(v.x, v.y), v.z));
  }
  core::Real scene_scale = std::min(std::min(size.x, size.y), size.z);
  core::Real scale = scene_scale / (max_coord - min_coord);
  for (auto& v : mesh.vertices) {
    v = (v - core::Vec3d(min_coord)) * scale;
    v *= 0.5;
    assert(
        v.x >= 0.0 && v.x <= scene_scale && v.y >= 0.0 && v.y <= scene_scale &&
        v.z >= 0.0 && v.z <= scene_scale);
  }
}

std::tuple<std::unique_ptr<OpenGLContext>, std::unique_ptr<ShaderProgram>>
initSdfRender(const fluid::SDF<3>& sdf) {
  auto shader = std::make_unique<ShaderProgram>(
      std::format("{}/perspective-sdf.vs", FLUIDSIM_SHADER_DIR).c_str(),
      std::format("{}/sdf.fs", FLUIDSIM_SHADER_DIR).c_str());
  auto ctx = std::make_unique<OpenGLContext>();
  ctx->vao.bind();
  ctx->newAttribute("aPos", sdf.positionSamples(), 3, 3 * sizeof(double),
                    GL_DOUBLE);
  ctx->newAttribute("aColor", sdf.fieldSamples(), 3, 3 * sizeof(double),
                    GL_DOUBLE);
  return {std::move(ctx), std::move(shader)};
}

std::tuple<std::unique_ptr<OpenGLContext>, std::unique_ptr<ShaderProgram>>
initFluidRender(const std::vector<core::Vec3d>& particles) {
  auto shader = std::make_unique<ShaderProgram>(
      std::format("{}/perspective.vs", FLUIDSIM_SHADER_DIR).c_str(),
      std::format("{}/point.fs", FLUIDSIM_SHADER_DIR).c_str());
  auto ctx = std::make_unique<OpenGLContext>();
  ctx->vao.bind();
  ctx->newAttribute("aPos", particles, 3, 3 * sizeof(double), GL_DOUBLE);
  return {std::move(ctx), std::move(shader)};
}

std::tuple<std::unique_ptr<OpenGLContext>, std::unique_ptr<ShaderProgram>>
initColliderRender(const core::Mesh& mesh) {
  auto shader = std::make_unique<ShaderProgram>(
      std::format("{}/perspective-mesh.vs", FLUIDSIM_SHADER_DIR).c_str(),
      std::format("{}/fluidRegion.fs", FLUIDSIM_SHADER_DIR).c_str());
  auto ctx = std::make_unique<OpenGLContext>();
  ctx->vao.bind();
  ctx->newAttribute("aPos", mesh.vertices, 3, 3 * sizeof(double),
                    GL_DOUBLE);
  ctx->ebo.bind();
  ctx->ebo.initAttributeData(mesh.indices);
  return {std::move(ctx), std::move(shader)};
}

void drawFluid(OpenGLContext* fluidCtx, ShaderProgram* fluidShader,
               TargetCamera& camera, const std::vector<core::Vec3d>& positions,
               int display_w, int display_h) {
  core::Mat4f model;
  model[0][0] = model[1][1] = model[2][2] = model[3][3] = 1.f;
  glEnable(GL_DEPTH_TEST);
  fluidCtx->vao.bind();
  fluidCtx->VBO("aPos").bind();
  fluidCtx->VBO("aPos").initAttributeData(positions);
  fluidShader->use();
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  fluidShader->setMat4f("view", camera.getViewMatrix());
  fluidShader->setMat4f("model", model);
  fluidShader->setMat4f(
      "proj", camera.getProjectionMatrix(display_w, display_h));
  glDrawArrays(GL_POINTS, 0, positions.size());
}

void drawCollider(OpenGLContext* colliderCtx, ShaderProgram* meshShader,
                  TargetCamera& camera, const core::Mesh& colliderMesh,
                  int display_w, int display_h) {
  core::Mat4f model;
  model[0][0] = model[1][1] = model[2][2] = model[3][3] = 1.f;
  colliderCtx->vao.bind();
  meshShader->use();
  meshShader->setMat4f("view", camera.getViewMatrix());
  meshShader->setMat4f("model", model);
  meshShader->setMat4f(
      "proj", camera.getProjectionMatrix(display_w, display_h));
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  OpenGLContext::draw(GL_TRIANGLES, colliderMesh.indices.size());
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void drawSDF(OpenGLContext* sdfCtx, ShaderProgram* sdfShader,
             TargetCamera& camera, const fluid::SDF<3>& sdf, int display_w,
             int display_h) {
  core::Mat4f model;
  model[0][0] = model[1][1] = model[2][2] = model[3][3] = 1.f;
  glEnable(GL_DEPTH_TEST);
  sdfCtx->vao.bind();
  sdfCtx->VBO("aColor").bind();
  sdfCtx->VBO("aColor").initAttributeData(sdf.fieldSamples());
  sdfShader->use();
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  sdfShader->setMat4f("view", camera.getViewMatrix());
  sdfShader->setMat4f("model", model);
  sdfShader->setMat4f(
      "proj", camera.getProjectionMatrix(display_w, display_h));
  glDrawArrays(GL_POINTS, 0, sdf.sampleCount());
}

int main(int argc, char** argv) {
  auto [nParticles, size, resolution, colliderPath] = parseOptions(argc, argv);
  GLFWwindow* window;
  if (!initGLFW(window)) return -1;
  auto simulator = std::make_unique<fluid::cpu::FluidSimulator>(
      nParticles, size, resolution);
  core::Mesh colliderMesh;
  if (!myLoadObj(colliderPath, &colliderMesh)) {
    std::cerr << "Failed to load mesh: " << colliderPath << std::endl;
    return -1;
  }
  relocateMesh(colliderMesh, size);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }
  simulator->setCollider(colliderMesh);
  simulator->setAdvector(fluid::AdvectionMethod::PIC);
  simulator->setProjector(fluid::ProjectSolver::FVM);
  simulator->setInitialFluid(colliderMesh);
  simulator->setParticleReconstructor(fluid::ReconstructionMethod::Naive);
  simulator->setCompressedSolver(fluid::CompressedSolverMethod::CG,
                                 fluid::PreconditionerMethod::ModifiedIncompleteCholesky);
  simulator->reconstruct();
  simulator->smoothFluidSurface(5);
  simulator->extrapolateFluidSdf(10);

  // initial config: for all the sample points atmost one of the two sdf is negative
  // check this, if not true then it means the sdf is not properly initialized
  auto [fluidCtx, fluidShader] = initFluidRender(simulator->positions());
  // auto [sdfCtx, sdfShader] = initSdfRender(simulator->exportFluidSurface());
  auto [colliderCtx, meshShader] = initColliderRender(colliderMesh);

  core::Frame frame;
  TargetCamera camera(core::Vec3f(size.x, size.y, size.z) * 0.5f, 90.f);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    processWindowInput(window, camera);
    // Rendering
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(kClearColor.x, kClearColor.y, kClearColor.z, kClearColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    simulator->step(frame);
    drawFluid(fluidCtx.get(), fluidShader.get(), camera, simulator->positions(),
              display_w, display_h);
    if (frame.idx >= 80 && frame.idx < 140) {
      simulator->reconstruct();
      simulator->erodeFluidSurface(2);
      simulator->smoothFluidSurface(5);
      core::Mesh fluidMesh;
      fluid::cpu::rebuildSurface(fluidMesh, simulator->exportFluidSurface());
      if (!core::exportObj("fluid.obj", fluidMesh)) {
        std::cerr << "Failed to export fluid mesh" << std::endl;
        exit(-1);
      }
      if (frame.idx == 139) {
        std::cout << "Done." << std::endl;
        break;
      }
    }
    drawCollider(colliderCtx.get(), meshShader.get(), camera, colliderMesh,
                 display_w, display_h);
    glfwSwapBuffers(window);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}