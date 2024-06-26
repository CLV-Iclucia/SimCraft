#include <Core/core.h>
#include <Core/animation.h>
#include <ogl-render/resource-handles.h>
#include <ogl-render/camera.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <iostream>
#include <memory>
#include <FluidSim/cuda/smoke-simulator.cuh>
using namespace opengl;
bool initGLFW(GLFWwindow *&window) {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window = glfwCreateWindow(1280, 1280, "FLUID-GL", nullptr, nullptr);
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
  int resolution = 64;
  int exportFrame = -1;
  core::Vec3f density = core::Vec3f(0.3f, 0.5f, 1.0f);
  core::Vec3f albedo = core::Vec3f(0.2f, 0.3f, 0.8f);
  std::string outputDir = "./smoke_b/";
} opt;

void processWindowInput(GLFWwindow *window, FpsCamera &camera) {
  static float lastFrame = 0.f;
  float currentFrame = glfwGetTime();
  float deltaTime = currentFrame - lastFrame;
  lastFrame = currentFrame;
  camera.processKeyBoard(window, deltaTime);
}

Options parseOptions(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-r") {
      opt.resolution = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "-e") {
      opt.exportFrame = std::atoi(argv[++i]);
    } else if (std::string(argv[i]) == "-o") {
      opt.outputDir = std::string(argv[++i]);
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;
      return opt;
    }
  }
  return opt;
}

// two triangles for a screen quad
std::vector<float> screen{
    -1.f, 1.f,
    1.f, 1.f,
    -1.f, -1.f,
    1.f, 1.f,
    1.f, -1.f,
    -1.f, -1.f,
};

void exportVolume(const std::vector<float> &density,
                  const std::vector<float> &T,
                  const std::string &filename,
                  const Options &opt) {
  std::string file = opt.outputDir + "/" + filename;
  std::ofstream outFile(file);
  outFile << opt.resolution << " " << opt.resolution << " " << opt.resolution <<
          std::endl;
  for (int i = 0; i < opt.resolution * opt.resolution * opt.resolution; i++) {
    outFile << density[i] * opt.density.x << " " << density[i] * opt.density.y
            << " "
            << density[i] * opt.density.z << std::endl;
  }

  for (int i = 0; i < opt.resolution * opt.resolution * opt.resolution; i++) {
    outFile << opt.albedo.x * density[i] << " " << opt.albedo.y * density[i] << " " << opt.albedo.z * density[i]
            << std::endl;
  }
  outFile << std::endl;
  outFile.close();
}

static void setFluidRegion(std::vector<uint8_t> &fluid_region, int resolution) {
  for (int i = 0; i < resolution; i++) {
    for (int j = 0; j < resolution; j++) {
      for (int k = 0; k < resolution; k++) {
        double x = (i + 0.5) / static_cast<double>(resolution);
        double y = (j + 0.5) / static_cast<double>(resolution);
        double z = (k + 0.5) / static_cast<double>(resolution);
        double r = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) +
            (z - 0.5) * (z - 0.5);
        if (r <= 0.25)
          fluid_region[i * resolution * resolution + j * resolution + k] = 1;
      }
    }
  }
}

int main(int argc, char **argv) {
  Options opt = parseOptions(argc, argv);
  GLFWwindow *window;
  if (!initGLFW(window)) return -1;
  auto simulator = std::make_unique<fluid::cuda::GpuSmokeSimulator>(opt.resolution);
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }
  std::vector<uint8_t> fluid_region(opt.resolution * opt.resolution *
      opt.resolution, 0);
  setFluidRegion(fluid_region, opt.resolution);
  simulator->setActiveRegion(fluid_region);
  ShaderProgram smokeShader(FLUIDSIM_SHADER_DIR"/default.vs",
                         FLUIDSIM_SHADER_DIR"/smoke.fs");
  std::unique_ptr<OpenGLContext> smokeCtx = std::make_unique<OpenGLContext>();
  smokeCtx->vao.bind();
  smokeCtx->newAttribute("aPos", screen, 2, 2 * sizeof(float), GL_FLOAT);
  GLuint smokeTex;
  glGenTextures(1, &smokeTex);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, smokeTex);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  std::vector<float> buffer(opt.resolution * opt.resolution * opt.resolution);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, opt.resolution, opt.resolution,
               opt.resolution, 0, GL_RED, GL_FLOAT, buffer.data());
  core::Frame frame;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    simulator->step(frame);
    simulator->rho->copyTo(buffer.data());
//    if (frame.idx >= 100 && frame.idx < 400) {
//      std::vector<float> T(opt.resolution * opt.resolution * opt.resolution);
//      std::string filename = "frame_" + std::to_string(frame.idx) + ".vol";
//      exportVolume(buffer, T, filename, opt);
//      std::cout << "exporting frame " << frame.idx << " to directory " << opt.outputDir << std::endl;
//      if (frame.idx == 399) {
//        std::cout << "done exporting" << std::endl;
//        break;
//      }
//    }
    std::cout << "frame " << frame.idx << std::endl;
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, opt.resolution, opt.resolution,
                    opt.resolution, GL_RED,
                    GL_FLOAT, buffer.data());
    // check the buffer
    // if there is non-zero value, print the coordinates and value
    smokeShader.use();
    smokeShader.setInt("smokeTex", 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glfwSwapBuffers(window);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}