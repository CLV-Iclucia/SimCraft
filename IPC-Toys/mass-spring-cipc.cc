#include <iostream>
#include <Core/core.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
using namespace core;
using VecXd = Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<Real>;

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

struct CipcParams {
  Real dhat;
  Real s;
  Real xi;
};

struct SystemConfig {
  int nMassesPerRope;
  int nRopes;
  Real dt;
  Real mass;
  Real k;
  Real restLength;
} cfg;

struct Rope {
  int n;
  Real mass;
  std::vector<Vec2d> pos;
};

std::vector<Rope> ropes;

void addElement(std::vector<Eigen::Triplet<Real>>& triplets, int i, int j,
                Real val) {
  triplets.emplace_back(i, j, val);
}

Eigen::Matrix4d hessian(const SystemConfig& cfg, const Vec2d& pa,
                        const Vec2d& pb) {
  Eigen::Matrix4d H;
  // the hessian matrix of a spring
  Vec2d d = pb - pa;
  Real l = glm::length(d);
  Real l0 = cfg.restLength;
  Real k = cfg.k;
  Real lsqr = l * l;
  Real l0sqr = l0 * l0;
  H(0, 0) = k * (1 - l0 / l + l0 * lsqr / (l * l0sqr));
  H(0, 1) = k * (l0 * lsqr / (l * l0sqr));
  H(1, 0) = k * (l0 * lsqr / (l * l0sqr));
  H(1, 1) = k * (1 - l0 / l + l0 * lsqr / (l * l0sqr));
  H(0, 2) = -H(0, 0);
  H(0, 3) = -H(0, 1);
  H(1, 2) = -H(1, 0);
  H(1, 3) = -H(1, 1);
  H(2, 0) = -H(0, 0);
  H(2, 1) = -H(0, 1);
  H(2, 2) = H(0, 0);
  H(2, 3) = H(0, 1);
  H(3, 0) = -H(1, 0);
  H(3, 1) = -H(1, 1);
  H(3, 2) = H(1, 0);
  H(3, 3) = H(1, 1);
  H *= 1 / l;
  return H;
}

void assembleStiffness(std::vector<Eigen::Triplet<Real>>& triplets, int i,
                       int j, const Eigen::Matrix4d& H) {
  for (int bi = 0; bi < 2; bi++) {
    for (int bj = 0; bj < 2; bj++) {
      addElement(triplets, 2 * i + bi, 2 * j + bj, H(2 * bi, 2 * bj));
      addElement(triplets, 2 * i + bi, 2 * j + 1 + bj, H(2 * bi, 2 * bj + 1));
      addElement(triplets, 2 * i + 1 + bi, 2 * j + bj, H(2 * bi + 1, 2 * bj));
      addElement(triplets, 2 * i + 1 + bi, 2 * j + 1 + bj,
                 H(2 * bi + 1, 2 * bj + 1));
    }
  }
}

void step(Real dt) {
  Real t = 0;
  while (t < dt) {
    SparseMatrix M;
    std::vector<Eigen::Triplet<Real>> triplets;
    int n = cfg.nRopes * cfg.nMassesPerRope;
    for (int i = 0; i < n; i++)
      addElement(triplets, i, i, cfg.mass);
    VecXd rhs(n);
    VecXd qdot(n);
    VecXd q(n);
    for (int i = 0; i < cfg.nRopes; i++) {
      for (int j = 0; j < cfg.nMassesPerRope - 1; j++) {
        int idx = i * cfg.nMassesPerRope + j;
        auto H = hessian(cfg, ropes[i].pos[j], ropes[i].pos[j + 1]);
        assembleStiffness(triplets, idx, idx + 1, H);
      }
    }

  }
}

int main(int argc, char** argv) {
  std::string path(argv[1]);
  GLFWwindow* window;
  if (!initGLFW(window)) return -1;
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }
  while (!glfwWindowShouldClose(window)) {
  }
}