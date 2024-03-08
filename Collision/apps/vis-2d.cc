//
// Created by creeper on 2/22/24.
//
#include <collision/lbvh.h>
#include <memory>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Core/rand-gen.h>

void drawAABB(const collision::AABB<core::Real, 2>& aabb) {
  glBegin(GL_LINE_LOOP);
  glVertex2f(aabb.lo.x, aabb.lo.y);
  glVertex2f(aabb.lo.x, aabb.hi.y);
  glVertex2f(aabb.hi.x, aabb.lo.y);
  glVertex2f(aabb.hi.x, aabb.hi.y);
  glEnd();
}

void drawTriangle(const core::Vector<core::Real, 2>& a,
                  const core::Vector<core::Real, 2>& b,
                  const core::Vector<core::Real, 2>& c) {
  glBegin(GL_LINE_LOOP);
  glVertex2f(a.x, a.y);
  glVertex2f(b.x, b.y);
  glVertex2f(c.x, c.y);
  glEnd();
}

void recursiveDrawBVH(const collision::LBVH<core::Real, 2>& lbvh, int nodeIdx) {
  std::cout << std::format("Node {}: lo: ({}, {}), hi: ({}, {})\n", nodeIdx,
                           lbvh.aabb(nodeIdx).lo.x, lbvh.aabb(nodeIdx).lo.y,
                           lbvh.aabb(nodeIdx).hi.x, lbvh.aabb(nodeIdx).hi.y);
  drawAABB(lbvh.aabb(nodeIdx));
  if (lbvh.isLeaf(nodeIdx))
    return;
  recursiveDrawBVH(lbvh, lbvh.lchild(nodeIdx));
  recursiveDrawBVH(lbvh, lbvh.rchild(nodeIdx));
}

void drawBVH(const collision::LBVH<core::Real, 2>& lbvh) {
  // recursiveDrawBVH(lbvh, 0);
  glBegin(GL_LINE_LOOP);
  glVertex2f(0, 0);
  glVertex2f(0.5, 0);
  glVertex2f(0.5, 0.5);
  glVertex2f(0, 0.5);
  glEnd();
}

bool initGLFW(GLFWwindow*& window) {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window = glfwCreateWindow(1280, 720, "LBVH-VIS", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  return true;
}

int main() {
  GLFWwindow* window;
  if (!initGLFW(window))
    return -1;

  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // random 10 AABBs
  std::vector<core::AABB<core::Real, 2>> aabbs;
  for (int i = 0; i < 5; ++i) {
    core::Vector<core::Real, 2> lo = core::randomVec<core::Real, 2>();
    core::Vector<core::Real, 2> hi = core::randomVec<core::Real, 2>();
    // adjust lo and hi
    for (int j = 0; j < 2; ++j)
      if (lo[j] > hi[j])
        std::swap(lo[j], hi[j]);
    aabbs.emplace_back(lo, hi);
    std::cout << std::format("AABB {}: lo: ({}, {}), hi: ({}, {})\n", i, lo.x,
                             lo.y, hi.x, hi.y);
  }
  auto lbvh = std::make_unique<collision::LBVH<core::Real, 2>>(5);
  lbvh->refit(aabbs);
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glColor3f(1.0, 0.0, 0.0);
    drawBVH(*lbvh);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}