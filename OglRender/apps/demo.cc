#include <ogl-render/shader-prog.h>
#include <ogl-render/ogl-ctx.h>
#include <imgui/imgui.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <glm/glm.hpp>

enum class InteractionMode {
  None,
  Draw,
  Edit,
};

struct UI {
  glm::vec3 color;
  int selectedPrimitive = -1;
  InteractionMode mode = InteractionMode::None;
};

enum class PrimitiveType {
  Rectangle,
  Triangle,
  Line,
};

struct Primitive {
  int startOffset, endOffset;
  PrimitiveType type;
  Primitive(int startOffset_, int endOffset_, PrimitiveType type_) :
      startOffset(startOffset_), endOffset(endOffset_), type(type_) {}
};

struct DrawBoard {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> colors;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs; // these two are not used by far
  std::vector<Primitive> primitives;
  std::vector<uint> idx;
  void addRectangle(float x, float y, float z, float width, float height, const glm::vec3 &color) {
    int num_vertices = positions.size();
    positions.emplace_back(x, y, z);
    positions.emplace_back(x + width, y, z);
    positions.emplace_back(x + width, y + height, z);
    positions.emplace_back(x, y + height, z);
    for (int i = 0; i < 4; ++i) {
      colors.push_back(color);
      normals.emplace_back(0.0f, 0.0f, 1.0f);
    }
    idx.push_back(num_vertices);
    idx.push_back(num_vertices + 1);
    idx.push_back(num_vertices + 2);
    idx.push_back(num_vertices);
    idx.push_back(num_vertices + 2);
    idx.push_back(num_vertices + 3);
    primitives.emplace_back(num_vertices, num_vertices + 4, PrimitiveType::Rectangle);
  }
  void addTriangle(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const glm::vec3 &color) {
    int num_vertices = positions.size();
    positions.emplace_back(a.x, a.y, a.z);
    positions.emplace_back(b.x, b.y, b.z);
    positions.emplace_back(c.x, c.y, c.z);
    for (int i = 0; i < 3; ++i) {
      colors.push_back(color);
      normals.emplace_back(0.0f, 0.0f, 1.0f);
    }
    idx.push_back(num_vertices);
    idx.push_back(num_vertices + 1);
    idx.push_back(num_vertices + 2);
    primitives.emplace_back(num_vertices, num_vertices + 3, PrimitiveType::Triangle);
  }
};

static DrawBoard board;
static UI ui;
static glm::vec2 cursor;

bool initGLFW(GLFWwindow *&window) {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window = glfwCreateWindow(1280, 720, "DEMO-GL", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  return true;
}

void initComputerOnDrawBoard() {
  // frame
  board.addRectangle(0.0f, 0.0f, -0.2f, 0.5f, 0.5f, glm::vec3(0.5f));
  // screen
  board.addRectangle(0.1f, 0.1f, -0.3f, 0.3f, 0.3f, glm::vec3(0.8f, 0.8f, 1.0f));
  // triangle
  board.addTriangle(glm::vec3(0.2f, 0.2f, -0.4f),
                    glm::vec3(0.25f, 0.25f, -0.4f),
                    glm::vec3(0.2f, 0.3f, -0.4f),
                    glm::vec3(1.0f, 1.0f, 0.0f));
  // Base
  board.addRectangle(0.2f, -0.3f, -0.2f, 0.1f, 0.3f, glm::vec3(0.5f));
  board.addRectangle(0.05f, -0.4f, -0.2f, 0.4f, 0.1f, glm::vec3(0.5f));
}

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

void drawUI(ImGuiIO &io) {
 // TODO: haven't implemented yet
}

struct Ray {
  glm::vec3 orig;
  glm::vec3 dir;
  Ray(const glm::vec3 &orig_, const glm::vec3 &dir_) : orig(orig_), dir(dir_) {}
};

bool rayTriangleIntersect(const Ray &ray,
                          const glm::vec3 &a,
                          const glm::vec3 &b,
                          const glm::vec3 &c) {
  glm::vec3 ab = b - a;
  glm::vec3 ac = c - a;
  glm::vec3 pvec = glm::cross(ac, ray.dir);
  float det = glm::dot(ab, pvec);
  if (det < 1e-8) return false;
  float inv_det = 1.0f / det;
  glm::vec3 tvec = ray.orig - a;
  float u = glm::dot(tvec, pvec) * inv_det;
  if (u < 0.0f || u > 1.0f) return false;
  glm::vec3 qvec = glm::cross(ab, tvec);
  float v = glm::dot(ray.dir, qvec) * inv_det;
  if (v < 0.0f || u + v > 1.0f) return false;
  return true;
}

int select() {
  int selected_primitive = -1;
  float depth_buf = 100.f;
  for (int i = 0; i < board.primitives.size(); i++) {
    const Primitive &pr = board.primitives[i];
    if (pr.type == PrimitiveType::Rectangle) {
      const auto &p1 = board.positions[pr.startOffset];
      const auto &p2 = board.positions[pr.startOffset + 1];
      const auto &p3 = board.positions[pr.startOffset + 2];
      const auto &p4 = board.positions[pr.startOffset + 3];
      auto &c1 = board.colors[pr.startOffset];
      if (rayTriangleIntersect
          (Ray(glm::vec3(cursor.x, cursor.y, -100.0f), glm::vec3(0.f, 0.f, 1.f)),
           p1, p2, p3) || rayTriangleIntersect
          (Ray(glm::vec3(cursor.x, cursor.y, -100.0f), glm::vec3(0.f, 0.f, 1.f)),
           p1, p3, p4)) {
        if (p1.z < depth_buf) {
          depth_buf = p1.z;
          selected_primitive = i;
          ui.color = c1;
        }
      }
    } else if (pr.type == PrimitiveType::Triangle) {
      const auto &p1 = board.positions[pr.startOffset];
      const auto &p2 = board.positions[pr.startOffset + 1];
      const auto &p3 = board.positions[pr.startOffset + 2];
      const auto &c1 = board.colors[pr.startOffset];
      if (rayTriangleIntersect
          (Ray(glm::vec3(cursor.x, cursor.y, -100.0f), glm::vec3(0.f, 0.f, 1.f)),
           p1, p2, p3)) {
        if (p1.z < depth_buf) {
          depth_buf = p1.z;
          selected_primitive = i;
          ui.color = c1;
        }
      }
    }
  }
  return selected_primitive;
}

void EditUI(ImGuiIO &io, opengl::OpenGLContext &ctx) {
  if (io.MouseClicked[0] && !io.WantCaptureMouse) {
    ui.selectedPrimitive = select();
  }
  if (ui.selectedPrimitive == -1) {
    ImGui::Text("No primitive selected");
    return;
  }
  ImGui::Text("Editing a %s",
              board.primitives[ui.selectedPrimitive].type == PrimitiveType::Rectangle ? "rectangle" : "triangle");
  // show a color picker to change the color of the selected primitive
  ImGui::ColorEdit3("Color", (float *) &ui.color);
  int startOffset = board.primitives[ui.selectedPrimitive].startOffset;
  int endOffset = board.primitives[ui.selectedPrimitive].endOffset;
  for (int i = startOffset; i < endOffset; i++)
    board.colors[i] = ui.color;
  ctx.vbo[ctx.attribute("aColor")].updateData(board.colors.data() + startOffset, startOffset,
                                              endOffset - startOffset);
}

int main() {
  GLFWwindow *window;
  if (!initGLFW(window)) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }
  ImGuiIO &io = initImGui(window);
  opengl::ShaderProg shader("/home/creeper/JeoCraft/OglRender/shaders/2d-default.vs",
                            "/home/creeper/JeoCraft/OglRender/shaders/default.fs");
  shader.initAttributeHandles();
  shader.initUniformHandles();
  initComputerOnDrawBoard();
  opengl::OpenGLContext ctx;
  ctx.vao.bind();
  ctx.newAttribute("aPos", board.positions, 3, 3 * sizeof(float), GL_FLOAT);
  ctx.newAttribute("aColor", board.colors, 3, 3 * sizeof(float), GL_FLOAT);
  ctx.ebo.bind();
  ctx.ebo.passData(board.idx);
  shader.use();
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    {
      cursor.x = (io.MousePos.x / width) * 2 - 1;
      cursor.y = -(io.MousePos.y / height) * 2 + 1;
      ImGui::Begin("Hello, computer graphics!");
      ImGui::Text("Mouse Cursor Position: %f %f", cursor.x, cursor.y);
      ImGui::Text("Mouse Interaction Mode:");
      if (ImGui::Button("None")) {
        ui.selectedPrimitive = -1;
        ui.mode = InteractionMode::None;
      }
      ImGui::SameLine();
      if (ImGui::Button("Draw")) {
        ui.selectedPrimitive = -1;
        ui.mode = InteractionMode::Draw;
      }
      ImGui::SameLine();
      if (ImGui::Button("Edit"))
        ui.mode = InteractionMode::Edit;
      if (ui.mode == InteractionMode::Draw)
        drawUI(io);
      if (ui.mode == InteractionMode::Edit)
        EditUI(io, ctx);
      ImGui::End();
    }
    // Rendering
    ImGui::Render();
    glViewport(0, 0, width, height);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    opengl::OpenGLContext::draw(GL_TRIANGLES, board.idx.size());
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }
  opengl::OpenGLContext::unbind();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}