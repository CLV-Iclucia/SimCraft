#include <ogl-render/ogl-gui.h>
#include <chrono>
using namespace opengl;
int main() {
  GuiOption option{1024, 1024, "Perf Lab"};
  std::unique_ptr<OpenglGui> gui = std::make_unique<OpenglGui>(option);
  int buffer_size = 4000000;
  std::vector<std::byte> buffer(buffer_size);
  GLuint gpu_buffer;
  glGenBuffers(1, &gpu_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, gpu_buffer);
  glBufferData(GL_ARRAY_BUFFER, buffer_size, buffer.data(), GL_STATIC_DRAW);
  glDeleteBuffers(1, &gpu_buffer);
}