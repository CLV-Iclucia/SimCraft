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
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10000; i++)
    glBindBuffer(GL_ARRAY_BUFFER, gpu_buffer);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken for 10000 calls to glBindBuffer: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "ns\n";
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10000; i++)
    glBufferData(GL_ARRAY_BUFFER, buffer_size, buffer.data(), GL_STATIC_DRAW);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken for 10000 calls to glBufferData: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "ns\n";
}