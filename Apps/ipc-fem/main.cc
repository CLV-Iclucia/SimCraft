//
// Created by creeper on 5/23/24.
//
#include <cxxopts.hpp>
#include <fem/fem-simulation.h>
#include <Renderer/renderer.h>
#include <iostream>
#include <thread>
#include <atomic>

using namespace sim;
using namespace sim::fem;

/// 从 System 构建渲染用的 SceneProxy
std::unique_ptr<renderer::SceneProxy> buildSceneProxy(const fem::System& system, int frame) {
  auto proxy = std::make_unique<renderer::SceneProxy>();
  proxy->frameIndex = frame;
  proxy->simulationTime = system.currentTime();

  // 遍历每个 Primitive，提取表面网格
  for (int i = 0; i < system.primitives().size(); i++) {
    const auto& pr = system.primitives()[i];
    auto surfaceView = pr.getSurfaceView();
    auto vertCount = pr.getVertexCount();
    int dofStart = pr.getDofStart();

    renderer::MeshProxy mesh;
    mesh.name = "primitive_" + std::to_string(i);

    // 将 double 位置转换为 float
    mesh.positions.resize(vertCount);
    for (size_t v = 0; v < vertCount; v++) {
      // system.x 是 BlockVector<3>，每个块是一个 glm::dvec3
      // dofStart 是标量索引，除以 3 得到块索引
      const auto& pos = system.x[(dofStart / 3) + static_cast<int>(v)];
      mesh.positions[v] = glm::vec3(pos);  // double → float
    }

    // 三角形索引（primitive 内部局部索引）
    mesh.triangles.resize(surfaceView.size());
    for (size_t t = 0; t < surfaceView.size(); t++) {
      auto tri = surfaceView[t];
      mesh.triangles[t] = {
          static_cast<unsigned>(tri.x),
          static_cast<unsigned>(tri.y),
          static_cast<unsigned>(tri.z)};
    }

    // 计算面法线（如果没有平滑法线）
    // TODO: 计算平滑法线
    mesh.normals.clear(); // 使用面法线或留空让 shader 处理

    proxy->meshes.push_back(std::move(mesh));
  }

  return proxy;
}

void checkArgs(const cxxopts::ParseResult& result) {
  if (!result.count("input")) {
    std::cerr << "Please specify input file" << std::endl;
    exit(1);
  }
}

int main(int argc, char** argv) {
  cxxopts::Options options("IPC FEM", "FEM Soft body simulator using IPC");
  options.add_options()
      ("i,input", "Input file", cxxopts::value<std::string>())
      ("no-render", "Disable rendering", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print help");
  auto result = options.parse(argc, argv);
  checkArgs(result);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  auto inputFile = result["input"].as<std::string>();
  auto simBuilder = FEMSimulationBuilder{};
  auto simConfig = core::loadJsonFile(inputFile);

  if (!simConfig) {
    std::cerr << "Failed to load simulation configuration from " << inputFile
              << std::endl;
    return 1;
  }

  core::Frame frame;
  auto femSim = simBuilder.build(*simConfig);

  bool useRenderer = !result["no-render"].as<bool>();

  if (!useRenderer) {
    // 无渲染模式：直接跑模拟
    while (frame.idx < 1000) {
      femSim.step(frame);
      if (frame.idx % 100 == 0) {
        std::cout << "Frame " << frame.idx << ", time = " << femSim.getSystem().currentTime() << std::endl;
      }
    }
    return 0;
  }

  // 有渲染模式：模拟线程 + 主线程渲染
  auto renderer = renderer::createRenderer({.windowTitle = "SimCraft - IPC FEM"});

  // 模拟在子线程
  std::atomic<bool> simFinished{false};
  std::thread simThread([&]() {
    core::Frame frame;
    while (frame.idx < 1000 && renderer->isRunning()) {
      femSim.step(frame);
      auto proxy = buildSceneProxy(femSim.getSystem(), frame.idx);
      renderer->queue().push(std::move(proxy));  // 阻塞提交（队列满则等渲染追上）

      if (frame.idx % 10 == 0) {
        std::cout << "Simulated frame " << frame.idx << ", time = " << femSim.getSystem().currentTime() << std::endl;
      }
    }
    renderer->shutdown();  // 模拟结束，通知渲染线程退出
    simFinished = true;
    std::cout << "Simulation finished" << std::endl;
  });

  // 渲染在主线程（满足 GLFW/macOS 要求）
  renderer->runOnCurrentThread();  // 阻塞直到 shutdown

  simThread.join();
  std::cout << "Application terminated" << std::endl;
  return 0;
}
