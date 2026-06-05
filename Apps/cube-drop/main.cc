//
// SimCraft Example: Cube Free Fall (C++)
// =======================================
//
// A cube mesh falls under gravity and collides with a ground plane.
// Equivalent to: python/examples/cube_drop.py
//
// Demonstrates:
//   TetMesh loading → material creation → System setup →
//   kinematic ground plane → IPC integrator → realtime rendering.
//
// Usage:
//   cube-drop [--dt 0.01] [--steps 500] [--no-render]
//

#include <cxxopts.hpp>
#include <fem/primitives/tet-mesh.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/primitive.h>
#include <fem/system.h>
#include <fem/colliders.h>
#include <fem/ipc/implicit-euler.h>
#include <Maths/block-solvers/block-pcg.h>
#include <Deform/strain-energy-density.h>
#include <Renderer/renderer.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <format>

using namespace sim;
using namespace sim::fem;
using namespace sim::deform;

// ─── Scene Proxy Builder ───────────────────────────────────────────────────────

/// 计算平滑法线
static void computeSmoothNormals(renderer::MeshProxy& mesh) {
  const auto& pos = mesh.positions;
  const auto& tris = mesh.triangles;

  mesh.normals.assign(pos.size(), glm::vec3(0.0f));

  for (const auto& tri : tris) {
    glm::vec3 e1 = pos[tri.y] - pos[tri.x];
    glm::vec3 e2 = pos[tri.z] - pos[tri.x];
    glm::vec3 fn = glm::cross(e1, e2);  // area-weighted

    mesh.normals[tri.x] += fn;
    mesh.normals[tri.y] += fn;
    mesh.normals[tri.z] += fn;
  }

  for (auto& n : mesh.normals) {
    float len = glm::length(n);
    n = (len > 1e-8f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
  }
}

/// 从 System 构建渲染帧
static std::unique_ptr<renderer::SceneProxy> buildSceneProxy(const System& system, int frame) {
  auto proxy = std::make_unique<renderer::SceneProxy>();
  proxy->frameIndex = frame;
  proxy->simulationTime = system.currentTime();

  for (int i = 0; i < static_cast<int>(system.primitives().size()); i++) {
    const auto& pr = system.primitives()[i];
    auto surfaceView = pr.getSurfaceView();
    auto vertCount = pr.getVertexCount();
    int dofStart = pr.getDofStart();

    renderer::MeshProxy mesh;
    mesh.name = "primitive_" + std::to_string(i);

    // double → float 位置转换
    mesh.positions.resize(vertCount);
    for (size_t v = 0; v < vertCount; v++) {
      const auto& pos = system.x[(dofStart / 3) + static_cast<int>(v)];
      mesh.positions[v] = glm::vec3(pos);
    }

    // 三角形索引
    mesh.triangles.resize(surfaceView.size());
    for (size_t t = 0; t < surfaceView.size(); t++) {
      auto tri = surfaceView[t];
      mesh.triangles[t] = {
          static_cast<unsigned>(tri.x),
          static_cast<unsigned>(tri.y),
          static_cast<unsigned>(tri.z)};
    }

    computeSmoothNormals(mesh);
    proxy->meshes.push_back(std::move(mesh));
  }

  return proxy;
}

// ─── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  cxxopts::Options options("cube-drop", "Cube free-fall with IPC collision");
  options.add_options()
      ("dt", "Timestep size", cxxopts::value<double>()->default_value("0.01"))
      ("steps", "Number of simulation steps", cxxopts::value<int>()->default_value("500"))
      ("no-render", "Disable rendering", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print help");
  auto args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const double dt = args["dt"].as<double>();
  const int maxSteps = args["steps"].as<int>();
  const bool noRender = args["no-render"].as<bool>();

  // ─── 1. Mesh ────────────────────────────────────────────────────────────────
  auto meshOpt = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube10x10.tobj");
  if (!meshOpt) {
    std::cerr << "Failed to load mesh: " FEM_TETS_DIR "/cube50x50.tobj" << std::endl;
    return 1;
  }
  auto tetMesh = std::move(*meshOpt);
  std::cout << std::format("Mesh: {} vertices, {} tets\n",
                           tetMesh.getVertices().size(), tetMesh.tets.size());

  // ─── 2. Material ────────────────────────────────────────────────────────────
  // NeoHookean: Young's modulus = 1e5, Poisson's ratio = 0.4
  auto energy = std::make_unique<StableNeoHookean<Real>>(
      ElasticityParameters<Real>{.E = 1e5, .nu = 0.4});

  // ─── 3. System ──────────────────────────────────────────────────────────────
  System system;

  // 弹性体 = mesh + material + density
  ElasticTetMesh elasticBody(std::move(tetMesh), std::move(energy), /*density=*/1000.0);
  system.addPrimitive(Primitive(std::move(elasticBody)));

  // 重力
  system.setGravity({0.0, -9.81, 0.0});

  // ─── 4. Ground Plane ────────────────────────────────────────────────────────
  // y = -1 平面 (法线朝上, offset = -1)
  {
    Collider ground;
    glm::dvec3 normal(0.0, 1.0, 0.0);
    double offset = -1.0;

    Collider::SDFGeometry sdf;
    sdf.signedDistance = [normal, offset](const glm::dvec3& p) -> Real {
      return glm::dot(normal, p) - offset;
    };
    sdf.gradient = [normal](const glm::dvec3&) -> glm::dvec3 {
      return normal;
    };

    ground.geometry = std::move(sdf);
    ground.motion = staticMotion();
    system.colliders().push_back(std::move(ground));
  }

  // 初始化系统（分配 DOF、构建质量矩阵等）
  system.init();

  // ─── 5. Integrator ──────────────────────────────────────────────────────────
  IpcIntegrator::Config ipcConfig;
  ipcConfig.dHat = 1e-3;
  ipcConfig.contactStiffness = 1e8;  // kappa

  auto integrator = std::make_unique<IpcImplicitEuler>(system, ipcConfig);
  // 必须手动设置线性求解器 — 直接构造绕过了工厂方法
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(/*maxIter=*/1000, /*tol=*/1e-6);

  std::cout << std::format("System: {} DOF, dt = {}, max_steps = {}\n",
                           system.dof(), dt, maxSteps);
  std::cout << std::format("IPC: dHat = {}, kappa = {}\n",
                           ipcConfig.dHat, ipcConfig.contactStiffness);

  // ─── 6. Run ─────────────────────────────────────────────────────────────────

  if (noRender) {
    // 无渲染模式：直接跑模拟
    for (int step = 0; step < maxSteps; step++) {
      integrator->step(dt);
      system.advanceTime(dt);
      if (step % 50 == 0) {
        std::cout << std::format("Step {:4d}, t = {:.4f}, E = {:.6f}\n",
                                 step, system.currentTime(), system.totalEnergy());
      }
    }
    std::cout << "Done.\n";
    return 0;
  }

  // 有渲染模式：模拟线程 + 主线程渲染
  auto renderer = renderer::createRenderer({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "SimCraft - Cube Drop",
  });

  std::atomic<int> stepsCompleted{0};
  std::thread simThread([&]() {
    for (int step = 0; step < maxSteps && renderer->isRunning(); step++) {
      integrator->step(dt);
      system.advanceTime(dt);
      system.advanceKinematicBodies(system.currentTime());

      auto proxy = buildSceneProxy(system, step);
      renderer->queue().push(std::move(proxy));
      stepsCompleted = step + 1;

      if (step % 50 == 0) {
        std::cout << std::format("Step {:4d}, t = {:.4f}\n", step, system.currentTime());
      }
    }
    renderer->shutdown();
  });

  // 渲染在主线程
  renderer->runOnCurrentThread();

  simThread.join();
  std::cout << std::format("Done. {} steps completed.\n", stepsCompleted.load());
  return 0;
}
