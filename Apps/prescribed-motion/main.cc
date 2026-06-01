//
// SimCraft Example: Prescribed Motion (C++)
// ==========================================
//
// A cube mesh has one end pinned and one vertex driven by sinusoidal motion.
// Equivalent to: python/examples/prescribed_motion.py
//
// Demonstrates:
//   TetMesh loading → material → System → pin constraints →
//   prescribed time-varying motion → IPC integrator → realtime rendering.
//
// Usage:
//   prescribed-motion [--dt 0.01] [--steps 300] [--no-render]
//

#include <cxxopts.hpp>
#include <fem/primitives/tet-mesh.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/primitive.h>
#include <fem/system.h>
#include <fem/constraints.h>
#include <fem/ipc/implicit-euler.h>
#include <Maths/block-solvers/block-pcg.h>
#include <Deform/strain-energy-density.h>
#include <Renderer/renderer.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <cmath>
#include <format>
#include <numbers>

using namespace sim;
using namespace sim::fem;
using namespace sim::deform;

// ─── Scene Proxy Builder ───────────────────────────────────────────────────────

static void computeSmoothNormals(renderer::MeshProxy& mesh) {
  const auto& pos = mesh.positions;
  const auto& tris = mesh.triangles;

  mesh.normals.assign(pos.size(), glm::vec3(0.0f));

  for (const auto& tri : tris) {
    glm::vec3 e1 = pos[tri.y] - pos[tri.x];
    glm::vec3 e2 = pos[tri.z] - pos[tri.x];
    glm::vec3 fn = glm::cross(e1, e2);

    mesh.normals[tri.x] += fn;
    mesh.normals[tri.y] += fn;
    mesh.normals[tri.z] += fn;
  }

  for (auto& n : mesh.normals) {
    float len = glm::length(n);
    n = (len > 1e-8f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
  }
}

static std::unique_ptr<renderer::SceneProxy> buildSceneProxy(
    const System& system, int frame, glm::vec3 objectColor) {
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
    mesh.objectColor = objectColor;

    mesh.positions.resize(vertCount);
    for (size_t v = 0; v < vertCount; v++) {
      const auto& pos = system.x[(dofStart / 3) + static_cast<int>(v)];
      mesh.positions[v] = glm::vec3(pos);
    }

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
  cxxopts::Options options("prescribed-motion",
      "Elastic body with pinned + prescribed sinusoidal motion");
  options.add_options()
      ("dt", "Timestep size", cxxopts::value<double>()->default_value("0.01"))
      ("steps", "Number of simulation steps", cxxopts::value<int>()->default_value("300"))
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
    std::cerr << "Failed to load mesh: " FEM_TETS_DIR "/cube10x10.tobj" << std::endl;
    return 1;
  }
  auto tetMesh = std::move(*meshOpt);
  const auto& verts = tetMesh.getVertices();
  std::cout << std::format("Mesh: {} vertices, {} tets\n", verts.size(), tetMesh.tets.size());

  // Find x_min face (pinned end) and x_max face (driven end)
  Real xMin = std::numeric_limits<Real>::max();
  Real xMax = std::numeric_limits<Real>::lowest();
  for (const auto& v : verts) {
    xMin = std::min(xMin, v.x());
    xMax = std::max(xMax, v.x());
  }
  Real eps = (xMax - xMin) * 1e-6;

  std::vector<int> pinnedFace, drivenFace;
  for (int i = 0; i < static_cast<int>(verts.size()); i++) {
    if (std::abs(verts[i].x() - xMin) < eps) pinnedFace.push_back(i);
    if (std::abs(verts[i].x() - xMax) < eps) drivenFace.push_back(i);
  }

  // ─── 2. Material ────────────────────────────────────────────────────────────
  // NeoHookean: Young's = 2e5, Poisson's = 0.45
  auto energy = std::make_unique<StableNeoHookean<Real>>(
      ElasticityParameters<Real>{.E = 2e5, .nu = 0.45});

  // ─── 3. System ──────────────────────────────────────────────────────────────
  System system;

  ElasticTetMesh elasticBody(std::move(tetMesh), std::move(energy), /*density=*/800.0);
  system.addPrimitive(Primitive(std::move(elasticBody)));

  system.setGravity({0.0, -9.81, 0.0});

  // 初始化系统（分配 DOF）
  system.init();

  // ─── 4. Constraints ─────────────────────────────────────────────────────────
  // Pin the entire x_min face
  system.constraints().pinVertices(pinnedFace, system.x);
  std::cout << std::format("Pinned {} vertices on x_min face\n", pinnedFace.size());

  // Prescribe sinusoidal y-motion on the entire x_max face
  constexpr double amplitude = 0.3;
  constexpr double freq = 1.0;  // Hz

  for (int vIdx : drivenFace) {
    system.constraints().prescribeMotion(vIdx,
        [amplitude, freq](Real t) -> glm::dvec3 {
          return glm::dvec3(
              0.0,
              amplitude * std::sin(2.0 * std::numbers::pi * freq * t),
              0.0);
        });
  }
  std::cout << std::format("Prescribed sinusoidal motion on {} vertices (x_max face)\n",
                           drivenFace.size());

  // Build constraint index
  system.constraints().build(system.x.numBlocks());

  // ─── 5. Integrator ──────────────────────────────────────────────────────────
  IpcIntegrator::Config ipcConfig;
  ipcConfig.dHat = 1e-3;
  ipcConfig.contactStiffness = 1e8;

  auto integrator = std::make_unique<IpcImplicitEuler>(system, ipcConfig);
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(1000, 1e-6);

  std::cout << std::format("System: {} DOF, dt = {}, max_steps = {}\n",
                           system.dof(), dt, maxSteps);

  // ─── 6. Run ─────────────────────────────────────────────────────────────────

  // 物体颜色：青色 (同 Python 例子)
  glm::vec3 color(0.35f, 0.80f, 0.80f);

  if (noRender) {
    for (int step = 0; step < maxSteps; step++) {
      integrator->step(dt);
      system.advanceTime(dt);
      if (step % 50 == 0) {
        std::cout << std::format("Step {:4d}, t = {:.4f}\n", step, system.currentTime());
      }
    }
    std::cout << "Done.\n";
    return 0;
  }

  // 有渲染模式
  auto renderer = renderer::createRenderer({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "SimCraft - Prescribed Motion",
  });

  std::atomic<int> stepsCompleted{0};
  std::thread simThread([&]() {
    for (int step = 0; step < maxSteps && renderer->isRunning(); step++) {
      integrator->step(dt);
      system.advanceTime(dt);

      auto proxy = buildSceneProxy(system, step, color);
      renderer->queue().push(std::move(proxy));
      stepsCompleted = step + 1;

      if (step % 50 == 0) {
        std::cout << std::format("Step {:4d}, t = {:.4f}\n", step, system.currentTime());
      }
    }
    renderer->shutdown();
  });

  renderer->runOnCurrentThread();

  simThread.join();
  std::cout << std::format("Done. {} steps completed.\n", stepsCompleted.load());
  return 0;
}
