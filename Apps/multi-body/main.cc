//
// SimCraft Example: Multi-Body Collision (C++)
// =============================================
//
// Two cubes fly toward each other in zero gravity and collide.
// Equivalent to: python/examples/multi_body.py
//
// Demonstrates:
//   - Multiple elastic bodies with initial velocities
//   - Zero gravity (pure momentum exchange)
//   - Elastic-elastic collision via IPC
//
// Usage:
//   multi-body [--dt 0.005] [--steps 500] [--no-render]
//

#include <cxxopts.hpp>
#include <fem/primitives/tet-mesh.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/primitive.h>
#include <fem/system.h>
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
    mesh.name = "body_" + std::to_string(i);

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
  cxxopts::Options options("multi-body", "Two cubes head-on collision with IPC");
  options.add_options()
      ("dt", "Timestep size", cxxopts::value<double>()->default_value("0.005"))
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

  // ─── 1. Load template mesh ──────────────────────────────────────────────────
  auto meshOpt = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube10x10.tobj");
  if (!meshOpt) {
    std::cerr << "Failed to load mesh: " FEM_TETS_DIR "/cube10x10.tobj" << std::endl;
    return 1;
  }
  auto templateMesh = std::move(*meshOpt);
  const auto& baseVerts = templateMesh.getVertices();
  const auto& baseTets = templateMesh.tets;
  const int nVerts = static_cast<int>(baseVerts.size());

  // Compute mesh width for spacing
  Real xMin = std::numeric_limits<Real>::max();
  Real xMax = std::numeric_limits<Real>::lowest();
  for (const auto& v : baseVerts) {
    xMin = std::min(xMin, v[0]);
    xMax = std::max(xMax, v[0]);
  }
  Real width = xMax - xMin;
  Real separation = 0.05;  // initial gap
  Real speed = 2.0;        // m/s

  std::cout << std::format("Template mesh: {} verts, {} tets, width = {:.4f}\n",
                           nVerts, baseTets.size(), width);

  // ─── 2. Material ────────────────────────────────────────────────────────────
  // NeoHookean: Young's modulus = 2e5, Poisson's ratio = 0.4
  auto makeEnergy = []() {
    return std::make_unique<StableNeoHookean<Real>>(
        ElasticityParameters<Real>{.E = 2e5, .nu = 0.4});
  };

  // ─── 3. System ──────────────────────────────────────────────────────────────
  System system;
  system.setGravity({0.0, 0.0, 0.0});  // zero gravity

  // Left cube: offset left, velocity → right (+x)
  {
    std::vector<Vector<Real, 3>> verts(nVerts);
    std::vector<Vector<Real, 3>> vels(nVerts);
    Real offsetX = -(width / 2.0 + separation / 2.0);
    for (int i = 0; i < nVerts; i++) {
      verts[i] = baseVerts[i];
      verts[i][0] += offsetX;
      vels[i] = Vector<Real, 3>(speed, 0.0, 0.0);
    }
    TetMesh leftMesh(verts, baseTets, vels);
    ElasticTetMesh leftBody(std::move(leftMesh), makeEnergy(), /*density=*/800.0);
    system.addPrimitive(Primitive(std::move(leftBody)));
    std::cout << std::format("  Left cube:  {} verts, vel = [+{}, 0, 0]\n", nVerts, speed);
  }

  // Right cube: offset right, velocity → left (-x)
  {
    std::vector<Vector<Real, 3>> verts(nVerts);
    std::vector<Vector<Real, 3>> vels(nVerts);
    Real offsetX = +(width / 2.0 + separation / 2.0);
    for (int i = 0; i < nVerts; i++) {
      verts[i] = baseVerts[i];
      verts[i][0] += offsetX;
      vels[i] = Vector<Real, 3>(-speed, 0.0, 0.0);
    }
    TetMesh rightMesh(verts, baseTets, vels);
    ElasticTetMesh rightBody(std::move(rightMesh), makeEnergy(), /*density=*/800.0);
    system.addPrimitive(Primitive(std::move(rightBody)));
    std::cout << std::format("  Right cube: {} verts, vel = [-{}, 0, 0]\n", nVerts, speed);
  }

  // 初始化系统
  system.init();
  std::cout << std::format("System: {} DOF, {} bodies\n", system.dof(), system.primitives().size());

  // ─── 4. Integrator ──────────────────────────────────────────────────────────
  IpcIntegrator::Config ipcConfig;
  ipcConfig.dHat = 2e-3;
  ipcConfig.contactStiffness = 1e9;  // kappa

  auto integrator = std::make_unique<IpcImplicitEuler>(system, ipcConfig);
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(1000, 1e-6);

  std::cout << std::format("IPC: dHat = {}, kappa = {}, dt = {}\n",
                           ipcConfig.dHat, ipcConfig.contactStiffness, dt);

  // ─── 5. Run ─────────────────────────────────────────────────────────────────

  if (noRender) {
    for (int step = 0; step < maxSteps; step++) {
      integrator->step(dt);
      if (step % 10 == 0) {
        std::cout << std::format("Step {:4d}, t = {:.4f}, T = {:.6e}, V = {:.6e}\n",
                                 step, system.currentTime(),
                                 system.kineticEnergy(), system.potentialEnergy());
      }
    }
    std::cout << "Done.\n";
    return 0;
  }

  // 有渲染模式
  auto renderer = renderer::createRenderer({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "SimCraft - Multi-Body Collision",
  });

  // Push initial frame
  renderer->queue().push(buildSceneProxy(system, 0));

  // Hook: push a frame after every Newton iteration so we can see intermediate states
  std::atomic<int> frameCounter{1};
  integrator->onNewtonIter = [&](int newtonIter) {
    if (!renderer->isRunning()) return;
    auto proxy = buildSceneProxy(system, frameCounter++);
    renderer->queue().push(std::move(proxy));
  };

  std::atomic<int> stepsCompleted{0};
  std::thread simThread([&]() {
    for (int step = 0; step < maxSteps && renderer->isRunning(); step++) {
      integrator->step(dt);

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
