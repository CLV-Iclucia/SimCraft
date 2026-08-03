#include <ABD/abd-scene.h>
#include <FEM/fem-scene.h>
#include <Renderer/simulation-app.h>
#include <Runtime/global-solver.h>

#include <array>
#include <cxxopts.hpp>
#include <iostream>
#include <memory>
#include <vector>

namespace {
constexpr int CLOTH_X_SEGMENTS = 64;
constexpr int CLOTH_Z_SEGMENTS = 40;
constexpr double CLOTH_WIDTH = 0.9;
constexpr double CLOTH_DEPTH = 0.52;
constexpr double CLOTH_HEIGHT = 0.62;

ksk::abd::ABDBodyDesc makeRigidBox(const glm::dvec3& half_extents,
                                   const glm::dvec3& center,
                                   const glm::dvec3& velocity) {
  ksk::abd::ABDBodyDesc body;
  const double x = half_extents.x;
  const double y = half_extents.y;
  const double z = half_extents.z;
  body.surface.vertices = {
      glm::dvec3{-x, -y, -z}, glm::dvec3{x, -y, -z},
      glm::dvec3{x, y, -z},   glm::dvec3{-x, y, -z},
      glm::dvec3{-x, -y, z},  glm::dvec3{x, -y, z},
      glm::dvec3{x, y, z},    glm::dvec3{-x, y, z},
  };
  body.surface.edges = {
      std::array<int, 2>{0, 1}, std::array<int, 2>{1, 2},
      std::array<int, 2>{2, 3}, std::array<int, 2>{3, 0},
      std::array<int, 2>{4, 5}, std::array<int, 2>{5, 6},
      std::array<int, 2>{6, 7}, std::array<int, 2>{7, 4},
      std::array<int, 2>{0, 4}, std::array<int, 2>{1, 5},
      std::array<int, 2>{2, 6}, std::array<int, 2>{3, 7},
  };
  body.surface.triangles = {
      std::array<int, 3>{0, 2, 1}, std::array<int, 3>{0, 3, 2},
      std::array<int, 3>{4, 5, 6}, std::array<int, 3>{4, 6, 7},
      std::array<int, 3>{0, 1, 5}, std::array<int, 3>{0, 5, 4},
      std::array<int, 3>{1, 2, 6}, std::array<int, 3>{1, 6, 5},
      std::array<int, 3>{2, 3, 7}, std::array<int, 3>{2, 7, 6},
      std::array<int, 3>{3, 0, 4}, std::array<int, 3>{3, 4, 7},
  };
  body.initialTransform[3] = center;
  body.initialVelocity[3] = velocity;
  body.density = 1000.0;
  return body;
}

ksk::engine::fem::ClothMeshDesc makeClothSheet() {
  ksk::engine::fem::ClothMeshDesc mesh;
  mesh.vertices.reserve((CLOTH_X_SEGMENTS + 1) * (CLOTH_Z_SEGMENTS + 1));
  for (int z = 0; z <= CLOTH_Z_SEGMENTS; ++z) {
    const double tz =
        static_cast<double>(z) / static_cast<double>(CLOTH_Z_SEGMENTS);
    for (int x = 0; x <= CLOTH_X_SEGMENTS; ++x) {
      const double tx =
          static_cast<double>(x) / static_cast<double>(CLOTH_X_SEGMENTS);
      mesh.vertices.emplace_back(-0.5 * CLOTH_WIDTH + CLOTH_WIDTH * tx,
                                 CLOTH_HEIGHT,
                                 -0.5 * CLOTH_DEPTH + CLOTH_DEPTH * tz);
    }
  }
  mesh.initialPositions = mesh.vertices;
  mesh.triangles.reserve(2 * CLOTH_X_SEGMENTS * CLOTH_Z_SEGMENTS);
  const auto vertex_id = [](int x, int z) {
    return z * (CLOTH_X_SEGMENTS + 1) + x;
  };
  for (int z = 0; z < CLOTH_Z_SEGMENTS; ++z) {
    for (int x = 0; x < CLOTH_X_SEGMENTS; ++x) {
      const int v00 = vertex_id(x, z);
      const int v10 = vertex_id(x + 1, z);
      const int v01 = vertex_id(x, z + 1);
      const int v11 = vertex_id(x + 1, z + 1);
      if ((x + z) % 2 == 0) {
        mesh.triangles.push_back(std::array<int, 3>{v00, v10, v01});
        mesh.triangles.push_back(std::array<int, 3>{v10, v11, v01});
      } else {
        mesh.triangles.push_back(std::array<int, 3>{v00, v10, v11});
        mesh.triangles.push_back(std::array<int, 3>{v00, v11, v01});
      }
    }
  }
  mesh.material.arealDensity = 0.25;
  mesh.material.thickness = 2.0e-3;
  mesh.material.stretchStiffness = 2e3;
  return mesh;
}

ksk::core::Vec3f toVec3f(const glm::dvec3& v) {
  return {
      static_cast<float>(v.x),
      static_cast<float>(v.y),
      static_cast<float>(v.z),
  };
}

void pinClothPoint(ksk::runtime::RuntimeSceneDesc& scene,
                   ksk::runtime::ObjectRef cloth, int point,
                   const glm::dvec3& position, double stiffness) {
  ksk::engine::fem::addConstraint(scene, cloth, "x", point, stiffness,
                                  [x = position.x](double) { return x; });
  ksk::engine::fem::addConstraint(scene, cloth, "y", point, stiffness,
                                  [y = position.y](double) { return y; });
  ksk::engine::fem::addConstraint(scene, cloth, "z", point, stiffness,
                                  [z = position.z](double) { return z; });
}

bool isRigidTriangle(const ksk::runtime::GlobalGeometryManager& geometry,
                     const ksk::runtime::GeometryTriangle& triangle) {
  const ksk::runtime::GeometryBodyIdx body = triangle.ref.owner.body;
  if (body < 0 ||
      body >= static_cast<int>(geometry.geometryBodies.size())) {
    return false;
  }
  return geometry.geometryBodies[static_cast<size_t>(body)].type ==
         ksk::runtime::EGeometryBody::Rigid;
}

std::unique_ptr<ksk::renderer::SceneProxy> buildProxy(
    const ksk::runtime::SimulationRunner& runner, int frame) {
  const ksk::runtime::RuntimeScene& scene = runner.simulation().scene();
  const auto& geometry = scene.geometry;

  auto proxy = std::make_unique<ksk::renderer::SceneProxy>();
  proxy->frameIndex = frame;
  proxy->simulationTime = static_cast<float>(runner.time());
  proxy->camera.position = {0.20f, 0.25f, 1.25f};
  proxy->camera.target = {0.05f, 0.02f, 0.12f};

  std::vector<ksk::core::Vec3f> positions;
  positions.reserve(geometry.points.size());
  for (int point = 0; point < geometry.pointCount(); ++point) {
    positions.push_back(
        toVec3f(geometry.worldPosition(ksk::runtime::PointIdx{point})));
  }

  ksk::renderer::MeshProxy surface;
  surface.name = "soft-body-surface";
  surface.positions = positions;
  surface.objectColor = {0.28f, 0.55f, 0.92f};

  ksk::renderer::MeshProxy rigid_surface;
  rigid_surface.name = "rigid-body-surface";
  rigid_surface.positions = positions;
  rigid_surface.objectColor = {0.92f, 0.74f, 0.22f};

  for (const auto& triangle : geometry.triangles) {
    auto& target = isRigidTriangle(geometry, triangle) ? rigid_surface : surface;
    target.triangles.emplace_back(triangle.p0, triangle.p1, triangle.p2);
  }
  ksk::renderer::computeSmoothNormals(surface);
  proxy->meshes.push_back(std::move(surface));
  if (!rigid_surface.triangles.empty()) {
    ksk::renderer::computeSmoothNormals(rigid_surface);
    proxy->meshes.push_back(std::move(rigid_surface));
  }

  ksk::renderer::WireframeProxy edges;
  edges.name = "scene-edges";
  edges.color = {0.94f, 0.37f, 0.18f};
  edges.positions = positions;
  edges.edges.reserve(geometry.edges.size());
  for (const auto& edge : geometry.edges) {
    if (edge.ref.owner.subsystem != 0)
      edges.edges.emplace_back(edge.p0, edge.p1);
  }
  proxy->wireframes.push_back(std::move(edges));

  ksk::renderer::ParticleProxy points;
  points.name = "scene-points";
  points.positions = std::move(positions);
  points.radius = 0.0015f;
  points.color = {0.10f, 0.82f, 0.68f};
  proxy->particles.push_back(std::move(points));

  return proxy;
}
}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("coupled-contact-flow",
                           "Coupled ABD/FEM contact scene with renderer");
  options.add_options()("steps", "number of simulation steps",
                        cxxopts::value<int>()->default_value("36000"))(
      "dt", "time step", cxxopts::value<double>()->default_value("0.005"))(
      "no-render", "disable rendering",
      cxxopts::value<bool>()->default_value("false"))(
      "no-contact", "disable global collision/contact solve",
      cxxopts::value<bool>()->default_value("false"))("h,help", "print help");

  const auto args = options.parse(argc, argv);
  if (args.count("help") > 0) {
    std::cout << options.help() << '\n';
    return 0;
  }

  const int steps = args["steps"].as<int>();
  const double dt = args["dt"].as<double>();
  const bool no_render = args["no-render"].as<bool>();
  const bool no_contact = args["no-contact"].as<bool>();
  if (steps <= 0) {
    std::cerr << "coupled-contact-flow: steps must be positive\n";
    return 1;
  }
  if (dt <= 0.0) {
    std::cerr << "coupled-contact-flow: dt must be positive\n";
    return 1;
  }

  ksk::runtime::RuntimeSceneDesc scene;
  scene.gravity = glm::dvec3(0.0, -9.8, 0.0);
  scene.timeStep = dt;
  scene.solverConfig.maxNewtonIterations = 100;
  scene.solverConfig.maxPcgIterations = 1000;
  scene.solverConfig.enableContact = !no_contact;
  scene.solverConfig.maxLineSearchIterations = 20;
  scene.solverConfig.contactBarrierDistance = 1.0e-3;
  scene.solverConfig.solveMode = ksk::runtime::GlobalSolveMode::Decoupled;
  scene.solverConfig.enableContact = true;

  const ksk::runtime::ObjectRef cloth =
      ksk::engine::fem::addClothMesh(scene, makeClothSheet(), "cloth-sheet");
  pinClothPoint(scene, cloth, 0,
                {-0.5 * CLOTH_WIDTH, CLOTH_HEIGHT, -0.5 * CLOTH_DEPTH}, 5e3);
  pinClothPoint(scene, cloth, CLOTH_X_SEGMENTS,
                {0.5 * CLOTH_WIDTH, CLOTH_HEIGHT, -0.5 * CLOTH_DEPTH}, 5e3);
  pinClothPoint(scene,
                cloth,
                CLOTH_Z_SEGMENTS * (CLOTH_X_SEGMENTS + 1),
                {-0.5 * CLOTH_WIDTH, CLOTH_HEIGHT, 0.5 * CLOTH_DEPTH},
                5e3);
  pinClothPoint(scene,
                cloth,
                CLOTH_Z_SEGMENTS * (CLOTH_X_SEGMENTS + 1) + CLOTH_X_SEGMENTS,
                {0.5 * CLOTH_WIDTH, CLOTH_HEIGHT, 0.5 * CLOTH_DEPTH},
                5e3);

  ksk::abd::addABDBody(
      scene,
      makeRigidBox({0.055, 0.035, 0.055}, {-0.18, 0.82, -0.12},
                   {0.0, -0.35, 0.0}),
      "rigid-box-left");
  ksk::abd::addABDBody(
      scene,
      makeRigidBox({0.045, 0.045, 0.045}, {0.12, 0.88, 0.04},
                   {0.0, -0.30, 0.0}),
      "rigid-box-center");
  ksk::abd::addABDBody(
      scene,
      makeRigidBox({0.07, 0.025, 0.04}, {0.32, 0.76, 0.14},
                   {0.0, -0.40, 0.0}),
      "rigid-box-right");

  ksk::runtime::SimulationRunner runner =
      ksk::runtime::buildSimulationRunner(scene);
  const ksk::runtime::RuntimeScene& initial_scene = runner.simulation().scene();

  std::cout << "coupled-contact-flow\n"
            << "  subsystems: " << initial_scene.subsystemBatches.size() << '\n'
            << "  dofs: " << initial_scene.dofs.totalScalars << '\n'
            << "  points: " << initial_scene.geometry.points.size() << '\n'
            << "  edges: " << initial_scene.geometry.edges.size() << '\n'
            << "  triangles: " << initial_scene.geometry.triangles.size()
            << '\n';

  bool solve_failed = false;
  auto step_once = [&](int step) {
    if (solve_failed) {
      return;
    }
    const ksk::runtime::RuntimeStepResult step_result = runner.step();
    const bool finite = std::isfinite(step_result.finalGradientNorm) &&
                        std::isfinite(step_result.finalStepNorm);

    if (!finite) solve_failed = true;

    std::cout << "step=" << step + 1 << " iterations=" << step_result.iterations
              << " converged=" << (step_result.converged ? "true" : "false")
              << " grad=" << step_result.finalGradientNorm
              << " step_norm=" << step_result.finalStepNorm << '\n';
  };

  if (no_render) {
    for (int step = 0; step < steps && !solve_failed; ++step) {
      step_once(step);
    }
    return solve_failed ? 2 : 0;
  }

  ksk::renderer::SimulationApp app({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "Kiseki - Coupled ABD/FEM Contact",
  });
  app.stepFn = [&](int step) { step_once(step); };
  app.buildProxy = [&](int frame) { return buildProxy(runner, frame); };
  app.logInterval = 30;
  app.run(steps);

  return solve_failed ? 2 : 0;
}
