#include <ABD/abd-scene.h>
#include <Renderer/simulation-app.h>
#include <Runtime/global-solver.h>

#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>

#include <array>
#include <cmath>
#include <cxxopts.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr double kRampAngle = 0.42;
constexpr double kRampLength = 3.2;
constexpr double kRampWidth = 1.15;
constexpr double kRampCenterY = 0.62;
constexpr double kSliderHalfX = 0.12;
constexpr double kSliderHalfY = 0.055;
constexpr double kSliderHalfZ = 0.09;

ksk::core::Vec3f toVec3f(const glm::dvec3& value)
{
  return {
      static_cast<float>(value.x),
      static_cast<float>(value.y),
      static_cast<float>(value.z),
  };
}

glm::dvec3 rampTangent() {
  return glm::normalize(glm::dvec3{std::cos(kRampAngle), -std::sin(kRampAngle),
                                   0.0});
}

glm::dvec3 rampNormal() {
  return glm::normalize(glm::dvec3{std::sin(kRampAngle), std::cos(kRampAngle),
                                   0.0});
}

glm::dvec3 rampBinormal() {
  return glm::dvec3{0.0, 0.0, 1.0};
}

glm::dvec3 rampPoint(double s, double z) {
  return glm::dvec3{0.0, kRampCenterY, 0.0} + s * rampTangent() +
         z * rampBinormal();
}

ksk::runtime::GeometryMeshDesc makeRampColliderMesh()
{
  ksk::runtime::GeometryMeshDesc mesh;
  const double half_length = 0.5 * kRampLength;
  const double half_width = 0.5 * kRampWidth;
  mesh.vertices = {
      rampPoint(-half_length, -half_width),
      rampPoint(half_length, -half_width),
      rampPoint(half_length, half_width),
      rampPoint(-half_length, half_width),
  };
  mesh.edges = {
      std::array<int, 2>{0, 1},
      std::array<int, 2>{1, 2},
      std::array<int, 2>{2, 3},
      std::array<int, 2>{3, 0},
  };
  mesh.triangles = {
      std::array<int, 3>{0, 1, 2},
      std::array<int, 3>{0, 2, 3},
  };
  mesh.radius = 1.0e-3;
  return mesh;
}

ksk::abd::ABDBodyDesc makeSlider(double lane,
                                 double start_s,
                                 double friction)
{
  ksk::abd::ABDBodyDesc body;
  const double x = kSliderHalfX;
  const double y = kSliderHalfY;
  const double z = kSliderHalfZ;
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
  body.initialTransform[0] = rampTangent();
  body.initialTransform[1] = rampNormal();
  body.initialTransform[2] = rampBinormal();
  body.initialTransform[3] =
      rampPoint(start_s, lane) + (kSliderHalfY + 4.0e-3) * rampNormal();
  body.density = 650.0;
  body.orthogonalStiffness = 5.0e6;
  body.contactThickness = 1.5e-3;
  body.frictionCoefficient = friction;
  return body;
}

std::unique_ptr<ksk::renderer::SceneProxy> buildProxy(
    const ksk::runtime::SimulationRunner& runner,
    int frame,
    const std::vector<ksk::core::Vec3f>& slider_colors)
{
  const ksk::runtime::RuntimeScene& scene = runner.simulation().scene();
  const auto& geometry = scene.geometry;
  auto proxy = std::make_unique<ksk::renderer::SceneProxy>();
  proxy->frameIndex = frame;
  proxy->simulationTime = static_cast<float>(runner.time());
  proxy->camera.position = {1.1f, 1.15f, 2.4f};
  proxy->camera.target = {0.15f, 0.35f, 0.0f};

  std::vector<ksk::core::Vec3f> positions;
  positions.reserve(geometry.points.size());
  for (int point = 0; point < geometry.pointCount(); ++point) {
    positions.push_back(toVec3f(
        geometry.worldPosition(ksk::runtime::PointIdx{point})));
  }

  for (int instance = 0; instance < static_cast<int>(geometry.instances.size());
       ++instance) {
    const auto& entry = geometry.instances[static_cast<size_t>(instance)];
    ksk::renderer::MeshProxy mesh;
    mesh.name = entry.owner.isCollider()
                    ? "static-ramp-collider"
                    : "abd-slider-" + std::to_string(entry.owner.body);
    mesh.positions = positions;
    mesh.objectColor = entry.owner.isCollider()
                           ? ksk::core::Vec3f{0.42f, 0.44f, 0.47f}
                           : slider_colors[static_cast<size_t>(
                                 entry.owner.body % slider_colors.size())];
    const ksk::runtime::GeometryRange tri_range =
        geometry.instanceTriangleRange(instance);
    for (int tri = tri_range.first; tri < tri_range.end(); ++tri) {
      const auto& triangle = geometry.triangles[static_cast<size_t>(tri)];
      mesh.triangles.emplace_back(triangle.p0, triangle.p1, triangle.p2);
    }
    ksk::renderer::computeSmoothNormals(mesh);
    proxy->meshes.push_back(std::move(mesh));
  }
  return proxy;
}

}  // namespace

int main(int argc, char** argv)
{
  cxxopts::Options options(
      "abd-friction-ramp",
      "ABD rigid sliders on a static mesh collider ramp with friction");
  options.add_options()
      ("steps", "number of simulation steps",
       cxxopts::value<int>()->default_value("1200"))
      ("dt", "time step", cxxopts::value<double>()->default_value("0.003"))
      ("no-render", "disable rendering",
       cxxopts::value<bool>()->default_value("false"))
      ("h,help", "print help");

  const auto args = options.parse(argc, argv);
  if (args.count("help") > 0) {
    std::cout << options.help() << '\n';
    return 0;
  }

  const int steps = args["steps"].as<int>();
  const double dt = args["dt"].as<double>();
  const bool no_render = args["no-render"].as<bool>();

  ksk::runtime::RuntimeSceneDesc scene;
  scene.timeStep = dt;
  scene.gravity = glm::dvec3{0.0, -9.81, 0.0};
  scene.solverConfig.enableContact = true;
  scene.solverConfig.enableFriction = true;
  scene.solverConfig.frictionCoefficient = 0.0;
  scene.solverConfig.frictionEpsilon = 1.0e-5;
  scene.solverConfig.contactBarrierDistance = 2.0e-3;
  scene.solverConfig.contactStiffness = 3.0e5;
  scene.solverConfig.maxNewtonIterations = 80;
  scene.solverConfig.maxPcgIterations = 600;
  scene.solverConfig.solveMode = ksk::runtime::GlobalSolveMode::Coupled;

  scene.addCollider(ksk::runtime::SceneColliderDesc{
      .mesh = makeRampColliderMesh(),
      .transform = glm::dmat4{1.0},
      .frictionCoefficient = 1.0,
      .tag = "static-ramp",
  });

  const std::array<double, 5> frictions{0.02, 0.08, 0.18, 0.36, 0.65};
  const std::vector<ksk::core::Vec3f> colors{
      {0.20f, 0.63f, 0.95f},
      {0.20f, 0.78f, 0.46f},
      {0.94f, 0.76f, 0.22f},
      {0.94f, 0.42f, 0.24f},
      {0.66f, 0.42f, 0.88f},
  };
  for (int i = 0; i < static_cast<int>(frictions.size()); ++i) {
    const double lane =
        -0.42 + 0.21 * static_cast<double>(i);
    const double start_s =
        -0.98 + 0.02 * static_cast<double>(i);
    ksk::abd::addABDBody(
        scene, makeSlider(lane, start_s, frictions[static_cast<size_t>(i)]),
        "slider-mu-" + std::to_string(i));
  }

  ksk::runtime::SimulationRunner runner =
      ksk::runtime::buildSimulationRunner(scene);
  const auto& runtime_scene = runner.simulation().scene();
  std::cout << "abd-friction-ramp\n"
            << "  sliders: " << frictions.size() << '\n'
            << "  collider points: "
            << runtime_scene.geometry.colliderPointRange(0).count << '\n'
            << "  dofs: " << runtime_scene.dofs.totalScalars << '\n';
  for (double mu : frictions) {
    std::cout << "  mu=" << mu << '\n';
  }

  bool solve_failed = false;
  auto step_once = [&](int step) {
    if (solve_failed) {
      return;
    }
    const ksk::runtime::RuntimeStepResult result = runner.step();
    const bool finite = std::isfinite(result.finalGradientNorm) &&
                        std::isfinite(result.finalStepNorm);
    solve_failed = !finite;
    std::cout << "step=" << step + 1
              << " iterations=" << result.iterations
              << " converged=" << (result.converged ? "true" : "false")
              << " grad=" << result.finalGradientNorm
              << " step_norm=" << result.finalStepNorm << '\n';
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
      .windowTitle = "Kiseki - ABD Friction Ramp",
  });
  app.stepFn = [&](int step) { step_once(step); };
  app.buildProxy = [&](int frame) {
    return buildProxy(runner, frame, colors);
  };
  app.logInterval = 30;
  app.run(steps);
  return solve_failed ? 2 : 0;
}
