// Phase 10: Renderer binding + run_and_display orchestration
// Design: Renderer is fully decoupled from Simulation.
// run_and_display() is a free function that composes them via SceneProxy.
#include "bindings.h"
#include <Renderer/renderer.h>
#include <Renderer/scene-proxy.h>
#include <thread>

using namespace sim::renderer;
using sim::core::Vec3f;
using sim::core::Vec3u;

/// Build a SceneProxy from the current System state
static std::unique_ptr<SceneProxy> buildSceneProxy(System& system, int frameIdx, float simTime)
{
  auto scene = std::make_unique<SceneProxy>();
  scene->frameIndex = frameIdx;
  scene->simulationTime = simTime;

  // Extract each primitive's surface mesh
  int dofStart = 0;
  for (int i = 0; i < static_cast<int>(system.primitives().size()); i++) {
    const auto& prim = system.primitive(i);
    auto surfaceView = prim.getSurfaceView();
    size_t vertCount = prim.getVertexCount();
    int primDofStart = prim.getDofStart();

    MeshProxy mesh;
    mesh.name = "body_" + std::to_string(i);

    // Extract vertex positions from system.x (BlockVector<3>)
    mesh.positions.resize(vertCount);
    for (size_t v = 0; v < vertCount; v++) {
      int blockIdx = primDofStart / 3 + static_cast<int>(v);
      auto pos = system.x[blockIdx]; // glm::dvec3
      mesh.positions[v] = Vec3f(
          static_cast<float>(pos.x),
          static_cast<float>(pos.y),
          static_cast<float>(pos.z));
    }

    // Extract surface triangles
    mesh.triangles.resize(surfaceView.size());
    for (size_t t = 0; t < surfaceView.size(); t++) {
      auto tri = surfaceView[t];
      mesh.triangles[t] = Vec3u(
          static_cast<unsigned>(tri[0]),
          static_cast<unsigned>(tri[1]),
          static_cast<unsigned>(tri[2]));
    }

    scene->meshes.push_back(std::move(mesh));
  }

  return scene;
}

/// Python-facing Renderer wrapper
struct PyRenderer {
  RendererConfig config;
  std::unique_ptr<Renderer> impl;

  void create() {
    if (!impl)
      impl = createRenderer(config);
  }
};

void bind_renderer(py::module_& m)
{
  // RendererConfig as Renderer constructor args
  py::class_<PyRenderer, std::shared_ptr<PyRenderer>>(m, "Renderer")
    .def(py::init([](int width, int height, const std::string& title, bool vsync) {
      auto r = std::make_shared<PyRenderer>();
      r->config.windowWidth = width;
      r->config.windowHeight = height;
      r->config.windowTitle = title;
      r->config.vsync = vsync;
      return r;
    }),
    py::arg("width") = 1280,
    py::arg("height") = 720,
    py::arg("title") = "SimCraft",
    py::arg("vsync") = true,
    "Create a renderer with window configuration")

    .def_property_readonly("width", [](const PyRenderer& self) {
      return self.config.windowWidth;
    })
    .def_property_readonly("height", [](const PyRenderer& self) {
      return self.config.windowHeight;
    });

  // run_and_display: top-level composition function
  m.def("run_and_display", [](std::shared_ptr<PySimulation> sim,
                               std::shared_ptr<PyRenderer> renderer,
                               Real dt, int steps) {
    if (!sim)
      throw py::type_error("simulation must be a valid Simulation object");
    if (!renderer)
      throw py::type_error("renderer must be a valid Renderer object");
    if (dt <= 0.0)
      throw py::value_error("dt must be positive");
    if (steps <= 0)
      throw py::value_error("steps must be positive");

    // Ensure simulation is initialized (system.init + integrator created)
    sim->ensure_initialized();
    sim->lock_all();

    // Create the renderer (OpenGL context)
    renderer->create();
    auto& rend = *renderer->impl;

    // Push initial frame
    auto initScene = buildSceneProxy(sim->py_system->system, 0,
                                     static_cast<float>(sim->py_system->system.currentTime()));
    rend.queue().push(std::move(initScene));

    // Simulation runs in background thread, pushes frames
    std::thread simThread([&]() {
      py::gil_scoped_release release; // Release GIL for sim thread
      for (int i = 0; i < steps; i++) {
        if (!rend.isRunning()) break;

        sim->integrator->step(dt);
        sim->py_system->system.advanceTime(dt);
        sim->steps_completed++;

        auto scene = buildSceneProxy(sim->py_system->system, i + 1,
                                     static_cast<float>(sim->py_system->system.currentTime()));
        rend.queue().push(std::move(scene));
      }
      // Signal renderer to stop after all frames consumed
      // (renderer will drain queue then exit)
      rend.shutdown();
    });

    // Render loop runs on main thread (GLFW requirement)
    {
      py::gil_scoped_release release;
      rend.runOnCurrentThread();
    }

    simThread.join();

  }, py::arg("simulation"), py::arg("renderer"),
     py::arg("dt") = 0.01, py::arg("steps") = 1000,
     "Run simulation with real-time display.\n"
     "Main thread renders (GLFW), background thread simulates.\n"
     "Window closes when simulation completes or user closes window.");
}
