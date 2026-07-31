#include "bindings.h"

#include <ABD/abd-scene.h>
#include <Runtime/global-solver.h>

#include <glm/glm.hpp>

#include <array>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace abd = ksk::abd;
namespace runtime = ksk::runtime;

struct PyABDRigidBody {
  abd::ABDBodyDesc desc;
};

struct PyABDScene {
  runtime::RuntimeSceneDesc scene;
  int body_count = 0;
  std::vector<int> body_vertex_counts;
  bool locked = false;

  void checkUnlocked() const
  {
    if (locked) {
      throw std::runtime_error(
          "ABDScene is locked after build(). Create a new scene to modify it.");
    }
  }
};

struct PyABDSimulation {
  runtime::SimulationRunner simulation;
  std::vector<int> body_vertex_counts;

  PyABDSimulation(runtime::SimulationRunner&& runtimeSimulation,
                  std::vector<int> vertexCounts)
      : simulation(std::move(runtimeSimulation))
      , body_vertex_counts(std::move(vertexCounts))
  {
  }

  runtime::RuntimeStepResult step()
  {
    return simulation.step();
  }

  runtime::RuntimeStepResult run(int steps)
  {
    if (steps <= 0) {
      throw py::value_error("steps must be positive");
    }
    return simulation.run(steps);
  }
};

glm::dvec3 parseVec3(py::array_t<double> values, const char* name)
{
  const auto buffer = values.unchecked<1>();
  if (buffer.ndim() != 1 || buffer.shape(0) != 3) {
    throw py::value_error(std::string(name) + " must have shape (3,)");
  }
  return glm::dvec3(buffer(0), buffer(1), buffer(2));
}

glm::dmat4x3 parseAffine(std::optional<py::array_t<double>> values,
                         const char* name,
                         const glm::dmat4x3& fallback)
{
  if (!values.has_value()) {
    return fallback;
  }

  const auto buffer = values->unchecked<2>();
  if (buffer.ndim() != 2) {
    throw py::value_error(std::string(name) +
                          " must have shape (4, 3) or (3, 4)");
  }

  glm::dmat4x3 affine{0.0};
  if (buffer.shape(0) == 4 && buffer.shape(1) == 3) {
    for (py::ssize_t col = 0; col < 4; ++col) {
      affine[static_cast<int>(col)] =
          glm::dvec3(buffer(col, 0), buffer(col, 1), buffer(col, 2));
    }
    return affine;
  }

  if (buffer.shape(0) == 3 && buffer.shape(1) == 4) {
    for (py::ssize_t col = 0; col < 4; ++col) {
      affine[static_cast<int>(col)] =
          glm::dvec3(buffer(0, col), buffer(1, col), buffer(2, col));
    }
    return affine;
  }

  throw py::value_error(std::string(name) +
                        " must have shape (4, 3) or (3, 4)");
}

std::vector<glm::dvec3> parseVertices(py::array_t<double> vertices)
{
  const auto buffer = vertices.unchecked<2>();
  if (buffer.ndim() != 2 || buffer.shape(1) != 3) {
    throw py::value_error("vertices must have shape (N, 3)");
  }
  if (buffer.shape(0) <= 0) {
    throw py::value_error("ABD rigid body requires at least one vertex");
  }

  std::vector<glm::dvec3> result(static_cast<size_t>(buffer.shape(0)));
  for (py::ssize_t i = 0; i < buffer.shape(0); ++i) {
    result[static_cast<size_t>(i)] =
        glm::dvec3(buffer(i, 0), buffer(i, 1), buffer(i, 2));
  }
  return result;
}

std::vector<std::array<int, 2>> parseEdges(
    std::optional<py::array_t<int>> edges)
{
  if (!edges.has_value()) {
    return {};
  }
  const auto buffer = edges->unchecked<2>();
  if (buffer.ndim() != 2 || buffer.shape(1) != 2) {
    throw py::value_error("edges must have shape (E, 2)");
  }

  std::vector<std::array<int, 2>> result(static_cast<size_t>(buffer.shape(0)));
  for (py::ssize_t i = 0; i < buffer.shape(0); ++i) {
    result[static_cast<size_t>(i)] = {buffer(i, 0), buffer(i, 1)};
  }
  return result;
}

std::vector<std::array<int, 3>> parseTriangles(
    std::optional<py::array_t<int>> triangles)
{
  if (!triangles.has_value()) {
    return {};
  }
  const auto buffer = triangles->unchecked<2>();
  if (buffer.ndim() != 2 || buffer.shape(1) != 3) {
    throw py::value_error("triangles must have shape (T, 3)");
  }

  std::vector<std::array<int, 3>> result(
      static_cast<size_t>(buffer.shape(0)));
  for (py::ssize_t i = 0; i < buffer.shape(0); ++i) {
    result[static_cast<size_t>(i)] = {buffer(i, 0), buffer(i, 1), buffer(i, 2)};
  }
  return result;
}

std::vector<std::array<int, 2>> deriveEdgesFromTriangles(
    const std::vector<std::array<int, 3>>& triangles)
{
  std::set<std::array<int, 2>> unique_edges;
  for (const std::array<int, 3>& triangle : triangles) {
    for (std::array<int, 2> edge :
         {std::array<int, 2>{triangle[0], triangle[1]},
          std::array<int, 2>{triangle[1], triangle[2]},
          std::array<int, 2>{triangle[2], triangle[0]}}) {
      if (edge[1] < edge[0]) {
        std::swap(edge[0], edge[1]);
      }
      unique_edges.insert(edge);
    }
  }
  return {unique_edges.begin(), unique_edges.end()};
}

py::array_t<double> affineToArray(const glm::dmat4x3& affine)
{
  auto result = py::array_t<double>({py::ssize_t{4}, py::ssize_t{3}});
  auto out = result.mutable_unchecked<2>();
  for (py::ssize_t col = 0; col < 4; ++col) {
    out(col, 0) = affine[static_cast<int>(col)].x;
    out(col, 1) = affine[static_cast<int>(col)].y;
    out(col, 2) = affine[static_cast<int>(col)].z;
  }
  return result;
}

py::array_t<double> geometryPositions(const runtime::SimulationContext& sim)
{
  const auto& geometry = sim.scene().geometry;
  auto result = py::array_t<double>(
      {static_cast<py::ssize_t>(geometry.pointCount()), py::ssize_t{3}});
  auto out = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0;
       i < static_cast<py::ssize_t>(geometry.pointCount());
       ++i) {
    const glm::dvec3 point =
        geometry.worldPosition(runtime::PointIdx{static_cast<int>(i)});
    out(i, 0) = point.x;
    out(i, 1) = point.y;
    out(i, 2) = point.z;
  }
  return result;
}

py::array_t<double> bodyPositions(const PyABDSimulation& sim, int bodyIndex)
{
  if (bodyIndex < 0 ||
      bodyIndex >= static_cast<int>(sim.body_vertex_counts.size())) {
    throw py::index_error("body index out of range");
  }

  int offset = 0;
  for (int i = 0; i < bodyIndex; ++i) {
    offset += sim.body_vertex_counts[static_cast<size_t>(i)];
  }

  const int count = sim.body_vertex_counts[static_cast<size_t>(bodyIndex)];
  const auto& geometry = sim.simulation.simulation().scene().geometry;
  auto result =
      py::array_t<double>({static_cast<py::ssize_t>(count), py::ssize_t{3}});
  auto out = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < count; ++i) {
    const glm::dvec3 point =
        geometry.worldPosition(runtime::PointIdx{static_cast<int>(offset + i)});
    out(i, 0) = point.x;
    out(i, 1) = point.y;
    out(i, 2) = point.z;
  }
  return result;
}

}  // namespace

void bind_abd(py::module_& m)
{
  py::class_<PyABDRigidBody, std::shared_ptr<PyABDRigidBody>>(
      m, "ABDRigidBody")
      .def(py::init([](py::array_t<double> vertices,
                       std::optional<py::array_t<int>> edges,
                       std::optional<py::array_t<int>> triangles,
                       std::optional<py::array_t<double>> initialTransform,
                       std::optional<py::array_t<double>> initialVelocity,
                       double density,
                       double contactThickness,
                       double orthogonalStiffness) {
             if (density <= 0.0) {
               throw py::value_error("density must be positive");
             }
             if (orthogonalStiffness < 0.0) {
               throw py::value_error(
                   "orthogonal_stiffness must be non-negative");
             }
             if (contactThickness < 0.0) {
               throw py::value_error("contact_thickness must be non-negative");
             }

             auto body = std::make_shared<PyABDRigidBody>();
             body->desc.surface.vertices = parseVertices(vertices);
             body->desc.surface.triangles = parseTriangles(triangles);
             body->desc.surface.edges = parseEdges(edges);
             if (!edges.has_value()) {
               body->desc.surface.edges =
                   deriveEdgesFromTriangles(body->desc.surface.triangles);
             }
             body->desc.initialTransform =
                 parseAffine(initialTransform, "initial_transform",
                             glm::dmat4x3{1.0});
             body->desc.initialVelocity =
                 parseAffine(initialVelocity, "initial_velocity",
                             glm::dmat4x3{0.0});
             body->desc.density = density;
             body->desc.orthogonalStiffness = orthogonalStiffness;
             body->desc.contactThickness = contactThickness;
             return body;
           }),
           py::arg("vertices"),
           py::arg("edges") = py::none(),
           py::arg("triangles") = py::none(),
           py::arg("initial_transform") = py::none(),
           py::arg("initial_velocity") = py::none(),
           py::arg("density") = 1.0,
           py::arg("contact_thickness") = 0.0,
           py::arg("orthogonal_stiffness") = 1.0e6)
      .def_property_readonly("num_vertices", [](const PyABDRigidBody& self) {
        return self.desc.surface.vertices.size();
      })
      .def_property_readonly("num_edges", [](const PyABDRigidBody& self) {
        return self.desc.surface.edges.size();
      })
      .def_property_readonly("num_triangles", [](const PyABDRigidBody& self) {
        return self.desc.surface.triangles.size();
      })
      .def_property_readonly("initial_transform",
                             [](const PyABDRigidBody& self) {
                               return affineToArray(
                                   self.desc.initialTransform);
                             })
      .def_property_readonly("initial_velocity", [](const PyABDRigidBody& self) {
        return affineToArray(self.desc.initialVelocity);
      })
      .def_property(
          "density",
          [](const PyABDRigidBody& self) {
            return self.desc.density;
          },
          [](PyABDRigidBody& self, double density) {
            if (density <= 0.0) {
              throw py::value_error("density must be positive");
            }
            self.desc.density = density;
          })
      .def_property(
          "orthogonal_stiffness",
          [](const PyABDRigidBody& self) {
            return self.desc.orthogonalStiffness;
          },
          [](PyABDRigidBody& self, double stiffness) {
            if (stiffness < 0.0) {
              throw py::value_error(
                  "orthogonal_stiffness must be non-negative");
            }
            self.desc.orthogonalStiffness = stiffness;
          })
      .def_property(
          "contact_thickness",
          [](const PyABDRigidBody& self) {
            return self.desc.contactThickness;
          },
          [](PyABDRigidBody& self, double contactThickness) {
            if (contactThickness < 0.0) {
              throw py::value_error("contact_thickness must be non-negative");
            }
            self.desc.contactThickness = contactThickness;
          });

  py::class_<PyABDScene, std::shared_ptr<PyABDScene>>(m, "ABDScene")
      .def(py::init([] {
        return std::make_shared<PyABDScene>();
      }))
      .def("add_body",
           [](PyABDScene& self,
              const std::shared_ptr<PyABDRigidBody>& body,
              const std::string& tag) {
             self.checkUnlocked();
             if (!body) {
               throw py::type_error("body must be an ABDRigidBody");
             }
             self.body_vertex_counts.push_back(
                 static_cast<int>(body->desc.surface.vertices.size()));
             abd::addABDBody(self.scene, body->desc, tag);
             ++self.body_count;
           },
           py::arg("body"),
           py::arg("tag") = "")
      .def_property(
          "gravity",
          [](const PyABDScene& self) {
            auto result = py::array_t<double>(3);
            auto out = result.mutable_unchecked<1>();
            out(0) = self.scene.gravity.x;
            out(1) = self.scene.gravity.y;
            out(2) = self.scene.gravity.z;
            return result;
          },
          [](PyABDScene& self, py::array_t<double> gravity) {
            self.checkUnlocked();
            self.scene.gravity = parseVec3(gravity, "gravity");
          })
      .def_property(
          "time_step",
          [](const PyABDScene& self) {
            return self.scene.timeStep;
          },
          [](PyABDScene& self, double timeStep) {
            self.checkUnlocked();
            if (timeStep <= 0.0) {
              throw py::value_error("time_step must be positive");
            }
            self.scene.timeStep = timeStep;
          })
      .def_property(
          "solver",
          [](PyABDScene& self) -> runtime::GlobalSolverConfig& {
            return self.scene.solverConfig;
          },
          [](PyABDScene& self, runtime::GlobalSolverConfig solver) {
            self.checkUnlocked();
            self.scene.solverConfig = solver;
          },
          py::return_value_policy::reference_internal)
      .def("build",
           [](std::shared_ptr<PyABDScene> self) {
             if (!self) {
               throw py::type_error("scene must be an ABDScene");
             }
             if (self->body_count == 0) {
               throw py::value_error("ABDScene requires at least one body");
             }
             self->locked = true;
             return std::make_shared<PyABDSimulation>(
                 runtime::buildSimulationRunner(self->scene),
                 self->body_vertex_counts);
           });

  py::class_<PyABDSimulation, std::shared_ptr<PyABDSimulation>>(
      m, "ABDSimulation")
      .def("step", &PyABDSimulation::step)
      .def("run", &PyABDSimulation::run, py::arg("steps") = 100)
      .def_property_readonly("positions", [](const PyABDSimulation& self) {
        return geometryPositions(self.simulation.simulation());
      })
      .def("body_positions", &bodyPositions, py::arg("body_index"))
      .def_property_readonly("body_vertex_counts",
                             [](const PyABDSimulation& self) {
                               return self.body_vertex_counts;
                             })
      .def_property_readonly("steps_completed",
                             [](const PyABDSimulation& self) {
                               return self.simulation.stepsCompleted();
                             })
      .def_property_readonly("time", [](const PyABDSimulation& self) {
        return self.simulation.time();
      })
      .def_property_readonly("time_step", [](const PyABDSimulation& self) {
        return self.simulation.timeStep();
      })
      .def_property_readonly("last_step", [](const PyABDSimulation& self) {
        return self.simulation.lastStep();
      });
}
