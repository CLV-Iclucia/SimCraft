// Phase 4: ElasticBody & System Core binding
#include "bindings.h"

void bind_system(py::module_& m)
{
  py::class_<PySystem, std::shared_ptr<PySystem>>(m, "System")
    .def(py::init([]() {
      return std::make_shared<PySystem>();
    }), "Create a new simulation system")

    .def("add_elastic_body", [](PySystem& self, TetMesh& mesh,
                                  std::shared_ptr<PyMaterial> material, Real density,
                                  std::optional<std::tuple<double,double,double>> color) {
           glm::vec3 c(-1.0f);
           if (color.has_value()) {
             auto [r, g, b] = color.value();
             c = glm::vec3(static_cast<float>(r), static_cast<float>(g), static_cast<float>(b));
           }
           self.add_elastic_body(mesh, material, density, c);
         },
         py::arg("mesh"), py::arg("material"), py::arg("density"),
         py::arg("color") = py::none(),
         "Add an elastic body to the system.\n"
         "color: optional (r, g, b) tuple with values in [0,1]")

    .def("add_kinematic_body", &PySystem::add_kinematic_body,
         py::arg("body"),
         "Add a kinematic (non-deformable) body to the system for collision")

    .def_property("gravity",
                  &PySystem::get_gravity,
                  &PySystem::set_gravity,
                  "Gravity vector [x, y, z]")

    .def_property_readonly("num_bodies", [](const PySystem& self) {
      return self.system.primitives().size();
    }, "Number of elastic bodies in the system")

    .def_property_readonly("constraints", [](std::shared_ptr<PySystem> self) {
      auto c = std::make_shared<PyConstraints>();
      c->owner = self;
      return c;
    }, "Access the constraint manager for this system")

    .def_property_readonly("locked", [](const PySystem& self) {
      return self.locked;
    }, "Whether the system is locked (after simulation started)");
}
