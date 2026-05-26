// Phase 3: Material Model binding (NeoHookean)
#include "bindings.h"

void bind_material(py::module_& m)
{
  py::class_<PyMaterial, std::shared_ptr<PyMaterial>>(m, "NeoHookean")
    .def(py::init([](Real young, Real poisson) {
      if (young <= 0.0)
        throw py::value_error("Young's modulus must be positive");
      if (poisson <= -1.0 || poisson >= 0.5)
        throw py::value_error("Poisson ratio must be in (-1, 0.5)");

      auto mat = std::make_shared<PyMaterial>();
      mat->young = young;
      mat->poisson = poisson;
      mat->impl = std::make_unique<StableNeoHookean<Real>>(
          ElasticityParameters<Real>{young, poisson});
      return mat;
    }), py::arg("young") = 1e6, py::arg("poisson") = 0.45,
       "Create a Stable NeoHookean material with Young's modulus and Poisson ratio")

    .def_property_readonly("young", [](const PyMaterial& self) {
      return self.young;
    }, "Young's modulus")

    .def_property_readonly("poisson", [](const PyMaterial& self) {
      return self.poisson;
    }, "Poisson ratio");
}
