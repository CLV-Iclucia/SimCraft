// Phase 7: IPC Integrator binding
#include "bindings.h"

void bind_integrator(py::module_& m)
{
  py::class_<PyIntegratorConfig, std::shared_ptr<PyIntegratorConfig>>(m, "IpcIntegrator")
    .def(py::init([](Real dHat, Real eps, Real kappa, Real stepSizeScale) {
      if (dHat <= 0.0)
        throw py::value_error("dHat must be positive");
      if (eps <= 0.0)
        throw py::value_error("eps must be positive");
      if (kappa <= 0.0)
        throw py::value_error("kappa must be positive");
      if (stepSizeScale <= 0.0 || stepSizeScale > 1.0)
        throw py::value_error("stepSizeScale must be in (0, 1]");

      auto config = std::make_shared<PyIntegratorConfig>();
      config->dHat = dHat;
      config->eps = eps;
      config->kappa = kappa;
      config->stepSizeScale = stepSizeScale;
      return config;
    }),
    py::arg("dHat") = 1e-3,
    py::arg("eps") = 1e-2,
    py::arg("kappa") = 1e10,
    py::arg("stepSizeScale") = 0.9,
    "Create an IPC integrator with collision handling parameters")

    .def_property("dHat",
      [](const PyIntegratorConfig& self) { return self.dHat; },
      [](PyIntegratorConfig& self, Real v) {
        self.check_locked();
        if (v <= 0.0) throw py::value_error("dHat must be positive");
        self.dHat = v;
      }, "Distance threshold for barrier activation")

    .def_property("eps",
      [](const PyIntegratorConfig& self) { return self.eps; },
      [](PyIntegratorConfig& self, Real v) {
        self.check_locked();
        if (v <= 0.0) throw py::value_error("eps must be positive");
        self.eps = v;
      }, "Convergence tolerance")

    .def_property("kappa",
      [](const PyIntegratorConfig& self) { return self.kappa; },
      [](PyIntegratorConfig& self, Real v) {
        self.check_locked();
        if (v <= 0.0) throw py::value_error("kappa must be positive");
        self.kappa = v;
      }, "Contact stiffness parameter")

    .def_property("stepSizeScale",
      [](const PyIntegratorConfig& self) { return self.stepSizeScale; },
      [](PyIntegratorConfig& self, Real v) {
        self.check_locked();
        if (v <= 0.0 || v > 1.0) throw py::value_error("stepSizeScale must be in (0, 1]");
        self.stepSizeScale = v;
      }, "CCD step size scaling factor")

    .def_property_readonly("locked", [](const PyIntegratorConfig& self) {
      return self.locked;
    }, "Whether parameters are locked");
}
