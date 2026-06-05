// Phase 6: KinematicBody binding
#include "bindings.h"

void bind_kinematic(py::module_& m)
{
  py::class_<PyKinematicBody, std::shared_ptr<PyKinematicBody>>(m, "KinematicBody")
    .def(py::init([]() {
      auto kb = std::make_shared<PyKinematicBody>();
      kb->body.motion = staticMotion();
      return kb;
    }), "Create an empty kinematic body")

    .def_static("plane", [](py::array_t<double> normal, double offset) {
      auto nbuf = normal.unchecked<1>();
      if (nbuf.shape(0) != 3)
        throw py::value_error("normal must be a 3-element array");

      glm::dvec3 n(nbuf(0), nbuf(1), nbuf(2));
      double len = glm::length(n);
      if (len < 1e-12)
        throw py::value_error("normal vector must be non-zero");
      n /= len;

      auto kb = std::make_shared<PyKinematicBody>();

      Collider::SDFGeometry sdf;
      sdf.signedDistance = [n, offset](const glm::dvec3& p) -> Real {
        return glm::dot(n, p) - offset;
      };
      sdf.gradient = [n](const glm::dvec3&) -> glm::dvec3 {
        return n;
      };

      kb->body.geometry = std::move(sdf);
      kb->body.motion = staticMotion();
      kb->motion_type = "static";
      return kb;
    }, py::arg("normal"), py::arg("offset") = 0.0,
       "Create a kinematic body with plane SDF geometry")

    .def_property("motion",
      [](const PyKinematicBody& self) { return self.motion_type; },
      [](PyKinematicBody& self, const std::string& type) {
        if (type == "static") {
          self.body.motion = staticMotion();
          self.motion_type = "static";
        } else {
          throw py::value_error(
              "Use set_constant_velocity() or set_rotation() for non-static motion");
        }
      }, "Motion type string (read/write 'static' only; use methods for others)")

    .def("set_constant_velocity", [](PyKinematicBody& self, py::array_t<double> vel) {
      auto buf = vel.unchecked<1>();
      if (buf.shape(0) != 3)
        throw py::value_error("velocity must be a 3-element array");
      glm::dvec3 v(buf(0), buf(1), buf(2));
      self.body.motion = constantVelocity(v);
      self.motion_type = "constant_velocity";
    }, py::arg("velocity"),
       "Set constant velocity motion [vx, vy, vz]")

    .def("set_rotation", [](PyKinematicBody& self,
                            py::array_t<double> axis,
                            py::array_t<double> center,
                            double omega) {
      auto abuf = axis.unchecked<1>();
      auto cbuf = center.unchecked<1>();
      if (abuf.shape(0) != 3 || cbuf.shape(0) != 3)
        throw py::value_error("axis and center must be 3-element arrays");
      glm::dvec3 a(abuf(0), abuf(1), abuf(2));
      glm::dvec3 c(cbuf(0), cbuf(1), cbuf(2));
      self.body.motion = constantRotation(a, c, omega);
      self.motion_type = "rotation";
    }, py::arg("axis"), py::arg("center"), py::arg("omega"),
       "Set constant rotation motion around an axis through center");
}
