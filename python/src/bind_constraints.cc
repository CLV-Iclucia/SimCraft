// Phase 5: Constraint System binding
#include "bindings.h"

void bind_constraints(py::module_& m)
{
  py::class_<PyConstraints, std::shared_ptr<PyConstraints>>(m, "Constraints")
    .def("pin_vertices", &PyConstraints::pin_vertices,
         py::arg("indices"),
         "Pin vertices at their current positions. indices: array of vertex indices.")

    .def("prescribe_motion", &PyConstraints::prescribe_motion,
         py::arg("vertex_idx"), py::arg("position_func"),
         "Prescribe time-varying motion for a vertex. "
         "position_func receives time t and returns [x, y, z] displacement.");
}
