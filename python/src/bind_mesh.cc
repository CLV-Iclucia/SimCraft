// Phase 2: TetMesh binding
#include "bindings.h"
#include <filesystem>

void bind_mesh(py::module_& m)
{
  py::class_<TetMesh>(m, "TetMesh")
    .def(py::init([](py::array_t<double> vertices, py::array_t<int> tets) {
      auto vbuf = vertices.unchecked<2>();
      auto tbuf = tets.unchecked<2>();

      if (vbuf.shape(1) != 3)
        throw py::value_error("vertices must have shape (N, 3)");
      if (tbuf.shape(1) != 4)
        throw py::value_error("tets must have shape (M, 4)");

      std::vector<Vector<Real, 3>> verts(vbuf.shape(0));
      for (py::ssize_t i = 0; i < vbuf.shape(0); i++)
        verts[i] = Vector<Real, 3>(vbuf(i, 0), vbuf(i, 1), vbuf(i, 2));

      std::vector<Tetrahedron> tetras(tbuf.shape(0));
      for (py::ssize_t i = 0; i < tbuf.shape(0); i++)
        tetras[i] = Tetrahedron(tbuf(i, 0), tbuf(i, 1), tbuf(i, 2), tbuf(i, 3));

      return TetMesh(verts, tetras);
    }), py::arg("vertices"), py::arg("tets"),
       "Construct TetMesh from numpy arrays of shape (N,3) and (M,4)")

    .def_static("from_file", [](const std::string& path) {
      auto mesh = readTetMeshFromTOBJ(std::filesystem::path(path));
      if (!mesh.has_value())
        throw py::value_error("Failed to load mesh from: " + path);
      return std::move(mesh.value());
    }, py::arg("path"),
       "Load TetMesh from a .tobj file")

    .def_property_readonly("num_vertices", [](const TetMesh& self) {
      return self.getVertices().size();
    }, "Number of vertices in the mesh")

    .def_property_readonly("num_elements", [](const TetMesh& self) {
      return self.tets.size();
    }, "Number of tetrahedral elements")

    .def_property_readonly("num_surface_triangles", [](const TetMesh& self) {
      return self.surfaces.size();
    }, "Number of surface triangles");
}
