project(simcraft-python LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(simcraft
    src/module.cc
    src/bind_mesh.cc
    src/bind_material.cc
    src/bind_system.cc
    src/bind_constraints.cc
    src/bind_kinematic.cc
    src/bind_integrator.cc
    src/bind_simulation.cc
    src/bind_renderer.cc
)
target_link_libraries(simcraft PRIVATE FEM Renderer)
target_include_directories(simcraft PRIVATE src)
target_compile_definitions(simcraft PRIVATE
    SIMCRAFT_VERSION="0.1.0"
)

# Install target — used by scikit-build-core to place .pyd into site-packages
install(TARGETS simcraft
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

# Also install runtime DLL dependencies (TBB, GLFW, etc.) alongside the .pyd
# so Python can find them at import time
if(WIN32)
  install(FILES $<TARGET_RUNTIME_DLLS:simcraft> DESTINATION . OPTIONAL)
endif()

enable_testing()
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(
      NAME python_test_import
      COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_import.py
  )
  set_tests_properties(python_test_import PROPERTIES
      ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:simcraft>"
  )
endif()
