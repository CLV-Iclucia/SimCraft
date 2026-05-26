project(SimCraftApps LANGUAGES CXX)

# IPC FEM Simulator with optional rendering
add_executable(ipc-fem ipc-fem/main.cc)
target_link_libraries(ipc-fem PUBLIC FEM cxxopts Renderer)
target_compile_features(ipc-fem PUBLIC cxx_std_20)

message(STATUS "Apps configured: ipc-fem")
