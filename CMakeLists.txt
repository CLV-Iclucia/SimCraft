cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52;70;75;86)
project(SimCraft)

find_package(Eigen3 REQUIRED)
add_subdirectory(external/imgui)
add_subdirectory(Core)
add_subdirectory(FluidSim)
add_subdirectory(MPM)
add_subdirectory(HairSim)

