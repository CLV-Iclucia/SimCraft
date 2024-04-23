cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
project(SimCraft)

set(GLAD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include)
set(EXTERNAL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(ASSETS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${EXTERNAL_PATH}/eigen/cmake/")
find_package(TBB REQUIRED)

add_subdirectory(external/glad)
add_subdirectory(external/imgui)
add_subdirectory(Core)
add_subdirectory(OglRender)
add_subdirectory(Spatify)
#add_subdirectory(PoissonSolver)
add_subdirectory(FluidSim)
#add_subdirectory(HairSim)
