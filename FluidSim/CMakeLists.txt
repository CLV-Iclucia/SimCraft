project(Fluid LANGUAGES CXX CUDA)

# file(GLOB_RECURSE gpu_srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/FluidSim/gpu-fluid/*.h)
file(GLOB_RECURSE common_srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp)
file(GLOB_RECURSE common_headers CONFIGURE_DEPENDS include/FluidSim/common/*.h)
add_library(FluidCommon STATIC ${common_srcs} ${common_headers})

target_include_directories(FluidCommon PUBLIC include)
# target_include_directories(FluidCommon PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(FluidCommon PUBLIC Core Eigen3::Eigen Spatify TBB::tbb)
# target_compile_options(Fluid PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_definitions(-DFLUIDSIM_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
add_definitions(-DSIMCRAFT_ASSETS_DIR="${ASSETS_PATH}")

add_executable(fluid-sim apps/fluid-sim.cc)
target_include_directories(fluid-sim PUBLIC ${EXTERNAL_PATH}/imgui)
target_include_directories(fluid-sim PUBLIC ${GLAD_INCLUDE_DIR})
target_link_libraries(fluid-sim PUBLIC ogl-render
        imgui imgui_impl_glfw imgui_impl_opengl3 FluidCommon glfw3)