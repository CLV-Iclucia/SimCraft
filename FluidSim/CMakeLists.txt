project(Fluid LANGUAGES CXX CUDA)

file(GLOB_RECURSE cpu_srcs CONFIGURE_DEPENDS src/cpu/*.cc src/cpu/*.cpp)
file(GLOB_RECURSE cpu_headers CONFIGURE_DEPENDS include/FluidSim/cpu/*.h)

file(GLOB_RECURSE gpu_srcs CONFIGURE_DEPENDS src/gpu/*.cc src/gpu/*.cpp src/gpu/*.cu)
file(GLOB_RECURSE gpu_headers CONFIGURE_DEPENDS include/FluidSim/gpu/*.h include/FluidSim/gpu/*.cuh)

file(GLOB common_headers CONFIGURE_DEPENDS include/FluidSim/*.h)

add_library(CpuFluid STATIC ${cpu_srcs} ${cpu_headers} ${common_headers})
add_library(GpuFluid STATIC ${gpu_srcs} ${gpu_headers} ${common_headers})

target_include_directories(CpuFluid PUBLIC include)
target_include_directories(GpuFluid PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} include)

target_link_libraries(CpuFluid PUBLIC Core Spatify TBB::tbb)
target_link_libraries(GpuFluid PUBLIC Core)

target_compile_options(GpuFluid PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_definitions(-DFLUIDSIM_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
add_definitions(-DSIMCRAFT_ASSETS_DIR="${ASSETS_PATH}")

add_executable(fluid-sim apps/fluid-sim.cc)
target_include_directories(fluid-sim PUBLIC ${EXTERNAL_PATH}/imgui)
target_include_directories(fluid-sim PUBLIC ${GLAD_INCLUDE_DIR})
target_link_libraries(fluid-sim PUBLIC ogl-render
        imgui imgui_impl_glfw imgui_impl_opengl3 CpuFluid glfw3)

add_executable(gpu-smoke apps/gpu-smoke.cc)
target_include_directories(gpu-smoke PUBLIC ${EXTERNAL_PATH}/imgui)
target_include_directories(gpu-smoke PUBLIC ${GLAD_INCLUDE_DIR})
target_link_libraries(gpu-smoke PUBLIC ogl-render
        imgui imgui_impl_glfw imgui_impl_opengl3 GpuFluid glfw3)
