project(Fluid LANGUAGES CXX)

file(GLOB_RECURSE cpu_srcs CONFIGURE_DEPENDS src/cpu/*.cc src/cpu/*.cpp)
file(GLOB_RECURSE cpu_headers CONFIGURE_DEPENDS include/FluidSim/cpu/*.h)

file(GLOB_RECURSE gpu_srcs CONFIGURE_DEPENDS src/cuda/*.cc src/cuda/*.cpp src/cuda/*.cu src/cuda/*.cc)
file(GLOB_RECURSE gpu_headers CONFIGURE_DEPENDS include/FluidSim/cuda/*.h include/FluidSim/cuda/*.cuh)

file(GLOB common_headers CONFIGURE_DEPENDS include/FluidSim/*.h)
file(GLOB common_srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp)

add_library(CpuFluid STATIC ${cpu_srcs} ${cpu_headers} ${common_headers} ${common_srcs})

target_include_directories(CpuFluid PUBLIC include/)
# target_include_directories(GpuFluid PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} include)

target_link_libraries(CpuFluid PUBLIC Core Spatify TBB::tbb)
# target_link_libraries(GpuFluid PUBLIC Core)

# target_compile_options(GpuFluid PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_definitions(-DFLUIDSIM_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
add_definitions(-DSIMCRAFT_ASSETS_DIR="${ASSETS_PATH}")

add_executable(fluid-sim apps/fluid-sim.cc)
target_include_directories(fluid-sim PUBLIC ${EXTERNAL_PATH}/imgui)
target_include_directories(fluid-sim PUBLIC ${GLAD_INCLUDE_DIR})
target_link_libraries(fluid-sim PUBLIC ogl-render CpuFluid glfw)

# add_executable(gpu-smoke apps/gpu-smoke.cu)
# target_include_directories(gpu-smoke PUBLIC ${GLAD_INCLUDE_DIR})
# target_link_libraries(gpu-smoke PUBLIC ogl-render GpuFluid glfw)

# add_executable(gpu-free-surface apps/gpu-free-surface.cu)
# target_include_directories(gpu-free-surface PUBLIC ${GLAD_INCLUDE_DIR})
# target_link_libraries(gpu-free-surface PUBLIC ogl-render GpuFluid glfw)

# find_package(GTest CONFIG REQUIRED)
# file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cu)
# foreach(test_src ${test_srcs})
#   get_filename_component(test_name ${test_src} NAME_WE)
#   add_executable(${test_name} ${test_src})
#   target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#   target_link_libraries(${test_name} PUBLIC Core GTest::gtest GTest::gtest_main GpuFluid)
#   add_test(NAME ${test_name} CONFIGURATIONS Release COMMAND ${test_name})
# endforeach()
