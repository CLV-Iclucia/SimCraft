project(Fluid LANGUAGES CXX)

file(GLOB_RECURSE cpu_srcs CONFIGURE_DEPENDS src/cpu/*.cc src/cpu/*.cpp)
file(GLOB_RECURSE cpu_headers CONFIGURE_DEPENDS include/FluidSim/cpu/*.h)
file(GLOB common_headers CONFIGURE_DEPENDS include/FluidSim/*.h)
file(GLOB common_srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp)

add_library(CpuFluid STATIC ${cpu_srcs} ${cpu_headers} ${common_headers} ${common_srcs})

target_include_directories(CpuFluid PUBLIC include/)
target_link_libraries(CpuFluid PUBLIC Core Spatify TBB::tbb)

target_compile_definitions(CpuFluid PRIVATE
    FLUIDSIM_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    SIMCRAFT_ASSETS_DIR="${ASSETS_PATH}"
)
