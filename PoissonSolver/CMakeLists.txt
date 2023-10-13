# poisson equation solvers on CPU square/cubes and GPU square/cubes
project(PoissonSolver LANGUAGES CXX CUDA)
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
add_library(PoissonSolver STATIC ${srcs})
target_link_libraries(PoissonSolver PUBLIC TBB::tbb)
target_include_directories(PoissonSolver PUBLIC include)
# target_compile_options(PoissonSolver PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
add_executable(poisson-test test/poisson-test.cc)
target_link_libraries(poisson-test PUBLIC PoissonSolver)