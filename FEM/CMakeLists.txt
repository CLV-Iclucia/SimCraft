project(FEM LANGUAGES CXX)
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
add_library(FEM STATIC ${srcs}
        src/collision-resolver.cc
        src/system.cc
        src/integrator.cc)
target_link_libraries(FEM PUBLIC Maths Core Spatify)
target_include_directories(FEM PUBLIC include)
add_executable(fem apps/fem.cc
        include/fem/system.h)
target_link_libraries(fem PUBLIC FEM)