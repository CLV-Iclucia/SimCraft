project(FEM LANGUAGES CXX)
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
add_library(FEM STATIC ${srcs}
        src/system.cc)
target_link_libraries(FEM PUBLIC Maths Core Spatify Deform)
target_include_directories(FEM PUBLIC include)
add_executable(fem apps/fem.cc)
target_link_libraries(fem PUBLIC FEM)