file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
add_library(MPM STATIC ${srcs})
target_include_directories(MPM PUBLIC include)
target_include_directories(MPM PUBLIC "${EXTERNAL_PATH}/imgui")
target_link_libraries(MPM PUBLIC Core)
target_link_libraries(MPM PUBLIC Eigen)

add_executable(rope-sim apps/rope-sim.cc)
