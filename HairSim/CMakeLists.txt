file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
add_library(HairSim STATIC ${srcs})
target_include_directories(HairSim PUBLIC include)
target_link_libraries(HairSim PUBLIC Eigen3::Eigen)
target_include_directories(HairSim PUBLIC "../external/glm")