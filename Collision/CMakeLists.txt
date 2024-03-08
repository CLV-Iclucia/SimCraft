file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h include/*.inl)
add_library(ColDetect STATIC ${srcs}
        include/collision/collision-detect.h)
target_include_directories(ColDetect PUBLIC include)
target_link_libraries(ColDetect PUBLIC Core TBB::tbb)

add_executable(lbvh-vis apps/vis-2d.cc)
target_link_libraries(lbvh-vis ColDetect GLAD glfw3 Core)