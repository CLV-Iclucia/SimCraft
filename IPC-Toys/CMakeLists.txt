add_executable(rope-demo mass-spring-cipc.cc)
target_link_libraries(rope-demo PUBLIC Core glfw3 glad Eigen)
target_include_directories(rope-demo PUBLIC ${EIGEN3_INCLUDE_DIR})