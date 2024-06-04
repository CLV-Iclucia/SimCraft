project(Core LANGUAGES CXX CUDA)
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
add_library(Core STATIC ${srcs})
target_include_directories(Core PUBLIC include)
target_include_directories(Core PUBLIC ${EXTERNAL_PATH}/glm)
target_include_directories(Core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(Core PUBLIC ${GLAD_INCLUDE_DIR})
target_link_libraries(Core PUBLIC OpenGL GLAD)

enable_testing()
# for all the files in tests, build test executables
file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src}
            include/Core/debug.h
            include/Core/cuda-utils.h
            include/Core/log.h
            include/Core/dynamic-reflection.h
            include/Core/zip.h
            tests/zip-test.cc
    )
    target_link_libraries(${test_name} PUBLIC Core gtest gtest_main)
    add_test(NAME ${test_name} CONFIGURATIONS Release COMMAND ${test_name})
endforeach()