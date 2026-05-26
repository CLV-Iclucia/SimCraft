project(FEM LANGUAGES CXX)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
add_library(FEM STATIC ${srcs})
target_link_libraries(FEM PUBLIC Maths Core Spatify Deform)
target_include_directories(FEM PUBLIC include)

set(FEM_ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(FEM_TETS_DIR ${FEM_ASSETS_DIR}/tets)
target_compile_definitions(FEM PUBLIC
    FEM_ASSETS_DIR="${FEM_ASSETS_DIR}"
    FEM_TETS_DIR="${FEM_TETS_DIR}"
)

# Developer test apps (headless)
add_executable(toy-ipc apps/toy-ipc.cc)
target_link_libraries(toy-ipc PUBLIC FEM)

enable_testing()
file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PUBLIC FEM GTest::gtest GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
