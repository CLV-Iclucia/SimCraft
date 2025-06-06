project(FEM LANGUAGES CXX)

option(FEM_BUILD_GUI "Build the GUI" OFF)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
add_library(FEM STATIC ${srcs})
target_link_libraries(FEM PUBLIC Maths Core Spatify Deform)
target_include_directories(FEM PUBLIC include)

set(FEM_ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(FEM_TETS_DIR ${FEM_ASSETS_DIR}/tets)
add_definitions(-DFEM_ASSETS_DIR="${FEM_ASSETS_DIR}")
add_definitions(-DFEM_TETS_DIR="${FEM_TETS_DIR}")

file(GLOB APPLICATIONS apps/*.cc)
foreach (app ${APPLICATIONS})
    get_filename_component(app_name ${app} NAME_WE)
    add_executable(${app_name} ${app})
    target_link_libraries(${app_name} PUBLIC FEM cxxopts)
    if (FEM_BUILD_GUI)
        target_link_libraries(${app_name} PUBLIC ogl-render)
    endif ()
endforeach ()

enable_testing()
file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PUBLIC FEM GTest::gtest GTest::gtest_main)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_test(NAME ${test_name} CONFIGURATIONS Debug COMMAND ${test_name})
    else()
        add_test(NAME ${test_name} CONFIGURATIONS Release COMMAND ${test_name})
    endif()
endforeach()