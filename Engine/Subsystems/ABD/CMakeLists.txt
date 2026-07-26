project(EngineABD LANGUAGES CXX)

set(ENGINE_ABD_SOURCES
    src/abd-scene.cc
    src/abd-subsystem.cc
    src/cpu/abd-cpu-backend.cc
)

set(ENGINE_ABD_HEADERS
    include/ABD/abd-scene.h
    include/ABD/abd-subsystem.h
    include/ABD/cpu/abd-cpu-backend.h
)

add_library(EngineABD STATIC ${ENGINE_ABD_SOURCES} ${ENGINE_ABD_HEADERS})
add_library(Kiseki::EngineABD ALIAS EngineABD)
target_include_directories(EngineABD PUBLIC include)
target_link_libraries(EngineABD PUBLIC Kiseki::Core Kiseki::Runtime Kiseki::SceneObjects Kiseki::EngineContactIPC Eigen3::Eigen)
target_compile_features(EngineABD PUBLIC cxx_std_20)

enable_testing()
file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PUBLIC
        Kiseki::EngineABD
        GTest::gtest
        GTest::gtest_main
    )
    target_compile_features(${test_name} PUBLIC cxx_std_20)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
