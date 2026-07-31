project(EngineFEM LANGUAGES CXX)

set(ENGINE_FEM_SOURCES
    src/fem-scene.cc
    src/fem-subsystem.cc
    src/cpu/fem-cpu-backend.cc
)

set(ENGINE_FEM_HEADERS
    include/FEM/fem-scene.h
    include/FEM/fem-subsystem.h
    include/FEM/cpu/fem-cpu-backend.h
)

add_library(EngineFEM STATIC ${ENGINE_FEM_SOURCES} ${ENGINE_FEM_HEADERS})
add_library(Kiseki::EngineFEM ALIAS EngineFEM)
target_include_directories(EngineFEM PUBLIC include)
target_link_libraries(EngineFEM PUBLIC Kiseki::Core Kiseki::Runtime Kiseki::SceneObjects Kiseki::Deform Kiseki::EngineContactIPC Eigen3::Eigen TBB::tbb)
if(KISEKI_USE_CHOLMOD)
    target_compile_definitions(EngineFEM PUBLIC KISEKI_USE_CHOLMOD)
    if(TARGET SuiteSparse::CHOLMOD)
        target_link_libraries(EngineFEM PUBLIC SuiteSparse::CHOLMOD)
    elseif(TARGET CHOLMOD::CHOLMOD)
        target_link_libraries(EngineFEM PUBLIC CHOLMOD::CHOLMOD)
    elseif(TARGET CHOLMOD)
        target_link_libraries(EngineFEM PUBLIC CHOLMOD)
    else()
        message(FATAL_ERROR "KISEKI_USE_CHOLMOD is ON, but no CHOLMOD CMake target was found")
    endif()
endif()
target_compile_features(EngineFEM PUBLIC cxx_std_20)
