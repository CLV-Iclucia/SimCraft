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
target_link_libraries(EngineFEM PUBLIC Kiseki::Core Kiseki::Runtime Kiseki::SceneObjects Kiseki::Deform Kiseki::EngineContactIPC Eigen3::Eigen)
target_compile_features(EngineFEM PUBLIC cxx_std_20)