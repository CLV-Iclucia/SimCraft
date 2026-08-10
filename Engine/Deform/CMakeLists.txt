project(EngineDeform LANGUAGES CXX)

set(ENGINE_DEFORM_SOURCES
    src/strain-energy-density.cc
)

set(ENGINE_DEFORM_HEADERS
    include/Engine/Deform/deformation-gradient.h
    include/Engine/Deform/invariants.h
    include/Engine/Deform/strain-energy-density.h
    include/Engine/Deform/types.h
)

add_library(EngineDeform STATIC ${ENGINE_DEFORM_SOURCES} ${ENGINE_DEFORM_HEADERS})
add_library(Kiseki::EngineDeform ALIAS EngineDeform)
target_include_directories(EngineDeform PUBLIC include)
target_link_libraries(EngineDeform PUBLIC Kiseki::Core Kiseki::Maths)
target_compile_features(EngineDeform PUBLIC cxx_std_20)
