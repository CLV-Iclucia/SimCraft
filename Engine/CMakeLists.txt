# Engine meta-module containing Runtime, SceneObjects, Subsystems, and Scene.
# DER is the first subsystem in the Subsystems layer.
add_subdirectory(Contact)
add_subdirectory(Runtime)
add_subdirectory(SceneObjects)
add_subdirectory(Subsystems)

include(${CMAKE_SOURCE_DIR}/ReflectGen/cmake/ReflectGen.cmake)
reflectgen_add_registry(
    TARGET engine_reflection
    SOURCE_TARGETS Runtime SceneObjects DER
)
target_link_libraries(engine_reflection PRIVATE Kiseki::Runtime)
get_target_property(engine_reflection_output_dir engine_reflection REFLECTGEN_OUTPUT_DIR)
get_target_property(engine_reflection_generate_target engine_reflection REFLECTGEN_GENERATE_TARGET)
target_include_directories(Runtime PUBLIC "${engine_reflection_output_dir}")
add_dependencies(Runtime "${engine_reflection_generate_target}")
add_dependencies(SceneObjects engine_reflection)
add_dependencies(DER engine_reflection)
target_link_libraries(SceneObjects PUBLIC engine_reflection)
target_link_libraries(DER PUBLIC engine_reflection)
