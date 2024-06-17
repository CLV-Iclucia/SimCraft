project(OglRender LANGUAGES CXX)
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h src/*.cc)

add_library(ogl-render STATIC ${srcs})
target_link_libraries(ogl-render PUBLIC glad glfw)
target_include_directories(ogl-render PUBLIC include ${EXTERNAL_PATH}/glm)

add_executable(mesh-view apps/mesh-view.cc
        include/ogl-render/glad-glfw.h)
target_include_directories(mesh-view PUBLIC ${EXTERNAL_PATH}/imgui)
target_link_libraries(mesh-view PUBLIC ogl-render imgui imgui_impl_glfw imgui_impl_opengl3 glfw)