project(OglRender LANGUAGES CXX)
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h src/*.cc)

add_library(ogl-render STATIC ${srcs})
target_link_libraries(ogl-render PUBLIC GLAD glfw)
target_include_directories(ogl-render PUBLIC include ${EXTERNAL_PATH}/glm)

add_executable(demo apps/demo.cc)
target_include_directories(demo PUBLIC ${EXTERNAL_PATH}/imgui)
target_link_libraries(demo PUBLIC ogl-render imgui imgui_impl_glfw imgui_impl_opengl3 glfw)