project(OglRender LANGUAGES CXX)

option(BUILD_EXTERNAL_GUI "Build external GUI" ON)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/ogl-render/*.h src/*.cc)
if (BUILD_EXTERNAL_GUI)
file(GLOB_RECURSE ext_srcs CONFIGURE_DEPENDS ext/*.cc include/ext-gui/*.h)
set(EXTERNAL_GUI_LIBS imgui imgui_impl_glfw imgui_impl_opengl3)
set(EXTERNAL_GUI_INCLUDE_DIR ${IMGUI_INCLUDE_DIR})
endif()

add_definitions(-DOGL_RENDER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_library(ogl-render STATIC ${srcs} ${ext_srcs})
target_link_libraries(ogl-render PUBLIC glad glfw ${EXTERNAL_GUI_LIBS})
target_include_directories(ogl-render PUBLIC include ${EXTERNAL_PATH}/glm ${EXTERNAL_GUI_INCLUDE_DIR})

add_executable(mesh-view apps/mesh-view.cc
        include/ogl-render/platform.h)
target_link_libraries(mesh-view PUBLIC ogl-render)

add_executable(perf-lab apps/perf-lab.cc)
target_link_libraries(perf-lab PUBLIC ogl-render)