cmake_minimum_required(VERSION 3.20)
project(imgui)
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")

set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)

set(ROOT ${IMGUI_DIR})

add_library(imgui STATIC
        ${ROOT}/imgui.cpp
        ${ROOT}/imgui.h
        ${ROOT}/imstb_rectpack.h
        ${ROOT}/imstb_textedit.h
        ${ROOT}/imstb_truetype.h
        ${ROOT}/imgui_demo.cpp
        ${ROOT}/imgui_draw.cpp
        ${ROOT}/imgui_internal.h
        ${ROOT}/imgui_tables.cpp
        ${ROOT}/imgui_widgets.cpp
)
target_include_directories(imgui PUBLIC
        $<BUILD_INTERFACE:${ROOT}>
        $<INSTALL_INTERFACE:include>)

set(INSTALL_TARGETS imgui)
set(INSTALL_HEADERS ${ROOT}/imgui.h ${ROOT}/imconfig.h)

foreach(BACKEND glfw opengl3 vulkan)
    set(NAME imgui_impl_${BACKEND})
    set(HEADER ${ROOT}/${NAME}.h)
    add_library(${NAME} STATIC ${ROOT}/${NAME}.cpp ${HEADER})
    target_link_libraries(${NAME} PUBLIC imgui)
    target_include_directories(${NAME} PUBLIC
            $<BUILD_INTERFACE:${ROOT}>
            $<INSTALL_INTERFACE:include>)
endforeach()

target_link_libraries(imgui_impl_glfw PRIVATE glfw)
target_sources(imgui_impl_opengl3 PRIVATE ${ROOT}/imgui_impl_opengl3_loader.h)
target_link_libraries(imgui_impl_opengl3 PRIVATE OpenGL::GL)