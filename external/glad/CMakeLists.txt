project(glad LANGUAGES C)
file(GLOB_RECURSE GLAD_SOURCES src/*.c include/*.h)
add_library(glad STATIC ${GLAD_SOURCES})
target_include_directories(glad PUBLIC include)