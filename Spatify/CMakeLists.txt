project(Spatify LANGUAGES CXX)
add_library(Spatify INTERFACE)
target_include_directories(Spatify INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_include_directories(Spatify INTERFACE ${EXTERNAL_PATH}/glm)