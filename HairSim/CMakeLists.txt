file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
set(MKL_THREADING "tbb_thread")
set(MKL_INTERFACE "lp64")

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
add_library(HairSim STATIC ${srcs})

target_include_directories(HairSim PUBLIC include)
target_include_directories(HairSim PUBLIC ${MKL_INCLUDE})
target_link_libraries(HairSim PUBLIC Core Eigen3::Eigen MKL::MKL)

file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src}
            include/HairSim/time-steppers.h)
    target_link_libraries(${test_name} PUBLIC Core gtest gtest_main HairSim)
    add_test(NAME ${test_name} CONFIGURATIONS Release COMMAND ${test_name})
endforeach()