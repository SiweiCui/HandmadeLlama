include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(../test DIR_TEST)

add_executable(test_handmadellama ${DIR_TEST})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
target_link_libraries(test_handmadellama ${link_ext_lib})
target_link_libraries(test_handmadellama handmadellama)

target_link_directories(test_handmadellama PUBLIC ${PROJECT_SOURCE_DIR}/lib)

target_include_directories(test_handmadellama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_handmadellama PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_handmadellama PUBLIC ../include)

set_target_properties(test_handmadellama PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_handmadellama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)