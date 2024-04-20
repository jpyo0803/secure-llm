cmake_minimum_required(VERSION 3.10)
project(layer_cpp)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

include_directories(./cpp)

file(GLOB SOURCES "cpp/*.cpp")
add_library(cipher_cpp SHARED ${SOURCES})

target_compile_options(cipher_cpp PRIVATE -Ofast -maes -march=native)

if (OpenMP_CXX_FOUND)
    target_link_libraries(cipher_cpp PUBLIC OpenMP::OpenMP_CXX)
endif ()

add_subdirectory(cpp_unittests)