cmake_minimum_required(VERSION 3.10)
project(layer_cpp)

set(CMAKE_CXX_STANDARD 17)

# find_package(OpenMP REQUIRED)

include_directories(./common)

file(GLOB SOURCES "common/*.cpp")
add_library(cipher_cpp SHARED ${SOURCES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

target_compile_options(cipher_cpp PRIVATE -mavx512f -maes -march=native)