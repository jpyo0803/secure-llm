cmake_minimum_required(VERSION 3.10)
project(layer_cpp)

set(CMAKE_CXX_STANDARD 17)

# find_package(OpenMP REQUIRED)

include_directories(./cpp)

file(GLOB SOURCES "c/*.c")
add_library(cipher_cpp SHARED ${SOURCES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# Set the optimization flag for C files
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# # Add OpenMP flags if found
# if (OpenMP_C_FOUND)
#     target_compile_options(cipher_cpp PRIVATE ${OpenMP_C_FLAGS})
#     target_link_libraries(cipher_cpp PUBLIC OpenMP::OpenMP_C)
# endif()

# if (OpenMP_CXX_FOUND)
#     target_compile_options(cipher_cpp PRIVATE ${OpenMP_CXX_FLAGS})
#     target_link_libraries(cipher_cpp PUBLIC OpenMP::OpenMP_CXX)
# endif()

# Additional compiler options
# target_compile_options(cipher_cpp PRIVATE -maes -fopenmp -march=native)
target_compile_options(cipher_cpp PRIVATE -mavx512f -maes -march=native)