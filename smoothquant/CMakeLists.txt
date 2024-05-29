# cmake_minimum_required(VERSION 3.10)
# project(layer_cpp)

# set(CMAKE_CXX_STANDARD 17)

# find_package(OpenMP REQUIRED)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

# include_directories(./cpp)

# file(GLOB SOURCES "c/*.c")
# add_library(cipher_cpp SHARED ${SOURCES})

# # target_compile_options(cipher_cpp PRIVATE -Ofast -maes -march=native)
# target_compile_options(cipher_cpp PRIVATE -maes -fopenmp)

# if (OpenMP_CXX_FOUND)
#     target_link_libraries(cipher_cpp PUBLIC OpenMP::OpenMP_CXX m)
#     # target_link_libraries(cipher_cpp PUBLIC m)
# endif ()

# # add_subdirectory(cpp_unittests)

cmake_minimum_required(VERSION 3.10)
project(layer_cpp)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

include_directories(./cpp)

file(GLOB SOURCES "c/*.c")
add_library(cipher_cpp SHARED ${SOURCES})

# Add OpenMP flags if found
if (OpenMP_C_FOUND)
    target_compile_options(cipher_cpp PRIVATE ${OpenMP_C_FLAGS})
    target_link_libraries(cipher_cpp PUBLIC OpenMP::OpenMP_C)
endif()

if (OpenMP_CXX_FOUND)
    target_compile_options(cipher_cpp PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(cipher_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

# Additional compiler options
target_compile_options(cipher_cpp PRIVATE -maes -fopenmp -march=native)