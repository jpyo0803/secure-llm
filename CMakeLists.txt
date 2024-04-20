cmake_minimum_required(VERSION 3.10)
project(cipher_cpp VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

# add_compile_options(-march=native)

# Find OpenMP
find_package(OpenMP REQUIRED)

find_package(pybind11 REQUIRED)

pybind11_add_module(cipher_cpp cipher_cpp/cipher_pybind.cpp cipher_cpp/cipher.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cipher_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()