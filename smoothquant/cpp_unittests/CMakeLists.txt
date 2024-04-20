cmake_minimum_required(VERSION 3.10)

# Set the project name
project(aes_stream_unittest)

# Add the directory with the source files
include_directories(../cpp)

# Add the executable
add_executable(${PROJECT_NAME} aes_stream_unittest.cpp)

# Link the libraries needed for the test
# Replace 'library_name' with the actual names of the libraries
target_link_libraries(${PROJECT_NAME} cipher_cpp)