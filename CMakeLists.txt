cmake_minimum_required(VERSION 4.1)
project(hands_on_profiling)

set(CMAKE_CXX_STANDARD 26)
find_package(OpenMP REQUIRED)

add_executable(hands_on_profiling main.cpp)
target_link_libraries(hands_on_profiling PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(hands_on_profiling PRIVATE -march=native -mavx2 -mfma)