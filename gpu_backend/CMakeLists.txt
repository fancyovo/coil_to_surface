cmake_minimum_required(VERSION 3.22)
project(stellarator_gpu_backend LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(stellarator_gpu SHARED
    src/coil_field.cu
)

target_include_directories(stellarator_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(stellarator_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
