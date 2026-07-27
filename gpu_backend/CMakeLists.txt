cmake_minimum_required(VERSION 3.22)
project(stellarator_gpu_backend LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(stellarator_gpu SHARED
    src/coil_field.cu
    src/score_pipeline.cu
)

target_include_directories(stellarator_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(stellarator_gpu PUBLIC CUDA::cublas CUDA::cusolver)
target_link_options(stellarator_gpu PRIVATE -static-libstdc++ -static-libgcc)

set(SGPU_WARPS_PER_BLOCK "8" CACHE STRING "Warps per block for warp-per-line kernels")
set(SGPU_SEG_TILE "256" CACHE STRING "Segment tile size")

target_compile_definitions(stellarator_gpu PRIVATE
    SGPU_WARPS_PER_BLOCK=${SGPU_WARPS_PER_BLOCK}
    SGPU_SEG_TILE=${SGPU_SEG_TILE}
)

set_target_properties(stellarator_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
