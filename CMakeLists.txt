cmake_minimum_required(VERSION 3.24)

project(mycuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(
    Python REQUIRED
    COMPONENTS Interpreter Development.Module
)

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(
    _candle
    candle.cpp
    kernels/add.cu
    kernels/broadcast_add.cu
    kernels/transpose.cu
    kernels/gemm.cu
    kernels/batched_gemm.cu
    kernels/broadcast_gemm.cu
)

target_include_directories(
    _candle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_link_libraries(
    _candle PRIVATE
    CUDA::cudart
)

set_target_properties(
    _candle PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
