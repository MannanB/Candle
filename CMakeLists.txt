cmake_minimum_required(VERSION 3.24)

project(candle LANGUAGES CXX CUDA)

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
    cpp/bindings.cpp
    cpp/tensor.cpp
    cpp/utils.cpp
    cpp/kernels/add.cu
    cpp/kernels/broadcast_add.cu
    cpp/kernels/transpose.cu
    cpp/kernels/gemm.cu
    cpp/kernels/batched_gemm.cu
    cpp/kernels/broadcast_gemm.cu
)

target_include_directories(
    _candle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/kernels
)

target_link_libraries(
    _candle PRIVATE
    CUDA::cudart
)

set_target_properties(
    _candle PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
