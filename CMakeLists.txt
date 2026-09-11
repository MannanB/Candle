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
    cpp/activations.cpp
    cpp/autograd.cpp
    cpp/layers.cpp
    cpp/losses.cpp
    cpp/optimizers.cpp
    cpp/tensor.cpp
    cpp/utils.cpp
    cpp/kernels/tensor/add.cu
    cpp/kernels/activation/relu.cu
    cpp/kernels/activation/relu_bwd.cu
    cpp/kernels/loss/mse.cu
    cpp/kernels/loss/mse_bwd.cu
    cpp/kernels/tensor/scalar_multiply.cu
    cpp/kernels/activation/softmax.cu
    cpp/kernels/tensor/broadcast_add.cu
    cpp/kernels/tensor/transpose.cu
    cpp/kernels/tensor/gemm.cu
    cpp/kernels/tensor/sum_reduce.cu
    cpp/kernels/tensor/batched_gemm.cu
    cpp/kernels/tensor/broadcast_gemm.cu
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
