#include "tensor.h"

#include <cuda_runtime_api.h>

#include <stdexcept>

#include "kernels/add.h"
#include "kernels/batched_gemm.h"
#include "kernels/broadcast_add.h"
#include "kernels/broadcast_gemm.h"
#include "kernels/gemm.h"
#include "kernels/transpose.h"
#include "utils.h"

Tensor::Tensor(float* host_data, int size, int* shape, int ndim) : size(size), shape(shape), ndim(ndim) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        data,
        host_data,
        size * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaFreeHost(host_data));
}

Tensor::Tensor(int size, int* shape, int ndim) : size(size), shape(shape), ndim(ndim) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(float)));
}

Tensor::~Tensor() {
    if (data != nullptr) {
        cudaFree(data);
    }
    delete[] shape;
}

std::unique_ptr<Tensor> Tensor::transpose(int dim1, int dim2) const {
    int prefix_dim = 1;
    for (int d = 0; d < dim1; ++d) {
        prefix_dim *= shape[d];
    }

    int suffix_dim = 1;
    for (int d = dim2 + 1; d < ndim; ++d) {
        suffix_dim *= shape[d];
    }

    int* new_shape = new int[ndim];
    for (int d = 0; d < ndim; ++d) {
        new_shape[d] = shape[d];
    }

    const int buffer = new_shape[dim1];
    new_shape[dim1] = new_shape[dim2];
    new_shape[dim2] = buffer;

    auto out = std::make_unique<Tensor>(size, new_shape, ndim);
    launch_mat_transpose_kernel(
        data,
        out->data,
        prefix_dim,
        shape[dim1],
        shape[dim2],
        suffix_dim
    );
    return out;
}

std::unique_ptr<Tensor> Tensor::add(const Tensor* a, const Tensor* b) {
    bool same_shape = a->ndim == b->ndim;
    if (same_shape) {
        for (int d = 0; d < a->ndim; ++d) {
            if (a->shape[d] != b->shape[d]) {
                same_shape = false;
                break;
            }
        }
    }

    if (same_shape) {
        int* out_shape = new int[a->ndim];
        for (int d = 0; d < a->ndim; ++d) {
            out_shape[d] = a->shape[d];
        }

        auto out = std::make_unique<Tensor>(a->size, out_shape, a->ndim);
        launch_vec_add_kernel(a->data, b->data, out->data, a->size);
        return out;
    }

    const Tensor* batched = a->size > b->size ? a : b;
    const Tensor* broadcasted = a->size > b->size ? b : a;

    if (broadcasted->ndim >= batched->ndim || batched->size % broadcasted->size != 0) {
        throw std::invalid_argument(
            "Tensor shapes cannot be broadcast for addition"
        );
    }

    const int dim_offset = batched->ndim - broadcasted->ndim;
    for (int d = 0; d < broadcasted->ndim; ++d) {
        if (batched->shape[dim_offset + d] != broadcasted->shape[d]) {
            throw std::invalid_argument(
                "Tensor shapes cannot be broadcast for addition"
            );
        }
    }

    int* out_shape = new int[batched->ndim];
    for (int d = 0; d < batched->ndim; ++d) {
        out_shape[d] = batched->shape[d];
    }

    auto out = std::make_unique<Tensor>(
        batched->size,
        out_shape,
        batched->ndim
    );
    launch_broadcast_vec_add_kernel(
        batched->data,
        broadcasted->data,
        out->data,
        broadcasted->size,
        batched->size / broadcasted->size
    );
    return out;
}

std::unique_ptr<Tensor> Tensor::uniform(
    int* shape,
    int ndim,
    float min,
    float max
) {
    int size = 1;
    for (int d = 0; d < ndim; ++d) {
        size *= shape[d];
    }

    float* host_data = nullptr;
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&host_data),
        size * sizeof(float)
    ));

    for (int i = 0; i < size; ++i) {
        host_data[i] = random_float(min, max);
    }

    return std::make_unique<Tensor>(host_data, size, shape, ndim);
}

std::unique_ptr<Tensor> Tensor::sum(int dim) {
    // sum across dim
    // TODO: implement
    return nullptr;
}


std::unique_ptr<Tensor> Tensor::matmul(const Tensor* a, const Tensor* b) {
    if (a->ndim < 2 || b->ndim < 1) {
        throw std::invalid_argument("Matmul requires a matrix on the left");
    }

    const int a_rows = a->shape[a->ndim - 2];
    const int a_cols = a->shape[a->ndim - 1];
    int b_rows = 1;
    int b_cols = 1;
    int batch_size = 1;
    int out_ndim = 2;
    int* new_shape = nullptr;
    const bool batched = a->ndim > 2;
    bool broadcast = false;

    if (batched) {
        if (b->ndim == a->ndim) {
            b_rows = b->shape[b->ndim - 2];
            b_cols = b->shape[b->ndim - 1];

            for (int d = 0; d < a->ndim - 2; ++d) {
                if (a->shape[d] != b->shape[d]) {
                    throw std::invalid_argument("Batch dimensions must match");
                }
                batch_size *= a->shape[d];
            }
        } else if (b->ndim == a->ndim - 1) {
            b_rows = b->shape[b->ndim - 1];

            for (int d = 0; d < b->ndim - 1; ++d) {
                if (a->shape[d] != b->shape[d]) {
                    throw std::invalid_argument("Batch dimensions must match");
                }
                batch_size *= a->shape[d];
            }
        } else {
            throw std::invalid_argument("Unsupported batched matmul dimensions");
        }

        out_ndim = a->ndim;
        new_shape = new int[out_ndim];
        for (int d = 0; d < out_ndim - 2; ++d) {
            new_shape[d] = a->shape[d];
        }
        new_shape[out_ndim - 2] = a_rows;
        new_shape[out_ndim - 1] = b_cols;
    } else if (b->ndim == 1) {
        b_rows = b->shape[0];
        new_shape = new int[2]{a_rows, 1};
    } else if (a_cols == b->shape[b->ndim - 2]) {
        b_rows = b->shape[b->ndim - 2];
        b_cols = b->shape[b->ndim - 1];
        broadcast = b->ndim > 2;
        out_ndim = b->ndim;
        new_shape = new int[out_ndim];

        for (int d = 0; d < b->ndim - 2; ++d) {
            new_shape[d] = b->shape[d];
            batch_size *= b->shape[d];
        }
        new_shape[out_ndim - 2] = a_rows;
        new_shape[out_ndim - 1] = b_cols;
    } else if (a_cols == b->shape[b->ndim - 1]) {
        b_rows = b->shape[b->ndim - 1];
        broadcast = true;
        out_ndim = b->ndim + 1;
        new_shape = new int[out_ndim];

        for (int d = 0; d < b->ndim - 1; ++d) {
            new_shape[d] = b->shape[d];
            batch_size *= b->shape[d];
        }
        new_shape[out_ndim - 2] = a_rows;
        new_shape[out_ndim - 1] = 1;
    } else {
        throw std::invalid_argument("Dimensions must line up for matmul");
    }

    if (a_cols != b_rows) {
        delete[] new_shape;
        throw std::invalid_argument("Dimensions must line up for matmul");
    }

    auto out = std::make_unique<Tensor>(
        batch_size * a_rows * b_cols,
        new_shape,
        out_ndim
    );

    if (batched) {
        launch_batched_mat_mul_kernel(
            a->data, b->data, out->data, 
            batch_size, a_rows, a_cols,  b_cols
        );
    } else if (broadcast) {
        launch_broadcast_mat_mul_kernel(
            a->data, b->data, out->data,
            batch_size, a_rows, a_cols, b_cols
        );
    } else {
        launch_mat_mul_kernel(
            a->data, b->data, out->data,
            a_rows, a_cols, b_cols
        );
    }

    return out;
}

std::unique_ptr<Tensor> Tensor::operator+(const Tensor& other) const {
    return add(this, &other);
}

std::unique_ptr<Tensor> Tensor::matmul(const Tensor& other) const {
    return matmul(this, &other);
}
