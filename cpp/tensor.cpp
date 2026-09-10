#include "tensor.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <utility>

#include "kernels/add.h"
#include "kernels/batched_gemm.h"
#include "kernels/broadcast_add.h"
#include "kernels/broadcast_gemm.h"
#include "kernels/gemm.h"
#include "kernels/transpose.h"
#include "utils.h"

TensorData::~TensorData() {
    if (data != nullptr) {
        cudaFree(data);
    }
}

Tensor::Tensor(float* host_data, int size, int* shape, int ndim) : shape(shape), ndim(ndim) {
    tensor_data = std::make_shared<TensorData>();
    tensor_data->size = size;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tensor_data->data), size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        tensor_data->data,
        host_data,
        tensor_data->size * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaFreeHost(host_data));
}

Tensor::Tensor(int size, int* shape, int ndim) : shape(shape), ndim(ndim) {
    tensor_data = std::make_shared<TensorData>();
    tensor_data->size = size;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tensor_data->data), size * sizeof(float)));
}

Tensor::Tensor(std::shared_ptr<TensorData> tensor_data, int* shape, int ndim) : tensor_data(tensor_data), shape(shape), ndim(ndim) {

}

Tensor::Tensor(Tensor&& other) noexcept : tensor_data(std::move(other.tensor_data)), shape(other.shape), ndim(other.ndim), grad_graph(other.grad_graph) {
    other.shape = nullptr;
    other.ndim = 0;
    other.grad_graph = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] shape;

        tensor_data = std::move(other.tensor_data);
        shape = other.shape;
        ndim = other.ndim;
        grad_graph = other.grad_graph;

        other.shape = nullptr;
        other.ndim = 0;
        other.grad_graph = nullptr;
    }
    return *this;
}

Tensor::~Tensor() {
    delete[] shape;
}

Tensor Tensor::transpose(int dim1, int dim2) const {
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

    Tensor out(tensor_data->size, new_shape, ndim);
    launch_mat_transpose_kernel(
        tensor_data->data,
        out.tensor_data->data,
        prefix_dim,
        shape[dim1],
        shape[dim2],
        suffix_dim
    );
    return out;
}

Tensor Tensor::add(const Tensor& a, const Tensor& b) {
    bool same_shape = a.ndim == b.ndim;
    if (same_shape) {
        for (int d = 0; d < a.ndim; ++d) {
            if (a.shape[d] != b.shape[d]) {
                same_shape = false;
                break;
            }
        }
    }

    if (same_shape) {
        int* out_shape = new int[a.ndim];
        for (int d = 0; d < a.ndim; ++d) {
            out_shape[d] = a.shape[d];
        }

        Tensor out(a.tensor_data->size, out_shape, a.ndim);
        launch_vec_add_kernel(a.tensor_data->data, b.tensor_data->data, out.tensor_data->data, a.tensor_data->size);
        
        if (a.requires_grad || b.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = std::make_shared<AddGradFn>(a, b);
        }
        
        return out;
    }

    const Tensor* batched = a.tensor_data->size > b.tensor_data->size ? &a : &b;
    const Tensor* broadcasted = a.tensor_data->size > b.tensor_data->size ? &b : &a;

    if (broadcasted->ndim >= batched->ndim || batched->tensor_data->size % broadcasted->tensor_data->size != 0) {
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

    Tensor out(
        batched->tensor_data->size,
        out_shape,
        batched->ndim
    );
    launch_broadcast_vec_add_kernel(
        batched->tensor_data->data,
        broadcasted->tensor_data->data,
        out.tensor_data->data,
        broadcasted->tensor_data->size,
        batched->tensor_data->size / broadcasted->tensor_data->size
    );

    if (a.requires_grad || b.requires_grad) {
        out.requires_grad = true;
        out.grad_fn = std::make_shared<AddGradFn>(a, b);
    }

    return out;
}

Tensor Tensor::uniform(int* shape, int ndim, float min, float max) {
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
    return Tensor(host_data, size, shape, ndim);
}

Tensor Tensor::ones(int* shape, int ndim) {
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
        host_data[i] = 1;
    }
    return Tensor(host_data, size, shape, ndim);
}

Tensor Tensor::sum(int dim) const {
    // sum across dim
    // TODO: implement
    throw std::logic_error("Tensor::sum is not implemented");
}


Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim < 2 || b.ndim < 1) {
        throw std::invalid_argument("Matmul requires a matrix on the left");
    }

    const int a_rows = a.shape[a.ndim - 2];
    const int a_cols = a.shape[a.ndim - 1];
    int b_rows = 1;
    int b_cols = 1;
    int batch_size = 1;
    int out_ndim = 2;
    int* new_shape = nullptr;
    const bool batched = a.ndim > 2;
    bool broadcast = false;

    if (batched) {
        if (b.ndim == a.ndim) {
            b_rows = b.shape[b.ndim - 2];
            b_cols = b.shape[b.ndim - 1];

            for (int d = 0; d < a.ndim - 2; ++d) {
                if (a.shape[d] != b.shape[d]) {
                    throw std::invalid_argument("Batch dimensions must match");
                }
                batch_size *= a.shape[d];
            }
        } else if (b.ndim == a.ndim - 1) {
            b_rows = b.shape[b.ndim - 1];

            for (int d = 0; d < b.ndim - 1; ++d) {
                if (a.shape[d] != b.shape[d]) {
                    throw std::invalid_argument("Batch dimensions must match");
                }
                batch_size *= a.shape[d];
            }
        } else {
            throw std::invalid_argument("Unsupported batched matmul dimensions");
        }

        out_ndim = a.ndim;
        new_shape = new int[out_ndim];
        for (int d = 0; d < out_ndim - 2; ++d) {
            new_shape[d] = a.shape[d];
        }
        new_shape[out_ndim - 2] = a_rows;
        new_shape[out_ndim - 1] = b_cols;
    } else if (b.ndim == 1) {
        b_rows = b.shape[0];
        new_shape = new int[2]{a_rows, 1};
    } else if (a_cols == b.shape[b.ndim - 2]) {
        b_rows = b.shape[b.ndim - 2];
        b_cols = b.shape[b.ndim - 1];
        broadcast = b.ndim > 2;
        out_ndim = b.ndim;
        new_shape = new int[out_ndim];

        for (int d = 0; d < b.ndim - 2; ++d) {
            new_shape[d] = b.shape[d];
            batch_size *= b.shape[d];
        }
        new_shape[out_ndim - 2] = a_rows;
        new_shape[out_ndim - 1] = b_cols;
    } else if (a_cols == b.shape[b.ndim - 1]) {
        b_rows = b.shape[b.ndim - 1];
        broadcast = true;
        out_ndim = b.ndim + 1;
        new_shape = new int[out_ndim];

        for (int d = 0; d < b.ndim - 1; ++d) {
            new_shape[d] = b.shape[d];
            batch_size *= b.shape[d];
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

    Tensor out(
        batch_size * a_rows * b_cols,
        new_shape,
        out_ndim
    );

    if (batched) {
        launch_batched_mat_mul_kernel(
            a.tensor_data->data, b.tensor_data->data, out.tensor_data->data,
            batch_size, a_rows, a_cols,  b_cols
        );
    } else if (broadcast) {
        launch_broadcast_mat_mul_kernel(
            a.tensor_data->data, b.tensor_data->data, out.tensor_data->data,
            batch_size, a_rows, a_cols, b_cols
        );
    } else {
        launch_mat_mul_kernel(
            a.tensor_data->data, b.tensor_data->data, out.tensor_data->data,
            a_rows, a_cols, b_cols
        );
    }

    if (a.requires_grad || b.requires_grad) {
        out.requires_grad = true;
        out.grad_fn = std::make_shared<MatMulGradFn>(a, b);
    }

    return out;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return add(*this, other);
}

Tensor Tensor::matmul(const Tensor& other) const {
    return matmul(*this, other);
}

void Tensor::backward() {
    if (grad_fn == nullptr || !requires_grad) {return;}

    if (grad == nullptr) {
        Tensor start_grad = Tensor::ones(this->shape, this->ndim);
        grad = std::make_shared<Tensor>(std::move(start_grad));
    }

    std::vector<Tensor> grads = grad_fn->backward(*grad);

    // DFS backward; probably wont work for all kinds of graphs
    for (int i=0; i<grad_fn->parents.size(); i++) {
        grad_fn->parents[i].grad = std::make_shared<Tensor>(std::move(grads[i]));
        grad_fn->parents[i].backward();
    }
}