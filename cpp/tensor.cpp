#include "tensor.h"
#include "autograd.h"

#include <cuda_runtime_api.h>

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kernels/tensor/add.h"
#include "kernels/tensor/batched_gemm.h"
#include "kernels/tensor/broadcast_add.h"
#include "kernels/tensor/broadcast_gemm.h"
#include "kernels/tensor/broadcast_subtract.h"
#include "kernels/tensor/gemm.h"
#include "kernels/tensor/scalar_multiply.h"
#include "kernels/tensor/subtract.h"
#include "kernels/tensor/sum_reduce.h"
#include "kernels/tensor/transpose.h"
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

Tensor::Tensor(const Tensor& other) : grad_fn(other.grad_fn), grad(other.grad), requires_grad(other.requires_grad), tensor_data(other.tensor_data), shape(new int[other.ndim]), ndim(other.ndim) {
    for (int d = 0; d < ndim; d++) {
        shape[d] = other.shape[d];
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        int* new_shape = new int[other.ndim];
        for (int d = 0; d < other.ndim; d++) {
            new_shape[d] = other.shape[d];
        }

        delete[] shape;
        grad_fn = other.grad_fn;
        grad = other.grad;
        requires_grad = other.requires_grad;
        tensor_data = other.tensor_data;
        shape = new_shape;
        ndim = other.ndim;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept : grad_fn(std::move(other.grad_fn)), grad(std::move(other.grad)), requires_grad(other.requires_grad), tensor_data(std::move(other.tensor_data)), shape(other.shape), ndim(other.ndim) {
    other.shape = nullptr;
    other.ndim = 0;
    other.requires_grad = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] shape;

        grad_fn = std::move(other.grad_fn);
        grad = std::move(other.grad);
        requires_grad = other.requires_grad;
        tensor_data = std::move(other.tensor_data);
        shape = other.shape;
        ndim = other.ndim;

        other.shape = nullptr;
        other.ndim = 0;
        other.requires_grad = false;
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
            if (a.requires_grad) {a.init_grad();}
            if (b.requires_grad) {b.init_grad();}
            out.requires_grad = true;
            out.init_grad();
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
        if (a.requires_grad) {a.init_grad();}
        if (b.requires_grad) {b.init_grad();}
        out.requires_grad = true;
        out.init_grad();
        out.grad_fn = std::make_shared<AddGradFn>(a, b);
    }

    return out;
}

Tensor Tensor::scalar_multiply(const Tensor& input, float scalar) {
    int* out_shape = new int[input.ndim];
    for (int d = 0; d < input.ndim; ++d) {
        out_shape[d] = input.shape[d];
    }

    Tensor out(input.tensor_data->size, out_shape, input.ndim);
    launch_scalar_multiply_kernel(
        input.tensor_data->data,
        scalar,
        out.tensor_data->data,
        input.tensor_data->size
    );
    return out;
}

Tensor Tensor::subtract(const Tensor& a, const Tensor& b) {
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
        launch_vec_subtract_kernel(
            a.tensor_data->data,
            b.tensor_data->data,
            out.tensor_data->data,
            a.tensor_data->size
        );
        return out;
    }

    const Tensor* batched = a.tensor_data->size > b.tensor_data->size ? &a : &b;
    const Tensor* broadcasted = a.tensor_data->size > b.tensor_data->size ? &b : &a;

    if (broadcasted->ndim >= batched->ndim || batched->tensor_data->size % broadcasted->tensor_data->size != 0) {
        throw std::invalid_argument(
            "Tensor shapes cannot be broadcast for subtraction"
        );
    }

    const int dim_offset = batched->ndim - broadcasted->ndim;
    for (int d = 0; d < broadcasted->ndim; ++d) {
        if (batched->shape[dim_offset + d] != broadcasted->shape[d]) {
            throw std::invalid_argument(
                "Tensor shapes cannot be broadcast for subtraction"
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
    launch_broadcast_vec_subtract_kernel(
        a.tensor_data->data,
        b.tensor_data->data,
        out.tensor_data->data,
        a.tensor_data->size,
        b.tensor_data->size,
        out.tensor_data->size
    );
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

Tensor Tensor::zeroes(int* shape, int ndim) {
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
        host_data[i] = 0;
    }
    return Tensor(host_data, size, shape, ndim);
}

Tensor Tensor::sum(int dim) const {
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Sum dimension is out of range");
    }

    int prefix_dim = 1;
    for (int d = 0; d < dim; ++d) {
        prefix_dim *= shape[d];
    }

    const int reduce_dim = shape[dim];

    int suffix_dim = 1;
    for (int d = dim + 1; d < ndim; ++d) {
        suffix_dim *= shape[d];
    }

    const int out_ndim = ndim - 1;
    int* out_shape = new int[out_ndim];
    for (int source_dim = 0, output_dim = 0; source_dim < ndim; ++source_dim) {
        if (source_dim != dim) {
            out_shape[output_dim++] = shape[source_dim];
        }
    }

    Tensor out(prefix_dim * suffix_dim, out_shape, out_ndim);
    launch_sum_reduce_kernel(
        tensor_data->data,
        out.tensor_data->data,
        prefix_dim,
        reduce_dim,
        suffix_dim
    );
    return out;
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
        if (a.requires_grad) {a.init_grad();}
        if (b.requires_grad) {b.init_grad();}
        out.requires_grad = true;
        out.init_grad();
        out.grad_fn = std::make_shared<MatMulGradFn>(a, b);
    }

    return out;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return subtract(*this, other);
}

Tensor Tensor::operator*(float scalar) const {
    return scalar_multiply(*this, scalar);
}

Tensor Tensor::matmul(const Tensor& other) const {
    return matmul(*this, other);
}

void Tensor::init_grad() const {
    if (grad != nullptr) {return;}

    int* grad_shape = new int[ndim];
    for (int d = 0; d < ndim; d++) {
        grad_shape[d] = shape[d];
    }
    grad = std::make_shared<Tensor>(Tensor::zeroes(grad_shape, ndim));
}

void Tensor::accumulate_grad(const Tensor& gradient) const {
    init_grad();
    Tensor detached_gradient = gradient;
    detached_gradient.requires_grad = false;
    detached_gradient.grad_fn = nullptr;
    *grad = *grad + detached_gradient;
}

void Tensor::backward() {
    if (!requires_grad) {return;}

    init_grad();
    int* grad_shape = new int[ndim];
    for (int d = 0; d < ndim; d++) {
        grad_shape[d] = shape[d];
    }
    Tensor start_grad = Tensor::ones(grad_shape, ndim);
    accumulate_grad(start_grad);

    std::queue<Tensor*> search_queue;
    std::unordered_set<TensorData*> visited;
    std::unordered_map<TensorData*, int> pending;
    std::unordered_map<TensorData*, Tensor*> tensors;
    search_queue.push(this);
    tensors[tensor_data.get()] = this;

    while (!search_queue.empty()) {
        Tensor* current = search_queue.front();
        search_queue.pop();

        if (!visited.insert(current->tensor_data.get()).second || current->grad_fn == nullptr) {
            continue;
        }

        for (Tensor& parent : current->grad_fn->parents) {
            if (!parent.requires_grad) {continue;}
            parent.init_grad();
            pending[parent.tensor_data.get()]++;
            tensors[parent.tensor_data.get()] = &parent;
            search_queue.push(&parent);
        }
    }

    std::queue<Tensor*> backward_queue;
    backward_queue.push(this);

    while (!backward_queue.empty()) {
        Tensor* current = backward_queue.front();
        backward_queue.pop();

        if (current->grad_fn == nullptr) {continue;}
        std::vector<Tensor> gradients = current->grad_fn->backward(*current->grad);

        for (int i = 0; i < current->grad_fn->parents.size(); i++) {
            Tensor& parent = current->grad_fn->parents[i];
            if (!parent.requires_grad) {continue;}

            parent.accumulate_grad(gradients[i]);
            pending[parent.tensor_data.get()]--;
            if (pending[parent.tensor_data.get()] == 0) {
                backward_queue.push(tensors[parent.tensor_data.get()]);
            }
        }
    }
}
