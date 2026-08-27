#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime_api.h>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <stdio.h>
#include <stdexcept>
#include <vector>
#include <random>

#include "utils.h"
#include "kernels/add.h"
#include "kernels/transpose.h"
#include "kernels/gemm.h"

namespace py = pybind11;

class Random {
private:
    inline static std::random_device rd;
    inline static std::mt19937 gen{rd()};

public:
    Random() = delete;

    static float get_float(float min, float max) {
        std::uniform_real_distribution<float> dis(min, max);
        return dis(gen);
    }

};

struct Tensor { // native CUDA memory
    Tensor(float* host_data, int size, int* shape, int ndim) : size(size), shape(shape), ndim(ndim) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), size*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(data, host_data, size*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFreeHost(host_data));
    };

    Tensor(int size, int* shape, int ndim) : size(size), shape(shape), ndim(ndim) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), size*sizeof(float)));
    };

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    ~Tensor() {
        if (data != nullptr) {
            cudaFree(data);
        }

        delete[] shape;
    }

    std::unique_ptr<Tensor> tranpose(int dim1, int dim2) const {
        int prefixDim = 1;
        if (0 < dim1) { for (int d=0; d<dim1; d++) { prefixDim *= this->shape[d]; } }
        int suffixDim = 1;
        if (dim2+1 < ndim) { for (int d=dim2+1; d<ndim; d++) { suffixDim *= this->shape[d]; } }

        int* newShape = new int[ndim];
        for (int d=0; d<ndim; d++) { newShape[d] = this->shape[d]; }
        int buf = newShape[dim1];
        newShape[dim1] = newShape[dim2];
        newShape[dim2] = buf;

        std::unique_ptr<Tensor> out = std::make_unique<Tensor>(
            this->size,
            newShape,
            this->ndim
        );

        launch_mat_transpose_kernel(this->data, out->data, prefixDim, this->shape[dim1], this->shape[dim2], suffixDim);

        return out; 
    }

    static std::unique_ptr<Tensor> add(const Tensor* a, const Tensor* b) {
        if (a->size != b->size || a->ndim != b->ndim) {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }

        int* out_shape = new int[a->ndim];
        for (int i = 0; i < a->ndim; ++i) {
            if (a->shape[i] != b->shape[i]) {
                delete[] out_shape;
                throw std::invalid_argument("Tensor shapes must match for addition");
            }
            out_shape[i] = a->shape[i];
        }

        auto out = std::make_unique<Tensor>(a->size, out_shape, a->ndim);
        launch_vec_add_kernel(a->data, b->data, out->data, a->size);
        return out;
    }

    static std::unique_ptr<Tensor> uniform(int* shape, int ndim, float min, float max) {
        int size = 1;
        for (int d=0; d<ndim; d++) { size *= shape[d]; }

        float* hostData = new float[size];

        for (int i=0; i<size; i++) {
            hostData[i] = Random::get_float(min, max);
        }

        std::unique_ptr<Tensor> out = std::make_unique<Tensor>(hostData, size, shape, ndim);

        return out;
    }


    static std::unique_ptr<Tensor> matmul(const Tensor* a, const Tensor* b) {
        if (a->shape[a->ndim-1] != b->shape[0]) {
            throw std::invalid_argument("Dimensions must line up for matmul");
        }

        int a_rows = 1; // collapse for matmul (or expand if a is 1 dim)
        if (a->ndim > 1) { for (int d=0; d < a->ndim-1; d++) { a_rows *= a->shape[d]; } }
        int b_cols = 1; // collapse for matmul (or expand if b is 1 dim)
        if (b->ndim > 1) { for (int d=1; d < b->ndim; d++) { b_cols *= b->shape[d]; } }

        int* newShape = new int[a->ndim-1 + b->ndim-1];
        for (int d=0; d < a->ndim-1; d++) {
            newShape[d] = a->shape[d];
        }
        for (int d=1; d < b->ndim; d++) {
            newShape[a->ndim-2+d] = b->shape[d];
        }

        std::unique_ptr<Tensor> out = std::make_unique<Tensor>(
            a_rows*b_cols,
            newShape,
            a->ndim-1 + b->ndim-1
        );

        launch_mat_mul_kernel(a->data, b->data, out->data, a_rows, b->shape[0], b_cols);

        return out;
    }

    std::unique_ptr<Tensor> operator+(const Tensor& other) const {
        return add(this, &other);
    }

    std::unique_ptr<Tensor> matmul(const Tensor& other) const {
        return matmul(this, &other);
    }

    float* data = nullptr;
    int size = 0;

    int* shape = nullptr;
    int ndim = 0;
};

int flatten_nested_list(const py::handle& obj, float* flat_data_list, int idx) {
    if (py::isinstance<py::list>(obj)) {
        for (const py::handle& item: obj) {
            idx = flatten_nested_list(item, flat_data_list, idx);
        }
        return idx;
    } else {
        float value = py::cast<float>(obj);
        flat_data_list[idx] = value;
        return idx+1;
    }
}

int get_ndim_nested_list(const py::handle& obj) {
    int n = 0;

    py::handle current = obj;

    while (py::isinstance<py::list>(current)) {
        py::list list = py::reinterpret_borrow<py::list>(current);
        if (list.empty()) { break; }
        current = list[0];
        n++;
    }
    return n;
}

int get_shape_nested_list(const py::handle& obj, int* shape) {
    int n = 0;
    int size = 1;
    py::handle current = obj;

    while (py::isinstance<py::list>(current)) {
        py::list list = py::reinterpret_borrow<py::list>(current);
        if (list.empty()) { break; }
        int length = static_cast<int>(py::len(list));

        shape[n] = length;
        size *= length;

        current = list[0];
        n++;
    }
    return size;
}

std::string format_tensor_data(const std::vector<float>& data, const int* shape,
                                int ndim, int dim, int& flat_index) {
    std::string out = "[";

    if (dim == ndim - 1) {
        for (int i = 0; i < shape[dim]; ++i) {
            if (i > 0) {
                out += ", ";
            }

            out += std::to_string(data[flat_index++]);
        }
    } else {
        for (int i = 0; i < shape[dim]; ++i) {
            if (i > 0) {
                out += ", ";
            }

            out += format_tensor_data(data, shape, ndim, dim + 1, flat_index);
        }
    }

    out += "]";
    return out;
}

/*
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9],
 [10,11,12]]
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 shape=[4,3]=len(out), len(out[0])

*/

PYBIND11_MODULE(candle, m, py::mod_gil_not_used()) {

    py::class_<Tensor>(m, "Tensor")
            .def(
                py::init([](py::list data_list) {
                    // need to traverse the list

                    int* shape = nullptr;

                    int ndim = get_ndim_nested_list(data_list);
                    shape = new int[ndim];

                    int size = get_shape_nested_list(data_list, shape);

                    float* host_data = nullptr;
                    CUDA_CHECK(cudaMallocHost(
                        reinterpret_cast<void**>(&host_data),
                        size*sizeof(float)
                    ));
                    
                    flatten_nested_list(data_list, host_data, 0);

                    return std::make_unique<Tensor>(
                        host_data,
                        size,
                        shape,
                        ndim
                    );

                }),
                py::arg("data_list")
            )

            .def_static(
                "uniform",
                [](const std::vector<int>& shape, float min, float max) {
                    int ndim = static_cast<int>(shape.size());

                    int* shape_copy = new int[ndim];
                    for (int i = 0; i < ndim; ++i) {
                        shape_copy[i] = shape[i];
                    }

                    return Tensor::uniform(shape_copy, ndim, min, max);
                },
                py::arg("shape"),
                py::arg("min") = 0.0f,
                py::arg("max") = 1.0f
            )

            .def("numpy", [](const Tensor& tensor) {
                    std::vector<py::ssize_t> shape(tensor.shape, tensor.shape + tensor.ndim);
                    py::array_t<float> out(shape);

                    CUDA_CHECK(cudaMemcpy(
                        out.mutable_data(),
                        tensor.data,
                        tensor.size * sizeof(float),
                        cudaMemcpyDeviceToHost
                    ));

                    return out;
                })

            .def_property_readonly(
                "shape",
                [](const Tensor& tensor) {
                    py::list result;

                    for (int i = 0; i < tensor.ndim; i++) {
                        result.append(tensor.shape[i]);
                    }

                    return result;
                }
            )

            .def(py::self + py::self) // + operator

            .def(
                "transpose",
                &Tensor::tranpose,
                py::arg("dim1"),
                py::arg("dim2")
            )

            .def(
                "matmul",
                py::overload_cast<const Tensor&>(&Tensor::matmul, py::const_),
                py::arg("other")
            )
            
            .def("__repr__", [](const Tensor& tensor) {
                std::vector<float> host_data(tensor.size);

                CUDA_CHECK(cudaMemcpy(
                    host_data.data(),
                    tensor.data,
                    tensor.size * sizeof(float),
                    cudaMemcpyDeviceToHost
                ));

                std::string data_string;

                if (tensor.ndim == 0) {
                    data_string = tensor.size > 0
                        ? std::to_string(host_data[0])
                        : "[]";
                } else {
                    int flat_index = 0;

                    data_string = format_tensor_data(
                        host_data,
                        tensor.shape,
                        tensor.ndim,
                        0,
                        flat_index
                    );
                }

                std::string shape_string = "(";

                for (int i = 0; i < tensor.ndim; ++i) {
                    if (i > 0) {
                        shape_string += ", ";
                    }

                    shape_string += std::to_string(tensor.shape[i]);
                }

                if (tensor.ndim == 1) {
                    shape_string += ",";
                }

                shape_string += ")";

                return "<Tensor shape=" + shape_string +
                    " data=" + data_string + ">";
            });
}
