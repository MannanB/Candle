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

#include "utils.h"
#include "kernels/add.h"

namespace py = pybind11;


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

    std::unique_ptr<Tensor> operator+(const Tensor& other) const {
        return add(this, &other);
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
                    shape = (int*)malloc(ndim * sizeof(int));

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
