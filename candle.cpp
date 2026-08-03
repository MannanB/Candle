#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>

#include "utils.h"
#include "add.h"

namespace py = pybind11;


struct Tensor { // native CUDA memory
    Tensor(float* host_data, int size, int* shape, int ndim) : size(size), shape(shape), ndim(ndim) {
        CUDA_CHECK(cudaMalloc(&data, size*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(data, host_data, size*sizeof(float), cudaMemcpyDefault));
        CUDA_CHECK(cudaFreeHost(host_data));
    };

    Tensor(int size, int* shape, int ndim) : size(size), shape(shape), ndim(ndim) {
        CUDA_CHECK(cudaMalloc(&data, size*sizeof(float)));
    };
    
    ~Tensor() {
        if (data != nullptr) {
            cudaFree(data);
        }

        delete[] shape;
    }

    static Tensor* add(Tensor* a, Tensor* b) {
        Tensor* out = std::make_unique<Tensor>(a->size, a->shape, a->ndim);
        launch_vec_add_kernel(a->data, b->data, out->data, a->size);
        return out;
    }

    Tensor operator+(const Tensor& other) const {
        return *add(&this, &other);
    }

    float* data = nullptr;
    int size = 0;

    int* shape = nullptr;
    int ndim = 0;
};


PYBIND11_MODULE(candle, m, py::mod_gil_not_used()) {

    py::class_<Tensor>(m, "Tensor")
            .def(
                py::init([](py::list data_list, py::list shape_list) {
                    int size = static_cast<int>(data_list.size());
                    int ndim = static_cast<int>(shape_list.size());

                    float* host_data = nullptr;
                    cudaMallocHost(&host_data, size*sizeof(float));
                    
                    for (int i=0; i<size; i++) {
                        host_data[i] = data_list[i].cast<float>();
                    }

                    int* shape = new int[ndim];
                    for (int i=0; i<ndim; i++) {
                        shape[i] = shape_list[i].cast<int>();
                    }

                    return std::make_unique<Tensor>(
                        host_data,
                        size,
                        shape,
                        ndim
                    );

                }),
                py::arg("data_list"),
                py::arg("shape_list")
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

            .def(py::self + py::self); // + operator
}