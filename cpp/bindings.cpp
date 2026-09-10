#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

#include "tensor.h"
#include "utils.h"

namespace py = pybind11;

namespace {

int flatten_nested_list(const py::handle& object, float* output, int index) {
    if (py::isinstance<py::list>(object)) {
        for (const py::handle& item : object) {
            index = flatten_nested_list(item, output, index);
        }
        return index;
    }

    output[index] = py::cast<float>(object);
    return index + 1;
}

int get_nested_list_ndim(const py::handle& object) {
    int ndim = 0;
    py::handle current = object;

    while (py::isinstance<py::list>(current)) {
        const py::list list = py::reinterpret_borrow<py::list>(current);
        if (list.empty()) {
            break;
        }
        current = list[0];
        ++ndim;
    }

    return ndim;
}

int get_nested_list_shape(const py::handle& object, int* shape) {
    int dimension = 0;
    int size = 1;
    py::handle current = object;

    while (py::isinstance<py::list>(current)) {
        const py::list list = py::reinterpret_borrow<py::list>(current);
        if (list.empty()) {
            break;
        }

        const int length = static_cast<int>(py::len(list));
        shape[dimension] = length;
        size *= length;
        current = list[0];
        ++dimension;
    }

    return size;
}

std::string format_tensor_data(
    const std::vector<float>& data,
    const int* shape,
    int ndim,
    int dimension,
    int& flat_index
) {
    std::string output = "[";

    if (dimension == ndim - 1) {
        for (int i = 0; i < shape[dimension]; ++i) {
            if (i > 0) {
                output += ", ";
            }
            output += std::to_string(data[flat_index++]);
        }
    } else {
        for (int i = 0; i < shape[dimension]; ++i) {
            if (i > 0) {
                output += ", ";
            }
            output += format_tensor_data(
                data,
                shape,
                ndim,
                dimension + 1,
                flat_index
            );
        }
    }

    output += "]";
    return output;
}

Tensor tensor_from_list(const py::list& data_list) {
    const int ndim = get_nested_list_ndim(data_list);
    int* shape = new int[ndim];
    const int size = get_nested_list_shape(data_list, shape);

    float* host_data = nullptr;
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&host_data),
        size * sizeof(float)
    ));
    flatten_nested_list(data_list, host_data, 0);

    return Tensor(host_data, size, shape, ndim);
}

Tensor uniform_tensor(
    const std::vector<int>& shape,
    float min,
    float max
) {
    const int ndim = static_cast<int>(shape.size());
    int* shape_copy = new int[ndim];
    for (int i = 0; i < ndim; ++i) {
        shape_copy[i] = shape[i];
    }
    return Tensor::uniform(shape_copy, ndim, min, max);
}

py::array_t<float> tensor_to_numpy(const Tensor& tensor) {
    const std::vector<py::ssize_t> shape(
        tensor.shape,
        tensor.shape + tensor.ndim
    );
    py::array_t<float> output(shape);
    CUDA_CHECK(cudaMemcpy(
        output.mutable_data(),
        tensor.tensor_data->data,
        tensor.tensor_data->size * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    return output;
}

py::list tensor_shape(const Tensor& tensor) {
    py::list result;
    for (int i = 0; i < tensor.ndim; ++i) {
        result.append(tensor.shape[i]);
    }
    return result;
}

std::string tensor_repr(const Tensor& tensor) {
    std::vector<float> host_data(tensor.tensor_data->size);
    CUDA_CHECK(cudaMemcpy(
        host_data.data(),
        tensor.tensor_data->data,
        tensor.tensor_data->size * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    std::string data_string;
    if (tensor.ndim == 0) {
        data_string = tensor.tensor_data->size > 0 ? std::to_string(host_data[0]) : "[]";
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

    return "<Tensor shape=" + shape_string + " data=" + data_string + ">";
}

}  // namespace

PYBIND11_MODULE(_candle, module, py::mod_gil_not_used()) {
    py::class_<Tensor>(module, "Tensor")
        .def(py::init(&tensor_from_list), py::arg("data_list"))
        .def_static(
            "uniform",
            &uniform_tensor,
            py::arg("shape"),
            py::arg("min") = 0.0f,
            py::arg("max") = 1.0f
        )
        .def("numpy", &tensor_to_numpy)
        .def_property_readonly("shape", &tensor_shape)
        .def(py::self + py::self)
        .def(
            "transpose",
            &Tensor::transpose,
            py::arg("dim1"),
            py::arg("dim2")
        )
        .def(
            "matmul",
            py::overload_cast<const Tensor&>(&Tensor::matmul, py::const_),
            py::arg("other")
        )
        .def("__repr__", &tensor_repr);
}
