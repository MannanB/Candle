#pragma once

#include <memory>



struct Tensor {
    Tensor(float* host_data, int size, int* shape, int ndim);
    Tensor(int size, int* shape, int ndim);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    ~Tensor();

    std::unique_ptr<Tensor> transpose(int dim1, int dim2) const;
    std::unique_ptr<Tensor> sum(int dim) const;

    static std::unique_ptr<Tensor> add(const Tensor* a, const Tensor* b);
    static std::unique_ptr<Tensor> uniform(
        int* shape,
        int ndim,
        float min,
        float max
    );
    static std::unique_ptr<Tensor> matmul(const Tensor* a, const Tensor* b);

    std::unique_ptr<Tensor> operator+(const Tensor& other) const;
    std::unique_ptr<Tensor> matmul(const Tensor& other) const;

    float* data = nullptr;
    int size = 0;
    int* shape = nullptr;
    int ndim = 0;
};
