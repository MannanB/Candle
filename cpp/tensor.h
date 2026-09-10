#pragma once

struct Tensor {
    Tensor() = default;
    Tensor(float* host_data, int size, int* shape, int ndim);
    Tensor(int size, int* shape, int ndim);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    Tensor transpose(int dim1, int dim2) const;
    Tensor sum(int dim) const;

    static Tensor add(const Tensor* a, const Tensor* b);
    static Tensor uniform(
        int* shape,
        int ndim,
        float min,
        float max
    );
    static Tensor matmul(const Tensor* a, const Tensor* b);

    Tensor operator+(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    float* data = nullptr;
    int size = 0;
    int* shape = nullptr;
    int ndim = 0;
};
