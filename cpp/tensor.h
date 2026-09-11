#pragma once
#include <memory>
#include <vector>

struct GradFn;
struct Tensor;

struct Parameter {
    Tensor* parameter;
    Tensor* gradient;
};



struct TensorData {
    float* data = nullptr;
    int size = 0;

    ~TensorData();
};

struct Tensor {
    Tensor() = default;
    Tensor(float* host_data, int size, int* shape, int ndim);
    Tensor(int size, int* shape, int ndim);
    Tensor(std::shared_ptr<TensorData> tensor_data, int* shape, int ndim);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    Tensor transpose(int dim1, int dim2) const;
    Tensor sum(int dim) const;
    Tensor flatten() const;
    Tensor flatten(int dim1, int dim2) const;
    Tensor unsqueeze() const;

    static Tensor add(const Tensor& a, const Tensor& b, float A_factor = 1.0f, float B_factor = 1.0f);
    static Tensor subtract(const Tensor& a, const Tensor& b);
    static Tensor scalar_multiply(const Tensor& input, float scalar);
    static Tensor uniform(int* shape, int ndim, float min, float max);
    static Tensor ones(int* shape, int ndim);
    static Tensor zeroes(int* shape, int ndim);

    static Tensor matmul(const Tensor& a, const Tensor& b);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    Tensor matmul(const Tensor& other) const;
    void inplace_add(const Tensor& other, float self_factor = 1.0f, float other_factor = 1.0f);

    void backward();
    // void backward(std::shared_ptr<TensorData> inp_grad);
    void init_grad() const;
    void zero_grad() const;
    void accumulate_grad(const Tensor& gradient) const;

    // stuff for autograd
    std::shared_ptr<GradFn> grad_fn;
    mutable std::shared_ptr<Tensor> grad;
    bool requires_grad = false;

    // so basically data is the only think we pass by ref
    // that way we can do in place ops and stuff without getting into some wierd ownership stuff
    std::shared_ptr<TensorData> tensor_data;
    int* shape = nullptr;
    int ndim = 0;
};
