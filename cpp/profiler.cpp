#include <cuda_runtime_api.h>

#include <chrono>
#include <climits>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kernels/activation/relu.h"
#include "kernels/activation/relu_bwd.h"
#include "kernels/activation/sigmoid.h"
#include "kernels/activation/sigmoid_bwd.h"
#include "kernels/activation/softmax.h"
#include "kernels/loss/mse.h"
#include "kernels/loss/mse_bwd.h"
#include "kernels/tensor/add.h"
#include "kernels/tensor/batched_gemm.h"
#include "kernels/tensor/broadcast_add.h"
#include "kernels/tensor/broadcast_gemm.h"
#include "kernels/tensor/gemm.h"
#include "kernels/tensor/gemm_2d_tile.h"
#include "kernels/tensor/gemm_naive.h"
#include "kernels/tensor/gemm_smem.h"
#include "kernels/tensor/gemm_vectorized.h"
#include "kernels/tensor/scalar_multiply.h"
#include "kernels/tensor/sum_reduce.h"
#include "kernels/tensor/transpose.h"

namespace {

constexpr const char* kKernelNames[] = {
    "add",
    "broadcast-add",
    "scalar-multiply",
    "relu",
    "relu-bwd",
    "sigmoid",
    "sigmoid-bwd",
    "softmax",
    "mse",
    "mse-bwd",
    "transpose",
    "sum-reduce",
    "gemm",
    "gemm_vectorized",
    "gemm_2d_tile",
    "gemm_smem",
    "gemm_naive",
    "batched-gemm",
    "broadcast-gemm",
};

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

std::size_t checked_product(std::initializer_list<std::size_t> factors) {
    std::size_t result = 1;
    for (const std::size_t factor : factors) {
        if (factor != 0 && result > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::runtime_error("tensor size overflow");
        }
        result *= factor;
    }
    return result;
}

class DeviceAllocations {
public:
    DeviceAllocations() = default;
    DeviceAllocations(const DeviceAllocations&) = delete;
    DeviceAllocations& operator=(const DeviceAllocations&) = delete;
    DeviceAllocations(DeviceAllocations&&) noexcept = default;
    DeviceAllocations& operator=(DeviceAllocations&&) noexcept = default;

    ~DeviceAllocations() {
        for (float* pointer : pointers_) {
            cudaFree(pointer);
        }
    }

    float* allocate(std::size_t element_count, bool initialize = true) {
        if (element_count == 0 ||
            element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error("invalid device allocation size");
        }

        float* pointer = nullptr;
        const std::size_t bytes = element_count * sizeof(float);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes), "cudaMalloc");
        pointers_.push_back(pointer);
        check_cuda(cudaMemset(pointer, initialize ? 0x3f : 0, bytes), "cudaMemset");
        return pointer;
    }

private:
    std::vector<float*> pointers_;
};

struct Options {
    std::string kernel;
    int size = 1 << 20;
    int rows = 1024;
    int inner = 1024;
    int cols = 1024;
    int batch = 8;
    int prefix = 1;
    int reduce = 1024;
    int suffix = 1;
    int a_size = 1;
    int b_size = 1024;
    int warmup = 0;
    int iterations = 1;
    bool list = false;
    bool help = false;
};

void print_usage(const char* executable) {
    std::cout
        << "Profile one Candle CUDA kernel in a standalone process.\n\n"
        << "Usage:\n"
        << "  " << executable << " --kernel NAME [options]\n"
        << "  " << executable << " --list\n\n"
        << "Common options:\n"
        << "  --kernel NAME       Kernel to launch (a positional NAME also works)\n"
        << "  --size N            Element count for elementwise kernels (default: 1048576)\n"
        << "  --warmup N          Unmeasured launches before the run (default: 0)\n"
        << "  --iterations N      Measured launches (default: 1)\n\n"
        << "Matrix options:\n"
        << "  --rows N            GEMM/transpose row count (default: 1024)\n"
        << "  --inner N           GEMM shared dimension (default: 1024)\n"
        << "  --cols N            GEMM/transpose column count (default: 1024)\n"
        << "  --batch N           Batched GEMM batch count (default: 8)\n\n"
        << "Reduction options:\n"
        << "  --prefix N          Outer dimension (default: 1)\n"
        << "  --reduce N          Reduction dimension (default: 1024)\n"
        << "  --suffix N          Inner dimension (default: 1)\n\n"
        << "Broadcast-add options:\n"
        << "  --a-size N          Repeating A length (default: 1)\n"
        << "  --b-size N          Repeating B length (default: 1024)\n"
        << "  --size N            Output length\n";
}

void print_kernel_list() {
    for (const char* name : kKernelNames) {
        std::cout << name << '\n';
    }
}

int parse_positive_int(const std::string& text, const std::string& option, bool allow_zero) {
    std::size_t consumed = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(option + " expects an integer, got '" + text + "'");
    }

    const long long minimum = allow_zero ? 0 : 1;
    if (consumed != text.size() || value < minimum || value > INT_MAX) {
        throw std::runtime_error(option + " is out of range: '" + text + "'");
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    const std::unordered_set<std::string> value_options = {
        "--kernel", "--size", "--rows", "--inner", "--cols", "--batch",
        "--prefix", "--reduce", "--suffix", "--a-size", "--b-size",
        "--warmup", "--iterations",
    };

    for (int index = 1; index < argc; ++index) {
        std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            options.help = true;
            continue;
        }
        if (argument == "--list") {
            options.list = true;
            continue;
        }

        std::string name = argument;
        std::string value;
        const std::size_t equals = argument.find('=');
        if (equals != std::string::npos) {
            name = argument.substr(0, equals);
            value = argument.substr(equals + 1);
        }

        if (value_options.count(name) != 0) {
            if (value.empty()) {
                if (++index >= argc) {
                    throw std::runtime_error("missing value for " + name);
                }
                value = argv[index];
            }

            if (name == "--kernel") options.kernel = value;
            else if (name == "--size") options.size = parse_positive_int(value, name, false);
            else if (name == "--rows") options.rows = parse_positive_int(value, name, false);
            else if (name == "--inner") options.inner = parse_positive_int(value, name, false);
            else if (name == "--cols") options.cols = parse_positive_int(value, name, false);
            else if (name == "--batch") options.batch = parse_positive_int(value, name, false);
            else if (name == "--prefix") options.prefix = parse_positive_int(value, name, false);
            else if (name == "--reduce") options.reduce = parse_positive_int(value, name, false);
            else if (name == "--suffix") options.suffix = parse_positive_int(value, name, false);
            else if (name == "--a-size") options.a_size = parse_positive_int(value, name, false);
            else if (name == "--b-size") options.b_size = parse_positive_int(value, name, false);
            else if (name == "--warmup") options.warmup = parse_positive_int(value, name, true);
            else if (name == "--iterations") options.iterations = parse_positive_int(value, name, false);
            continue;
        }

        if (!argument.empty() && argument[0] == '-') {
            throw std::runtime_error("unknown option: " + argument);
        }
        if (!options.kernel.empty()) {
            throw std::runtime_error("only one kernel may be selected");
        }
        options.kernel = argument;
    }
    return options;
}

struct PreparedKernel {
    DeviceAllocations allocations;
    std::function<void()> launch;
    std::string shape;
};

PreparedKernel prepare_kernel(const Options& options) {
    PreparedKernel prepared;
    const std::string& name = options.kernel;

    if (name == "add") {
        float* a = prepared.allocations.allocate(options.size);
        float* b = prepared.allocations.allocate(options.size);
        float* output = prepared.allocations.allocate(options.size, false);
        prepared.launch = [=] { launch_vec_add_kernel(a, b, output, 1.0f, 1.0f, options.size); };
        prepared.shape = "elements=" + std::to_string(options.size);
    } else if (name == "broadcast-add") {
        if (options.size % options.a_size != 0 || options.size % options.b_size != 0) {
            throw std::runtime_error("--a-size and --b-size must divide --size for broadcast-add");
        }
        float* a = prepared.allocations.allocate(options.a_size);
        float* b = prepared.allocations.allocate(options.b_size);
        float* output = prepared.allocations.allocate(options.size, false);
        prepared.launch = [=] {
            launch_broadcast_vec_add_kernel(
                a, b, output, 1.0f, 1.0f, options.a_size, options.b_size, options.size);
        };
        prepared.shape = "a=" + std::to_string(options.a_size) + ", b=" +
                         std::to_string(options.b_size) + ", output=" +
                         std::to_string(options.size);
    } else if (name == "scalar-multiply") {
        float* input = prepared.allocations.allocate(options.size);
        float* output = prepared.allocations.allocate(options.size, false);
        prepared.launch = [=] { launch_scalar_multiply_kernel(input, 2.0f, output, options.size); };
        prepared.shape = "elements=" + std::to_string(options.size);
    } else if (name == "relu") {
        float* input = prepared.allocations.allocate(options.size);
        float* output = prepared.allocations.allocate(options.size, false);
        prepared.launch = [=] { launch_relu_kernel(input, output, options.size); };
        prepared.shape = "elements=" + std::to_string(options.size);
    } else if (name == "relu-bwd") {
        float* input = prepared.allocations.allocate(options.size);
        float* output_gradient = prepared.allocations.allocate(options.size);
        float* input_gradient = prepared.allocations.allocate(options.size, false);
        prepared.launch = [=] {
            launch_relu_bwd_kernel(input, output_gradient, input_gradient, options.size);
        };
        prepared.shape = "elements=" + std::to_string(options.size);
    } else if (name == "sigmoid") {
        float* input = prepared.allocations.allocate(options.size);
        float* output = prepared.allocations.allocate(options.size, false);
        prepared.launch = [=] { launch_sigmoid_kernel(input, output, options.size); };
        prepared.shape = "elements=" + std::to_string(options.size);
    } else if (name == "sigmoid-bwd") {
        float* input = prepared.allocations.allocate(options.size);
        float* output_gradient = prepared.allocations.allocate(options.size);
        float* input_gradient = prepared.allocations.allocate(options.size, false);
        prepared.launch = [=] {
            launch_sigmoid_bwd_kernel(input, output_gradient, input_gradient, options.size);
        };
        prepared.shape = "elements=" + std::to_string(options.size);
    } else if (name == "mse") {
        float* predicted = prepared.allocations.allocate(options.size);
        float* actual = prepared.allocations.allocate(options.size);
        float* output = prepared.allocations.allocate(1, false);
        prepared.launch = [=] { launch_mse_kernel(predicted, actual, output, options.size); };
        prepared.shape = "elements=" + std::to_string(options.size);
    } else if (name == "mse-bwd") {
        float* predicted = prepared.allocations.allocate(options.size);
        float* actual = prepared.allocations.allocate(options.size);
        float* output_gradient = prepared.allocations.allocate(1);
        float* predicted_gradient = prepared.allocations.allocate(options.size, false);
        float* actual_gradient = prepared.allocations.allocate(options.size, false);
        prepared.launch = [=] {
            launch_mse_bwd_kernel(
                predicted, actual, output_gradient, predicted_gradient, actual_gradient, options.size);
        };
        prepared.shape = "elements=" + std::to_string(options.size);
    } else if (name == "softmax") {
        const std::size_t input_size = checked_product(
            {static_cast<std::size_t>(options.prefix), static_cast<std::size_t>(options.reduce),
             static_cast<std::size_t>(options.suffix)});
        float* input = prepared.allocations.allocate(input_size);
        float* output = prepared.allocations.allocate(input_size, false);
        prepared.launch = [=] {
            launch_softmax_kernel(input, output, options.prefix, options.reduce, options.suffix);
        };
        prepared.shape = "prefix=" + std::to_string(options.prefix) + ", reduce=" +
                         std::to_string(options.reduce) + ", suffix=" +
                         std::to_string(options.suffix);
    } else if (name == "sum-reduce") {
        const std::size_t input_size = checked_product(
            {static_cast<std::size_t>(options.prefix), static_cast<std::size_t>(options.reduce),
             static_cast<std::size_t>(options.suffix)});
        const std::size_t output_size = checked_product(
            {static_cast<std::size_t>(options.prefix), static_cast<std::size_t>(options.suffix)});
        float* input = prepared.allocations.allocate(input_size);
        float* output = prepared.allocations.allocate(output_size, false);
        prepared.launch = [=] {
            launch_sum_reduce_kernel(input, output, options.prefix, options.reduce, options.suffix);
        };
        prepared.shape = "prefix=" + std::to_string(options.prefix) + ", reduce=" +
                         std::to_string(options.reduce) + ", suffix=" +
                         std::to_string(options.suffix);
    } else if (name == "transpose") {
        const std::size_t element_count = checked_product(
            {static_cast<std::size_t>(options.prefix), static_cast<std::size_t>(options.rows),
             static_cast<std::size_t>(options.cols), static_cast<std::size_t>(options.suffix)});
        float* input = prepared.allocations.allocate(element_count);
        float* output = prepared.allocations.allocate(element_count, false);
        prepared.launch = [=] {
            launch_mat_transpose_kernel(
                input, output, options.prefix, options.rows, options.cols, options.suffix);
        };
        prepared.shape = "prefix=" + std::to_string(options.prefix) + ", rows=" +
                         std::to_string(options.rows) + ", cols=" +
                         std::to_string(options.cols) + ", suffix=" +
                         std::to_string(options.suffix);
    } else if (name == "gemm" || name == "gemm_vectorized" || name == "gemm_2d_tile" || name == "gemm_smem" || name == "gemm_naive" ||
               name == "gemm-smem" || name == "gemm-naive") {
        const std::size_t a_count = checked_product(
            {static_cast<std::size_t>(options.rows), static_cast<std::size_t>(options.inner)});
        const std::size_t b_count = checked_product(
            {static_cast<std::size_t>(options.inner), static_cast<std::size_t>(options.cols)});
        const std::size_t output_count = checked_product(
            {static_cast<std::size_t>(options.rows), static_cast<std::size_t>(options.cols)});
        float* a = prepared.allocations.allocate(a_count);
        float* b = prepared.allocations.allocate(b_count);
        float* output = prepared.allocations.allocate(output_count, false);
        if (name == "gemm_vectorized") {
            prepared.launch = [=] {
                launch_mat_mul_vectorized_kernel(a, b, output, options.rows, options.inner, options.cols);
            };
        } else if (name == "gemm_2d_tile") {
            prepared.launch = [=] {
                launch_mat_mul_2d_tile_kernel(a, b, output, options.rows, options.inner, options.cols);
            };
        } else if (name == "gemm_smem" || name == "gemm-smem") {
            prepared.launch = [=] {
                launch_mat_mul_smem_kernel(a, b, output, options.rows, options.inner, options.cols);
            };
        } else if (name == "gemm_naive" || name == "gemm-naive") {
            prepared.launch = [=] {
                launch_mat_mul_naive_kernel(a, b, output, options.rows, options.inner, options.cols);
            };
        } else {
            prepared.launch = [=] {
                launch_mat_mul_kernel(a, b, output, options.rows, options.inner, options.cols);
            };
        }
        prepared.shape = "m=" + std::to_string(options.rows) + ", k=" +
                         std::to_string(options.inner) + ", n=" + std::to_string(options.cols);
    } else if (name == "batched-gemm" || name == "broadcast-gemm") {
        const bool broadcast = name == "broadcast-gemm";
        const std::size_t a_batch = broadcast ? 1 : static_cast<std::size_t>(options.batch);
        const std::size_t a_count = checked_product(
            {a_batch, static_cast<std::size_t>(options.rows), static_cast<std::size_t>(options.inner)});
        const std::size_t b_count = checked_product(
            {static_cast<std::size_t>(options.batch), static_cast<std::size_t>(options.inner),
             static_cast<std::size_t>(options.cols)});
        const std::size_t output_count = checked_product(
            {static_cast<std::size_t>(options.batch), static_cast<std::size_t>(options.rows),
             static_cast<std::size_t>(options.cols)});
        float* a = prepared.allocations.allocate(a_count);
        float* b = prepared.allocations.allocate(b_count);
        float* output = prepared.allocations.allocate(output_count, false);
        if (broadcast) {
            prepared.launch = [=] {
                launch_broadcast_mat_mul_kernel(
                    a, b, output, options.batch, options.rows, options.inner, options.cols);
            };
        } else {
            prepared.launch = [=] {
                launch_batched_mat_mul_kernel(
                    a, b, output, options.batch, options.rows, options.inner, options.cols);
            };
        }
        prepared.shape = "batch=" + std::to_string(options.batch) + ", m=" +
                         std::to_string(options.rows) + ", k=" +
                         std::to_string(options.inner) + ", n=" + std::to_string(options.cols);
    } else {
        throw std::runtime_error("unknown kernel '" + name + "' (use --list)");
    }

    check_cuda(cudaDeviceSynchronize(), "CUDA setup");
    return prepared;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (options.list) {
            print_kernel_list();
            return EXIT_SUCCESS;
        }
        if (options.kernel.empty()) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        PreparedKernel prepared = prepare_kernel(options);
        std::cout << "Kernel: " << options.kernel << " (" << prepared.shape << ")\n"
                  << "Warmup launches: " << options.warmup
                  << ", measured launches: " << options.iterations << '\n';

        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            prepared.launch();
        }
        check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

        check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            prepared.launch();
        }
        check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop), "kernel synchronization");

        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
        check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

        std::cout << std::fixed << std::setprecision(3)
                  << "Average CUDA time: " << (elapsed_ms * 1000.0f / options.iterations)
                  << " us\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
