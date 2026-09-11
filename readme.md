# Candle: torch but candle (wip)

A simpler recreation of pytorch with cuda kernels from scratch (no AI for kernels)

Candle now works standalone! An example training an mnist classifier is in python/mnist. Lots more coming soon

## Layout

- `cpp/` contains the native tensor implementation, pybind11 bindings, utilities,
  and CUDA kernels.
- `python/` contains the public Python module and Python tests.
- `scripts/` contains the Windows setup, build, and test helpers.

TODOs

- [x] Simple kernels (add, mm, transpose) 
    - [x] simple gemm (batched,broadcast)
    - [ ] simple gemv 
    - [ ] add tranpose flag to gemm
- [x] Simple python bindings
- [ ] Better matmul (mma, tmem / tensor core usage, coarse)
    - [x] batched matmul / normal pytorch style matmul dimensions
    - [ ] fused forward kernels (activations)
    - [ ] fused backward kernels (add batch reduction)
- [x] Separate native C++/CUDA code from the Python API
- [ ] Python-side type hints / QOL
- [ ] half precision
- [ ] Optimizers
    - [x] SGD
    - [ ] Adam
    - [ ] AdamW
- [x] simple autograd
- [ ] separate tensors from host mem / device mem
- [x] working Mnist w/o pytorch
    - [x] binary classifier
    - [ ] multi-class classification model
    - [ ] image processing kernels?
- [ ] tiny transformer model
- [ ] Flash attention
