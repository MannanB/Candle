# Candle: torch but candle (very wip)

A simpler recreation of pytorch with cuda kernels from scratch (no AI for kernels)

TODOs

- [x] Simple kernels (add, mm, transpose) 
    - [x] simple gemm (batched,broadcast)
    - [ ] simple gemv 
- [x] Simple python bindings
- [ ] Better matmul (mma, tmem / tensor core usage, coarse)
    - [x] batched matmul / normal pytorch style matmul dimensions
    - [ ] fused forward kernels (activations)
    - [ ] fused backward kernels
- [ ] cleanup file structure
- [ ] Python-side type hints / QOL
- [ ] half precision
- [ ] Optimizers
    - [ ] SGD
    - [ ] Adam
    - [ ] AdamW
- [ ] simple autograd
- [ ] separate tensors from host mem / device mem
- [ ] working Mnist w/o pytorch
    - [ ] image processing kernels?
- [ ] tiny transformer model
- [ ] Flash attention
