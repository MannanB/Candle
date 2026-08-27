# Candle: torch but candle (very wip)

A simpler recreation of pytorch with cuda kernels from scratch (no AI for kernels)

TODOs

- [x] Simple kernels (add, mm, transpose) 
- [x] Simple python bindings
- [ ] Better matmul (mma, tmem / tensor core usage)
    - [ ] batched matmul / normal pytorch style matmul dimensions
    - [ ] fused forward kernels (activations)
    - [ ] fused backward kernels
- [ ] cleanup file structure
- [ ] Python-side type hints / QOL
- [ ] Optimizers
    - [ ] SGD
    - [ ] Adam
    - [ ] AdamW
- [ ] simple autograd
- [ ] working Mnist w/o pytorch
    - [ ] image processing kernels?
- [ ] tiny transformer model
- [ ] Flash attention
