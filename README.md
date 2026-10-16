# Naive2Cublas-GEMM
Repository demonstrates step-by-step optimization to achieve performance close to the ***cuBLAS Gemm***. Kernels perform matrix multiplication in cuda-style
$$C = \alpha \cdot op(A) \cdot op(B)+\beta C, \quad op(X) \in \{X, X^T\}$$
All matrices are store in row-major order

This project is inspired by Simon Boehm's blog post: [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM).

### List of Kernels
1. Naive without coalescing memory read
2. Naive with coalescing memory read
3. Shared Memory Tiling
4. Thread Tiling 1D
5. Thread Tiling 2D
6. Vectorized Memory Access


## Naive without coalescing memory read
In CUDA threads are grouped in packages of size 32 called warps. Thread indices increase along X axis (`threadId.x`).
By setting up
```cuda
int const col = blockIdx.y * blockDim.y + threadIdx.y;
int const row = blockIdx.x * blockDim.x + threadIdx.x;
```
we are denying natural order in which thread are laid out in the CUDA architecture. It results in inefficient
memory access caused by the fact that threads from the same warp are calling to the memory addresses that are
away from each other by the stride of K (common dimension for A and B matrices).




