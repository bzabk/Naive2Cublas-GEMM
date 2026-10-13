# Naive2Cublas-GEMM
Repository demonstrates step-by-step optimization to achieve performance close to the ***cuBLAS Gemm***. Kernels perform matrix multiplication in cuda-style
$$C = \alpha \cdot op(A) \cdot op(B) + \beta C, \quad op(X) \in \{X, X^T\}$$
All matrices are store in row-major order

This project is inspired by Simon Boehm's blog post: [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM).

### List of Kernels
1. Naive without coalescing memory read
2. Naive with coalescing memory read
3. Shared Memory Tiling
4. Thread Tiling 1D
5. Thread Tiling 2D
6. Vectorized Memory Access



