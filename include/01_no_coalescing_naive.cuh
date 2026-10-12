#pragma once
#include <cuda_runtime.h>

namespace gemm::kernel {

    template<bool const transA, bool const transB>
    __global__ void naive_no_coalescing(
        float const * __restrict__ A,
        float const * __restrict__ B,
        float * __restrict__ C,
        float const alpha, float const beta,
        int const M,
        int const N,
        int const K) {

        int const col = blockIdx.y * blockDim.y + threadIdx.y;
        int const row = blockIdx.x * blockDim.x + threadIdx.x;
        // because threadIdx.x indexing works in a row major order, adjacent threads within the same warp
        // read from memory addresses separated by a stride of K causing uncoalescing memory access
        // It forces hardware to retrieve 32 seperate cache lines instead of a single transaction
        float local_result = 0.0f;
        if (row < M && col < N) {
            int const c_idx = N * row + col;
            for (int k = 0; k < K; k++) {
                int a_idx;
                int b_idx;

                if constexpr (transA) {
                    a_idx = M * k + row;
                } else {
                    a_idx = K * row + k;
                }

                if constexpr (transB) {
                    b_idx = K * col + k;
                } else {
                    b_idx = N * k + col;
                }

                local_result += A[a_idx] * B[b_idx];
            }
            C[c_idx] = alpha * local_result + beta * C[c_idx];
        }
    }

}