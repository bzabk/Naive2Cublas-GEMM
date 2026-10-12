#pragma once
#include <cuda_runtime.h>

namespace gemm::kernel {

    template<bool const transA, bool const transB>
    __global__ void naive_coalescing(
        float const * __restrict__ A,
        float const * __restrict__ B,
        float * __restrict__ C,
        float const alpha, float const beta,
        int const M,
        int const N,
        int const K) {

        int const row = blockIdx.y * blockDim.y + threadIdx.y;
        int const col = blockIdx.x * blockDim.x + threadIdx.x;

        float local_result = 0.0f;
        if (col < N && row < M) {
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

    static __global__ void naive_manual_transpose(
        float const * __restrict__ A,
        float const * __restrict__ B,
        float * __restrict__ C,
        float const alpha, float const beta,
        bool const transA, bool const transB,
        int const M,
        int const N,
        int const K) {

        int const row = blockIdx.y * blockDim.y + threadIdx.y;
        int const col = blockIdx.x * blockDim.x + threadIdx.x;

        float local_result = 0.0f;
        if (col < N && row < M) {
            int const c_idx = N * row + col;
            for (int k = 0; k < K; k++) {
                int a_idx;
                int b_idx;

                if (transA) {
                    a_idx = M * k + row;
                } else {
                    a_idx = K * row + k;
                }

                if (transB) {
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
