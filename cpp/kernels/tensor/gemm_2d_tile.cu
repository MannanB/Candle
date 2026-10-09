#include "gemm_2d_tile.h"

#define BM 64 // rows of C computed by a block
#define BN 64 // cols of C computed by a block
#define BK 8  // # of K elements loaded per iteration
#define TM 4  // rows of C computed by a thread
#define TN 4  // cols of C computed by a thread

// A -> M x K, B -> K x N, C -> M x N

__global__ void mat_mul_2d_tile_kernel(float alpha, float* A, float* B, float beta, float* C, int A_rows, int shared_dim, int B_cols) {
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int num_tiles = ceil((float)shared_dim / BK);

    __shared__ float tile_shared_mem_A[BM][BK];
    __shared__ float tile_shared_mem_B[BK][BN];

    float values[TM][TN] = {0.0};
    float reg_A[TM];
    float reg_B[TN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // y*8+x
    const int num_threads = blockDim.x * blockDim.y; // 8 * 8 = 64

    for (int tile = 0; tile < num_tiles; tile++) {
        for (int idx = tid; idx < BM * BK; idx += num_threads) { // BM * BK = 64*8 = 512, 512 / 64 = 8
            // each thread loads 8 more values (K more values)
            // only local_row increases per a single a thread, local col=static
            const int local_row = idx / BK;
            const int local_col = idx % BK;

            // basically, a single thread loads the row of A_tile, but not necessarily uses those for its calcuation
            const int inner_row_A = blockIdx.y * BM + local_row;
            const int inner_col_A = tile * BK + local_col;

            if (inner_row_A < A_rows && inner_col_A < shared_dim) {
                tile_shared_mem_A[local_row][local_col] = A[inner_row_A * shared_dim + inner_col_A];
            } else {
                tile_shared_mem_A[local_row][local_col] = 0;
            }
        }

        for (int idx = tid; idx < BK * BN; idx += num_threads) {
            // same thing but transposed (cuz K dim is diff)
            const int local_row = idx / BN;
            const int local_col = idx % BN;

            const int inner_row_B = tile * BK + local_row;
            const int inner_col_B = blockIdx.x * BN + local_col;

            if (inner_row_B < shared_dim && inner_col_B < B_cols) {
                tile_shared_mem_B[local_row][local_col] = B[inner_row_B * B_cols + inner_col_B];
            } else {
                tile_shared_mem_B[local_row][local_col] = 0;
            }
        }

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (int resIdx = 0; resIdx < TM; resIdx++) {
                reg_A[resIdx] = tile_shared_mem_A[thread_row * TM + resIdx][dotIdx];
            }

            for (int resIdx = 0; resIdx < TN; resIdx++) {
                reg_B[resIdx] = tile_shared_mem_B[dotIdx][thread_col * TN + resIdx];
            }

            for (int rowIdx = 0; rowIdx < TM; rowIdx++) {
                for (int colIdx = 0; colIdx < TN; colIdx++) {
                    values[rowIdx][colIdx] += reg_A[rowIdx] * reg_B[colIdx];
                }
            }
        }

        __syncthreads();
    }

    for (int rowIdx = 0; rowIdx < TM; rowIdx++) {
        for (int colIdx = 0; colIdx < TN; colIdx++) {
            const int row = blockIdx.y * BM + thread_row * TM + rowIdx;
            const int col = blockIdx.x * BN + thread_col * TN + colIdx;

            if (row < A_rows && col < B_cols) {
                if (beta != 0.0f) {
                    C[row * B_cols + col] = alpha * values[rowIdx][colIdx] + beta * C[row * B_cols + col];
                } else {
                    C[row * B_cols + col] = alpha * values[rowIdx][colIdx];
                }
            }
        }
    }
}

void launch_mat_mul_2d_tile_kernel(float* A, float* B, float* C, int A_rows, int shared_dim, int B_cols) {
    dim3 threadsPerBlock(BN / TN, BM / TM);
    dim3 numBlocks(cuda::ceil_div(B_cols, BN), cuda::ceil_div(A_rows, BM));

    mat_mul_2d_tile_kernel<<<numBlocks, threadsPerBlock>>>(1.0f, A, B, 0.0f, C, A_rows, shared_dim, B_cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
