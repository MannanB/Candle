#include "gemm_vectorized.h"

#define BM 128 // rows of C computed by a block
#define BN 128 // cols of C computed by a block
#define BK 8  // # of K elements loaded per iteration
#define TM 8  // rows of C computed by a thread
#define TN 8  // cols of C computed by a thread

// A -> M x K, B -> K x N, C -> M x N

__global__ void mat_mul_vectorized_kernel(float alpha, float* A, float* B, float beta, float* C, int A_rows, int shared_dim, int B_cols) {
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int num_tiles = ceil((float)shared_dim / BK);

    __shared__ float tile_shared_mem_A[BK][BM];
    __shared__ float tile_shared_mem_B[BK][BN];

    float values[TM][TN] = {0.0};
    float reg_A[TM];
    float reg_B[TN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // y*16+x
    const int num_threads = blockDim.x * blockDim.y; // 16 * 16 = 256

    for (int tile = 0; tile < num_tiles; tile++) {
        // we are trying to make a 128x128 block
        // we need to load 128 x 8 from A and 8 x 128 from B
        // There are (128 / 8)^2 = 256 threads
        // 128 * 8 / 256 = 4. each thread does a perfect 4 number vectorized load
        // each tid does next 4. better to imagine tile shared and A as flattened


        // basically, a single thread loads the row of A_tile, but not necessarily uses those for its calcuation
        const int local_row_A = tid / (BK / 4);
        const int local_col_A = (tid % (BK / 4)) * 4;
        const int inner_row_A = blockIdx.y * BM + local_row_A;
        const int inner_col_A = tile * BK + local_col_A;
        const int A_start_idx_four = inner_row_A * shared_dim + inner_col_A; // for vec 4
        // max = 255 * 4 = 1020 (1021,1022,1023) -> which make sense cuz 128 * 8 = 1024


        float4 tmp = reinterpret_cast<float4 *>(&A[A_start_idx_four])[0];

        tile_shared_mem_A[local_col_A + 0][local_row_A] = tmp.x;
        tile_shared_mem_A[local_col_A + 1][local_row_A] = tmp.y;
        tile_shared_mem_A[local_col_A + 2][local_row_A] = tmp.z;
        tile_shared_mem_A[local_col_A + 3][local_row_A] = tmp.w;


        const int local_row_B = tid / (BN / 4);
        const int local_col_B = (tid % (BN / 4)) * 4;
        const int inner_row_B = tile * BK + local_row_B;
        const int inner_col_B = blockIdx.x * BN + local_col_B;
        const int B_start_idx_four = inner_row_B * B_cols + inner_col_B; // for vec 4

        reinterpret_cast<float4 *>(&tile_shared_mem_B[local_row_B][local_col_B])[0] =
            reinterpret_cast<float4 *>(&B[B_start_idx_four])[0];


        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (int resIdx = 0; resIdx < TM; resIdx++) {
                reg_A[resIdx] = tile_shared_mem_A[dotIdx][thread_row * TM + resIdx];
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
        for (int colIdx = 0; colIdx < TN; colIdx += 4) {
            const int row = blockIdx.y * BM + thread_row * TM + rowIdx;
            const int col = blockIdx.x * BN + thread_col * TN + colIdx;

            float4 result = {
                alpha * values[rowIdx][colIdx + 0],
                alpha * values[rowIdx][colIdx + 1],
                alpha * values[rowIdx][colIdx + 2],
                alpha * values[rowIdx][colIdx + 3]
            };

            if (beta != 0.0f) {
                const float4 old = reinterpret_cast<float4 *>(&C[row * B_cols + col])[0];
                result.x += beta * old.x;
                result.y += beta * old.y;
                result.z += beta * old.z;
                result.w += beta * old.w;
            }

            reinterpret_cast<float4 *>(&C[row * B_cols + col])[0] = result;
        }
    }
}

void launch_mat_mul_vectorized_kernel(float* A, float* B, float* C, int A_rows, int shared_dim, int B_cols) {
    dim3 threadsPerBlock(BN / TN, BM / TM);
    dim3 numBlocks(cuda::ceil_div(B_cols, BN), cuda::ceil_div(A_rows, BM));

    mat_mul_vectorized_kernel<<<numBlocks, threadsPerBlock>>>(1.0f, A, B, 0.0f, C, A_rows, shared_dim, B_cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
