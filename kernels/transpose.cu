#include "transpose.h"


// TODO: prefix dim and suffix dim lowkey really dependant on the actual prefix/suffix prob need dynamic choosing?
#define THREADS_PREFIX_DIM 2
#define THREADS_ROW_DIM 4
#define THREADS_COL_DIM 8
#define THREADS_SUFFIX_DIM 8

#define THREADS_PER_BLOCK THREADS_PREFIX_DIM*THREADS_ROW_DIM*THREADS_COL_DIM*THREADS_SUFFIX_DIM
#define idx(p, r, c, s, R, C, S) ( p * R * C * S + r * S * C + c * S + s )

/* CUDA kernel for shared memory matrix transpose */
__global__ void mat_transpose_kernel(float *a, float *c, int numPrefix, int numRows, int numCols, int numSuffix,
                                                         int numBlocksPre, int numBlocksRow, int numBlocksCols, int numBlockSuf)
{

    /* declare a statically allocated shared memory array */

    __shared__ float smemArray[THREADS_PER_BLOCK]; // i hope this is banked

    /* determine my row and column indices for the error checking code */


    const int preBlockIdx = blockIdx.x / (numBlocksRow * numBlocksCols * numBlockSuf);
    const int rowBlockIdx = (blockIdx.x / (numBlocksCols * numBlockSuf)) % numBlocksRow;
    const int colBlockIdx = (blockIdx.x / numBlockSuf) % numBlocksCols;
    const int sufBlockIdx = blockIdx.x % numBlockSuf;

    const int preThreadIdx = threadIdx.x / (THREADS_ROW_DIM * THREADS_COL_DIM * THREADS_SUFFIX_DIM);
    const int rowThreadIdx = (threadIdx.x / (THREADS_COL_DIM * THREADS_SUFFIX_DIM)) % THREADS_ROW_DIM;
    const int colThreadIdx = (threadIdx.x / THREADS_SUFFIX_DIM) % THREADS_COL_DIM;
    const int sufThreadIdx = threadIdx.x % THREADS_SUFFIX_DIM;

    const int pIdx = preBlockIdx * THREADS_PREFIX_DIM + preThreadIdx;
    const int rIdx = rowBlockIdx * THREADS_ROW_DIM + rowThreadIdx;
    const int cIdx = colBlockIdx * THREADS_COL_DIM + colThreadIdx;
    const int sIdx = sufBlockIdx * THREADS_SUFFIX_DIM + sufThreadIdx;

    if( pIdx < numPrefix && rIdx < numRows && cIdx < numCols && sIdx < numSuffix )
    {
        smemArray[idx(preThreadIdx, rowThreadIdx, colThreadIdx, sufThreadIdx, THREADS_ROW_DIM, THREADS_COL_DIM, THREADS_SUFFIX_DIM)] 
                = a[idx(pIdx, rIdx, cIdx, sIdx, numRows, numCols, numSuffix)];
    } 

    /* synchronize the threads in the thread block */
    __syncthreads();

    const int rcLinear = rowThreadIdx * THREADS_COL_DIM + colThreadIdx;

    const int outRowLocal = rcLinear / THREADS_ROW_DIM;
    const int outColLocal = rcLinear % THREADS_ROW_DIM;

    const int rIdxNew = colBlockIdx * THREADS_COL_DIM + outRowLocal;

    const int cIdxNew = rowBlockIdx * THREADS_ROW_DIM + outColLocal;

    if (pIdx < numPrefix && rIdxNew < numCols && cIdxNew < numRows && sIdx < numSuffix ) {
        c[idx(pIdx, rIdxNew, cIdxNew, sIdx, numCols, numRows, numSuffix)] = 
        smemArray[idx(preThreadIdx, outColLocal, outRowLocal, sufThreadIdx, THREADS_ROW_DIM, THREADS_COL_DIM, THREADS_SUFFIX_DIM)];
    } 
    return;

} 

void launch_mat_transpose_kernel(float* A, float* C, int numPrefix, int numRows, int numCols, int numSuffix) {

    int numBlocksPre = cuda::ceil_div(numPrefix, THREADS_PREFIX_DIM);
    int numBlocksRow = cuda::ceil_div(numRows, THREADS_ROW_DIM);
    int numBlocksCol = cuda::ceil_div(numCols, THREADS_COL_DIM);
    int numBlocksSuf = cuda::ceil_div(numSuffix, THREADS_SUFFIX_DIM);
    int numBlocks = numBlocksPre*numBlocksRow*numBlocksCol*numBlocksSuf;

    mat_transpose_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(A, C, numPrefix, numRows, numCols, numSuffix, numBlocksPre, numBlocksRow, numBlocksCol, numBlocksSuf);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}
