#include "transpose.h"


#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 8

/* macro to index a 1D memory array with 2D indices in column-major order */
/* ld is the leading dimension, i.e. the number of rows in the matrix     */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* CUDA kernel for shared memory matrix transpose */
__global__ void mat_transpose_kernel(float *a, float *c, int numRows, int numCols)
{

    /* declare a statically allocated shared memory array */

    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y+1];

    /* determine my row and column indices for the error checking code */

    const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

    /* determine my row tile and column tile index */

    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    if( myRow < numRows && myCol < numCols )
    {
        /* read from global memory into shared memory array */
        // todo: is this numRows or numCols
        smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileX + threadIdx.x, tileY + threadIdx.y, numRows )];
    } /* end if */

    /* synchronize the threads in the thread block */
    __syncthreads();

    if ((tileY + threadIdx.x) < numCols &&
        (tileX + threadIdx.y) < numRows) {
        /* write the result from shared memory to global memory */
        // todo: is this numCols or numRows
        c[INDX( tileY + threadIdx.x, tileX + threadIdx.y, numCols)] = smemArray[threadIdx.y][threadIdx.x];
    } /* end if */
    return;

} 

void launch_mat_transpose_kernel(float* A, float* C, int numRows, int numCols) {
    // assumes A and B are already on device
    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks(
        cuda::ceil_div(numRows, threadsPerBlock.x),
        cuda::ceil_div(numCols, threadsPerBlock.y)
    );

    mat_transpose_kernel<<<numBlocks, threadsPerBlock>>>(A, C, numRows, numCols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}
