#include <stdio.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

#define n 4096
#define m 4096
#define BLOCK_SIZE 32

__global__
void compute_Anew(float *A, float *Anew) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > 0 && row < n - 1 && col > 0 && col < m - 1) {
        Anew[row*m + col] = (A[(row+1)*m + col] + A[(row-1)*m + col] + A[row*m + (col+1)] + A[row*m + (col-1)]) / 4;
    }
}

__global__
void compute_error(float *A, float *Anew, float *block_errors) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    __shared__ float squared_errors[BLOCK_SIZE*BLOCK_SIZE];
    if (row > 0 && row < n - 1 && col > 0 && col < m - 1) {
        // Each thread compute an error
        squared_errors[threadIdx.y * BLOCK_SIZE + threadIdx.x] = fabs(Anew[row*m + col] - A[row*m + col]);
    } else {
        squared_errors[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0;
    }
    

    // Take the maximum error of every block, in the shared memory
    __syncthreads();
    for (i=2; i<=BLOCK_SIZE; i *= 2) {
        if (threadIdx.x % i == 0)
            if (squared_errors[threadIdx.y * BLOCK_SIZE + threadIdx.x] < squared_errors[threadIdx.y * BLOCK_SIZE + threadIdx.x + i / 2])
                squared_errors[threadIdx.y * BLOCK_SIZE + threadIdx.x] = squared_errors[threadIdx.y * BLOCK_SIZE + threadIdx.x + i / 2];
        __syncthreads();
    }

    // Now, the first thread of the first row, stores the maximum value among its column
    if (threadIdx.x == 0) {
        for (i=2; i<=BLOCK_SIZE; i *= 2) {
            if (threadIdx.y % i == 0)
                if (squared_errors[threadIdx.y * BLOCK_SIZE] < squared_errors[(threadIdx.y + i / 2) * BLOCK_SIZE])
                    squared_errors[threadIdx.y * BLOCK_SIZE] = squared_errors[(threadIdx.y + i / 2) * BLOCK_SIZE];
            __syncthreads();
        }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        block_errors[blockIdx.y * gridDim.x + blockIdx.x] = squared_errors[0];
        __syncthreads();
        if (blockIdx.x == 0) {
            for (i=0; i<gridDim.x; i++) {
                if (block_errors[blockIdx.y * gridDim.x] < block_errors[blockIdx.y * gridDim.x + i])
                    block_errors[blockIdx.y * gridDim.x] = block_errors[blockIdx.y * gridDim.x + i];
            }
            __syncthreads();
            if (blockIdx.y == 0) {
                for (i=0; i<gridDim.y; i++) {
                    if (block_errors[0] < block_errors[i * gridDim.x])
                        block_errors[0] = block_errors[i * gridDim.x];
                }
            }
        }
    }
    // Store the result in max error in block_errors[0]
}


__global__
void copy_Anew(float *A, float *Anew) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > 0 && row < n - 1 && col > 0 && col < m - 1) {
        A[row*m + col] = Anew[row*m + col];
    }
}

int main(int argc, char** argv) 
{ 
    // Set block and grid sizes
    dim3 block_shape = dim3(BLOCK_SIZE, BLOCK_SIZE);
    int blocks_in_x_grid = max(1.0, ceil((float) m / (float) block_shape.x));
    int blocks_in_y_grid = max(1.0, ceil((float) n / (float) block_shape.y));
    dim3 grid_shape = dim3(blocks_in_x_grid, blocks_in_y_grid);
    int n_blocks = blocks_in_x_grid * blocks_in_y_grid;

    // Set parameters
    const float tol = 3.0e-3f; 
    float error = 1.0;
    int i, iter_max=100000, iter=0; 

    // get iter_max from command line at execution time 
    if (argc>1) {  iter_max = atoi(argv[1]); } 

    // Initialize pointers
    float *A_host, *A, *Anew;
    float *block_errors;
    A_host = (float *) malloc(n * m * sizeof(float));
    cudaMalloc((void **) &A, n * m * sizeof(float));
    cudaMalloc((void **) &Anew, n * m * sizeof(float));
    cudaMalloc((void **) &block_errors, n_blocks * sizeof(float));

    // set all values in matrix as zero  
    memset(A_host, 0, n * m * sizeof(float)); 

    // set boundary conditions 
    for (i=0; i < n; i++) {
        A_host[i*m] = sin(i*M_PI / (n-1)); 
        A_host[i*m + m - 1] = sin(i*M_PI / (n-1))*exp(-M_PI); 
    } 

    // Transfer data from host to device
    cudaMemcpy(A, A_host, n * m * sizeof(float), cudaMemcpyHostToDevice);

    // Main loop: iterate until error <= tol a maximum of iter_max iterations 
    while ( error > tol && iter < iter_max ) { 
        // Compute new values using main matrix and writing into auxiliary matrix
        compute_Anew<<<grid_shape, block_shape>>>(A, Anew);

        // Compute error = maximum of the square root of the absolute differences
        compute_error<<<grid_shape, block_shape>>>(A, Anew, block_errors);
        cudaMemcpy(&error, block_errors, sizeof(float), cudaMemcpyDeviceToHost);
        error = sqrt(error);

        // Copy from auxiliary matrix to main matrix
        copy_Anew<<<grid_shape, block_shape>>>(A, Anew);

        // if number of iterations is multiple of 10 then print error on the screen    
        iter++; 
        if (iter % (iter_max/100) == 0)
            printf("For iteration %d error is still %f\n", iter, error); 
    } // while 
    printf("For iteration %d error is %f\n", iter, error);
    free(A_host);
    cudaFree(A); cudaFree(Anew); cudaFree(block_errors);
    return 0;
} 
