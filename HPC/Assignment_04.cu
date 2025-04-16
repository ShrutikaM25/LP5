#include <iostream>
#include <cuda_runtime.h>

#define N 1000000     // Vector size
#define WIDTH 512     // Matrix size (WIDTH x WIDTH)

// ------------------------------
// Vector Addition Kernel
// ------------------------------
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

// ------------------------------
// Matrix Multiplication Kernel
// ------------------------------
__global__ void matrixMul(float *A, float *B, float *C, int width) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k)
            sum += A[row * width + k] * B[k * width + col];
        C[row * width + col] = sum;
    }
}

int main() {
    // ----------- VECTOR ADDITION ----------
    float *a, *b, *c;               // host
    float *d_a, *d_b, *d_c;         // device
    size_t sizeVec = N * sizeof(float);

    a = (float*)malloc(sizeVec);
    b = (float*)malloc(sizeVec);
    c = (float*)malloc(sizeVec);

    for (int i = 0; i < N; ++i) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    cudaMalloc(&d_a, sizeVec);
    cudaMalloc(&d_b, sizeVec);
    cudaMalloc(&d_c, sizeVec);

    cudaMemcpy(d_a, a, sizeVec, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeVec, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaMemcpy(c, d_c, sizeVec, cudaMemcpyDeviceToHost);

    std::cout << "\n=== Vector Addition ===\n";
    std::cout << "Sample result: c[0] = " << c[0] << ", c[N-1] = " << c[N-1] << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);

    // ----------- MATRIX MULTIPLICATION ----------
    int sizeMat = WIDTH * WIDTH * sizeof(float);
    float *A = (float*)malloc(sizeMat);
    float *B = (float*)malloc(sizeMat);
    float *C = (float*)malloc(sizeMat);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeMat); cudaMalloc(&d_B, sizeMat); cudaMalloc(&d_C, sizeMat);

    for (int i = 0; i < WIDTH * WIDTH; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    cudaMemcpy(d_A, A, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeMat, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((WIDTH + threads.x - 1) / threads.x, (WIDTH + threads.y - 1) / threads.y);

    matrixMul<<<blocks, threads>>>(d_A, d_B, d_C, WIDTH);
    cudaMemcpy(C, d_C, sizeMat, cudaMemcpyDeviceToHost);

    std::cout << "\n=== Matrix Multiplication ===\n";
    std::cout << "Sample result: C[0] = " << C[0] << ", C[WIDTH*WIDTH-1] = " << C[WIDTH * WIDTH - 1] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);

    return 0;
}
