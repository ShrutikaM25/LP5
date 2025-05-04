#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda.h>

#define VECTOR_SIZE 1024 

using namespace std;


__global__ void vectorAdd(float* A, float* B, float* C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        C[i] = A[i] + B[i];
}


__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}


void fillRandom(float* arr, int size) {
    for (int i = 0; i < size; ++i)
        arr[i] = static_cast<float>(rand() % 100);
}


void printPartial(const float* arr, int size, int cols = -1) {
    int limit = min(size, 10);
    for (int i = 0; i < limit; ++i) {
        if (cols > 0 && i % cols == 0) cout << "\n";
        cout << arr[i] << " ";
    }
    cout << "\n...\n";
}

int main() {
    srand(time(0));

    int size = VECTOR_SIZE;
    int bytes = size * sizeof(float);

    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C = new float[size];

    fillRandom(h_A, size);
    fillRandom(h_B, size);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, size);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cout << "\n=== Vector Addition (first 10 elements) ===\n";
    printPartial(h_C, size);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    // =============================
    // 2. MATRIX MULTIPLICATION
    // =============================
    int matrixSize = VECTOR_SIZE;
    int matrixBytes = matrixSize * matrixSize * sizeof(float);

    float *h_M1 = new float[matrixSize * matrixSize];
    float *h_M2 = new float[matrixSize * matrixSize];
    float *h_Mout = new float[matrixSize * matrixSize];

    fillRandom(h_M1, matrixSize * matrixSize);
    fillRandom(h_M2, matrixSize * matrixSize);

    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc(&d_M1, matrixBytes);
    cudaMalloc(&d_M2, matrixBytes);
    cudaMalloc(&d_Mout, matrixBytes);

    cudaMemcpy(d_M1, h_M1, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, h_M2, matrixBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((matrixSize + 15) / 16, (matrixSize + 15) / 16);

    matrixMul<<<numBlocks, threadsPerBlock>>>(d_M1, d_M2, d_Mout, matrixSize);
    cudaMemcpy(h_Mout, d_Mout, matrixBytes, cudaMemcpyDeviceToHost);

    cout << "\n=== Matrix Multiplication (first 10 elements of result) ===\n";
    printPartial(h_Mout, matrixSize * matrixSize, matrixSize);

    cudaFree(d_M1); cudaFree(d_M2); cudaFree(d_Mout);
    delete[] h_M1; delete[] h_M2; delete[] h_Mout;

    return 0;
}
