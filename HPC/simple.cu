// !nvcc -arch=sm_75 <assignmentName>.cu -o output
// !./output

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA Kernel for Vector Addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMul(int *a, int *b, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++)
            sum += a[row * N + k] * b[k * N + col];
        c[row * N + col] = sum;
    }
}

// Host function for Vector Addition (CPU)
void vectorAddCPU(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

// Host function for Matrix Multiplication (CPU)
void matrixMulCPU(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += a[i * N + k] * b[k * N + j];
            c[i * N + j] = sum;
        }
}

// Utility: Print vector
void printVector(int *v, int n) {
    for (int i = 0; i < n; i++)
        cout << v[i] << " ";
    cout << endl;
}

// Utility: Print matrix
void printMatrix(int *m, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << m[i * N + j] << " ";
        cout << endl;
    }
}

int main() {
    const int vecSize = 8;
    const int matrixSize = 2;

    // ---------------- Vector Addition ----------------
    int a[vecSize], b[vecSize], c_cpu[vecSize], c_gpu[vecSize];
    for (int i = 0; i < vecSize; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    vectorAddCPU(a, b, c_cpu, vecSize);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, vecSize * sizeof(int));
    cudaMalloc(&d_b, vecSize * sizeof(int));
    cudaMalloc(&d_c, vecSize * sizeof(int));

    cudaMemcpy(d_a, a, vecSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, vecSize * sizeof(int), cudaMemcpyHostToDevice);

    vectorAdd<<<1, vecSize>>>(d_a, d_b, d_c, vecSize);
    cudaMemcpy(c_gpu, d_c, vecSize * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Vector A: "; printVector(a, vecSize);
    cout << "Vector B: "; printVector(b, vecSize);
    cout << "CPU Result: "; printVector(c_cpu, vecSize);
    cout << "GPU Result: "; printVector(c_gpu, vecSize);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    // ---------------- Matrix Multiplication ----------------
    int A[matrixSize * matrixSize], B[matrixSize * matrixSize], C_cpu[matrixSize * matrixSize], C_gpu[matrixSize * matrixSize];
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    matrixMulCPU(A, B, C_cpu, matrixSize);

    int *d_A, *d_B, *d_C;
    size_t size = matrixSize * matrixSize * sizeof(int);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((matrixSize + 15) / 16, (matrixSize + 15) / 16);
    matrixMul<<<blocks, threads>>>(d_A, d_B, d_C, matrixSize);
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    cout << "\nMatrix A:\n"; printMatrix(A, matrixSize);
    cout << "Matrix B:\n"; printMatrix(B, matrixSize);
    cout << "CPU Result:\n"; printMatrix(C_cpu, matrixSize);
    cout << "GPU Result:\n"; printMatrix(C_gpu, matrixSize);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
