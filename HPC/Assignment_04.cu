// !nvcc -arch=sm_75 <assignmentName>.cu -o output
// !./output

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
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

// CPU Vector Add
void vectorAddCPU(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

// CPU Matrix Mul
void matrixMulCPU(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += a[i * N + k] * b[k * N + j];
            c[i * N + j] = sum;
        }
}

void printVector(int *v, int n) {
    for (int i = 0; i < n; i++) cout << v[i] << " ";
    cout << endl;
}

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

    // ------------------- Vector Addition -------------------
    int a[vecSize], b[vecSize], c_cpu[vecSize], c_gpu[vecSize];
    for (int i = 0; i < vecSize; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // CPU Vector Add Time
    auto startCPU = chrono::high_resolution_clock::now();
    vectorAddCPU(a, b, c_cpu, vecSize);
    auto endCPU = chrono::high_resolution_clock::now();
    auto durationCPU = chrono::duration_cast<chrono::microseconds>(endCPU - startCPU);
    cout << "CPU Vector Add Time: " << durationCPU.count() << " microseconds\n";

    // GPU Vector Add Time
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, vecSize * sizeof(int));
    cudaMalloc(&d_b, vecSize * sizeof(int));
    cudaMalloc(&d_c, vecSize * sizeof(int));

    cudaMemcpy(d_a, a, vecSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, vecSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t startGPU, endGPU;
    float timeGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&endGPU);

    cudaEventRecord(startGPU);
    vectorAdd<<<1, vecSize>>>(d_a, d_b, d_c, vecSize);
    cudaEventRecord(endGPU);
    cudaEventSynchronize(endGPU);
    cudaEventElapsedTime(&timeGPU, startGPU, endGPU);

    cudaMemcpy(c_gpu, d_c, vecSize * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "GPU Vector Add Time: " << timeGPU * 1000 << " microseconds\n";

    cout << "Vector A: "; printVector(a, vecSize);
    cout << "Vector B: "; printVector(b, vecSize);
    cout << "CPU Result: "; printVector(c_cpu, vecSize);
    cout << "GPU Result: "; printVector(c_gpu, vecSize);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    // ------------------- Matrix Multiplication -------------------
    int A[matrixSize * matrixSize], B[matrixSize * matrixSize];
    int C_cpu[matrixSize * matrixSize], C_gpu[matrixSize * matrixSize];
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    startCPU = chrono::high_resolution_clock::now();
    matrixMulCPU(A, B, C_cpu, matrixSize);
    endCPU = chrono::high_resolution_clock::now();
    durationCPU = chrono::duration_cast<chrono::microseconds>(endCPU - startCPU);
    cout << "\nCPU Matrix Mul Time: " << durationCPU.count() << " microseconds\n";

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrixSize * matrixSize * sizeof(int));
    cudaMalloc(&d_B, matrixSize * matrixSize * sizeof(int));
    cudaMalloc(&d_C, matrixSize * matrixSize * sizeof(int));

    cudaMemcpy(d_A, A, matrixSize * matrixSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, matrixSize * matrixSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(startGPU);
    dim3 threads(16, 16);
    dim3 blocks((matrixSize + 15) / 16, (matrixSize + 15) / 16);
    matrixMul<<<blocks, threads>>>(d_A, d_B, d_C, matrixSize);
    cudaEventRecord(endGPU);
    cudaEventSynchronize(endGPU);
    cudaEventElapsedTime(&timeGPU, startGPU, endGPU);

    cudaMemcpy(C_gpu, d_C, matrixSize * matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "GPU Matrix Mul Time: " << timeGPU * 1000 << " microseconds\n";

    cout << "\nMatrix A:\n"; printMatrix(A, matrixSize);
    cout << "Matrix B:\n"; printMatrix(B, matrixSize);
    cout << "CPU Result:\n"; printMatrix(C_cpu, matrixSize);
    cout << "GPU Result:\n"; printMatrix(C_gpu, matrixSize);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(startGPU); cudaEventDestroy(endGPU);

    return 0;
}
