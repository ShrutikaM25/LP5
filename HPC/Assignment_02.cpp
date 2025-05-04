#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Utility to print array
void printArray(const vector<int>& arr) {
    for (int val : arr)
        cout << val << " ";
    cout << endl;
}

// Sequential Bubble Sort
void bubbleSortSequential(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Parallel Bubble Sort using Odd-Even Transposition Sort
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; ++i) {
        #pragma omp parallel for
        for (int j = (i % 2); j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// Merge two subarrays
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    vector<int> L(n1), R(n2);
    
    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

// Sequential Merge Sort
void mergeSortSequential(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left)/2;
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid+1, right);
        merge(arr, left, mid, right);
    }
}

// Parallel Merge Sort using OpenMP Tasks
void mergeSortParallel(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left)/2;

        #pragma omp task shared(arr)
        mergeSortParallel(arr, left, mid);

        #pragma omp task shared(arr)
        mergeSortParallel(arr, mid+1, right);

        #pragma omp taskwait
        merge(arr, left, mid, right);
    }
}

// Helper to copy vector
vector<int> copyArray(const vector<int>& original) {
    return vector<int>(original.begin(), original.end());
}

// Generate random array
vector<int> generateRandomArray(int size, int maxVal = 10000) {
    vector<int> arr(size);
    for (int i = 0; i < size; ++i)
        arr[i] = rand() % maxVal;
    return arr;
}

int main() {
    srand(time(0));
    const int N = 5000; // Reasonable size for bubble sort
    vector<int> base = generateRandomArray(N);

    // === Bubble Sort Timing ===
    vector<int> bubble_seq = copyArray(base);
    double start1 = omp_get_wtime();
    bubbleSortSequential(bubble_seq);
    double end1 = omp_get_wtime();

    vector<int> bubble_par = copyArray(base);
    double start2 = omp_get_wtime();
    bubbleSortParallel(bubble_par);
    double end2 = omp_get_wtime();

    // === Merge Sort Timing ===
    vector<int> merge_seq = copyArray(base);
    double start3 = omp_get_wtime();
    mergeSortSequential(merge_seq, 0, N - 1);
    double end3 = omp_get_wtime();

    vector<int> merge_par = copyArray(base);
    double start4 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSortParallel(merge_par, 0, N - 1);
    }
    double end4 = omp_get_wtime();

    // === Output Results ===
    cout << "=== Bubble Sort ===" << endl;
    cout << "Sequential Time: " << (end1 - start1) * 1000 << " ms" << endl;
    cout << "Parallel Time  : " << (end2 - start2) * 1000 << " ms" << endl;

    cout << "\n=== Merge Sort ===" << endl;
    cout << "Sequential Time: " << (end3 - start3) * 1000 << " ms" << endl;
    cout << "Parallel Time  : " << (end4 - start4) * 1000 << " ms" << endl;

    return 0;
}


// command to run this code: cd "d:\4th year sem 8\HPC\PRACTICALS\" ; if ($?) { g++ -fopenmp Assignment_02.cpp -o Assignment_02 } ; if ($?) { .\Assignment_02 }
// OR
// g++ -fopenmp Assignment_02.cpp -o Assignment_02 && ./Assignment_02 
