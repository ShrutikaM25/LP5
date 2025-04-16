#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
using namespace std;

// Sequential Bubble Sort
void bubbleSortSequential(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; ++i)
        for (int j = 0; j < n-i-1; ++j)
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
}

// Parallel Bubble Sort using OpenMP
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; ++i) {
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Merge function
void merge(vector<int>& arr, int l, int m, int r) {
    int i = l, j = m+1;
    vector<int> temp;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j])
            temp.push_back(arr[i++]);
        else
            temp.push_back(arr[j++]);
    }
    while (i <= m) temp.push_back(arr[i++]);
    while (j <= r) temp.push_back(arr[j++]);

    for (int k = 0; k < temp.size(); ++k)
        arr[l + k] = temp[k];
}

// Sequential Merge Sort
void mergeSortSequential(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    mergeSortSequential(arr, l, m);
    mergeSortSequential(arr, m + 1, r);
    merge(arr, l, m, r);
}

// Parallel Merge Sort using OpenMP
void mergeSortParallel(vector<int>& arr, int l, int r, int depth=0) {
    if (l >= r) return;
    int m = (l + r) / 2;

    if (depth <= 3) { // limit parallel recursion depth
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortParallel(arr, l, m, depth+1);
            #pragma omp section
            mergeSortParallel(arr, m + 1, r, depth+1);
        }
    } else {
        mergeSortSequential(arr, l, m);
        mergeSortSequential(arr, m + 1, r);
    }

    merge(arr, l, m, r);
}

// Utility: generate random array
vector<int> generateRandomArray(int n) {
    vector<int> arr(n);
    for (int i = 0; i < n; ++i)
        arr[i] = rand() % 10000;
    return arr;
}

int main() {
    srand(time(0));
    int n = 10000;
    vector<int> arr1 = generateRandomArray(n);
    vector<int> arr2 = arr1;
    vector<int> arr3 = arr1;
    vector<int> arr4 = arr1;

    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();
    bubbleSortSequential(arr1);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << end - start << " sec\n";

    // Parallel Bubble Sort
    start = omp_get_wtime();
    bubbleSortParallel(arr2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << end - start << " sec\n";

    // Sequential Merge Sort
    start = omp_get_wtime();
    mergeSortSequential(arr3, 0, n - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << end - start << " sec\n";

    // Parallel Merge Sort
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSortParallel(arr4, 0, n - 1);
    }
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << " sec\n";

    return 0;
}


// command to run this code: cd "d:\4th year sem 8\HPC\PRACTICALS\" ; if ($?) { g++ -fopenmp Assignment_02.cpp -o Assignment_02 } ; if ($?) { .\Assignment_02 }
// OR
// g++ -fopenmp Assignment_02.cpp -o Assignment_02 && ./Assignment_02 
