#include <iostream>
#include <omp.h>
#include <limits>

using namespace std;

int main() {
    const int N = 1000000;
    int* arr = new int[N];

    // Initialize array
    for (int i = 0; i < N; ++i)
        arr[i] = rand() % 1000;

    int min_val = INT_MAX;
    int max_val = INT_MIN;
    long long sum = 0;
    double avg = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < N; ++i) {
        if (arr[i] < min_val)
            min_val = arr[i];
    }

    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < N; ++i) {
        if (arr[i] > max_val)
            max_val = arr[i];
    }

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += arr[i];
    }

    avg = static_cast<double>(sum) / N;

    double end = omp_get_wtime();

    cout << "Min: " << min_val << endl;
    cout << "Max: " << max_val << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;
    cout << "Execution Time: " << (end - start) << " seconds" << endl;

    delete[] arr;
    return 0;
}

// command to run this code: cd "d:\4th year sem 8\HPC\PRACTICALS\" ; if ($?) { g++ -fopenmp Assignment_02.cpp -o Assignment_02 } ; if ($?) { .\Assignment_02 }
// OR
// g++ -fopenmp Assignment_02.cpp -o Assignment_02 && ./Assignment_02 
