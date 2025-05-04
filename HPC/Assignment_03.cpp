#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>

using namespace std;

int main() {
    const int N = 1e7; // 10 million elements
    vector<int> data(N);

    // Initialize data
    for (int i = 0; i < N; ++i) {
        data[i] = i + 1;
    }

    // === Sequential Implementation ===
    long long seq_sum = 0, seq_min = INT_MAX, seq_max = INT_MIN;
    double seq_avg = 0.0;

    double start_seq = omp_get_wtime();

    for (int i = 0; i < N; ++i) {
        seq_sum += data[i];
        if (data[i] < seq_min) seq_min = data[i];
        if (data[i] > seq_max) seq_max = data[i];
    }
    seq_avg = static_cast<double>(seq_sum) / N;

    double end_seq = omp_get_wtime();
    double duration_seq = (end_seq - start_seq) * 1000; // in ms

    // === Parallel Implementation using OpenMP Reduction ===
    long long par_sum = 0, par_min = INT_MAX, par_max = INT_MIN;
    double par_avg = 0.0;

    double start_par = omp_get_wtime();

    #pragma omp parallel for reduction(+:par_sum)
    for (int i = 0; i < N; ++i) {
        par_sum += data[i];
    }

    #pragma omp parallel for reduction(min:par_min)
    for (int i = 0; i < N; ++i) {
        if (data[i] < par_min) par_min = data[i];
    }

    #pragma omp parallel for reduction(max:par_max)
    for (int i = 0; i < N; ++i) {
        if (data[i] > par_max) par_max = data[i];
    }

    par_avg = static_cast<double>(par_sum) / N;

    double end_par = omp_get_wtime();
    double duration_par = (end_par - start_par) * 1000; // in ms

    // === Output Results ===
    cout << "=== Sequential Results ===" << endl;
    cout << "Sum     = " << seq_sum << endl;
    cout << "Average = " << seq_avg << endl;
    cout << "Min     = " << seq_min << endl;
    cout << "Max     = " << seq_max << endl;
    cout << "Time    = " << duration_seq << " ms" << endl;

    cout << "\n=== Parallel Results ===" << endl;
    cout << "Sum     = " << par_sum << endl;
    cout << "Average = " << par_avg << endl;
    cout << "Min     = " << par_min << endl;
    cout << "Max     = " << par_max << endl;
    cout << "Time    = " << duration_par << " ms" << endl;

    return 0;
}


// command to run this code: cd "d:\4th year sem 8\HPC\PRACTICALS\" ; if ($?) { g++ -fopenmp Assignment_02.cpp -o Assignment_02 } ; if ($?) { .\Assignment_02 }
// OR
// g++ -fopenmp Assignment_02.cpp -o Assignment_02 && ./Assignment_02 
