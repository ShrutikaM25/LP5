#include <iostream>
#include <vector>
#include <string>
#include <omp.h>

// Define the Employee structure, representing a row in our "Employees" table
struct Employee {
    int id;
    std::string name;
    int age;
};

// Simulate a simple "Employees" table with some sample data
std::vector<Employee> table = {
    {1, "Alice", 30},
    {2, "Bob", 24},
    {3, "Charlie", 29},
    {4, "David", 35},
    {5, "Eve", 28}
};

// Sequential filter query to select employees older than minAge
std::vector<Employee> filterQuery(int minAge) {
    std::vector<Employee> result;
    for (const auto& emp : table) {
        if (emp.age >= minAge) {
            result.push_back(emp);
        }
    }
    return result;
}

// Parallel filter query using OpenMP
std::vector<Employee> filterQueryParallel(int minAge) {
    std::vector<Employee> result;
    
    #pragma omp parallel for
    for (int i = 0; i < table.size(); ++i) {
        if (table[i].age >= minAge) {
            #pragma omp critical
            result.push_back(table[i]);
        }
    }
    return result;
}

// Parallel join query: Joining two tables (cross-product) using OpenMP
std::vector<std::pair<Employee, Employee>> joinQueryParallel(const std::vector<Employee>& table1, const std::vector<Employee>& table2) {
    std::vector<std::pair<Employee, Employee>> result;

    #pragma omp parallel for
    for (int i = 0; i < table1.size(); ++i) {
        for (int j = 0; j < table2.size(); ++j) {
            #pragma omp critical
            result.push_back(std::make_pair(table1[i], table2[j]));
        }
    }

    return result;
}

// Optimized query that applies filter before join (filter pushdown optimization)
std::vector<std::pair<Employee, Employee>> filterAndJoinQueryParallel(const std::vector<Employee>& table1, const std::vector<Employee>& table2, int minAge) {
    // First, apply the filter to table1 before performing join
    std::vector<Employee> filteredTable;
    
    #pragma omp parallel for
    for (int i = 0; i < table1.size(); ++i) {
        if (table1[i].age >= minAge) {
            #pragma omp critical
            filteredTable.push_back(table1[i]);
        }
    }

    // Now perform the join on the filtered table
    std::vector<std::pair<Employee, Employee>> result;
    #pragma omp parallel for
    for (int i = 0; i < filteredTable.size(); ++i) {
        for (int j = 0; j < table2.size(); ++j) {
            #pragma omp critical
            result.push_back(std::make_pair(filteredTable[i], table2[j]));
        }
    }

    return result;
}

// Main function to test and compare sequential vs parallel queries
int main() {
    int minAge = 30;  // Define a minimum age for filtering
    
    // Sequential Query
    double start = omp_get_wtime();
    auto resultSeq = filterQuery(minAge);
    double end = omp_get_wtime();
    std::cout << "Sequential Query Time: " << (end - start) << " seconds\n";
    std::cout << "Result of Sequential Query (filtering employees older than " << minAge << "):\n";
    for (const auto& emp : resultSeq) {
        std::cout << emp.name << ", Age: " << emp.age << "\n";
    }

    // Parallel Query
    start = omp_get_wtime();
    auto resultPar = filterQueryParallel(minAge);
    end = omp_get_wtime();
    std::cout << "Parallel Query Time: " << (end - start) << " seconds\n";
    std::cout << "Result of Parallel Query (filtering employees older than " << minAge << "):\n";
    for (const auto& emp : resultPar) {
        std::cout << emp.name << ", Age: " << emp.age << "\n";
    }

    // Join Query: Performing join between two tables (employees joining with themselves)
    start = omp_get_wtime();
    auto resultJoin = joinQueryParallel(table, table);
    end = omp_get_wtime();
    std::cout << "Parallel Join Query Time: " << (end - start) << " seconds\n";
    std::cout << "Result of Parallel Join (Employees joined with themselves):\n";
    for (const auto& pair : resultJoin) {
        std::cout << pair.first.name << " (ID: " << pair.first.id << ") joined with "
                  << pair.second.name << " (ID: " << pair.second.id << ")\n";
    }

    // Filter and Join Query (Optimization: Filter before join)
    start = omp_get_wtime();
    auto resultFilterJoin = filterAndJoinQueryParallel(table, table, minAge);
    end = omp_get_wtime();
    std::cout << "Filter and Join Query Time (with Filter Pushdown): " << (end - start) << " seconds\n";
    std::cout << "Result of Filter and Join (Employees older than " << minAge << " joined with themselves):\n";
    for (const auto& pair : resultFilterJoin) {
        std::cout << pair.first.name << " (ID: " << pair.first.id << ") joined with "
                  << pair.second.name << " (ID: " << pair.second.id << ")\n";
    }

    return 0;
}

// command to run this code: cd "d:\4th year sem 8\HPC\PRACTICALS\" ; if ($?) { g++ -fopenmp Assignment_02.cpp -o Assignment_02 } ; if ($?) { .\Assignment_02 }
// OR
// g++ -fopenmp Assignment_02.cpp -o Assignment_02 && ./Assignment_02 
