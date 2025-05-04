#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <cstdlib>
using namespace std;

class Graph {
public:
    int V; // Number of vertices
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected
    }

    // === Sequential BFS ===
    void BFS_Sequential(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(start);
        visited[start] = true;

        cout << "Sequential BFS: ";
        while (!q.empty()) {
            int u = q.front(); q.pop();
            cout << u << " ";

            for (int v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
        cout << endl;
    }

    // === Parallel BFS ===
    void BFS_Parallel(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(start);
        visited[start] = true;

        cout << "Parallel BFS: ";
        while (!q.empty()) {
            int size = q.size();

            // Process current level in parallel
            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                int u;
                #pragma omp critical
                {
                    if (!q.empty()) {
                        u = q.front();
                        q.pop();
                        cout << u << " ";
                    }
                }

                #pragma omp parallel for
                for (int j = 0; j < adj[u].size(); ++j) {
                    int v = adj[u][j];
                    if (!visited[v]) {
                        #pragma omp critical
                        {
                            if (!visited[v]) {
                                visited[v] = true;
                                q.push(v);
                            }
                        }
                    }
                }
            }
        }
        cout << endl;
    }

    // === Sequential DFS ===
    void DFS_Sequential(int u, vector<bool>& visited) {
        visited[u] = true;
        cout << u << " ";
        for (int v : adj[u]) {
            if (!visited[v]) {
                DFS_Sequential(v, visited);
            }
        }
    }

    // === Parallel DFS ===
    void DFS_Parallel(int u, vector<bool>& visited) {
        #pragma omp critical
        {
            if (visited[u]) return;
            visited[u] = true;
            cout << u << " ";
        }

        #pragma omp parallel for
        for (int i = 0; i < adj[u].size(); ++i) {
            int v = adj[u][i];
            if (!visited[v]) {
                #pragma omp task
                DFS_Parallel(v, visited);
            }
        }
    }

    void runDFS_Sequential(int start) {
        vector<bool> visited(V, false);
        cout << "Sequential DFS: ";
        DFS_Sequential(start, visited);
        cout << endl;
    }

    void runDFS_Parallel(int start) {
        vector<bool> visited(V, false);
        cout << "Parallel DFS: ";
        #pragma omp parallel
        {
            #pragma omp single
            DFS_Parallel(start, visited);
        }
        cout << endl;
    }
};

int main() {
    Graph g(8);

    // Create sample undirected graph
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);
    g.addEdge(6, 7);

    // int V, E;
    // cout << "Enter number of vertices: ";
    // cin >> V;
    // Graph g(V);

    // cout << "Enter number of edges: ";
    // cin >> E;

    // cout << "Enter " << E << " edges (u v):\n";
    // for (int i = 0; i < E; ++i) {
    //     int u, v;
    //     cin >> u >> v;
    //     g.addEdge(u, v);
    // }


    double t1, t2;

    // BFS Sequential
    t1 = omp_get_wtime();
    g.BFS_Sequential(0);
    t2 = omp_get_wtime();
    cout << "Time (Sequential BFS): " << (t2 - t1) * 1000 << " ms\n\n";

    // BFS Parallel
    t1 = omp_get_wtime();
    g.BFS_Parallel(0);
    t2 = omp_get_wtime();
    cout << "Time (Parallel BFS): " << (t2 - t1) * 1000 << " ms\n\n";

    // DFS Sequential
    t1 = omp_get_wtime();
    g.runDFS_Sequential(0);
    t2 = omp_get_wtime();
    cout << "Time (Sequential DFS): " << (t2 - t1) * 1000 << " ms\n\n";

    // DFS Parallel
    t1 = omp_get_wtime();
    g.runDFS_Parallel(0);
    t2 = omp_get_wtime();
    cout << "Time (Parallel DFS): " << (t2 - t1) * 1000 << " ms\n\n";

    return 0;
}
