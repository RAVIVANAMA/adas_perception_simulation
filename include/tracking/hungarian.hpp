#pragma once
// Hungarian algorithm for optimal bipartite assignment.
// Solves min-cost perfect matching on an NxM cost matrix.
// Returns assignment[i] = j or -1 if row i is unmatched.
#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>

namespace adas::tracking {

inline std::vector<int>
solveHungarian(const std::vector<std::vector<double>>& costMatrix) {
    int rows = static_cast<int>(costMatrix.size());
    if (rows == 0) return {};
    int cols = static_cast<int>(costMatrix[0].size());
    int n    = std::max(rows, cols);

    // Pad to square
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 1e18));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            C[r][c] = costMatrix[r][c];

    // Munkres / Hungarian implementation (O(n³))
    std::vector<double> u(n+1, 0.0), v(n+1, 0.0);
    std::vector<int>    p(n+1, 0), way(n+1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n+1,  1e18);
        std::vector<bool>   used(n+1, false);

        do {
            used[j0] = true;
            int i0 = p[j0], j1 = -1;
            double delta = 1e18;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    double cur = C[i0-1][j-1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j]  = j0;
                    }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0]  = p[j1];
            j0     = j1;
        } while (j0);
    }

    std::vector<int> assignment(rows, -1);
    for (int j = 1; j <= cols; ++j) {
        if (p[j] > 0 && p[j] <= rows)
            assignment[p[j]-1] = j - 1;
    }
    return assignment;
}

} // namespace adas::tracking
