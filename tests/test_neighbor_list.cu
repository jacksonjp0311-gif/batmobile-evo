/**
 * Test and benchmark neighbor list construction.
 */

#include "neighbor_list.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <set>

using namespace batteries;

// Generate random positions in a box
void generate_positions(float* h_positions, int N, float box_size, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, box_size);

    for (int i = 0; i < N; i++) {
        h_positions[i * 3 + 0] = dist(gen);
        h_positions[i * 3 + 1] = dist(gen);
        h_positions[i * 3 + 2] = dist(gen);
    }
}

// Brute force neighbor list for verification
void brute_force_neighbors(
    const float* positions,
    float box_size,
    int N,
    float cutoff,
    bool self_loops,
    std::vector<std::pair<int, int>>& edges,
    std::vector<float>& vectors
) {
    float cutoff_sq = cutoff * cutoff;

    for (int i = 0; i < N; i++) {
        float xi = positions[i * 3 + 0];
        float yi = positions[i * 3 + 1];
        float zi = positions[i * 3 + 2];

        for (int j = 0; j < N; j++) {
            if (!self_loops && i == j) continue;

            float dx = positions[j * 3 + 0] - xi;
            float dy = positions[j * 3 + 1] - yi;
            float dz = positions[j * 3 + 2] - zi;

            // Minimum image convention
            dx -= rintf(dx / box_size) * box_size;
            dy -= rintf(dy / box_size) * box_size;
            dz -= rintf(dz / box_size) * box_size;

            float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq <= cutoff_sq) {
                edges.push_back({i, j});
                vectors.push_back(dx);
                vectors.push_back(dy);
                vectors.push_back(dz);
            }
        }
    }
}

bool test_correctness(int N, float density, float cutoff) {
    // Calculate box size from density
    float volume = N / density;
    float box_size = cbrtf(volume);

    printf("Testing N=%d, box=%.2f, cutoff=%.2f, density=%.4f\n",
           N, box_size, cutoff, density);

    // Allocate and generate positions
    std::vector<float> h_positions(N * 3);
    generate_positions(h_positions.data(), N, box_size);

    // Create cell matrix (orthorhombic)
    float h_cell[9] = {0};
    h_cell[0] = box_size;  // cell[0,0]
    h_cell[4] = box_size;  // cell[1,1]
    h_cell[8] = box_size;  // cell[2,2]

    // Copy to device
    float *d_positions, *d_cell;
    cudaMalloc(&d_positions, N * 3 * sizeof(float));
    cudaMalloc(&d_cell, 9 * sizeof(float));
    cudaMemcpy(d_positions, h_positions.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cell, h_cell, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // Run CUDA neighbor list
    NeighborListResult result;
    cudaError_t err = neighbor_list_cuda(d_positions, d_cell, N, cutoff, false, &result);

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return false;
    }

    // Copy results back
    std::vector<int> h_edge_index(2 * result.num_edges);
    std::vector<float> h_edge_vectors(3 * result.num_edges);

    if (result.num_edges > 0) {
        cudaMemcpy(h_edge_index.data(), result.edge_index,
                   2 * result.num_edges * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_edge_vectors.data(), result.edge_vectors,
                   3 * result.num_edges * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Compute reference
    std::vector<std::pair<int, int>> ref_edges;
    std::vector<float> ref_vectors;
    brute_force_neighbors(h_positions.data(), box_size, N, cutoff, false, ref_edges, ref_vectors);

    printf("  CUDA edges: %ld, Reference edges: %zu\n", result.num_edges, ref_edges.size());

    // Check edge count
    if (result.num_edges != (int64_t)ref_edges.size()) {
        printf("  FAIL: Edge count mismatch!\n");

        // Debug: show some edges
        printf("  First 10 CUDA edges: ");
        for (int i = 0; i < std::min(10, (int)result.num_edges); i++) {
            printf("(%d,%d) ", h_edge_index[i], h_edge_index[result.num_edges + i]);
        }
        printf("\n");

        neighbor_list_free(&result);
        cudaFree(d_positions);
        cudaFree(d_cell);
        return false;
    }

    // Build set of CUDA edges for comparison
    std::set<std::pair<int, int>> cuda_edge_set;
    for (int64_t i = 0; i < result.num_edges; i++) {
        int src = h_edge_index[i];
        int dst = h_edge_index[result.num_edges + i];
        cuda_edge_set.insert({src, dst});
    }

    // Check all reference edges are present
    int missing = 0;
    for (const auto& edge : ref_edges) {
        if (cuda_edge_set.find(edge) == cuda_edge_set.end()) {
            missing++;
            if (missing <= 5) {
                printf("  Missing edge: (%d, %d)\n", edge.first, edge.second);
            }
        }
    }

    if (missing > 0) {
        printf("  FAIL: %d missing edges\n", missing);
        neighbor_list_free(&result);
        cudaFree(d_positions);
        cudaFree(d_cell);
        return false;
    }

    // Check edge vectors (sample a few)
    float max_error = 0.0f;
    for (int64_t i = 0; i < std::min((int64_t)100, result.num_edges); i++) {
        int src = h_edge_index[i];
        int dst = h_edge_index[result.num_edges + i];

        // Find this edge in reference
        for (size_t k = 0; k < ref_edges.size(); k++) {
            if (ref_edges[k].first == src && ref_edges[k].second == dst) {
                float dx = h_edge_vectors[i * 3 + 0] - ref_vectors[k * 3 + 0];
                float dy = h_edge_vectors[i * 3 + 1] - ref_vectors[k * 3 + 1];
                float dz = h_edge_vectors[i * 3 + 2] - ref_vectors[k * 3 + 2];
                float err = sqrtf(dx * dx + dy * dy + dz * dz);
                max_error = std::max(max_error, err);
                break;
            }
        }
    }

    printf("  Max vector error: %.2e\n", max_error);

    bool pass = max_error < 1e-5f;
    printf("  %s\n", pass ? "PASS" : "FAIL");

    // Stats
    float avg_neighbors = (float)result.num_edges / N;
    printf("  Avg neighbors per atom: %.1f\n", avg_neighbors);

    neighbor_list_free(&result);
    cudaFree(d_positions);
    cudaFree(d_cell);

    return pass;
}

float benchmark_cuda(int N, float density, float cutoff, int warmup = 5, int iterations = 20) {
    float volume = N / density;
    float box_size = cbrtf(volume);

    std::vector<float> h_positions(N * 3);
    generate_positions(h_positions.data(), N, box_size);

    float h_cell[9] = {0};
    h_cell[0] = h_cell[4] = h_cell[8] = box_size;

    float *d_positions, *d_cell;
    cudaMalloc(&d_positions, N * 3 * sizeof(float));
    cudaMalloc(&d_cell, 9 * sizeof(float));
    cudaMemcpy(d_positions, h_positions.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cell, h_cell, 9 * sizeof(float), cudaMemcpyHostToDevice);

    NeighborListResult result;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        neighbor_list_cuda(d_positions, d_cell, N, cutoff, false, &result);
        neighbor_list_free(&result);
    }

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int64_t total_edges = 0;

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        neighbor_list_cuda(d_positions, d_cell, N, cutoff, false, &result);
        total_edges = result.num_edges;
        neighbor_list_free(&result);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_positions);
    cudaFree(d_cell);

    float avg_ms = ms / iterations;
    float avg_neighbors = (float)total_edges / N;

    printf("N=%6d: %.4f ms, %ld edges (%.1f neighbors/atom)\n",
           N, avg_ms, total_edges, avg_neighbors);

    return avg_ms;
}

int main() {
    printf("=== Neighbor List Correctness Tests ===\n\n");

    bool all_pass = true;

    // Small test for debugging
    all_pass &= test_correctness(100, 0.05f, 5.0f);
    all_pass &= test_correctness(500, 0.05f, 5.0f);
    all_pass &= test_correctness(1000, 0.05f, 5.0f);

    if (!all_pass) {
        printf("\nCorrectness tests failed!\n");
        return 1;
    }

    printf("\n=== Neighbor List Benchmarks (RTX 3090) ===\n");
    printf("Density: 0.05 atoms/A^3, Cutoff: 5.0 A\n\n");

    benchmark_cuda(1000, 0.05f, 5.0f);
    benchmark_cuda(10000, 0.05f, 5.0f);
    benchmark_cuda(50000, 0.05f, 5.0f);
    benchmark_cuda(100000, 0.05f, 5.0f);

    return 0;
}
