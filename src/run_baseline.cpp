#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>

#include "../MlcDec/Core/MLCTree.h"
#include "../MlcDec/Core/PMLCTreeBuilder.h"
#include "../MlcDec/Graphs/MultilayerGraph.h"
#include "DynamicMLCore.h"
#include "baseline_io.h"

using namespace std;

using Clock = chrono::steady_clock;

static double elapsed_ms(Clock::time_point start) {
    return chrono::duration<double, milli>(Clock::now() - start).count();
}

static tuple<uint, uint, uint> canonical_edge(uint u, uint v, uint l) {
    return make_tuple(min(u, v), max(u, v), l);
}

static bool contains_edge_key(const vector<tuple<uint, uint, uint>>& edges,
    const tuple<uint, uint, uint>& key) {
    for (const auto& e : edges) {
        if (canonical_edge(get<0>(e), get<1>(e), get<2>(e)) == key) {
            return true;
        }
    }
    return false;
}

static bool edge_exists(MultilayerGraph& mg, uint u, uint v, uint l) {
    uint** adj = mg.GetGraph(l).GetAdjLst();
    for (uint i = 1; i <= adj[u][0]; ++i) {
        if (adj[u][i] == v) return true;
    }
    return false;
}

static vector<tuple<uint, uint, uint>> sample_random_insert_edges(
    MultilayerGraph& mg,
    uint n,
    uint L,
    int num_edges) {
    mt19937 rng(123);
    vector<tuple<uint, uint, uint>> edges;
    while (static_cast<int>(edges.size()) < num_edges) {
        uint u = uniform_int_distribution<uint>(0, n - 1)(rng);
        uint v = uniform_int_distribution<uint>(0, n - 1)(rng);
        uint l = uniform_int_distribution<uint>(0, L - 1)(rng);
        if (u == v) continue;
        if (edge_exists(mg, u, v, l)) continue;
        auto key = canonical_edge(u, v, l);
        if (!contains_edge_key(edges, key)) edges.emplace_back(u, v, l);
    }
    return edges;
}

static vector<SkylineSet> run_peeling(MultilayerGraph& mg,
    uint n,
    uint L,
    double& computation_ms) {
    const auto start = Clock::now();
    vector<SkylineSet> scvs(n);
    {
        MLCTree tree(L, n);
        PMLCTreeBuilder::PRun_async_with_merge(mg, tree, 2, 0.1f);
        extract_peeling_scvs(tree, scvs, n, L);
    }
    computation_ms = elapsed_ms(start);
    return scvs;
}

static void print_usage() {
    cerr << "Usage: ./run_baseline <dataset_path> [num_edges=1000] "
    "[num_threads=32]\n";
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        print_usage();
        return 1;
    }

    string dataset_path = argv[1];
    int num_edges = (argc >= 3) ? atoi(argv[2]) : 1000;
    int num_threads = (argc >= 4) ? atoi(argv[3]) : 32;
    if (num_edges <= 0) num_edges = 1000;
    if (num_threads <= 0) num_threads = 32;

    const auto total_start = Clock::now();
    string dataset_name = extract_dataset_name(dataset_path);
    string baseline_dir = "baseline";
    ensure_dir(baseline_dir);
    string prefix = baseline_dir + "/" + dataset_name;

    MultilayerGraph mg;
    const auto graph_start = Clock::now();
    mg.LoadFromFile(dataset_path);
    const double graph_loading_ms = elapsed_ms(graph_start);
    uint L = mg.GetLayerNumber();
    uint n = mg.GetN();
    omp_set_num_threads(num_threads);

    cout << fixed << setprecision(3);
    cout << "=== Baseline Generation ===\n";
    cout << "Dataset: " << dataset_name << " | Nodes: " << n
    << " | Layers: " << L << " | Updated Edges: " << num_edges
    << " | Threads: " << num_threads << "\n\n";

    cout << "Graph data loading time: " << graph_loading_ms << " ms\n\n";

    cout << "[1/3] Full-graph MCV initialization (decomposition)...\n";
    double peeling_ms = 0.0;
    vector<SkylineSet> peeling_scvs = run_peeling(mg, n, L, peeling_ms);
    save_scvs(prefix + "_peeling.bin", peeling_scvs, n, L);
    cout << "  Time: " << peeling_ms << " ms\n";

    vector<tuple<uint, uint, uint>> insert_edges =
    sample_random_insert_edges(mg, n, L, num_edges);
    save_edges(prefix + "_insert.bin", insert_edges);

    for (const auto& e : insert_edges) {
        DynamicMLCore::AddEdgeToGraph(mg, get<0>(e), get<1>(e), get<2>(e));
    }

    cout << "[2/3] Full-graph MCV recomputation after insert (decomposition)...\n";
    double after_insert_peeling_ms = 0.0;
    vector<SkylineSet> after_insert_scvs =
    run_peeling(mg, n, L, after_insert_peeling_ms);
    save_scvs(prefix + "_after_insert.bin", after_insert_scvs, n, L);
    cout << "  Time: " << after_insert_peeling_ms << " ms\n";

    vector<tuple<uint, uint, uint>> delete_edges = insert_edges;
    save_edges(prefix + "_delete.bin", delete_edges);

    for (const auto& e : delete_edges) {
        DynamicMLCore::RemoveEdgeFromGraph(mg, get<0>(e), get<1>(e),
            get<2>(e));
    }

    cout << "[3/3] Full-graph MCV recomputation after delete (decomposition)...\n";
    double after_delete_peeling_ms = 0.0;
    vector<SkylineSet> after_delete_scvs =
    run_peeling(mg, n, L, after_delete_peeling_ms);
    save_scvs(prefix + "_after_delete.bin", after_delete_scvs, n, L);
    cout << "  Time: " << after_delete_peeling_ms << " ms\n";

    ofstream metrics(prefix + "_metrics.txt");
    metrics << "dataset: " << dataset_name << "\n";
    metrics << "nodes: " << n << "\n";
    metrics << "layers: " << L << "\n";
    metrics << "num_edges: " << num_edges << "\n";
    metrics << "num_threads: " << num_threads << "\n";
    metrics << "mode: batch\n";
    metrics << "insert_pattern: insert_delete_same\n";
    metrics << "update_order: insert_then_delete\n";
    metrics << fixed << setprecision(3);
    metrics << "graph_loading_ms: " << graph_loading_ms << "\n";
    metrics << "peeling_ms: " << peeling_ms << "\n";
    metrics << "after_insert_peeling_ms: " << after_insert_peeling_ms << "\n";
    metrics << "after_delete_peeling_ms: " << after_delete_peeling_ms << "\n";
    metrics.close();

    cout << "\nBaseline saved to " << baseline_dir << "/" << dataset_name
    << "_*\n";
    return 0;
}
