#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>

#include "DynamicMLCore.h"
#include "baseline_io.h"

using namespace std;

static double display_ms(double ms) {
    return round(ms * 10.0) / 10.0;
}

static double displayed_total_ms(double propagation_ms, double iteration_ms) {
    return display_ms(propagation_ms) + display_ms(iteration_ms);
}

static bool load_metric_int(const string& metrics_file,
                            const string& key,
                            int& value) {
    ifstream in(metrics_file);
    string line;
    const string prefix = key + ":";
    while (getline(in, line)) {
        if (line.rfind(prefix, 0) == 0) {
            value = atoi(line.substr(prefix.size()).c_str());
            return true;
        }
    }
    return false;
}

static bool restore_prepared_or_snapshot(DynamicMLCore& solver,
                                         const vector<SkylineSet>& scvs,
                                         const string& prepared_file,
                                         const string& label) {
    (void)label;
    if (solver.RestorePreparedSnapshotForIndependentRun(scvs, prepared_file)) {
        return true;
    }
    solver.RestoreSnapshotForIndependentRun(scvs);
    return false;
}

static void print_usage() {
    cerr << "Usage: ./run_test <dataset_path> [num_threads=32]\n";
}

template <typename Fn>
static void run_silencing_cerr(Fn&& fn) {
    ofstream devnull("/dev/null");
    streambuf* old_cerr = cerr.rdbuf();
    if (devnull) cerr.rdbuf(devnull.rdbuf());
    fn();
    cerr.rdbuf(old_cerr);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    string dataset_path = argv[1];
    int num_threads = (argc >= 3) ? atoi(argv[2]) : 32;
    if (num_threads <= 0) num_threads = 32;

    string dataset_name = extract_dataset_name(dataset_path);
    string prefix = "baseline/" + dataset_name;
    string metrics_file = prefix + "_metrics.txt";

    MultilayerGraph mg;
    mg.LoadFromFile(dataset_path);
    uint n = mg.GetN();
    uint L = mg.GetLayerNumber();
    omp_set_num_threads(num_threads);

    int num_edges = 0;
    if (!load_metric_int(metrics_file, "num_edges", num_edges)) {
        cerr << "Failed to load baseline metrics. Run ./run_baseline first.\n";
        return 1;
    }

    vector<SkylineSet> peeling_scvs = load_scvs(prefix + "_peeling.bin", n, L);
    vector<SkylineSet> after_insert_scvs =
        load_scvs(prefix + "_after_insert.bin", n, L);
    vector<tuple<uint, uint, uint>> insert_edges =
        load_edges(prefix + "_insert.bin");
    vector<tuple<uint, uint, uint>> delete_edges =
        load_edges(prefix + "_delete.bin");

    if (peeling_scvs.empty() || after_insert_scvs.empty() ||
        insert_edges.empty() ||
        delete_edges.empty()) {
        cerr << "Failed to load maintenance inputs. Run ./run_baseline first.\n";
        return 1;
    }

    cout << "Dataset: " << dataset_name << " | Nodes: " << n
         << " | Layers: " << L << " | Updated Edges: " << num_edges
         << " | Threads: " << num_threads << "\n";

    double insert_propagation_ms = 0.0;
    double insert_iteration_ms = 0.0;
    {
        MultilayerGraph mg_insert;
        mg_insert.LoadFromFile(dataset_path);
        DynamicMLCore solver(mg_insert, num_threads);
        restore_prepared_or_snapshot(solver, peeling_scvs,
                                     prefix + "_peeling.prepared.bin",
                                     "peeling");
        run_silencing_cerr([&]() {
            solver.BatchInsertEdges(insert_edges);
        });
        insert_propagation_ms = solver.last_bfs_ms;
        insert_iteration_ms = solver.last_iter_ms;
    }

    double delete_propagation_ms = 0.0;
    double delete_iteration_ms = 0.0;
    {
        MultilayerGraph mg_delete;
        mg_delete.LoadFromFile(dataset_path);
        for (const auto& e : insert_edges) {
            DynamicMLCore::AddEdgeToGraph(mg_delete, get<0>(e), get<1>(e),
                                          get<2>(e));
        }
        DynamicMLCore solver(mg_delete, num_threads);
        restore_prepared_or_snapshot(solver, after_insert_scvs,
                                     prefix + "_after_insert.prepared.bin",
                                     "after_insert");
        run_silencing_cerr([&]() {
            solver.BatchDeleteEdges(delete_edges);
        });
        delete_propagation_ms = solver.last_bfs_ms;
        delete_iteration_ms = solver.last_iter_ms;
    }

    cout << "Insert time (x" << num_edges << "): "
         << displayed_total_ms(insert_propagation_ms, insert_iteration_ms)
         << " ms\n";
    cout << "Delete time (x" << num_edges << "): "
         << displayed_total_ms(delete_propagation_ms, delete_iteration_ms)
         << " ms\n";

    return 0;
}
