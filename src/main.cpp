#include <algorithm>
#include <chrono>
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

using Clock = chrono::steady_clock;

static double elapsed_ms(Clock::time_point start) {
    return chrono::duration<double, milli>(Clock::now() - start).count();
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

static void print_usage() {
    cerr << "Usage: ./run_test <dataset_path> [num_threads=32]\n";
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        print_usage();
        return 1;
    }

    string dataset_path = argv[1];
    int num_threads = (argc >= 3) ? atoi(argv[2]) : 32;
    if (num_threads <= 0) num_threads = 32;

    int num_edges = 0;
    {
        string dataset_name = extract_dataset_name(dataset_path);
        string prefix = "baseline/" + dataset_name;
        string metrics_file = prefix + "_metrics.txt";

        MultilayerGraph mg;
        mg.LoadFromFile(dataset_path);
        uint n = mg.GetN();
        uint L = mg.GetLayerNumber();
        omp_set_num_threads(num_threads);

        if (!load_metric_int(metrics_file, "num_edges", num_edges)) {
            cerr << "Failed to load baseline metrics. Run ./run_baseline first.\n";
            return 1;
        }

        auto stage_start = Clock::now();
        vector<SkylineSet> peeling_scvs = load_scvs(prefix + "_peeling.bin", n, L);
        const double insert_snapshot_loading_ms = elapsed_ms(stage_start);
        stage_start = Clock::now();
        vector<SkylineSet> after_insert_scvs =
        load_scvs(prefix + "_after_insert.bin", n, L);
        const double delete_snapshot_loading_ms = elapsed_ms(stage_start);
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

        cout << fixed << setprecision(3);
        cout << "Dataset: " << dataset_name << " | Nodes: " << n
        << " | Layers: " << L << " | Updated Edges: " << num_edges
        << " | Threads: " << num_threads << "\n";

        {
            MultilayerGraph mg_insert;
            stage_start = Clock::now();
            mg_insert.LoadFromFile(dataset_path);
            const double graph_loading_ms = elapsed_ms(stage_start);
            cout << "Graph data loading time: " << graph_loading_ms << " ms\n";

            stage_start = Clock::now();
            DynamicMLCore solver(mg_insert, num_threads);

            solver.RestoreSnapshot(peeling_scvs);
            const double mcv_initialization_ms = insert_snapshot_loading_ms
                + elapsed_ms(stage_start);
            cout << "MCV initialization time: " << mcv_initialization_ms << " ms\n";

            solver.BatchInsertEdges(insert_edges);
            cout << "Insertion maintenance time (x" << num_edges << "): "
            << solver.last_bfs_ms + solver.last_iter_ms << " ms\n\n";
        }

        {
            MultilayerGraph mg_delete;
            stage_start = Clock::now();
            mg_delete.LoadFromFile(dataset_path);
            const double graph_loading_ms = elapsed_ms(stage_start);
            cout << "Graph data loading time: " << graph_loading_ms << " ms\n";
            for (const auto& e : insert_edges) {
                DynamicMLCore::AddEdgeToGraph(mg_delete, get<0>(e), get<1>(e),
                    get<2>(e));
            }
            stage_start = Clock::now();
            DynamicMLCore solver(mg_delete, num_threads);

            solver.RestoreSnapshot(after_insert_scvs);
            const double mcv_initialization_ms = delete_snapshot_loading_ms
                + elapsed_ms(stage_start);
            cout << "MCV initialization time: " << mcv_initialization_ms << " ms\n";

            solver.BatchDeleteEdges(delete_edges);
            cout << "Deletion maintenance time (x" << num_edges << "): "
            << solver.last_bfs_ms + solver.last_iter_ms << " ms\n";
        }
    }

    return 0;
}
