#ifndef BASELINE_IO_H
#define BASELINE_IO_H

#include "SkylineSet.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <sys/stat.h>


inline std::string extract_dataset_name(const std::string& path) {
    std::string p = path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    auto pos = p.rfind('/');
    return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

inline bool dir_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline void ensure_dir(const std::string& path) {
    if (!dir_exists(path)) {
        mkdir(path.c_str(), 0755);
    }
}





inline void save_scvs(const std::string& filename,
                      const std::vector<SkylineSet>& scvs, uint n, uint L) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) { std::cerr << "Cannot open " << filename << " for writing\n"; return; }
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint));
    out.write(reinterpret_cast<const char*>(&L), sizeof(uint));
    for (uint v = 0; v < n; v++) {
        const auto& s = scvs[v].GetSCVs();
        uint num = static_cast<uint>(s.size());
        out.write(reinterpret_cast<const char*>(&num), sizeof(uint));
        for (const auto& scv : s) {
            out.write(reinterpret_cast<const char*>(scv.data()), L * sizeof(uint));
        }
    }
}

inline std::vector<SkylineSet> load_scvs(const std::string& filename,
                                         uint n_expected, uint L_expected) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << " for reading\n";
        return {};
    }
    uint n, L;
    in.read(reinterpret_cast<char*>(&n), sizeof(uint));
    in.read(reinterpret_cast<char*>(&L), sizeof(uint));
    if (n != n_expected || L != L_expected) {
        std::cerr << "SCV file dimension mismatch: expected (" << n_expected << "," << L_expected
                  << ") got (" << n << "," << L << ")\n";
        return {};
    }
    std::vector<SkylineSet> scvs(n);
    for (uint v = 0; v < n; v++) {
        uint num;
        in.read(reinterpret_cast<char*>(&num), sizeof(uint));
        for (uint i = 0; i < num; i++) {
            std::vector<uint> scv(L);
            in.read(reinterpret_cast<char*>(scv.data()), L * sizeof(uint));
            scvs[v].Insert(scv);
        }
    }
    return scvs;
}




inline void save_edges(const std::string& filename,
                       const std::vector<std::tuple<uint,uint,uint>>& edges) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) { std::cerr << "Cannot open " << filename << " for writing\n"; return; }
    uint num = static_cast<uint>(edges.size());
    out.write(reinterpret_cast<const char*>(&num), sizeof(uint));
    for (const auto& e : edges) {
        uint arr[3] = {std::get<0>(e), std::get<1>(e), std::get<2>(e)};
        out.write(reinterpret_cast<const char*>(arr), 3 * sizeof(uint));
    }
}

inline std::vector<std::tuple<uint,uint,uint>> load_edges(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << " for reading\n";
        return {};
    }
    uint num;
    in.read(reinterpret_cast<char*>(&num), sizeof(uint));
    std::vector<std::tuple<uint,uint,uint>> edges(num);
    for (uint i = 0; i < num; i++) {
        uint arr[3];
        in.read(reinterpret_cast<char*>(arr), 3 * sizeof(uint));
        edges[i] = std::make_tuple(arr[0], arr[1], arr[2]);
    }
    return edges;
}


inline void extract_peeling_scvs(MLCTree& tree, std::vector<SkylineSet>& out_scvs,
                                 uint n, uint L) {
    std::queue<Node*> q;
    if (tree.GetRoot()) q.push(tree.GetRoot());
    while (!q.empty()) {
        Node* curr = q.front(); q.pop();
        uint* k_ptr = MLCTree::GetK(curr);
        std::vector<uint> k_vec(L);
        for (uint l = 0; l < L; l++) k_vec[l] = k_ptr[l];
        Diff* diff = tree.GetDiff(curr);
        if (diff && diff->num > 0)
            for (uint j = 0; j < diff->num; j++)
                out_scvs[diff->vtx_ptr[j]].Insert(k_vec);
        uint inc_k = tree.GetIncK(curr);
        for (uint i = inc_k; i < L; i++) {
            Node* chd = tree.GetRelChd(curr, i);
            if (chd) q.push(chd);
        }
    }
    
    std::vector<uint> zero_k(L, 0);
    for (uint v = 0; v < n; v++) out_scvs[v].Insert(zero_k);
}

#endif 
