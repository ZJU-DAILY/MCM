#ifndef DYNAMIC_ML_CORE_H
#define DYNAMIC_ML_CORE_H

#include "IterativeMLCore.h"
#include <array>
#include <cassert>
#include <climits>
#include <queue>
#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include <smmintrin.h>

struct EdgeTuple {
    uint u, v, l;
    EdgeTuple(uint u_, uint v_, uint l_) : u(u_), v(v_), l(l_) {}
};

struct FailEntry {
    uint node;
    std::vector<uint> scv;
    uint broken_dim;
    uint scv_idx;
    FailEntry() = default;
    FailEntry(uint n, const std::vector<uint>& k, uint d, uint idx = UINT_MAX)
    : node(n), scv(k), broken_dim(d), scv_idx(idx) {}
    FailEntry(uint n, std::vector<uint>&& k, uint d, uint idx = UINT_MAX)
    : node(n), scv(std::move(k)), broken_dim(d), scv_idx(idx) {}
};

class DynamicMLCore : public IterativeMLCore {
public:

    explicit DynamicMLCore(MultilayerGraph& mg_, int num_threads_ = 0);
    virtual ~DynamicMLCore() = default;

    virtual void BatchInsertEdges(const std::vector<std::tuple<uint, uint, uint>>& edges);

    virtual void BatchDeleteEdges(const std::vector<std::tuple<uint, uint, uint>>& edges);

    void RestoreSnapshot(const std::vector<SkylineSet>& snapshot);

    static void AddEdgeToGraph(MultilayerGraph& mg, uint u, uint v, uint l);

    static void RemoveEdgeFromGraph(MultilayerGraph& mg, uint u, uint v, uint l);

    double last_bfs_ms = 0;

protected:

    bool IsSCVFeasible(uint v, const std::vector<uint>& k_vec) const;

    std::vector<uint> TryHIndexCandidate(uint v, const std::vector<uint>& k_vec,
        const NeighborMaxCBuffer& buf) const override;

    virtual void RecoverTrueSCV();

    static void BatchAddEdgesToGraph(MultilayerGraph& mg, uint num_layers,
        const std::vector<EdgeTuple>& edges);

    void BatchComputeCappedUpperBound(
        const std::vector<EdgeTuple>& batch_edges,
        const std::vector<SkylineSet>& old_scv_sets,
        std::vector<std::pair<uint, std::vector<uint>>>& candidate_records);

    void DeletePropagationOnly(
        const std::vector<std::tuple<uint, uint, uint>>& deleted_edges,
        std::vector<char>& needs_check,
        std::vector<char>* affected_domain = nullptr,
        std::vector<std::pair<uint, std::vector<uint>>>* affected_scvs = nullptr);

};

#endif
