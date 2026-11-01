#ifndef ITERATIVE_ML_CORE_H
#define ITERATIVE_ML_CORE_H

#include "../MlcDec/Graphs/MultilayerGraph.h"
#include "SkylineSet.h"
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <cassert>
#include <climits>
#include <cstddef>

struct SCVSnapshot {
    const SkylineSet* view = nullptr;

    void Clear() {
        view = nullptr;
    }
    bool Empty() const { return view == nullptr || view->NumSCVs() == 0; }
    uint NumSCVs(uint L) const {
        return L == 0 || view == nullptr ? 0 : view->NumSCVs();
    }
    void SetView(const SkylineSet& set) {
        view = &set;
    }
    const uint* GetRaw(uint idx, uint L) const {
        (void)L;
        return view->GetSCVFlat(idx);
    }
};

struct NeighborSnapshot {
    std::vector<uint> nodes;
    std::vector<SCVSnapshot> snapshots;
    std::vector<uint> index_keys;
    std::vector<uint> index_pos;
    SCVSnapshot empty;
    static constexpr uint kEmptyKey = UINT_MAX;

    void Clear() {
        nodes.clear();
        snapshots.clear();
        index_keys.clear();
        index_pos.clear();
        empty.Clear();
    }

    void Reserve(size_t count) {
        nodes.reserve(count);
        snapshots.reserve(count);
        size_t cap = 1;
        size_t needed = std::max<size_t>(count * 4, 8);
        while (cap < needed) cap <<= 1;
        index_keys.assign(cap, kEmptyKey);
        index_pos.assign(cap, 0);
    }

    void RebuildIndex(size_t new_cap) {
        std::vector<uint> old_nodes = nodes;
        size_t cap = 1;
        while (cap < std::max<size_t>(new_cap, 8)) cap <<= 1;
        index_keys.assign(cap, kEmptyKey);
        index_pos.assign(cap, 0);
        for (uint pos = 0; pos < old_nodes.size(); pos++) {
            InsertIndex(old_nodes[pos], pos);
        }
    }

    void InsertIndex(uint node, uint pos) {
        if (index_keys.empty()) Reserve(8);
        size_t mask = index_keys.size() - 1;
        size_t slot = static_cast<size_t>(node) & mask;
        while (index_keys[slot] != kEmptyKey) {
            slot = (slot + 1) & mask;
        }
        index_keys[slot] = node;
        index_pos[slot] = pos;
    }

    uint FindIndex(uint node) const {
        if (index_keys.empty()) return UINT_MAX;
        size_t mask = index_keys.size() - 1;
        size_t slot = static_cast<size_t>(node) & mask;
        while (true) {
            uint key = index_keys[slot];
            if (key == node) return index_pos[slot];
            if (key == kEmptyKey) return UINT_MAX;
            slot = (slot + 1) & mask;
        }
    }

    SCVSnapshot& GetOrCreate(uint node) {
        uint found = FindIndex(node);
        if (found != UINT_MAX) return snapshots[found];
        if ((nodes.size() + 1) * 2 >= index_keys.size()) {
            RebuildIndex(index_keys.empty() ? 16 : index_keys.size() * 2);
        }
        uint pos = static_cast<uint>(snapshots.size());
        InsertIndex(node, pos);
        nodes.push_back(node);
        snapshots.emplace_back();
        return snapshots.back();
    }

    SCVSnapshot& operator[](uint node) {
        return GetOrCreate(node);
    }

    const SCVSnapshot& operator[](uint node) const {
        uint found = FindIndex(node);
        if (found == UINT_MAX) return empty;
        return snapshots[found];
    }
};

struct LazyVerifyState {

    struct NeighborInfo {
        bool is_supporter = false;

        uint support_scv_idx = UINT_MAX;
    };

    struct LayerState {
        std::vector<uint> support_nodes;
        std::vector<uint> fail_nodes;
        std::vector<uint> candidate_nodes;
        std::vector<NeighborInfo> neighbor_info;

        void ClearInfo(uint pos) {
            neighbor_info[pos] = NeighborInfo();

        }
        uint support_count = 0;
        bool initialized = false;

        NeighborInfo& GetInfo(uint pos) {
            if (neighbor_info.size() <= pos) neighbor_info.resize(pos + 1);
            return neighbor_info[pos];
        }

    };

    std::vector<LayerState> layers;

    explicit LazyVerifyState(uint num_layers) : layers(num_layers) {}

    void Clear() {
        for (auto& ls : layers) {
            for (uint pos : ls.support_nodes) ls.ClearInfo(pos);
            for (uint pos : ls.fail_nodes) ls.ClearInfo(pos);
            for (uint pos : ls.candidate_nodes) ls.ClearInfo(pos);

            ls.support_nodes.clear();
            ls.fail_nodes.clear();
            ls.candidate_nodes.clear();
            ls.support_count = 0;
            ls.initialized = false;
        }
    }
};

struct CandidateHash {
    size_t operator()(const std::vector<uint>& v) const {
        size_t h = 1469598103934665603ull;
        for (uint x : v) {
            h ^= static_cast<size_t>(x) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        }
        return h;
    }
};

using CandidateSet = std::unordered_set<std::vector<uint>, CandidateHash>;

struct CandidateTracker {
    CandidateSet entries;

    void Reserve(size_t n) {
        entries.reserve(n);
    }

    bool Empty() const {
        return entries.empty();
    }

    bool Insert(const std::vector<uint>& k) {
        return entries.insert(k).second;
    }

    bool Contains(const std::vector<uint>& k) const {
        return entries.find(k) != entries.end();
    }
};

class IterativeMLCore {
public:

    explicit IterativeMLCore(MultilayerGraph& mg_, int num_threads_ = 0);

    virtual ~IterativeMLCore() = default;

    virtual void Decompose(bool hot_start = false);

    const SkylineSet& GetNodeSkyline(uint v) const;

    const std::vector<SkylineSet>& GetAllSkylines() const;

    void RebuildAllLayerNeighborCache();

    double last_iter_ms = 0;

protected:

    MultilayerGraph& mg;
    uint L;
    uint n;
    int num_threads;

    std::vector<SkylineSet> scv_sets;

    std::vector<std::unique_ptr<std::atomic<uint>[]>> max_c;

    std::vector<std::vector<uint>> all_layer_neighbors;
    bool all_layer_neighbors_valid = false;

    std::vector<std::unique_ptr<std::shared_mutex>> scv_rw_locks;
    std::unique_ptr<std::atomic<bool>[]> is_in_inbox;

    std::deque<uint> shared_queue;
    mutable std::mutex shared_queue_mtx;
    std::atomic<long long> active_task_count;

    void InitializeBounds();

    void GetUniqueNeighbors(uint v, std::vector<uint>& out) const;
    const std::vector<uint>& GetCachedUniqueNeighbors(uint v, std::vector<uint>& scratch) const;
    void AddAllLayerNeighborCacheEdge(uint u, uint v);
    void RemoveAllLayerNeighborCacheEdgeIfDisconnected(uint u, uint v);

    std::vector<SkylineSet> lower_bounds_snapshot;
    std::vector<char> lower_bounds_active;
    bool has_lower_bounds = false;

    std::unordered_map<uint, std::vector<unsigned long long>> hot_start_scv_bits_by_node;
    bool hot_start_scv_filter_enabled = false;

    std::vector<char> hot_start_wake_domain;
    bool limit_hot_start_wake_to_domain = false;
    bool fixed_refinement_dim_enabled = false;
    uint fixed_refinement_dim = UINT_MAX;

    struct NeighborMaxCBuffer {
        std::vector<std::vector<uint>> per_layer_maxc;
        std::vector<std::vector<uint>> per_layer_fixed_maxc;

        uint fixed_dim = UINT_MAX;
        bool has_fixed_maxc = false;

        void Init(uint num_layers) {

            has_fixed_maxc = false;
            fixed_dim = UINT_MAX;

            per_layer_maxc.resize(num_layers);
            for (auto& v : per_layer_maxc) v.clear();
            per_layer_fixed_maxc.clear();
        }
    };

    void PrecomputeNeighborMaxC(uint v, NeighborMaxCBuffer& buf) const;

    void ClearHotStartSCVFilter();
    void SeedHotStartSCVFilter(const std::vector<std::pair<uint, std::vector<uint>>>& records);
    void ComputeHFromBuffer(const std::vector<uint>& current_k, const NeighborMaxCBuffer& buf,
        std::vector<uint>& h_vec, bool& h_pruned) const;
    void EnableFixedRefinementDim(uint dim) {
        fixed_refinement_dim_enabled = (dim < L);
        fixed_refinement_dim = fixed_refinement_dim_enabled ? dim : UINT_MAX;
    }
    void DisableFixedRefinementDim() {
        fixed_refinement_dim_enabled = false;
        fixed_refinement_dim = UINT_MAX;
    }
    bool IsRefinementDimFixed(uint dim) const {
        return fixed_refinement_dim_enabled && dim == fixed_refinement_dim;
    }
    bool ApplyFixedRefinementDim(const std::vector<uint>& source,
        std::vector<uint>& candidate) const {
        if (fixed_refinement_dim_enabled && fixed_refinement_dim < L &&
            fixed_refinement_dim < candidate.size() &&
            fixed_refinement_dim < source.size()) {
            candidate[fixed_refinement_dim] = source[fixed_refinement_dim];
        }
        for (uint d = 0; d < L; d++) {
            if (candidate[d] < source[d]) return true;
        }
        return false;
    }
    bool FixedRefinementSubtreeFails(const std::vector<uint>& current_k,
        const NeighborMaxCBuffer& buf) const;

    struct RoundNodeResult {

        uint v = UINT_MAX;
        bool changed = false;
        uint old_scv_count = 0;
        uint new_scv_count = 0;
        std::vector<uint> old_scvs_flat;
        std::vector<uint> new_scvs_flat;
        SkylineSet next_scv_set;
        std::vector<uint> next_hot_indices;
    };
    void ProcessRoundNode(uint v, RoundNodeResult& result);
    void DecomposeHotStartByRounds(bool use_parallel);

    bool CheckSupport(uint v, const std::vector<uint>& k_vec) const;
    bool CheckSupportNoLock(uint v, const std::vector<uint>& k_vec) const;
    bool CheckSupportFromSnapshot(uint v, const std::vector<uint>& k_vec,
        const NeighborSnapshot& snapshot) const;

    void SplitSCV(uint v, const std::vector<uint>& k_vec,
        SkylineSet& valid_results,
        const NeighborMaxCBuffer* precomputed_buf,
        const NeighborSnapshot* precomputed_snapshot,
        SkylineSet* completed_roots) const;

    void SearchOrderedSplit(uint v, std::vector<uint> lower, std::vector<uint> upper,
        uint idx, SkylineSet& valid_results,
        const NeighborSnapshot& snapshot,
        const LazyVerifyState* ancestor_state,
        const std::vector<uint>* ancestor_k,
        uint depth, const SkylineSet* completed_roots) const;

    virtual std::vector<uint> TryHIndexCandidate(uint, const std::vector<uint>&,
        const NeighborMaxCBuffer&) const {
        return {};
    }

    void BuildNeighborSnapshotNoLock(uint v, NeighborSnapshot& snapshot) const;
    bool LazyVerifyAndClassify(uint v, const std::vector<uint>& k_vec,
        const NeighborSnapshot& snapshot,
        LazyVerifyState& state,
        bool stop_on_first_failure = false) const;

    bool LazyVerifyIncremental(uint v, const std::vector<uint>& child_k, uint changed_dim,
        const NeighborSnapshot& snapshot,
        const LazyVerifyState& parent_state,
        LazyVerifyState& child_state) const;

    uint CalcHIndex(const std::vector<uint>& support_vals, uint max_bound) const;

    void UpdateMaxC(uint v);

};

#endif
