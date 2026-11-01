#include "IterativeMLCore.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <cassert>
#include <omp.h>

namespace {

    bool ValidationNeighborSupports(uint, const SkylineSet& set,
        const std::vector<uint>& k, uint L) {
        return set.IsDominatedRaw(k.data(), L);
    }

    unsigned long long HotStartRoundTaskThreshold() {
        return 8;
    }

    unsigned long long HotStartRoundSCVThreshold() {
        return 512;
    }

    bool DominatesVectorRaw(const uint* a, const std::vector<uint>& b, uint L) {
        for (uint d = 0; d < L; d++) {
            if (a[d] < b[d]) return false;
        }
        return true;
    }

    bool RootDominatesCandidate(const SkylineSet* completed_roots,
        const std::vector<uint>& candidate,
        uint L) {
        return completed_roots && completed_roots->IsDominatedRaw(candidate.data(), L);
    }

    bool ScanLazyNeighborSCVs(const SCVSnapshot& u_scvs,
        const std::vector<uint>& k_vec,
        uint L,
        LazyVerifyState::NeighborInfo& info) {
        info.is_supporter = false;
        info.support_scv_idx = UINT_MAX;
        uint count = u_scvs.NumSCVs(L);

        const uint* k_raw = k_vec.data();
        for (uint si = 0; si < count; si++) {

            if (SkylineSet::DominatesRaw(u_scvs.GetRaw(si, L), k_raw, L)) {
                info.is_supporter = true;
                info.support_scv_idx = si;

                return true;
            }

        }
        return false;
    }

    void MarkLazyNeighborFailed(const SCVSnapshot&,
        uint,
        LazyVerifyState::NeighborInfo& info) {
        info.is_supporter = false;
        info.support_scv_idx = UINT_MAX;

    }

    bool SnapshotDominates(const SCVSnapshot& u_scvs, const uint* k_raw, uint L) {
        uint scv_count = u_scvs.NumSCVs(L);
        for (uint si = 0; si < scv_count; si++) {
            if (SkylineSet::DominatesRaw(u_scvs.GetRaw(si, L), k_raw, L)) return true;
        }
        return false;
    }

    inline bool MaxCAtLeast(const std::vector<std::unique_ptr<std::atomic<uint>[]>>& max_c,
        uint u, const uint* k_raw, uint L) {
        for (uint j = 0; j < L; j++) {
            if (max_c[u][j].load(std::memory_order_relaxed) < k_raw[j]) return false;
        }
        return true;
    }
}

IterativeMLCore::IterativeMLCore(MultilayerGraph& mg_, int num_threads_) : mg(mg_) {
    L = mg.GetLayerNumber();
    n = mg.GetN();

    if (num_threads_ <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    } else {
        num_threads = num_threads_;
    }

    scv_sets.resize(n);
    max_c.resize(n);
    scv_rw_locks.resize(n);

    for (uint i = 0; i < n; i++) {
        scv_rw_locks[i] = std::make_unique<std::shared_mutex>();
    }

    active_task_count.store(0);
    is_in_inbox = std::make_unique<std::atomic<bool>[]>(n);
    for (uint i = 0; i < n; i++) {
        is_in_inbox[i].store(false, std::memory_order_relaxed);
    }
    RebuildAllLayerNeighborCache();
}

void IterativeMLCore::InitializeBounds() {
    std::vector<std::vector<uint>> bound(n, std::vector<uint>(L, 0));
    for (uint v = 0; v < n; v++) {
        for (uint l = 0; l < L; l++) {
            bound[v][l] = mg.GetGraph(l).GetAdjLst()[v][0];
        }
    }

    std::vector<uint> h_buf;
    for (uint v = 0; v < n; v++) {
        for (uint l = 0; l < L; l++) {
            uint** adj = mg.GetGraph(l).GetAdjLst();
            uint deg = adj[v][0];
            h_buf.clear();
            for (uint i = 1; i <= deg; i++) {
                uint u = adj[v][i];
                h_buf.push_back(mg.GetGraph(l).GetAdjLst()[u][0]);
            }
            bound[v][l] = CalcHIndex(h_buf, bound[v][l]);
        }
    }

    std::vector<std::pair<uint, uint>> sorted_nodes;
    sorted_nodes.reserve(n);
    for (uint v = 0; v < n; v++) {
        scv_sets[v] = SkylineSet();
        scv_sets[v].Insert(bound[v]);
        max_c[v] = std::make_unique<std::atomic<uint>[]>(L);
        uint scv_level = 0;
        for (uint l = 0; l < L; l++) {
            max_c[v][l].store(bound[v][l], std::memory_order_relaxed);
            scv_level += bound[v][l];
        }
        sorted_nodes.emplace_back(scv_level, v);
    }
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), std::greater<>{});
    {
        std::lock_guard<std::mutex> lock(shared_queue_mtx);
        for (auto& [_, v] : sorted_nodes) {
            is_in_inbox[v].store(true, std::memory_order_relaxed);
            shared_queue.push_back(v);
        }
    }
    active_task_count.store(n);
}

void IterativeMLCore::GetUniqueNeighbors(uint v, std::vector<uint>& out) const {
    if (all_layer_neighbors_valid && v < all_layer_neighbors.size()) {
        out = all_layer_neighbors[v];
        return;
    }

    thread_local std::vector<uint> visited_epoch;
    thread_local uint current_epoch = 0;

    if (visited_epoch.size() < n) visited_epoch.resize(n, 0);

    current_epoch++;
    if (current_epoch == 0) {
        std::fill(visited_epoch.begin(), visited_epoch.end(), 0);
        current_epoch = 1;
    }

    out.clear();
    for (uint l = 0; l < L; l++) {
        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];
        for (uint i = 1; i <= degree; i++) {
            uint u = adj_lst[v][i];
            if (visited_epoch[u] != current_epoch) {
                visited_epoch[u] = current_epoch;
                out.push_back(u);
            }
        }
    }
}

void IterativeMLCore::RebuildAllLayerNeighborCache() {
    all_layer_neighbors.clear();
    all_layer_neighbors.resize(n);

#pragma omp parallel for schedule(dynamic, 256)
    for (uint v = 0; v < n; v++) {
        size_t degree_sum = 0;
        for (uint l = 0; l < L; l++) {
            degree_sum += mg.GetGraph(l).GetAdjLst()[v][0];
        }

        std::vector<uint> neighbors;
        neighbors.reserve(degree_sum);
        for (uint l = 0; l < L; l++) {
            uint** adj = mg.GetGraph(l).GetAdjLst();
            uint degree = adj[v][0];
            for (uint i = 1; i <= degree; i++) {
                neighbors.push_back(adj[v][i]);
            }
        }
        if (neighbors.size() > 1) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
        all_layer_neighbors[v].swap(neighbors);
    }
    all_layer_neighbors_valid = true;
}

const std::vector<uint>& IterativeMLCore::GetCachedUniqueNeighbors(
    uint v, std::vector<uint>& scratch) const {
    if (all_layer_neighbors_valid && v < all_layer_neighbors.size()) {
        return all_layer_neighbors[v];
    }
    GetUniqueNeighbors(v, scratch);
    return scratch;
}

void IterativeMLCore::AddAllLayerNeighborCacheEdge(uint u, uint v) {
    if (!all_layer_neighbors_valid || u >= all_layer_neighbors.size() ||
        v >= all_layer_neighbors.size()) {
        return;
    }

    auto contains = [](const std::vector<uint>& values, uint target) {
        return std::find(values.begin(), values.end(), target) != values.end();
    };
    if (!contains(all_layer_neighbors[u], v)) all_layer_neighbors[u].push_back(v);
    if (!contains(all_layer_neighbors[v], u)) all_layer_neighbors[v].push_back(u);
}

void IterativeMLCore::RemoveAllLayerNeighborCacheEdgeIfDisconnected(uint u, uint v) {
    if (!all_layer_neighbors_valid || u >= all_layer_neighbors.size() ||
        v >= all_layer_neighbors.size()) {
        return;
    }

    bool still_connected = false;
    for (uint l = 0; l < L && !still_connected; l++) {
        uint** adj = mg.GetGraph(l).GetAdjLst();
        uint degree = adj[u][0];
        for (uint i = 1; i <= degree; i++) {
            if (adj[u][i] == v) {
                still_connected = true;
                break;
            }
        }
    }
    if (still_connected) return;

    auto erase_one = [](std::vector<uint>& values, uint target) {
        auto it = std::find(values.begin(), values.end(), target);
        if (it != values.end()) {
            *it = values.back();
            values.pop_back();
        }
    };
    erase_one(all_layer_neighbors[u], v);
    erase_one(all_layer_neighbors[v], u);
}

void IterativeMLCore::UpdateMaxC(uint v) {
    std::vector<uint> current_max(L, 0);
    const SkylineSet& set = scv_sets[v];
    uint count = set.NumSCVs();
    for (uint i = 0; i < count; i++) {
        const uint* k_vec = set.GetSCVFlat(i);
        for (uint l = 0; l < L; l++) {
            if (k_vec[l] > current_max[l]) {
                current_max[l] = k_vec[l];
            }
        }
    }
    for (uint l = 0; l < L; l++) {
        max_c[v][l].store(current_max[l], std::memory_order_relaxed);
    }
}

bool IterativeMLCore::CheckSupport(uint v, const std::vector<uint>& k_vec) const {
    const uint* k_raw = k_vec.data();
    for (uint l = 0; l < L; l++) {
        uint required_k = k_raw[l];
        if (required_k == 0) continue;

        uint support_count = 0;
        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];

        if (degree < required_k) return false;

        for (uint i = 1; i <= degree; i++) {
            if (support_count >= required_k) break;
            if (support_count + (degree - i + 1) < required_k) return false;

            uint u = adj_lst[v][i];
            bool can_support = MaxCAtLeast(max_c, u, k_raw, L);
            if (!can_support) continue;

            scv_rw_locks[u]->lock_shared();
            bool dominated = ValidationNeighborSupports(u, scv_sets[u], k_vec, L);
            scv_rw_locks[u]->unlock_shared();

            if (dominated) support_count++;
        }

        if (support_count < required_k) return false;
    }
    return true;
}

bool IterativeMLCore::CheckSupportNoLock(uint v, const std::vector<uint>& k_vec) const {
    const uint* k_raw = k_vec.data();
    for (uint l = 0; l < L; l++) {
        uint required_k = k_raw[l];
        if (required_k == 0) continue;

        uint support_count = 0;
        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];

        if (degree < required_k) return false;

        for (uint i = 1; i <= degree; i++) {
            if (support_count >= required_k) break;
            if (support_count + (degree - i + 1) < required_k) return false;

            uint u = adj_lst[v][i];
            bool can_support = MaxCAtLeast(max_c, u, k_raw, L);
            if (!can_support) continue;

            if (ValidationNeighborSupports(u, scv_sets[u], k_vec, L)) support_count++;
        }

        if (support_count < required_k) return false;
    }
    return true;
}

bool IterativeMLCore::CheckSupportFromSnapshot(uint v, const std::vector<uint>& k_vec,
    const NeighborSnapshot& snapshot) const {
    const uint* k_raw = k_vec.data();
    for (uint l = 0; l < L; l++) {
        uint required_k = k_raw[l];
        if (required_k == 0) continue;

        uint support_count = 0;
        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];

        if (degree < required_k) return false;

        for (uint i = 1; i <= degree; i++) {
            if (support_count >= required_k) break;
            if (support_count + (degree - i + 1) < required_k) return false;

            uint u = adj_lst[v][i];
            bool can_support = MaxCAtLeast(max_c, u, k_raw, L);
            if (!can_support) continue;

            if (!snapshot[u].Empty() && SnapshotDominates(snapshot[u], k_vec.data(), L)) support_count++;
        }

        if (support_count < required_k) return false;
    }
    return true;
}

uint IterativeMLCore::CalcHIndex(const std::vector<uint>& support_vals, uint max_bound) const {
    if (support_vals.empty()) return 0;

    thread_local std::vector<uint> counts;
    size_t needed = static_cast<size_t>(max_bound) + 2;
    if (counts.size() < needed) counts.resize(needed);
    std::fill(counts.begin(), counts.begin() + needed, 0);

    for (uint val : support_vals) {
        uint capped_val = std::min(val, max_bound);
        counts[capped_val]++;
    }

    uint count_ge = 0;
    for (int i = max_bound; i >= 0; --i) {
        count_ge += counts[i];
        if (count_ge >= (uint)i) {
            return i;
        }
    }
    return 0;
}

void IterativeMLCore::PrecomputeNeighborMaxC(uint v, NeighborMaxCBuffer& buf) const {
    buf.Init(L);
    bool collect_fixed_dim =
    has_lower_bounds && fixed_refinement_dim_enabled && fixed_refinement_dim < L;

    if (collect_fixed_dim) {
        buf.has_fixed_maxc = true;
        buf.fixed_dim = fixed_refinement_dim;
        buf.per_layer_fixed_maxc.resize(L);
    }
    for (uint l = 0; l < L; l++) {
        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];
        buf.per_layer_maxc[l].reserve(degree);
        if (collect_fixed_dim) buf.per_layer_fixed_maxc[l].reserve(degree);
        for (uint i = 1; i <= degree; i++) {
            uint u = adj_lst[v][i];
            buf.per_layer_maxc[l].push_back(max_c[u][l].load(std::memory_order_relaxed));
            if (collect_fixed_dim) {
                buf.per_layer_fixed_maxc[l].push_back(
                    max_c[u][fixed_refinement_dim].load(std::memory_order_relaxed));
            }
        }
    }
}

void IterativeMLCore::ClearHotStartSCVFilter() {
    hot_start_scv_bits_by_node.clear();
    hot_start_scv_filter_enabled = false;
}

void IterativeMLCore::SeedHotStartSCVFilter(
    const std::vector<std::pair<uint, std::vector<uint>>>& records) {
    if (records.empty()) return;
    for (const auto& item : records) {
        uint node = item.first;
        if (node >= n || item.second.size() != L) continue;
        const SkylineSet& set = scv_sets[node];
        if (!set.ContainsRaw(item.second.data(), L)) continue;
        uint count = set.NumSCVs();
        for (uint si = 0; si < count; si++) {
            const uint* raw = set.GetSCVFlat(si);
            bool equal = true;
            for (uint d = 0; d < L; d++) {
                if (raw[d] != item.second[d]) {
                    equal = false;
                    break;
                }
            }
            if (!equal) continue;
            auto& bits = hot_start_scv_bits_by_node[node];
            size_t word = static_cast<size_t>(si >> 6);
            if (bits.size() <= word) bits.resize(word + 1, 0);
            bits[word] |= 1ULL << (si & 63);
            break;
        }
    }
    hot_start_scv_filter_enabled = !hot_start_scv_bits_by_node.empty();
}

void IterativeMLCore::ComputeHFromBuffer(const std::vector<uint>& current_k, const NeighborMaxCBuffer& buf,
    std::vector<uint>& h_vec, bool& h_pruned) const {
    thread_local std::vector<uint> counts;
    h_pruned = false;
    bool use_fixed_conditioned_cap =
    has_lower_bounds && fixed_refinement_dim_enabled &&
    fixed_refinement_dim < L && fixed_refinement_dim < current_k.size() &&
    buf.has_fixed_maxc && buf.fixed_dim == fixed_refinement_dim &&
    current_k[fixed_refinement_dim] > 0;

    for (uint l = 0; l < L; l++) {
        uint bound = current_k[l];
        if (bound == 0) { h_vec[l] = 0; continue; }

        size_t needed = static_cast<size_t>(bound) + 1;
        if (counts.size() < needed) counts.resize(needed);
        std::fill(counts.begin(), counts.begin() + needed, 0);

        if (use_fixed_conditioned_cap &&
            l < buf.per_layer_fixed_maxc.size() &&
            buf.per_layer_fixed_maxc[l].size() == buf.per_layer_maxc[l].size()) {
            uint fixed_required = current_k[fixed_refinement_dim];
            for (size_t idx = 0; idx < buf.per_layer_maxc[l].size(); idx++) {
                if (buf.per_layer_fixed_maxc[l][idx] < fixed_required) continue;
                uint val = buf.per_layer_maxc[l][idx];
                if (val == 0) continue;
                counts[std::min(bound, val)]++;
            }
        } else {
            for (uint val : buf.per_layer_maxc[l]) {
                if (val == 0) continue;
                counts[std::min(bound, val)]++;
            }
        }

        uint count_ge = 0;
        uint h = 0;
        for (int i = static_cast<int>(bound); i >= 1; --i) {
            count_ge += counts[i];
            if (count_ge >= static_cast<uint>(i)) {
                h = static_cast<uint>(i);
                break;
            }
        }
        h_vec[l] = h;
        if (h_vec[l] < current_k[l]) h_pruned = true;
    }
}

bool IterativeMLCore::FixedRefinementSubtreeFails(const std::vector<uint>& current_k,
    const NeighborMaxCBuffer& buf) const {
    if (!has_lower_bounds || !fixed_refinement_dim_enabled || fixed_refinement_dim >= L ||
        fixed_refinement_dim >= current_k.size()) {
        return false;
    }
    uint g = fixed_refinement_dim;
    uint required = current_k[g];
    if (required == 0) return false;

    if (g >= buf.per_layer_maxc.size()) return false;

    uint possible = 0;
    for (uint val : buf.per_layer_maxc[g]) {
        if (val >= required) {
            possible++;
            if (possible >= required) return false;
        }
    }
    return true;
}

void IterativeMLCore::SplitSCV(
    uint v, const std::vector<uint>& k_vec, SkylineSet& valid_results,
    const NeighborMaxCBuffer* precomputed_buf,
    const NeighborSnapshot* precomputed_snapshot, SkylineSet* completed_roots) const {
    if (valid_results.IsDominatedRaw(k_vec.data(), L) ||
        RootDominatesCandidate(completed_roots, k_vec, L)) return;
    NeighborMaxCBuffer local_buf;
    if (!precomputed_buf) {
        PrecomputeNeighborMaxC(v, local_buf);
        precomputed_buf = &local_buf;
    }
    if (FixedRefinementSubtreeFails(k_vec, *precomputed_buf)) {
        if (completed_roots) completed_roots->Insert(k_vec);
        return;
    }
    std::vector<uint> upper(L), lower(L, 0);
    bool pruned = false;
    ComputeHFromBuffer(k_vec, *precomputed_buf, upper, pruned);
    ApplyFixedRefinementDim(k_vec, upper);
    for (uint d = 0; d < L; d++) {
        if (IsRefinementDimFixed(d)) lower[d] = upper[d];
    }
    NeighborSnapshot local_snapshot;
    if (!precomputed_snapshot) {
        BuildNeighborSnapshotNoLock(v, local_snapshot);
        precomputed_snapshot = &local_snapshot;
    }

    SearchOrderedSplit(v, std::move(lower), std::move(upper), 0, valid_results,
        *precomputed_snapshot, nullptr, nullptr, 0, completed_roots);

    if (completed_roots) completed_roots->Insert(k_vec);
}

void IterativeMLCore::SearchOrderedSplit(
    uint v, std::vector<uint> lower, std::vector<uint> upper,
    uint idx, SkylineSet& valid_results, const NeighborSnapshot& snapshot,
    const LazyVerifyState* ancestor_state, const std::vector<uint>* ancestor_k,
    uint depth, const SkylineSet* completed_roots) const {
    assert(idx < L);

    for (uint d = 0; d < idx; ++d) assert(lower[d] == upper[d]);

    if (valid_results.IsDominatedRaw(upper.data(), L) ||
        RootDominatesCandidate(completed_roots, upper, L)) return;

    auto trim_covered = [&](const SkylineSet& certificates) {
        for (uint si = 0; si < certificates.NumSCVs(); si++) {
            const uint* cert = certificates.GetSCVFlat(si);
            uint missing = L, count = 0;
            for (uint d = 0; d < L; d++) {
                if (cert[d] < upper[d]) { missing = d; ++count; }
            }
            if (count == 0) return false;
            if (count == 1) lower[missing] = std::max(lower[missing], cert[missing] + 1);
        }
        return true;
    };
    if (!trim_covered(valid_results)) return;
    if (completed_roots && !trim_covered(*completed_roots)) return;
    if (has_lower_bounds && v < lower_bounds_active.size() &&
        lower_bounds_active[v] && v < lower_bounds_snapshot.size() &&
        !trim_covered(lower_bounds_snapshot[v])) return;
    for (uint d = 0; d < L; d++) if (lower[d] > upper[d]) return;

    struct Workspace {
        LazyVerifyState root;
        std::unique_ptr<LazyVerifyState> failed, probe;
        explicit Workspace(uint layers)
        : root(layers), failed(new LazyVerifyState(layers)),
        probe(new LazyVerifyState(layers)) {}
    };
    thread_local std::vector<std::unique_ptr<Workspace>> pool;
    if (!pool.empty() && pool[0]->root.layers.size() != L) pool.clear();
    while (pool.size() <= depth) pool.emplace_back(new Workspace(L));
    Workspace& ws = *pool[depth];
    bool valid;
    if (ancestor_state) {
        assert(ancestor_k && DominatesVectorRaw(ancestor_k->data(), upper, L));
        valid = LazyVerifyIncremental(v, upper, L, snapshot, *ancestor_state, ws.root);
    } else {
        valid = LazyVerifyAndClassify(v, upper, snapshot, ws.root, true);
    }
    if (valid) { valid_results.Insert(upper); return; }
    uint split_dim = L;

    for (int d = static_cast<int>(L) - 1; d >= static_cast<int>(idx); --d) {
        if (upper[d] > lower[d]) {
            split_dim = static_cast<uint>(d);
            break;
        }
    }
    if (split_dim == L) return;

    long long lo = static_cast<long long>(lower[split_dim]) - 1;
    uint hi = upper[split_dim];
    const LazyVerifyState* failed_state = &ws.root;
    std::vector<uint> trial = upper;
    while (static_cast<long long>(hi) - lo > 1) {
        uint mid = static_cast<uint>(lo + (static_cast<long long>(hi) - lo) / 2);
        trial[split_dim] = mid;

        bool supported = LazyVerifyIncremental(v, trial, split_dim, snapshot, *failed_state, *ws.probe);
        if (supported) {
            lo = mid;
        } else {
            hi = mid;
            ws.failed.swap(ws.probe);
            failed_state = ws.failed.get();
        }
    }
    if (lo >= static_cast<long long>(lower[split_dim])) {
        trial[split_dim] = static_cast<uint>(lo);
        valid_results.Insert(trial);
        lower[split_dim] = static_cast<uint>(lo) + 1;
    }

    std::vector<uint> child_lower = upper;
    child_lower[split_dim] = lower[split_dim];
    for (int d = static_cast<int>(L) - 1; d >= static_cast<int>(idx); --d) {
        if (static_cast<uint>(d) == split_dim) continue;
        child_lower[d] = lower[d];
        if (lower[d] == upper[d]) continue;
        std::vector<uint> child_upper = upper;
        --child_upper[d];
        assert(static_cast<uint>(d) >= idx);

        SearchOrderedSplit(v, child_lower, std::move(child_upper), static_cast<uint>(d),
            valid_results, snapshot, &ws.root, &upper,
            depth + 1, completed_roots);
    }

}

void IterativeMLCore::BuildNeighborSnapshotNoLock(uint v, NeighborSnapshot& snapshot) const {
    thread_local std::vector<uint> neighbors;
    GetUniqueNeighbors(v, neighbors);
    snapshot.Clear();
    snapshot.Reserve(neighbors.size());

    for (uint u : neighbors) {
        const SkylineSet& u_set = scv_sets[u];
        auto& out = snapshot.GetOrCreate(u);
        out.SetView(u_set);
    }
}

bool IterativeMLCore::LazyVerifyAndClassify(uint v, const std::vector<uint>& k_vec,
    const NeighborSnapshot& snapshot,
    LazyVerifyState& state,
    bool stop_on_first_failure) const {

    state.Clear();
    bool all_supported = true;

    for (uint l = 0; l < L; l++) {
        uint required_k = k_vec[l];
        if (required_k == 0) {
            state.layers[l].initialized = true;
            continue;
        }

        auto& ls = state.layers[l];
        ls.initialized = true;
        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];
        ls.neighbor_info.resize(degree + 1);

        if (degree < required_k) {
            for (uint i = 1; i <= degree; i++) {
                uint u = adj_lst[v][i];
                ls.fail_nodes.push_back(i);
                auto& ni = ls.GetInfo(i);
                MarkLazyNeighborFailed(snapshot[u], L, ni);
            }
            all_supported = false;
            if (stop_on_first_failure) return false;
            continue;
        }

        for (uint i = 1; i <= degree; i++) {
            uint u = adj_lst[v][i];

            bool can_support = true;
            for (uint j = 0; j < L; j++) {
                if (max_c[u][j].load(std::memory_order_relaxed) < k_vec[j]) {
                    can_support = false;
                    break;
                }
            }

            auto& ni = ls.GetInfo(i);

            if (!can_support) {
                MarkLazyNeighborFailed(snapshot[u], L, ni);
                ls.fail_nodes.push_back(i);
                continue;
            }

            const auto& u_scvs = snapshot[u];
            if (u_scvs.Empty()) {
                MarkLazyNeighborFailed(u_scvs, L, ni);
                ls.fail_nodes.push_back(i);
                continue;
            }

            bool found_support = ScanLazyNeighborSCVs(u_scvs, k_vec, L, ni);

            if (found_support) {
                ls.support_nodes.push_back(i);
                ls.support_count++;
                if (ls.support_count >= required_k) {
                    for (uint j = i + 1; j <= degree; j++) {
                        ls.candidate_nodes.push_back(j);
                    }
                    break;
                }
            } else {
                ni.is_supporter = false;
                ni.support_scv_idx = UINT_MAX;
                ls.fail_nodes.push_back(i);
            }
        }

        if (ls.support_count < required_k) {
            all_supported = false;
            if (stop_on_first_failure) return false;
        }
    }

    return all_supported;
}

bool IterativeMLCore::LazyVerifyIncremental(uint v, const std::vector<uint>& child_k, uint changed_dim,
    const NeighborSnapshot& snapshot,
    const LazyVerifyState& parent_state,
    LazyVerifyState& child_state) const {

    (void)changed_dim;

    child_state.Clear();
    bool all_supported = true;

    auto verify_layer_from_scratch = [&](uint l, LazyVerifyState::LayerState& cls) {
        cls.initialized = true;
        uint required_k = child_k[l];
        if (required_k == 0) return true;

        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];
        cls.neighbor_info.resize(degree + 1);

        if (degree < required_k) {
            for (uint i = 1; i <= degree; i++) {
                uint u = adj_lst[v][i];
                cls.fail_nodes.push_back(i);
                auto& ni = cls.GetInfo(i);
                MarkLazyNeighborFailed(snapshot[u], L, ni);
            }
            return false;
        }

        for (uint i = 1; i <= degree; i++) {
            uint u = adj_lst[v][i];

            bool can_support = true;
            for (uint j = 0; j < L; j++) {
                if (max_c[u][j].load(std::memory_order_relaxed) < child_k[j]) {
                    can_support = false;
                    break;
                }
            }

            auto& ni = cls.GetInfo(i);
            if (!can_support) {
                MarkLazyNeighborFailed(snapshot[u], L, ni);
                cls.fail_nodes.push_back(i);
                continue;
            }

            const auto& u_scvs = snapshot[u];
            if (u_scvs.Empty()) {
                MarkLazyNeighborFailed(u_scvs, L, ni);
                cls.fail_nodes.push_back(i);
                continue;
            }

            bool found_support = ScanLazyNeighborSCVs(u_scvs, child_k, L, ni);
            if (found_support) {
                cls.support_nodes.push_back(i);
                cls.support_count++;
                if (cls.support_count >= required_k) {
                    for (uint j = i + 1; j <= degree; j++) {
                        cls.candidate_nodes.push_back(j);
                    }
                    return true;
                }
            } else {
                ni.is_supporter = false;
                ni.support_scv_idx = UINT_MAX;
                cls.fail_nodes.push_back(i);
            }
        }

        return cls.support_count >= required_k;
    };

    for (uint l = 0; l < L; l++) {
        uint required_k = child_k[l];
        if (required_k == 0) {
            child_state.layers[l].initialized = true;
            continue;
        }

        auto& cls = child_state.layers[l];
        const auto& pls = parent_state.layers[l];
        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];
        cls.neighbor_info.resize(degree + 1);
        cls.initialized = true;

        if (!pls.initialized) {
            if (!verify_layer_from_scratch(l, cls)) {
                all_supported = false;
            }
            continue;
        }

        cls.support_count = pls.support_count;

        if (cls.support_count >= required_k) {

            continue;
        }
        cls.support_nodes = pls.support_nodes;
        for (uint pos : cls.support_nodes) {
            cls.GetInfo(pos) = pls.neighbor_info[pos];

        }

        auto recheck_pos = [&](uint pos, bool inherited_candidate) {
            uint u = adj_lst[v][pos];
            auto& ni = cls.GetInfo(pos);
            if (inherited_candidate && pos < pls.neighbor_info.size()) {
                ni = pls.neighbor_info[pos];
                if (ni.is_supporter && ni.support_scv_idx != UINT_MAX &&
                    ni.support_scv_idx < snapshot[u].NumSCVs(L)) {

                    cls.support_nodes.push_back(pos);
                    cls.support_count++;
                    if (cls.support_count >= required_k) {
                        return true;
                    }
                    return false;
                }
            }

            bool can_support = true;
            for (uint j = 0; j < L; j++) {
                if (max_c[u][j].load(std::memory_order_relaxed) < child_k[j]) {
                    can_support = false;
                    break;
                }
            }

            if (!can_support) {
                MarkLazyNeighborFailed(snapshot[u], L, ni);
                cls.fail_nodes.push_back(pos);
                return false;
            }

            const auto& u_scvs = snapshot[u];
            if (u_scvs.Empty()) {
                MarkLazyNeighborFailed(u_scvs, L, ni);
                cls.fail_nodes.push_back(pos);
                return false;
            }

            bool found_support = ScanLazyNeighborSCVs(u_scvs, child_k, L, ni);

            if (found_support) {
                cls.support_nodes.push_back(pos);
                cls.support_count++;
                if (cls.support_count >= required_k) {
                    return true;
                }
            } else {
                ni.is_supporter = false;
                ni.support_scv_idx = UINT_MAX;
                cls.fail_nodes.push_back(pos);
            }
            return false;
        };

        for (uint pos : pls.fail_nodes) {

            if (recheck_pos(pos, false)) break;
        }

        if (cls.support_count < required_k) {
            for (uint pos : pls.candidate_nodes) {

                if (recheck_pos(pos, true)) break;
            }
        }

        if (cls.support_count < required_k) {
            all_supported = false;
        }
    }

    return all_supported;
}

void IterativeMLCore::ProcessRoundNode(uint v, RoundNodeResult& result) {
    result = RoundNodeResult();

    result.v = v;

    std::vector<uint> k_vec;

    result.old_scv_count = scv_sets[v].NumSCVs();
    result.old_scvs_flat.reserve(static_cast<size_t>(result.old_scv_count) * L);
    for (uint si = 0; si < result.old_scv_count; si++) {
        const uint* raw = scv_sets[v].GetSCVFlat(si);
        result.old_scvs_flat.insert(result.old_scvs_flat.end(), raw, raw + L);
    }
    SkylineSet next_scv_set;
    bool node_may_change = false;
    SkylineSet shared_split_results;
    SkylineSet completed_invalid_roots;

    NeighborMaxCBuffer node_buf;
    bool has_node_buf = false;
    NeighborSnapshot node_snapshot;
    bool has_node_snapshot = false;
    uint eager_snapshot_threshold = 128;
    if (result.old_scv_count >= eager_snapshot_threshold) {
        BuildNeighborSnapshotNoLock(v, node_snapshot);
        has_node_snapshot = true;
    }

    if (has_lower_bounds &&
        v < lower_bounds_snapshot.size() &&
        v < lower_bounds_active.size() &&
        lower_bounds_active[v]) {

        const SkylineSet& lb_set = lower_bounds_snapshot[v];
        uint lb_count = lb_set.NumSCVs();
        for (uint si = 0; si < lb_count; si++) {
            const uint* lb = lb_set.GetSCVFlat(si);
            next_scv_set.InsertRaw(lb, L);
            shared_split_results.InsertRaw(lb, L);
        }
    }

    std::vector<unsigned long long> local_hot_bits;
    bool use_hot_start_filter = false;
    CandidateTracker affected_candidates;
    if (hot_start_scv_filter_enabled) {
        auto it = hot_start_scv_bits_by_node.find(v);
        if (it != hot_start_scv_bits_by_node.end() && !it->second.empty()) {
            local_hot_bits.swap(it->second);
            use_hot_start_filter = true;
        }
    }
    auto remember_generated_hot_record = [&](const std::vector<uint>& rec) {
        if (!hot_start_scv_filter_enabled) return;
        affected_candidates.Insert(rec);
    };

    k_vec.resize(L);
    for (uint si = 0; si < result.old_scv_count; si++) {
        const uint* k_raw = result.old_scvs_flat.data() + static_cast<size_t>(si) * L;
        for (uint d = 0; d < L; d++) k_vec[d] = k_raw[d];

        size_t hot_word = static_cast<size_t>(si >> 6);
        bool is_affected_scv =
        use_hot_start_filter && hot_word < local_hot_bits.size() &&
        (local_hot_bits[hot_word] & (1ULL << (si & 63))) != 0;
        if (hot_start_scv_filter_enabled && !is_affected_scv) {
            next_scv_set.InsertRaw(k_raw, L);
            continue;
        }
        if (is_affected_scv) affected_candidates.Insert(k_vec);

        if (next_scv_set.IsDominatedRaw(k_raw, L) ||
            shared_split_results.IsDominatedRaw(k_raw, L) ||
            completed_invalid_roots.IsDominatedRaw(k_raw, L)) {
            continue;
        }

        bool supported = has_node_snapshot
        ? CheckSupportFromSnapshot(v, k_vec, node_snapshot)
        : CheckSupportNoLock(v, k_vec);

        if (supported) {
            next_scv_set.Insert(k_vec);
            shared_split_results.Insert(k_vec);
        } else {
            node_may_change = true;

            if (!has_node_buf) {
                PrecomputeNeighborMaxC(v, node_buf);
                has_node_buf = true;
            }

            auto binary_result = TryHIndexCandidate(v, k_vec, node_buf);

            if (!binary_result.empty() &&
                !next_scv_set.IsDominatedRaw(binary_result.data(), L) &&
                !completed_invalid_roots.IsDominatedRaw(binary_result.data(), L)) {
                bool inserted_next = next_scv_set.Insert(binary_result);
                shared_split_results.Insert(binary_result);
                if (inserted_next) remember_generated_hot_record(binary_result);
            } else {
                if (FixedRefinementSubtreeFails(k_vec, node_buf)) {
                    completed_invalid_roots.Insert(k_vec);
                    continue;
                }
                if (!has_node_snapshot) {
                    BuildNeighborSnapshotNoLock(v, node_snapshot);
                    has_node_snapshot = true;
                }
                CandidateTracker split_results_before;
                if (hot_start_scv_filter_enabled) {
                    uint before_count = shared_split_results.NumSCVs();
                    split_results_before.Reserve(before_count * 2 + 1);
                    std::vector<uint> before_rec(L);
                    for (uint ri = 0; ri < before_count; ri++) {
                        const uint* raw = shared_split_results.GetSCVFlat(ri);
                        for (uint d = 0; d < L; d++) before_rec[d] = raw[d];
                        split_results_before.Insert(before_rec);
                    }
                }
                SplitSCV(v, k_vec, shared_split_results,
                    &node_buf, &node_snapshot, &completed_invalid_roots);
                if (hot_start_scv_filter_enabled) {
                    uint after_count = shared_split_results.NumSCVs();
                    std::vector<uint> after_rec(L);
                    for (uint ri = 0; ri < after_count; ri++) {
                        const uint* raw = shared_split_results.GetSCVFlat(ri);
                        for (uint d = 0; d < L; d++) after_rec[d] = raw[d];
                        if (!split_results_before.Contains(after_rec)) {
                            remember_generated_hot_record(after_rec);
                        }
                    }
                }
            }
        }
    }

    uint split_count = shared_split_results.NumSCVs();
    for (uint si = 0; si < split_count; si++) {
        next_scv_set.InsertRaw(shared_split_results.GetSCVFlat(si), L);
    }
    if (hot_start_scv_filter_enabled && !affected_candidates.Empty()) {
        uint final_count = next_scv_set.NumSCVs();
        result.next_hot_indices.reserve(final_count);
        for (uint si = 0; si < final_count; si++) {
            const uint* raw = next_scv_set.GetSCVFlat(si);
            std::vector<uint> final_scv(raw, raw + L);
            if (affected_candidates.Contains(final_scv)) {
                result.next_hot_indices.push_back(si);
            }
        }
    }

    if (!node_may_change) return;

    bool equal = scv_sets[v].IsEqual(next_scv_set);
    if (equal) return;

    result.changed = true;
    result.new_scv_count = next_scv_set.NumSCVs();
    result.new_scvs_flat.reserve(static_cast<size_t>(result.new_scv_count) * L);
    for (uint si = 0; si < result.new_scv_count; si++) {
        const uint* raw = next_scv_set.GetSCVFlat(si);
        result.new_scvs_flat.insert(result.new_scvs_flat.end(), raw, raw + L);
    }
    result.next_scv_set = std::move(next_scv_set);
}

void IterativeMLCore::DecomposeHotStartByRounds(bool use_parallel) {
    std::vector<uint> frontier;
    {
        std::lock_guard<std::mutex> lock(shared_queue_mtx);
        frontier.reserve(shared_queue.size());
        while (!shared_queue.empty()) {
            uint v = shared_queue.front();
            shared_queue.pop_front();
            if (v < n) frontier.push_back(v);
        }
    }
    active_task_count.store(0, std::memory_order_relaxed);
    for (uint v : frontier) {
        if (v < n) is_in_inbox[v].store(false, std::memory_order_relaxed);
    }

    std::vector<char> next_seen(n, 0);
    std::vector<uint> next_seen_nodes;
    while (!frontier.empty()) {
        std::vector<RoundNodeResult> results(frontier.size());
        if (use_parallel) {
#pragma omp parallel for schedule(dynamic, 1)
            for (long long i = 0; i < static_cast<long long>(frontier.size()); i++) {
                ProcessRoundNode(frontier[static_cast<size_t>(i)],
                    results[static_cast<size_t>(i)]);
            }
        } else {
            for (size_t i = 0; i < frontier.size(); i++) {
                ProcessRoundNode(frontier[i], results[i]);
            }
        }

        auto commit_result = [&](RoundNodeResult& res) {
            if (!res.changed) return;
            scv_sets[res.v] = std::move(res.next_scv_set);
            UpdateMaxC(res.v);
        };
        if (use_parallel) {
#pragma omp parallel for schedule(dynamic, 64)
            for (long long i = 0; i < static_cast<long long>(results.size()); i++) {
                commit_result(results[static_cast<size_t>(i)]);
            }
        } else {
            for (RoundNodeResult& res : results) {
                commit_result(res);
            }
        }

        next_seen_nodes.clear();
        std::vector<uint> next_frontier;
        auto add_next = [&](uint node) {
            if (node >= n || next_seen[node]) return;
            next_seen[node] = 1;
            next_seen_nodes.push_back(node);
            next_frontier.push_back(node);
        };
        auto mark_hot_idx = [&](uint node, uint idx) {
            if (!hot_start_scv_filter_enabled || node >= n) return;
            auto& bits = hot_start_scv_bits_by_node[node];
            size_t word = static_cast<size_t>(idx >> 6);
            if (bits.size() <= word) bits.resize(word + 1, 0);
            bits[word] |= 1ULL << (idx & 63);
        };
        for (const RoundNodeResult& res : results) {
            for (uint idx : res.next_hot_indices) {
                mark_hot_idx(res.v, idx);
            }
        }
        struct WakeLocal {
            std::vector<uint> next_candidates;
            std::vector<std::pair<uint, uint>> hot_indices;
        };
        int wake_threads = std::max(1, omp_get_max_threads());
        std::vector<WakeLocal> wake_locals(wake_threads);

        auto process_wake_result = [&](RoundNodeResult& res, WakeLocal& local) {
            if (!res.changed) return;

            std::vector<uint> neighbor_buf;
            const std::vector<uint>* wake_neighbors = &neighbor_buf;
            wake_neighbors = &GetCachedUniqueNeighbors(res.v, neighbor_buf);

            std::vector<uint> mc_u_buf(L);
            for (uint u : *wake_neighbors) {
                if (limit_hot_start_wake_to_domain &&
                    (u >= hot_start_wake_domain.size() || !hot_start_wake_domain[u])) {
                    continue;
                }

                for (uint d = 0; d < L; d++) {
                    mc_u_buf[d] = max_c[u][d].load(std::memory_order_relaxed);
                }

                bool v_super_dominates = false;
                for (uint si = 0; si < res.new_scv_count; si++) {
                    const uint* k_v = res.new_scvs_flat.data() + static_cast<size_t>(si) * L;
                    bool dom = true;
                    for (uint d = 0; d < L; d++) {
                        if (k_v[d] < mc_u_buf[d]) { dom = false; break; }
                    }
                    if (dom) { v_super_dominates = true; break; }
                }
                if (v_super_dominates) continue;

                const SkylineSet& u_set = scv_sets[u];
                const std::vector<unsigned long long>* affected_u = nullptr;
                if (hot_start_scv_filter_enabled) {
                    auto it = hot_start_scv_bits_by_node.find(u);
                    if (it == hot_start_scv_bits_by_node.end() || it->second.empty()) {
                        continue;
                    }
                    affected_u = &it->second;
                }
                bool v_lost_support_for_u = false;
                uint u_count = u_set.NumSCVs();
                for (uint si = 0; si < u_count; si++) {
                    const uint* k_u = u_set.GetSCVFlat(si);
                    if (affected_u) {
                        size_t word = static_cast<size_t>(si >> 6);
                        if (word >= affected_u->size() ||
                            (((*affected_u)[word] & (1ULL << (si & 63))) == 0)) {
                            continue;
                        }
                    }
                    if (hot_start_scv_filter_enabled && has_lower_bounds &&
                        u < lower_bounds_snapshot.size() &&
                        u < lower_bounds_active.size() &&
                        lower_bounds_active[u] &&
                        lower_bounds_snapshot[u].IsDominatedRaw(k_u, L)) {
                        continue;
                    }
                    bool supported_before = false;
                    for (uint vsi = 0; vsi < res.old_scv_count; vsi++) {
                        const uint* k_v = res.old_scvs_flat.data() + static_cast<size_t>(vsi) * L;
                        if (SkylineSet::DominatesRaw(k_v, k_u, L)) {
                            supported_before = true;
                            break;
                        }
                    }
                    if (!supported_before) continue;

                    bool supported_after = false;
                    for (uint vsi = 0; vsi < res.new_scv_count; vsi++) {
                        const uint* k_v = res.new_scvs_flat.data() + static_cast<size_t>(vsi) * L;
                        if (SkylineSet::DominatesRaw(k_v, k_u, L)) {
                            supported_after = true;
                            break;
                        }
                    }
                    if (!supported_after) {
                        v_lost_support_for_u = true;
                        if (hot_start_scv_filter_enabled) {
                            local.hot_indices.emplace_back(u, si);
                        }
                    }
                }

                if (!v_lost_support_for_u) continue;

                local.next_candidates.push_back(u);
            }
        };
        if (use_parallel) {
#pragma omp parallel for schedule(dynamic, 1)
            for (long long ri = 0; ri < static_cast<long long>(results.size()); ri++) {
                WakeLocal& local = wake_locals[omp_get_thread_num()];
                process_wake_result(results[static_cast<size_t>(ri)], local);
            }
        } else {
            WakeLocal& local = wake_locals[0];
            for (RoundNodeResult& res : results) {
                process_wake_result(res, local);
            }
        }

        for (WakeLocal& local : wake_locals) {
            for (uint u : local.next_candidates) add_next(u);
            for (const auto& item : local.hot_indices) {
                mark_hot_idx(item.first, item.second);
            }
        }
        for (uint v : next_frontier) {
            is_in_inbox[v].store(true, std::memory_order_relaxed);
        }
        for (uint v : frontier) {
            if (v < n && !next_seen[v]) is_in_inbox[v].store(false, std::memory_order_relaxed);
        }
        frontier.swap(next_frontier);
        for (uint v : frontier) {
            is_in_inbox[v].store(false, std::memory_order_relaxed);
        }
        for (uint v : next_seen_nodes) {
            next_seen[v] = 0;
        }
    }
}

void IterativeMLCore::Decompose(bool hot_start) {
    if (!hot_start) {
        DisableFixedRefinementDim();
        InitializeBounds();
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    long long initial_tasks = active_task_count.load(std::memory_order_relaxed);
    unsigned long long round_task_threshold = HotStartRoundTaskThreshold();
    unsigned long long round_scv_threshold = HotStartRoundSCVThreshold();
    unsigned long long initial_scv_work = 0;

    if (initial_tasks > 0 &&
        static_cast<unsigned long long>(initial_tasks) <= round_task_threshold) {
        std::vector<uint> queued_nodes;
        {
            std::lock_guard<std::mutex> lock(shared_queue_mtx);
            queued_nodes.assign(shared_queue.begin(), shared_queue.end());
        }
        for (uint v : queued_nodes) {
            if (v >= n) continue;
            initial_scv_work += scv_sets[v].NumSCVs();
            if (initial_scv_work >= round_scv_threshold) break;
        }
    }
    bool use_round_parallel =
    !hot_start ||
    (initial_tasks > 0 &&
        (static_cast<unsigned long long>(initial_tasks) > round_task_threshold ||
        initial_scv_work >= round_scv_threshold));

    DecomposeHotStartByRounds(use_round_parallel);

    shared_queue.clear();
    for (uint i = 0; i < n; i++) is_in_inbox[i].store(false, std::memory_order_relaxed);
    auto t_iter = std::chrono::high_resolution_clock::now();
    double iter_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_iter - t_start).count() / 1000.0;
    if (hot_start) last_iter_ms = iter_ms;
}

const SkylineSet& IterativeMLCore::GetNodeSkyline(uint v) const {
    return scv_sets[v];
}

const std::vector<SkylineSet>& IterativeMLCore::GetAllSkylines() const {
    return scv_sets;
}
