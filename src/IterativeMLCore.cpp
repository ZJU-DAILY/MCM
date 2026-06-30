#include "IterativeMLCore.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <omp.h>

namespace {
unsigned long long HotStartRoundTaskThreshold() {
    return 8;
}

unsigned long long HotStartRoundSCVThreshold() {
    return 512;
}

unsigned long long ReadMemAvailableKB() {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    }
    std::fclose(f);
    return kb;
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

void ApplyRootLowerBound(const SkylineSet* completed_roots,
                         const std::vector<uint>& current_k,
                         uint split_dim,
                         uint L,
                         uint& lo) {
    if (!completed_roots) return;
    uint root_count = completed_roots->NumSCVs();
    for (uint ri = 0; ri < root_count; ri++) {
        const uint* root = completed_roots->GetSCVFlat(ri);
        bool covers_other_dims = true;
        for (uint d = 0; d < L; d++) {
            if (d == split_dim) continue;
            if (root[d] < current_k[d]) {
                covers_other_dims = false;
                break;
            }
        }
        if (covers_other_dims) {
            lo = std::max(lo, root[split_dim] + 1);
        }
    }
}

void ApplyValidResultLowerBound(const SkylineSet& valid_results,
                                const std::vector<uint>& current_k,
                                uint split_dim,
                                uint L,
                                uint& lo) {
    uint result_count = valid_results.NumSCVs();
    for (uint ri = 0; ri < result_count; ri++) {
        const uint* result = valid_results.GetSCVFlat(ri);
        bool covers_other_dims = true;
        for (uint d = 0; d < L; d++) {
            if (d == split_dim) continue;
            if (result[d] < current_k[d]) {
                covers_other_dims = false;
                break;
            }
        }
        if (covers_other_dims) {
            lo = std::max(lo, result[split_dim]);
        }
    }
}

bool ScanLazyNeighborSCVsFrom(const SCVSnapshot& u_scvs,
                              const std::vector<uint>& k_vec,
                              uint L,
                              uint start,
                              LazyVerifyState::NeighborInfo& info) {
    info.is_supporter = false;
    info.support_scv_idx = UINT_MAX;
    uint count = u_scvs.NumSCVs(L);
    if (start > count) start = count;
    info.fail_scv_count = start;
    info.candidate_scv_start = start;

    const uint* k_raw = k_vec.data();
    for (uint si = start; si < count; si++) {
        if (SkylineSet::DominatesRaw(u_scvs.GetRaw(si, L), k_raw, L)) {
            info.is_supporter = true;
            info.support_scv_idx = si;
            info.fail_scv_count = si;
            info.candidate_scv_start = si + 1;
            return true;
        }
        info.fail_scv_count = si + 1;
        info.candidate_scv_start = si + 1;
    }
    return false;
}

bool ScanLazyNeighborSCVs(const SCVSnapshot& u_scvs,
                          const std::vector<uint>& k_vec,
                          uint L,
                          LazyVerifyState::NeighborInfo& info) {
    return ScanLazyNeighborSCVsFrom(u_scvs, k_vec, L, 0, info);
}

void MarkLazyNeighborFailed(const SCVSnapshot& u_scvs,
                            uint L,
                            LazyVerifyState::NeighborInfo& info) {
    info.is_supporter = false;
    info.support_scv_idx = UINT_MAX;
    uint count = u_scvs.NumSCVs(L);
    info.fail_scv_count = count;
    info.candidate_scv_start = count;
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

void IterativeMLCore::UpdateMaxCIncremental(uint v, const std::vector<uint>& k_vec) {
    for (uint l = 0; l < L; l++) {
        uint cur = max_c[v][l].load(std::memory_order_relaxed);
        if (k_vec[l] > cur) {
            max_c[v][l].store(k_vec[l], std::memory_order_relaxed);
        }
    }
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
            bool dominated = scv_sets[u].IsDominatedRaw(k_raw, L);
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

            if (scv_sets[u].IsDominatedRaw(k_raw, L)) support_count++;
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

            if (SnapshotDominates(snapshot[u], k_raw, L)) support_count++;
        }

        if (support_count < required_k) return false;
    }
    return true;
}

bool IterativeMLCore::CheckSupportDim(uint v, const std::vector<uint>& k_vec, uint dim) const {
    if (dim >= L) return CheckSupport(v, k_vec);
    const uint* k_raw = k_vec.data();
    uint required_k = k_raw[dim];
    if (required_k == 0) return true;

    uint** adj_lst = mg.GetGraph(dim).GetAdjLst();
    uint degree = adj_lst[v][0];
    if (degree < required_k) return false;

    uint support_count = 0;
    for (uint i = 1; i <= degree; i++) {
        if (support_count >= required_k) break;
        if (support_count + (degree - i + 1) < required_k) return false;

        uint u = adj_lst[v][i];
        bool can_support = MaxCAtLeast(max_c, u, k_raw, L);
        if (!can_support) continue;

        scv_rw_locks[u]->lock_shared();
        bool dominated = scv_sets[u].IsDominatedRaw(k_raw, L);
        scv_rw_locks[u]->unlock_shared();
        if (dominated) support_count++;
    }

    return support_count >= required_k;
}

bool IterativeMLCore::CheckSupportDimNoLock(uint v, const std::vector<uint>& k_vec, uint dim) const {
    if (dim >= L) return CheckSupportNoLock(v, k_vec);
    const uint* k_raw = k_vec.data();
    uint required_k = k_raw[dim];
    if (required_k == 0) return true;

    uint** adj_lst = mg.GetGraph(dim).GetAdjLst();
    uint degree = adj_lst[v][0];
    if (degree < required_k) return false;

    uint support_count = 0;
    for (uint i = 1; i <= degree; i++) {
        if (support_count >= required_k) break;
        if (support_count + (degree - i + 1) < required_k) return false;

        uint u = adj_lst[v][i];
        bool can_support = MaxCAtLeast(max_c, u, k_raw, L);
        if (!can_support) continue;

        if (scv_sets[u].IsDominatedRaw(k_raw, L)) support_count++;
    }

    return support_count >= required_k;
}

bool IterativeMLCore::CheckSupportDimFromSnapshot(uint v, const std::vector<uint>& k_vec,
                                                   const NeighborSnapshot& snapshot,
                                                   uint dim) const {
    if (dim >= L) return CheckSupportFromSnapshot(v, k_vec, snapshot);
    const uint* k_raw = k_vec.data();
    uint required_k = k_raw[dim];
    if (required_k == 0) return true;

    uint** adj_lst = mg.GetGraph(dim).GetAdjLst();
    uint degree = adj_lst[v][0];
    if (degree < required_k) return false;

    uint support_count = 0;
    for (uint i = 1; i <= degree; i++) {
        if (support_count >= required_k) break;
        if (support_count + (degree - i + 1) < required_k) return false;

        uint u = adj_lst[v][i];
        bool can_support = MaxCAtLeast(max_c, u, k_raw, L);
        if (!can_support) continue;

        if (SnapshotDominates(snapshot[u], k_raw, L)) support_count++;
    }

    return support_count >= required_k;
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

uint IterativeMLCore::GetDimLowerBound(uint v, uint d, const std::vector<uint>& current_k) const {
    if (!has_lower_bounds ||
        v >= lower_bounds_snapshot.size() ||
        v >= lower_bounds_active.size() ||
        !lower_bounds_active[v]) {
        return 0;
    }

    const SkylineSet& lb_set = lower_bounds_snapshot[v];
    uint lb_count = lb_set.NumSCVs();
    if (lb_count == 0) return 0;

    uint best_lb = 0;
    for (uint i = 0; i < lb_count; i++) {
        const uint* lb = lb_set.GetSCVFlat(i);
        bool applicable = true;
        for (uint dd = 0; dd < L; dd++) {
            if (dd != d && current_k[dd] > lb[dd]) {
                applicable = false;
                break;
            }
        }
        if (applicable && lb[d] > best_lb) {
            best_lb = lb[d];
        }
    }
    return best_lb;
}

uint IterativeMLCore::ComputeConditionedSnapshotUpperBound(
    uint v,
    uint split_dim,
    const std::vector<uint>& current_k,
    const NeighborSnapshot& snapshot,
    uint current_hi) const {
    if (split_dim >= L || current_hi == 0) return current_hi;

    uint** adj_lst = mg.GetGraph(split_dim).GetAdjLst();
    uint degree = adj_lst[v][0];
    if (degree == 0) return 0;

    thread_local std::vector<uint> counts;
    size_t needed = static_cast<size_t>(current_hi) + 1;
    if (counts.size() < needed) counts.resize(needed);
    std::fill(counts.begin(), counts.begin() + needed, 0);

    for (uint i = 1; i <= degree; i++) {
        uint u = adj_lst[v][i];
        const auto& u_scvs = snapshot[u];
        uint scv_count = u_scvs.NumSCVs(L);
        uint best = 0;
        for (uint si = 0; si < scv_count; si++) {
            const uint* raw = u_scvs.GetRaw(si, L);
            bool covers_other_dims = true;
            for (uint d = 0; d < L; d++) {
                if (d == split_dim) continue;
                if (raw[d] < current_k[d]) {
                    covers_other_dims = false;
                    break;
                }
            }
            if (covers_other_dims && raw[split_dim] > best) {
                best = raw[split_dim];
            }
        }
        if (best > 0) counts[std::min(best, current_hi)]++;
    }

    uint count_ge = 0;
    for (int h = static_cast<int>(current_hi); h >= 1; h--) {
        count_ge += counts[h];
        if (count_ge >= static_cast<uint>(h)) return static_cast<uint>(h);
    }
    return 0;
}

void IterativeMLCore::PrecomputeNeighborMaxC(uint v, NeighborMaxCBuffer& buf) const {
    buf.Init(L);
    bool collect_fixed_dim =
        has_lower_bounds && fixed_refinement_dim_enabled && fixed_refinement_dim < L;
    if (neighbor_hindex_cap_cache_enabled &&
        !collect_fixed_dim &&
        v < neighbor_hindex_cap_cache.size() &&
        !neighbor_hindex_cap_cache[v].empty()) {
        buf.hindex_cap = neighbor_hindex_cap_cache[v];
        buf.has_hindex_cap = true;
        return;
    }
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

void IterativeMLCore::BuildNeighborHIndexCapCache(const std::vector<uint>& nodes) {
    neighbor_hindex_cap_cache.clear();
    neighbor_hindex_cap_cache.resize(n);

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        uint v = nodes[idx];
        auto& node_cache = neighbor_hindex_cap_cache[v];
        node_cache.assign(L, 0);
        std::vector<uint> counts;
        for (uint l = 0; l < L; l++) {
            uint** adj_lst = mg.GetGraph(l).GetAdjLst();
            uint degree = adj_lst[v][0];
            if (degree == 0) continue;
            counts.assign(degree + 1, 0);
            for (uint i = 1; i <= degree; i++) {
                uint u = adj_lst[v][i];
                uint val = max_c[u][l].load(std::memory_order_relaxed);
                if (val > 0) counts[std::min(degree, val)]++;
            }
            uint count_ge = 0;
            for (int h = static_cast<int>(degree); h >= 1; h--) {
                count_ge += counts[h];
                if (count_ge >= static_cast<uint>(h)) {
                    node_cache[l] = static_cast<uint>(h);
                    break;
                }
            }
        }
    }

    neighbor_hindex_cap_cache_enabled = true;
}

void IterativeMLCore::ClearNeighborHIndexCapCache() {
    neighbor_hindex_cap_cache_enabled = false;
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
    if (buf.has_hindex_cap) {
        for (uint l = 0; l < L; l++) {
            uint bound = current_k[l];
            if (bound == 0) {
                h_vec[l] = 0;
                continue;
            }
            h_vec[l] = std::min(bound, buf.hindex_cap[l]);
            if (h_vec[l] < current_k[l]) h_pruned = true;
        }
        return;
    }
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

    if (buf.has_hindex_cap && g < buf.hindex_cap.size()) {
        return buf.hindex_cap[g] < required;
    }
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

void IterativeMLCore::DFSSplit(uint v, std::vector<uint>& current_k, uint last_dim,
                               SkylineSet& valid_results, const NeighborMaxCBuffer& buf) const {
    if (valid_results.IsDominatedRaw(current_k.data(), L)) return;

    thread_local std::vector<uint> h_vec;
    h_vec.resize(L);
    bool h_pruned = false;
    ComputeHFromBuffer(current_k, buf, h_vec, h_pruned);
    if (h_pruned) h_pruned = ApplyFixedRefinementDim(current_k, h_vec);

    if (h_pruned) {
        std::vector<uint> k_prime = current_k;
        for (uint l = 0; l < L; l++) k_prime[l] = h_vec[l];
        DFSSplit(v, k_prime, 0, valid_results, buf);
        return;
    }

    if (CheckSupport(v, current_k)) {
        valid_results.Insert(current_k);
        return;
    }

    for (int d = (int)L - 1; d >= (int)last_dim; --d) {
        if (IsRefinementDimFixed(static_cast<uint>(d))) continue;
        if (current_k[d] > 0) {
            std::vector<uint> safe_step = current_k;
            safe_step[d] = current_k[d] - 1;
            DFSSplit(v, safe_step, d, valid_results, buf);
        }
    }
}

void IterativeMLCore::SplitSCV(uint v, const std::vector<uint>& k_vec, SkylineSet& shared_results,
                                const VerifyState* precomputed_state,
                                SCVVisitTracker* visited,
                                const NeighborMaxCBuffer* precomputed_buf,
                                const NeighborSnapshot* precomputed_snapshot,
                                SkylineSet* completed_roots) const {
    (void)precomputed_state;
    DFSSplitLazyRoot(v, k_vec, shared_results, visited, precomputed_buf,
                     precomputed_snapshot, completed_roots);
}

bool IterativeMLCore::VerifyAndClassify(uint v, const std::vector<uint>& k_vec, VerifyState& state,
                                         const NeighborSnapshot* snapshot) const {
    bool all_supported = true;

    for (uint l = 0; l < L; l++) {
        uint required_k = k_vec[l];
        if (required_k == 0) {
            state.supporters[l].clear();
            state.failers[l].clear();
            state.candidates[l].clear();
            continue;
        }

        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];

        state.supporters[l].clear();
        state.failers[l].clear();
        state.candidates[l].clear();

        if (degree < required_k) {
            for (uint i = 1; i <= degree; i++) {
                state.failers[l].push_back(adj_lst[v][i]);
            }
            all_supported = false;
            continue;
        }

        bool early_stop = false;
        for (uint i = 1; i <= degree; i++) {
            uint u = adj_lst[v][i];

            bool can_support = MaxCAtLeast(max_c, u, k_vec.data(), L);

            if (!can_support) {
                state.failers[l].push_back(u);
                continue;
            }

            bool dominated;
            if (snapshot) {
                const auto& u_scvs = (*snapshot)[u];
                dominated = SnapshotDominates(u_scvs, k_vec.data(), L);
            } else {
                scv_rw_locks[u]->lock_shared();
                dominated = scv_sets[u].IsDominatedRaw(k_vec.data(), L);
                scv_rw_locks[u]->unlock_shared();
            }

            if (dominated) {
                state.supporters[l].push_back(u);
                if (state.supporters[l].size() >= required_k) {
                    for (uint j = i + 1; j <= degree; j++) {
                        state.candidates[l].push_back(adj_lst[v][j]);
                    }
                    early_stop = true;
                    break;
                }
            } else {
                state.failers[l].push_back(u);
            }
        }

        if (state.supporters[l].size() < required_k) {
            all_supported = false;
        }
    }

    return all_supported;
}

bool IterativeMLCore::VerifyIncremental(uint v, const std::vector<uint>& child_k, uint changed_dim,
                                         const VerifyState& parent_state, VerifyState& child_state,
                                         const NeighborSnapshot* snapshot) const {
    bool all_supported = true;

    for (uint l = 0; l < L; l++) {
        uint required_k = child_k[l];
        if (required_k == 0) {
            child_state.supporters[l].clear();
            child_state.failers[l].clear();
            child_state.candidates[l].clear();
            continue;
        }

        child_state.supporters[l] = parent_state.supporters[l];

        if (child_state.supporters[l].size() >= required_k) {
            child_state.failers[l] = parent_state.failers[l];
            child_state.candidates[l] = parent_state.candidates[l];
            continue;
        }

        child_state.failers[l].clear();
        child_state.candidates[l].clear();

        bool enough = false;
        for (auto it = parent_state.failers[l].begin(); it != parent_state.failers[l].end(); ++it) {
            uint u = *it;

            bool can_support = MaxCAtLeast(max_c, u, child_k.data(), L);

            if (!can_support) {
                child_state.failers[l].push_back(u);
                continue;
            }

            bool dominated;
            if (snapshot) {
                const auto& u_scvs = (*snapshot)[u];
                dominated = SnapshotDominates(u_scvs, child_k.data(), L);
            } else {
                scv_rw_locks[u]->lock_shared();
                dominated = scv_sets[u].IsDominatedRaw(child_k.data(), L);
                scv_rw_locks[u]->unlock_shared();
            }

            if (dominated) {
                child_state.supporters[l].push_back(u);
                if (child_state.supporters[l].size() >= required_k) {
                    child_state.failers[l].insert(child_state.failers[l].end(),
                                                  it + 1, parent_state.failers[l].end());
                    child_state.candidates[l] = parent_state.candidates[l];
                    enough = true;
                    break;
                }
            } else {
                child_state.failers[l].push_back(u);
            }
        }

        if (enough) continue;

        for (auto it = parent_state.candidates[l].begin(); it != parent_state.candidates[l].end(); ++it) {
            uint u = *it;

            bool can_support = true;
            for (uint j = 0; j < L; j++) {
                if (max_c[u][j].load(std::memory_order_relaxed) < child_k[j]) {
                    can_support = false;
                    break;
                }
            }

            if (!can_support) {
                child_state.failers[l].push_back(u);
                continue;
            }

            bool dominated;
            if (snapshot) {
                const auto& u_scvs = (*snapshot)[u];
                dominated = SnapshotDominates(u_scvs, child_k.data(), L);
            } else {
                scv_rw_locks[u]->lock_shared();
                dominated = scv_sets[u].IsDominatedRaw(child_k.data(), L);
                scv_rw_locks[u]->unlock_shared();
            }

            if (dominated) {
                child_state.supporters[l].push_back(u);
                if (child_state.supporters[l].size() >= required_k) {
                    child_state.candidates[l].insert(child_state.candidates[l].end(),
                                                     it + 1, parent_state.candidates[l].end());
                    enough = true;
                    break;
                }
            } else {
                child_state.failers[l].push_back(u);
            }
        }

        if (child_state.supporters[l].size() < required_k) {
            all_supported = false;
        }
    }

    return all_supported;
}

void IterativeMLCore::DFSSplitRoot(uint v, const std::vector<uint>& current_k, SkylineSet& valid_results,
                                    const VerifyState* precomputed_state,
                                    const NeighborSnapshot* ext_snapshot) const {
    if (valid_results.IsDominatedRaw(current_k.data(), L)) {
        return;
    }

    thread_local NeighborMaxCBuffer buf;
    PrecomputeNeighborMaxC(v, buf);

    thread_local std::vector<uint> h_vec;
    h_vec.resize(L);
    bool h_pruned = false;
    ComputeHFromBuffer(current_k, buf, h_vec, h_pruned);
    if (h_pruned) h_pruned = ApplyFixedRefinementDim(current_k, h_vec);

    const NeighborSnapshot* snap_ptr = ext_snapshot;

    VerifyState root_state(L);
    bool is_valid;
    std::vector<uint> effective_k;

    if (h_pruned) {
        effective_k.resize(L);
        for (uint l = 0; l < L; l++) effective_k[l] = h_vec[l];
        is_valid = VerifyAndClassify(v, effective_k, root_state, snap_ptr);
    } else {
        effective_k = current_k;
        if (precomputed_state) {
            root_state = *precomputed_state;
            is_valid = false;
        } else {
            is_valid = VerifyAndClassify(v, effective_k, root_state, snap_ptr);
        }
    }

    if (is_valid) {
        valid_results.Insert(effective_k);
        return;
    }

    for (int d = (int)L - 1; d >= 0; --d) {
        if (IsRefinementDimFixed(static_cast<uint>(d))) continue;
        if (effective_k[d] > 0) {
            std::vector<uint> jumped = effective_k;

            thread_local std::vector<uint> h_bs;
            h_bs.resize(L);
            uint lo = GetDimLowerBound(v, d, effective_k);
            uint hi = effective_k[d] - 1;
            ApplyValidResultLowerBound(valid_results, effective_k, d, L, lo);
            if (lo > hi) continue;
            while (lo < hi) {
                uint mid = lo + (hi - lo + 1) / 2;
                jumped[d] = mid;
                bool bp = false;
                ComputeHFromBuffer(jumped, buf, h_bs, bp);
                if (bp && h_bs[d] < mid) hi = h_bs[d];
                else lo = mid;
            }
            uint h_target = lo;

            jumped[d] = h_target;
            DFSSplitIncremental(v, jumped, d, valid_results, root_state, d, buf, snap_ptr);
        }
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
    if (enable_lazy_support_seed && TrySeedLazyFailersFromSupportData(v, k_vec, snapshot, state)) {
        bool all_supported = true;
        for (uint l = 0; l < L; l++) {
            uint required_k = k_vec[l];
            if (required_k > 0 && state.layers[l].support_count < required_k) {
                all_supported = false;
                break;
            }
        }
        return all_supported;
    }

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

bool IterativeMLCore::TrySeedLazyFailersFromSupportData(uint v, const std::vector<uint>& k_vec,
                                                         const NeighborSnapshot& snapshot,
                                                         LazyVerifyState& state) const {
    if (!enable_support_pruning || v >= support_data.size()) return false;

    const SCVSupportEntry* entry = nullptr;
    bool exact_entry = false;
    for (const auto& item : support_data[v]) {
        if (item.scv == k_vec) {
            entry = &item;
            exact_entry = true;
            break;
        }
    }
    if (!entry) return false;

    state.Clear();
    if (exact_entry) {
        bool all_supported = true;

        for (uint l = 0; l < L; l++) {
            uint required_k = k_vec[l];
            if (required_k == 0) continue;

            auto& ls = state.layers[l];
            uint** adj_lst = mg.GetGraph(l).GetAdjLst();
            uint degree = adj_lst[v][0];
            ls.neighbor_info.resize(degree + 1);

            for (uint pos : entry->dim_supporter_positions[l]) {
                if (pos == 0 || pos > degree) continue;
                uint u = adj_lst[v][pos];
                auto& ni = ls.GetInfo(pos);
                const auto& u_scvs = snapshot[u];
                bool found_support = !u_scvs.Empty() && ScanLazyNeighborSCVs(u_scvs, k_vec, L, ni);
                if (found_support) {
                    ls.support_nodes.push_back(pos);
                    ls.support_count++;
                    if (ls.support_count >= required_k) break;
                } else {
                    ni.is_supporter = false;
                    ni.support_scv_idx = UINT_MAX;
                    ls.fail_nodes.push_back(pos);
                }
            }

            if (ls.support_count < required_k) {
                for (uint pos : entry->dim_failer_positions[l]) {
                    if (pos == 0 || pos > degree) continue;
                    uint u = adj_lst[v][pos];
                    auto& ni = ls.GetInfo(pos);
                    MarkLazyNeighborFailed(snapshot[u], L, ni);
                    ls.fail_nodes.push_back(pos);
                }
                all_supported = false;
            }
        }

        return all_supported;
    }

    thread_local std::vector<uint> support_tag;
    thread_local std::vector<uint> fail_tag;
    thread_local uint epoch = 1;
    if (support_tag.size() < n) {
        support_tag.assign(n, 0);
        fail_tag.assign(n, 0);
    }
    epoch++;
    if (epoch == 0) {
        std::fill(support_tag.begin(), support_tag.end(), 0);
        std::fill(fail_tag.begin(), fail_tag.end(), 0);
        epoch = 1;
    }

    for (uint l = 0; l < L; l++) {
        uint required_k = k_vec[l];
        if (required_k == 0) continue;

        auto& ls = state.layers[l];
        uint** adj_lst = mg.GetGraph(l).GetAdjLst();
        uint degree = adj_lst[v][0];
        ls.neighbor_info.resize(degree + 1);

        for (uint u : entry->dim_supporters[l]) support_tag[u] = epoch;
        for (uint u : entry->dim_failers[l]) fail_tag[u] = epoch;

        bool enough = false;
        thread_local std::vector<uint> unknown_positions;
        thread_local std::vector<uint> seeded_fail_positions;
        unknown_positions.clear();
        seeded_fail_positions.clear();
        for (uint i = 1; i <= degree; i++) {
            uint u = adj_lst[v][i];
            auto& ni = ls.GetInfo(i);

            if (fail_tag[u] == epoch) {
                if (exact_entry) {
                    MarkLazyNeighborFailed(snapshot[u], L, ni);
                    ls.fail_nodes.push_back(i);
                } else {
                    seeded_fail_positions.push_back(i);
                }
                continue;
            }

            if (support_tag[u] == epoch) {
                const auto& u_scvs = snapshot[u];
                bool found_support = !u_scvs.Empty() && ScanLazyNeighborSCVs(u_scvs, k_vec, L, ni);
                if (found_support) {
                    ls.support_nodes.push_back(i);
                    ls.support_count++;
                    if (ls.support_count >= required_k) {
                        enough = true;
                        break;
                    }
                } else {
                    ni.is_supporter = false;
                    ni.support_scv_idx = UINT_MAX;
                    ls.fail_nodes.push_back(i);
                }
            } else {
                unknown_positions.push_back(i);
            }
        }

        if (enough) {
            for (uint pos : seeded_fail_positions) ls.candidate_nodes.push_back(pos);
            for (uint pos : unknown_positions) ls.candidate_nodes.push_back(pos);
            for (uint j = ls.support_nodes.back() + 1; j <= degree; j++) {
                uint u = adj_lst[v][j];
                if (fail_tag[u] != epoch && support_tag[u] != epoch) {
                    ls.candidate_nodes.push_back(j);
                }
            }
            if (!exact_entry) {
                for (uint j = ls.support_nodes.back() + 1; j <= degree; j++) {
                    uint u = adj_lst[v][j];
                    if (fail_tag[u] == epoch) ls.candidate_nodes.push_back(j);
                }
            }
            continue;
        }

        auto try_positions = [&](const std::vector<uint>& positions) {
            for (size_t idx = 0; idx < positions.size(); idx++) {
                uint pos = positions[idx];
                uint u = adj_lst[v][pos];
                auto& ni = ls.GetInfo(pos);
                const auto& u_scvs = snapshot[u];
                bool found_support = !u_scvs.Empty() && ScanLazyNeighborSCVs(u_scvs, k_vec, L, ni);
                if (found_support) {
                    ls.support_nodes.push_back(pos);
                    ls.support_count++;
                    if (ls.support_count >= required_k) {
                        for (size_t rest = idx + 1; rest < positions.size(); rest++) {
                            ls.candidate_nodes.push_back(positions[rest]);
                        }
                        return true;
                    }
                } else {
                    ni.is_supporter = false;
                    ni.support_scv_idx = UINT_MAX;
                    ls.fail_nodes.push_back(pos);
                }
            }
            return false;
        };

        if (!exact_entry && try_positions(seeded_fail_positions)) {
            for (uint pos : unknown_positions) ls.candidate_nodes.push_back(pos);
            continue;
        }

        for (size_t idx = 0; idx < unknown_positions.size(); idx++) {
            uint pos = unknown_positions[idx];
            uint u = adj_lst[v][pos];
            auto& ni = ls.GetInfo(pos);
            const auto& u_scvs = snapshot[u];
            bool found_support = !u_scvs.Empty() && ScanLazyNeighborSCVs(u_scvs, k_vec, L, ni);
            if (found_support) {
                ls.support_nodes.push_back(pos);
                ls.support_count++;
                if (ls.support_count >= required_k) {
                    for (size_t rest = idx + 1; rest < unknown_positions.size(); rest++) {
                        ls.candidate_nodes.push_back(unknown_positions[rest]);
                    }
                    break;
                }
            } else {
                ni.is_supporter = false;
                ni.support_scv_idx = UINT_MAX;
                ls.fail_nodes.push_back(pos);
            }
        }
    }

    return true;
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

void IterativeMLCore::DFSSplitLazyIncremental(uint v, std::vector<uint>& current_k, uint last_dim,
                                              SkylineSet& valid_results,
                                              const LazyVerifyState& parent_state,
                                              uint changed_dim, const NeighborMaxCBuffer& buf,
                                              const NeighborSnapshot& snapshot,
                                              SCVVisitTracker* visited,
                                              uint depth,
                                              SkylineSet* completed_roots) const {
    if (valid_results.IsDominatedRaw(current_k.data(), L)) return;
    if (RootDominatesCandidate(completed_roots, current_k, L)) return;

    thread_local std::vector<uint> h_entry;
    h_entry.resize(L);
    bool h_pruned = false;
    ComputeHFromBuffer(current_k, buf, h_entry, h_pruned);
    if (FixedRefinementSubtreeFails(current_k, buf)) {
        if (completed_roots) completed_roots->Insert(current_k);
        return;
    }
    if (h_pruned) h_pruned = ApplyFixedRefinementDim(current_k, h_entry);
    if (h_pruned) {
        for (uint l = 0; l < L; l++) current_k[l] = h_entry[l];
        bool all_zero = true;
        for (uint l = 0; l < L; l++) {
            if (current_k[l] > 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero || valid_results.IsDominatedRaw(current_k.data(), L)) return;
        if (RootDominatesCandidate(completed_roots, current_k, L)) return;
    }
    if (visited && !visited->Insert(current_k)) return;

    thread_local std::vector<std::unique_ptr<LazyVerifyState>> state_pool;
    if (state_pool.size() <= depth) {
        while (state_pool.size() <= depth) {
            state_pool.emplace_back(std::make_unique<LazyVerifyState>(L));
        }
    }
    LazyVerifyState& current_state = *state_pool[depth];
    bool is_valid = LazyVerifyIncremental(v, current_k, changed_dim, snapshot, parent_state, current_state);

    if (is_valid) {
        valid_results.Insert(current_k);
        return;
    }

    for (int d = (int)L - 1; d >= (int)last_dim; --d) {
        if (IsRefinementDimFixed(static_cast<uint>(d))) continue;
        if (current_k[d] > 0) {
            std::vector<uint> jumped = current_k;

            thread_local std::vector<uint> h_bs;
            h_bs.resize(L);
            uint lo = GetDimLowerBound(v, d, current_k);
            uint hi = current_k[d] - 1;
            ApplyValidResultLowerBound(valid_results, current_k, d, L, lo);
            ApplyRootLowerBound(completed_roots, current_k, d, L, lo);
            if (lo > hi) continue;
            while (lo < hi) {
                uint mid = lo + (hi - lo + 1) / 2;
                jumped[d] = mid;
                bool bp = false;
                ComputeHFromBuffer(jumped, buf, h_bs, bp);
                if (bp && h_bs[d] < mid) hi = h_bs[d];
                else lo = mid;
            }
            uint h_target = lo;

            jumped[d] = h_target;
            if (valid_results.IsDominatedRaw(jumped.data(), L)) continue;
            if (RootDominatesCandidate(completed_roots, jumped, L)) continue;
            if (visited && visited->Contains(jumped)) continue;

            DFSSplitLazyIncremental(v, jumped, d, valid_results, current_state, d, buf,
                                    snapshot, visited, depth + 1,
                                    completed_roots);
        }
    }
}

void IterativeMLCore::DFSSplitLazyRoot(uint v, const std::vector<uint>& current_k,
                                        SkylineSet& valid_results,
                                        SCVVisitTracker* visited,
                                        const NeighborMaxCBuffer* precomputed_buf,
                                        const NeighborSnapshot* precomputed_snapshot,
                                        SkylineSet* completed_roots) const {
    if (valid_results.IsDominatedRaw(current_k.data(), L)) return;
    if (RootDominatesCandidate(completed_roots, current_k, L)) return;

    thread_local NeighborMaxCBuffer local_buf;
    const NeighborMaxCBuffer* buf_ptr = precomputed_buf;
    if (!buf_ptr) {
        PrecomputeNeighborMaxC(v, local_buf);
        buf_ptr = &local_buf;
    }

    thread_local std::vector<uint> h_vec;
    h_vec.resize(L);
    bool h_pruned = false;
    ComputeHFromBuffer(current_k, *buf_ptr, h_vec, h_pruned);
    if (FixedRefinementSubtreeFails(current_k, *buf_ptr)) {
        if (completed_roots) completed_roots->Insert(current_k);
        return;
    }
    if (h_pruned) h_pruned = ApplyFixedRefinementDim(current_k, h_vec);

    std::vector<uint> effective_k = current_k;
    if (h_pruned) {
        for (uint l = 0; l < L; l++) effective_k[l] = h_vec[l];
        if (valid_results.IsDominatedRaw(effective_k.data(), L)) return;
        if (RootDominatesCandidate(completed_roots, effective_k, L)) return;
    }
    if (visited && !visited->Insert(effective_k)) return;

    thread_local NeighborSnapshot local_snapshot;
    const NeighborSnapshot* snapshot_ptr = precomputed_snapshot;
    if (!snapshot_ptr) {
        BuildNeighborSnapshotNoLock(v, local_snapshot);
        snapshot_ptr = &local_snapshot;
    }

    thread_local std::unique_ptr<LazyVerifyState> root_state_storage;
    if (!root_state_storage || root_state_storage->layers.size() != L) {
        root_state_storage = std::make_unique<LazyVerifyState>(L);
    }
    LazyVerifyState& root_state = *root_state_storage;
    root_state.Init(n);
    bool partial_root_verify =
        has_lower_bounds && fixed_refinement_dim_enabled;
    bool is_valid = LazyVerifyAndClassify(v, effective_k, *snapshot_ptr, root_state,
                                          partial_root_verify);
    if (is_valid) {
        valid_results.Insert(effective_k);
        return;
    }

    for (int d = (int)L - 1; d >= 0; --d) {
        if (IsRefinementDimFixed(static_cast<uint>(d))) continue;
        if (effective_k[d] == 0) continue;

        std::vector<uint> jumped = effective_k;

        thread_local std::vector<uint> h_bs;
        h_bs.resize(L);
        uint lo = GetDimLowerBound(v, d, effective_k);
        uint hi = effective_k[d] - 1;
        ApplyValidResultLowerBound(valid_results, effective_k, d, L, lo);
        ApplyRootLowerBound(completed_roots, effective_k, d, L, lo);
        if (lo > hi) continue;
        while (lo < hi) {
            uint mid = lo + (hi - lo + 1) / 2;
            jumped[d] = mid;
            bool bp = false;
            ComputeHFromBuffer(jumped, *buf_ptr, h_bs, bp);
            if (bp && h_bs[d] < mid) hi = h_bs[d];
            else lo = mid;
        }
        uint h_target = lo;

        jumped[d] = h_target;
        if (valid_results.IsDominatedRaw(jumped.data(), L)) continue;
        if (RootDominatesCandidate(completed_roots, jumped, L)) continue;
        if (visited && visited->Contains(jumped)) continue;

        DFSSplitLazyIncremental(v, jumped, d, valid_results, root_state, d, *buf_ptr,
                                *snapshot_ptr, visited, 0,
                                completed_roots);
    }
    if (completed_roots) completed_roots->Insert(effective_k);
}

void IterativeMLCore::DFSSplitIncremental(uint v, std::vector<uint>& current_k, uint last_dim,
                                          SkylineSet& valid_results, const VerifyState& parent_state,
                                          uint changed_dim, const NeighborMaxCBuffer& buf,
                                          const NeighborSnapshot* snapshot) const {
    if (valid_results.IsDominatedRaw(current_k.data(), L)) return;

    thread_local std::vector<uint> h_entry;
    h_entry.resize(L);
    bool h_pruned = false;
    ComputeHFromBuffer(current_k, buf, h_entry, h_pruned);
    if (h_pruned) h_pruned = ApplyFixedRefinementDim(current_k, h_entry);
    if (h_pruned) {
        for (uint l = 0; l < L; l++) current_k[l] = h_entry[l];
        bool all_zero = true;
        for (uint l = 0; l < L; l++) if (current_k[l] > 0) { all_zero = false; break; }
        if (all_zero) return;
        if (valid_results.IsDominatedRaw(current_k.data(), L)) return;
    }

    VerifyState current_state(L);
    bool is_valid = VerifyIncremental(v, current_k, changed_dim, parent_state, current_state, snapshot);

    if (is_valid) {
        valid_results.Insert(current_k);
        return;
    }

    for (int d = (int)L - 1; d >= (int)last_dim; --d) {
        if (IsRefinementDimFixed(static_cast<uint>(d))) continue;
        if (current_k[d] > 0) {
            std::vector<uint> jumped = current_k;

            thread_local std::vector<uint> h_bs;
            h_bs.resize(L);
            uint lo = GetDimLowerBound(v, d, current_k);
            uint hi = current_k[d] - 1;
            ApplyValidResultLowerBound(valid_results, current_k, d, L, lo);
            if (lo > hi) continue;
            while (lo < hi) {
                uint mid = lo + (hi - lo + 1) / 2;
                jumped[d] = mid;
                bool bp = false;
                ComputeHFromBuffer(jumped, buf, h_bs, bp);
                if (bp && h_bs[d] < mid) hi = h_bs[d];
                else lo = mid;
            }
            uint h_target = lo;

            jumped[d] = h_target;
            DFSSplitIncremental(v, jumped, d, valid_results, current_state, d, buf, snapshot);
        }
    }
}

void IterativeMLCore::ProcessRoundNode(uint v, RoundNodeResult& result) {
    result = RoundNodeResult();
    result.v = v;

    std::vector<uint> k_vec;
    SCVVisitSet failed_support_cache_slow;
    failed_support_cache_slow.reserve(128);

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
    SCVVisitTracker split_visited;
    split_visited.Init(L);
    split_visited.Reserve(1024);
    NeighborMaxCBuffer node_buf;
    bool has_node_buf = false;
    NeighborSnapshot node_snapshot;
    bool has_node_snapshot = false;
    uint eager_snapshot_threshold = 128;
    if (result.old_scv_count >= eager_snapshot_threshold) {
        BuildNeighborSnapshotNoLock(v, node_snapshot);
        has_node_snapshot = true;
    }

    bool lower_bound_seeded = false;
    if (has_lower_bounds &&
        v < lower_bounds_snapshot.size() &&
        v < lower_bounds_active.size() &&
        lower_bounds_active[v]) {
        lower_bound_seeded = true;
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
    SCVVisitTracker affected_candidates;
    affected_candidates.Init(L);
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

        bool supported = false;
        std::vector<uint> support_key;
        support_key.reserve(L + 1);
        support_key.push_back(v);
        support_key.insert(support_key.end(), k_vec.begin(), k_vec.end());
        bool cached_failed = failed_support_cache_slow.find(support_key) != failed_support_cache_slow.end();
        if (!cached_failed) {
            supported = has_node_snapshot
                ? CheckSupportFromSnapshot(v, k_vec, node_snapshot)
                : CheckSupportNoLock(v, k_vec);
            if (!supported) {
                failed_support_cache_slow.insert(std::move(support_key));
            }
        }

        if (supported) {
            next_scv_set.Insert(k_vec);
            shared_split_results.Insert(k_vec);
        } else {
            node_may_change = true;

            if (!has_node_buf) {
                PrecomputeNeighborMaxC(v, node_buf);
                has_node_buf = true;
            }

            auto binary_result = TryBinarySplit(v, k_vec, node_buf);

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
                SCVVisitTracker split_results_before;
                if (hot_start_scv_filter_enabled) {
                    split_results_before.Init(L);
                    uint before_count = shared_split_results.NumSCVs();
                    split_results_before.Reserve(before_count * 2 + 1);
                    std::vector<uint> before_rec(L);
                    for (uint ri = 0; ri < before_count; ri++) {
                        const uint* raw = shared_split_results.GetSCVFlat(ri);
                        for (uint d = 0; d < L; d++) before_rec[d] = raw[d];
                        split_results_before.Insert(before_rec);
                    }
                }
                SplitSCV(v, k_vec, shared_split_results, nullptr, &split_visited,
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
            if (delete_wake_use_reverse_support &&
                res.v < reverse_support.size() &&
                !reverse_support[res.v].empty()) {
                neighbor_buf.reserve(reverse_support[res.v].size());
                for (uint u : reverse_support[res.v]) neighbor_buf.push_back(u);
            } else {
                wake_neighbors = &GetCachedUniqueNeighbors(res.v, neighbor_buf);
            }

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
                if (limit_delete_wake_to_bfs &&
                    (u >= debug_delete_bfs_affected.size() || !debug_delete_bfs_affected[u])) {
                    continue;
                }

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

IterativeMLCore::MemoryBreakdown IterativeMLCore::EstimateMemoryUsageBytes() const {
    MemoryBreakdown out;

    auto vector_u32_bytes = [](const std::vector<uint>& v) -> size_t {
        return sizeof(v) + v.capacity() * sizeof(uint);
    };
    auto vector_char_bytes = [](const std::vector<char>& v) -> size_t {
        return sizeof(v) + v.capacity() * sizeof(char);
    };
    auto unordered_u32_set_bytes = [](const std::unordered_set<uint>& s) -> size_t {
        return sizeof(s) + s.bucket_count() * sizeof(void*) +
               s.size() * (sizeof(uint) + 2 * sizeof(void*));
    };
    auto unordered_ull_set_bytes = [](const std::unordered_set<unsigned long long>& s) -> size_t {
        return sizeof(s) + s.bucket_count() * sizeof(void*) +
               s.size() * (sizeof(unsigned long long) + 2 * sizeof(void*));
    };
    auto unordered_support_map_bytes =
        [](const std::unordered_map<uint, SCVSupportEntry::SupporterSCVInfo>& m) -> size_t {
            return sizeof(m) + m.bucket_count() * sizeof(void*) +
                   m.size() * (sizeof(uint) +
                               sizeof(SCVSupportEntry::SupporterSCVInfo) +
                               2 * sizeof(void*));
        };
    auto reverse_ref_vector_bytes = [](const std::vector<ReverseSupportEntryRef>& v) -> size_t {
        return sizeof(v) + v.capacity() * sizeof(ReverseSupportEntryRef);
    };
    auto skyline_bytes = [](const SkylineSet& set) -> size_t {
        return sizeof(SkylineSet) + set.CapacityValues() * sizeof(uint);
    };

    out.decomposition_bytes += sizeof(*this);
    out.decomposition_bytes += scv_sets.capacity() * sizeof(SkylineSet);
    for (const auto& set : scv_sets) {
        out.decomposition_bytes += skyline_bytes(set);
    }
    out.decomposition_bytes += max_c.capacity() * sizeof(max_c[0]);
    for (const auto& ptr : max_c) {
        if (ptr) out.decomposition_bytes += static_cast<size_t>(L) * sizeof(std::atomic<uint>);
    }
    out.decomposition_bytes += scv_rw_locks.capacity() * sizeof(scv_rw_locks[0]);
    out.decomposition_bytes += static_cast<size_t>(n) * sizeof(std::shared_mutex);
    if (is_in_inbox) out.decomposition_bytes += static_cast<size_t>(n) * sizeof(std::atomic<bool>);

    out.maintenance_bytes += support_data.capacity() * sizeof(std::vector<SCVSupportEntry>);
    for (const auto& entries : support_data) {
        out.maintenance_bytes += sizeof(entries) + entries.capacity() * sizeof(SCVSupportEntry);
        for (const auto& entry : entries) {
            out.maintenance_bytes += vector_u32_bytes(entry.scv);
            out.maintenance_bytes += entry.dim_supporters.capacity() * sizeof(std::unordered_set<uint>);
            for (const auto& s : entry.dim_supporters) {
                out.maintenance_bytes += unordered_u32_set_bytes(s);
            }
            out.maintenance_bytes += entry.dim_failers.capacity() * sizeof(std::unordered_set<uint>);
            for (const auto& s : entry.dim_failers) {
                out.maintenance_bytes += unordered_u32_set_bytes(s);
            }
            out.maintenance_bytes += entry.dim_supporter_scv_info.capacity() *
                                     sizeof(std::unordered_map<uint, SCVSupportEntry::SupporterSCVInfo>);
            for (const auto& m : entry.dim_supporter_scv_info) {
                out.maintenance_bytes += unordered_support_map_bytes(m);
            }
            out.maintenance_bytes += entry.dim_supporter_positions.capacity() * sizeof(std::vector<uint>);
            for (const auto& v : entry.dim_supporter_positions) {
                out.maintenance_bytes += vector_u32_bytes(v);
            }
            out.maintenance_bytes += entry.dim_failer_positions.capacity() * sizeof(std::vector<uint>);
            for (const auto& v : entry.dim_failer_positions) {
                out.maintenance_bytes += vector_u32_bytes(v);
            }
        }
    }

    out.maintenance_bytes += reverse_support.capacity() * sizeof(std::unordered_set<uint>);
    for (const auto& s : reverse_support) {
        out.maintenance_bytes += unordered_u32_set_bytes(s);
    }
    out.maintenance_bytes += reverse_support_entries.capacity() * sizeof(std::vector<ReverseSupportEntryRef>);
    for (const auto& refs : reverse_support_entries) {
        out.maintenance_bytes += reverse_ref_vector_bytes(refs);
    }

    auto break_index_bytes =
        [&](const std::vector<std::vector<std::unordered_map<uint, std::vector<ReverseSupportEntryRef>>>>& index) {
            size_t bytes = index.capacity() * sizeof(std::vector<std::unordered_map<uint, std::vector<ReverseSupportEntryRef>>>);
            for (const auto& per_node : index) {
                bytes += sizeof(per_node) +
                         per_node.capacity() * sizeof(std::unordered_map<uint, std::vector<ReverseSupportEntryRef>>);
                for (const auto& m : per_node) {
                    bytes += sizeof(m) + m.bucket_count() * sizeof(void*) +
                             m.size() * (sizeof(uint) + sizeof(std::vector<ReverseSupportEntryRef>) + 2 * sizeof(void*));
                    for (const auto& item : m) {
                        bytes += item.second.capacity() * sizeof(ReverseSupportEntryRef);
                    }
                }
            }
            return bytes;
        };
    out.maintenance_bytes += break_index_bytes(reverse_support_entries_by_break);
    out.maintenance_bytes += break_index_bytes(reverse_support_entries_by_break_ambiguous);
    out.maintenance_bytes += reverse_support_entries_by_unique_break.capacity() *
                             sizeof(std::unordered_map<unsigned long long, std::vector<ReverseSupportEntryRef>>);
    for (const auto& m : reverse_support_entries_by_unique_break) {
        out.maintenance_bytes += sizeof(m) + m.bucket_count() * sizeof(void*) +
                                 m.size() * (sizeof(unsigned long long) +
                                             sizeof(std::vector<ReverseSupportEntryRef>) +
                                             2 * sizeof(void*));
        for (const auto& item : m) {
            out.maintenance_bytes += item.second.capacity() * sizeof(ReverseSupportEntryRef);
        }
    }

    out.maintenance_bytes += neighbor_hindex_cap_cache.capacity() * sizeof(std::vector<uint>);
    for (const auto& v : neighbor_hindex_cap_cache) out.maintenance_bytes += vector_u32_bytes(v);
    out.maintenance_bytes += all_layer_neighbors.capacity() * sizeof(std::vector<uint>);
    for (const auto& v : all_layer_neighbors) out.maintenance_bytes += vector_u32_bytes(v);
    out.maintenance_bytes += lower_bounds_snapshot.capacity() * sizeof(SkylineSet);
    for (const auto& set : lower_bounds_snapshot) out.maintenance_bytes += skyline_bytes(set);
    out.maintenance_bytes += vector_char_bytes(lower_bounds_active);
    out.maintenance_bytes += vector_char_bytes(debug_delete_bfs_affected);
    out.maintenance_bytes += debug_delete_bfs_failed_sets.capacity() * sizeof(SkylineSet);
    for (const auto& set : debug_delete_bfs_failed_sets) out.maintenance_bytes += skyline_bytes(set);
    out.maintenance_bytes += sizeof(hot_start_scv_bits_by_node) +
                             hot_start_scv_bits_by_node.bucket_count() * sizeof(void*);
    for (const auto& item : hot_start_scv_bits_by_node) {
        out.maintenance_bytes += sizeof(item) +
                                 item.second.capacity() * sizeof(unsigned long long);
    }
    out.maintenance_bytes += vector_char_bytes(hot_start_wake_domain);
    out.maintenance_bytes += shared_queue.size() * sizeof(uint);

    return out;
}


void IterativeMLCore::BuildSupportData() {
    support_data.clear();
    reverse_support.clear();
    reverse_support_entries.clear();
    reverse_support_entries_by_break.clear();
    reverse_support_entries_by_break_ambiguous.clear();
    reverse_support_entries_by_unique_break.clear();
    if (!enable_support_pruning) return;

    support_data.resize(n);
    bool old_suppress = suppress_support_warnings;
    suppress_support_warnings = true;
    #pragma omp parallel for schedule(dynamic, 64)
    for (uint v = 0; v < n; v++) {
        BuildNodeSupportData(v);
    }
    suppress_support_warnings = old_suppress;
    BuildReverseSupportIndex();
}

void IterativeMLCore::BuildNodeSupportData(uint v) {
    support_data[v].clear();
    const SkylineSet& set = scv_sets[v];
    uint scv_count = set.NumSCVs();
    bool build_scv_support_info = n <= 100000;

    for (uint si = 0; si < scv_count; si++) {
        const uint* k_raw = set.GetSCVFlat(si);
        std::vector<uint> k_vec(k_raw, k_raw + L);
        SCVSupportEntry entry(k_vec, L, build_scv_support_info);

        for (uint d = 0; d < L; d++) {
            uint required_k = k_raw[d];
            if (required_k == 0) continue;

            uint** adj_lst = mg.GetGraph(d).GetAdjLst();
            uint degree = adj_lst[v][0];

            for (uint i = 1; i <= degree; i++) {
                uint u = adj_lst[v][i];

                SCVSupportEntry::SupporterSCVInfo info;
                bool is_supporter = false;
                if (build_scv_support_info) {
                    const SkylineSet& u_set = scv_sets[u];
                    uint u_count = u_set.NumSCVs();
                    for (uint ui = 0; ui < u_count; ui++) {
                        const uint* u_scv = u_set.GetSCVFlat(ui);
                        if (SkylineSet::DominatesRaw(u_scv, k_raw, L)) {
                            if (info.count == 0) info.first_idx = ui;
                            info.count++;
                        }
                    }
                    is_supporter = info.count > 0;
                } else {
                    is_supporter = scv_sets[u].IsDominatedRaw(k_raw, L);
                }

                if (is_supporter) {
                    entry.dim_supporters[d].insert(u);
                    if (build_scv_support_info) entry.dim_supporter_scv_info[d][u] = info;
                    entry.dim_supporter_positions[d].push_back(i);
                } else {
                    entry.dim_failers[d].insert(u);
                    entry.dim_failer_positions[d].push_back(i);
                }
            }

            if (!suppress_support_warnings && entry.dim_supporters[d].size() < required_k) {
                cerr << "[WARN] BuildNodeSupportData: v=" << v << " d=" << d
                     << " required_k=" << required_k
                     << " actual=" << entry.dim_supporters[d].size()
                     << " k_vec=[";
                for (uint dd = 0; dd < L; dd++) cerr << k_vec[dd] << (dd+1<L?",":"");
                cerr << "]" << endl;
            }
        }

        support_data[v].push_back(std::move(entry));
    }
}

bool IterativeMLCore::IsSupporter(uint v, const std::vector<uint>& k, uint d, uint u) const {
    if (v >= support_data.size()) return false;
    for (const auto& entry : support_data[v]) {
        if (entry.scv == k) {
            if (d < entry.dim_supporters.size()) {
                return entry.dim_supporters[d].count(u) > 0;
            }
            return false;
        }
    }
    return false;
}

uint IterativeMLCore::GetSupporterCount(uint v, const std::vector<uint>& k, uint d) const {
    if (v >= support_data.size()) return 0;
    for (const auto& entry : support_data[v]) {
        if (entry.scv == k) {
            if (d < entry.dim_supporters.size()) {
                return entry.dim_supporters[d].size();
            }
            return 0;
        }
    }
    return 0;
}

const std::unordered_set<uint>* IterativeMLCore::GetSupporters(uint v, const std::vector<uint>& k, uint d) const {
    if (v >= support_data.size()) return nullptr;
    for (const auto& entry : support_data[v]) {
        if (entry.scv == k) {
            if (d < entry.dim_supporters.size()) {
                return &entry.dim_supporters[d];
            }
            return nullptr;
        }
    }
    return nullptr;
}

void IterativeMLCore::BuildReverseSupportIndex() {
    reverse_support.clear();
    reverse_support.resize(n);
    reverse_support_entries.clear();
    reverse_support_entries.resize(n);
    reverse_support_entries_by_break.clear();
    reverse_support_entries_by_break_ambiguous.clear();
    reverse_support_entries_by_unique_break.clear();
    bool build_break_index = n <= 100000;
    if (build_break_index) {
        reverse_support_entries_by_break.resize(n);
        reverse_support_entries_by_break_ambiguous.resize(n);
        reverse_support_entries_by_unique_break.resize(n);
        for (uint x = 0; x < n; x++) {
            reverse_support_entries_by_break[x].resize(L);
            reverse_support_entries_by_break_ambiguous[x].resize(L);
        }
    }

    std::vector<std::vector<std::pair<uint, uint>>> thread_pairs(omp_get_max_threads());
    std::vector<std::vector<std::pair<uint, ReverseSupportEntryRef>>> thread_entry_refs(omp_get_max_threads());

    #pragma omp parallel
    {
        auto& local = thread_pairs[omp_get_thread_num()];
        auto& local_refs = thread_entry_refs[omp_get_thread_num()];
        #pragma omp for schedule(dynamic) nowait
        for (uint y = 0; y < n; y++) {
            for (uint si = 0; si < support_data[y].size(); si++) {
                const auto& entry = support_data[y][si];
                for (uint d = 0; d < L; d++) {
                    for (uint x : entry.dim_supporters[d]) {
                        local.emplace_back(x, y);
                        ReverseSupportEntryRef ref{y, si, d};
                        if (d < entry.dim_supporter_scv_info.size()) {
                            auto info_it = entry.dim_supporter_scv_info[d].find(x);
                            if (info_it != entry.dim_supporter_scv_info[d].end()) {
                                ref.support_scv_count = info_it->second.count;
                                ref.unique_support_scv_idx = info_it->second.count == 1
                                    ? info_it->second.first_idx
                                    : UINT_MAX;
                            }
                        }
                        local_refs.push_back({x, ref});
                    }
                }
            }
        }
    }

    for (auto& pairs : thread_pairs) {
        for (auto& p : pairs) {
            reverse_support[p.first].insert(p.second);
        }
    }
    for (auto& refs : thread_entry_refs) {
        for (auto& p : refs) {
            reverse_support_entries[p.first].push_back(p.second);
            if (build_break_index) {
                const ReverseSupportEntryRef& ref = p.second;
                if (ref.node < support_data.size() && ref.scv_idx < support_data[ref.node].size()) {
                    const auto& target = support_data[ref.node][ref.scv_idx].scv;
                    for (uint bd = 0; bd < L && bd < target.size(); bd++) {
                        reverse_support_entries_by_break[p.first][bd][target[bd]].push_back(ref);
                        if (ref.support_scv_count == 1 &&
                            ref.unique_support_scv_idx != UINT_MAX &&
                            target[bd] < (1u << 21) &&
                            ref.unique_support_scv_idx < (1u << 21)) {
                            unsigned long long key =
                                (static_cast<unsigned long long>(bd) << 42) |
                                (static_cast<unsigned long long>(target[bd]) << 21) |
                                static_cast<unsigned long long>(ref.unique_support_scv_idx);
                            reverse_support_entries_by_unique_break[p.first][key].push_back(ref);
                        } else {
                            reverse_support_entries_by_break_ambiguous[p.first][bd][target[bd]].push_back(ref);
                        }
                    }
                }
            }
        }
    }
}

void IterativeMLCore::RebuildSupportData() {
    support_data.clear();
    reverse_support.clear();
    reverse_support_entries.clear();
    reverse_support_entries_by_break.clear();
    reverse_support_entries_by_break_ambiguous.clear();
    reverse_support_entries_by_unique_break.clear();
    BuildSupportData();
}

unsigned long long IterativeMLCore::ComputeNodeCost(uint v) const {
    unsigned long long degree_sum = 0;
    for (uint l = 0; l < L; l++) {
        degree_sum += mg.GetGraph(l).GetAdjLst()[v][0];
    }

    scv_rw_locks[v]->lock_shared();
    uint num_scvs = scv_sets[v].NumSCVs();
    scv_rw_locks[v]->unlock_shared();
    unsigned long long cost = degree_sum + static_cast<unsigned long long>(num_scvs) * L * 16ULL;

    if (v >= support_data.size() || support_data[v].empty()) return cost;

    unsigned long long support_cost = 0;
    for (const auto& entry : support_data[v]) {
        support_cost += L * 8ULL;
        for (uint d = 0; d < L; d++) {
            uint required_k = (d < entry.scv.size()) ? entry.scv[d] : 0;
            if (required_k == 0) continue;

            unsigned long long supporters = entry.dim_supporters[d].size();
            unsigned long long failers = entry.dim_failers[d].size();
            unsigned long long missing = (supporters < required_k)
                ? static_cast<unsigned long long>(required_k) - supporters
                : 0ULL;

            support_cost += supporters * 4ULL;
            support_cost += failers * 10ULL;
            support_cost += missing * 48ULL;
        }
    }
    cost += support_cost;

    return cost;
}
