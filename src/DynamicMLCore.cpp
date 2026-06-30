#include "DynamicMLCore.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <climits>
#include <functional>
#include <fstream>
#include <iterator>
#include <mutex>

DynamicMLCore::DynamicMLCore(MultilayerGraph& mg_, int num_threads_)
    : IterativeMLCore(mg_, num_threads_) {
}

namespace {

constexpr uint kMaxTrackedBatchLayers = 64;
constexpr unsigned long long kAffectedSCVFilterMinAverage = 8;
std::array<char, kMaxTrackedBatchLayers> g_batch_insert_active_layers{};
bool g_has_batch_insert_active_layers = false;
std::array<char, kMaxTrackedBatchLayers> g_batch_delete_active_layers{};
bool g_has_batch_delete_active_layers = false;
thread_local bool g_disable_pending_candidate_locks = false;

inline unsigned long long InsertSingleEdgeParallelThreshold() {
    return 512;
}

inline unsigned long long InsertSingleEdgeParallelWorkThreshold() {
    return 4096;
}

inline unsigned long long InsertSingleEdgeParallelNeighborWorkThreshold() {
    return 1024;
}

inline unsigned long long DeleteSingleEdgeParallelThreshold() {
    return 512;
}

inline unsigned long long DeleteSingleEdgeParallelWorkThreshold() {
    return 4096;
}

inline unsigned long long DeleteSingleEdgeParallelNeighborWorkThreshold() {
    return 1024;
}

inline bool SameSCVRaw(const uint* a, const uint* b, uint L) {
    for (uint d = 0; d < L; d++) {
        if (a[d] != b[d]) return false;
    }
    return true;
}

inline std::string MakeVectorRecordKey(uint node, const std::vector<uint>& k) {
    std::string key;
    key.reserve(sizeof(uint) * (k.size() + 1));
    key.append(reinterpret_cast<const char*>(&node), sizeof(uint));
    key.append(reinterpret_cast<const char*>(k.data()), sizeof(uint) * k.size());
    return key;
}

inline std::string MakeDeleteFailKey(uint node, const std::vector<uint>& k, uint broken_dim) {
    std::string key = MakeVectorRecordKey(node, k);
    key.append(reinterpret_cast<const char*>(&broken_dim), sizeof(uint));
    return key;
}

inline std::string MakeDeleteFailKey(uint node, const std::vector<uint>& k) {
    return MakeVectorRecordKey(node, k);
}

inline std::string MakeIndexedDeleteFailKey(uint node, uint scv_idx, uint broken_dim) {
    std::string key;
    key.reserve(sizeof(uint) * 3);
    key.append(reinterpret_cast<const char*>(&node), sizeof(uint));
    key.append(reinterpret_cast<const char*>(&scv_idx), sizeof(uint));
    key.append(reinterpret_cast<const char*>(&broken_dim), sizeof(uint));
    return key;
}

inline std::string MakeIndexedDeleteFailKey(uint node, uint scv_idx) {
    std::string key;
    key.reserve(sizeof(uint) * 2);
    key.append(reinterpret_cast<const char*>(&node), sizeof(uint));
    key.append(reinterpret_cast<const char*>(&scv_idx), sizeof(uint));
    return key;
}

void CollectUniqueNeighbors(uint node,
                            const std::vector<uint**>& adj_arrays,
                            uint L,
                            uint total_nodes,
                            std::vector<uint>& out) {
    out.clear();
    size_t degree_sum = 0;
    for (uint d = 0; d < L; d++) {
        degree_sum += adj_arrays[d][node][0];
    }

    out.reserve(degree_sum);
    for (uint d = 0; d < L; d++) {
        uint** adj = adj_arrays[d];
        uint degree = adj[node][0];
        for (uint i = 1; i <= degree; i++) {
            out.push_back(adj[node][i]);
        }
    }
    if (out.size() < 2) return;
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

void CompactCandidateRecordsByNode(std::vector<std::pair<uint, std::vector<uint>>>& records,
                                   uint L) {
    if (records.size() < 2) return;

    std::sort(records.begin(), records.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
              });

    std::vector<std::pair<uint, std::vector<uint>>> compacted;
    compacted.reserve(records.size());
    size_t pos = 0;
    while (pos < records.size()) {
        uint node = records[pos].first;
        SkylineSet skyline;
        size_t next = pos;
        while (next < records.size() && records[next].first == node) {
            skyline.Insert(records[next].second);
            next++;
        }
        uint count = skyline.NumSCVs();
        for (uint si = 0; si < count; si++) {
            const uint* raw = skyline.GetSCVFlat(si);
            compacted.emplace_back(node, std::vector<uint>(raw, raw + L));
        }
        pos = next;
    }
    records.swap(compacted);
}

inline unsigned long long Mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline std::pair<unsigned long long, unsigned long long> MakeDeleteStateFingerprint(
    uint node,
    const std::vector<uint>& k,
    uint broken_dim)
{
    unsigned long long h1 = Mix64(static_cast<unsigned long long>(node) << 32 | broken_dim);
    unsigned long long h2 = Mix64((static_cast<unsigned long long>(broken_dim) << 32) ^ node);
    for (uint value : k) {
        h1 = Mix64(h1 ^ value);
        h2 = Mix64(h2 + 0x9e3779b97f4a7c15ULL + value);
    }
    h1 &= ~3ULL;
    if (h1 == 0) h1 = 4;
    if (h2 == 0) h2 = 0x9e3779b97f4a7c15ULL;
    return {h1, h2};
}

inline std::pair<unsigned long long, unsigned long long> MakeDeleteStateFingerprintRaw(
    uint node,
    const uint* k,
    uint L,
    uint broken_dim)
{
    unsigned long long h1 = Mix64(static_cast<unsigned long long>(node) << 32 | broken_dim);
    unsigned long long h2 = Mix64((static_cast<unsigned long long>(broken_dim) << 32) ^ node);
    for (uint d = 0; d < L; d++) {
        uint value = k[d];
        h1 = Mix64(h1 ^ value);
        h2 = Mix64(h2 + 0x9e3779b97f4a7c15ULL + value);
    }
    h1 &= ~3ULL;
    if (h1 == 0) h1 = 4;
    if (h2 == 0) h2 = 0x9e3779b97f4a7c15ULL;
    return {h1, h2};
}

inline std::pair<unsigned long long, unsigned long long> MakeIndexedStateFingerprint(
    uint node,
    uint scv_idx,
    uint dim)
{
    unsigned long long h1 = Mix64(static_cast<unsigned long long>(node) << 32 | dim);
    h1 = Mix64(h1 ^ scv_idx);
    unsigned long long h2 = Mix64((static_cast<unsigned long long>(dim) << 32) ^ node);
    h2 = Mix64(h2 + scv_idx);
    h1 &= ~3ULL;
    if (h1 == 0) h1 = 4;
    if (h2 == 0) h2 = 0x9e3779b97f4a7c15ULL;
    return {h1, h2};
}

inline std::pair<unsigned long long, unsigned long long> MakeIndexedStateFingerprint(
    uint node,
    uint scv_idx)
{
    unsigned long long h1 = Mix64(static_cast<unsigned long long>(node) << 32);
    h1 = Mix64(h1 ^ scv_idx);
    unsigned long long h2 = Mix64(static_cast<unsigned long long>(node) ^ 0x9e3779b97f4a7c15ULL);
    h2 = Mix64(h2 + scv_idx);
    h1 &= ~3ULL;
    if (h1 == 0) h1 = 4;
    if (h2 == 0) h2 = 0x9e3779b97f4a7c15ULL;
    return {h1, h2};
}

struct AtomicDeleteStateSet {
    static constexpr unsigned long long kEmpty = 0;
    static constexpr unsigned long long kReserved = 1;

    explicit AtomicDeleteStateSet(size_t requested_capacity) {
        capacity = 1;
        while (capacity < requested_capacity) capacity <<= 1;
        keys = std::make_unique<std::atomic<unsigned long long>[]>(capacity);
        checks = std::make_unique<std::atomic<unsigned long long>[]>(capacity);
        for (size_t i = 0; i < capacity; i++) {
            keys[i].store(kEmpty, std::memory_order_relaxed);
            checks[i].store(0, std::memory_order_relaxed);
        }
    }

    bool TryMark(uint node, const std::vector<uint>& k, uint broken_dim) {
        auto fp = MakeDeleteStateFingerprint(node, k, broken_dim);
        return TryMarkFingerprint(fp.first, fp.second);
    }

    bool TryMarkFingerprint(unsigned long long key,
                            unsigned long long check) {
        size_t mask = capacity - 1;
        size_t pos = key & mask;
        for (size_t probe = 0; probe < 16; probe++) {
            size_t slot = (pos + probe) & mask;
            unsigned long long cur = keys[slot].load(std::memory_order_acquire);
            if (cur == key) {
                if (checks[slot].load(std::memory_order_acquire) == check) return false;
                continue;
            }
            if (cur == kReserved) continue;
            if (cur == kEmpty) {
                unsigned long long expected = kEmpty;
                if (keys[slot].compare_exchange_strong(expected, kReserved,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire)) {
                    checks[slot].store(check, std::memory_order_release);
                    keys[slot].store(key, std::memory_order_release);
                    return true;
                }
            }
        }
        return true;
    }

    bool TryMarkRaw(uint node, const uint* k_raw, uint L, uint broken_dim) {
        auto fp = MakeDeleteStateFingerprintRaw(node, k_raw, L, broken_dim);
        unsigned long long key = fp.first;
        unsigned long long check = fp.second;
        size_t mask = capacity - 1;
        size_t pos = key & mask;
        for (size_t probe = 0; probe < 16; probe++) {
            size_t slot = (pos + probe) & mask;
            unsigned long long cur = keys[slot].load(std::memory_order_acquire);
            if (cur == key) {
                if (checks[slot].load(std::memory_order_acquire) == check) return false;
                continue;
            }
            if (cur == kReserved) continue;
            if (cur == kEmpty) {
                unsigned long long expected = kEmpty;
                if (keys[slot].compare_exchange_strong(expected, kReserved,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire)) {
                    checks[slot].store(check, std::memory_order_release);
                    keys[slot].store(key, std::memory_order_release);
                    return true;
                }
            }
        }
        return true;
    }

    bool TryMarkId(uint node, uint scv_idx, uint dim) {
        auto fp = MakeIndexedStateFingerprint(node, scv_idx, dim);
        unsigned long long key = fp.first;
        unsigned long long check = fp.second;
        size_t mask = capacity - 1;
        size_t pos = key & mask;
        for (size_t probe = 0; probe < 16; probe++) {
            size_t slot = (pos + probe) & mask;
            unsigned long long cur = keys[slot].load(std::memory_order_acquire);
            if (cur == key) {
                if (checks[slot].load(std::memory_order_acquire) == check) return false;
                continue;
            }
            if (cur == kReserved) continue;
            if (cur == kEmpty) {
                unsigned long long expected = kEmpty;
                if (keys[slot].compare_exchange_strong(expected, kReserved,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire)) {
                    checks[slot].store(check, std::memory_order_release);
                    keys[slot].store(key, std::memory_order_release);
                    return true;
                }
            }
        }
        return true;
    }

    bool TryMarkId(uint node, uint scv_idx) {
        auto fp = MakeIndexedStateFingerprint(node, scv_idx);
        unsigned long long key = fp.first;
        unsigned long long check = fp.second;
        size_t mask = capacity - 1;
        size_t pos = key & mask;
        for (size_t probe = 0; probe < 16; probe++) {
            size_t slot = (pos + probe) & mask;
            unsigned long long cur = keys[slot].load(std::memory_order_acquire);
            if (cur == key) {
                if (checks[slot].load(std::memory_order_acquire) == check) return false;
                continue;
            }
            if (cur == kReserved) continue;
            if (cur == kEmpty) {
                unsigned long long expected = kEmpty;
                if (keys[slot].compare_exchange_strong(expected, kReserved,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire)) {
                    checks[slot].store(check, std::memory_order_release);
                    keys[slot].store(key, std::memory_order_release);
                    return true;
                }
            }
        }
        return true;
    }

    bool Contains(uint node, const uint* k_raw, uint L, uint broken_dim) const {
        auto fp = MakeDeleteStateFingerprintRaw(node, k_raw, L, broken_dim);
        unsigned long long key = fp.first;
        unsigned long long check = fp.second;
        size_t mask = capacity - 1;
        size_t pos = key & mask;
        for (size_t probe = 0; probe < 16; probe++) {
            size_t slot = (pos + probe) & mask;
            unsigned long long cur = keys[slot].load(std::memory_order_acquire);
            if (cur == key && checks[slot].load(std::memory_order_acquire) == check) return true;
            if (cur == kEmpty) return false;
        }
        return false;
    }

    size_t capacity = 0;
    std::unique_ptr<std::atomic<unsigned long long>[]> keys;
    std::unique_ptr<std::atomic<unsigned long long>[]> checks;
};

} 

void DynamicMLCore::RestoreSnapshotForIndependentRun(const std::vector<SkylineSet>& snapshot) {
    scv_sets = snapshot;
    if (max_c.size() != n) max_c.resize(n);
    for (uint v = 0; v < n; v++) {
        if (!max_c[v]) max_c[v] = std::make_unique<std::atomic<uint>[]>(L);
        UpdateMaxC(v);
    }
    {
        std::lock_guard<std::mutex> lock(shared_queue_mtx);
        shared_queue.clear();
    }
    active_task_count.store(0, std::memory_order_relaxed);
    if (is_in_inbox) {
        for (uint v = 0; v < n; v++) {
            is_in_inbox[v].store(false, std::memory_order_relaxed);
        }
    }
    has_lower_bounds = false;
    lower_bounds_snapshot.clear();
    lower_bounds_active.clear();
    ClearNeighborHIndexCapCache();
    ClearHotStartSCVFilter();
    g_has_batch_insert_active_layers = false;
    g_has_batch_delete_active_layers = false;
    debug_delete_bfs_failed_sets.clear();
    last_bfs_ms = 0.0;
    last_iter_ms = 0.0;
    support_data.clear();
    reverse_support.clear();
    reverse_support_entries.clear();
    reverse_support_entries_by_break.clear();
    reverse_support_entries_by_break_ambiguous.clear();
    reverse_support_entries_by_unique_break.clear();
}

bool DynamicMLCore::SavePreparedSnapshotForIndependentRun(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    const char magic[8] = {'M', 'C', 'M', 'P', 'R', 'E', 'P', '1'};
    uint version = 1;
    uint persistent_index = 0;
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(uint));
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint));
    out.write(reinterpret_cast<const char*>(&L), sizeof(uint));
    out.write(reinterpret_cast<const char*>(&persistent_index), sizeof(uint));

    for (uint v = 0; v < n; v++) {
        for (uint d = 0; d < L; d++) {
            uint value = max_c[v]
                ? max_c[v][d].load(std::memory_order_relaxed)
                : 0;
            out.write(reinterpret_cast<const char*>(&value), sizeof(uint));
        }
    }

    return out.good();
}

bool DynamicMLCore::RestorePreparedSnapshotForIndependentRun(
    const std::vector<SkylineSet>& snapshot,
    const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    char magic[8];
    in.read(magic, sizeof(magic));
    const char expected_magic[8] = {'M', 'C', 'M', 'P', 'R', 'E', 'P', '1'};
    if (!in || !std::equal(std::begin(magic), std::end(magic), std::begin(expected_magic))) {
        return false;
    }

    uint version = 0, file_n = 0, file_L = 0, persistent_index = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(uint));
    in.read(reinterpret_cast<char*>(&file_n), sizeof(uint));
    in.read(reinterpret_cast<char*>(&file_L), sizeof(uint));
    in.read(reinterpret_cast<char*>(&persistent_index), sizeof(uint));
    if (!in || version != 1 || file_n != n || file_L != L) return false;
    if (snapshot.size() != n) return false;
    if (persistent_index != 0) return false;

    scv_sets = snapshot;
    if (max_c.size() != n) max_c.resize(n);
    for (uint v = 0; v < n; v++) {
        if (!max_c[v]) max_c[v] = std::make_unique<std::atomic<uint>[]>(L);
        for (uint d = 0; d < L; d++) {
            uint value = 0;
            in.read(reinterpret_cast<char*>(&value), sizeof(uint));
            if (!in) return false;
            max_c[v][d].store(value, std::memory_order_relaxed);
        }
    }

    {
        std::lock_guard<std::mutex> lock(shared_queue_mtx);
        shared_queue.clear();
    }
    active_task_count.store(0, std::memory_order_relaxed);
    if (is_in_inbox) {
        for (uint v = 0; v < n; v++) {
            is_in_inbox[v].store(false, std::memory_order_relaxed);
        }
    }
    has_lower_bounds = false;
    lower_bounds_snapshot.clear();
    lower_bounds_active.clear();
    ClearNeighborHIndexCapCache();
    ClearHotStartSCVFilter();
    g_has_batch_insert_active_layers = false;
    g_has_batch_delete_active_layers = false;
    debug_delete_bfs_failed_sets.clear();
    last_bfs_ms = 0.0;
    last_iter_ms = 0.0;
    support_data.clear();
    reverse_support.clear();
    reverse_support_entries.clear();
    reverse_support_entries_by_break.clear();
    reverse_support_entries_by_break_ambiguous.clear();
    reverse_support_entries_by_unique_break.clear();
    return true;
}

bool DynamicMLCore::TryInsertPendingCandidate(uint node,
                                              const std::vector<uint>& k_cand,
                                              const std::vector<SkylineSet>& old_scv_sets,
                                              std::vector<SkylineSet>& pending_sets,
                                              std::vector<char>& pending_touched,
                                              std::vector<uint>& pending_nodes) {
    bool old_max_can_dominate = true;
    for (uint d = 0; d < L; d++) {
        if (max_c[node][d].load(std::memory_order_relaxed) < k_cand[d]) {
            old_max_can_dominate = false;
            break;
        }
    }
    if (old_max_can_dominate && old_scv_sets[node].IsDominated(k_cand)) return false;

    if (g_disable_pending_candidate_locks) {
        bool inserted = pending_sets[node].Insert(k_cand);
        if (inserted && !pending_touched[node]) {
            pending_touched[node] = 1;
            pending_nodes.push_back(node);
        }
        return inserted;
    }

    scv_rw_locks[node]->lock();
    bool inserted = pending_sets[node].Insert(k_cand);
    if (inserted && !pending_touched[node]) {
        pending_touched[node] = 1;
        pending_nodes.push_back(node);
    }
    scv_rw_locks[node]->unlock();
    return inserted;
}

void DynamicMLCore::ParallelBFSCascadePendingCapped(std::vector<PropagationEntry>& F_curr,
                                                    const std::vector<SkylineSet>& old_scv_sets,
                                                    std::vector<SkylineSet>& pending_sets,
                                                    std::vector<char>& pending_touched,
                                                    std::vector<uint>& pending_nodes) {
    std::vector<PropagationEntry> F_next;
    auto max_c_can_dominate = [&](uint node, const uint* k) {
        for (uint d = 0; d < L; d++) {
            if (max_c[node][d].load(std::memory_order_relaxed) < k[d]) return false;
        }
        return true;
    };

    while (!F_curr.empty()) {
        if (F_curr.size() <= 64) {
            std::vector<uint**> adj_arrays(L);
            for (uint d = 0; d < L; d++) adj_arrays[d] = mg.GetGraph(d).GetAdjLst();

            F_next.clear();
            std::vector<uint> k_y_cand(L);
            for (const PropagationEntry& pe : F_curr) {
                uint x = pe.node;
                const uint* k_new = pe.K();

                scv_rw_locks[x]->lock_shared();
                bool still_pending = pending_sets[x].ContainsRaw(k_new, L);
                scv_rw_locks[x]->unlock_shared();
                if (!still_pending) continue;

                const SkylineSet& old_x_set = old_scv_sets[x];
                uint old_x_count = old_x_set.NumSCVs();

                for (uint l_edge = 0; l_edge < L; l_edge++) {
                    uint** adj_le = adj_arrays[l_edge];
                    uint deg_x = adj_le[x][0];
                    for (uint j = 1; j <= deg_x; j++) {
                        uint y = adj_le[x][j];

                        for (uint d = 0; d < L; d++)
                            k_y_cand[d] = std::min(k_new[d], adj_arrays[d][y][0]);

                        bool x_old_supported = false;
                        if (max_c_can_dominate(x, k_y_cand.data())) {
                            for (uint si = 0; si < old_x_count; si++) {
                                if (SkylineSet::DominatesRaw(old_x_set.GetSCVFlat(si), k_y_cand.data(), L)) {
                                    x_old_supported = true;
                                    break;
                                }
                            }
                        }
                        if (x_old_supported) continue;

                        if (max_c_can_dominate(y, k_y_cand.data()) &&
                            old_scv_sets[y].IsDominatedRaw(k_y_cand.data(), L)) {
                            continue;
                        }

                        scv_rw_locks[y]->lock_shared();
                        bool pending_dominated = pending_sets[y].IsDominatedRaw(k_y_cand.data(), L);
                        scv_rw_locks[y]->unlock_shared();
                        if (pending_dominated) continue;

                        if (TryInsertPendingCandidate(y, k_y_cand, old_scv_sets,
                                                      pending_sets, pending_touched,
                                                      pending_nodes)) {
                            F_next.push_back(PropagationEntry(y, k_y_cand, l_edge));
                        }
                    }
                }
            }
            swap(F_curr, F_next);
            continue;
        }

        int T = omp_get_max_threads();
        std::vector<std::vector<PropagationEntry>> F_local(T);
        std::vector<std::vector<uint>> touched_local(T);

        std::vector<uint**> adj_arrays(L);
        for (uint d = 0; d < L; d++) adj_arrays[d] = mg.GetGraph(d).GetAdjLst();

        #pragma omp parallel
        {
            std::vector<uint> k_y_cand(L);
            int tid = omp_get_thread_num();

            #pragma omp for schedule(dynamic, 64)
            for (size_t idx = 0; idx < F_curr.size(); idx++) {
                const PropagationEntry& pe = F_curr[idx];
                uint x = pe.node;
                const uint* k_new = pe.K();

                scv_rw_locks[x]->lock_shared();
                bool still_pending = pending_sets[x].ContainsRaw(k_new, L);
                scv_rw_locks[x]->unlock_shared();
                if (!still_pending) continue;

                const SkylineSet& old_x_set = old_scv_sets[x];
                uint old_x_count = old_x_set.NumSCVs();

                for (uint l_edge = 0; l_edge < L; l_edge++) {
                    uint** adj_le = adj_arrays[l_edge];
                    uint deg_x = adj_le[x][0];
                    for (uint j = 1; j <= deg_x; j++) {
                        uint y = adj_le[x][j];

                        for (uint d = 0; d < L; d++)
                            k_y_cand[d] = std::min(k_new[d], adj_arrays[d][y][0]);

                        bool x_old_supported = false;
                        if (max_c_can_dominate(x, k_y_cand.data())) {
                            for (uint si = 0; si < old_x_count; si++) {
                                if (SkylineSet::DominatesRaw(old_x_set.GetSCVFlat(si), k_y_cand.data(), L)) {
                                    x_old_supported = true;
                                    break;
                                }
                            }
                        }
                        if (x_old_supported) {
                            continue;
                        }

                        if (max_c_can_dominate(y, k_y_cand.data()) &&
                            old_scv_sets[y].IsDominatedRaw(k_y_cand.data(), L)) {
                            continue;
                        }

                        scv_rw_locks[y]->lock_shared();
                        bool pending_dominated = pending_sets[y].IsDominatedRaw(k_y_cand.data(), L);
                        scv_rw_locks[y]->unlock_shared();
                        if (pending_dominated) {
                            continue;
                        }

                        bool inserted = TryInsertPendingCandidate(y, k_y_cand, old_scv_sets,
                                                                  pending_sets, pending_touched,
                                                                  touched_local[tid]);
                        if (inserted) {
                            F_local[tid].push_back(PropagationEntry(y, k_y_cand, l_edge));
                        }
                    }
                }
            }
        }

        size_t next_size = 0;
        size_t touched_size = 0;
        for (int t = 0; t < T; t++) {
            next_size += F_local[t].size();
            touched_size += touched_local[t].size();
        }
        F_next.clear();
        F_next.reserve(next_size);
        pending_nodes.reserve(pending_nodes.size() + touched_size);
        for (int t = 0; t < T; t++) {
            F_next.insert(F_next.end(), F_local[t].begin(), F_local[t].end());
            F_local[t].clear();
            pending_nodes.insert(pending_nodes.end(), touched_local[t].begin(), touched_local[t].end());
            touched_local[t].clear();
        }
        swap(F_curr, F_next);
    }
}

void DynamicMLCore::DeletePropagationOnly(
    const std::vector<std::tuple<uint, uint, uint>>& deleted_edges,
    std::vector<char>& needs_check,
    std::vector<char>* affected_domain,
    std::vector<std::pair<uint, std::vector<uint>>>* affected_scvs)
{
    std::deque<FailEntry> frontier;
    std::unordered_set<std::string> processed;
    if (affected_domain) affected_domain->assign(n, 0);
    std::vector<uint**> adj_arrays(L);
    for (uint d = 0; d < L; d++) adj_arrays[d] = mg.GetGraph(d).GetAdjLst();
    unsigned long long single_edge_endpoint_scvs = 0;
    unsigned long long single_edge_endpoint_work = 0;
    unsigned long long single_edge_neighbor_work = 0;
    unsigned long long initial_cost = 0;

    auto mark_domain = [&](uint node) {
        if (node < needs_check.size()) needs_check[node] = 1;
        if (affected_domain && node < affected_domain->size()) {
            (*affected_domain)[node] = 1;
        }
    };

    auto enqueue_failed = [&](uint node,
                              const std::vector<uint>& k,
                              uint delete_layer,
                              uint scv_idx = UINT_MAX) {
        std::string key = scv_idx != UINT_MAX
            ? MakeIndexedDeleteFailKey(node, scv_idx)
            : MakeDeleteFailKey(node, k);
        if (!processed.insert(std::move(key)).second) return false;
        mark_domain(node);
        if (affected_scvs) affected_scvs->emplace_back(node, k);
        frontier.emplace_back(node, k, delete_layer, scv_idx);
        return true;
    };

    auto has_direct_support_from_endpoint = [&](uint other, const uint* target) {
        for (uint d = 0; d < L; d++) {
            if (max_c[other][d].load(std::memory_order_relaxed) < target[d]) return false;
        }
        return scv_sets[other].IsDominatedRaw(target, L);
    };

    auto deletion_transition_holds = [&](uint x,
                                         const std::vector<uint>& k_fail,
                                         uint delete_layer,
                                         const uint* k_y) {
        if (delete_layer >= L) return false;

        for (uint d = 0; d < L; d++) {
            if (k_fail[d] < k_y[d]) return false;
        }

        for (uint d = 0; d < L; d++) {
            uint value = max_c[x][d].load(std::memory_order_relaxed);
            if (d == delete_layer && value > 0) value--;
            if (value < k_y[d]) return true;
        }

        uint x_count = scv_sets[x].NumSCVs();
        for (uint si = 0; si < x_count; si++) {
            const uint* h = scv_sets[x].GetSCVFlat(si);
            bool reduced_dominates = true;
            for (uint d = 0; d < L; d++) {
                uint value = h[d];
                if (d == delete_layer && value > 0) value--;
                if (value < k_y[d]) {
                    reduced_dominates = false;
                    break;
                }
            }
            if (reduced_dominates) return false;
        }
        return true;
    };

    for (const auto& e : deleted_edges) {
        uint u = std::get<0>(e), v = std::get<1>(e), l = std::get<2>(e);
        if (deleted_edges.size() == 1) {
            unsigned long long u_scvs = scv_sets[u].NumSCVs();
            unsigned long long v_scvs = v == u ? 0 : scv_sets[v].NumSCVs();
            single_edge_endpoint_scvs = u_scvs + v_scvs;
            single_edge_endpoint_work = v == u
                ? u_scvs
                : 2ULL * u_scvs * std::max<unsigned long long>(1, v_scvs);
        }
        auto seed_endpoint = [&](uint node, uint other, uint layer) {
            if (layer >= L) return;
            uint scv_count = scv_sets[node].NumSCVs();
            for (uint si = 0; si < scv_count; si++) {
                
                const uint* raw = scv_sets[node].GetSCVFlat(si);
                if (raw[layer] == 0) continue;
                if (!has_direct_support_from_endpoint(other, raw)) continue;
                std::vector<uint> k(raw, raw + L);
                enqueue_failed(node, k, layer, si);
            }
        };
        seed_endpoint(u, v, l);
        if (v != u) seed_endpoint(v, u, l);
    }

    auto initial_frontier_cost = [&]() {
        unsigned long long cost = 0;
        for (const FailEntry& fe : frontier) {
            if (fe.broken_dim >= L) continue;
            for (uint li = 0; li < L; li++) {
                cost += mg.GetGraph(li).GetAdjLst()[fe.node][0];
            }
            cost += static_cast<unsigned long long>(scv_sets[fe.node].NumSCVs()) * 8ULL;
        }
        return cost;
    };
    initial_cost = initial_frontier_cost();

    if (deleted_edges.size() == 1 && !frontier.empty()) {
        unsigned long long neighbor_threshold =
            DeleteSingleEdgeParallelNeighborWorkThreshold();
        std::vector<uint> neighbor_scratch;
        for (const FailEntry& fe : frontier) {
            if (fe.node >= n) continue;
            neighbor_scratch.clear();
            const std::vector<uint>& neighbors =
                GetCachedUniqueNeighbors(fe.node, neighbor_scratch);
            for (uint y : neighbors) {
                single_edge_neighbor_work += scv_sets[y].NumSCVs();
                if (single_edge_neighbor_work >= neighbor_threshold) break;
            }
            if (single_edge_neighbor_work >= neighbor_threshold) break;
        }
    }

    bool use_parallel_delete_bfs =
        omp_get_max_threads() > 1 && !frontier.empty() &&
        (frontier.size() >= 64 || initial_cost >= 4096 ||
         (deleted_edges.size() == 1 &&
          (single_edge_endpoint_scvs >= DeleteSingleEdgeParallelThreshold() ||
           single_edge_endpoint_work >= DeleteSingleEdgeParallelWorkThreshold() ||
           single_edge_neighbor_work >= DeleteSingleEdgeParallelNeighborWorkThreshold())));

    if (use_parallel_delete_bfs) {
        std::vector<FailEntry> initial_frontier(frontier.begin(), frontier.end());
        frontier.clear();

        size_t state_capacity = std::max<size_t>(static_cast<size_t>(n) * 8, 4096);
        state_capacity = std::min<size_t>(state_capacity, 1u << 23);
        AtomicDeleteStateSet propagated_delete_states(state_capacity);
        for (const FailEntry& fe : initial_frontier) {
            if (fe.scv_idx != UINT_MAX) {
                propagated_delete_states.TryMarkId(fe.node, fe.scv_idx);
            } else {
                propagated_delete_states.TryMark(fe.node, fe.scv, fe.broken_dim);
            }
        }

        int T = omp_get_max_threads();
        std::vector<std::deque<FailEntry>> work_queues(T);
        std::vector<std::mutex> work_mutexes(T);
        std::vector<std::vector<uint>> local_touched(T);
        std::vector<std::vector<std::pair<uint, std::vector<uint>>>> local_affected_scvs(
            affected_scvs ? T : 0);
        std::atomic<long long> active_tasks{0};

        auto push_task = [&](int owner, FailEntry&& task) {
            {
                std::lock_guard<std::mutex> lock(work_mutexes[owner]);
                work_queues[owner].push_back(std::move(task));
            }
            active_tasks.fetch_add(1, std::memory_order_release);
        };

        auto pop_task = [&](int tid, FailEntry& task) {
            {
                std::lock_guard<std::mutex> lock(work_mutexes[tid]);
                if (!work_queues[tid].empty()) {
                    task = std::move(work_queues[tid].back());
                    work_queues[tid].pop_back();
                    return true;
                }
            }
            for (int offset = 1; offset < T; offset++) {
                int victim = (tid + offset) % T;
                std::lock_guard<std::mutex> lock(work_mutexes[victim]);
                if (!work_queues[victim].empty()) {
                    task = std::move(work_queues[victim].front());
                    work_queues[victim].pop_front();
                    return true;
                }
            }
            return false;
        };

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::vector<uint> unique_neighbors;
            auto& touched = local_touched[tid];

            auto deletion_transition_holds_local = [&](uint x,
                                                       const std::vector<uint>& k_fail,
                                                       uint delete_layer,
                                                       const uint* k_y) {
                if (delete_layer >= L) return false;
                for (uint d = 0; d < L; d++) {
                    if (k_fail[d] < k_y[d]) return false;
                }
                for (uint d = 0; d < L; d++) {
                    uint value = max_c[x][d].load(std::memory_order_relaxed);
                    if (d == delete_layer && value > 0) value--;
                    if (value < k_y[d]) return true;
                }

                uint x_count = scv_sets[x].NumSCVs();
                for (uint si = 0; si < x_count; si++) {
                    const uint* h = scv_sets[x].GetSCVFlat(si);
                    bool reduced_dominates = true;
                    for (uint d = 0; d < L; d++) {
                        uint value = h[d];
                        if (d == delete_layer && value > 0) value--;
                        if (value < k_y[d]) {
                            reduced_dominates = false;
                            break;
                        }
                    }
                    if (reduced_dominates) return false;
                }
                return true;
            };

            #pragma omp for schedule(dynamic, 64)
            for (size_t idx = 0; idx < initial_frontier.size(); idx++) {
                push_task(tid, FailEntry(initial_frontier[idx]));
            }

            while (active_tasks.load(std::memory_order_acquire) > 0) {
                FailEntry fe;
                if (!pop_task(tid, fe)) continue;

                uint x = fe.node;
                const std::vector<uint>& k_fail = fe.scv;
                uint delete_layer = fe.broken_dim;
                
                touched.push_back(x);
                if (delete_layer >= L) {
                    active_tasks.fetch_sub(1, std::memory_order_acq_rel);
                    continue;
                }

                const std::vector<uint>& neighbors =
                    GetCachedUniqueNeighbors(x, unique_neighbors);
                

                uint level = k_fail[delete_layer];
                for (uint y : neighbors) {
                    if (max_c[y][delete_layer].load(std::memory_order_relaxed) < level) continue;
                    uint y_count = scv_sets[y].NumSCVs();
                    
                    for (uint yi = 0; yi < y_count; yi++) {
                        const uint* raw_y = scv_sets[y].GetSCVFlat(yi);
                        if (raw_y[delete_layer] != level) continue;
                        
                        if (!deletion_transition_holds_local(x, k_fail, delete_layer, raw_y)) continue;
                        if (!propagated_delete_states.TryMarkId(y, yi)) continue;
                        std::vector<uint> k_y(raw_y, raw_y + L);
                        touched.push_back(y);
                        if (affected_scvs) {
                            local_affected_scvs[tid].emplace_back(y, k_y);
                        }
                        
                        push_task(tid, FailEntry(y, std::move(k_y), delete_layer, yi));
                    }
                }

                active_tasks.fetch_sub(1, std::memory_order_acq_rel);
            }
        }

        for (auto& local : local_touched) {
            for (uint node : local) {
                if (node < needs_check.size()) needs_check[node] = 1;
                if (affected_domain && node < affected_domain->size()) {
                    (*affected_domain)[node] = 1;
                }
            }
        }
        if (affected_scvs) {
            for (auto& local : local_affected_scvs) {
                affected_scvs->insert(affected_scvs->end(),
                                      std::make_move_iterator(local.begin()),
                                      std::make_move_iterator(local.end()));
            }
        }
        
        return;
    }

    std::vector<uint> unique_neighbors;

    while (!frontier.empty()) {
        FailEntry fe = std::move(frontier.front());
        frontier.pop_front();

        uint x = fe.node;
        const std::vector<uint>& k_fail = fe.scv;
        uint delete_layer = fe.broken_dim;
        
        mark_domain(x);
        if (delete_layer >= L) continue;

        const std::vector<uint>& neighbors =
            GetCachedUniqueNeighbors(x, unique_neighbors);
        

        for (uint y : neighbors) {
            uint level = k_fail[delete_layer];
            if (max_c[y][delete_layer].load(std::memory_order_relaxed) < level) continue;
            uint y_count = scv_sets[y].NumSCVs();
            
            for (uint yi = 0; yi < y_count; yi++) {
                const uint* raw_y = scv_sets[y].GetSCVFlat(yi);
                if (raw_y[delete_layer] != level) continue;
                
                if (!deletion_transition_holds(x, k_fail, delete_layer, raw_y)) continue;
                enqueue_failed(y, std::vector<uint>(raw_y, raw_y + L),
                               delete_layer, yi);
            }
        }
    }
    
}

void DynamicMLCore::RecoverTrueSCV() {
    Decompose(true);

    has_lower_bounds = false;
    lower_bounds_snapshot.clear();
    lower_bounds_active.clear();
}

void DynamicMLCore::AddEdgeToGraph(MultilayerGraph& mg, uint u, uint v, uint l) {
    uint** adj = mg.GetGraph(l).GetAdjLst();

    uint deg_u = adj[u][0];
    uint* new_adj_u = new uint[deg_u + 2];
    new_adj_u[0] = deg_u + 1;
    for (uint i = 1; i <= deg_u; i++) new_adj_u[i] = adj[u][i];
    new_adj_u[deg_u + 1] = v;
    adj[u] = new_adj_u;

    uint deg_v = adj[v][0];
    uint* new_adj_v = new uint[deg_v + 2];
    new_adj_v[0] = deg_v + 1;
    for (uint i = 1; i <= deg_v; i++) new_adj_v[i] = adj[v][i];
    new_adj_v[deg_v + 1] = u;
    adj[v] = new_adj_v;
}

void DynamicMLCore::BatchAddEdgesToGraph(MultilayerGraph& mg, uint num_layers,
                                         const std::vector<EdgeTuple>& edges) {
    std::vector<std::vector<std::pair<uint, uint>>> layer_edges(num_layers);
    for (const auto& e : edges) {
        layer_edges[e.l].emplace_back(e.u, e.v);
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (uint l = 0; l < num_layers; l++) {
        if (layer_edges[l].empty()) continue;

        uint** adj = mg.GetGraph(l).GetAdjLst();

        std::unordered_map<uint, std::vector<uint>> new_neighbors;
        for (const auto& uv : layer_edges[l]) {
            new_neighbors[uv.first].push_back(uv.second);
            new_neighbors[uv.second].push_back(uv.first);
        }

        for (auto& kv : new_neighbors) {
            uint v = kv.first;
            auto& neighbors = kv.second;
            uint deg_v = adj[v][0];
            uint new_deg = deg_v + neighbors.size();
            uint* new_adj = new uint[new_deg + 1];
            new_adj[0] = new_deg;
            for (uint i = 1; i <= deg_v; i++) new_adj[i] = adj[v][i];
            for (uint i = 0; i < neighbors.size(); i++) new_adj[deg_v + 1 + i] = neighbors[i];
            adj[v] = new_adj;
        }
    }
}

void DynamicMLCore::BatchComputeCappedUpperBound(
    const std::vector<EdgeTuple>& batch_edges,
    const std::vector<SkylineSet>& old_scv_sets,
    std::vector<char>& needs_check,
    bool force_affected_scope,
    const std::vector<char>* commit_domain,
    const std::vector<SkylineSet>* commit_dominance_sets,
    std::vector<SkylineSet>* out_candidate_sets,
    std::vector<uint>* out_candidate_nodes,
    std::vector<SkylineSet>* out_inserted_scv_sets,
    std::vector<std::pair<uint, std::vector<uint>>>* out_candidate_records) {
    auto t0 = std::chrono::high_resolution_clock::now();

    bool same_layer_group = !batch_edges.empty();
    uint group_layer = same_layer_group ? batch_edges[0].l : UINT_MAX;
    for (const auto& e : batch_edges) {
        if (e.l != group_layer) {
            same_layer_group = false;
            break;
        }
    }
    bool output_only = out_candidate_sets != nullptr || out_candidate_records != nullptr;
    struct PendingLockGuard {
        bool old_value;
        PendingLockGuard(bool enabled) : old_value(g_disable_pending_candidate_locks) {
            if (enabled) g_disable_pending_candidate_locks = true;
        }
        ~PendingLockGuard() {
            g_disable_pending_candidate_locks = old_value;
        }
    } pending_lock_guard(output_only && omp_in_parallel());
    bool grouped_affected_scope =
        same_layer_group && force_affected_scope &&
        true;
    const std::vector<SkylineSet>& commit_basis =
        commit_dominance_sets != nullptr ? *commit_dominance_sets : old_scv_sets;
    auto max_c_can_dominate = [&](uint node, const uint* k) {
        for (uint d = 0; d < L; d++) {
            if (max_c[node][d].load(std::memory_order_relaxed) < k[d]) return false;
        }
        return true;
    };

    if (grouped_affected_scope && output_only) {
        uint grow_layer = group_layer;
        std::vector<uint**> adj_arrays(L);
        for (uint d = 0; d < L; d++) adj_arrays[d] = mg.GetGraph(d).GetAdjLst();
        uint** adj_grow = adj_arrays[grow_layer];
        size_t insert_state_capacity = std::max<size_t>(static_cast<size_t>(n) * 4, 4096);
        insert_state_capacity = std::min<size_t>(insert_state_capacity, 1u << 23);
        auto dominated_by_original = [&](const SkylineSet& set,
                                         const std::vector<uint>& k) {
            return set.IsDominatedRaw(k.data(), L);
        };

        auto same_level_insertion_transition = [&](const std::vector<uint>& base,
                                                   uint level) {
            return base[grow_layer] == level;
        };

        auto target_old_dominates_candidate = [&](uint target_node,
                                                  const uint* candidate) {
            if (!max_c_can_dominate(target_node, candidate)) return false;
            return commit_basis[target_node].IsDominatedRaw(candidate, L);
        };

        auto insertion_transition_holds = [&](uint supporter_node,
                                              const SkylineSet& supporter_old_set,
                                              uint target_node,
                                              const uint* candidate) {
            if (max_c_can_dominate(supporter_node, candidate) &&
		                supporter_old_set.IsDominatedRaw(candidate, L)) return false;
		            if (target_old_dominates_candidate(target_node, candidate)) {
	                return false;
	            }
				            return true;
				        };

        bool use_parallel_single_edge = false;
        unsigned long long single_edge_endpoint_scvs = 0;
        unsigned long long single_edge_endpoint_work = 0;
        unsigned long long single_edge_neighbor_work = 0;
        if (batch_edges.size() == 1) {
            const EdgeTuple& edge = batch_edges[0];
            unsigned long long u_scvs = old_scv_sets[edge.u].NumSCVs();
            unsigned long long v_scvs = edge.v == edge.u ? 0 : old_scv_sets[edge.v].NumSCVs();
            single_edge_endpoint_scvs = u_scvs + v_scvs;
            if (edge.v != edge.u) {
                single_edge_endpoint_work =
                    2ULL * u_scvs * std::max<unsigned long long>(1, v_scvs);
            } else {
                single_edge_endpoint_work = u_scvs;
            }
            unsigned long long neighbor_threshold =
                InsertSingleEdgeParallelNeighborWorkThreshold();
            std::vector<uint> neighbor_scratch;
            auto accumulate_neighbor_work = [&](uint node) {
                neighbor_scratch.clear();
                const std::vector<uint>& neighbors =
                    GetCachedUniqueNeighbors(node, neighbor_scratch);
                for (uint y : neighbors) {
                    single_edge_neighbor_work += old_scv_sets[y].NumSCVs();
                    if (single_edge_neighbor_work >= neighbor_threshold) break;
                }
            };
            accumulate_neighbor_work(edge.u);
            if (edge.v != edge.u && single_edge_neighbor_work < neighbor_threshold) {
                accumulate_neighbor_work(edge.v);
            }
            use_parallel_single_edge =
                omp_get_max_threads() > 1 &&
                (single_edge_endpoint_scvs >= InsertSingleEdgeParallelThreshold() ||
                 single_edge_endpoint_work >= InsertSingleEdgeParallelWorkThreshold() ||
                 single_edge_neighbor_work >= neighbor_threshold);
        }

        if (batch_edges.size() == 1 && !use_parallel_single_edge) {
            using InsertTask = std::pair<uint, std::vector<uint>>;
            std::vector<std::pair<uint, std::vector<uint>>> records;
            std::vector<uint> touched;
            std::deque<InsertTask> frontier;
            size_t seen_capacity = std::max<size_t>(static_cast<size_t>(n) / 16, 4096);
            seen_capacity = std::min<size_t>(seen_capacity, 1u << 20);
            AtomicDeleteStateSet seen_records(seen_capacity);
            std::unordered_map<unsigned long long, SkylineSet> affected_skyline_by_level;
            records.reserve(64);
            touched.reserve(64);
            affected_skyline_by_level.reserve(128);

            std::vector<uint> r(L);
		            std::vector<uint> base(L);
		            std::vector<uint> candidate(L);
		            std::vector<uint> unique_neighbors;
                std::vector<std::pair<uint, std::vector<uint>>> task_candidates;
            unsigned long long endpoint_max_pruned = 0;
            unsigned long long transition_max_pruned = 0;
            unsigned long long transition_scv_scanned = 0;
            unsigned long long emitted_records = 0;
            auto keep_nonredundant_base_serial = [&](uint node, const std::vector<uint>& base) {
                unsigned long long key =
                    (static_cast<unsigned long long>(node) << 32) |
                    static_cast<unsigned long long>(base[grow_layer]);
                SkylineSet& level_skyline = affected_skyline_by_level[key];
                if (level_skyline.IsDominatedRaw(base.data(), L)) return false;
                level_skyline.InsertKnownUndominatedRaw(base.data(), L);
                return true;
            };

	            auto emit_record_serial = [&](uint node, const std::vector<uint>& base) {
                    bool mark_ok = false;
                    
                        mark_ok = seen_records.TryMark(node, base, grow_layer);
                    
	                if (!mark_ok) return false;
	                if (commit_domain != nullptr && !(*commit_domain)[node]) {
                        
                        return false;
                    }

                uint limit_l = adj_grow[node][0];
                if (base[grow_layer] >= limit_l) {
                    
                    return false;
                }

                std::vector<uint> upper = base;
                upper[grow_layer]++;
	                if (max_c_can_dominate(node, upper.data()) &&
	                    dominated_by_original(commit_basis[node], upper)) {
                        
	                    return false;
	                }
                if (!keep_nonredundant_base_serial(node, base)) {
                    
                    return false;
                }

		                records.emplace_back(node, std::move(upper));
	                touched.push_back(node);
	                frontier.emplace_back(node, base);
                emitted_records++;
                
	                return true;
            };

            const EdgeTuple& edge = batch_edges[0];
            auto activate_endpoint = [&](uint node, uint other) {
                const SkylineSet& node_scvs = old_scv_sets[node];
                const SkylineSet& other_scvs = old_scv_sets[other];
                uint node_count = node_scvs.NumSCVs();
                uint other_count = other_scvs.NumSCVs();
                for (uint si = 0; si < node_count; si++) {
                    const uint* k = node_scvs.GetSCVFlat(si);
                    if (k[grow_layer] >= adj_grow[node][0]) continue;
                    std::vector<uint> upper(k, k + L);
                    upper[grow_layer]++;
                    if (max_c_can_dominate(node, upper.data()) &&
                        dominated_by_original(commit_basis[node], upper)) {
                        continue;
                    }
                    if (max_c[other][grow_layer].load(std::memory_order_relaxed) < k[grow_layer]) {
                        endpoint_max_pruned++;
                        continue;
                    }
                    for (uint oi = 0; oi < other_count; oi++) {
                        const uint* h = other_scvs.GetSCVFlat(oi);
                        if (h[grow_layer] < k[grow_layer]) continue;
                        for (uint d = 0; d < L; d++) r[d] = std::min(k[d], h[d]);
                        emit_record_serial(node, r);
                        if (SameSCVRaw(r.data(), k, L)) break;
                    }
                }
            };
            activate_endpoint(edge.u, edge.v);
            if (edge.v != edge.u) activate_endpoint(edge.v, edge.u);

            while (!frontier.empty()) {
                InsertTask item = std::move(frontier.front());
                frontier.pop_front();
	                uint x = item.first;
	                const std::vector<uint>& k_x_base = item.second;
	                const SkylineSet& x_scvs = old_scv_sets[x];
	                uint level = k_x_base[grow_layer];

					                unique_neighbors.clear();
		                const std::vector<uint>& neighbors =
		                    GetCachedUniqueNeighbors(x, unique_neighbors);

	                for (uint y : neighbors) {
                    if (max_c[y][grow_layer].load(std::memory_order_relaxed) < level) {
                        transition_max_pruned++;
                        continue;
                    }
                    const SkylineSet& y_scvs = old_scv_sets[y];
                    uint y_count = y_scvs.NumSCVs();
                    uint limit_y = adj_grow[y][0];
                    if (level >= limit_y) continue;
                    for (uint yi = 0; yi < y_count; yi++) {
                        const uint* k_y = y_scvs.GetSCVFlat(yi);
                        transition_scv_scanned++;
                        if (k_y[grow_layer] != level) continue;
                        for (uint d = 0; d < L; d++) base[d] = std::min(k_x_base[d], k_y[d]);
                        if (!same_level_insertion_transition(base, level)) continue;
                        candidate = base;
                        candidate[grow_layer]++;
                        if (!insertion_transition_holds(x, x_scvs, y, candidate.data())) continue;
                        emit_record_serial(y, base);
                    }
                }
            }

	            if (out_candidate_records != nullptr) {
	                out_candidate_records->reserve(out_candidate_records->size() + records.size());
	                out_candidate_records->insert(out_candidate_records->end(),
	                                              std::make_move_iterator(records.begin()),
	                                              std::make_move_iterator(records.end()));
	            }
            for (uint node : touched) needs_check[node] = 1;
            
            
            
	            return;
	        }

        AtomicDeleteStateSet propagated_insert_states(insert_state_capacity);
        bool enable_redundant_base_filter = batch_edges.size() >= 16;
        static constexpr size_t kAffectedSkylineShards = 4096;
        std::vector<std::mutex> affected_skyline_mutexes(enable_redundant_base_filter ? kAffectedSkylineShards : 0);
        std::vector<std::unordered_map<unsigned long long, SkylineSet>> affected_skyline_by_key(
            enable_redundant_base_filter ? kAffectedSkylineShards : 0);

		        int T = omp_get_max_threads();
			        std::vector<std::vector<std::pair<uint, std::vector<uint>>>> local_records(T);
		        std::vector<std::vector<uint>> local_touched(T);
		        using InsertTask = std::pair<uint, std::vector<uint>>;
		        std::vector<std::deque<InsertTask>> work_queues(T);
		        std::vector<std::mutex> work_mutexes(T);
        std::vector<std::vector<std::pair<uint, std::vector<uint>>>> local_endpoint_records(T);
        std::vector<std::pair<uint, std::vector<uint>>> compacted_endpoint_records;
        std::atomic<long long> active_tasks{0};

        auto push_task = [&](int owner, InsertTask&& task) {
            {
                std::lock_guard<std::mutex> lock(work_mutexes[owner]);
                work_queues[owner].push_back(std::move(task));
            }
            active_tasks.fetch_add(1, std::memory_order_relaxed);
        };

        auto pop_task = [&](int tid, InsertTask& task) {
            {
                std::lock_guard<std::mutex> lock(work_mutexes[tid]);
                if (!work_queues[tid].empty()) {
                    task = std::move(work_queues[tid].back());
                    work_queues[tid].pop_back();
                    return true;
                }
            }
            for (int offset = 1; offset < T; offset++) {
                int victim = (tid + offset) % T;
                std::lock_guard<std::mutex> lock(work_mutexes[victim]);
                if (!work_queues[victim].empty()) {
                    task = std::move(work_queues[victim].front());
                    work_queues[victim].pop_front();
                    return true;
                }
            }
            return false;
        };

        auto keep_nonredundant_base = [&](uint node, const std::vector<uint>& base) {
            if (!enable_redundant_base_filter) return true;
            uint level = base[grow_layer];
            unsigned long long key =
                (static_cast<unsigned long long>(node) << 32) |
                static_cast<unsigned long long>(level);
            size_t shard = Mix64(key) & (kAffectedSkylineShards - 1);
            std::lock_guard<std::mutex> lock(affected_skyline_mutexes[shard]);
            SkylineSet& level_skyline = affected_skyline_by_key[shard][key];
            if (level_skyline.IsDominatedRaw(base.data(), L)) return false;
            level_skyline.InsertKnownUndominatedRaw(base.data(), L);
            return true;
        };

        auto try_mark_insert_state = [&](AtomicDeleteStateSet& states,
                                         uint node,
                                         const std::vector<uint>& base) {
            auto fp = MakeDeleteStateFingerprint(node, base, grow_layer);
            return states.TryMarkFingerprint(fp.first, fp.second);
        };

        auto compact_endpoint_records = [&](std::vector<std::pair<uint, std::vector<uint>>>& records) {
            if (records.size() < 2) return;
            std::sort(records.begin(), records.end(),
                      [&](const auto& a, const auto& b) {
                          if (a.first != b.first) return a.first < b.first;
                          uint a_level = a.second[grow_layer];
                          uint b_level = b.second[grow_layer];
                          if (a_level != b_level) return a_level < b_level;
                          return a.second < b.second;
                      });

            std::vector<std::pair<uint, std::vector<uint>>> compacted;
            compacted.reserve(records.size());
            size_t pos = 0;
            while (pos < records.size()) {
                uint node = records[pos].first;
                uint level = records[pos].second[grow_layer];
                SkylineSet skyline;
                size_t next = pos;
                while (next < records.size() &&
                       records[next].first == node &&
                       records[next].second[grow_layer] == level) {
                    skyline.Insert(records[next].second);
                    next++;
                }
                uint count = skyline.NumSCVs();
                for (uint si = 0; si < count; si++) {
                    const uint* raw = skyline.GetSCVFlat(si);
                    compacted.emplace_back(node, std::vector<uint>(raw, raw + L));
                }
                pos = next;
            }
            records.swap(compacted);
        };

        struct EndpointSCVChunk {
            uint node;
            uint other;
            uint begin;
            uint end;
        };
        static constexpr uint kEndpointSCVChunkSize = 16;
        std::vector<EndpointSCVChunk> endpoint_scv_chunks;
        bool enable_group_endpoint_mcv_parallel = batch_edges.size() > 1;
        auto append_endpoint_chunks = [&](uint node, uint other) {
            uint count = old_scv_sets[node].NumSCVs();
            if (count == 0) return;
            for (uint begin = 0; begin < count; begin += kEndpointSCVChunkSize) {
                uint end = std::min<uint>(count, begin + kEndpointSCVChunkSize);
                endpoint_scv_chunks.push_back({node, other, begin, end});
            }
        };
        unsigned long long group_endpoint_total_work = 0;
        unsigned long long group_endpoint_max_edge_work = 0;
        if (batch_edges.size() > 1) {
            for (const EdgeTuple& edge : batch_edges) {
                unsigned long long u_count = old_scv_sets[edge.u].NumSCVs();
                unsigned long long v_count = old_scv_sets[edge.v].NumSCVs();
                unsigned long long edge_work = u_count * v_count;
                if (edge.v != edge.u) edge_work *= 2;
                group_endpoint_total_work += edge_work;
                group_endpoint_max_edge_work =
                    std::max(group_endpoint_max_edge_work, edge_work);
            }
            unsigned long long parallel_slots =
                static_cast<unsigned long long>(std::max(1, std::min<int>(T, batch_edges.size())));
            bool endpoint_work_is_skewed =
                group_endpoint_total_work >= 4096 &&
                group_endpoint_max_edge_work * parallel_slots >=
                    group_endpoint_total_work * 4;
            enable_group_endpoint_mcv_parallel = endpoint_work_is_skewed;
        }
        if (enable_group_endpoint_mcv_parallel || batch_edges.size() == 1) {
            endpoint_scv_chunks.reserve(batch_edges.size() * 2);
            for (const EdgeTuple& edge : batch_edges) {
                append_endpoint_chunks(edge.u, edge.v);
                if (edge.v != edge.u) append_endpoint_chunks(edge.v, edge.u);
            }
        }

		auto emit_record = [&](int tid,
		                               uint node,
		                               const std::vector<uint>& base) {
	            if (!try_mark_insert_state(propagated_insert_states, node, base)) return false;
	            if (commit_domain != nullptr && !(*commit_domain)[node]) {
                    
                    return false;
                }

            uint limit_l = adj_grow[node][0];
            if (base[grow_layer] >= limit_l) {
                
                return false;
            }

            std::vector<uint> upper = base;
            upper[grow_layer]++;
	            if (max_c_can_dominate(node, upper.data()) &&
				                dominated_by_original(commit_basis[node], upper)) {
                            
				                return false;
				            }
		            if (!keep_nonredundant_base(node, base)) {
                        
                        return false;
                    }

						            local_records[tid].emplace_back(node, std::move(upper));
					            local_touched[tid].push_back(node);
                            
                            
				            push_task(tid, InsertTask(node, base));
			            return true;
		        };

        #pragma omp parallel
        {
		            int tid = omp_get_thread_num();
			            std::vector<uint> r(L);
				            std::vector<uint> base(L);
				            std::vector<uint> candidate(L);
				            std::vector<uint> unique_neighbors;
		                std::vector<std::pair<uint, std::vector<uint>>> task_candidates;
				            auto activate_endpoint_scv = [&](uint node, uint other, uint si) {
		                const SkylineSet& node_scvs = old_scv_sets[node];
		                const SkylineSet& other_scvs = old_scv_sets[other];
	                uint other_count = other_scvs.NumSCVs();
	                const uint* k = node_scvs.GetSCVFlat(si);
	                if (k[grow_layer] >= adj_grow[node][0]) return;
	                std::vector<uint> upper(k, k + L);
	                upper[grow_layer]++;
	                if (max_c_can_dominate(node, upper.data()) &&
	                    dominated_by_original(commit_basis[node], upper)) {
	                    return;
	                }
	                if (max_c[other][grow_layer].load(std::memory_order_relaxed) < k[grow_layer]) {
	                    
	                    return;
	                }
	                for (uint oi = 0; oi < other_count; oi++) {
	                    const uint* h = other_scvs.GetSCVFlat(oi);
	                    if (h[grow_layer] < k[grow_layer]) {
	                        continue;
		                    }
		                    for (uint d = 0; d < L; d++) r[d] = std::min(k[d], h[d]);
                            local_endpoint_records[tid].emplace_back(node, r);
		                    if (SameSCVRaw(r.data(), k, L)) break;
		                }
		            };

	            if (batch_edges.size() == 1 || enable_group_endpoint_mcv_parallel) {
	                #pragma omp for schedule(dynamic, 1)
	                for (size_t job = 0; job < endpoint_scv_chunks.size(); job++) {
	                    const EndpointSCVChunk& chunk = endpoint_scv_chunks[job];
	                    for (uint si = chunk.begin; si < chunk.end; si++) {
	                        activate_endpoint_scv(chunk.node, chunk.other, si);
	                    }
	                }
	            } else {
	                #pragma omp for schedule(dynamic, 1)
	                for (size_t edge_idx = 0; edge_idx < batch_edges.size(); edge_idx++) {
	                    const EdgeTuple& edge = batch_edges[edge_idx];
	                    auto activate_endpoint = [&](uint node, uint other) {
	                        const SkylineSet& node_scvs = old_scv_sets[node];
	                        uint node_count = node_scvs.NumSCVs();
	                        for (uint si = 0; si < node_count; si++) {
	                            activate_endpoint_scv(node, other, si);
	                        }
	                    };
	                    activate_endpoint(edge.u, edge.v);
		                    if (edge.v != edge.u) activate_endpoint(edge.v, edge.u);
		                }
		            }

                #pragma omp single
                {
                    size_t total_endpoint_records = 0;
                    for (const auto& local : local_endpoint_records) {
                        total_endpoint_records += local.size();
                    }
                    compacted_endpoint_records.clear();
                    compacted_endpoint_records.reserve(total_endpoint_records);
                    for (auto& local : local_endpoint_records) {
                        compact_endpoint_records(local);
                        compacted_endpoint_records.insert(
                            compacted_endpoint_records.end(),
                            std::make_move_iterator(local.begin()),
                            std::make_move_iterator(local.end()));
                        local.clear();
                    }
                    compact_endpoint_records(compacted_endpoint_records);
                }

                #pragma omp for schedule(dynamic, 64)
                for (size_t idx = 0; idx < compacted_endpoint_records.size(); idx++) {
                    const auto& item = compacted_endpoint_records[idx];
                    emit_record(tid, item.first, item.second);
                }

	            while (active_tasks.load(std::memory_order_acquire) > 0) {
	                InsertTask item;
	                if (!pop_task(tid, item)) continue;
                    
                    
				                uint x = item.first;
				                const std::vector<uint>& k_x_base = item.second;
				                const SkylineSet& x_scvs = old_scv_sets[x];
				                uint level = k_x_base[grow_layer];
                    

					                unique_neighbors.clear();
		                const std::vector<uint>& neighbors =
		                    GetCachedUniqueNeighbors(x, unique_neighbors);
                    

		                for (uint y : neighbors) {
                    if (max_c[y][grow_layer].load(std::memory_order_relaxed) < level) {
                        
                        
                        continue;
                    }
                    const SkylineSet& y_scvs = old_scv_sets[y];
		                    uint y_count = y_scvs.NumSCVs();
		                    uint limit_y = adj_grow[y][0];
				                    if (level >= limit_y) {
                            
                            continue;
                        }
                                
                                
	                                
	                                auto process_transition_scv = [&](uint yi) {
	                                    const uint* k_y = y_scvs.GetSCVFlat(yi);
                                    
	                                    
	                                    if (k_y[grow_layer] != level) {
	                                        return;
	                                    }
                                    
	                                    for (uint d = 0; d < L; d++) base[d] = std::min(k_x_base[d], k_y[d]);
	                                    if (!same_level_insertion_transition(base, level)) {
	                                        return;
	                                    }
	                                    candidate = base;
	                                    candidate[grow_layer]++;
                                    if (!insertion_transition_holds(x, x_scvs, y, candidate.data())) {
                                        return;
                                    }
                                    
	                                    
	                                    emit_record(tid, y, base);
                                            
	                                    
	                                };

                                    for (uint yi = 0; yi < y_count; yi++) {
                                        process_transition_scv(yi);
                                    }
                                
			                }
	                active_tasks.fetch_sub(1, std::memory_order_acq_rel);
	            }
	        }

	        if (out_candidate_records != nullptr) {
	            size_t total = 0;
	            for (const auto& local : local_records) total += local.size();
	            out_candidate_records->reserve(out_candidate_records->size() + total);
	            for (auto& local : local_records) {
	                out_candidate_records->insert(out_candidate_records->end(),
	                                              std::make_move_iterator(local.begin()),
	                                              std::make_move_iterator(local.end()));
	            }
	        }
            for (const auto& local : local_touched) {
                for (uint node : local) {
                    needs_check[node] = 1;
                }
            }
            
            
            
            
            
		        return;
		    }

    if (grouped_affected_scope) {
        uint grow_layer = group_layer;
        std::deque<std::pair<uint, std::vector<uint>>> frontier;
        size_t seen_capacity = std::max<size_t>(static_cast<size_t>(n) * 4, 4096);
        seen_capacity = std::min<size_t>(seen_capacity, 1u << 23);
        AtomicDeleteStateSet seen_records(seen_capacity);
        uint capped_sources = 0;
        unsigned long long endpoint_total_scvs = 0;
        unsigned long long endpoint_initial_scvs = 0;
        unsigned long long endpoint_refine_only_scvs = 0;
        unsigned long long scope_refine_only_scvs = 0;
        unsigned long long scope_reached_scvs = 0;
        unsigned long long scope_upper_inserted = 0;
        unsigned long long scope_neighbor_scvs = 0;
        unsigned long long scope_level_pruned = 0;
        unsigned long long scope_dom_pruned = 0;
        unsigned long long scope_seen_pruned = 0;
        unsigned long long level_scan_inserted = 0;
        unsigned long long level_scan_scvs = 0;
        auto dominated_by_original = [&](const SkylineSet& set,
                                         const std::vector<uint>& k) {
            return set.IsDominatedRaw(k.data(), L);
        };

        auto same_level_insertion_transition = [&](const std::vector<uint>& base,
                                                   uint level) {
            return base[grow_layer] == level;
        };

	        auto insertion_transition_holds = [&](uint supporter_node,
	                                              const SkylineSet& supporter_old_set,
	                                              uint target_node,
	                                              const uint* candidate) {
	            if (max_c_can_dominate(supporter_node, candidate) &&
	                supporter_old_set.IsDominatedRaw(candidate, L)) return false;
	            if (max_c_can_dominate(target_node, candidate) &&
                commit_basis[target_node].IsDominatedRaw(candidate, L)) {
                return false;
            }
            return true;
        };

        auto insert_upper_record = [&](uint node, const std::vector<uint>& upper) {
            if (commit_domain != nullptr && !(*commit_domain)[node]) return false;
            uint limit_l = mg.GetGraph(grow_layer).GetAdjLst()[node][0];
            if (upper[grow_layer] == 0) return false;

            std::vector<uint> k_prime = upper;
            if (k_prime[grow_layer] > limit_l) {
                k_prime[grow_layer] = limit_l;
                capped_sources++;
            }
            if (max_c_can_dominate(node, k_prime.data()) &&
                dominated_by_original(commit_basis[node], k_prime)) {
                return false;
            }

            if (output_only) {
                if (out_candidate_sets != nullptr) {
                    (*out_candidate_sets)[node].Insert(k_prime);
                }
                if (out_inserted_scv_sets != nullptr) {
                    (*out_inserted_scv_sets)[node].Insert(k_prime);
                }
                if (out_candidate_records != nullptr) {
                    out_candidate_records->emplace_back(node, std::move(k_prime));
                }
                needs_check[node] = 1;
                scope_upper_inserted++;
                return true;
            }

            scv_rw_locks[node]->lock();
            bool inserted = scv_sets[node].Insert(k_prime);
            if (inserted) {
                if (out_inserted_scv_sets != nullptr) {
                    (*out_inserted_scv_sets)[node].Insert(k_prime);
                }
                if (out_candidate_nodes != nullptr && !needs_check[node]) {
                    out_candidate_nodes->push_back(node);
                }
                UpdateMaxC(node);
                needs_check[node] = 1;
                scope_upper_inserted++;
            }
            scv_rw_locks[node]->unlock();
            return true;
        };

        auto push_base_record = [&](uint node, const std::vector<uint>& base, bool enqueue) {
            if (!seen_records.TryMark(node, base, grow_layer)) {
                scope_seen_pruned++;
                return false;
            }
            std::vector<uint> upper = base;
            upper[grow_layer]++;
            if (!insert_upper_record(node, upper)) return false;
            if (enqueue) frontier.emplace_back(node, base);
            scope_reached_scvs++;
            return true;
        };

        auto activate_endpoint = [&](uint node, uint other) {
            const SkylineSet& node_scvs = old_scv_sets[node];
            const SkylineSet& other_scvs = old_scv_sets[other];
            uint node_count = node_scvs.NumSCVs();
            uint other_count = other_scvs.NumSCVs();
            endpoint_total_scvs += node_count;

            std::vector<uint> r(L);
            for (uint si = 0; si < node_count; si++) {
                const uint* k = node_scvs.GetSCVFlat(si);
                if (k[grow_layer] >= mg.GetGraph(grow_layer).GetAdjLst()[node][0]) continue;
                std::vector<uint> upper(k, k + L);
                upper[grow_layer]++;
                if (max_c_can_dominate(node, upper.data()) &&
                    dominated_by_original(commit_basis[node], upper)) {
                    continue;
                }
                for (uint oi = 0; oi < other_count; oi++) {
                    const uint* h = other_scvs.GetSCVFlat(oi);
                    if (h[grow_layer] < k[grow_layer]) continue;
                    for (uint d = 0; d < L; d++) r[d] = std::min(k[d], h[d]);

                    if (SameSCVRaw(r.data(), k, L)) endpoint_initial_scvs++;
                    else endpoint_refine_only_scvs++;
                    push_base_record(node, r, true);
                    if (SameSCVRaw(r.data(), k, L)) break;
                }
            }
        };

        for (const EdgeTuple& edge : batch_edges) {
            activate_endpoint(edge.u, edge.v);
            if (edge.v != edge.u) activate_endpoint(edge.v, edge.u);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        std::vector<uint**> adj_arrays(L);
        for (uint d = 0; d < L; d++) adj_arrays[d] = mg.GetGraph(d).GetAdjLst();
        uint** adj_grow = adj_arrays[grow_layer];
        std::vector<uint> unique_neighbors;
        while (!frontier.empty()) {
            auto [x, k_x_base] = frontier.front();
            frontier.pop_front();

			            const SkylineSet& x_scvs = old_scv_sets[x];
			            uint level = k_x_base[grow_layer];

				            unique_neighbors.clear();
            const std::vector<uint>& neighbors =
                GetCachedUniqueNeighbors(x, unique_neighbors);

            std::vector<uint> base(L);
            std::vector<uint> candidate(L);
            for (uint y : neighbors) {
                const SkylineSet& y_scvs = old_scv_sets[y];
                uint y_count = y_scvs.NumSCVs();
                uint limit_y = adj_grow[y][0];
                if (level >= limit_y) continue;
                for (uint yi = 0; yi < y_count; yi++) {
                    const uint* k_y = y_scvs.GetSCVFlat(yi);
                    scope_neighbor_scvs++;
                    if (k_y[grow_layer] != level) continue;
                    for (uint d = 0; d < L; d++) base[d] = std::min(k_x_base[d], k_y[d]);
                    if (!same_level_insertion_transition(base, level)) {
                        scope_level_pruned++;
                        continue;
	                    }
	                    candidate = base;
	                    candidate[grow_layer]++;
	                    if (!insertion_transition_holds(x, x_scvs, y, candidate.data())) {
	                        scope_dom_pruned++;
	                        continue;
	                    }
                    push_base_record(y, base, true);
                }
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        uint committed_nodes = 0;
        uint committed_scvs = 0;
        for (uint i = 0; i < n; i++) {
            if (!needs_check[i]) continue;
            committed_nodes++;
            uint old_count = old_scv_sets[i].NumSCVs();
            uint now_count = scv_sets[i].NumSCVs();
            if (now_count > old_count) committed_scvs += now_count - old_count;
        }
        auto t3 = std::chrono::high_resolution_clock::now();

        double activate_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        double bfs_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
        double commit_ms = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0;
        unsigned long long endpoint_kept = endpoint_initial_scvs + endpoint_refine_only_scvs;
        unsigned long long endpoint_pruned =
            endpoint_total_scvs >= endpoint_kept ? endpoint_total_scvs - endpoint_kept : 0;
        
        return;
    }

    std::vector<SkylineSet> pending_sets(n);
    std::vector<char> pending_touched(n, 0);
    std::vector<uint> pending_nodes;
    pending_nodes.reserve(batch_edges.size() * 2);
    std::vector<PropagationEntry> F_curr;

    std::vector<uint> endpoint_delta(static_cast<size_t>(n) * L, 0);
    std::vector<char> endpoint_seen(n, 0);
    std::vector<uint> activation_nodes;
    activation_nodes.reserve(batch_edges.size() * 2);
    auto add_endpoint_delta = [this, &endpoint_delta, &endpoint_seen, &activation_nodes](uint node, uint layer) {
        if (!endpoint_seen[node]) {
            endpoint_seen[node] = 1;
            activation_nodes.push_back(node);
        }
        endpoint_delta[static_cast<size_t>(node) * L + layer]++;
    };
    for (const auto& e : batch_edges) {
        add_endpoint_delta(e.u, e.l);
        add_endpoint_delta(e.v, e.l);
    }
    std::sort(activation_nodes.begin(), activation_nodes.end());

    uint capped_sources = 0;
    bool small_activation = activation_nodes.size() <= 64;
    if (small_activation) {
        std::vector<uint> k_prime(L);
        for (size_t idx = 0; idx < activation_nodes.size(); idx++) {
            uint node = activation_nodes[idx];
            const uint* delta = endpoint_delta.data() + static_cast<size_t>(node) * L;
            const SkylineSet& current_scvs = old_scv_sets[node];
            uint current_count = current_scvs.NumSCVs();

            for (uint si = 0; si < current_count; si++) {
                const uint* k = current_scvs.GetSCVFlat(si);
                for (uint layer = 0; layer < L; layer++) k_prime[layer] = k[layer];
                uint first_grow_dim = L;
                for (uint layer = 0; layer < L; layer++) {
                    if (delta[layer] == 0) continue;
                    uint max_deg_l = mg.GetGraph(layer).GetAdjLst()[node][0];
                    uint raised = k_prime[layer] + delta[layer];
                    uint next_value = std::min(raised, max_deg_l);
                    if (next_value != k_prime[layer] && first_grow_dim == L) first_grow_dim = layer;
                    k_prime[layer] = next_value;
                }
                if (first_grow_dim == L) continue;
                if (max_c_can_dominate(node, k_prime.data()) &&
                    commit_basis[node].IsDominated(k_prime)) {
                    continue;
                }

                bool inserted = TryInsertPendingCandidate(node, k_prime, old_scv_sets,
                                                          pending_sets, pending_touched,
                                                          pending_nodes);
                if (inserted) {
                    F_curr.push_back(PropagationEntry(node, k_prime, first_grow_dim));
                }
            }
        }
    } else {
        int T = omp_get_max_threads();
        std::vector<std::vector<PropagationEntry>> activation_frontiers(T);
        std::vector<std::vector<uint>> activation_touched(T);

        #pragma omp parallel reduction(+:capped_sources)
        {
            int tid = omp_get_thread_num();
            std::vector<uint> k_prime(L);

            #pragma omp for schedule(dynamic, 64)
            for (size_t idx = 0; idx < activation_nodes.size(); idx++) {
                uint node = activation_nodes[idx];
                const uint* delta = endpoint_delta.data() + static_cast<size_t>(node) * L;
                const SkylineSet& current_scvs = old_scv_sets[node];
                uint current_count = current_scvs.NumSCVs();

                for (uint si = 0; si < current_count; si++) {
                    const uint* k = current_scvs.GetSCVFlat(si);
                    for (uint layer = 0; layer < L; layer++) k_prime[layer] = k[layer];
                    uint first_grow_dim = L;
                    for (uint layer = 0; layer < L; layer++) {
                        if (delta[layer] == 0) continue;
                        uint max_deg_l = mg.GetGraph(layer).GetAdjLst()[node][0];
                        uint raised = k_prime[layer] + delta[layer];
                        uint next_value = std::min(raised, max_deg_l);
                        if (next_value != k_prime[layer] && first_grow_dim == L) first_grow_dim = layer;
                        k_prime[layer] = next_value;
                    }
                    if (first_grow_dim == L) continue;
                    if (max_c_can_dominate(node, k_prime.data()) &&
                        commit_basis[node].IsDominated(k_prime)) {
                        continue;
                    }

                    bool inserted = TryInsertPendingCandidate(node, k_prime, old_scv_sets,
                                                              pending_sets, pending_touched,
                                                              activation_touched[tid]);
                    if (inserted) {
                        activation_frontiers[tid].push_back(PropagationEntry(node, k_prime, first_grow_dim));
                    }
                }
            }
        }

        for (int t = 0; t < T; t++) {
            F_curr.insert(F_curr.end(), activation_frontiers[t].begin(), activation_frontiers[t].end());
            pending_nodes.insert(pending_nodes.end(), activation_touched[t].begin(), activation_touched[t].end());
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    uint initial_activation = F_curr.size();

    ParallelBFSCascadePendingCapped(F_curr, old_scv_sets,
                                    pending_sets, pending_touched, pending_nodes);
    auto t2 = std::chrono::high_resolution_clock::now();

    uint committed_nodes = 0;
    uint committed_scvs = 0;
    uint pending_node_count = pending_nodes.size();

    if (output_only) {
        if (out_candidate_nodes != nullptr) {
            out_candidate_nodes->reserve(out_candidate_nodes->size() + pending_nodes.size());
        }
        for (uint node : pending_nodes) {
            if (commit_domain != nullptr && !(*commit_domain)[node]) continue;
            uint candidate_count = pending_sets[node].NumSCVs();
            if (candidate_count == 0) continue;

            bool node_changed = false;
            for (uint si = 0; si < candidate_count; si++) {
                const uint* raw = pending_sets[node].GetSCVFlat(si);
                if (max_c_can_dominate(node, raw) &&
                    commit_basis[node].IsDominatedRaw(raw, L)) continue;
                bool inserted_record = false;
                if (out_candidate_sets != nullptr) {
                    inserted_record = (*out_candidate_sets)[node].InsertRaw(raw, L);
                    if (inserted_record && out_inserted_scv_sets != nullptr) {
                        (*out_inserted_scv_sets)[node].InsertRaw(raw, L);
                    }
                }
                if (out_candidate_records != nullptr) {
                    out_candidate_records->emplace_back(node, std::vector<uint>(raw, raw + L));
                    inserted_record = true;
                }
                if (inserted_record) {
                    node_changed = true;
                    committed_scvs++;
                }
            }
            if (node_changed) {
                needs_check[node] = 1;
                committed_nodes++;
                if (out_candidate_nodes != nullptr) out_candidate_nodes->push_back(node);
            }
        }
    } else if (pending_nodes.size() <= 64) {
        for (uint node : pending_nodes) {
            if (commit_domain != nullptr && !(*commit_domain)[node]) continue;
            uint candidate_count = pending_sets[node].NumSCVs();
            if (candidate_count == 0) continue;

            bool node_changed = false;
            scv_rw_locks[node]->lock();
            for (uint si = 0; si < candidate_count; si++) {
                const uint* raw = pending_sets[node].GetSCVFlat(si);
                if (max_c_can_dominate(node, raw) &&
                    commit_basis[node].IsDominatedRaw(raw, L)) continue;
                if (scv_sets[node].InsertRaw(raw, L)) {
                    if (out_inserted_scv_sets != nullptr) {
                        (*out_inserted_scv_sets)[node].InsertRaw(raw, L);
                    }
                    node_changed = true;
                    committed_scvs++;
                }
            }
            if (node_changed) {
                UpdateMaxC(node);
                needs_check[node] = 1;
                committed_nodes++;
                if (out_candidate_nodes != nullptr) out_candidate_nodes->push_back(node);
            }
            scv_rw_locks[node]->unlock();
        }
    } else {
        std::vector<std::vector<uint>> committed_node_locals;
        if (out_candidate_nodes != nullptr) {
            committed_node_locals.resize(omp_get_max_threads());
        }
        #pragma omp parallel for schedule(dynamic, 256) reduction(+:committed_nodes,committed_scvs)
        for (size_t idx = 0; idx < pending_nodes.size(); idx++) {
            uint node = pending_nodes[idx];
            if (commit_domain != nullptr && !(*commit_domain)[node]) continue;
            uint candidate_count = pending_sets[node].NumSCVs();
            if (candidate_count == 0) continue;

            bool node_changed = false;
            scv_rw_locks[node]->lock();
            for (uint si = 0; si < candidate_count; si++) {
                const uint* raw = pending_sets[node].GetSCVFlat(si);
                if (max_c_can_dominate(node, raw) &&
                    commit_basis[node].IsDominatedRaw(raw, L)) continue;
                if (scv_sets[node].InsertRaw(raw, L)) {
                    if (out_inserted_scv_sets != nullptr) {
                        (*out_inserted_scv_sets)[node].InsertRaw(raw, L);
                    }
                    node_changed = true;
                    committed_scvs++;
                }
            }
            if (node_changed) {
                UpdateMaxC(node);
                needs_check[node] = 1;
                committed_nodes++;
                if (out_candidate_nodes != nullptr) {
                    committed_node_locals[omp_get_thread_num()].push_back(node);
                }
            }
            scv_rw_locks[node]->unlock();
        }
        if (out_candidate_nodes != nullptr) {
            size_t extra = 0;
            for (const auto& local : committed_node_locals) extra += local.size();
            out_candidate_nodes->reserve(out_candidate_nodes->size() + extra);
            for (auto& local : committed_node_locals) {
                out_candidate_nodes->insert(out_candidate_nodes->end(),
                                            local.begin(), local.end());
            }
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    double activate_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    double bfs_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
    double commit_ms = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0;
    
}

void DynamicMLCore::BatchInsertEdges(const std::vector<std::tuple<uint, uint, uint>>& edges) {
    if (edges.empty()) return;

    g_batch_insert_active_layers.fill(0);
    g_has_batch_insert_active_layers = (L <= kMaxTrackedBatchLayers);
    if (g_has_batch_insert_active_layers) {
        for (const auto& e : edges) g_batch_insert_active_layers[std::get<2>(e)] = 1;
    }

    has_lower_bounds = false;
    lower_bounds_snapshot.clear();
    lower_bounds_active.clear();

    std::vector<EdgeTuple> edge_tuples;
    edge_tuples.reserve(edges.size());
    for (const auto& e : edges) {
        edge_tuples.emplace_back(std::get<0>(e), std::get<1>(e), std::get<2>(e));
    }

    bool use_grouped_insert = !edge_tuples.empty();
    if (use_grouped_insert) {
        cerr << "[BATCH-INSERT-GROUP] " << edge_tuples.size()
             << " edges -> layer-ordered hybrid group loop" << endl;

        unsigned long long total_groups = 0;
        unsigned long long total_components = 0;
        unsigned long long total_affected = 0;
        unsigned long long total_queued = 0;
        double grouped_build_sum = 0.0;
        double grouped_bfs_sum = 0.0;
        double grouped_iter_sum = 0.0;

        if (lower_bounds_snapshot.size() != n) lower_bounds_snapshot.resize(n);
        if (lower_bounds_active.size() != n) lower_bounds_active.assign(n, 0);

        auto apply_insert_group = [&](const std::vector<EdgeTuple>& group_edges,
                                      double group_build_ms) {
            has_lower_bounds = false;
            double snapshot_ms = 0.0;
            double total_group_build_ms = group_build_ms + snapshot_ms;

            g_batch_insert_active_layers.fill(0);
            g_has_batch_insert_active_layers = (L <= kMaxTrackedBatchLayers);
            if (g_has_batch_insert_active_layers) {
                for (const auto& e : group_edges) g_batch_insert_active_layers[e.l] = 1;
            }

            auto group_start = std::chrono::high_resolution_clock::now();
            if (group_edges.size() == 1) {
                const auto& e = group_edges[0];
	                AddAllLayerNeighborCacheEdge(e.u, e.v);
	                AddEdgeToGraph(mg, e.u, e.v, e.l);
	            } else {
	                for (const auto& e : group_edges) {
	                    AddAllLayerNeighborCacheEdge(e.u, e.v);
	                }
	                BatchAddEdgesToGraph(mg, L, group_edges);
	            }

            std::vector<char> needs_check(n, 0);
            std::vector<char> affected_seen(n, 0);
            std::vector<uint> affected_nodes;
            std::vector<std::pair<uint, std::vector<uint>>> candidate_records;
            std::vector<std::pair<uint, std::vector<uint>>> initial_affected_scvs;
            std::vector<char> group_needs(n, 0);
	            BatchComputeCappedUpperBound(group_edges, scv_sets, group_needs,
	                                         true, nullptr, &scv_sets, nullptr,
	                                         nullptr, nullptr, &candidate_records);
	            CompactCandidateRecordsByNode(candidate_records, L);

	            std::vector<uint> lower_bound_nodes;
            lower_bound_nodes.reserve(candidate_records.size());
            auto snapshot_update_start = std::chrono::high_resolution_clock::now();
            uint last_snapshot_node = UINT_MAX;
            for (const auto& record : candidate_records) {
                uint node = record.first;
                if (node >= n || node == last_snapshot_node) continue;
                last_snapshot_node = node;
                if (!lower_bounds_active[node]) {
                    scv_rw_locks[node]->lock_shared();
                    lower_bounds_snapshot[node] = scv_sets[node];
                    scv_rw_locks[node]->unlock_shared();
                    lower_bounds_active[node] = 1;
                    lower_bound_nodes.push_back(node);
                }
            }
            auto snapshot_update_end = std::chrono::high_resolution_clock::now();
            double snapshot_update_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                snapshot_update_end - snapshot_update_start).count() / 1000.0;
            snapshot_ms += snapshot_update_ms;
            total_group_build_ms += snapshot_update_ms;

            auto candidate_commit_start = std::chrono::high_resolution_clock::now();
            for (auto& record : candidate_records) {
                uint node = record.first;
                if (node >= n) continue;
                scv_rw_locks[node]->lock();
                bool inserted = scv_sets[node].Insert(record.second);
                if (inserted) {
                    initial_affected_scvs.emplace_back(node, record.second);
                    UpdateMaxC(node);
                    needs_check[node] = 1;
                    if (!affected_seen[node]) {
                        affected_seen[node] = 1;
                        affected_nodes.push_back(node);
                    }
                }
                scv_rw_locks[node]->unlock();
            }
            auto candidate_commit_end = std::chrono::high_resolution_clock::now();
            double candidate_commit_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                candidate_commit_end - candidate_commit_start).count() / 1000.0;

            std::vector<uint> ordered_nodes;
            ordered_nodes.reserve(affected_nodes.size());
            auto order_filter_start = std::chrono::high_resolution_clock::now();
            for (uint node : affected_nodes) {
                if (node < n &&
                    needs_check[node] &&
                    node < lower_bounds_active.size() &&
                    lower_bounds_active[node] &&
                    !scv_sets[node].IsEqual(lower_bounds_snapshot[node])) {
                    ordered_nodes.push_back(node);
                }
            }
            auto order_filter_end = std::chrono::high_resolution_clock::now();
            double order_filter_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                order_filter_end - order_filter_start).count() / 1000.0;
            unsigned long long affected_scv_count = 0;
            bool use_affected_scv_filter = false;
            for (uint node : ordered_nodes) {
                affected_scv_count += scv_sets[node].NumSCVs();
            }
            if (!ordered_nodes.empty() &&
                affected_scv_count >=
                    kAffectedSCVFilterMinAverage * ordered_nodes.size()) {
                use_affected_scv_filter = true;
            }

            uint fixed_dim = group_edges.empty() ? UINT_MAX : group_edges[0].l;
            bool same_insert_layer = fixed_dim < L;
            for (const auto& e : group_edges) {
                if (e.l != fixed_dim) {
                    same_insert_layer = false;
                    break;
                }
            }
            double hindex_cap_cache_ms = 0.0;
            bool use_hindex_cap_cache =
                ordered_nodes.size() > 64 &&
                !same_insert_layer;
            if (use_hindex_cap_cache) {
                auto cache_start = std::chrono::high_resolution_clock::now();
                BuildNeighborHIndexCapCache(ordered_nodes);
                auto cache_end = std::chrono::high_resolution_clock::now();
                hindex_cap_cache_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    cache_end - cache_start).count() / 1000.0;
            } else {
                ClearNeighborHIndexCapCache();
            }

            long long pushed_count = 0;
            auto enqueue_start = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(shared_queue_mtx);
                for (uint node : ordered_nodes) {
                    bool expected = false;
                    if (is_in_inbox[node].compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                        shared_queue.push_back(node);
                        pushed_count++;
                    }
                }
                active_task_count.fetch_add(pushed_count, std::memory_order_relaxed);
            }
            auto enqueue_end = std::chrono::high_resolution_clock::now();
            double enqueue_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                enqueue_end - enqueue_start).count() / 1000.0;

            auto group_scope_end = std::chrono::high_resolution_clock::now();
            double group_bfs_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                group_scope_end - group_start).count() / 1000.0;
            group_bfs_ms = std::max(0.0, group_bfs_ms - hindex_cap_cache_ms);

            if (pushed_count > 0) {
                hot_start_wake_domain.assign(n, 0);
                for (uint node : ordered_nodes) hot_start_wake_domain[node] = 1;
                limit_hot_start_wake_to_domain = true;
                has_lower_bounds = true;
                ClearHotStartSCVFilter();
                if (use_affected_scv_filter) {
                    SeedHotStartSCVFilter(initial_affected_scvs);
                }
                if (same_insert_layer) EnableFixedRefinementDim(fixed_dim);
                Decompose(true);
                if (same_insert_layer) DisableFixedRefinementDim();
                ClearHotStartSCVFilter();
                limit_hot_start_wake_to_domain = false;
                hot_start_wake_domain.clear();
                grouped_iter_sum += last_iter_ms;
            }
            ClearNeighborHIndexCapCache();
            has_lower_bounds = false;

            auto snapshot_clear_start = std::chrono::high_resolution_clock::now();
            for (uint node : lower_bound_nodes) {
                if (node < lower_bounds_active.size()) lower_bounds_active[node] = 0;
            }
            auto snapshot_clear_end = std::chrono::high_resolution_clock::now();
            double snapshot_clear_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                snapshot_clear_end - snapshot_clear_start).count() / 1000.0;
            snapshot_ms += snapshot_clear_ms;
            total_group_build_ms += snapshot_clear_ms;

            grouped_build_sum += total_group_build_ms;
            grouped_bfs_sum += group_bfs_ms;
            total_affected += ordered_nodes.size();
            total_queued += pushed_count;

            cerr << "[BATCH-INSERT-GROUP] group=" << total_groups
                 << " layer=" << group_edges[0].l
                 << " edges=" << group_edges.size()
                 << " candidates=" << candidate_records.size()
                 << " affected=" << ordered_nodes.size()
                 << " queued=" << pushed_count
	                 << " build=" << total_group_build_ms
	                 << " snapshot=" << snapshot_ms
	                 << " candidate_commit=" << candidate_commit_ms;
            cerr << " order_filter=" << order_filter_ms
	                 << " enqueue=" << enqueue_ms
	                 << " bfs=" << group_bfs_ms
	                 << " iter=" << (pushed_count > 0 ? last_iter_ms : 0.0) << "ms" << endl;
        };

        auto insert_node_layer_minmax = [&](uint node, uint layer) {
            uint count = scv_sets[node].NumSCVs();
            uint mn = UINT_MAX;
            uint mx = 0;
            for (uint si = 0; si < count; si++) {
                const uint* raw = scv_sets[node].GetSCVFlat(si);
                mn = std::min(mn, raw[layer]);
                mx = std::max(mx, raw[layer]);
            }
            if (mn == UINT_MAX) mn = 0;
            return std::pair<uint, uint>(mn, mx);
        };

        auto insert_hybrid_group_valid = [&](const std::vector<EdgeTuple>& group_edges) {
            if (group_edges.empty()) return true;
            uint layer = group_edges[0].l;
            std::unordered_map<uint, uint> degree;
            degree.reserve(group_edges.size() * 2 + 1);
            for (const auto& edge : group_edges) {
                if (edge.l != layer) return false;
                degree[edge.u]++;
                degree[edge.v]++;
            }
            for (const auto& edge : group_edges) {
                bool u_center = degree[edge.u] > 1;
                bool v_center = degree[edge.v] > 1;
                if (u_center && v_center) return false;
            }
            for (const auto& item : degree) {
                uint center = item.first;
                if (item.second <= 1) continue;
                uint center_min = insert_node_layer_minmax(center, layer).first;
                for (const auto& edge : group_edges) {
                    uint leaf = UINT_MAX;
                    if (edge.u == center) leaf = edge.v;
                    else if (edge.v == center) leaf = edge.u;
                    else continue;
                    if (degree[leaf] != 1) return false;
                    uint leaf_max = insert_node_layer_minmax(leaf, layer).second;
                    if (center_min <= leaf_max) return false;
                }
            }
            return true;
        };

        std::vector<std::vector<EdgeTuple>> by_layer(L);
        for (const EdgeTuple& edge : edge_tuples) {
            by_layer[edge.l].push_back(edge);
        }

        for (uint layer = 0; layer < L; layer++) {
            std::vector<EdgeTuple> remaining = by_layer[layer];
            if (remaining.empty()) continue;
            total_components++;

            while (!remaining.empty()) {
                auto group_build_start = std::chrono::high_resolution_clock::now();
                std::vector<EdgeTuple> group_edges;
                std::vector<char> picked(remaining.size(), 0);
                group_edges.reserve(remaining.size());
                for (size_t i = 0; i < remaining.size(); i++) {
                    std::vector<EdgeTuple> trial = group_edges;
                    trial.push_back(remaining[i]);
                    if (!insert_hybrid_group_valid(trial)) continue;
                    group_edges.push_back(remaining[i]);
                    picked[i] = 1;
                }
                if (group_edges.empty()) {
                    group_edges.push_back(remaining[0]);
                    picked[0] = 1;
                }
                total_groups++;

                std::vector<EdgeTuple> next_remaining;
                next_remaining.reserve(remaining.size() - group_edges.size());
                for (size_t i = 0; i < remaining.size(); i++) {
                    if (!picked[i]) next_remaining.push_back(remaining[i]);
                }
                auto group_build_end = std::chrono::high_resolution_clock::now();
                double group_build_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    group_build_end - group_build_start).count() / 1000.0;
                apply_insert_group(group_edges, group_build_ms);
                remaining.swap(next_remaining);
            }
        }

        enable_lazy_support_seed = false;
        g_has_batch_insert_active_layers = false;
        has_lower_bounds = false;
        lower_bounds_snapshot.clear();
        lower_bounds_active.clear();
        last_bfs_ms = grouped_build_sum + grouped_bfs_sum;
        last_iter_ms = grouped_iter_sum;
        cerr << "[BATCH-INSERT-GROUP] total_groups=" << total_groups
             << " components=" << total_components
             << " affected=" << total_affected
             << " queued=" << total_queued
             << " total_build=" << grouped_build_sum
             << " total_bfs=" << grouped_bfs_sum
             << " total_bfs_with_build=" << last_bfs_ms
             << " total_iter=" << last_iter_ms << "ms" << endl;
        return;
    }
}

void DynamicMLCore::RemoveEdgeFromGraph(MultilayerGraph& mg, uint u, uint v, uint l) {
    uint** adj = mg.GetGraph(l).GetAdjLst();

    auto remove_neighbor = [](uint** adj, uint node, uint neighbor) {
        uint deg = adj[node][0];
        bool found = false;
        for (uint i = 1; i <= deg; i++) {
            if (adj[node][i] == neighbor) { found = true; break; }
        }
        if (!found) return;  
        uint* new_adj = new uint[deg];
        new_adj[0] = deg - 1;
        uint idx = 1;
        for (uint i = 1; i <= deg; i++) {
            if (adj[node][i] != neighbor) {
                new_adj[idx++] = adj[node][i];
            }
        }
        adj[node] = new_adj;
    };

    remove_neighbor(adj, u, v);
    remove_neighbor(adj, v, u);
}

void DynamicMLCore::BatchDeleteEdges(const std::vector<std::tuple<uint, uint, uint>>& edges) {
    if (edges.empty()) return;

    g_batch_delete_active_layers.fill(0);
    g_has_batch_delete_active_layers = (L <= kMaxTrackedBatchLayers);
    if (g_has_batch_delete_active_layers) {
        for (const auto& e : edges) g_batch_delete_active_layers[std::get<2>(e)] = 1;
    }
    std::vector<EdgeTuple> edge_tuples;
    edge_tuples.reserve(edges.size());
    for (const auto& e : edges) {
        edge_tuples.emplace_back(std::get<0>(e), std::get<1>(e), std::get<2>(e));
    }

    if (!edge_tuples.empty()) {
        cerr << "[BATCH-DELETE-GROUP] " << edge_tuples.size()
             << " edges -> layer-ordered hybrid group loop" << endl;

        auto delete_node_layer_minmax = [&](uint node, uint layer) {
            uint count = scv_sets[node].NumSCVs();
            uint mn = UINT_MAX;
            uint mx = 0;
            for (uint si = 0; si < count; si++) {
                const uint* raw = scv_sets[node].GetSCVFlat(si);
                mn = std::min(mn, raw[layer]);
                mx = std::max(mx, raw[layer]);
            }
            if (mn == UINT_MAX) mn = 0;
            return std::pair<uint, uint>(mn, mx);
        };
        auto delete_hybrid_group_valid = [&](const std::vector<EdgeTuple>& group_edges) {
            if (group_edges.empty()) return true;
            uint layer = group_edges[0].l;
            std::unordered_map<uint, uint> degree;
            degree.reserve(group_edges.size() * 2 + 1);
            for (const auto& edge : group_edges) {
                if (edge.l != layer) return false;
                degree[edge.u]++;
                degree[edge.v]++;
            }
            for (const auto& edge : group_edges) {
                bool u_center = degree[edge.u] > 1;
                bool v_center = degree[edge.v] > 1;
                if (u_center && v_center) return false;
            }
            for (const auto& item : degree) {
                uint center = item.first;
                if (item.second <= 1) continue;
                uint center_min = delete_node_layer_minmax(center, layer).first;
                for (const auto& edge : group_edges) {
                    uint leaf = UINT_MAX;
                    if (edge.u == center) leaf = edge.v;
                    else if (edge.v == center) leaf = edge.u;
                    else continue;
                    if (degree[leaf] != 1) return false;
                    uint leaf_max = delete_node_layer_minmax(leaf, layer).second;
                    if (center_min <= leaf_max) return false;
                }
            }
            return true;
        };

        unsigned long long total_groups = 0;
        unsigned long long total_components = 0;
        unsigned long long total_affected = 0;
        unsigned long long total_queued = 0;
        double grouped_build_sum = 0.0;
        double grouped_bfs_sum = 0.0;
        double grouped_iter_sum = 0.0;

        std::vector<std::vector<EdgeTuple>> by_layer(L);
        for (const EdgeTuple& edge : edge_tuples) {
            by_layer[edge.l].push_back(edge);
        }

        for (uint layer = 0; layer < L; layer++) {
            std::vector<EdgeTuple> remaining = by_layer[layer];
            if (remaining.empty()) continue;
            total_components++;

            while (!remaining.empty()) {
                auto group_build_start = std::chrono::high_resolution_clock::now();
                std::vector<EdgeTuple> group_edges;
                std::vector<char> picked(remaining.size(), 0);
                group_edges.reserve(remaining.size());
                for (size_t i = 0; i < remaining.size(); i++) {
                    std::vector<EdgeTuple> trial = group_edges;
                    trial.push_back(remaining[i]);
                    if (!delete_hybrid_group_valid(trial)) continue;
                    group_edges.push_back(remaining[i]);
                    picked[i] = 1;
                }
                if (group_edges.empty()) {
                    group_edges.push_back(remaining[0]);
                    picked[0] = 1;
                }
                std::vector<EdgeTuple> next_remaining;
                next_remaining.reserve(remaining.size() - group_edges.size());
                for (size_t i = 0; i < remaining.size(); i++) {
                    if (!picked[i]) next_remaining.push_back(remaining[i]);
                }
                auto group_build_end = std::chrono::high_resolution_clock::now();
                double group_build_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    group_build_end - group_build_start).count() / 1000.0;
                grouped_build_sum += group_build_ms;

                total_groups++;
                g_batch_delete_active_layers.fill(0);
                g_has_batch_delete_active_layers = (L <= kMaxTrackedBatchLayers);
                if (g_has_batch_delete_active_layers) {
                    g_batch_delete_active_layers[layer] = 1;
                }

                auto group_start = std::chrono::high_resolution_clock::now();
                std::vector<std::tuple<uint, uint, uint>> group_tuples;
                group_tuples.reserve(group_edges.size());
                for (const auto& e : group_edges) {
                    RemoveEdgeFromGraph(mg, e.u, e.v, e.l);
                    RemoveAllLayerNeighborCacheEdgeIfDisconnected(e.u, e.v);
                    group_tuples.emplace_back(e.u, e.v, e.l);
                }

                std::vector<char> needs_check(n, 0);
                std::vector<char> lc_domain;
                std::vector<std::pair<uint, std::vector<uint>>> affected_scvs;
                DeletePropagationOnly(group_tuples, needs_check, &lc_domain,
                                      &affected_scvs);
                if (lc_domain.empty()) lc_domain = needs_check;

                std::vector<uint> ordered_nodes;
                ordered_nodes.reserve(n);
                for (uint node = 0; node < n; node++) {
                    if (lc_domain[node]) ordered_nodes.push_back(node);
                }
                unsigned long long affected_scv_count = 0;
                bool use_affected_scv_filter = false;
                for (uint node : ordered_nodes) {
                    affected_scv_count += scv_sets[node].NumSCVs();
                }
                if (!ordered_nodes.empty() &&
                    affected_scv_count >=
                        kAffectedSCVFilterMinAverage * ordered_nodes.size()) {
                    use_affected_scv_filter = true;
                }

                long long pushed = 0;
                {
                    std::lock_guard<std::mutex> lock(shared_queue_mtx);
                    for (uint node : ordered_nodes) {
                        bool expected = false;
                        if (is_in_inbox[node].compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                            shared_queue.push_back(node);
                            pushed++;
                        }
                    }
                    active_task_count.fetch_add(pushed, std::memory_order_relaxed);
                }

                double group_bfs_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - group_start).count() / 1000.0;

                if (pushed > 0) {
                    debug_delete_bfs_affected = lc_domain;
                    hot_start_wake_domain = lc_domain;
                    limit_delete_wake_to_bfs = true;
                    limit_hot_start_wake_to_domain = true;
                    has_lower_bounds = false;
                    ClearHotStartSCVFilter();
                    if (use_affected_scv_filter) {
                        SeedHotStartSCVFilter(affected_scvs);
                    }
                    RecoverTrueSCV();
                    ClearHotStartSCVFilter();
                    grouped_iter_sum += last_iter_ms;
                }

                delete_wake_use_reverse_support = false;
                debug_delete_wake_enabled = false;
                limit_delete_wake_to_bfs = false;
                limit_hot_start_wake_to_domain = false;
                hot_start_wake_domain.clear();
                debug_delete_bfs_affected.clear();
                debug_delete_bfs_failed_sets.clear();

                grouped_bfs_sum += group_bfs_ms;
                total_affected += ordered_nodes.size();
                total_queued += pushed;

                cerr << "[BATCH-DELETE-GROUP] group=" << total_groups
                     << " layer=" << layer
                     << " edges=" << group_edges.size()
                     << " affected=" << ordered_nodes.size()
                     << " queued=" << pushed
                     << " build=" << group_build_ms
                     << " bfs=" << group_bfs_ms
                     << " iter=" << (pushed > 0 ? last_iter_ms : 0.0) << "ms" << endl;
                remaining.swap(next_remaining);
            }
        }

        last_bfs_ms = grouped_build_sum + grouped_bfs_sum;
        last_iter_ms = grouped_iter_sum;
        cerr << "[BATCH-DELETE] " << edges.size() << " edges, "
             << total_queued << " affected / " << n << " total" << endl;
        cerr << "[BATCH-DELETE] build=" << grouped_build_sum
             << " BFS+delete=" << grouped_bfs_sum
             << " BFS+delete+build=" << last_bfs_ms << "ms" << endl;
        cerr << "[BATCH-DELETE-GROUP] total_groups=" << total_groups
             << " components=" << total_components
             << " affected=" << total_affected
             << " queued=" << total_queued
             << " total_build=" << grouped_build_sum
             << " total_bfs=" << grouped_bfs_sum
             << " total_bfs_with_build=" << last_bfs_ms
             << " total_iter=" << last_iter_ms << "ms" << endl;
        g_has_batch_delete_active_layers = false;
        return;
    }
}

bool DynamicMLCore::IsSCVFeasible(uint v, const std::vector<uint>& k_vec) const {

    return CheckSupport(v, k_vec);
}

std::vector<uint> DynamicMLCore::TryBinarySplit(
    uint v,
    const std::vector<uint>& k_vec,
    const NeighborMaxCBuffer& buf) const {

    thread_local std::vector<uint> h_vec;
    h_vec.resize(L);
    bool h_pruned = false;
    ComputeHFromBuffer(k_vec, buf, h_vec, h_pruned);
    if (h_pruned) h_pruned = ApplyFixedRefinementDim(k_vec, h_vec);
    if (!h_pruned) return {};

    bool all_zero = true;
    for (uint d = 0; d < L; d++) {
        if (h_vec[d] > 0) { all_zero = false; break; }
    }
    if (all_zero) return {};

    if (IsSCVFeasible(v, h_vec)) return h_vec;
    return {};
}
