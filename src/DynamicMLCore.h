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


struct PropagationEntry {
    static constexpr uint kInlineLayers = 8;
    uint node;                      
    uint l_grow;                    
    std::array<uint, kInlineLayers> k_new{}; 
    std::vector<uint> k_overflow;    
    PropagationEntry() = default;
    PropagationEntry(uint n, const std::vector<uint>& k, uint l) : node(n), l_grow(l) {
        if (k.size() <= kInlineLayers) {
            for (uint i = 0; i < k.size(); i++) k_new[i] = k[i];
        } else {
            k_overflow = k;
        }
    }
    const uint* K() const { return k_overflow.empty() ? k_new.data() : k_overflow.data(); }
};


class DynamicMLCore : public IterativeMLCore {
public:
    
    explicit DynamicMLCore(MultilayerGraph& mg_, int num_threads_ = 0);
    virtual ~DynamicMLCore() = default;

    
    
    

    
    virtual void BatchInsertEdges(const std::vector<std::tuple<uint, uint, uint>>& edges);

    
    virtual void BatchDeleteEdges(const std::vector<std::tuple<uint, uint, uint>>& edges);

    
    
    void RestoreSnapshotForIndependentRun(const std::vector<SkylineSet>& snapshot);
    bool SavePreparedSnapshotForIndependentRun(const std::string& filename) const;
    bool RestorePreparedSnapshotForIndependentRun(const std::vector<SkylineSet>& snapshot,
                                                  const std::string& filename);
    
    static void AddEdgeToGraph(MultilayerGraph& mg, uint u, uint v, uint l);

    
    static void RemoveEdgeFromGraph(MultilayerGraph& mg, uint u, uint v, uint l);

public:
    double last_bfs_ms = 0;          

protected:
    
    
    
    
    
    bool IsSCVFeasible(uint v, const std::vector<uint>& k_vec) const;

    
    std::vector<uint> TryBinarySplit(uint v, const std::vector<uint>& k_vec,
                                     const NeighborMaxCBuffer& buf) const override;

    virtual void RecoverTrueSCV();


    
    
    

    
    static void BatchAddEdgesToGraph(MultilayerGraph& mg, uint num_layers,
                                     const std::vector<EdgeTuple>& edges);

    
    
    
    
    void BatchComputeCappedUpperBound(
        const std::vector<EdgeTuple>& batch_edges,
        const std::vector<SkylineSet>& old_scv_sets,
        std::vector<char>& needs_check,
        bool force_affected_scope = false,
        const std::vector<char>* commit_domain = nullptr,
        const std::vector<SkylineSet>* commit_dominance_sets = nullptr,
        std::vector<SkylineSet>* out_candidate_sets = nullptr,
        std::vector<uint>* out_candidate_nodes = nullptr,
        std::vector<SkylineSet>* out_inserted_scv_sets = nullptr,
        std::vector<std::pair<uint, std::vector<uint>>>* out_candidate_records = nullptr);

    
    
    
    
    
    
    void ParallelBFSCascadePendingCapped(std::vector<PropagationEntry>& F_curr,
                                         const std::vector<SkylineSet>& old_scv_sets,
                                         std::vector<SkylineSet>& pending_sets,
                                         std::vector<char>& pending_touched,
                                         std::vector<uint>& pending_nodes);

    bool TryInsertPendingCandidate(uint node,
                                   const std::vector<uint>& k_cand,
                                   const std::vector<SkylineSet>& old_scv_sets,
                                   std::vector<SkylineSet>& pending_sets,
                                   std::vector<char>& pending_touched,
                                   std::vector<uint>& pending_nodes);

    
    
    
    

    
    
    
    
    
    
    
    void DeletePropagationOnly(
        const std::vector<std::tuple<uint, uint, uint>>& deleted_edges,
        std::vector<char>& needs_check,
        std::vector<char>* affected_domain = nullptr,
        std::vector<std::pair<uint, std::vector<uint>>>* affected_scvs = nullptr);

    
    
    
    
private:
    inline bool Dominates(const std::vector<uint>& k1, const std::vector<uint>& k2) const {
        const size_t size = k1.size();

        size_t i = 0;
        const __m128i all_ones = _mm_set1_epi32(-1); 

        for (; i + 4 <= size; i += 4) {
            
            __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&k1[i]));
            __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&k2[i]));

            
            
            __m128i max_v = _mm_max_epu32(v1, v2);

            
            __m128i eq_res = _mm_cmpeq_epi32(max_v, v1);

            
            
            if (_mm_testc_si128(eq_res, all_ones) == 0) {
                return false;
            }
        }

        
        for (; i < size; ++i) {
            if (k1[i] < k2[i]) return false;
        }

        return true;
    }
};

#endif 
