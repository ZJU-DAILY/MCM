#ifndef SKYLINE_SET_H
#define SKYLINE_SET_H

#include <vector>
#include <smmintrin.h>

typedef unsigned int uint;

class SkylineSet {
public:
    SkylineSet() : L_(0) {}
    ~SkylineSet() = default;

    bool Insert(const std::vector<uint>& k_vec);
    bool InsertKnownUndominated(const std::vector<uint>& k_vec);
    bool InsertRaw(const uint* k_vec, uint L);
    bool InsertKnownUndominatedRaw(const uint* k_vec, uint L);
    bool Remove(const std::vector<uint>& k_vec);
    bool IsDominated(const std::vector<uint>& k_vec) const;
    bool IsDominatedRaw(const uint* k_vec, uint L) const;
    bool Contains(const std::vector<uint>& k_vec) const;
    bool ContainsRaw(const uint* k_vec, uint L) const;

    std::vector<std::vector<uint>> GetSCVs() const;
    void Clear();
    bool IsEqual(const SkylineSet& other) const;

    
    uint NumSCVs() const { return L_ == 0 ? 0 : scvs_.size() / L_; }
    uint GetL() const { return L_; }
    const uint* GetSCVFlat(uint idx) const { return scvs_.data() + idx * L_; }
    size_t CapacityValues() const { return scvs_.capacity(); }

    
    static bool DominatesRaw(const uint* k1, const uint* k2, uint L) {
        for (uint i = 0; i < L; i++) {
            if (k1[i] < k2[i]) return false;
        }
        return true;
    }

    static bool Dominates(const std::vector<uint>& k1, const std::vector<uint>& k2);

private:
    std::vector<uint> scvs_;  
    uint L_;                  

    void InitL(uint L) {
        if (L_ == 0) L_ = L;
    }
};

#endif 
