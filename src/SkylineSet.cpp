#include "SkylineSet.h"
#include <algorithm>
#include <numeric>

bool SkylineSet::Dominates(const std::vector<uint>& k1, const std::vector<uint>& k2) {
    return DominatesRaw(k1.data(), k2.data(), k1.size());
}

bool SkylineSet::IsDominated(const std::vector<uint>& k_vec) const {
    return IsDominatedRaw(k_vec.data(), k_vec.size());
}

bool SkylineSet::IsDominatedRaw(const uint* k, uint L) const {
    if (L_ == 0 || scvs_.empty()) return false;
    if (L != L_) return false;

    const uint* data = scvs_.data();
    uint n = scvs_.size() / L_;

    if (L_ == 2) {
        uint lo = 0, hi = n;
        while (lo < hi) {
            uint mid = lo + (hi - lo) / 2;
            if (data[mid * 2] < k[0]) lo = mid + 1;
            else hi = mid;
        }
        return lo < n && data[lo * 2 + 1] >= k[1];
    } else if (L_ == 3) {
        uint lo = 0, hi = n;
        while (lo < hi) {
            uint mid = lo + (hi - lo) / 2;
            if (data[mid * 3] < k[0]) lo = mid + 1;
            else hi = mid;
        }
        for (uint i = lo; i < n; i++) {
            const uint* s = data + i * 3;
            if (s[0] >= k[0] && s[1] >= k[1] && s[2] >= k[2]) return true;
        }
    } else if (L_ == 4) {
        const __m128i all_ones = _mm_set1_epi32(-1);
        for (uint i = 0; i < n; i++) {
            const uint* s = data + i * 4;
            __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
            __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k));
            __m128i max_v = _mm_max_epu32(v1, v2);
            __m128i eq = _mm_cmpeq_epi32(max_v, v1);
            if (_mm_testc_si128(eq, all_ones)) return true;
        }
    } else if (L_ == 7) {
        const __m128i all_ones = _mm_set1_epi32(-1);
        __m128i k0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k));
        uint k4 = k[4], k5 = k[5], k6 = k[6];
        for (uint i = 0; i < n; i++) {
            const uint* s = data + i * 7;
            __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
            __m128i max_v = _mm_max_epu32(v0, k0);
            __m128i eq = _mm_cmpeq_epi32(max_v, v0);
            if (!_mm_testc_si128(eq, all_ones)) continue;
            if (s[4] >= k4 && s[5] >= k5 && s[6] >= k6) return true;
        }
    } else {
        
        const __m128i all_ones = _mm_set1_epi32(-1);
        for (uint i = 0; i < n; i++) {
            const uint* s = data + i * L_;
            bool dom = true;
            uint j = 0;
            for (; j + 4 <= L_; j += 4) {
                __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + j));
                __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k + j));
                __m128i max_v = _mm_max_epu32(v1, v2);
                __m128i eq = _mm_cmpeq_epi32(max_v, v1);
                if (_mm_testc_si128(eq, all_ones) == 0) { dom = false; break; }
            }
            if (dom) {
                for (; j < L_; j++) {
                    if (s[j] < k[j]) { dom = false; break; }
                }
            }
            if (dom) return true;
        }
    }
    return false;
}

bool SkylineSet::Insert(const std::vector<uint>& k_vec) {
    if (IsDominated(k_vec)) return false;
    return InsertKnownUndominated(k_vec);
}

bool SkylineSet::InsertRaw(const uint* k_vec, uint L) {
    if (IsDominatedRaw(k_vec, L)) return false;
    return InsertKnownUndominatedRaw(k_vec, L);
}

bool SkylineSet::InsertKnownUndominated(const std::vector<uint>& k_vec) {
    return InsertKnownUndominatedRaw(k_vec.data(), k_vec.size());
}

bool SkylineSet::InsertKnownUndominatedRaw(const uint* k_vec, uint L) {
    InitL(L);
    if (L_ != L) return false;

    uint n = NumSCVs();

    if (L_ == 2) {
        uint write = 0;
        for (uint i = 0; i < n; i++) {
            const uint* s = scvs_.data() + i * 2;
            uint s0 = s[0];
            uint s1 = s[1];
            if (s0 <= k_vec[0] && s1 <= k_vec[1]) continue;

            if (write != i) {
                uint* dst = scvs_.data() + write * 2;
                dst[0] = s0;
                dst[1] = s1;
            }
            write++;
        }

        uint pos = 0;
        while (pos < write && scvs_[pos * 2] < k_vec[0]) pos++;

        scvs_.resize((write + 1) * 2);
        for (uint i = write; i > pos; i--) {
            scvs_[i * 2] = scvs_[(i - 1) * 2];
            scvs_[i * 2 + 1] = scvs_[(i - 1) * 2 + 1];
        }
        scvs_[pos * 2] = k_vec[0];
        scvs_[pos * 2 + 1] = k_vec[1];
        return true;
    }

    if (L_ == 3) {
        uint write = 0;
        for (uint i = 0; i < n; i++) {
            const uint* s = scvs_.data() + i * 3;
            uint s0 = s[0];
            uint s1 = s[1];
            uint s2 = s[2];
            if (s0 <= k_vec[0] && s1 <= k_vec[1] && s2 <= k_vec[2]) continue;

            if (write != i) {
                uint* dst = scvs_.data() + write * 3;
                dst[0] = s0;
                dst[1] = s1;
                dst[2] = s2;
            }
            write++;
        }

        uint pos = 0;
        while (pos < write) {
            const uint* s = scvs_.data() + pos * 3;
            if (s[0] > k_vec[0]) break;
            if (s[0] == k_vec[0] && s[1] > k_vec[1]) break;
            if (s[0] == k_vec[0] && s[1] == k_vec[1] && s[2] > k_vec[2]) break;
            pos++;
        }

        scvs_.resize((write + 1) * 3);
        for (uint i = write; i > pos; i--) {
            scvs_[i * 3] = scvs_[(i - 1) * 3];
            scvs_[i * 3 + 1] = scvs_[(i - 1) * 3 + 1];
            scvs_[i * 3 + 2] = scvs_[(i - 1) * 3 + 2];
        }
        scvs_[pos * 3] = k_vec[0];
        scvs_[pos * 3 + 1] = k_vec[1];
        scvs_[pos * 3 + 2] = k_vec[2];
        return true;
    }

    uint write = 0;
    for (uint i = 0; i < n; i++) {
        const uint* s = scvs_.data() + i * L_;
        if (!DominatesRaw(k_vec, s, L_)) {
            if (write != i) {
                uint* dst = scvs_.data() + write * L_;
                std::copy(s, s + L_, dst);
            }
            write++;
        }
    }

    scvs_.resize(write * L_);
    scvs_.insert(scvs_.end(), k_vec, k_vec + L_);
    return true;
}

bool SkylineSet::Contains(const std::vector<uint>& k_vec) const {
    return ContainsRaw(k_vec.data(), k_vec.size());
}

bool SkylineSet::ContainsRaw(const uint* k, uint L) const {
    if (L_ == 0 || L != L_) return false;
    uint n = NumSCVs();
    if (L_ == 2) {
        uint lo = 0, hi = n;
        while (lo < hi) {
            uint mid = lo + (hi - lo) / 2;
            uint mid0 = scvs_[mid * 2];
            if (mid0 < k[0]) lo = mid + 1;
            else hi = mid;
        }
        return lo < n && scvs_[lo * 2] == k[0] && scvs_[lo * 2 + 1] == k[1];
    }
    if (L_ == 3) {
        uint lo = 0, hi = n;
        while (lo < hi) {
            uint mid = lo + (hi - lo) / 2;
            uint mid0 = scvs_[mid * 3];
            if (mid0 < k[0]) lo = mid + 1;
            else hi = mid;
        }
        for (uint i = lo; i < n && scvs_[i * 3] == k[0]; i++) {
            const uint* s = scvs_.data() + i * 3;
            if (s[1] == k[1] && s[2] == k[2]) return true;
        }
        return false;
    }
    for (uint i = 0; i < n; i++) {
        const uint* s = scvs_.data() + i * L_;
        bool equal = true;
        for (uint j = 0; j < L_; j++) {
            if (s[j] != k[j]) { equal = false; break; }
        }
        if (equal) return true;
    }
    return false;
}

bool SkylineSet::Remove(const std::vector<uint>& k_vec) {
    if (L_ == 0) return false;
    uint n = NumSCVs();
    const uint* k = k_vec.data();
    const uint* data = scvs_.data();

    for (uint i = 0; i < n; i++) {
        const uint* s = data + i * L_;
        bool equal = true;
        for (uint j = 0; j < L_; j++) {
            if (s[j] != k[j]) { equal = false; break; }
        }
        if (equal) {
            if (L_ == 2 || L_ == 3) {
                scvs_.erase(scvs_.begin() + i * L_, scvs_.begin() + (i + 1) * L_);
            } else {
                
                if (i < n - 1) {
                    uint* dst = scvs_.data() + i * L_;
                    const uint* src = scvs_.data() + (n - 1) * L_;
                    std::copy(src, src + L_, dst);
                }
                scvs_.resize(scvs_.size() - L_);
            }
            return true;
        }
    }
    return false;
}

std::vector<std::vector<uint>> SkylineSet::GetSCVs() const {
    std::vector<std::vector<uint>> result;
    uint n = NumSCVs();
    result.reserve(n);
    for (uint i = 0; i < n; i++) {
        const uint* s = scvs_.data() + i * L_;
        result.emplace_back(s, s + L_);
    }
    return result;
}

void SkylineSet::Clear() {
    scvs_.clear();
    L_ = 0;
}

bool SkylineSet::IsEqual(const SkylineSet& other) const {
    if (NumSCVs() != other.NumSCVs()) return false;
    if (L_ != other.L_) return L_ == 0 && other.L_ == 0;
    if (L_ == 0) return true;

    
    if (L_ == 2) {
        if (scvs_.size() != other.scvs_.size()) return false;
        return std::equal(scvs_.begin(), scvs_.end(), other.scvs_.begin());
    }

    
    auto sorted_flat = [](const std::vector<uint>& flat, uint L) -> std::vector<uint> {
        uint count = flat.size() / L;
        std::vector<uint> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](uint a, uint b) {
            return std::lexicographical_compare(flat.data() + a * L, flat.data() + (a + 1) * L,
                                                 flat.data() + b * L, flat.data() + (b + 1) * L);
        });
        std::vector<uint> result;
        result.reserve(flat.size());
        for (uint i : indices) {
            const uint* s = flat.data() + i * L;
            result.insert(result.end(), s, s + L);
        }
        return result;
    };

    return sorted_flat(scvs_, L_) == sorted_flat(other.scvs_, L_);
}
