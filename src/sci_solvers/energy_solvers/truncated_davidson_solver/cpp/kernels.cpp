/*
 * kernels.cpp
 * ===========
 * C++17 / OpenMP translation of numba_kernels.py.
 *
 * Exported Python symbols (via pybind11):
 *   add_basis_vector_fused(...)
 *   calc_trunc_pcr_fused(...)
 *   singles_kernel(...)
 *   doubles_same_spin_kernel(...)
 *   doubles_cross_spin_kernel(...)
 *   build_addr_dict(...)   [not needed at runtime, exported for completeness]
 *
 * Parallelization strategy mirrors the numba version:
 *   - Pass 1 (serial)  : register every determinant of new_vector.
 *   - Pass 2 (parallel): each thread accumulates into a private buffer;
 *                        final serial reduction.
 *
 * Hash-table design mirrors hash_table.py:
 *   open-addressing linear probing, Fibonacci mixing, load <= 0.5,
 *   sentinel EMPTY = -1.
 *
 * (C) Copyright IBM Corp. 2026
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
using i64  = int64_t;
using f64  = double;
using AI   = py::array_t<i64, py::array::c_style | py::array::forcecast>;
using AI3  = py::array_t<i64, py::array::c_style | py::array::forcecast>;
using AF   = py::array_t<f64, py::array::c_style | py::array::forcecast>;
using AF2  = py::array_t<f64, py::array::c_style | py::array::forcecast>;
using AF3  = py::array_t<f64, py::array::c_style | py::array::forcecast>;
using AF4  = py::array_t<f64, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// Flat hash table (mirrors hash_table.py)
// ---------------------------------------------------------------------------
static constexpr i64 EMPTY = i64(-1);

// Address lookup into an addr table.
// When use_dense=true: O(1) direct index into the 2^norb sentinel-filled array.
// When use_dense=false: O(1) average hash map lookup (norb > 24 path).
inline i64 addr_lookup(const i64* table, i64 table_size, i64 key, bool use_dense,
                       const std::unordered_map<i64,i64>* map = nullptr) {
    if (use_dense) {
        return (key < table_size) ? table[key] : i64(-1);
    } else {
        auto it = map->find(key);
        return (it == map->end()) ? i64(-1) : it->second;
    }
}

inline i64 ht_cap_for(i64 capacity) {
    i64 target = capacity * 2;
    i64 cap = 1;
    while (cap < target) cap <<= 1;
    return cap;
}

struct HT {
    std::vector<i64> keys;
    std::vector<f64> vals;
    i64 cnt = 0;

    explicit HT(i64 capacity) {
        i64 cap = ht_cap_for(std::max(capacity, i64(1)));
        keys.assign(cap, EMPTY);
        vals.assign(cap, 0.0);
    }

    i64 mask() const { return (i64)keys.size() - 1; }

    i64 slot(i64 key) const {
        i64 m = mask();
        i64 h = (key * i64(0x9E3779B97F4A7C15LL)) & m;
        while (keys[h] != EMPTY && keys[h] != key)
            h = (h + 1) & m;
        return h;
    }

    void grow() {
        i64 old_cap = (i64)keys.size();
        i64 new_cap = old_cap * 2;
        std::vector<i64> nk(new_cap, EMPTY);
        std::vector<f64> nv(new_cap, 0.0);
        i64 nm = new_cap - 1;
        for (i64 i = 0; i < old_cap; ++i) {
            i64 k = keys[i];
            if (k == EMPTY) continue;
            i64 h = (k * i64(0x9E3779B97F4A7C15LL)) & nm;
            while (nk[h] != EMPTY) h = (h + 1) & nm;
            nk[h] = k;
            nv[h] = vals[i];
        }
        keys = std::move(nk);
        vals = std::move(nv);
    }

    // accumulate delta into key; grow if needed.
    void add(i64 key, f64 delta) {
        if ((cnt + 1) * 2 > (i64)keys.size()) grow();
        i64 s = slot(key);
        if (keys[s] == EMPTY) { keys[s] = key; ++cnt; }
        vals[s] += delta;
    }

    f64 get(i64 key, f64 def = 0.0) const {
        i64 s = slot(key);
        return (keys[s] == key) ? vals[s] : def;
    }

    bool contains(i64 key) const {
        i64 s = slot(key);
        return keys[s] == key;
    }
};

// ---------------------------------------------------------------------------
// Registry: flat_det_idx  ->  {basis_idx -> coeff}
// Mirrors the NumbaDict[int64, DictType(int64,float64)] registry.
// ---------------------------------------------------------------------------
using InnerMap = std::unordered_map<i64, f64>;
using Registry = std::unordered_map<i64, InnerMap>;

// ---------------------------------------------------------------------------
// Bit utilities
// ---------------------------------------------------------------------------
// Portable 64-bit popcount. C++17 has no std::popcount (that arrives in C++20),
// so wrap the compiler intrinsics behind a single inline helper instead of
// scattering non-standard __builtin_popcountll calls through the kernels.
inline int popcount64(uint64_t x) {
#if defined(_MSC_VER)
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}

inline int popcount_between(i64 bits, int lo, int hi) {
    // count set bits strictly between lo and hi
    i64 mask = ((i64(1) << hi) - 1) & ~((i64(1) << (lo + 1)) - 1);
    return popcount64((uint64_t)(bits & mask));
}

inline int sign_single(i64 bits, int i, int a) {
    int lo = std::min(i, a);
    int hi = std::max(i, a);
    return (popcount_between(bits, lo, hi) % 2 == 0) ? 1 : -1;
}

inline void decode_bits(i64 bits, int norb, std::vector<int>& occ, std::vector<int>& virt) {
    occ.clear(); virt.clear();
    for (int p = 0; p < norb; ++p) {
        if ((bits >> p) & 1) occ.push_back(p);
        else                 virt.push_back(p);
    }
}

// ---------------------------------------------------------------------------
// Diagonal helpers
// ---------------------------------------------------------------------------
inline f64 hii_from_bits(
    i64 bits_a, i64 bits_b,
    const f64* hdiag1,
    const f64* jdiag, const f64* kdiag,   // [norb x norb] row-major
    int norb, int nelec_a, int nelec_b)
{
    // extract occupied indices
    std::array<int, 64> occ_a{}, occ_b{};
    {
        int oa = 0;
        for (int p = 0; p < norb; ++p) if ((bits_a >> p) & 1) occ_a[oa++] = p;
    }
    {
        int ob = 0;
        for (int p = 0; p < norb; ++p) if ((bits_b >> p) & 1) occ_b[ob++] = p;
    }
    f64 e = 0.0;
    for (int i = 0; i < nelec_a; ++i) e += hdiag1[occ_a[i]];
    for (int i = 0; i < nelec_b; ++i) e += hdiag1[occ_b[i]];
    for (int i = 0; i < nelec_a; ++i)
        for (int j = i+1; j < nelec_a; ++j) {
            int pi = occ_a[i], pj = occ_a[j];
            e += jdiag[pi*norb+pj] - kdiag[pi*norb+pj];
        }
    for (int i = 0; i < nelec_b; ++i)
        for (int j = i+1; j < nelec_b; ++j) {
            int pi = occ_b[i], pj = occ_b[j];
            e += jdiag[pi*norb+pj] - kdiag[pi*norb+pj];
        }
    for (int i = 0; i < nelec_a; ++i)
        for (int j = 0; j < nelec_b; ++j)
            e += jdiag[occ_a[i]*norb + occ_b[j]];
    return e;
}

inline f64 ediag(
    bool use_dense_hdiag, i64 new_flat,
    i64 bits_a_val, i64 bits_b_val,
    const f64* hdiag, const f64* hdiag1,
    const f64* jdiag, const f64* kdiag,
    int norb, int nelec_a, int nelec_b)
{
    if (use_dense_hdiag) return hdiag[new_flat];
    return hii_from_bits(bits_a_val, bits_b_val, hdiag1, jdiag, kdiag, norb, nelec_a, nelec_b);
}

// O(1) diagonal delta helpers
inline f64 delta_single(int i, int a,
    const f64* hdiag1, const f64* jdiag, const f64* kdiag,
    const f64* Sx, const f64* Ty, int norb)
{
    return (hdiag1[a] - hdiag1[i])
         + (Sx[a] - Sx[i] - (jdiag[a*norb+i] - kdiag[a*norb+i]))
         + (Ty[a] - Ty[i]);
}

inline f64 delta_double_same_spin(int i, int j, int a, int b,
    const f64* hdiag1, const f64* jdiag, const f64* kdiag,
    const f64* Sx, const f64* Ty, int norb)
{
    return (hdiag1[a] + hdiag1[b] - hdiag1[i] - hdiag1[j])
         + (Sx[a] + Sx[b] - Sx[i] - Sx[j])
         - (jdiag[a*norb+i] - kdiag[a*norb+i])
         - (jdiag[a*norb+j] - kdiag[a*norb+j])
         - (jdiag[b*norb+i] - kdiag[b*norb+i])
         - (jdiag[b*norb+j] - kdiag[b*norb+j])
         + (jdiag[a*norb+b] - kdiag[a*norb+b])
         + (jdiag[i*norb+j] - kdiag[i*norb+j])
         + (Ty[a] + Ty[b] - Ty[i] - Ty[j]);
}

// Build Sa/Sb/Ta/Tb per-I rank-update arrays
inline void build_diag_ST(
    const std::vector<int>& occ_a, const std::vector<int>& occ_b,
    int norb, const f64* jdiag, const f64* kdiag,
    f64* Sa, f64* Sb, f64* Ta, f64* Tb)
{
    for (int p = 0; p < norb; ++p) {
        f64 sa = 0, ta = 0, sb = 0, tb = 0;
        for (int k : occ_a) { sa += jdiag[p*norb+k] - kdiag[p*norb+k]; ta += jdiag[p*norb+k]; }
        for (int k : occ_b) { sb += jdiag[p*norb+k] - kdiag[p*norb+k]; tb += jdiag[p*norb+k]; }
        Sa[p] = sa; Ta[p] = ta; Sb[p] = sb; Tb[p] = tb;
    }
}

// ---------------------------------------------------------------------------
// addr_get: address table is either int64[:] (dense) or unordered_map
// We use a tagged union approach via a simple wrapper.
// ---------------------------------------------------------------------------
struct AddrTable {
    bool is_dense;
    const i64* arr;     // used when is_dense
    i64 arr_size;
    const std::unordered_map<i64,i64>* dict; // used when !is_dense

    i64 get(i64 key) const {
        if (is_dense) {
            if (key < 0 || key >= arr_size) return i64(-1);
            return arr[key];
        }
        auto it = dict->find(key);
        return (it != dict->end()) ? it->second : i64(-1);
    }
};

// ---------------------------------------------------------------------------
// singles_kernel
// ---------------------------------------------------------------------------
py::tuple singles_kernel_py(
    AI occ_py, AI virt_py, AI occ_other_py,
    i64 bits0, AF2 h1e_py, AF4 eri_py)
{
    auto occ       = occ_py.unchecked<1>();
    auto virt      = virt_py.unchecked<1>();
    auto occ_other = occ_other_py.unchecked<1>();
    auto h1e       = h1e_py.unchecked<2>();
    auto eri       = eri_py.unchecked<4>();

    int n_occ  = (int)occ.shape(0);
    int n_virt = (int)virt.shape(0);
    (void)h1e.shape(0); // norb unused here
    int n      = n_occ * n_virt;

    AI new_bits_py((py::ssize_t)n);
    AF elem_py((py::ssize_t)n);
    AI i_arr_py((py::ssize_t)n);
    AI a_arr_py((py::ssize_t)n);

    auto new_bits = new_bits_py.mutable_unchecked<1>();
    auto elem     = elem_py.mutable_unchecked<1>();
    auto i_arr    = i_arr_py.mutable_unchecked<1>();
    auto a_arr    = a_arr_py.mutable_unchecked<1>();

    int idx = 0;
    for (int oi = 0; oi < n_occ; ++oi) {
        int i = (int)occ(oi);
        for (int vi = 0; vi < n_virt; ++vi) {
            int a = (int)virt(vi);
            int lo = std::min(i,a), hi = std::max(i,a);
            // parity of occupied orbitals strictly between lo and hi, via a
            // bit-mask + popcount (O(1)); equivalent to the scan over `occ`
            // since `occ` enumerates exactly the set bits of `bits0`.
            i64 mask = ((i64(1)<<hi)-1) & ~((i64(1)<<(lo+1))-1);
            f64 sign = (popcount64((uint64_t)(bits0&mask))%2==0) ? 1.0 : -1.0;

            f64 e = h1e(a, i);
            for (int ki = 0; ki < n_occ; ++ki) {
                int k = (int)occ(ki);
                e += eri(a, i, k, k) - eri(a, k, k, i);
            }
            for (int ki = 0; ki < (int)occ_other.shape(0); ++ki) {
                int k = (int)occ_other(ki);
                e += eri(a, i, k, k);
            }

            new_bits(idx) = (bits0 & ~(i64(1) << i)) | (i64(1) << a);
            elem(idx)     = e * sign;
            i_arr(idx)    = i;
            a_arr(idx)    = a;
            ++idx;
        }
    }
    return py::make_tuple(new_bits_py, elem_py, i_arr_py, a_arr_py);
}

// ---------------------------------------------------------------------------
// doubles_same_spin_kernel
// ---------------------------------------------------------------------------
py::tuple doubles_same_spin_kernel_py(
    AI occ_py, AI virt_py, i64 bits0, AF4 eri_py)
{
    auto occ  = occ_py.unchecked<1>();
    auto virt = virt_py.unchecked<1>();
    auto eri  = eri_py.unchecked<4>();

    int n_occ  = (int)occ.shape(0);
    int n_virt = (int)virt.shape(0);
    int n_occ_pairs  = n_occ  * (n_occ  - 1) / 2;
    int n_virt_pairs = n_virt * (n_virt - 1) / 2;
    int n = n_occ_pairs * n_virt_pairs;

    AI new_bits_py((py::ssize_t)n);
    AF elem_py((py::ssize_t)n);
    auto new_bits = new_bits_py.mutable_unchecked<1>();
    auto elem     = elem_py.mutable_unchecked<1>();

    int idx = 0;
    for (int oi = 0; oi < n_occ; ++oi) {
        int i = (int)occ(oi);
        for (int oj = oi+1; oj < n_occ; ++oj) {
            int j = (int)occ(oj);
            for (int vi = 0; vi < n_virt; ++vi) {
                int a = (int)virt(vi);
                int lo1 = std::min(i,a), hi1 = std::max(i,a);
                int p1 = 0;
                for (int xi = 0; xi < n_occ; ++xi) {
                    int x = (int)occ(xi);
                    if (lo1 < x && x < hi1) p1++;
                }
                f64 s1 = (p1 % 2 == 0) ? 1.0 : -1.0;

                for (int vj = vi+1; vj < n_virt; ++vj) {
                    int b = (int)virt(vj);
                    int lo2 = std::min(j,b), hi2 = std::max(j,b);
                    int p2 = 0;
                    for (int xi = 0; xi < n_occ; ++xi) {
                        int x = (int)occ(xi);
                        if (x == i) continue;
                        if (lo2 < x && x < hi2) p2++;
                    }
                    if (lo2 < a && a < hi2) p2++;
                    f64 s2 = (p2 % 2 == 0) ? 1.0 : -1.0;

                    f64 sign = s1 * s2;
                    f64 e = sign * (eri(a,i,b,j) - eri(a,j,b,i));
                    new_bits(idx) = (bits0 & ~(i64(1)<<i) & ~(i64(1)<<j))
                                  | (i64(1)<<a) | (i64(1)<<b);
                    elem(idx) = e;
                    ++idx;
                }
            }
        }
    }
    return py::make_tuple(new_bits_py, elem_py);
}

// ---------------------------------------------------------------------------
// doubles_cross_spin_kernel
// ---------------------------------------------------------------------------
py::tuple doubles_cross_spin_kernel_py(
    AI occ_a_py, AI virt_a_py,
    AI occ_b_py, AI virt_b_py,
    i64 bits_a0, i64 bits_b0, AF4 eri_py)
{
    auto occ_a  = occ_a_py.unchecked<1>();
    auto virt_a = virt_a_py.unchecked<1>();
    auto occ_b  = occ_b_py.unchecked<1>();
    auto virt_b = virt_b_py.unchecked<1>();
    auto eri    = eri_py.unchecked<4>();

    int n_oa = (int)occ_a.shape(0);
    int n_va = (int)virt_a.shape(0);
    int n_ob = (int)occ_b.shape(0);
    int n_vb = (int)virt_b.shape(0);
    int n    = n_oa * n_va * n_ob * n_vb;

    AI new_bits_a_py((py::ssize_t)n);
    AI new_bits_b_py((py::ssize_t)n);
    AF elem_py((py::ssize_t)n);
    auto nba = new_bits_a_py.mutable_unchecked<1>();
    auto nbb = new_bits_b_py.mutable_unchecked<1>();
    auto elem = elem_py.mutable_unchecked<1>();

    int idx = 0;
    for (int oi = 0; oi < n_oa; ++oi) {
        int i = (int)occ_a(oi);
        for (int vi = 0; vi < n_va; ++vi) {
            int a = (int)virt_a(vi);
            int lo1 = std::min(i,a), hi1 = std::max(i,a);
            int p1 = 0;
            for (int xi = 0; xi < n_oa; ++xi) {
                int x = (int)occ_a(xi);
                if (lo1 < x && x < hi1) p1++;
            }
            f64 s1 = (p1 % 2 == 0) ? 1.0 : -1.0;
            i64 new_ba = (bits_a0 & ~(i64(1)<<i)) | (i64(1)<<a);

            for (int oj = 0; oj < n_ob; ++oj) {
                int j = (int)occ_b(oj);
                for (int vj = 0; vj < n_vb; ++vj) {
                    int b = (int)virt_b(vj);
                    int lo2 = std::min(j,b), hi2 = std::max(j,b);
                    int p2 = 0;
                    for (int xj = 0; xj < n_ob; ++xj) {
                        int x = (int)occ_b(xj);
                        if (lo2 < x && x < hi2) p2++;
                    }
                    f64 s2 = (p2 % 2 == 0) ? 1.0 : -1.0;

                    f64 sign = s1 * s2;
                    f64 e    = sign * eri(a, i, b, j);
                    i64 new_bb = (bits_b0 & ~(i64(1)<<j)) | (i64(1)<<b);

                    nba(idx)  = new_ba;
                    nbb(idx)  = new_bb;
                    elem(idx) = e;
                    ++idx;
                }
            }
        }
    }
    return py::make_tuple(new_bits_a_py, new_bits_b_py, elem_py);
}

// ---------------------------------------------------------------------------
// pcr_doubles_same_spin_kernel
// ---------------------------------------------------------------------------
// Returns (out_idx, out_val, n_out)
py::tuple pcr_doubles_same_spin_kernel_py(
    i64 bits0, i64 ib_fixed, i64 nb, i64 use_alpha, i64 ia_fixed,
    AI a_arr_py, AI b_arr_py, AF val_arr_py,
    AI addr_table_py,
    bool use_dense_addr,
    bool use_dense_hdiag,
    AF hdiag_py, AF hdiag1_py, AF2 jdiag_py, AF2 kdiag_py,
    f64 E_I, AF Sx_py, AF Ty_py,
    f64 cI, f64 E, f64 delta,
    i64 i_orb, i64 j_orb,
    AI out_idx_py, AF out_val_py)
{
    auto a_arr    = a_arr_py.unchecked<1>();
    auto b_arr    = b_arr_py.unchecked<1>();
    auto val_arr  = val_arr_py.unchecked<1>();
    const i64* addr_p    = addr_table_py.data();
    const i64  addr_size = addr_table_py.shape(0);
    const f64* Sx_p    = Sx_py.data();
    const f64* Ty_p    = Ty_py.data();
    auto out_idx  = out_idx_py.mutable_unchecked<1>();
    auto out_val  = out_val_py.mutable_unchecked<1>();

    const f64* hdiag_p  = use_dense_hdiag ? hdiag_py.data()  : nullptr;
    const f64* hdiag1_p = use_dense_hdiag ? nullptr           : hdiag1_py.data();
    const f64* jdiag_p  = use_dense_hdiag ? nullptr           : jdiag_py.data();
    const f64* kdiag_p  = use_dense_hdiag ? nullptr           : kdiag_py.data();
    int norb = use_dense_hdiag ? 0 : (int)jdiag_py.shape(0);

    constexpr f64 floor = 1e-10;
    i64 n_out = 0;
    i64 n_ab  = a_arr.shape(0);
    i64 i = i_orb, j = j_orb;

    for (i64 k = 0; k < n_ab; ++k) {
        int a = (int)a_arr(k);
        int b = (int)b_arr(k);
        if ((bits0 >> a) & i64(1)) continue;
        if ((bits0 >> b) & i64(1)) continue;

        f64 raw_c = val_arr(k) * cI;
        if (std::abs(raw_c) < delta) break;

        // sign i->a
        int lo1 = (int)std::min(i,(i64)a), hi1 = (int)std::max(i,(i64)a);
        i64 mask1 = ((i64(1)<<hi1)-1) & ~((i64(1)<<(lo1+1))-1);
        int s1 = (popcount64((uint64_t)(bits0 & mask1)) % 2 == 0) ? 1 : -1;

        i64 bits_mid = (bits0 & ~(i64(1)<<i)) | (i64(1)<<a);

        // sign j->b on intermediate
        int lo2 = (int)std::min(j,(i64)b), hi2 = (int)std::max(j,(i64)b);
        i64 mask2 = ((i64(1)<<hi2)-1) & ~((i64(1)<<(lo2+1))-1);
        int s2 = (popcount64((uint64_t)(bits_mid & mask2)) % 2 == 0) ? 1 : -1;

        f64 cIp = (f64)(s1*s2) * val_arr(k) * cI;

        i64 new_bits = (bits0 & ~(i64(1)<<i) & ~(i64(1)<<j))
                     | (i64(1)<<a) | (i64(1)<<b);

        i64 new_addr = addr_lookup(addr_p, addr_size, new_bits, use_dense_addr);
        if (new_addr < 0) continue;

        i64 new_flat = use_alpha ? (new_addr * nb + ib_fixed)
                                 : (ia_fixed * nb + new_addr);

        f64 e_diag;
        if (use_dense_hdiag) {
            e_diag = hdiag_p[new_flat];
        } else {
            e_diag = E_I + delta_double_same_spin(
                (int)i, (int)j, a, b, hdiag1_p, jdiag_p, kdiag_p,
                Sx_p, Ty_p, norb);
        }

        f64 denom = e_diag - E;
        if (std::abs(denom) < floor) denom = (denom >= 0.0) ? floor : -floor;
        cIp /= denom;

        if (std::abs(cIp) > delta) {
            out_idx(n_out) = new_flat;
            out_val(n_out) = cIp;
            ++n_out;
        }
    }
    return py::make_tuple(out_idx_py, out_val_py, n_out);
}

// ---------------------------------------------------------------------------
// pcr_doubles_cross_spin_kernel
// ---------------------------------------------------------------------------
py::tuple pcr_doubles_cross_spin_kernel_py(
    i64 bits_a0, i64 bits_b0, i64 nb,
    AI a_arr_py, AI b_arr_py, AF val_arr_py,
    AI addr_table_a_py, AI addr_table_b_py,
    bool use_dense_addr,
    bool use_dense_hdiag,
    AF hdiag_py, AF hdiag1_py, AF2 jdiag_py, AF2 kdiag_py,
    i64 nelec_a, i64 nelec_b,
    f64 cI, f64 E, f64 delta,
    i64 i_orb, i64 j_orb,
    AI out_idx_py, AF out_val_py)
{
    auto a_arr   = a_arr_py.unchecked<1>();
    auto b_arr   = b_arr_py.unchecked<1>();
    auto val_arr = val_arr_py.unchecked<1>();
    const i64* addr_a_p    = addr_table_a_py.data();
    const i64  addr_a_size = addr_table_a_py.shape(0);
    const i64* addr_b_p    = addr_table_b_py.data();
    const i64  addr_b_size = addr_table_b_py.shape(0);
    auto out_idx = out_idx_py.mutable_unchecked<1>();
    auto out_val = out_val_py.mutable_unchecked<1>();

    const f64* hdiag_p  = use_dense_hdiag ? hdiag_py.data()  : nullptr;
    const f64* hdiag1_p = use_dense_hdiag ? nullptr           : hdiag1_py.data();
    const f64* jdiag_p  = use_dense_hdiag ? nullptr           : jdiag_py.data();
    const f64* kdiag_p  = use_dense_hdiag ? nullptr           : kdiag_py.data();
    int norb = use_dense_hdiag ? 0 : (int)jdiag_py.shape(0);

    constexpr f64 floor = 1e-10;
    i64 n_out = 0;
    i64 n_ab  = a_arr.shape(0);
    i64 i = i_orb, j = j_orb;

    for (i64 k = 0; k < n_ab; ++k) {
        int a = (int)a_arr(k);
        int b = (int)b_arr(k);
        if ((bits_a0 >> a) & i64(1)) continue;
        if ((bits_b0 >> b) & i64(1)) continue;

        f64 raw_c = val_arr(k) * cI;
        if (std::abs(raw_c) < delta) break;

        int lo1 = (int)std::min(i,(i64)a), hi1 = (int)std::max(i,(i64)a);
        i64 mask1 = ((i64(1)<<hi1)-1) & ~((i64(1)<<(lo1+1))-1);
        int s1 = (popcount64((uint64_t)(bits_a0 & mask1)) % 2 == 0) ? 1 : -1;

        int lo2 = (int)std::min(j,(i64)b), hi2 = (int)std::max(j,(i64)b);
        i64 mask2 = ((i64(1)<<hi2)-1) & ~((i64(1)<<(lo2+1))-1);
        int s2 = (popcount64((uint64_t)(bits_b0 & mask2)) % 2 == 0) ? 1 : -1;

        f64 cIp = (f64)(s1*s2) * val_arr(k) * cI;

        i64 new_ba = (bits_a0 & ~(i64(1)<<i)) | (i64(1)<<a);
        i64 new_bb = (bits_b0 & ~(i64(1)<<j)) | (i64(1)<<b);
        i64 new_ia = addr_lookup(addr_a_p, addr_a_size, new_ba, use_dense_addr, nullptr);
        i64 new_ib = addr_lookup(addr_b_p, addr_b_size, new_bb, use_dense_addr, nullptr);
        if (new_ia < 0 || new_ib < 0) continue;

        i64 new_flat = new_ia * nb + new_ib;

        f64 e_diag = ediag(
            use_dense_hdiag, new_flat, new_ba, new_bb,
            hdiag_p, hdiag1_p, jdiag_p, kdiag_p,
            norb, (int)nelec_a, (int)nelec_b);

        f64 denom = e_diag - E;
        if (std::abs(denom) < floor) denom = (denom >= 0.0) ? floor : -floor;
        cIp /= denom;

        if (std::abs(cIp) > delta) {
            out_idx(n_out) = new_flat;
            out_val(n_out) = cIp;
            ++n_out;
        }
    }
    return py::make_tuple(out_idx_py, out_val_py, n_out);
}

// ---------------------------------------------------------------------------
// add_basis_vector_fused
// ---------------------------------------------------------------------------
// registry is passed as two parallel arrays (keys, vals flattened from
// the nested Python structure is complex; instead we keep a C++ Registry
// in a py::capsule and return it so Python can hold it opaquely).
//
// To match the numba API exactly, registry is a py::object (capsule).
// ---------------------------------------------------------------------------

// Opaque registry handle exposed to Python
struct RegistryCapsule {
    Registry reg;
    // Sparse address maps for norb > 24 (use_dense_addr=false).
    // Built once via build_addr_maps(); empty when use_dense_addr=true.
    std::unordered_map<i64,i64> addr_map_a, addr_map_b;
};

py::capsule make_registry() {
    auto* r = new RegistryCapsule();
    return py::capsule(r, [](void* p) { delete static_cast<RegistryCapsule*>(p); });
}

void build_addr_maps(py::object registry_cap, AI bits_a_py, AI bits_b_py) {
    py::capsule cap = registry_cap.cast<py::capsule>();
    auto* rc = static_cast<RegistryCapsule*>(static_cast<void*>(cap));
    auto bits_a = bits_a_py.unchecked<1>();
    auto bits_b = bits_b_py.unchecked<1>();
    rc->addr_map_a.reserve(bits_a.shape(0) * 2);
    for (i64 i = 0; i < bits_a.shape(0); ++i) rc->addr_map_a[bits_a(i)] = i;
    rc->addr_map_b.reserve(bits_b.shape(0) * 2);
    for (i64 i = 0; i < bits_b.shape(0); ++i) rc->addr_map_b[bits_b(i)] = i;
}


void add_basis_vector_fused_py(
    AI new_idx_py, AF new_val_py, i64 n_new,
    py::object registry_cap,
    i64 norb, i64 nb,
    i64 nelec_a, i64 nelec_b,
    AI bits_a_py, AI bits_b_py,
    AI addr_table_a_py, AI addr_table_b_py,
    bool use_dense_addr,
    AF2 h1e_py, AF4 eri_py,
    bool use_dense_hdiag,
    AF hdiag_py, AF hdiag1_py, AF2 jdiag_py, AF2 kdiag_py,
    AF col_py)
{
    py::capsule cap = registry_cap.cast<py::capsule>();
    auto* rc = static_cast<RegistryCapsule*>(static_cast<void*>(cap));
    auto& registry = rc->reg;
    const std::unordered_map<i64,i64>* map_a = use_dense_addr ? nullptr : &rc->addr_map_a;
    const std::unordered_map<i64,i64>* map_b = use_dense_addr ? nullptr : &rc->addr_map_b;

    auto new_idx   = new_idx_py.unchecked<1>();
    auto new_val   = new_val_py.unchecked<1>();
    auto bits_a    = bits_a_py.unchecked<1>();
    auto bits_b    = bits_b_py.unchecked<1>();
    const i64* addr_a_p    = addr_table_a_py.data();
    const i64  addr_a_size = addr_table_a_py.shape(0);
    const i64* addr_b_p    = addr_table_b_py.data();
    const i64  addr_b_size = addr_table_b_py.shape(0);
    auto h1e       = h1e_py.unchecked<2>();
    auto eri       = eri_py.unchecked<4>();
    auto col       = col_py.mutable_unchecked<1>();

    const f64* hdiag_p  = use_dense_hdiag ? hdiag_py.data()  : nullptr;
    const f64* hdiag1_p = use_dense_hdiag ? nullptr           : hdiag1_py.data();
    const f64* jdiag_p  = use_dense_hdiag ? nullptr           : jdiag_py.data();
    const f64* kdiag_p  = use_dense_hdiag ? nullptr           : kdiag_py.data();

    i64 n_J = new_idx.shape(0);
    int ncol = (int)col.shape(0);

    // Pass 1: serial register
    for (i64 _p = 0; _p < n_J; ++_p) {
        i64 J_idx = new_idx(_p);
        f64 cJ    = new_val(_p);
        registry[J_idx][n_new] = cJ;
    }

    // Pass 2: parallel accumulate — each thread owns a private col_local slice.
#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif

    std::vector<f64> col_local((size_t)n_threads * ncol, 0.0);
    i64 chunk = (n_J + n_threads - 1) / n_threads;

#pragma omp parallel for schedule(static,1) num_threads(n_threads)
    for (int t = 0; t < n_threads; ++t) {
        i64 start = (i64)t * chunk;
        i64 end   = std::min(start + chunk, n_J);

        std::vector<int> occ_a, virt_a, occ_b, virt_b;
        occ_a.reserve(nelec_a); virt_a.reserve(norb - nelec_a);
        occ_b.reserve(nelec_b); virt_b.reserve(norb - nelec_b);

        for (i64 idx = start; idx < end; ++idx) {
            i64 J_idx = new_idx(idx);
            f64 cJ    = new_val(idx);
            i64 ia    = J_idx / nb;
            i64 ib    = J_idx % nb;
            i64 ba0   = bits_a(ia);
            i64 bb0   = bits_b(ib);

            decode_bits(ba0, (int)norb, occ_a, virt_a);
            decode_bits(bb0, (int)norb, occ_b, virt_b);

            // Diagonal
            auto it = registry.find(J_idx);
            if (it != registry.end()) {
                f64 e_diag = ediag(
                    use_dense_hdiag, J_idx, ba0, bb0,
                    hdiag_p, hdiag1_p, jdiag_p, kdiag_p,
                    (int)norb, (int)nelec_a, (int)nelec_b);
                for (auto& [i_basis, ci] : it->second)
                    col_local[(size_t)t * ncol + i_basis] += ci * cJ * e_diag;
            }

            // We rebuild single/double bit arrays inline here to avoid
            // building Python arrays inside the parallel region.
            // Alpha singles
            for (int oi = 0; oi < (int)occ_a.size(); ++oi) {
                int i = occ_a[oi];
                for (int vi = 0; vi < (int)virt_a.size(); ++vi) {
                    int a = virt_a[vi];
                    int lo = std::min(i,a), hi = std::max(i,a);
                    int p = 0;
                    for (int x : occ_a) if (lo < x && x < hi) p++;
                    f64 sign = (p % 2 == 0) ? 1.0 : -1.0;

                    f64 e = h1e(a, i);
                    for (int k : occ_a) e += eri(a,i,k,k) - eri(a,k,k,i);
                    for (int k : occ_b) e += eri(a,i,k,k);
                    e *= sign;

                    i64 nb_bits = (ba0 & ~(i64(1)<<i)) | (i64(1)<<a);
                    i64 na = addr_lookup(addr_a_p, addr_a_size, nb_bits, use_dense_addr, map_a);
                    if (na < 0) continue;
                    i64 I_idx = na * nb + ib;
                    auto it2 = registry.find(I_idx);
                    if (it2 == registry.end()) continue;
                    for (auto& [i_basis, cI] : it2->second)
                        col_local[(size_t)t * ncol + i_basis] += cI * cJ * e;
                }
            }

            // Beta singles
            for (int oi = 0; oi < (int)occ_b.size(); ++oi) {
                int i = occ_b[oi];
                for (int vi = 0; vi < (int)virt_b.size(); ++vi) {
                    int a = virt_b[vi];
                    int lo = std::min(i,a), hi = std::max(i,a);
                    int p = 0;
                    for (int x : occ_b) if (lo < x && x < hi) p++;
                    f64 sign = (p % 2 == 0) ? 1.0 : -1.0;

                    f64 e = h1e(a, i);
                    for (int k : occ_b) e += eri(a,i,k,k) - eri(a,k,k,i);
                    for (int k : occ_a) e += eri(a,i,k,k);
                    e *= sign;

                    i64 nb_bits = (bb0 & ~(i64(1)<<i)) | (i64(1)<<a);
                    i64 nb_addr = addr_lookup(addr_b_p, addr_b_size, nb_bits, use_dense_addr, map_b);
                    if (nb_addr < 0) continue;
                    i64 I_idx = ia * nb + nb_addr;
                    auto it2 = registry.find(I_idx);
                    if (it2 == registry.end()) continue;
                    for (auto& [i_basis, cI] : it2->second)
                        col_local[(size_t)t * ncol + i_basis] += cI * cJ * e;
                }
            }

            // Alpha-alpha doubles
            for (int oi = 0; oi < (int)occ_a.size(); ++oi) {
                int i = occ_a[oi];
                for (int oj = oi+1; oj < (int)occ_a.size(); ++oj) {
                    int j = occ_a[oj];
                    for (int vi = 0; vi < (int)virt_a.size(); ++vi) {
                        int a = virt_a[vi];
                        int lo1 = std::min(i,a), hi1 = std::max(i,a);
                        int p1 = 0;
                        for (int x : occ_a) if (lo1 < x && x < hi1) p1++;
                        f64 s1 = (p1 % 2 == 0) ? 1.0 : -1.0;
                        for (int vj = vi+1; vj < (int)virt_a.size(); ++vj) {
                            int b = virt_a[vj];
                            int lo2 = std::min(j,b), hi2 = std::max(j,b);
                            int p2 = 0;
                            for (int x : occ_a) { if (x==i) continue; if (lo2<x&&x<hi2) p2++; }
                            if (lo2 < a && a < hi2) p2++;
                            f64 s2 = (p2 % 2 == 0) ? 1.0 : -1.0;
                            f64 sign = s1 * s2;
                            f64 e    = sign * (eri(a,i,b,j) - eri(a,j,b,i));
                            i64 nb_bits = (ba0 & ~(i64(1)<<i) & ~(i64(1)<<j))
                                        | (i64(1)<<a) | (i64(1)<<b);
                            i64 na = addr_lookup(addr_a_p, addr_a_size, nb_bits, use_dense_addr, map_a);
                            if (na < 0) continue;
                            i64 I_idx = na * nb + ib;
                            auto it2 = registry.find(I_idx);
                            if (it2 == registry.end()) continue;
                            for (auto& [i_basis, cI] : it2->second)
                                col_local[(size_t)t * ncol + i_basis] += cI * cJ * e;
                        }
                    }
                }
            }

            // Beta-beta doubles
            for (int oi = 0; oi < (int)occ_b.size(); ++oi) {
                int i = occ_b[oi];
                for (int oj = oi+1; oj < (int)occ_b.size(); ++oj) {
                    int j = occ_b[oj];
                    for (int vi = 0; vi < (int)virt_b.size(); ++vi) {
                        int a = virt_b[vi];
                        int lo1 = std::min(i,a), hi1 = std::max(i,a);
                        int p1 = 0;
                        for (int x : occ_b) if (lo1 < x && x < hi1) p1++;
                        f64 s1 = (p1 % 2 == 0) ? 1.0 : -1.0;
                        for (int vj = vi+1; vj < (int)virt_b.size(); ++vj) {
                            int b = virt_b[vj];
                            int lo2 = std::min(j,b), hi2 = std::max(j,b);
                            int p2 = 0;
                            for (int x : occ_b) { if (x==i) continue; if (lo2<x&&x<hi2) p2++; }
                            if (lo2 < a && a < hi2) p2++;
                            f64 s2 = (p2 % 2 == 0) ? 1.0 : -1.0;
                            f64 sign = s1 * s2;
                            f64 e    = sign * (eri(a,i,b,j) - eri(a,j,b,i));
                            i64 nb_bits = (bb0 & ~(i64(1)<<i) & ~(i64(1)<<j))
                                        | (i64(1)<<a) | (i64(1)<<b);
                            i64 nb_addr = addr_lookup(addr_b_p, addr_b_size, nb_bits, use_dense_addr, map_b);
                            if (nb_addr < 0) continue;
                            i64 I_idx = ia * nb + nb_addr;
                            auto it2 = registry.find(I_idx);
                            if (it2 == registry.end()) continue;
                            for (auto& [i_basis, cI] : it2->second)
                                col_local[(size_t)t * ncol + i_basis] += cI * cJ * e;
                        }
                    }
                }
            }

            // Cross alpha-beta doubles
            for (int oi = 0; oi < (int)occ_a.size(); ++oi) {
                int i = occ_a[oi];
                for (int vi = 0; vi < (int)virt_a.size(); ++vi) {
                    int a = virt_a[vi];
                    int lo1 = std::min(i,a), hi1 = std::max(i,a);
                    int p1 = 0;
                    for (int x : occ_a) if (lo1 < x && x < hi1) p1++;
                    f64 s1 = (p1 % 2 == 0) ? 1.0 : -1.0;
                    i64 new_ba = (ba0 & ~(i64(1)<<i)) | (i64(1)<<a);

                    for (int oj = 0; oj < (int)occ_b.size(); ++oj) {
                        int j = occ_b[oj];
                        for (int vj = 0; vj < (int)virt_b.size(); ++vj) {
                            int b = virt_b[vj];
                            int lo2 = std::min(j,b), hi2 = std::max(j,b);
                            int p2 = 0;
                            for (int x : occ_b) if (lo2 < x && x < hi2) p2++;
                            f64 s2 = (p2 % 2 == 0) ? 1.0 : -1.0;

                            f64 sign = s1 * s2;
                            f64 e    = sign * eri(a,i,b,j);
                            i64 new_bb_bits = (bb0 & ~(i64(1)<<j)) | (i64(1)<<b);

                            i64 na = addr_lookup(addr_a_p, addr_a_size, new_ba, use_dense_addr, map_a);
                            i64 nb_addr = addr_lookup(addr_b_p, addr_b_size, new_bb_bits, use_dense_addr, map_b);
                            if (na < 0 || nb_addr < 0) continue;
                            i64 I_idx = na * nb + nb_addr;
                            auto it2 = registry.find(I_idx);
                            if (it2 == registry.end()) continue;
                            for (auto& [i_basis, cI] : it2->second)
                                col_local[(size_t)t * ncol + i_basis] += cI * cJ * e;
                        }
                    }
                }
            }
        } // end idx loop
    } // end parallel

    // Serial reduction
    for (int t = 0; t < n_threads; ++t)
        for (int i = 0; i < ncol; ++i)
            col(i) += col_local[(size_t)t * ncol + i];
}

// ---------------------------------------------------------------------------
// calc_trunc_pcr_fused
// Returns (out_keys: int64[:], out_vals: float64[:])
// ---------------------------------------------------------------------------
py::tuple calc_trunc_pcr_fused_py(
    AI psi_keys_py, AF psi_vals_py,
    i64 norb, i64 nb, i64 nelec_a, i64 nelec_b,
    AI bits_a_py, AI bits_b_py,
    AI addr_table_a_py, AI addr_table_b_py,
    bool use_dense_addr,
    py::object registry_cap,
    AF2 h1e_py, AF4 eri_py,
    bool use_dense_hdiag,
    AF hdiag_py, AF hdiag1_py, AF2 jdiag_py, AF2 kdiag_py,
    // Sorted integral tables as dense, contiguous 3-D arrays [norb, norb, m]:
    // same_* has fixed width m_same = norb*(norb-1) per (i, j) (same-spin, i < j),
    // cross_* has fixed width m_cross = norb*norb per (i, j) (cross-spin, all i, j).
    // Passed flat (not as obj[i][j]) so the kernel indexes them by pointer
    // arithmetic and holds the GIL for no Python table access at all.
    AI3 same_a_py, AI3 same_b_py, AF3 same_v_py,
    AI3 cross_a_py, AI3 cross_b_py, AF3 cross_v_py,
    f64 E, f64 delta,
    f64 level_shift, f64 trunc_eps)
{
    auto psi_keys = psi_keys_py.unchecked<1>();
    auto psi_vals = psi_vals_py.unchecked<1>();
    auto bits_a   = bits_a_py.unchecked<1>();
    auto bits_b   = bits_b_py.unchecked<1>();
    auto h1e      = h1e_py.unchecked<2>();
    auto eri      = eri_py.unchecked<4>();

    const f64* hdiag_p  = use_dense_hdiag ? hdiag_py.data()  : nullptr;
    const f64* hdiag1_p = use_dense_hdiag ? nullptr           : hdiag1_py.data();
    const f64* jdiag_p  = use_dense_hdiag ? nullptr           : jdiag_py.data();
    const f64* kdiag_p  = use_dense_hdiag ? nullptr           : kdiag_py.data();

    // Table slices point directly into the dense 3-D input arrays: row (i, j)
    // of a flat [norb, norb, m] buffer starts at data() + (i*norb + j)*m and
    // has fixed width m. No Python access, so this is valid to read from the
    // parallel region (the arrays outlive this function and are contiguous).
    struct RawSlice { const i64* a; const i64* b; const f64* v; i64 n; };
    const i64 m_same  = norb * (norb - 1);  // same-spin entries per (i, j)
    const i64 m_cross = norb * norb;        // cross-spin entries per (i, j)
    const i64* same_a_p  = same_a_py.data();
    const i64* same_b_p  = same_b_py.data();
    const f64* same_v_p  = same_v_py.data();
    const i64* cross_a_p = cross_a_py.data();
    const i64* cross_b_p = cross_b_py.data();
    const f64* cross_v_p = cross_v_py.data();
    std::vector<RawSlice> same_slices((size_t)norb * norb, {nullptr,nullptr,nullptr,0});
    std::vector<RawSlice> cross_slices((size_t)norb * norb, {nullptr,nullptr,nullptr,0});
    for (i64 i = 0; i < norb; ++i) {
        for (i64 j = i + 1; j < norb; ++j) {  // same-spin: upper triangle only
            size_t off = (size_t)(i * norb + j);
            same_slices[off] = {same_a_p + (i64)off * m_same,
                                same_b_p + (i64)off * m_same,
                                same_v_p + (i64)off * m_same, m_same};
        }
        for (i64 j = 0; j < norb; ++j) {  // cross-spin: all (i, j)
            size_t off = (size_t)(i * norb + j);
            cross_slices[off] = {cross_a_p + (i64)off * m_cross,
                                 cross_b_p + (i64)off * m_cross,
                                 cross_v_p + (i64)off * m_cross, m_cross};
        }
    }

    // Raw pointers for addr tables — safe since arrays outlive this function.
    const i64* addr_a_p    = addr_table_a_py.data();
    const i64  addr_a_size = addr_table_a_py.shape(0);
    const i64* addr_b_p    = addr_table_b_py.data();
    const i64  addr_b_size = addr_table_b_py.shape(0);

    py::capsule cap2 = registry_cap.cast<py::capsule>();
    auto* rc2 = static_cast<RegistryCapsule*>(static_cast<void*>(cap2));
    const std::unordered_map<i64,i64>* map_a = use_dense_addr ? nullptr : &rc2->addr_map_a;
    const std::unordered_map<i64,i64>* map_b = use_dense_addr ? nullptr : &rc2->addr_map_b;

    // Snapshot EMPTY-holed psi into dense arrays
    i64 n_total = psi_keys.shape(0);
    i64 n_I = 0;
    for (i64 _i = 0; _i < n_total; ++_i)
        if (psi_keys(_i) != EMPTY) n_I++;

    std::vector<i64> I_idx_arr(n_I);
    std::vector<f64> cI_arr(n_I);
    {
        i64 p = 0;
        for (i64 _i = 0; _i < n_total; ++_i) {
            if (psi_keys(_i) != EMPTY) {
                I_idx_arr[p] = psi_keys(_i);
                cI_arr[p]    = psi_vals(_i);
                p++;
            }
        }
    }

    // Per-thread parallel accumulate.
#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif

    // Per-thread hash tables
    i64 n_cap = std::max(i64(64), i64(2) * n_I / n_threads);
    // round up to power of 2
    { i64 p = 1; while (p < n_cap * 2) p <<= 1; n_cap = p; }

    std::vector<HT> r_locals;
    r_locals.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t)
        r_locals.emplace_back(n_cap);

    // Parallelize over determinants (not over threads): each thread owns its
    // own r_locals[tid] hash table and scratch, but the idx-range is handed out
    // dynamically. Per-determinant work is highly non-uniform (it scales with
    // occ*virt and integral-table lengths), so a static split leaves threads
    // idle; schedule(dynamic) rebalances as heavy determinants are drawn.
    //
    // Chunk size is scaled to the core count and problem size rather than fixed:
    // we aim for ~8 chunks per thread. A fixed chunk misbehaves at the extremes
    // -- on many cores it either thrashes the shared work counter (chunk too
    // small relative to threads) or leaves cores idle on the tail (too few
    // chunks to redistribute). n_I/(n_threads*8) keeps grain coarse enough to
    // avoid counter contention yet fine enough that late heavy determinants can
    // still be rebalanced; clamp to >=1 for tiny n_I.
    // maybe_unused: without OpenMP the schedule() clause below is dropped, so
    // this variable would otherwise warn (and break a -Werror no-OpenMP build).
    [[maybe_unused]] i64 dyn_chunk = std::max<i64>(1, n_I / ((i64)n_threads * 8));

#pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        HT& ht = r_locals[tid];
        std::vector<f64> Sa(norb), Sb(norb), Ta(norb), Tb(norb);
        std::vector<int> occ_a, virt_a, occ_b, virt_b;
        occ_a.reserve(nelec_a); virt_a.reserve(norb - nelec_a);
        occ_b.reserve(nelec_b); virt_b.reserve(norb - nelec_b);

#pragma omp for schedule(dynamic, dyn_chunk)
        for (i64 idx = 0; idx < n_I; ++idx) {
            i64 I_idx = I_idx_arr[idx];
            f64 cI    = cI_arr[idx];

            // seed r <- psi
            ht.add(I_idx, cI);

            i64 ia    = I_idx / nb;
            i64 ib    = I_idx % nb;
            i64 ba0   = bits_a(ia);
            i64 bb0   = bits_b(ib);

            decode_bits(ba0, (int)norb, occ_a, virt_a);
            decode_bits(bb0, (int)norb, occ_b, virt_b);

            f64 E_I = use_dense_hdiag
                    ? hdiag_p[I_idx]
                    : hii_from_bits(ba0, bb0, hdiag1_p, jdiag_p, kdiag_p,
                                    (int)norb, (int)nelec_a, (int)nelec_b);
            if (!use_dense_hdiag)
                build_diag_ST(occ_a, occ_b, (int)norb, jdiag_p, kdiag_p,
                              Sa.data(), Sb.data(), Ta.data(), Tb.data());

            constexpr f64 floor = 1e-10;

            // ---- alpha singles ----
            for (int i : occ_a) {
                for (int a : virt_a) {
                    int lo = std::min(i,a), hi = std::max(i,a);
                    // parity of occupied orbitals strictly between lo and hi,
                    // via a bit-mask + popcount (O(1)); matches the scan
                    // `for x in occ if lo < x < hi` but avoids the inner loop.
                    i64 mask = ((i64(1)<<hi)-1) & ~((i64(1)<<(lo+1))-1);
                    f64 sign = (popcount64((uint64_t)(ba0&mask))%2==0) ? 1.0 : -1.0;
                    f64 e = h1e(a,i);
                    for (int k : occ_a) e += eri(a,i,k,k) - eri(a,k,k,i);
                    for (int k : occ_b) e += eri(a,i,k,k);
                    e *= sign;
                    f64 cIp = e * cI;
                    i64 nb_bits = (ba0 & ~(i64(1)<<i)) | (i64(1)<<a);
                    i64 na = addr_lookup(addr_a_p, addr_a_size, nb_bits, use_dense_addr, map_a);
                    if (na < 0) continue;
                    i64 new_flat = na * nb + ib;
                    f64 e_diag = use_dense_hdiag ? hdiag_p[new_flat]
                        : E_I + delta_single(i, a, hdiag1_p, jdiag_p, kdiag_p,
                                             Sa.data(), Tb.data(), (int)norb);
                    f64 denom = e_diag - (E - level_shift);
                    if (std::abs(denom) < trunc_eps) denom = denom>=0.0 ? trunc_eps : -trunc_eps;
                    cIp /= denom;
                    if (std::abs(cIp) > delta) ht.add(new_flat, cIp);
                }
            }

            // ---- beta singles ----
            for (int i : occ_b) {
                for (int a : virt_b) {
                    int lo = std::min(i,a), hi = std::max(i,a);
                    i64 mask = ((i64(1)<<hi)-1) & ~((i64(1)<<(lo+1))-1);
                    f64 sign = (popcount64((uint64_t)(bb0&mask))%2==0) ? 1.0 : -1.0;
                    f64 e = h1e(a,i);
                    for (int k : occ_b) e += eri(a,i,k,k) - eri(a,k,k,i);
                    for (int k : occ_a) e += eri(a,i,k,k);
                    e *= sign;
                    f64 cIp = e * cI;
                    i64 nb_bits = (bb0 & ~(i64(1)<<i)) | (i64(1)<<a);
                    i64 nb_addr = addr_lookup(addr_b_p, addr_b_size, nb_bits, use_dense_addr, map_b);
                    if (nb_addr < 0) continue;
                    i64 new_flat = ia * nb + nb_addr;
                    f64 e_diag = use_dense_hdiag ? hdiag_p[new_flat]
                        : E_I + delta_single(i, a, hdiag1_p, jdiag_p, kdiag_p,
                                             Sb.data(), Ta.data(), (int)norb);
                    f64 denom = e_diag - (E - level_shift);
                    if (std::abs(denom) < trunc_eps) denom = denom>=0.0 ? trunc_eps : -trunc_eps;
                    cIp /= denom;
                    if (std::abs(cIp) > delta) ht.add(new_flat, cIp);
                }
            }

            // ---- same-spin alpha-alpha ----
            for (int oi = 0; oi < (int)occ_a.size(); ++oi) {
                int i = occ_a[oi];
                for (int oj = oi+1; oj < (int)occ_a.size(); ++oj) {
                    int j = occ_a[oj];
                    const RawSlice& sl = same_slices[(size_t)i*norb+j];
                    for (i64 k = 0; k < sl.n; ++k) {
                        int a = (int)sl.a[k], b = (int)sl.b[k];
                        if ((ba0>>a)&i64(1)) continue;
                        if ((ba0>>b)&i64(1)) continue;
                        f64 raw_c = sl.v[k] * cI;
                        if (std::abs(raw_c) < delta) break;
                        int lo1 = std::min(i,a), hi1 = std::max(i,a);
                        i64 mask1 = ((i64(1)<<hi1)-1) & ~((i64(1)<<(lo1+1))-1);
                        int s1 = (popcount64((uint64_t)(ba0&mask1))%2==0)?1:-1;
                        i64 bits_mid = (ba0 & ~(i64(1)<<i)) | (i64(1)<<a);
                        int lo2 = std::min(j,b), hi2 = std::max(j,b);
                        i64 mask2 = ((i64(1)<<hi2)-1) & ~((i64(1)<<(lo2+1))-1);
                        int s2 = (popcount64((uint64_t)(bits_mid&mask2))%2==0)?1:-1;
                        f64 cIp = (f64)(s1*s2) * sl.v[k] * cI;
                        i64 nb_bits = (ba0 & ~(i64(1)<<i) & ~(i64(1)<<j)) | (i64(1)<<a) | (i64(1)<<b);
                        i64 na = addr_lookup(addr_a_p, addr_a_size, nb_bits, use_dense_addr, map_a);
                        if (na < 0) continue;
                        i64 new_flat = na * nb + ib;
                        f64 e_diag = use_dense_hdiag ? hdiag_p[new_flat]
                            : E_I + delta_double_same_spin(i, j, a, b, hdiag1_p, jdiag_p, kdiag_p,
                                                           Sa.data(), Tb.data(), (int)norb);
                        f64 denom = e_diag - E;
                        if (std::abs(denom) < floor) denom = denom>=0.0 ? floor : -floor;
                        cIp /= denom;
                        if (std::abs(cIp) > delta) ht.add(new_flat, cIp);
                    }
                }
            }

            // ---- same-spin beta-beta ----
            for (int oi = 0; oi < (int)occ_b.size(); ++oi) {
                int i = occ_b[oi];
                for (int oj = oi+1; oj < (int)occ_b.size(); ++oj) {
                    int j = occ_b[oj];
                    const RawSlice& sl = same_slices[(size_t)i*norb+j];
                    for (i64 k = 0; k < sl.n; ++k) {
                        int a = (int)sl.a[k], b = (int)sl.b[k];
                        if ((bb0>>a)&i64(1)) continue;
                        if ((bb0>>b)&i64(1)) continue;
                        f64 raw_c = sl.v[k] * cI;
                        if (std::abs(raw_c) < delta) break;
                        int lo1 = std::min(i,a), hi1 = std::max(i,a);
                        i64 mask1 = ((i64(1)<<hi1)-1) & ~((i64(1)<<(lo1+1))-1);
                        int s1 = (popcount64((uint64_t)(bb0&mask1))%2==0)?1:-1;
                        i64 bits_mid = (bb0 & ~(i64(1)<<i)) | (i64(1)<<a);
                        int lo2 = std::min(j,b), hi2 = std::max(j,b);
                        i64 mask2 = ((i64(1)<<hi2)-1) & ~((i64(1)<<(lo2+1))-1);
                        int s2 = (popcount64((uint64_t)(bits_mid&mask2))%2==0)?1:-1;
                        f64 cIp = (f64)(s1*s2) * sl.v[k] * cI;
                        i64 nb_bits = (bb0 & ~(i64(1)<<i) & ~(i64(1)<<j)) | (i64(1)<<a) | (i64(1)<<b);
                        i64 nb_addr = addr_lookup(addr_b_p, addr_b_size, nb_bits, use_dense_addr, map_b);
                        if (nb_addr < 0) continue;
                        i64 new_flat = ia * nb + nb_addr;
                        f64 e_diag = use_dense_hdiag ? hdiag_p[new_flat]
                            : E_I + delta_double_same_spin(i, j, a, b, hdiag1_p, jdiag_p, kdiag_p,
                                                           Sb.data(), Ta.data(), (int)norb);
                        f64 denom = e_diag - E;
                        if (std::abs(denom) < floor) denom = denom>=0.0 ? floor : -floor;
                        cIp /= denom;
                        if (std::abs(cIp) > delta) ht.add(new_flat, cIp);
                    }
                }
            }

            // ---- cross alpha-beta ----
            for (int i : occ_a) {
                for (int j : occ_b) {
                    const RawSlice& sl = cross_slices[(size_t)i*norb+j];
                    for (i64 k = 0; k < sl.n; ++k) {
                        int a = (int)sl.a[k], b = (int)sl.b[k];
                        if ((ba0>>a)&i64(1)) continue;
                        if ((bb0>>b)&i64(1)) continue;
                        f64 raw_c = sl.v[k] * cI;
                        if (std::abs(raw_c) < delta) break;
                        int lo1 = std::min(i,a), hi1 = std::max(i,a);
                        i64 mask1 = ((i64(1)<<hi1)-1) & ~((i64(1)<<(lo1+1))-1);
                        int s1 = (popcount64((uint64_t)(ba0&mask1))%2==0)?1:-1;
                        int lo2 = std::min(j,b), hi2 = std::max(j,b);
                        i64 mask2 = ((i64(1)<<hi2)-1) & ~((i64(1)<<(lo2+1))-1);
                        int s2 = (popcount64((uint64_t)(bb0&mask2))%2==0)?1:-1;
                        f64 cIp = (f64)(s1*s2) * sl.v[k] * cI;
                        i64 new_ba = (ba0 & ~(i64(1)<<i)) | (i64(1)<<a);
                        i64 new_bb = (bb0 & ~(i64(1)<<j)) | (i64(1)<<b);
                        i64 na_ = addr_lookup(addr_a_p, addr_a_size, new_ba, use_dense_addr, map_a);
                        i64 nb_ = addr_lookup(addr_b_p, addr_b_size, new_bb, use_dense_addr, map_b);
                        if (na_ < 0 || nb_ < 0) continue;
                        i64 new_flat = na_ * nb + nb_;
                        f64 e_diag = use_dense_hdiag ? hdiag_p[new_flat]
                            : ediag(false, new_flat, new_ba, new_bb,
                                    hdiag_p, hdiag1_p, jdiag_p, kdiag_p,
                                    (int)norb, (int)nelec_a, (int)nelec_b);
                        f64 denom = e_diag - E;
                        if (std::abs(denom) < floor) denom = denom>=0.0 ? floor : -floor;
                        cIp /= denom;
                        if (std::abs(cIp) > delta) ht.add(new_flat, cIp);
                    }
                }
            }
        } // end idx
    } // end parallel

    // Serial merge
    i64 total = 0;
    for (int t = 0; t < n_threads; ++t) total += r_locals[t].cnt;

    HT out_ht(total);
    for (int t = 0; t < n_threads; ++t) {
        HT& ht = r_locals[t];
        i64 cap = (i64)ht.keys.size();
        for (i64 i = 0; i < cap; ++i) {
            i64 k = ht.keys[i];
            if (k != EMPTY) out_ht.add(k, ht.vals[i]);
        }
    }

    // Pack into numpy arrays
    i64 out_cap = (i64)out_ht.keys.size();
    AI out_keys_py((py::ssize_t)out_cap);
    AF out_vals_py((py::ssize_t)out_cap);
    auto ok = out_keys_py.mutable_unchecked<1>();
    auto ov = out_vals_py.mutable_unchecked<1>();
    for (i64 i = 0; i < out_cap; ++i) {
        ok(i) = out_ht.keys[i];
        ov(i) = out_ht.vals[i];
    }
    return py::make_tuple(out_keys_py, out_vals_py);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(kernels_cpp, m) {
    m.doc() = "C++/OpenMP kernels for the truncated Davidson solver";

    m.def("make_registry", &make_registry,
          "Create an opaque C++ registry capsule.");

    m.def("build_addr_maps", &build_addr_maps,
          py::arg("registry"), py::arg("bits_a"), py::arg("bits_b"),
          "Populate the registry's hash maps from sorted CI string arrays (norb > 24 path).");

    m.def("add_basis_vector_fused", &add_basis_vector_fused_py,
          py::arg("new_idx"), py::arg("new_val"), py::arg("n_new"),
          py::arg("registry"),
          py::arg("norb"), py::arg("nb"),
          py::arg("nelec_a"), py::arg("nelec_b"),
          py::arg("bits_a"), py::arg("bits_b"),
          py::arg("addr_table_a"), py::arg("addr_table_b"),
          py::arg("use_dense_addr"),
          py::arg("h1e"), py::arg("eri"),
          py::arg("use_dense_hdiag"),
          py::arg("hdiag"), py::arg("hdiag1"), py::arg("jdiag"), py::arg("kdiag"),
          py::arg("col"));

    m.def("calc_trunc_pcr_fused", &calc_trunc_pcr_fused_py,
          py::arg("psi_keys"), py::arg("psi_vals"),
          py::arg("norb"), py::arg("nb"), py::arg("nelec_a"), py::arg("nelec_b"),
          py::arg("bits_a"), py::arg("bits_b"),
          py::arg("addr_table_a"), py::arg("addr_table_b"),
          py::arg("use_dense_addr"),
          py::arg("registry"),
          py::arg("h1e"), py::arg("eri"),
          py::arg("use_dense_hdiag"),
          py::arg("hdiag"), py::arg("hdiag1"), py::arg("jdiag"), py::arg("kdiag"),
          py::arg("same_a"), py::arg("same_b"), py::arg("same_v"),
          py::arg("cross_a"), py::arg("cross_b"), py::arg("cross_v"),
          py::arg("E"), py::arg("delta"),
          py::arg("level_shift") = 1e-3,
          py::arg("trunc_eps") = 1e-8);

    m.def("singles_kernel", &singles_kernel_py,
          py::arg("occ"), py::arg("virt"), py::arg("occ_other"),
          py::arg("bits0"), py::arg("h1e"), py::arg("eri"));

    m.def("doubles_same_spin_kernel", &doubles_same_spin_kernel_py,
          py::arg("occ"), py::arg("virt"), py::arg("bits0"), py::arg("eri"));

    m.def("doubles_cross_spin_kernel", &doubles_cross_spin_kernel_py,
          py::arg("occ_a"), py::arg("virt_a"), py::arg("occ_b"), py::arg("virt_b"),
          py::arg("bits_a0"), py::arg("bits_b0"), py::arg("eri"));

    m.attr("EMPTY") = py::int_(EMPTY);
}
