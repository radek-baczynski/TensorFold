// Decode (tree) attention for Turing (sm_75): the fp16 m16n8k8 tensor core.
//
// Two kernels share one tile step, so a key gets the same bits whether a chunk of committed keys is folded by
// `decode_chunks` or the chunk holding the stream's end and a row's own path is folded by `decode_tail`:
//   scores = Q K^T over 16 keys (A: the pair tile's Q fragments, B: K), the online softmax of the (o, m, l) state with
//   masked keys at -inf, then O += P V (A: P straight from the score fragments, B: V).
// A pair is a window row and one query head of the KV head's group. A pair's bits depend on its own query and keys
// only, never on the other pairs of its tile. The partials (o, m, l) are stored as the Triton kernels stored them, so
// `_merge` reads them unchanged.
//
// decode_chunks: a block is one (stream, chunk, KV head); its warps take the stream's 16-pair tiles and read the
// chunk's keys and values from one shared-memory copy, staged as fp16 a 16-key tile at a time while the next tile's
// global loads are in flight.
// decode_tail: a warp is one (row, KV head, tail chunk) and loads its fragments from global memory.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

constexpr int D = 256;                  // head dim
constexpr int CH = 512;                 // keys a chunk
constexpr int KT = 16;                  // keys a tile
constexpr int NWARP = 6;                // 16-pair tiles a chunks block runs at once
constexpr int THREADS = NWARP * 32;
constexpr int ROWH = D + 8;             // halves a staged row: 528 bytes, so eight rows hit eight bank groups
constexpr int KV_BYTES = KT * ROWH * 2;
constexpr int TILES = CH / KT;
constexpr int TASKS = 2 * KT * (D / 8); // 16-byte pieces of a K and a V tile
constexpr int PER_THREAD = (TASKS + THREADS - 1) / THREADS;
constexpr int MAX_NODES = 128;          // rows a window path holds (attention.py MAX_NODES)
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t half2_bits(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

struct Fold {
    float o[D / 8][4];                  // C fragments: (row g, cols 2t, 2t+1) and (row g + 8, ...) of each 8 dims
    float m0, m1, l0, l1;               // running max (log2 domain) and this lane's share of the sum, rows g and g + 8
    __device__ void clear() {
#pragma unroll
        for (int j = 0; j < D / 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[j][e] = 0.0f;
        m0 = m1 = -INFINITY;
        l0 = l1 = 0.0f;
    }
};

// One 16-key tile into ``f``. ``vmask`` bit k: key k of the tile exists (the keys of a tile that does not are 0 in
// kfrag and vfrag). ``kfrag(nt, j0, b)``: the four k8 B fragments (dims 32 * j0 ..) of key tile nt;
// ``vfrag(kk, n0, b)``: the four n8 B fragments (dims 32 * n0 ..) of key step kk.
template <class KF, class VF>
__device__ __forceinline__ void tile_step(Fold& f, const uint32_t (&qf)[D / 8][2], float k2, uint32_t vmask, int t,
                                          KF&& kfrag, VF&& vfrag) {
    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[nt][e] = 0.0f;
#pragma unroll
        for (int j0 = 0; j0 < D / 32; ++j0) {
            uint32_t b[4];
            kfrag(nt, j0, b);
#pragma unroll
            for (int i = 0; i < 4; ++i) mma_k8(sc[nt], qf[j0 * 4 + i][0], qf[j0 * 4 + i][1], b[i]);
        }
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
            sc[nt][e] = ((vmask >> (nt * 8 + 2 * t + (e & 1))) & 1u) ? sc[nt][e] * k2 : -INFINITY;
        mx0 = fmaxf(mx0, fmaxf(sc[nt][0], sc[nt][1]));
        mx1 = fmaxf(mx1, fmaxf(sc[nt][2], sc[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float n0 = fmaxf(f.m0, mx0), n1 = fmaxf(f.m1, mx1);      // finite: a tile has at least its first key
    const float a0 = ex2(f.m0 - n0), a1 = ex2(f.m1 - n1);
    f.m0 = n0;
    f.m1 = n1;
    uint32_t pa[2][2];
    float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const float p00 = ex2(sc[nt][0] - n0), p01 = ex2(sc[nt][1] - n0);
        const float p10 = ex2(sc[nt][2] - n1), p11 = ex2(sc[nt][3] - n1);
        ps0 += p00 + p01;
        ps1 += p10 + p11;
        pa[nt][0] = half2_bits(p00, p01);
        pa[nt][1] = half2_bits(p10, p11);
    }
    f.l0 = f.l0 * a0 + ps0;                           // each lane keeps its own columns' sum; the quad adds at the end
    f.l1 = f.l1 * a1 + ps1;
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
        f.o[j][0] *= a0; f.o[j][1] *= a0; f.o[j][2] *= a1; f.o[j][3] *= a1;
    }
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
        for (int n0g = 0; n0g < D / 32; ++n0g) {
            uint32_t b[4];
            vfrag(kk, n0g, b);
#pragma unroll
            for (int i = 0; i < 4; ++i) mma_k8(f.o[n0g * 4 + i], pa[kk][0], pa[kk][1], b[i]);
        }
    }
}

// The pair tile's partial (o, m, l) to the Triton kernels' layout; pairs past ``valid`` store nothing.
template <int H>
__device__ __forceinline__ void store_partial(Fold& f, float* po, float* pm, float* pl, int lane, const size_t (&base)[2],
                                              const bool (&ok)[2]) {
    const int t = lane & 3;
    f.l0 += __shfl_xor_sync(0xffffffffu, f.l0, 1);
    f.l0 += __shfl_xor_sync(0xffffffffu, f.l0, 2);
    f.l1 += __shfl_xor_sync(0xffffffffu, f.l1, 1);
    f.l1 += __shfl_xor_sync(0xffffffffu, f.l1, 2);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        if (!ok[h]) continue;
        float* dst = po + base[h] * D + 2 * t;
#pragma unroll
        for (int j = 0; j < D / 8; ++j)
            *reinterpret_cast<float2*>(dst + 8 * j) = make_float2(f.o[j][2 * h], f.o[j][2 * h + 1]);
        if (t == 0) {
            pm[base[h]] = (h ? f.m1 : f.m0) / LOG2E;  // natural-log domain, as `_merge` reads it
            pl[base[h]] = h ? f.l1 : f.l0;
        }
    }
}

template <int H, int HK, int G>
__global__ void __launch_bounds__(THREADS, 1) decode_chunks_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ origin,
        const int64_t* __restrict__ off, const int* __restrict__ stream, const int* __restrict__ items,
        float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int W, float scale) {
    extern __shared__ __align__(128) unsigned char smem_buf[];
    const int item = blockIdx.x / HK, hk = blockIdx.x % HK;
    const int s = items[item * 3], first = items[item * 3 + 1], chunk = items[item * 3 + 2];
    if (first != 0) return;                           // the stream's first query tile leads its other tiles
    const int start = stream[s * 4], rows = stream[s * 4 + 1], p = stream[s * 4 + 2];
    if ((chunk + 1) * CH > p) return;                 // a plan padded for a longer context skips missing chunks
    const int pairs = rows * G;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const __nv_bfloat16* kbase = origin + off[s * 2];
    const __nv_bfloat16* vbase = origin + off[s * 2 + 1];
    const float k2 = scale * LOG2E;                   // scores in the log2 domain

    uint4 stage[PER_THREAD];
    auto fetch = [&](int tile) {
#pragma unroll
        for (int k = 0; k < PER_THREAD; ++k) {
            const int c = tid + k * THREADS;
            stage[k] = make_uint4(0, 0, 0, 0);
            if (c < TASKS) {
                const int which = c / (TASKS / 2), rem = c % (TASKS / 2);
                const int key = rem / (D / 8), ch = rem % (D / 8);
                const __nv_bfloat16* src = (which ? vbase : kbase) +
                    (static_cast<size_t>(chunk * CH + tile * KT + key) * HK + hk) * D + ch * 8;
                stage[k] = __ldg(reinterpret_cast<const uint4*>(src));
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int k = 0; k < PER_THREAD; ++k) {
            const int c = tid + k * THREADS;
            if (c < TASKS) {
                const int which = c / (TASKS / 2), rem = c % (TASKS / 2);
                const int key = rem / (D / 8), ch = rem % (D / 8);
                *reinterpret_cast<uint4*>(smem_buf + (buf * 2 + which) * KV_BYTES + key * ROWH * 2 + ch * 16) =
                    bf16x8_to_fp16(stage[k]);
            }
        }
    };

    const int rounds = (pairs + NWARP * 16 - 1) / (NWARP * 16);    // windows past 96 pairs take the chunk again
    for (int round = 0; round < rounds; ++round) {
        const int tile_first = (round * NWARP + warp) * 16;
        // this warp's query fragments: A of m16k8 is (row g, k 2t..2t+1) and (row g + 8, ...), fp16
        uint32_t qf[D / 8][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rr = tile_first + g + h * 8;
            const bool ok = rr < pairs;
            const int node = start + rr / G, head = hk * G + rr % G;
            const __nv_bfloat16* src = q + (static_cast<size_t>(node) * H + head) * D + 2 * t;
#pragma unroll
            for (int j = 0; j < D / 8; ++j)
                qf[j][h] = ok ? bf16_to_fp16(*reinterpret_cast<const uint32_t*>(src + 8 * j)) : 0u;
        }
        Fold f;
        f.clear();
        __syncthreads();                              // the previous round's last reads are done
        fetch(0);
        store(0);
        __syncthreads();
        for (int tile = 0; tile < TILES; ++tile) {
            const int buf = tile & 1;
            if (tile + 1 < TILES) fetch(tile + 1);
            if (tile_first < pairs) {
                const unsigned char* ks = smem_buf + buf * 2 * KV_BYTES;
                const unsigned char* vs = ks + KV_BYTES;
                tile_step(f, qf, k2, 0xFFFFu, t,
                          [&](int nt, int j0, uint32_t (&b)[4]) {
                              ldmatrix4(b, ks + (nt * 8 + (lane & 7)) * ROWH * 2 + (j0 * 4 + (lane >> 3)) * 16);
                          },
                          [&](int kk, int n0g, uint32_t (&b)[4]) {
                              ldmatrix4t(b, vs + (kk * 8 + (lane & 7)) * ROWH * 2 + (n0g * 4 + (lane >> 3)) * 16);
                          });
            }
            if (tile + 1 < TILES) store((tile + 1) & 1);
            __syncthreads();
        }
        if (tile_first < pairs) {
            size_t base[2];
            bool ok[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int rr = tile_first + g + h * 8;
                ok[h] = rr < pairs;
                base[h] = (static_cast<size_t>(chunk) * W + start + rr / G) * H + hk * G + rr % G;
            }
            store_partial<H>(f, po, pm, pl, lane, base, ok);
        }
    }
}

// A warp: one row's G query heads against one chunk holding its stream's end (committed keys then the row's path).
template <int H, int HK, int G>
__global__ void __launch_bounds__(32, 8) decode_tail_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kn, const __nv_bfloat16* __restrict__ vn,
        const __nv_bfloat16* __restrict__ origin, const int64_t* __restrict__ off, const int* __restrict__ stream,
        const int* __restrict__ rows, const int* __restrict__ paths, const int* __restrict__ depths,
        float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int W, float scale) {
    const int node = blockIdx.x, hk = blockIdx.y;
    const int s = rows[node];
    const int p = stream[s * 4 + 2], nch = stream[s * 4 + 3];
    const int chunk = p / CH + blockIdx.z;
    if (chunk >= nch) return;
    const int depth = depths[node];
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    const __nv_bfloat16* kbase = origin + off[s * 2];
    const __nv_bfloat16* vbase = origin + off[s * 2 + 1];
    const float k2 = scale * LOG2E;

    uint32_t qf[D / 8][2];
    size_t base[2];
    bool ok[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        ok[h] = g + h * 8 < G;
        const int head = hk * G + (ok[h] ? g + h * 8 : 0);
        base[h] = (static_cast<size_t>(chunk) * W + node) * H + head;
        const __nv_bfloat16* src = q + (static_cast<size_t>(node) * H + head) * D + 2 * t;
#pragma unroll
        for (int j = 0; j < D / 8; ++j)
            qf[j][h] = ok[h] ? bf16_to_fp16(*reinterpret_cast<const uint32_t*>(src + 8 * j)) : 0u;
    }
    Fold f;
    f.clear();
    const int total = p + depth;                      // keys the row sees: committed, then its path
    const int lo = chunk * CH;
    const int valid = min(total, lo + CH) - lo;
    const int ntiles = valid > 0 ? (valid + KT - 1) / KT : 0;
    for (int tile = 0; tile < ntiles; ++tile) {
        const int tbase = lo + tile * KT;
        const int nv = min(KT, total - tbase);
        // the row of key ``local`` of this tile in the committed cache or among the row's path nodes (null: no key)
        auto row_of = [&](const __nv_bfloat16* cache, const __nv_bfloat16* nodes, int local) -> const __nv_bfloat16* {
            if (local >= nv) return nullptr;
            const int pos = tbase + local;
            if (pos < p) return cache + (static_cast<size_t>(pos) * HK + hk) * D;
            return nodes + (static_cast<size_t>(paths[node * MAX_NODES + pos - p]) * HK + hk) * D;
        };
        const __nv_bfloat16* krow[2] = {row_of(kbase, kn, g), row_of(kbase, kn, 8 + g)};
        const __nv_bfloat16* vrow[2][2] = {{row_of(vbase, vn, 2 * t), row_of(vbase, vn, 2 * t + 1)},
                                           {row_of(vbase, vn, 8 + 2 * t), row_of(vbase, vn, 8 + 2 * t + 1)}};
        tile_step(f, qf, k2, nv >= KT ? 0xFFFFu : (1u << nv) - 1u, t,
                  [&](int nt, int j0, uint32_t (&b)[4]) {
#pragma unroll
                      for (int i = 0; i < 4; ++i)
                          b[i] = krow[nt] ? bf16_to_fp16(__ldg(reinterpret_cast<const uint32_t*>(
                                                krow[nt] + (j0 * 4 + i) * 8 + 2 * t))) : 0u;
                  },
                  [&](int kk, int n0g, uint32_t (&b)[4]) {
#pragma unroll
                      for (int i = 0; i < 4; ++i) {
                          const int d = (n0g * 4 + i) * 8 + g;
                          const float lo_v = vrow[kk][0] ? __bfloat162float(__ldg(vrow[kk][0] + d)) : 0.0f;
                          const float hi_v = vrow[kk][1] ? __bfloat162float(__ldg(vrow[kk][1] + d)) : 0.0f;
                          b[i] = half2_bits(lo_v, hi_v);
                      }
                  });
    }
    store_partial<H>(f, po, pm, pl, lane, base, ok);
}

} // namespace

// Whole-chunk partials for Turing. q (W, H, 256) bf16; ``origin`` the cache offsets' base; ``off`` (S, 2) int64 element
// offsets of each stream's K and V; ``stream`` (S, 4) and ``items`` (n, 3) int32 as `Plan` holds them.
void decode_chunks_cuda(const at::Tensor& q, const at::Tensor& origin, const at::Tensor& off, const at::Tensor& stream,
                        const at::Tensor& items, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int hk, int group,
                        float scale) {
    const int W = q.size(0), H = q.size(1), n = items.size(0);
    if (n == 0) return;
    constexpr int SMEM = 4 * KV_BYTES;
    TORCH_CHECK(H == 24 && hk == 4 && group == 6, "decode_chunks: 24 query heads over 4 KV heads");
    auto kernel = decode_chunks_kernel<24, 4, 6>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        configured = true;
    }
    kernel<<<n * hk, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(origin.data_ptr()),
        off.data_ptr<int64_t>(), stream.data_ptr<int>(), items.data_ptr<int>(), po.data_ptr<float>(),
        pm.data_ptr<float>(), pl.data_ptr<float>(), W, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The chunks holding each stream's end: committed keys, then each row's own path (``paths``, ``depths`` as `Plan`).
void decode_tail_cuda(const at::Tensor& q, const at::Tensor& kn, const at::Tensor& vn, const at::Tensor& origin,
                      const at::Tensor& off, const at::Tensor& stream, const at::Tensor& rows, const at::Tensor& paths,
                      const at::Tensor& depths, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int hk, int group,
                      float scale, int tails) {
    const int W = q.size(0), H = q.size(1);
    TORCH_CHECK(H == 24 && hk == 4 && group == 6, "decode_tail: 24 query heads over 4 KV heads");
    TORCH_CHECK(paths.size(1) == MAX_NODES, "paths: (W, 128) int32");
    decode_tail_kernel<24, 4, 6><<<dim3(W, hk, tails), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(kn.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(vn.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(origin.data_ptr()),
        off.data_ptr<int64_t>(), stream.data_ptr<int>(), rows.data_ptr<int>(), paths.data_ptr<int>(),
        depths.data_ptr<int>(), po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), W, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
