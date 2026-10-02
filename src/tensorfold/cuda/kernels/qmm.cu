// Lane matmul: acc = fma(xs, b, fma(P, s, acc)) per group in order, K slices set by shape, so no row affects another.

#include <ATen/ATen.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

// Tile shapes: BM rows by BN columns a block, WM x WN warps, each warp (BM / WM) x (BN / WN).
template <int GS, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;               // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;                // n8 tiles a warp
    static constexpr int ROW = GS * 2;                    // bytes of one input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;                    // stage bytes: inputs,
    static constexpr int W = BN * GS / 2;                 // weights,
    static constexpr int S = BN * 2;                      // scales, biases (bf16),
    static constexpr int XS = BM * 4;                     // and input sums (fp32)
    static constexpr int STAGE = X + W + 2 * S + XS;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;   // a K slice's partial, parked for the cluster sum
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool PIPE = false>
__global__ void __launch_bounds__(WM * WN * 32) qmm_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
        void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx, int group) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, per = KG / SK;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y, slice = blockIdx.z, g0 = slice * per;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // one group's inputs, weights, scales, biases and input sums into stage s
    auto load = [&](int s, int g, int m0) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z_act(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                      m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = 64 * GS / 2;           // one stored 64-column tile's group block
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
            cp16(ps + which * T::S + off * 16, src);
        }
        float* px = reinterpret_cast<float*>(ps + 2 * T::S);
        for (int r = tid; r < BM; r += T::THREADS)
            cp4(px + r, xs + static_cast<size_t>(min(m0 + r, M - 1)) * KG + g);
    };

    {
        float acc[T::MT][T::NT][4];
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
        // acc = fma(xs, b, fma(p, s, acc)) per output, the order every row gets
        auto epilogue = [&](float (&acc_)[T::MT][T::NT][4], const float (&d_)[T::MT][T::NT][4],
                            const float (&s_)[T::NT][2], const float (&b_)[T::NT][2], const float (&x_)[T::MT][2]) {
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc_[i][j][e] = __fmaf_rn(x_[i][e >> 1], b_[j][e & 1], __fmaf_rn(d_[i][j][e], s_[j][e & 1],
                                                                                       acc_[i][j][e]));
        };
        float dp[T::MT][T::NT][4], sp[T::NT][2], bp[T::NT][2], xp[T::MT][2];
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load(s, g0 + s, m0);
            commit();
        }
        for (int it = 0; it < per; ++it) {
            wait<STAGES - 2>();
            __syncthreads();
            const int next = it + STAGES - 1;
            if (next < per) load(next % STAGES, g0 + next, m0);
            commit();
            const unsigned char* p = stage(it % STAGES);
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
            const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
            const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
            uint32_t words[T::NT][GS / 32];
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int v = 0; v < GS / 32; ++v)
                    words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            float d[T::MT][T::NT][4];
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt) {
                uint32_t a[T::MT][4];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    const uint32_t b0 = pair_act(words[j][kt / 2], (kt & 1) * 8);
                    const uint32_t b1 = pair_act(words[j][kt / 2], (kt & 1) * 8 + 4);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) {
                        if (kt == 0) mma0_act(d[i][j], a[i], b0, b1);
                        else mma_act(d[i][j], a[i], b0, b1);
                    }
                }
                if (PIPE && kt == 0 && it > 0) epilogue(acc, dp, sp, bp, xp);      // the previous group's, between MMAs
            }
            float sv[T::NT][2], bv[T::NT][2], xv[T::MT][2];
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                sv[j][0] = __low2float(s2); sv[j][1] = __high2float(s2);
                bv[j][0] = __low2float(b2); bv[j][1] = __high2float(b2);
            }
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                xv[i][0] = px[row];
                xv[i][1] = px[row + 8];
            }
            if constexpr (PIPE) {
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) dp[i][j][e] = d[i][j][e];
                    xp[i][0] = xv[i][0]; xp[i][1] = xv[i][1];
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    sp[j][0] = sv[j][0]; sp[j][1] = sv[j][1]; bp[j][0] = bv[j][0]; bp[j][1] = bv[j][1];
                }
            } else {
                epilogue(acc, d, sv, bv, xv);
            }
        }
        if (PIPE && per > 0) epilogue(acc, dp, sp, bp, xp);
        wait<0>();
        __syncthreads();
        if constexpr (CLUSTER) {
#if __CUDA_ARCH__ < 900
            __trap();                                     // no clusters before sm_90: the host never launches this
#else
            // K slices of a tile form a cluster: slice 0 adds peers' partials in slice order, as reduce_kernel does
            constexpr int E = T::MT * T::NT * 4;
            auto cluster = cooperative_groups::this_cluster();
            float* mine = reinterpret_cast<float*>(buf);
            if (slice != 0) {
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) mine[((i * T::NT + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
            }
            cluster.sync();
            if (slice == 0) {
                for (int peer = 1; peer < SK; ++peer) {
                    const float* theirs = cluster.map_shared_rank(mine, peer);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i)
#pragma unroll
                        for (int j = 0; j < T::NT; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[i][j][e] = acc[i][j][e] + theirs[((i * T::NT + j) * 4 + e) * T::THREADS + tid];
                }
            }
            cluster.sync();                               // peers keep their memory until slice 0 has read it
            if (slice != 0) return;
#endif
        }
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                    if (row >= M) continue;
                    const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                    if (SK > 1 && !CLUSTER) {
                        float* dst = part + (static_cast<size_t>(slice) * M + row) * N + col;
                        if (col < N) dst[0] = v0;
                        if (col + 1 < N) dst[1] = v1;
                    } else if (F32) {
                        float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                        if (col < N) dst[0] = v0;
                        if (col + 1 < N) dst[1] = v1;
                    } else {
                        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                        if (col + 1 < N && (N & 1) == 0)
                            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                        else {
                            if (col < N) dst[0] = __float2bfloat16_rn(v0);
                            if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                        }
                    }
                }
            }
    }
}

// Turing's decode matmul. Without cp.async every staged byte stalls its warp, and a weight is used once, so the weights
// skip shared memory: each lane reads its own B-fragment words (the pack is stored in lane order) straight into
// registers a batch of U groups ahead. Shared memory holds only the inputs (fp16, double-buffered by batch).
// The arithmetic is qmm_kernel's, group by group: P by mma from zero, then acc = fma(xs, b, fma(P, s, acc)).
template <int BM, int WN, int U, bool F32>
__global__ void __launch_bounds__(WN * 32) qmm_stream_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const unsigned char* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
        void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx, int group) {
    constexpr int GS = 64, BN = WN * 16, NT = 2, MT = BM / 16, THREADS = WN * 32;
    constexpr int ROW = GS * 2, CHUNKS = ROW / 16;
    constexpr int XCHUNKS = U * BM * CHUNKS, XPT = (XCHUNKS + THREADS - 1) / THREADS;
    constexpr int XBYTES = U * BM * ROW;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / GS, per = KG / SK;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y, g0 = blockIdx.z * per;
    const int nb = (per + U - 1) / U;

    struct Regs {
        uint2 words[NT];
        uint32_t sv[NT], bv[NT];
        float xv[MT][2];
    };
    Regs A[U], B[U];

    auto fetch = [&](Regs (&dst)[U], int b) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int rel = b * U + u;
            if (rel >= per) continue;
            const int g = g0 + rel;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int nt = n0 / 8 + wn * NT + j;                  // this n8 tile, counted from column 0
                const size_t at64 = static_cast<size_t>(nt / 8) * KG + g;
                dst[u].words[j] = __ldg(reinterpret_cast<const uint2*>(w + at64 * 2048 + ((nt % 8) * 32 + lane) * 8));
                const int col = n0 + wn * 16 + j * 8 + (lane & 3) * 2;
                dst[u].sv[j] = __ldg(reinterpret_cast<const uint32_t*>(scales + static_cast<size_t>(g) * npad + col));
                dst[u].bv[j] = __ldg(reinterpret_cast<const uint32_t*>(biases + static_cast<size_t>(g) * npad + col));
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int row = i * 16 + (lane >> 2);
                dst[u].xv[i][0] = __ldg(xs + static_cast<size_t>(min(m0 + row, M - 1)) * KG + g);
                dst[u].xv[i][1] = __ldg(xs + static_cast<size_t>(min(m0 + row + 8, M - 1)) * KG + g);
            }
        }
    };
    uint4 xr[XPT];
    auto fetch_x = [&](int b) {
#pragma unroll
        for (int k = 0; k < XPT; ++k) {
            const int c = tid + k * THREADS;
            xr[k] = make_uint4(0, 0, 0, 0);
            if (c < XCHUNKS) {
                const int u = c / (BM * CHUNKS), r = (c % (BM * CHUNKS)) / CHUNKS, ch = c % CHUNKS;
                const int rel = b * U + u;
                if (rel < per && m0 + r < M)
                    xr[k] = bf16x8_to_fp16(*reinterpret_cast<const uint4*>(
                        x + static_cast<size_t>(m0 + r) * ldx + (g0 + rel) * GS + ch * 8));
            }
        }
    };
    auto store_x = [&](int which) {
#pragma unroll
        for (int k = 0; k < XPT; ++k) {
            const int c = tid + k * THREADS;
            if (c < XCHUNKS) {
                const int u = c / (BM * CHUNKS), r = (c % (BM * CHUNKS)) / CHUNKS, ch = c % CHUNKS;
                *reinterpret_cast<uint4*>(buf + which * XBYTES + u * (BM * ROW) + r * ROW + swz<CHUNKS>(r, ch) * 16) = xr[k];
            }
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    auto compute = [&](const Regs (&src)[U], int b, int which) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (b * U + u >= per) continue;
            const unsigned char* p = buf + which * XBYTES + u * (BM * ROW);
            float d[MT][NT][4];
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt) {
                uint32_t a[MT][4];
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    ldmatrix4(a[i], p + r * ROW + swz<CHUNKS>(r, kt * 2 + (lane >> 4)) * 16);
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const uint32_t word = (kt / 2) ? src[u].words[j].y : src[u].words[j].x;
                    const uint32_t b0 = pair_act(word, (kt & 1) * 8);
                    const uint32_t b1 = pair_act(word, (kt & 1) * 8 + 4);
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (kt == 0) mma0_act(d[i][j], a[i], b0, b1);
                        else mma_act(d[i][j], a[i], b0, b1);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const float s0 = __uint_as_float(src[u].sv[j] << 16), s1 = __uint_as_float(src[u].sv[j] & 0xffff0000u);
                const float b0 = __uint_as_float(src[u].bv[j] << 16), b1 = __uint_as_float(src[u].bv[j] & 0xffff0000u);
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc[i][j][e] = __fmaf_rn(src[u].xv[i][e >> 1], (e & 1) ? b1 : b0,
                                                 __fmaf_rn(d[i][j][e], (e & 1) ? s1 : s0, acc[i][j][e]));
            }
        }
    };

    if (nb > 0) {
        fetch(A, 0);
        fetch_x(0);
        store_x(0);
        __syncthreads();
        for (int b = 0; b < nb; b += 2) {
            const bool more = b + 1 < nb;
            if (more) { fetch(B, b + 1); fetch_x(b + 1); }
            compute(A, b, 0);
            if (more) store_x(1);
            __syncthreads();
            if (more) {
                const bool again = b + 2 < nb;
                if (again) { fetch(A, b + 2); fetch_x(b + 2); }
                compute(B, b + 1, 1);
                if (again) store_x(0);
                __syncthreads();
            }
        }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * 16 + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (SK > 1) {
                    float* dst = part + (static_cast<size_t>(blockIdx.z) * M + row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else if (F32) {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0)
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
}

template <int BM, int WN, int U, bool F32>
void launch_stream(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& scales,
                   const at::Tensor& biases, at::Tensor& out, const at::Tensor& part, int N, int SK) {
    constexpr int BN = WN * 16, SMEM = 2 * U * BM * 128;
    const int M = x.size(0), K = x.size(1);
    auto kernel = qmm_stream_kernel<BM, WN, U, F32>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        configured = true;
    }
    const int rows_t = (M + BM - 1) / BM;
    const int group = std::max(1, std::min(rows_t, static_cast<int>(
        static_cast<long long>(l2_group_bytes()) / (static_cast<long long>(BM) * K * 2))));
    kernel<<<dim3(rows_t * ((N + BN - 1) / BN), 1, SK), WN * 32, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), xs.data_ptr<float>(),
        reinterpret_cast<const unsigned char*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr()), out.data_ptr(),
        part.defined() ? part.data_ptr<float>() : nullptr, M, N, K, SK, static_cast<int>(scales.stride(0)),
        M == 1 ? K : static_cast<int>(x.stride(0)), group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// 16-row tiles of 4 warps (64 columns), one group a batch: 96 registers, so five blocks share an SM and the loads of
// many blocks are in flight at once (a 2080 Ti streams the weights at 340-480 GB/s, the shared-memory kernel 150-260).
// TF_QMM_STREAM="WN,U" picks another warp count and batch while tuning, "off" the shared-memory kernel;
// false when the shape has no streaming kernel.
template <bool F32>
bool dispatch_stream(int bm, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& s,
                     const at::Tensor& b, at::Tensor& out, const at::Tensor& part, int N, int SK) {
    static int wn = 4, u = 1;
    static bool read = false;
    if (!read) {
        read = true;
        if (const char* env = getenv("TF_QMM_STREAM")) sscanf(env, "%d,%d", &wn, &u);
    }
    if (bm != 16) return false;
#define GO(WN_, U_) if (wn == WN_ && u == U_) { launch_stream<16, WN_, U_, F32>(x, xs, w, s, b, out, part, N, SK); return true; }
    GO(4, 1) GO(8, 1) GO(2, 1) GO(16, 1) GO(4, 2) GO(4, 3) GO(4, 4) GO(2, 2) GO(2, 4) GO(2, 8) GO(8, 2) GO(8, 4) GO(1, 4) GO(1, 8) GO(4, 6) GO(4, 8)
#undef GO
    return false;
}

// The K slices' partial sums added in slice order, then rounded once.
template <bool F32>
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long long total, int SK) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < SK; ++s) acc = acc + part[s * total + i];
    if (F32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool PIPE = false>
void launch(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& scales,
            const at::Tensor& biases, at::Tensor& out, const at::Tensor& part, int N, int SK) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    const int M = x.size(0), K = x.size(1);
    auto kernel = qmm_kernel<GS, BM, BN, WM, WN, STAGES, F32, CLUSTER, PIPE>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        configured = true;
    }
    cudaLaunchConfig_t config = {};
    const int rows_t = (M + BM - 1) / BM;
    // a group's inputs stay near 12 MB of L2 while its blocks sweep the column tiles
    const int group = std::max(1, std::min(rows_t, static_cast<int>(
        static_cast<long long>(l2_group_bytes()) / (static_cast<long long>(BM) * K * 2))));
    config.gridDim = dim3(rows_t * ((N + BN - 1) / BN), 1, SK);
    config.blockDim = dim3(T::THREADS);
    config.dynamicSmemBytes = T::SMEM;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    if (CLUSTER) {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1;   // a tile's K slices share blockIdx.x
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = SK;
        config.attrs = attr;
        config.numAttrs = 1;
    }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel,
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), xs.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr()), out.data_ptr(),
        part.defined() ? part.data_ptr<float>() : nullptr, M, N, K, SK, static_cast<int>(scales.stride(0)),
        M == 1 ? K : static_cast<int>(x.stride(0)), group));
}

template <int GS, bool F32, bool CLUSTER>
void dispatch(int bm, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& s,
              const at::Tensor& b, at::Tensor& out, const at::Tensor& part, int N, int SK) {
    switch (bm) {
        case 16: launch<GS, 16, 64, 1, 4, 4, F32, CLUSTER>(x, xs, w, s, b, out, part, N, SK); break;
        case 32: launch<GS, 32, 64, 1, 4, 4, F32, CLUSTER>(x, xs, w, s, b, out, part, N, SK); break;
        default: launch<GS, 64, 64, 1, 4, 4, F32, CLUSTER>(x, xs, w, s, b, out, part, N, SK); break;
    }
}

} // namespace

// Clusters hold up to 8 K slices (the portable size) from sm_90; more, unreduced slices or older GPUs use the buffer.
bool qmm_clusters(int SK, bool reduce) {
    return SK > 1 && SK <= 8 && reduce && at::cuda::getCurrentDeviceProperties()->major >= 9;
}

void qmm_cuda(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& scales,
              const at::Tensor& biases, at::Tensor& out, const at::Tensor& part, int N, int SK, int gs, int bm,
              bool f32, bool reduce) {
    const int M = x.size(0);
    const bool cluster = qmm_clusters(SK, reduce);
#define GO(G, F, C) dispatch<G, F, C>(bm, x, xs, w, scales, biases, out, part, N, SK)
    static const bool stream_off = getenv("TF_QMM_STREAM") && !strcmp(getenv("TF_QMM_STREAM"), "off");
    const bool streamed = !stream_off && gs == 64 && at::cuda::getCurrentDeviceProperties()->major < 8 &&
        (f32 ? dispatch_stream<true>(bm, x, xs, w, scales, biases, out, part, N, SK)
             : dispatch_stream<false>(bm, x, xs, w, scales, biases, out, part, N, SK));
    if (streamed) {
    } else if (gs == 64) {
        if (f32) { if (cluster) GO(64, true, true); else GO(64, true, false); }
        else { if (cluster) GO(64, false, true); else GO(64, false, false); }
    } else {
        if (f32) { if (cluster) GO(32, true, true); else GO(32, true, false); }
        else { if (cluster) GO(32, false, true); else GO(32, false, false); }
    }
#undef GO
    if (SK > 1 && reduce && !cluster) {
        const long long total = static_cast<long long>(M) * N;
        const int threads = 256;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();
        if (f32) reduce_kernel<true><<<blocks, threads, 0, stream>>>(part.data_ptr<float>(), out.data_ptr(), total, SK);
        else reduce_kernel<false><<<blocks, threads, 0, stream>>>(part.data_ptr<float>(), out.data_ptr(), total, SK);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
