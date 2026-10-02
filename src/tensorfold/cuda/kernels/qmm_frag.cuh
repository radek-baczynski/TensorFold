// Shared by the 4-bit matmuls: staging, ldmatrix, the mma, nibble decoding from B-fragment order.
// Ampere and newer keep cp.async and bf16 m16n8k16. Turing (sm_75) has ldmatrix and fp16 m16n8k8,
// so copies are synchronous and each bf16 k16 step is two fp16 k8 steps on the same fragments.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace qmm_frag {

__device__ __forceinline__ uint32_t smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
#else
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#endif
}

// 16 bytes, or zeros without reading when ``read`` is false (a row past M: computed, never stored)
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem(dst)), "l"(src), "r"(read ? 16 : 0));
#else
    uint4 z = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(dst) = read ? *reinterpret_cast<const uint4*>(src) : z;
#endif
}

__device__ __forceinline__ void cp8(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem(dst)), "l"(src));
#else
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
#endif
}

__device__ __forceinline__ void cp4(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem(dst)), "l"(src));
#else
    *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(src);
#endif
}

__device__ __forceinline__ void commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}

// One bf16 pair as fp16. Integers 0-15 survive; other values round to fp16's 10-bit mantissa.
__device__ __forceinline__ uint32_t bf16_to_fp16(uint32_t v) {
    const float lo = __uint_as_float((v & 0xffffu) << 16);
    const float hi = __uint_as_float(v & 0xffff0000u);
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// m16n8k8, the Turing tensor core, with the same A/B/C registers as one half of m16n8k16.
__device__ __forceinline__ void mma_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    mma_k8(d, bf16_to_fp16(a[0]), bf16_to_fp16(a[1]), bf16_to_fp16(b0));
    mma_k8(d, bf16_to_fp16(a[2]), bf16_to_fp16(a[3]), bf16_to_fp16(b1));
#endif
}

// The same mma from a zero accumulator (a group's first k16 step): no registers to clear.
__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
#else
    d[0] = d[1] = d[2] = d[3] = 0.0f;
    mma(d, a, b0, b1);
#endif
}

// Turing's prompt matmul keeps its inputs as fp16 in shared memory: converted once while copied in, not at every mma.
__device__ __forceinline__ uint4 bf16x8_to_fp16(uint4 v) {
    return make_uint4(bf16_to_fp16(v.x), bf16_to_fp16(v.y), bf16_to_fp16(v.z), bf16_to_fp16(v.w));
}

// 16 bytes of inputs: bf16 on Ampere and newer, fp16 below (zeros without reading when ``read`` is false).
__device__ __forceinline__ void cp16z_act(void* dst, const void* src, bool read) {
#if __CUDA_ARCH__ >= 800
    cp16z(dst, src, read);
#else
    uint4 v = make_uint4(0, 0, 0, 0);
    if (read) v = bf16x8_to_fp16(*reinterpret_cast<const uint4*>(src));
    *reinterpret_cast<uint4*>(dst) = v;
#endif
}

// The mma on fragments that are already fp16 (A from ldmatrix of fp16 inputs, B from ``weight_fp16x2``).
__device__ __forceinline__ void mma_h(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    mma_k8(d, a[0], a[1], b0);
    mma_k8(d, a[2], a[3], b1);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float((v & 0xffffu) << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// Prefill scales a decoded pair in bf16. Turing has no bf16x2, so the fma is fp32 and rounds back once.
__device__ __forceinline__ uint32_t fma_bf16x2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
#if __CUDA_ARCH__ >= 800
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
#else
    d = pack_bf16(__fmaf_rn(bf16_lo(a), bf16_lo(b), bf16_lo(c)), __fmaf_rn(bf16_hi(a), bf16_hi(b), bf16_hi(c)));
#endif
    return d;
}

// A float rounded to the nearest bf16 (ties to even), kept as a float.
__device__ __forceinline__ float round_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

// Turing: the fp16 pair of bf16(fma(q, scale, bias)) for the nibbles at bits [s, s + 4) and [16 + s, 20 + s).
// Same values as bf16_to_fp16(fma_bf16x2(pair(w, s), scale_pair, bias_pair)) with fewer conversions.
__device__ __forceinline__ uint32_t weight_fp16x2(uint32_t w, int s, uint32_t scale_pair, uint32_t bias_pair) {
    const float sc = bf16_lo(scale_pair), bi = bf16_lo(bias_pair);
    const float q0 = static_cast<float>((w >> s) & 0xFu);
    const float q1 = static_cast<float>((w >> (s + 16)) & 0xFu);
    const __half2 h = __floats2half2_rn(round_bf16(__fmaf_rn(q0, sc, bi)), round_bf16(__fmaf_rn(q1, sc, bi)));
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Nibbles at bits [s, s + 4) and [16 + s, 20 + s) as a bf16 pair: exponent bits make 128 + q, then minus 128.
__device__ __forceinline__ uint32_t pair(uint32_t w, int s) {
    const uint32_t t = ((w >> s) & 0x000F000Fu) | 0x43004300u;
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
#elif __CUDA_ARCH__ >= 800
    // no sub.bf16x2 before sm_90: t * 1 - 128 rounds the exact q once too (an exact value, the same bits)
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC300C300u));
#else
    // 0..15 are exact in bf16. Shift the nibble out and round the integer, same bits as the fma above.
    const float q0 = static_cast<float>((w >> s) & 0xFu);
    const float q1 = static_cast<float>((w >> (s + 16)) & 0xFu);
    r = pack_bf16(q0, q1);
#endif
    return r;
}

// ``pair`` as the operand the arch's mma takes: bf16 on Ampere and newer, fp16 below. 0..15 are exact in both; in
// fp16, bits 0x6400 | q are 1024 + q and one HSUB2 leaves q.
__device__ __forceinline__ uint32_t pair_act(uint32_t w, int s) {
#if __CUDA_ARCH__ >= 800
    return pair(w, s);
#else
    const uint32_t t = ((w >> s) & 0x000F000Fu) | 0x64006400u;
    const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&t), __float2half2_rn(1024.0f));
    return *reinterpret_cast<const uint32_t*>(&r);
#endif
}

// The mma on ``pair_act`` operands and inputs staged by ``cp16z_act``.
__device__ __forceinline__ void mma_act(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    mma(d, a, b0, b1);
#else
    mma_h(d, a, b0, b1);
#endif
}

__device__ __forceinline__ void mma0_act(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    mma0(d, a, b0, b1);
#else
    d[0] = d[1] = d[2] = d[3] = 0.0f;
    mma_h(d, a, b0, b1);
#endif
}

// Block b's (first row, first column), row tiles fastest in bands of ``group`` so blocks in flight share L2.
__device__ __forceinline__ int2 tile_of(int b, int M, int N, int BM, int BN, int group) {
    const int rows_t = (M + BM - 1) / BM, cols_t = (N + BN - 1) / BN, band = group * cols_t;
    const int first = b / band * group, in_band = b % band, height = min(group, rows_t - first);
    return make_int2((first + in_band % height) * BM, in_band / height * BN);
}

// A 16-byte chunk c of row r sits at chunk c ^ (r % CHUNKS) so ldmatrix's eight rows hit eight bank groups.
template <int CHUNKS>
__device__ __forceinline__ int swz(int r, int c) { return c ^ (r % CHUNKS); }

// Bytes of L2 a group's inputs should stay inside: 12 MiB on large caches, three quarters of a 2080 Ti's 5.5 MiB.
inline int l2_group_bytes() {
    int bytes = 0;
    cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, 0);
    if (bytes <= 0)
        bytes = 5 << 20;
    int budget = bytes - bytes / 4;
    if (budget > (12 << 20))
        budget = 12 << 20;
    if (budget < (2 << 20))
        budget = 2 << 20;
    return budget;
}

} // namespace qmm_frag
