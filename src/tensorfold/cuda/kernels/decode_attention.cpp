#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void decode_chunks_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        at::Tensor&, at::Tensor&, at::Tensor&, int, int, float);
void decode_tail_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                      at::Tensor&, at::Tensor&, int, int, float, int);

static void check_partials(const at::Tensor& po, const at::Tensor& pm, const at::Tensor& pl) {
    TORCH_CHECK(po.is_contiguous() && pm.is_contiguous() && pl.is_contiguous() && po.scalar_type() == at::kFloat &&
                pm.scalar_type() == at::kFloat && pl.scalar_type() == at::kFloat, "partials: contiguous fp32");
}

static void check_plan(const at::Tensor& q, const at::Tensor& off, const at::Tensor& stream, int64_t hk, int64_t group) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 &&
                q.size(2) == 256 && q.size(1) == hk * group, "q: contiguous (W, H, 256) bf16");
    TORCH_CHECK(off.is_cuda() && off.scalar_type() == at::kLong && off.dim() == 2 && off.size(1) == 2, "off: (S, 2) int64");
    TORCH_CHECK(stream.is_cuda() && stream.scalar_type() == at::kInt && stream.is_contiguous(), "stream: int32");
}

// Partials (o, m, l) of whole 512-key chunks for the tree attention's (stream, query tile, chunk) items.
void decode_chunks(const at::Tensor& q, const at::Tensor& origin, const at::Tensor& off, const at::Tensor& stream,
                   const at::Tensor& items, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int64_t hk, int64_t group,
                   double scale) {
    check_plan(q, off, stream, hk, group);
    check_partials(po, pm, pl);
    TORCH_CHECK(items.is_cuda() && items.scalar_type() == at::kInt && items.is_contiguous(), "items: int32");
    c10::cuda::CUDAGuard guard(q.device());
    decode_chunks_cuda(q, origin, off, stream, items, po, pm, pl, static_cast<int>(hk), static_cast<int>(group),
                       static_cast<float>(scale));
}

// Partials of the chunks that hold each stream's end: ``tails`` chunks from the one its committed keys reach.
void decode_tail(const at::Tensor& q, const at::Tensor& kn, const at::Tensor& vn, const at::Tensor& origin,
                 const at::Tensor& off, const at::Tensor& stream, const at::Tensor& rows, const at::Tensor& paths,
                 const at::Tensor& depths, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int64_t hk, int64_t group,
                 double scale, int64_t tails) {
    check_plan(q, off, stream, hk, group);
    check_partials(po, pm, pl);
    TORCH_CHECK(kn.is_cuda() && kn.is_contiguous() && kn.scalar_type() == at::kBFloat16 && kn.sizes() == vn.sizes() &&
                vn.is_contiguous() && vn.scalar_type() == at::kBFloat16 && kn.dim() == 3 && kn.size(1) == hk &&
                kn.size(2) == 256, "node keys and values: contiguous (W, HK, 256) bf16");
    TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kInt && paths.scalar_type() == at::kInt &&
                depths.scalar_type() == at::kInt && paths.is_contiguous(), "rows, paths, depths: int32");
    c10::cuda::CUDAGuard guard(q.device());
    decode_tail_cuda(q, kn, vn, origin, off, stream, rows, paths, depths, po, pm, pl, static_cast<int>(hk),
                     static_cast<int>(group), static_cast<float>(scale), static_cast<int>(tails));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_chunks", &decode_chunks);
    m.def("decode_tail", &decode_tail);
}
