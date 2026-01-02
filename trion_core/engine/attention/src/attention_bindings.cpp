// -----------------------------------------------------------------------------
// WINDOWS BUILD FIX (CRITICAL)
// _addcarry_u64 hatası için bu satırlar EN TEPEDE ve KOŞULSUZ olmalı.
// -----------------------------------------------------------------------------
#include <cstdint>
#include <intrin.h> 
#include <limits>

#include <torch/extension.h>

// Forward Declarations
void rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor out, float epsilon);

void attention_fused_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Out, torch::Tensor L,
    torch::Tensor Mask, float scale, bool causal, bool has_mask, 
    float attn_threshold 
);

void attention_fused_backward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Out,
    torch::Tensor dO, torch::Tensor L, torch::Tensor Mask,
    torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV,
    float scale, bool causal, bool has_mask
);

void attention_inference_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Out, float scale);

// -----------------------------------------------------------------------------
// WRAPPERS
// -----------------------------------------------------------------------------

void forward_wrapper(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Out, torch::Tensor L,
    torch::Tensor Mask, float scale, bool causal, float attn_threshold 
) {
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kHalf, "Q must be float16");
    
    bool has_mask = (Mask.numel() > 0);
    if (has_mask) { TORCH_CHECK(Mask.is_contiguous(), "Mask must be contiguous"); }

    attention_fused_forward_cuda(Q, K, V, Out, L, Mask, scale, causal, has_mask, attn_threshold);
}

void backward_wrapper(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Out,
    torch::Tensor dO, torch::Tensor L, torch::Tensor Mask,
    torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV,
    float scale, bool causal
) {
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");
    TORCH_CHECK(dK.is_contiguous(), "dK must be contiguous");
    TORCH_CHECK(dV.is_contiguous(), "dV must be contiguous");

    bool has_mask = (Mask.numel() > 0);
    attention_fused_backward_cuda(Q, K, V, Out, dO, L, Mask, dQ, dK, dV, scale, causal, has_mask);
}

void inference_wrapper(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Out, float scale) {
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.stride(-1) == 1, "K last dim stride must be 1");
    TORCH_CHECK(V.stride(-1) == 1, "V last dim stride must be 1");
    attention_inference_cuda(Q, K, V, Out, scale);
}

void rmsnorm_wrapper(torch::Tensor x, torch::Tensor weight, torch::Tensor out, float epsilon) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    rmsnorm_cuda(x, weight, out, epsilon);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "Ghost Attention Forward");
    m.def("backward", &backward_wrapper, "Standard Attention Backward");
    m.def("inference", &inference_wrapper, "Inference Kernel");
    m.def("rmsnorm", &rmsnorm_wrapper, "RMSNorm CUDA");
}