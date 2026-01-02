#include <torch/extension.h>

// CUDA forward declaration
void nucleus_sample_cuda(
    torch::Tensor logits,
    float temperature,
    float top_p,
    torch::Tensor output
);

// Python binding
torch::Tensor nucleus_sample(
    torch::Tensor logits,
    float temperature,
    float top_p
) {
    auto output = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kInt32).device(logits.device())
    );

    nucleus_sample_cuda(logits, temperature, top_p, output);
    return output;
}

// 🔑 BURASI ÇOK ÖNEMLİ
PYBIND11_MODULE(trion_nucleus, m) {
    m.def(
        "nucleus_sample",
        &nucleus_sample,
        "CUDA nucleus sampling (warp-level, branch-free)"
    );
}
