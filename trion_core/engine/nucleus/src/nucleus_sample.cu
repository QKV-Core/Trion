// nucleus_sample.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/extension.h>
#include <chrono>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ------------------------------------------------------------
// Warp reduce: MAX
// ------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return val;
}

// ------------------------------------------------------------
// Warp reduce: SUM
// ------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// ------------------------------------------------------------
// Warp prefix sum (inclusive)
// ------------------------------------------------------------
__device__ __forceinline__ float warp_prefix_sum(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(FULL_MASK, val, offset);
        if (threadIdx.x >= offset)
            val += n;
    }
    return val;
}

// ------------------------------------------------------------
// CUDA KERNEL (SADECE DEVICE KODU)
// ------------------------------------------------------------
__global__ void nucleus_sample_kernel(
    const float* __restrict__ logits,   // [V]
    int vocab_size,
    float temperature,
    float top_p,
    unsigned long long seed,
    int* __restrict__ output_token       // [1]
) {
    int lane = threadIdx.x;
    int idx  = lane;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, lane, 0, &rng);

    // -----------------------------
    // 1) Max
    // -----------------------------
    float local_max = -1e20f;
    for (int i = idx; i < vocab_size; i += WARP_SIZE) {
        local_max = fmaxf(local_max, logits[i]);
    }
    float warp_max = warp_reduce_max(local_max);

    // -----------------------------
    // 2) Sum exp
    // -----------------------------
    float local_sum = 0.0f;
    for (int i = idx; i < vocab_size; i += WARP_SIZE) {
        float v = (logits[i] - warp_max) / temperature;
        local_sum += expf(v);
    }
    float warp_sum = warp_reduce_sum(local_sum);

    // -----------------------------
    // 3) Nucleus sampling
    // -----------------------------
    float r = curand_uniform(&rng);
    float running = 0.0f;
    float selected_token = -1.0f;

    for (int i = idx; i < vocab_size; i += WARP_SIZE) {
        float v = (logits[i] - warp_max) / temperature;
        float p = expf(v) / warp_sum;

        running = warp_prefix_sum(p);

        int keep = (running <= top_p) || (i == idx);
        p *= keep;

        float prev = __shfl_up_sync(FULL_MASK, running, 1);
        if (lane == 0) prev = 0.0f;

        if (selected_token < 0 && running >= r && prev < r) {
            selected_token = (float)i;
        }
    }

    // -----------------------------
    // 4) Warp select
    // -----------------------------
    int mask = __ballot_sync(FULL_MASK, selected_token >= 0);
    if (mask) {
        int leader = __ffs(mask) - 1;
        if (lane == leader) {
            *output_token = (int)selected_token;
        }
    }
}

// ------------------------------------------------------------
// PYTORCH ENTRY POINT (HOST SAFE)
// ------------------------------------------------------------
void nucleus_sample_cuda(
    torch::Tensor logits,
    float temperature,
    float top_p,
    torch::Tensor output
) {
    const int threads = WARP_SIZE;
    const int blocks = 1;

    // ✅ HOST-SAFE SEED
    unsigned long long seed =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();

    nucleus_sample_kernel<<<blocks, threads>>>(
        logits.data_ptr<float>(),
        logits.numel(),
        temperature,
        top_p,
        seed,
        output.data_ptr<int>()
    );
}
