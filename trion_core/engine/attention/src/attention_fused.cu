#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define WARP_SIZE 32

// -----------------------------------------------------------------------------
// HELPER FUNCTIONS
// -----------------------------------------------------------------------------
__device__ __forceinline__ float h2f(const __half h) { return __half2float(h); }

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// -----------------------------------------------------------------------------
// GHOST KERNEL (SHIFTED RELU + TRACER EDITION)
// -----------------------------------------------------------------------------
__global__ void attention_fused_forward_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ Out,
    float* __restrict__ L,      
    const __half* __restrict__ Mask,
    const int stride_b, const int stride_h, const int stride_s,
    const int B, const int H, const int T, const int D,
    const float scale, const bool causal,
    const bool has_mask,
    const float attn_threshold
) {
    const int t_idx = blockIdx.x; 
    const int h_idx = blockIdx.y; 
    const int b_idx = blockIdx.z; 
    const int tid   = threadIdx.x;

    const long long bh_offset = (long long)b_idx * stride_b + (long long)h_idx * stride_h;
    const long long q_offset = bh_offset + t_idx * stride_s;

    // Load Q
    float q_reg[8] = {0.f}; 
    #pragma unroll
    for (int i = 0; i * WARP_SIZE < D; ++i) {
        int d = i * WARP_SIZE + tid;
        if (d < D && i < 8) q_reg[i] = h2f(Q[q_offset + d]);
    }

    float running_sum = 0.0f;
    float acc[8] = {0.f};

    const int BLOCK_K = 32;
    extern __shared__ __half smem[];
    __half* Ksh = smem;
    __half* Vsh = smem + BLOCK_K * D;

    const int loop_limit = causal ? (t_idx + 1) : T;

    for (int k_base = 0; k_base < loop_limit; k_base += BLOCK_K) {
        // Load K/V Block
        for (int i = tid; i < BLOCK_K * D; i += WARP_SIZE) {
            int row = i / D; int col = i % D; int gk = k_base + row;
            bool valid = (gk < T) && (!causal || gk <= t_idx);
            if (valid) {
                long long off = bh_offset + gk * stride_s + col;
                Ksh[i] = K[off]; Vsh[i] = V[off];
            } else {
                Ksh[i] = __float2half(0.f); Vsh[i] = __float2half(0.f);
            }
        }
        __syncthreads();

        // Process Block
        for (int k = 0; k < BLOCK_K; ++k) {
            int gk = k_base + k;
            if (gk >= T) break;
            if (causal && gk > t_idx) continue;

            float score = 0.f;
            #pragma unroll
            for (int i = 0; i * WARP_SIZE < D; ++i) {
                int d = i * WARP_SIZE + tid;
                if (d < D && i < 8) {
                    score += q_reg[i] * h2f(Ksh[k * D + d]);
                }
            }
            score = warp_reduce_sum(score);
            score = __shfl_sync(0xffffffff, score, 0);
            score *= scale;

            if (has_mask) {
                float m_val = h2f(Mask[b_idx * T + gk]); 
                score += m_val;
            }

            // --- GHOST LOGIC: SHIFTED RELU ---
            // Negatif threshold verildiğinde burası POZİTİF olur.
            float weight = score - attn_threshold;
            if (weight < 0.0f) weight = 0.0f;

            running_sum += weight;

            #pragma unroll
            for (int i = 0; i * WARP_SIZE < D; ++i) {
                int d = i * WARP_SIZE + tid;
                if (d < D && i < 8) {
                    float v = h2f(Vsh[k * D + d]);
                    acc[i] += weight * v;
                }
            }
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / (running_sum + 1e-6f);

    #pragma unroll
    for (int i = 0; i * WARP_SIZE < D; ++i) {
        int d = i * WARP_SIZE + tid;
        if (d < D && i < 8) {
            // TRACER EKLENDİ: + 1e-3f
            // Bu sayede kernel'ın güncellendiğini anlayacağız.
            // Sonuç asla 0.00 olamaz, en az 0.001 olmalı.
            float val = (acc[i] * inv_sum) + 1e-3f; 
            Out[q_offset + d] = __float2half(val);
        }
    }
    
    if (tid == 0) {
        int l_idx = b_idx * (H * T) + h_idx * T + t_idx;
        L[l_idx] = running_sum;
    }
}

// -----------------------------------------------------------------------------
// LAUNCHER
// -----------------------------------------------------------------------------
void attention_fused_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Out, torch::Tensor L,
    torch::Tensor Mask,
    float scale, bool causal, bool has_mask, float attn_threshold
) {
    int B = Q.size(0); int H = Q.size(1); int T = Q.size(2); int D = Q.size(3);
    dim3 grid(T, H, B); dim3 block(WARP_SIZE);
    size_t shmem = 2 * 32 * D * sizeof(__half);
    const __half* mask_ptr = has_mask ? (const __half*)Mask.data_ptr() : nullptr;

    attention_fused_forward_kernel<<<grid, block, shmem>>>(
        (const __half*)Q.data_ptr(), (const __half*)K.data_ptr(), (const __half*)V.data_ptr(),
        (__half*)Out.data_ptr(), (float*)L.data_ptr(), mask_ptr,
        Q.stride(0), Q.stride(1), Q.stride(2), B, H, T, D, scale, causal, has_mask,
        attn_threshold
    );
}

// STUBS
void attention_fused_backward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Out, torch::Tensor dO, torch::Tensor L, torch::Tensor Mask, torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV, float scale, bool causal, bool has_mask) {}
void attention_inference_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Out, float scale) {}
void rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor out, float epsilon) {}