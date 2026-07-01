#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "rwkv7_cuda_utils.h"
#include "rwkv7_cuda_time_parallel_forward.h"
#include "rwkv7_cuda_time_parallel_backward.h"
#include "parallel_scan.h"

using namespace nvcuda;

namespace rwkv {
template <int CHUNK_LEN=32, typename F>
__global__ void rwkv7_wkv_forward_kernel(
    const int B,
    const int T,
    const int H,
    const F* __restrict__ r_BTHK,
    const F* __restrict__ k_BTHK,
    const F* __restrict__ v_BTHK,
    const float* __restrict__ w_BTHK,
    const F* __restrict__ a_BTHK,
    const F* __restrict__ k_deformed_BTHK,
    const bool* __restrict__ skip_BT,
    F* __restrict__ out_BTHK,
    const int L,
    float* __restrict__ state_checkpoints_BLHKK,
    // Stateful BPTT: optional initial state (carried from the previous chunk) and optional
    // final-state output (to carry into the next chunk). Both fp32 [B,H,K,K]. When null, the
    // kernel is byte-for-byte the original (state starts at 0, no final write).
    const float* __restrict__ state0_BHKK,
    float* __restrict__ final_state_BHKK
    ) {
    const int K = blockDim.x;
    int b = blockIdx.x;
    int h = blockIdx.y;

    // Swapped for better memory coalescing. We want x to refer to rows and y to refer to columns and one entire row = 1 warp
    int x = threadIdx.y;
    int y = threadIdx.x;

    int s_idx = ((b * H + h) * K + x) * K + y;  // index into a [B,H,K,K] state tensor
    float state_xy = (state0_BHKK != nullptr) ? state0_BHKK[s_idx] : 0.0f;
    int state_loc = get_index4(b, 0, h, x, y, L, H, K, K);
    int64_t global_y = get_index3(b, 0, h, y, T, H, K);
    int64_t global_x = get_index3(b, 0, h, x, T, H, K);
    for (int t = 0; t < T; t++) {
        if (t % CHUNK_LEN == 0) {
            state_checkpoints_BLHKK[state_loc] = state_xy;
            state_loc += H * K * K;
        }
        // load in the relevant values
        float r_y = to_float<F>(r_BTHK[global_y]);
        float k_y = to_float<F>(k_BTHK[global_y]);
        float v_x = to_float<F>(v_BTHK[global_x]);
        float w_y = w_BTHK[global_y];
        float a_y = to_float<F>(a_BTHK[global_y]);
        float k_deformed_y = to_float<F>(k_deformed_BTHK[global_y]);
        bool skip = skip_BT[get_index1(b, t, T)];
        float in_state_xy = state_xy;

        // compute decayed state value at (x, y)
        float state_xy_decayed = state_xy * w_y;
        float state_k_dot = state_xy * k_deformed_y;
        // compute S@k. We do this in parallel at the row (warp) level
        // Parallel reduction: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
        for (int offset = K / 2; offset > 0; offset /= 2) {
            state_k_dot += __shfl_down_sync(FULL_MASK, state_k_dot, offset, K);
        }
        state_k_dot = __shfl_sync(FULL_MASK, state_k_dot, 0, K);
        state_xy = state_xy_decayed - state_k_dot * a_y * k_deformed_y;
        state_xy += v_x * k_y;
        // Compute S@r and store the result in out
        float state_r_dot = state_xy * r_y;
        for (int offset = K / 2; offset > 0; offset /= 2) {
            state_r_dot += __shfl_down_sync(FULL_MASK, state_r_dot, offset, K);
        }
        if (y == 0) {
            out_BTHK[global_x] = to_F<F>(state_r_dot);
        }
        if (skip) {
            state_xy = in_state_xy;
        }
        global_x += H * K;
        global_y += H * K;
    }
    // Stateful BPTT: emit the post-last-step state so the next chunk can resume from it.
    if (final_state_BHKK != nullptr) {
        final_state_BHKK[s_idx] = state_xy;
    }
}

template <int CHUNK_LEN=32, typename F>
__global__ void rwkv7_wkv_backward_kernel(
    const int B,
    const int T,
    const int H,
    const F* __restrict__ r_BTHK,
    const F* __restrict__ k_BTHK,
    const F* __restrict__ v_BTHK,
    const float* __restrict__ w_BTHK,
    const F* __restrict__ a_BTHK,
    const F* __restrict__ k_deformed_BTHK,
    const bool* __restrict__ skip_BT,
    const F* __restrict__ grad_BTHK,
    const int L,
    const float* __restrict__ state_checkpoints_BLHKK,
    F* __restrict__ r_grad_BTHK,
    F* __restrict__ k_grad_BTHK,
    F* __restrict__ v_grad_BTHK,
    float* __restrict__ w_grad_BTHK,
    F* __restrict__ a_grad_BTHK,
    F* __restrict__ k_deformed_grad_BTHK
    ) {
    const int K = blockDim.x;
    // Shared scratch sized for the max supported K (32); indices use the runtime K, so K-general.
    // (Removed KK_grad_decay_remove, which was declared but never read -- a dead shared allocation.)
    __shared__ float KK_state[32 * (32 + 1)];
    __shared__ float KK_state_prev[32 * (32 + 1)];
    __shared__ float KK_dS[32 * (32 + 1)];
    __shared__ float KK_grad_decay[32 * (32 + 1)];
    __shared__ float K_k_deformed[32];
    __shared__ float K_a[32];
    float state_xy_chunk[CHUNK_LEN];
    float state_prev_xy_chunk[CHUNK_LEN];
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int x = threadIdx.y;
    const int y = threadIdx.x;

    if (x == 0) {
        a_grad_BTHK[get_index3(b, 0, h, y, T, H, K)] = to_F<F>(0.0);
        k_deformed_grad_BTHK[get_index3(b, 0, h, y, T, H, K)] = to_F<F>(0.0);
    }

    float dS_xy_contrib = 0.0;
    for (int l = L - 1; l >= 0; l--) {
        // recompute the states from the checkpoints
        float state_xy = state_checkpoints_BLHKK[get_index4(b, l, h, x, y, L, H, K, K)];
        for (int c = 0; c < CHUNK_LEN; c++) {
            int t = l * CHUNK_LEN + c;
            if (t >= T) break;

            bool skip = skip_BT[get_index1(b, t, T)];
            state_prev_xy_chunk[c] = state_xy;
            float in_state_xy = state_xy;
            int64_t global_y = get_index3(b, t, h, y, T, H, K);
            int64_t global_x = get_index3(b, t, h, x, T, H, K);
            float k_y = to_float<F>(k_BTHK[global_y]);
            float v_x = to_float<F>(v_BTHK[global_x]);
            float w_y = w_BTHK[global_y];
            float a_y = to_float<F>(a_BTHK[global_y]);
            float k_deformed_y = to_float<F>(k_deformed_BTHK[global_y]);

            // compute decayed state value at (x, y)
            float state_xy_decayed = state_xy * w_y;
            float state_k_dot = state_xy * k_deformed_y;
            for (int offset = K / 2; offset > 0; offset /= 2) {
                state_k_dot += __shfl_down_sync(FULL_MASK, state_k_dot, offset, K);
            }

            state_k_dot = __shfl_sync(FULL_MASK, state_k_dot, 0, K);
            state_xy = state_xy_decayed - state_k_dot * a_y * k_deformed_y;
            state_xy += v_x * k_y;
            state_xy_chunk[c] = state_xy;
            if (skip) {
                state_xy = in_state_xy;
            }
        }

        for (int t = std::min(T - 1, (l + 1) * CHUNK_LEN - 1); t >= l * CHUNK_LEN; t--) {
            int c = t - l * CHUNK_LEN;
            float state_xy = state_xy_chunk[c];
            KK_state[get_index1(x, y, K+1)] = state_xy;
            KK_state_prev[get_index1(x, y, K+1)] = state_prev_xy_chunk[c];

            int64_t global_x = get_index3(b, t, h, x, T, H, K);
            int64_t global_y = get_index3(b, t, h, y, T, H, K);
            float r_y = to_float<F>(r_BTHK[global_y]);
            float k_y = to_float<F>(k_BTHK[global_y]);
            float v_y = to_float<F>(v_BTHK[global_y]);
            float w_y = w_BTHK[global_y];
            float a_y = to_float<F>(a_BTHK[global_y]);
            float k_deformed_x = to_float<F>(k_deformed_BTHK[global_x]);
            float k_deformed_y = to_float<F>(k_deformed_BTHK[global_y]);
            bool skip = skip_BT[get_index1(b, t, T)];
            float grad_x = to_float<F>(grad_BTHK[global_x]);
            float grad_y = to_float<F>(grad_BTHK[global_y]);
            float dS_xy = grad_x * r_y;
            if (!skip) {
                dS_xy += dS_xy_contrib;
                dS_xy_contrib = 0.0;
            }
            float dS_xy_decay = dS_xy * w_y;
            float dS_xy_remove = dS_xy * a_y * k_deformed_y;
            KK_dS[get_index1(x, y, K + 1)] = dS_xy;
            if (x == 0) {
                K_k_deformed[y] = k_deformed_y;
                K_a[y] = a_y;
            }

            __syncthreads(); // for KK_state, KK_dS

            float grad_decay_remove_xy = 0.0;
            for (int k = 0; k < K; k++) {
                grad_decay_remove_xy += KK_state_prev[get_index1(k, x, K+1)] * KK_dS[get_index1(k, y, K+1)];
            }
            if (x == y) {
                w_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = grad_decay_remove_xy;
            }
            KK_grad_decay[get_index1(x, y, K+1)] = grad_decay_remove_xy;

            float state_mT_xy = KK_state[get_index1(y, x, K + 1)];
            float state_grad_dot = state_mT_xy * grad_y;
            float v_grad_x = dS_xy * k_y;
            float k_grad_x = KK_dS[get_index1(y, x, K + 1)] * v_y;

            // TODO dS_xy_remove must stay as float for accurate propagation, but the rest can be batched up in a 32x3 matrix and matmull'd?
            // Looks like no, they each have a different multiplier matrix.
            // But we can still use 3x4 = 12 warps to do this on the tensor cores instead.
            for (int offset = K / 2; offset > 0; offset /= 2) {
                v_grad_x += __shfl_down_sync(FULL_MASK, v_grad_x, offset, K);
                k_grad_x += __shfl_down_sync(FULL_MASK, k_grad_x, offset, K);
                state_grad_dot += __shfl_down_sync(FULL_MASK, state_grad_dot, offset, K);
                dS_xy_remove += __shfl_down_sync(FULL_MASK, dS_xy_remove, offset, K);
            }
            if (y == 0) {
                v_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(v_grad_x);
                k_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(k_grad_x);
                r_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(state_grad_dot);
            }
            __syncthreads(); // for KK_grad_decay
            float KK_grad_decay_yx = KK_grad_decay[get_index1(y, x, K+1)];
            float a_grad_x = -KK_grad_decay_yx * K_k_deformed[y];
            float k_deformed_t1 = -grad_decay_remove_xy * K_a[y] * K_k_deformed[y];
            float k_deformed_t2 = -K_a[x] * KK_grad_decay_yx * K_k_deformed[y];
            // TODO potential tensor core optimization
            for (int offset = K / 2; offset > 0; offset /= 2) {
                a_grad_x += __shfl_down_sync(FULL_MASK, a_grad_x, offset, K);
                k_deformed_t1 += __shfl_down_sync(FULL_MASK, k_deformed_t1, offset, K);
                k_deformed_t2 += __shfl_down_sync(FULL_MASK, k_deformed_t2, offset, K);
            }
            
            if (y == 0) {
                a_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(a_grad_x * K_k_deformed[x]);
                k_deformed_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(k_deformed_t1 + k_deformed_t2);
            }

            dS_xy_remove = __shfl_sync(FULL_MASK, dS_xy_remove, 0, K);
            dS_xy_contrib += dS_xy_decay - dS_xy_remove * k_deformed_y;
            __syncthreads();
        }
    }
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor> rwkv7_wkv_forward_cuda(
    const at::Tensor& r_BTHK, 
    const at::Tensor& k_BTHK,
    const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK,
    const at::Tensor& a_BTHK,
    const at::Tensor& k_deformed_BTHK,
    const at::Tensor& skip_BT
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = r_BTHK.size(3);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CUDA);
    const F* r_ptr = (F*)r_BTHK.data_ptr();
    const F* k_ptr = (F*)k_BTHK.data_ptr();
    const F* v_ptr = (F*)v_BTHK.data_ptr();
    const float* w_ptr = w_BTHK.data_ptr<float>();
    const F* a_ptr = (F*)a_BTHK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_BTHK.data_ptr();
    const bool* skip_ptr = (bool*)skip_BT.data_ptr();
    
    at::Tensor out_BTHK = torch::empty(r_BTHK.sizes(), r_BTHK.options());
    F* out_ptr = (F*)out_BTHK.data_ptr();
    int L = (T + CHUNK_LEN) / CHUNK_LEN;
    at::Tensor state_checkpoints_BLHKK = torch::empty({B, L, H, K, K}, r_BTHK.options().dtype(torch::kFloat32)).requires_grad_(false);
    float* state_checkpoints_ptr = state_checkpoints_BLHKK.data_ptr<float>();

    const int BASE_COARSE = 512;
    if (T >= 3 * BASE_COARSE) {
        int M = (T + BASE_COARSE - 1) / BASE_COARSE;
        dim3 base_block_dim(B, H, M);
        dim3 grid_dim(K, K);

        // Scratch for the time-parallel scan, via PyTorch's CUDA caching allocator instead of raw
        // cudaMalloc/cudaFree. cudaFree is a SYNCHRONIZING call (it stalls the stream until all prior
        // GPU work completes); with ~14 WKV layers x (fwd+bwd) per step this serialized dozens of
        // kernels/step. torch::empty pulls from the cached pool (no real malloc after warmup) and the
        // RAII free is recorded against the current stream (no device sync). Byte-identical numerics --
        // same contiguous 2*B*M*H*K*K float layout, only the memory source differs.
        at::Tensor scan_buf = torch::empty({2, B, M, H, K, K}, r_BTHK.options().dtype(torch::kFloat32));
        float *buffer = scan_buf.data_ptr<float>();
        float *partial_mul_BMHKK = buffer;
        float *partial_add_BMHKK = buffer + (int64_t) B * M * H * K * K;
        assert(M >= 3);
        rwkv7_wkv_forward_time_parallel_base_kernel<F><<<base_block_dim, grid_dim>>>(B, T, H, BASE_COARSE, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr, M, partial_mul_BMHKK, partial_add_BMHKK);
        rwkv7_scan_forward(B, M, H, K, partial_mul_BMHKK, partial_add_BMHKK);
        rwkv7_wkv_forward_time_parallel_final_kernel<CHUNK_LEN, F><<<base_block_dim, grid_dim>>>(B, T, H, BASE_COARSE, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr, out_ptr, M, partial_add_BMHKK, L, state_checkpoints_ptr);
    } else {
        dim3 block_dim(K, K);
        dim3 grid_dim(B, H);
        rwkv7_wkv_forward_kernel<CHUNK_LEN><<<grid_dim, block_dim>>>(B, T, H, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr, out_ptr, L, state_checkpoints_ptr, nullptr, nullptr);
    }
    return std::make_tuple(out_BTHK, state_checkpoints_BLHKK);
}

// Stateful WKV forward (truncated BPTT): like rwkv7_wkv_forward_cuda but takes an initial state
// (carried from the previous chunk) and returns the final state (to carry into the next). ALWAYS
// uses the sequential kernel -- the time-parallel scan path starts from a zero/identity state and
// would ignore state0; stateful chunks are small, so sequential is both correct and the right regime.
template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor, at::Tensor> rwkv7_wkv_forward_stateful_cuda(
    const at::Tensor& r_BTHK,
    const at::Tensor& k_BTHK,
    const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK,
    const at::Tensor& a_BTHK,
    const at::Tensor& k_deformed_BTHK,
    const at::Tensor& skip_BT,
    const at::Tensor& state0_BHKK
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = r_BTHK.size(3);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CUDA);
    const F* r_ptr = (F*)r_BTHK.data_ptr();
    const F* k_ptr = (F*)k_BTHK.data_ptr();
    const F* v_ptr = (F*)v_BTHK.data_ptr();
    const float* w_ptr = w_BTHK.data_ptr<float>();
    const F* a_ptr = (F*)a_BTHK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_BTHK.data_ptr();
    const bool* skip_ptr = (bool*)skip_BT.data_ptr();
    const float* state0_ptr = state0_BHKK.data_ptr<float>();

    at::Tensor out_BTHK = torch::empty(r_BTHK.sizes(), r_BTHK.options());
    F* out_ptr = (F*)out_BTHK.data_ptr();
    int L = (T + CHUNK_LEN) / CHUNK_LEN;
    at::Tensor state_checkpoints_BLHKK = torch::empty({B, L, H, K, K}, r_BTHK.options().dtype(torch::kFloat32)).requires_grad_(false);
    float* state_checkpoints_ptr = state_checkpoints_BLHKK.data_ptr<float>();
    at::Tensor final_state_BHKK = torch::empty({B, H, K, K}, r_BTHK.options().dtype(torch::kFloat32)).requires_grad_(false);
    float* final_state_ptr = final_state_BHKK.data_ptr<float>();

    dim3 block_dim(K, K);
    dim3 grid_dim(B, H);
    rwkv7_wkv_forward_kernel<CHUNK_LEN><<<grid_dim, block_dim>>>(B, T, H, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr, out_ptr, L, state_checkpoints_ptr, state0_ptr, final_state_ptr);
    return std::make_tuple(out_BTHK, state_checkpoints_BLHKK, final_state_BHKK);
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rwkv7_wkv_backward_cuda(
    const at::Tensor& r_BTHK, 
    const at::Tensor& k_BTHK,
    const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK,
    const at::Tensor& a_BTHK,
    const at::Tensor& k_deformed_BTHK,
    const at::Tensor& skip_BT,
    const at::Tensor& state_checkpoints_BLHKK,
    const at::Tensor& grad_BTHK
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = r_BTHK.size(3);
    const int L = state_checkpoints_BLHKK.size(1);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CUDA);
    const F* r_ptr = (F*)r_BTHK.data_ptr();
    const F* k_ptr = (F*)k_BTHK.data_ptr();
    const F* v_ptr = (F*)v_BTHK.data_ptr();
    const float* w_ptr = w_BTHK.data_ptr<float>();
    const F* a_ptr = (F*)a_BTHK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_BTHK.data_ptr();
    const bool* skip_ptr = (bool*)skip_BT.data_ptr();
    const float* state_checkpoints_ptr = state_checkpoints_BLHKK.data_ptr<float>();
    const F* grad_ptr = (F*)grad_BTHK.data_ptr();
    at::Tensor r_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor k_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor v_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor w_grad_BTHK = torch::zeros_like(r_BTHK, torch::dtype(torch::kFloat32));
    at::Tensor a_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor k_deformed_grad_BTHK = torch::zeros_like(r_BTHK);
    F* r_grad_ptr = (F*)r_grad_BTHK.data_ptr();
    F* k_grad_ptr = (F*)k_grad_BTHK.data_ptr();
    F* v_grad_ptr = (F*)v_grad_BTHK.data_ptr();
    float* w_grad_ptr = w_grad_BTHK.data_ptr<float>();
    F* a_grad_ptr = (F*)a_grad_BTHK.data_ptr();
    F* k_deformed_grad_ptr = (F*)k_deformed_grad_BTHK.data_ptr();

    const int BASE_COARSE = 128;
    if (T >= 3 * BASE_COARSE) {
        int M = (T + BASE_COARSE - 1) / BASE_COARSE;
        dim3 base_block_dim(B, H, M);
        dim3 grid_dim(K, K);

        // Caching-allocator scratch (see forward): replaces synchronizing cudaMalloc/cudaFree. The
        // backward time-parallel path triggers for any stream > 3*BASE_COARSE = 384 reviews -- i.e.
        // most users -- so this removed a per-layer device sync on the hot path. Byte-identical layout.
        at::Tensor scan_buf = torch::empty({2, B, M, H, K, K}, r_BTHK.options().dtype(torch::kFloat32));
        float *buffer = scan_buf.data_ptr<float>();
        float *partial_mul_BMHKK = buffer;
        float *partial_add_BMHKK = buffer + (int64_t) B * M * H * K * K;
        assert(BASE_COARSE % CHUNK_LEN == 0);
        rwkv7_wkv_backward_time_parallel_base_kernel<F><<<base_block_dim, grid_dim>>>(B, T, H, BASE_COARSE, grad_ptr, r_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr, M, partial_mul_BMHKK, partial_add_BMHKK);
        rwkv7_scan_backward(B, M, H, K, partial_mul_BMHKK, partial_add_BMHKK);
        rwkv7_wkv_backward_time_parallel_final_kernel<CHUNK_LEN, F><<<base_block_dim, grid_dim>>>(B, T, H, BASE_COARSE, grad_ptr, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr, M, partial_add_BMHKK, L, state_checkpoints_ptr, r_grad_ptr, k_grad_ptr, v_grad_ptr, w_grad_ptr, a_grad_ptr, k_deformed_grad_ptr);
    } else {
        dim3 block_dim(K, K);
        dim3 grid_dim(B, H);
        rwkv7_wkv_backward_kernel<CHUNK_LEN><<<grid_dim, block_dim>>>(B, T, H, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr, 
        grad_ptr, L, state_checkpoints_ptr, r_grad_ptr, k_grad_ptr, v_grad_ptr, w_grad_ptr, a_grad_ptr, k_deformed_grad_ptr);
    }

    return std::make_tuple(r_grad_BTHK, k_grad_BTHK, v_grad_BTHK, w_grad_BTHK, a_grad_BTHK, k_deformed_grad_BTHK);
}

// Stateful WKV backward (truncated BPTT): identical math to rwkv7_wkv_backward_cuda but ALWAYS uses
// the sequential kernel. The sequential backward recomputes each chunk forward from its checkpoint;
// checkpoint[0] is the (possibly nonzero) state0 written by the stateful forward, so it is already
// correct for a nonzero initial state -- including the nonzero w/a/k_deformed grads at t=0 that the
// decay acting on state0 produces. The leftover dS into state0 is simply not emitted (truncated BPTT,
// state0 is treated as a constant). The time-parallel backward is NOT used (it assumes a zero start).
template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rwkv7_wkv_backward_stateful_cuda(
    const at::Tensor& r_BTHK,
    const at::Tensor& k_BTHK,
    const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK,
    const at::Tensor& a_BTHK,
    const at::Tensor& k_deformed_BTHK,
    const at::Tensor& skip_BT,
    const at::Tensor& state_checkpoints_BLHKK,
    const at::Tensor& grad_BTHK
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = r_BTHK.size(3);
    const int L = state_checkpoints_BLHKK.size(1);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CUDA);
    const F* r_ptr = (F*)r_BTHK.data_ptr();
    const F* k_ptr = (F*)k_BTHK.data_ptr();
    const F* v_ptr = (F*)v_BTHK.data_ptr();
    const float* w_ptr = w_BTHK.data_ptr<float>();
    const F* a_ptr = (F*)a_BTHK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_BTHK.data_ptr();
    const bool* skip_ptr = (bool*)skip_BT.data_ptr();
    const float* state_checkpoints_ptr = state_checkpoints_BLHKK.data_ptr<float>();
    const F* grad_ptr = (F*)grad_BTHK.data_ptr();
    at::Tensor r_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor k_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor v_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor w_grad_BTHK = torch::zeros_like(r_BTHK, torch::dtype(torch::kFloat32));
    at::Tensor a_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor k_deformed_grad_BTHK = torch::zeros_like(r_BTHK);
    F* r_grad_ptr = (F*)r_grad_BTHK.data_ptr();
    F* k_grad_ptr = (F*)k_grad_BTHK.data_ptr();
    F* v_grad_ptr = (F*)v_grad_BTHK.data_ptr();
    float* w_grad_ptr = w_grad_BTHK.data_ptr<float>();
    F* a_grad_ptr = (F*)a_grad_BTHK.data_ptr();
    F* k_deformed_grad_ptr = (F*)k_deformed_grad_BTHK.data_ptr();

    dim3 block_dim(K, K);
    dim3 grid_dim(B, H);
    rwkv7_wkv_backward_kernel<CHUNK_LEN><<<grid_dim, block_dim>>>(B, T, H, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr,
    grad_ptr, L, state_checkpoints_ptr, r_grad_ptr, k_grad_ptr, v_grad_ptr, w_grad_ptr, a_grad_ptr, k_deformed_grad_ptr);

    return std::make_tuple(r_grad_BTHK, k_grad_BTHK, v_grad_BTHK, w_grad_BTHK, a_grad_BTHK, k_deformed_grad_BTHK);
}

const int CHECKPOINT_LEN = 32;
TORCH_LIBRARY_IMPL(rwkv, CUDA, m) {
    m.impl("rwkv7_wkv_forward_float", &rwkv7_wkv_forward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_wkv_backward_float", &rwkv7_wkv_backward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_wkv_forward_bfloat16", &rwkv7_wkv_forward_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_wkv_backward_bfloat16", &rwkv7_wkv_backward_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_wkv_forward_half", &rwkv7_wkv_forward_cuda<CHECKPOINT_LEN, __half>);
    m.impl("rwkv7_wkv_backward_half", &rwkv7_wkv_backward_cuda<CHECKPOINT_LEN, __half>);
    m.impl("rwkv7_wkv_forward_stateful_float", &rwkv7_wkv_forward_stateful_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_wkv_backward_stateful_float", &rwkv7_wkv_backward_stateful_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_wkv_forward_stateful_bfloat16", &rwkv7_wkv_forward_stateful_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_wkv_backward_stateful_bfloat16", &rwkv7_wkv_backward_stateful_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_wkv_forward_stateful_half", &rwkv7_wkv_forward_stateful_cuda<CHECKPOINT_LEN, __half>);
    m.impl("rwkv7_wkv_backward_stateful_half", &rwkv7_wkv_backward_stateful_cuda<CHECKPOINT_LEN, __half>);
}
}