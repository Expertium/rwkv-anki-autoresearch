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

// ===================== FUSED QAT KERNELS (per-step WKV + full-matrix int-N state quant) =====================
// Fuses the slow Python `quant_aware_rwkv7` loop (rwkv_ops.py): each step does the WKV update, then
// round-trips the recurrent state through symmetric per-batch int-N quant (STE) before carrying it. It
// matches `fake_quant_state` EXACTLY: amax over (H,K,K) per batch element, round-half-to-even, clamp to
// +-qmax, and the same nan_to_num(nan=0,+-inf=+-1e4) safeguard. Because the amax couples both heads, the
// FORWARD uses one block per batch element (looping over H) so the cross-head reduction is intra-block; it
// PERSISTS the per-step scale (scale_BT) so the BACKWARD can stay per-(b,h) -- once the scale is known,
// quantizing a head's entries is independent of the other head. STE => the quant is transparent to the
// gradient, so the backward's grad math is identical to the plain WKV backward; only the state trajectory
// it recomputes must re-apply the same quant. qmax: int8=127, int4=7, int2=1.

__device__ inline float qat_sanitize(float x) {
    // torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4) -- the _STATE_NAN_CAP=1e4 safeguard.
    if (isfinite(x)) return x;
    if (x > 0.0f) return 1e4f;
    if (x < 0.0f) return -1e4f;
    return 0.0f;  // NaN
}

// Block-wide max reduction; every thread receives the result. smem must hold >= #warps floats (<=32).
__device__ inline float qat_blockmax(float v, float* smem) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));  // warp all-reduce
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (tid == 0) {
        int nwarps = (nthreads + 31) / 32;
        float m = smem[0];
        for (int i = 1; i < nwarps; i++) m = fmaxf(m, smem[i]);
        smem[0] = m;
    }
    __syncthreads();
    return smem[0];
}

#define QAT_MAX_H 8

// ---- PRODUCT-QUANTIZATION codebook for the rank-1 factor DIRECTIONS (roles 0=u, 1=v). GLOBAL + fixed
// (uploaded once via rwkv7_set_pq_codebook), so qat_lr_rank1 can codebook-encode the split-sqrt factors
// INSTEAD of per-column int-N quant -- the train==deploy analog of the engine PqCodebook::encode_decode.
// When c_pq_active==0 (default, never uploaded) qat_lr_rank1 keeps its int-N path -> existing int QAT
// runs are byte-for-byte unaffected. Layout: g_pq_cb[((role*m + pos)*ncent + c)*sub + j], roles 0,1 only.
// Size 8192 = 2 roles * ncent(<=256) * K(=m*sub=16). Constant flag/metadata; array in device global mem. ----
__constant__ int c_pq_active = 0;
__constant__ int c_pq_m = 0;
__constant__ int c_pq_subdim = 0;
__constant__ int c_pq_ncent = 0;
// JOINT-UV mode (task23): the single catalog holds concat(u_unit, v_unit) 2K-dim entries; ONE index
// per head selects BOTH factor directions, each half rescaled by its own (quantized) norm. subdim=2K,
// layout g_pq_cb[c*subdim + j] (no role blocks). Mirror of engine PqCodebook::encode_decode_joint.
__constant__ int c_pq_joint = 0;
// task24: warm-started joint search (previous step's pick primes the pruning bound). Provably
// pick-identical; RWKV_QAT_NO_WARM=1 (read at codebook upload) disables it for bitwise A/B / fallback.
__constant__ int c_pq_warm = 1;
__device__ float g_pq_cb[32768];   // grown 8192 -> 32768 for joint-uv (ncent<=1024 x subdim 32)
// ---- NORM QUANT for the PQ branch (train==deploy analog of engine RWKV_PQ_NORM_BITS): reconstruct with
// the factor norm quantized to n bits, log2-uniform over [c_nq_lo, c_nq_hi] octaves; centroid MATCHING
// still uses the TRUE norm (mirrors PqCodebook::encode_decode). c_nq_levels = 2^bits - 1; 0 = off. ----
__constant__ float c_nq_levels = 0.f;
__constant__ float c_nq_lo = 0.f;
__constant__ float c_nq_hi = 0.f;
// ---- LEARNABLE WKV codebook (RWKV_QAT_PQ_LEARN): when c_pq_learn=1 the lr backward kernel accumulates
// dL/d(centroid) into g_pq_cb_grad via atomicAdd (order nondeterministic ~1e-7). The chunk re-run records
// each step's selected centroid indices + quantized norms; at the STE consumption point the contrib
// gradient dL/dQ_t is outer-product-reduced against the counterpart factor. Python fetches the buffer
// into the codebook Parameter's .grad each step and re-uploads the stepped codebook. ----
__constant__ int c_pq_learn = 0;
__device__ float g_pq_cb_grad[32768];

// Quantize a factor norm exactly like engine encode_decode: t -> round (HALF-AWAY, matches Rust
// f32::round) -> clamp -> exp2. Caller guarantees norm is finite and >= 1e-20.
__device__ inline float nq_quant_norm(float norm) {
    float t = (log2f(norm) - c_nq_lo) / (c_nq_hi - c_nq_lo);
    float q = roundf(t * c_nq_levels);
    q = fminf(fmaxf(q, 0.f), c_nq_levels);
    return exp2f(c_nq_lo + q / c_nq_levels * (c_nq_hi - c_nq_lo));
}

// In-place: normalize `col` (K-dim) to unit, replace each of m sub-vectors by its nearest of ncent
// centroids, rescale by the original norm. EXACT mirror of engine model.rs PqCodebook::encode_decode.
// NOTE: no longer called on the hot path -- qat_lr_rank1 inlines a BLOCK-PARALLEL version of this search
// (same per-distance FMA order + first-strict-min centroid choice, so bit-identical results). Kept as the
// readable serial reference of the deploy algorithm.
__device__ inline void pq_encode_decode(int role, float* col, int K) {
    float nn = 0.f;
    for (int i = 0; i < K; i++) nn += col[i] * col[i];
    float norm = sqrtf(nn);
    if (!isfinite(norm) || norm < 1e-20f) return;
    float inv = 1.0f / norm;
    if (c_nq_levels > 0.f) norm = nq_quant_norm(norm);
    int m = c_pq_m, sub = c_pq_subdim, ncent = c_pq_ncent;
    for (int p = 0; p < m; p++) {
        int s = p * sub;
        const float* cents = &g_pq_cb[((role * m + p) * ncent) * sub];
        int best = 0; float bestd = INFINITY;
        for (int c = 0; c < ncent; c++) {
            const float* cc = cents + c * sub;
            float d = 0.f;
            for (int j = 0; j < sub; j++) { float diff = col[s + j] * inv - cc[j]; d += diff * diff; }
            if (d < bestd) { bestd = d; best = c; }
        }
        const float* bc = cents + best * sub;
        for (int j = 0; j < sub; j++) col[s + j] = bc[j] * norm;
    }
}

// ---- Rank-1 int-N low-rank truncation of one head's KxK state (matches deploy compress_wkv_state r==1
// fast path + quant_factor_percol_inplace). The whole block cooperates on ONE KxK matrix: `a_val` is this
// thread's entry (x=threadIdx.y row, y=threadIdx.x col), the K vector ops run on lanes tid<K, and the
// return value is this thread's rank-1 reconstruction entry recon[x][y] = uf_q[x]*vf_q[y]. HALF-AWAY
// rounding (roundf) to match Rust f64::round. All threads call it (it has __syncthreads inside).
// rec_idx/rec_norm (nullable): learnable-codebook recording — the PQ branch writes the 2m selected
// centroid indices (rec_idx[role*m+p], -1 = column not quantized) and the 2 reconstruction norms.
// warm (nullable, joint path only): per-thread register carrying the PREVIOUS step's winning joint
// centroid across the caller's sequential t-loop (-1 = none). The state drifts slowly step-to-step,
// so it is a near-optimal distance bound that lets the scan prune most of the 1024-entry catalog.
// Provably pick-identical to the full scan (see the joint branch). Uniform across the block. ----
__device__ inline float qat_lr_rank1(float a_val, int K, float qmax, int* rec_idx = nullptr,
                                     float* rec_norm = nullptr, int* warm = nullptr) {
    const int x = threadIdx.y, y = threadIdx.x;
    const int tid = x * blockDim.x + y;
    const int nthreads = blockDim.x * blockDim.y;
    __shared__ float As[32 * 32];
    __shared__ float uvec[32], tvec[32], nvec[32], ufq[32], vfq[32];
    __shared__ float red[32];
    __shared__ int s_bidx[32];
    __shared__ float sc_nrm, sc_dot, sc_su, sc_sv, sc_sgn, sc_nu, sc_nv;
    __shared__ int sc_ok, sc_best;
    As[x * K + y] = a_val;
    float amax = qat_blockmax(fabsf(a_val), red);          // block max-abs (has __syncthreads inside)
    float scale = (isfinite(amax) && amax > 1e-30f) ? amax : 1.0f;
    float invs = 1.0f / scale;
    // ---- WARP-0 REGION. K <= 32, so every participant (tid < K workers + the tid==0 scalar steps) lives
    // in warp 0: the whole power iteration + factor extraction can synchronize with __syncwarp instead of
    // block-wide barriers (the other warps just wait at the single publishing __syncthreads below). Same
    // arithmetic, same participants, same order as the original block-wide version -> bit-identical.
    if (tid < 32) {
        if (tid < K) uvec[tid] = rsqrtf((float)K);             // u init = 1/sqrt(K)
        __syncwarp();
        for (int it = 0; it < 64; it++) {
            if (tid < K) {                                     // atu[j] = invs * sum_x As[x,j]*u[x]
                int j = tid; float s = 0.f;
                for (int xx = 0; xx < K; xx++) s += As[xx * K + j] * uvec[xx];
                tvec[j] = s * invs;
            }
            __syncwarp();
            if (tid < K) {                                     // nu[i] = invs * sum_y As[i,y]*atu[y]
                int i = tid; float s = 0.f;
                for (int yy = 0; yy < K; yy++) s += As[i * K + yy] * tvec[yy];
                nvec[i] = s * invs;
            }
            __syncwarp();
            if (tid == 0) { float nn = 0.f; for (int i = 0; i < K; i++) nn += nvec[i] * nvec[i]; sc_nrm = sqrtf(nn); }
            __syncwarp();
            float nrm = sc_nrm;
            if (!isfinite(nrm) || nrm < 1e-30f) break;         // uniform across warp 0 -> safe
            if (tid < K) nvec[tid] = nvec[tid] / nrm;
            __syncwarp();
            if (tid == 0) { float d = 0.f; for (int i = 0; i < K; i++) d += uvec[i] * nvec[i]; sc_dot = fabsf(d); }
            __syncwarp();
            if (tid < K) uvec[tid] = nvec[tid];
            __syncwarp();
            if (1.0f - sc_dot < 1e-7f) break;                  // uniform
        }
        if (tid < K) {                                         // v_un[j] = sum_x As[x,j]*u[x] (ORIGINAL A)
            int j = tid; float s = 0.f;
            for (int xx = 0; xx < K; xx++) s += As[xx * K + j] * uvec[xx];
            tvec[j] = s;
        }
        __syncwarp();
        if (tid == 0) {
            float ss = 0.f; for (int j = 0; j < K; j++) ss += tvec[j] * tvec[j];
            float sigma = sqrtf(ss);
            sc_nrm = sigma;
            int uok = 1; for (int i = 0; i < K; i++) if (!isfinite(uvec[i])) uok = 0;
            sc_ok = (sigma > 1e-20f && isfinite(sigma) && uok) ? 1 : 0;
        }
        __syncwarp();
        float sigma = sc_nrm; int ok = sc_ok;
        if (tid < K) {                                         // split-sqrt factors uf=u*sj, vf=(v_un/sigma)*sj
            float ufi = 0.f, vfi = 0.f;
            if (ok) { float sj = sqrtf(sigma); ufi = uvec[tid] * sj; vfi = (tvec[tid] / sigma) * sj; }
            ufq[tid] = ufi; vfq[tid] = vfi;
        }
        __syncwarp();
        if (!c_pq_active) {                                    // ---- int-N per-column quant (default deploy path)
            if (tid == 0) {                                    // per-column int-N scales (amax/qmax, clamp 1e-12)
                float au = 0.f, av = 0.f;
                for (int i = 0; i < K; i++) { au = fmaxf(au, fabsf(ufq[i])); av = fmaxf(av, fabsf(vfq[i])); }
                sc_su = fmaxf(au / qmax, 1e-12f); sc_sv = fmaxf(av / qmax, 1e-12f);
            }
            __syncwarp();
            if (tid < K) {                                     // quantize (round HALF-AWAY = deploy f64::round)
                float su = sc_su, sv = sc_sv;
                ufq[tid] = fminf(fmaxf(roundf(ufq[tid] / su), -qmax), qmax) * su;
                vfq[tid] = fminf(fmaxf(roundf(vfq[tid] / sv), -qmax), qmax) * sv;
            }
        } else {                                               // ---- PQ prep: sign-canon + direction norms
            if (tid == 0) {                                    // sign-canon: flip so u's dominant-abs entry >= 0
                float am = 0.f, sgn = 1.f;
                for (int i = 0; i < K; i++) { float av = fabsf(ufq[i]); if (av > am) { am = av; sgn = ufq[i] >= 0.f ? 1.f : -1.f; } }
                sc_sgn = sgn;
            }
            __syncwarp();
            if (sc_sgn < 0.f && tid < K) { ufq[tid] = -ufq[tid]; vfq[tid] = -vfq[tid]; }
            __syncwarp();
            if (tid == 0) {                                    // norms, same serial order as pq_encode_decode
                float nu = 0.f, nv = 0.f;
                for (int i = 0; i < K; i++) nu += ufq[i] * ufq[i];
                for (int i = 0; i < K; i++) nv += vfq[i] * vfq[i];
                sc_nu = sqrtf(nu); sc_nv = sqrtf(nv);
            }
        }
    }
    __syncthreads();                                       // publish ufq/vfq (+ sc_nu/sc_nv) to the whole block
    if (c_pq_active) {
        // ---- BLOCK-PARALLEL PQ codebook search (replaces the single-threaded pq_encode_decode calls;
        // bit-identical: per-distance FMA order unchanged, and the (dist, index) reduction keeps the
        // FIRST strict minimum exactly like the serial "d < bestd" scan -- ties resolve to the lower c,
        // all-NaN/inf distance sets fall back to centroid 0 just like the serial init best=0).
        if (rec_idx != nullptr && tid == 0) {                  // learnable-cb recording: default -1/0
            for (int i = 0; i < 2 * c_pq_m; i++) rec_idx[i] = -1;
            rec_norm[0] = 0.f; rec_norm[1] = 0.f;
        }
        if (c_pq_joint) {
            // ---- JOINT-UV: one 2K-dim code selects BOTH directions (engine encode_decode_joint
            // mirror: u-half distances first then v-half, first-strict-min, per-half norm rescale).
            // A degenerate norm on EITHER factor skips the pair (block-uniform: sc_nu/sc_nv shared).
            float nu = sc_nu, nv = sc_nv;
            if (isfinite(nu) && nu >= 1e-20f && isfinite(nv) && nv >= 1e-20f) {
                float iu = 1.0f / nu, iv = 1.0f / nv;          // matching by the TRUE norms
                float qu = (c_nq_levels > 0.f) ? nq_quant_norm(nu) : nu;
                float qv = (c_nq_levels > 0.f) ? nq_quant_norm(nv) : nv;
                if (rec_norm != nullptr && tid == 0) { rec_norm[0] = qu; rec_norm[1] = qv; }
                // Warm-started, partial-distance-pruned scan (task24). Candidate order per thread:
                // the previous step's winner wc first (ALL threads, exact bound), then this thread's
                // stride share. ONE loop body -> one FP compilation -> the warm bound is bit-equal to
                // what the plain scan computes for that centroid. PICK-IDENTICAL to the serial scan:
                // (a) the survival predicate below equals the update predicate, (b) d is a monotone
                // non-decreasing sum of squares, so a pruned candidate's final d can never satisfy it,
                // (c) (d, then lower c) tie order == the serial first-strict-min semantics.
                const int wc = (warm != nullptr && c_pq_warm) ? *warm : -1;
                float bd = INFINITY; int bi = 0x7fffffff;
                for (int s = (wc >= 0 ? -1 : tid); s < c_pq_ncent; s = (s < 0 ? tid : s + nthreads)) {
                    const int c = (s < 0) ? wc : s;
                    if (s >= 0 && c == wc) continue;           // warm already scanned at s == -1
                    const float* cc = &g_pq_cb[c * c_pq_subdim];
                    float d = 0.f;
                    bool alive = true;
                    for (int j0 = 0; j0 < K && alive; j0 += 8) {   // u-half, prune every 8 dims
                        const int je = (j0 + 8 < K) ? j0 + 8 : K;
                        for (int j = j0; j < je; j++) { float diff = ufq[j] * iu - cc[j]; d += diff * diff; }
                        alive = (d < bd) || (d == bd && c < bi);
                    }
                    for (int j0 = 0; j0 < K && alive; j0 += 8) {   // v-half (same serial-scan order)
                        const int je = (j0 + 8 < K) ? j0 + 8 : K;
                        for (int j = j0; j < je; j++) { float diff = vfq[j] * iv - cc[K + j]; d += diff * diff; }
                        alive = (d < bd) || (d == bd && c < bi);
                    }
                    if (alive) { bd = d; bi = c; }             // final alive == update predicate
                }
                for (int o = 16; o > 0; o >>= 1) {             // warp argmin (same as the role path)
                    float od = __shfl_down_sync(FULL_MASK, bd, o);
                    int oi = __shfl_down_sync(FULL_MASK, bi, o);
                    if (od < bd || (od == bd && oi < bi)) { bd = od; bi = oi; }
                }
                if ((tid & 31) == 0) { red[tid >> 5] = bd; s_bidx[tid >> 5] = bi; }
                __syncthreads();
                if (tid == 0) {                                // cross-warp argmin + all-bad fallback
                    int nwarps = (nthreads + 31) / 32;
                    float fd = red[0]; int fi = s_bidx[0];
                    for (int i = 1; i < nwarps; i++) {
                        if (red[i] < fd || (red[i] == fd && s_bidx[i] < fi)) { fd = red[i]; fi = s_bidx[i]; }
                    }
                    sc_best = (fi == 0x7fffffff) ? 0 : fi;
                    if (rec_idx != nullptr) rec_idx[0] = sc_best;
                }
                __syncthreads();
                if (tid < K) {
                    ufq[tid] = g_pq_cb[sc_best * c_pq_subdim + tid] * qu;
                    vfq[tid] = g_pq_cb[sc_best * c_pq_subdim + K + tid] * qv;
                }
                __syncthreads();                               // write-back + red/s_bidx reuse fence
                if (warm != nullptr) *warm = sc_best;          // every thread updates its own register
            }
            return ufq[x] * vfq[y];
        }
        for (int role = 0; role < 2; role++) {
            float norm = (role == 0) ? sc_nu : sc_nv;
            if (!isfinite(norm) || norm < 1e-20f) continue;    // mirror the early return (block-uniform)
            float inv = 1.0f / norm;                           // matching by the TRUE norm (engine parity)
            if (c_nq_levels > 0.f) norm = nq_quant_norm(norm); // reconstruct with the quantized norm
            if (rec_norm != nullptr && tid == 0) rec_norm[role] = norm;
            float* col = (role == 0) ? ufq : vfq;
            for (int p = 0; p < c_pq_m; p++) {
                int s = p * c_pq_subdim;
                const float* cents = &g_pq_cb[((role * c_pq_m + p) * c_pq_ncent) * c_pq_subdim];
                float bd = INFINITY; int bi = 0x7fffffff;
                for (int c = tid; c < c_pq_ncent; c += nthreads) {
                    const float* cc = cents + c * c_pq_subdim;
                    float d = 0.f;
                    for (int j = 0; j < c_pq_subdim; j++) { float diff = col[s + j] * inv - cc[j]; d += diff * diff; }
                    if (d < bd) { bd = d; bi = c; }
                }
                for (int o = 16; o > 0; o >>= 1) {             // warp argmin
                    float od = __shfl_down_sync(FULL_MASK, bd, o);
                    int oi = __shfl_down_sync(FULL_MASK, bi, o);
                    if (od < bd || (od == bd && oi < bi)) { bd = od; bi = oi; }
                }
                if ((tid & 31) == 0) { red[tid >> 5] = bd; s_bidx[tid >> 5] = bi; }
                __syncthreads();
                if (tid == 0) {                                // cross-warp argmin + all-bad fallback
                    int nwarps = (nthreads + 31) / 32;
                    float fd = red[0]; int fi = s_bidx[0];
                    for (int i = 1; i < nwarps; i++) {
                        if (red[i] < fd || (red[i] == fd && s_bidx[i] < fi)) { fd = red[i]; fi = s_bidx[i]; }
                    }
                    sc_best = (fi == 0x7fffffff) ? 0 : fi;
                    if (rec_idx != nullptr) rec_idx[role * c_pq_m + p] = sc_best;
                }
                __syncthreads();
                if (tid < c_pq_subdim) col[s + tid] = cents[sc_best * c_pq_subdim + tid] * norm;
                __syncthreads();                               // col write-back + red/s_bidx reuse fence
            }
        }
    }
    return ufq[x] * vfq[y];                                // recon[x][y] = uf_q[x] * vf_q[y]
}

// Stage-A validation op: apply qat_lr_rank1 to each (b,h) KxK of a state tensor. grid(B,H), block(K,K).
template <typename F>
__global__ void rwkv7_lr_trunc_test_kernel(const int B, const int H,
    const F* __restrict__ state_BHKK, F* __restrict__ out_BHKK, const float qmax) {
    const int K = blockDim.x;
    const int b = blockIdx.x, h = blockIdx.y;
    const int x = threadIdx.y, y = threadIdx.x;
    int idx = ((b * H + h) * K + x) * K + y;
    float recon = qat_lr_rank1(to_float<F>(state_BHKK[idx]), K, qmax);
    out_BHKK[idx] = to_F<F>(recon);
}

template <int CHUNK_LEN=32, typename F>
__global__ void rwkv7_wkv_qat_forward_kernel(
    const int B, const int T, const int H,
    const F* __restrict__ r_BTHK, const F* __restrict__ k_BTHK, const F* __restrict__ v_BTHK,
    const float* __restrict__ w_BTHK, const F* __restrict__ a_BTHK, const F* __restrict__ k_deformed_BTHK,
    const bool* __restrict__ skip_BT, const float qmax,
    F* __restrict__ out_BTHK, const int L,
    float* __restrict__ state_checkpoints_BLHKK, float* __restrict__ scale_BT
    ) {
    const int K = blockDim.x;
    const int b = blockIdx.x;
    const int x = threadIdx.y;  // row
    const int y = threadIdx.x;  // col
    __shared__ float smem[32];

    float st[QAT_MAX_H];                     // this thread's state entry (x,y) for each head (carried, quantized)
    for (int h = 0; h < H; h++) st[h] = 0.0f;

    for (int t = 0; t < T; t++) {
        if (t % CHUNK_LEN == 0) {            // checkpoint the (quantized) state entering this step
            int l = t / CHUNK_LEN;
            for (int h = 0; h < H; h++)
                state_checkpoints_BLHKK[get_index4(b, l, h, x, y, L, H, K, K)] = st[h];
        }
        bool skip = skip_BT[get_index1(b, t, T)];
        float newv[QAT_MAX_H];               // raw post-update state (pre-quant) per head
        for (int h = 0; h < H; h++) {
            int64_t gy = get_index3(b, t, h, y, T, H, K);
            int64_t gx = get_index3(b, t, h, x, T, H, K);
            float r_y = to_float<F>(r_BTHK[gy]);
            float k_y = to_float<F>(k_BTHK[gy]);
            float v_x = to_float<F>(v_BTHK[gx]);
            float w_y = w_BTHK[gy];
            float a_y = to_float<F>(a_BTHK[gy]);
            float kd_y = to_float<F>(k_deformed_BTHK[gy]);
            float s = st[h];
            float decayed = s * w_y;
            float k_dot = s * kd_y;
            for (int o = K / 2; o > 0; o /= 2) k_dot += __shfl_down_sync(FULL_MASK, k_dot, o, K);
            k_dot = __shfl_sync(FULL_MASK, k_dot, 0, K);
            float ns = decayed - k_dot * a_y * kd_y + v_x * k_y;
            float r_dot = ns * r_y;          // output from the RAW post-update state (matches single_timestep)
            for (int o = K / 2; o > 0; o /= 2) r_dot += __shfl_down_sync(FULL_MASK, r_dot, o, K);
            if (y == 0) out_BTHK[gx] = to_F<F>(r_dot);
            newv[h] = ns;
        }
        // per-batch amax over sanitized new states (both heads) -> shared scale
        float local = 0.0f;
        for (int h = 0; h < H; h++) local = fmaxf(local, fabsf(qat_sanitize(newv[h])));
        float amax = qat_blockmax(local, smem);
        float scale = fmaxf(amax / qmax, 1e-12f);
        if (x == 0 && y == 0) scale_BT[get_index1(b, t, T)] = scale;
        for (int h = 0; h < H; h++) {        // quantize (STE forward) then carry; skip keeps the old state
            float sv = qat_sanitize(newv[h]);
            float q = fminf(fmaxf(rintf(sv / scale), -qmax), qmax) * scale;
            st[h] = skip ? st[h] : q;
        }
    }
}

// Per-(b,h) backward: identical grad math to rwkv7_wkv_backward_kernel, but the state trajectory it
// recomputes from checkpoints re-applies the per-step quant using the precomputed cross-head scale_BT.
template <int CHUNK_LEN=32, typename F>
__global__ void rwkv7_wkv_qat_backward_kernel(
    const int B, const int T, const int H,
    const F* __restrict__ r_BTHK, const F* __restrict__ k_BTHK, const F* __restrict__ v_BTHK,
    const float* __restrict__ w_BTHK, const F* __restrict__ a_BTHK, const F* __restrict__ k_deformed_BTHK,
    const bool* __restrict__ skip_BT, const float qmax, const float* __restrict__ scale_BT,
    const F* __restrict__ grad_BTHK, const int L, const float* __restrict__ state_checkpoints_BLHKK,
    F* __restrict__ r_grad_BTHK, F* __restrict__ k_grad_BTHK, F* __restrict__ v_grad_BTHK,
    float* __restrict__ w_grad_BTHK, F* __restrict__ a_grad_BTHK, F* __restrict__ k_deformed_grad_BTHK
    ) {
    const int K = blockDim.x;
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
        // recompute the states from the checkpoint, re-applying the per-step quant
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

            float state_xy_decayed = state_xy * w_y;
            float state_k_dot = state_xy * k_deformed_y;
            for (int offset = K / 2; offset > 0; offset /= 2) {
                state_k_dot += __shfl_down_sync(FULL_MASK, state_k_dot, offset, K);
            }
            state_k_dot = __shfl_sync(FULL_MASK, state_k_dot, 0, K);
            state_xy = state_xy_decayed - state_k_dot * a_y * k_deformed_y;
            state_xy += v_x * k_y;
            state_xy_chunk[c] = state_xy;    // raw post-update (used by the grad formulas; matches forward out)
            // QAT: carry the quantized state (per-head; cross-head scale precomputed in the forward)
            float sc = scale_BT[get_index1(b, t, T)];
            float sv = qat_sanitize(state_xy);
            float q = fminf(fmaxf(rintf(sv / sc), -qmax), qmax) * sc;
            state_xy = skip ? in_state_xy : q;
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
std::tuple<at::Tensor, at::Tensor, at::Tensor> rwkv7_wkv_qat_forward_cuda(
    const at::Tensor& r_BTHK, const at::Tensor& k_BTHK, const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK, const at::Tensor& a_BTHK, const at::Tensor& k_deformed_BTHK,
    const at::Tensor& skip_BT, double qmax
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = r_BTHK.size(3);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(H <= QAT_MAX_H);
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
    at::Tensor scale_BT = torch::empty({B, T}, r_BTHK.options().dtype(torch::kFloat32)).requires_grad_(false);
    float* scale_ptr = scale_BT.data_ptr<float>();

    dim3 block_dim(K, K);
    dim3 grid_dim(B);
    rwkv7_wkv_qat_forward_kernel<CHUNK_LEN, F><<<grid_dim, block_dim>>>(
        B, T, H, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr, (float)qmax,
        out_ptr, L, state_checkpoints_ptr, scale_ptr);
    return std::make_tuple(out_BTHK, state_checkpoints_BLHKK, scale_BT);
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rwkv7_wkv_qat_backward_cuda(
    const at::Tensor& r_BTHK, const at::Tensor& k_BTHK, const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK, const at::Tensor& a_BTHK, const at::Tensor& k_deformed_BTHK,
    const at::Tensor& skip_BT, const at::Tensor& state_checkpoints_BLHKK, const at::Tensor& scale_BT,
    const at::Tensor& grad_BTHK, double qmax
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
    const float* scale_ptr = scale_BT.data_ptr<float>();
    const float* state_checkpoints_ptr = state_checkpoints_BLHKK.data_ptr<float>();
    const F* grad_ptr = (F*)grad_BTHK.data_ptr();
    at::Tensor r_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor k_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor v_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor w_grad_BTHK = torch::empty_like(r_BTHK, torch::dtype(torch::kFloat32));
    at::Tensor a_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor k_deformed_grad_BTHK = torch::empty_like(r_BTHK);
    F* r_grad_ptr = (F*)r_grad_BTHK.data_ptr();
    F* k_grad_ptr = (F*)k_grad_BTHK.data_ptr();
    F* v_grad_ptr = (F*)v_grad_BTHK.data_ptr();
    float* w_grad_ptr = w_grad_BTHK.data_ptr<float>();
    F* a_grad_ptr = (F*)a_grad_BTHK.data_ptr();
    F* k_deformed_grad_ptr = (F*)k_deformed_grad_BTHK.data_ptr();

    dim3 block_dim(K, K);
    dim3 grid_dim(B, H);
    rwkv7_wkv_qat_backward_kernel<CHUNK_LEN, F><<<grid_dim, block_dim>>>(
        B, T, H, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, skip_ptr, (float)qmax, scale_ptr,
        grad_ptr, L, state_checkpoints_ptr,
        r_grad_ptr, k_grad_ptr, v_grad_ptr, w_grad_ptr, a_grad_ptr, k_deformed_grad_ptr);
    return std::make_tuple(r_grad_BTHK, k_grad_BTHK, v_grad_BTHK, w_grad_BTHK, a_grad_BTHK, k_deformed_grad_BTHK);
}

// ---- FUSED RANK-1 int-N LOW-RANK QAT (matches deploy compress_wkv_state rank-1). Per-head (grid B,H),
// NO cross-head coupling (the rank-1 truncation is self-contained per KxK matrix), so no scale_BT. Each
// step: WKV update -> output from raw post-update state -> qat_lr_rank1 truncation (STE) -> carry. STE
// makes the truncation gradient-transparent, so the backward is the plain WKV backward over the truncated
// trajectory (the recompute re-applies the truncation). Checkpoints store the TRUNCATED carried state. ----
template <int CHUNK_LEN=32, typename F>
__global__ void rwkv7_wkv_qat_lr_forward_kernel(
    const int B, const int T, const int H,
    const F* __restrict__ r_BTHK, const F* __restrict__ k_BTHK, const F* __restrict__ v_BTHK,
    const float* __restrict__ w_BTHK, const F* __restrict__ a_BTHK, const F* __restrict__ k_deformed_BTHK,
    const bool* __restrict__ skip_BT, const float qmax,
    F* __restrict__ out_BTHK, const int L, float* __restrict__ state_checkpoints_BLHKK
    ) {
    const int K = blockDim.x;
    const int b = blockIdx.x, h = blockIdx.y;
    const int x = threadIdx.y, y = threadIdx.x;
    float st = 0.0f;                                       // this thread's carried (truncated) state entry
    int warm_joint = -1;                                   // previous step's joint-cb pick (block-uniform)
    int64_t global_y = get_index3(b, 0, h, y, T, H, K);
    int64_t global_x = get_index3(b, 0, h, x, T, H, K);
    for (int t = 0; t < T; t++) {
        if (t % CHUNK_LEN == 0)
            state_checkpoints_BLHKK[get_index4(b, t / CHUNK_LEN, h, x, y, L, H, K, K)] = st;
        float r_y = to_float<F>(r_BTHK[global_y]);
        float k_y = to_float<F>(k_BTHK[global_y]);
        float v_x = to_float<F>(v_BTHK[global_x]);
        float w_y = w_BTHK[global_y];
        float a_y = to_float<F>(a_BTHK[global_y]);
        float kd_y = to_float<F>(k_deformed_BTHK[global_y]);
        bool skip = skip_BT[get_index1(b, t, T)];
        float in_st = st;
        float decayed = st * w_y;
        float k_dot = st * kd_y;
        for (int o = K / 2; o > 0; o /= 2) k_dot += __shfl_down_sync(FULL_MASK, k_dot, o, K);
        k_dot = __shfl_sync(FULL_MASK, k_dot, 0, K);
        float ns = decayed - k_dot * a_y * kd_y + v_x * k_y;
        float r_dot = ns * r_y;                            // output from RAW post-update state
        for (int o = K / 2; o > 0; o /= 2) r_dot += __shfl_down_sync(FULL_MASK, r_dot, o, K);
        if (y == 0) out_BTHK[global_x] = to_F<F>(r_dot);
        // Skip-step elision: on skip (query) rows the carried state reverts to in_st and the truncation
        // result is used NOWHERE (out already came from the raw ns) -- so don't compute it. ~half of all
        // rows are query duplicates => ~2x less rank-1+quant work. skip is uniform per (b,t) across the
        // block, so branching around the barrier-bearing call is safe. Bit-identical outputs.
        if (skip) {
            st = in_st;
        } else {
            st = qat_lr_rank1(ns, K, qmax, nullptr, nullptr, &warm_joint);  // rank-1 int-N truncation (STE forward)
        }
        global_x += H * K;
        global_y += H * K;
    }
}

// Backward: recompute the trajectory from checkpoints re-applying qat_lr_rank1; STE => grad math identical
// to plain WKV backward. Per-(b,h), sequential. Mirrors rwkv7_wkv_qat_backward_kernel with the truncation.
template <int CHUNK_LEN=32, typename F>
__global__ void rwkv7_wkv_qat_lr_backward_kernel(
    const int B, const int T, const int H,
    const F* __restrict__ r_BTHK, const F* __restrict__ k_BTHK, const F* __restrict__ v_BTHK,
    const float* __restrict__ w_BTHK, const F* __restrict__ a_BTHK, const F* __restrict__ k_deformed_BTHK,
    const bool* __restrict__ skip_BT, const float qmax,
    const F* __restrict__ grad_BTHK, const int L, const float* __restrict__ state_checkpoints_BLHKK,
    F* __restrict__ r_grad_BTHK, F* __restrict__ k_grad_BTHK, F* __restrict__ v_grad_BTHK,
    float* __restrict__ w_grad_BTHK, F* __restrict__ a_grad_BTHK, F* __restrict__ k_deformed_grad_BTHK
    ) {
    const int K = blockDim.x;
    __shared__ float KK_state[32 * (32 + 1)];
    __shared__ float KK_state_prev[32 * (32 + 1)];
    __shared__ float KK_dS[32 * (32 + 1)];
    __shared__ float KK_grad_decay[32 * (32 + 1)];
    __shared__ float K_k_deformed[32];
    __shared__ float K_a[32];
    // learnable WKV cb (c_pq_learn): per-step recorded centroid picks + recon norms from the chunk
    // re-run, a staging tile for dL/dQ_t and the step's reconstructed factors (see grad block below).
    __shared__ int rec_idx_chunk[CHUNK_LEN * 8];
    __shared__ float rec_norm_chunk[CHUNK_LEN * 2];
    __shared__ float KK_G[32 * (32 + 1)];
    __shared__ float K_ufq[32], K_vfq[32];
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
    int warm_joint = -1;  // joint-cb warm pick: chunks recompute in reverse order, but any previously
                          // selected centroid is an EXACT distance bound, so correctness is unaffected
    for (int l = L - 1; l >= 0; l--) {
        // recompute states from the checkpoint, re-applying the rank-1 truncation each step
        float state_xy = state_checkpoints_BLHKK[get_index4(b, l, h, x, y, L, H, K, K)];
        for (int c = 0; c < CHUNK_LEN; c++) {
            int t = l * CHUNK_LEN + c;
            if (t >= T) break;                             // uniform (same t/T for all threads) -> safe

            bool skip = skip_BT[get_index1(b, t, T)];
            state_prev_xy_chunk[c] = state_xy;             // carried (truncated) entering state
            float in_state_xy = state_xy;
            int64_t global_y = get_index3(b, t, h, y, T, H, K);
            int64_t global_x = get_index3(b, t, h, x, T, H, K);
            float k_y = to_float<F>(k_BTHK[global_y]);
            float v_x = to_float<F>(v_BTHK[global_x]);
            float w_y = w_BTHK[global_y];
            float a_y = to_float<F>(a_BTHK[global_y]);
            float k_deformed_y = to_float<F>(k_deformed_BTHK[global_y]);

            float state_xy_decayed = state_xy * w_y;
            float state_k_dot = state_xy * k_deformed_y;
            for (int offset = K / 2; offset > 0; offset /= 2) {
                state_k_dot += __shfl_down_sync(FULL_MASK, state_k_dot, offset, K);
            }
            state_k_dot = __shfl_sync(FULL_MASK, state_k_dot, 0, K);
            state_xy = state_xy_decayed - state_k_dot * a_y * k_deformed_y;
            state_xy += v_x * k_y;
            state_xy_chunk[c] = state_xy;                  // raw post-update (grad formulas + out use this)
            if (skip) {                                    // skip-step elision (see forward kernel): the
                state_xy = in_state_xy;                    // truncation of a reverted state is unused
            } else {
                state_xy = qat_lr_rank1(state_xy, K, qmax, // re-apply truncation (block-uniform branch)
                    c_pq_learn ? &rec_idx_chunk[c * 8] : nullptr,
                    c_pq_learn ? &rec_norm_chunk[c * 2] : nullptr,
                    &warm_joint);
            }
        }

        for (int t = std::min(T - 1, (l + 1) * CHUNK_LEN - 1); t >= l * CHUNK_LEN; t--) {
            int c = t - l * CHUNK_LEN;
            float G_xy = 0.f;                              // dL/dQ_t at the STE consumption (cb grads)
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
                G_xy = dS_xy_contrib;                  // dL/dQ_t: Q_t was consumed by later steps
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

            __syncthreads();

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
            __syncthreads();
            float KK_grad_decay_yx = KK_grad_decay[get_index1(y, x, K+1)];
            float a_grad_x = -KK_grad_decay_yx * K_k_deformed[y];
            float k_deformed_t1 = -grad_decay_remove_xy * K_a[y] * K_k_deformed[y];
            float k_deformed_t2 = -K_a[x] * KK_grad_decay_yx * K_k_deformed[y];
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
            // ---- learnable-cb grads: dL/dcent_u[p][ci][j] = norm_u * sum_y G[s+j][y]*vfq[y] (v symm.).
            // Q_t = ufq (x) vfq was consumed at this !skip step (G_xy captured above); its centroid
            // selections + recon norms were recorded during the chunk re-run. atomicAdd accumulation
            // into g_pq_cb_grad (float add order nondeterministic, ~1e-7 jitter). ----
            if (c_pq_learn && c_pq_active && !skip) {
                KK_G[get_index1(x, y, K + 1)] = G_xy;
                if (x == 0 && y < K) {                 // reconstruct step-t factors from the recording
                    if (c_pq_joint) {                  // joint-uv: ONE code, halves u=[0,K) v=[K,2K)
                        int ci = rec_idx_chunk[c * 8];
                        K_ufq[y] = (ci >= 0) ? g_pq_cb[ci * c_pq_subdim + y] * rec_norm_chunk[c * 2] : 0.f;
                        K_vfq[y] = (ci >= 0) ? g_pq_cb[ci * c_pq_subdim + K + y] * rec_norm_chunk[c * 2 + 1] : 0.f;
                    } else {
                        int p = y / c_pq_subdim, j = y - p * c_pq_subdim;
                        int ciu = rec_idx_chunk[c * 8 + p];
                        int civ = rec_idx_chunk[c * 8 + c_pq_m + p];
                        K_ufq[y] = (ciu >= 0) ? g_pq_cb[(p * c_pq_ncent + ciu) * c_pq_subdim + j] * rec_norm_chunk[c * 2] : 0.f;
                        K_vfq[y] = (civ >= 0) ? g_pq_cb[((c_pq_m + p) * c_pq_ncent + civ) * c_pq_subdim + j] * rec_norm_chunk[c * 2 + 1] : 0.f;
                    }
                }
                __syncthreads();
                float gu = KK_G[get_index1(x, y, K + 1)] * K_vfq[y];  // row x: sum_y G[x][y]*vfq[y]
                float gv = KK_G[get_index1(y, x, K + 1)] * K_ufq[y];  // col x: sum_row G[row][x]*ufq[row]
                for (int o = K / 2; o > 0; o >>= 1) {
                    gu += __shfl_down_sync(FULL_MASK, gu, o, K);
                    gv += __shfl_down_sync(FULL_MASK, gv, o, K);
                }
                if (y == 0) {
                    if (c_pq_joint) {                  // joint-uv: both halves of ONE centroid
                        int ci = rec_idx_chunk[c * 8];
                        if (ci >= 0) {
                            atomicAdd(&g_pq_cb_grad[ci * c_pq_subdim + x], rec_norm_chunk[c * 2] * gu);
                            atomicAdd(&g_pq_cb_grad[ci * c_pq_subdim + K + x], rec_norm_chunk[c * 2 + 1] * gv);
                        }
                    } else {
                        int p = x / c_pq_subdim, j = x - p * c_pq_subdim;
                        int ciu = rec_idx_chunk[c * 8 + p];
                        int civ = rec_idx_chunk[c * 8 + c_pq_m + p];
                        if (ciu >= 0) atomicAdd(&g_pq_cb_grad[(p * c_pq_ncent + ciu) * c_pq_subdim + j], rec_norm_chunk[c * 2] * gu);
                        if (civ >= 0) atomicAdd(&g_pq_cb_grad[((c_pq_m + p) * c_pq_ncent + civ) * c_pq_subdim + j], rec_norm_chunk[c * 2 + 1] * gv);
                    }
                }
                __syncthreads();                       // KK_G/K_ufq/K_vfq reuse fence for the next t
            }
        }
    }
}

// Stage-A validation host wrapper: rank-1 int-N truncate each (b,h) KxK of a [B,H,K,K] state.
template <typename F>
at::Tensor rwkv7_lr_trunc_test_cuda(const at::Tensor& state_BHKK, double qmax) {
    const int B = state_BHKK.size(0), H = state_BHKK.size(1), K = state_BHKK.size(2);
    TORCH_INTERNAL_ASSERT(state_BHKK.device().type() == at::DeviceType::CUDA);
    at::Tensor out = torch::empty_like(state_BHKK);
    dim3 block_dim(K, K), grid_dim(B, H);
    rwkv7_lr_trunc_test_kernel<F><<<grid_dim, block_dim>>>(
        B, H, (F*)state_BHKK.data_ptr(), (F*)out.data_ptr(), (float)qmax);
    return out;
}

// Upload the rank-1 PQ codebook (roles 0,1) to the device globals; m<=0 disables PQ (qat_lr_rank1 reverts
// to its int-N path). cb_flat = float[2*m*ncent*sub] in layout ((role*m+p)*ncent+c)*sub+j (roles 0=u,1=v).
// Called ONCE before a PQ-QAT run. Not templated (global state), registered as a plain CUDA op.
static int h_pq_m = 0, h_pq_sub = 0, h_pq_ncent = 0, h_pq_joint = 0;   // host mirror of the codebook shape

void rwkv7_set_pq_codebook_cuda(const at::Tensor& cb_flat, int64_t m, int64_t sub, int64_t ncent, int64_t joint) {
    int active = (m > 0) ? 1 : 0;
    int mi = (int)m, si = (int)sub, ni = (int)ncent, ji = joint ? 1 : 0;
    h_pq_m = mi; h_pq_sub = si; h_pq_ncent = ni; h_pq_joint = ji;
    cudaMemcpyToSymbol(c_pq_active, &active, sizeof(int));
    cudaMemcpyToSymbol(c_pq_m, &mi, sizeof(int));
    cudaMemcpyToSymbol(c_pq_subdim, &si, sizeof(int));
    cudaMemcpyToSymbol(c_pq_ncent, &ni, sizeof(int));
    cudaMemcpyToSymbol(c_pq_joint, &ji, sizeof(int));
    const char* nw = getenv("RWKV_QAT_NO_WARM");           // task24 escape hatch: disable the
    int wa = (nw != nullptr && nw[0] == '1') ? 0 : 1;      // warm-started joint search (bitwise A/B)
    cudaMemcpyToSymbol(c_pq_warm, &wa, sizeof(int));
    if (active) {
        auto cbf = cb_flat.to(torch::kFloat32).to(torch::kCPU).contiguous();
        size_t n = (size_t)cbf.numel();
        size_t want = ji ? (size_t)(ncent * sub) : (size_t)(2 * m * ncent * sub);  // joint = 1 block
        TORCH_CHECK(n == want, "PQ codebook size mismatch: ", n, " != ", want);
        TORCH_CHECK(n <= 32768, "PQ codebook too large: ", n, " > 32768");
        TORCH_CHECK(!ji || m == 1, "joint PQ codebook requires m == 1");
        cudaMemcpyToSymbol(g_pq_cb, cbf.data_ptr<float>(), n * sizeof(float));
    }
    cudaDeviceSynchronize();
}

// Enable norm quantization in the PQ branch (bits<=0 disables). lo/hi = the fixed log2 range in octaves
// (engine Model::load uses [-3,0] for the WKV codebook). Global state, set once before a QAT run.
// `dev` is only a dispatch anchor (any CUDA tensor) — tensor-less schemas can't route to a CUDA impl.
void rwkv7_set_norm_quant_cuda(const at::Tensor& dev, int64_t bits, double lo_log2, double hi_log2) {
    float levels = (bits > 0) ? (float)((1 << (int)bits) - 1) : 0.f;
    float lo = (float)lo_log2, hi = (float)hi_log2;
    cudaMemcpyToSymbol(c_nq_levels, &levels, sizeof(float));
    cudaMemcpyToSymbol(c_nq_lo, &lo, sizeof(float));
    cudaMemcpyToSymbol(c_nq_hi, &hi, sizeof(float));
    cudaDeviceSynchronize();
}

// Learnable-codebook control (RWKV_QAT_PQ_LEARN). `dev` = dispatch anchor (any CUDA tensor).
void rwkv7_set_pq_learn_cuda(const at::Tensor& dev, int64_t on) {
    int v = on ? 1 : 0;
    cudaMemcpyToSymbol(c_pq_learn, &v, sizeof(int));
    cudaDeviceSynchronize();
}

// Zero the centroid-gradient accumulator. Call once per optimizer step BEFORE the backward pass(es).
void rwkv7_pq_cb_grad_zero_cuda(const at::Tensor& dev) {
    void* addr = nullptr;
    cudaGetSymbolAddress(&addr, g_pq_cb_grad);
    cudaMemsetAsync(addr, 0, sizeof(float) * 32768);
}

// Fetch the accumulated centroid grads (same layout as g_pq_cb) as a CUDA float tensor.
at::Tensor rwkv7_pq_cb_grad_get_cuda(const at::Tensor& dev) {
    int n = h_pq_joint ? h_pq_ncent * h_pq_sub : 2 * h_pq_m * h_pq_ncent * h_pq_sub;
    TORCH_CHECK(n > 0, "pq_cb_grad_get: no codebook uploaded");
    at::Tensor out = torch::empty({n}, torch::dtype(torch::kFloat32).device(dev.device()));
    cudaMemcpyFromSymbol(out.data_ptr<float>(), g_pq_cb_grad, n * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    return out;
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor> rwkv7_wkv_qat_lr_forward_cuda(
    const at::Tensor& r_BTHK, const at::Tensor& k_BTHK, const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK, const at::Tensor& a_BTHK, const at::Tensor& k_deformed_BTHK,
    const at::Tensor& skip_BT, double qmax
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = r_BTHK.size(3);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CUDA);
    at::Tensor out_BTHK = torch::empty(r_BTHK.sizes(), r_BTHK.options());
    int L = (T + CHUNK_LEN) / CHUNK_LEN;
    at::Tensor state_checkpoints_BLHKK = torch::empty({B, L, H, K, K}, r_BTHK.options().dtype(torch::kFloat32)).requires_grad_(false);
    dim3 block_dim(K, K);
    dim3 grid_dim(B, H);
    rwkv7_wkv_qat_lr_forward_kernel<CHUNK_LEN, F><<<grid_dim, block_dim>>>(
        B, T, H, (F*)r_BTHK.data_ptr(), (F*)k_BTHK.data_ptr(), (F*)v_BTHK.data_ptr(),
        w_BTHK.data_ptr<float>(), (F*)a_BTHK.data_ptr(), (F*)k_deformed_BTHK.data_ptr(),
        (bool*)skip_BT.data_ptr(), (float)qmax,
        (F*)out_BTHK.data_ptr(), L, state_checkpoints_BLHKK.data_ptr<float>());
    return std::make_tuple(out_BTHK, state_checkpoints_BLHKK);
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rwkv7_wkv_qat_lr_backward_cuda(
    const at::Tensor& r_BTHK, const at::Tensor& k_BTHK, const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK, const at::Tensor& a_BTHK, const at::Tensor& k_deformed_BTHK,
    const at::Tensor& skip_BT, const at::Tensor& state_checkpoints_BLHKK, const at::Tensor& grad_BTHK, double qmax
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = r_BTHK.size(3);
    const int L = state_checkpoints_BLHKK.size(1);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CUDA);
    at::Tensor r_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor k_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor v_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor w_grad_BTHK = torch::empty_like(r_BTHK, torch::dtype(torch::kFloat32));
    at::Tensor a_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor k_deformed_grad_BTHK = torch::empty_like(r_BTHK);
    dim3 block_dim(K, K);
    dim3 grid_dim(B, H);
    rwkv7_wkv_qat_lr_backward_kernel<CHUNK_LEN, F><<<grid_dim, block_dim>>>(
        B, T, H, (F*)r_BTHK.data_ptr(), (F*)k_BTHK.data_ptr(), (F*)v_BTHK.data_ptr(),
        w_BTHK.data_ptr<float>(), (F*)a_BTHK.data_ptr(), (F*)k_deformed_BTHK.data_ptr(),
        (bool*)skip_BT.data_ptr(), (float)qmax,
        (F*)grad_BTHK.data_ptr(), L, state_checkpoints_BLHKK.data_ptr<float>(),
        (F*)r_grad_BTHK.data_ptr(), (F*)k_grad_BTHK.data_ptr(), (F*)v_grad_BTHK.data_ptr(),
        w_grad_BTHK.data_ptr<float>(), (F*)a_grad_BTHK.data_ptr(), (F*)k_deformed_grad_BTHK.data_ptr());
    return std::make_tuple(r_grad_BTHK, k_grad_BTHK, v_grad_BTHK, w_grad_BTHK, a_grad_BTHK, k_deformed_grad_BTHK);
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
    at::Tensor r_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor k_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor v_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor w_grad_BTHK = torch::empty_like(r_BTHK, torch::dtype(torch::kFloat32));
    at::Tensor a_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor k_deformed_grad_BTHK = torch::empty_like(r_BTHK);
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
    at::Tensor r_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor k_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor v_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor w_grad_BTHK = torch::empty_like(r_BTHK, torch::dtype(torch::kFloat32));
    at::Tensor a_grad_BTHK = torch::empty_like(r_BTHK);
    at::Tensor k_deformed_grad_BTHK = torch::empty_like(r_BTHK);
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

// ---------------------------------------------------------------------------------------------
// Standalone PQ nearest-centroid search for the python shift-PQ path (fake_pq_shift):
// out[n] = argmin_c sum_j (u[n][j] - cb[c][j])^2, ties -> LOWEST c (the same first-strict-min
// semantics as torch.cdist().argmin() and the in-WKV joint search above). Direct squared-distance
// accumulation (always >= 0: no clamp subtlety) and the N x ncent distance matrix is NEVER
// materialized -- the eager cdist path wrote + re-read ~1.8 GB per call at the q72u m2b12 catalog
// (4096 x 16), ~370 ms/step total. One block per row; cb is re-read by every block -> L2-resident.
// Arch-agnostic: ncent/sub come from the tensor shapes.
// Row-tiled: PQA_ROWS rows share one block, so each centroid is loaded into registers ONCE per
// thread and reused across all PQA_ROWS rows (a one-row-per-block version was L2-bound re-reading
// the 256 KB catalog per row: 28 GB/call at N=110k -- slower than cdist). Templated on SUB so the
// bd/bi/cc tiles are compile-time-indexed (runtime bounds spill them to local memory: 9 ms/call
// instead of ~1). OOB row slots are zero-padded and computed anyway (results discarded on write).
#define PQA_ROWS 16
template <int SUB>
__global__ void rwkv7_pq_argmin_kernel(const float* __restrict__ u, const float* __restrict__ cb,
                                       int64_t* __restrict__ out, const int nrows, const int ncent) {
    __shared__ float srows[PQA_ROWS * SUB];             // this block's u rows (0-padded tail)
    __shared__ float rbd[32];                           // per-warp best distance (reused per row)
    __shared__ int rbi[32];                             // per-warp best index
    const int64_t row0 = (int64_t)blockIdx.x * PQA_ROWS;
    const int tid = threadIdx.x, nthreads = blockDim.x;
    const int nrow_here = (row0 + PQA_ROWS <= nrows) ? PQA_ROWS : (int)(nrows - row0);
    for (int j = tid; j < PQA_ROWS * SUB; j += nthreads)
        srows[j] = (j / SUB < nrow_here) ? u[row0 * SUB + j] : 0.f;
    __syncthreads();
    float bd[PQA_ROWS]; int bi[PQA_ROWS];
    #pragma unroll
    for (int r = 0; r < PQA_ROWS; r++) { bd[r] = INFINITY; bi[r] = 0x7fffffff; }
    float cc[SUB];
    for (int c = tid; c < ncent; c += nthreads) {       // ascending c per thread + strict < = lowest c on ties
        const float* cg = cb + (int64_t)c * SUB;
        #pragma unroll
        for (int j = 0; j < SUB; j++) cc[j] = cg[j];
        #pragma unroll
        for (int r = 0; r < PQA_ROWS; r++) {
            float d = 0.f;
            #pragma unroll
            for (int j = 0; j < SUB; j++) { float diff = srows[r * SUB + j] - cc[j]; d += diff * diff; }
            if (d < bd[r]) { bd[r] = d; bi[r] = c; }
        }
    }
    #pragma unroll
    for (int r = 0; r < PQA_ROWS; r++) {                // per-row block argmin, ties -> lower index
        if (r < nrow_here) {                            // nrow_here is block-uniform: syncs stay uniform
            float d = bd[r]; int i = bi[r];
            for (int o = 16; o > 0; o >>= 1) {
                float od = __shfl_down_sync(FULL_MASK, d, o);
                int oi = __shfl_down_sync(FULL_MASK, i, o);
                if (od < d || (od == d && oi < i)) { d = od; i = oi; }
            }
            if ((tid & 31) == 0) { rbd[tid >> 5] = d; rbi[tid >> 5] = i; }
            __syncthreads();
            if (tid == 0) {
                const int nwarps = (nthreads + 31) / 32;
                float fd = rbd[0]; int fi = rbi[0];
                for (int w = 1; w < nwarps; w++)
                    if (rbd[w] < fd || (rbd[w] == fd && rbi[w] < fi)) { fd = rbd[w]; fi = rbi[w]; }
                out[row0 + r] = (fi == 0x7fffffff) ? 0 : (int64_t)fi;  // all-NaN row -> 0 (masked upstream)
            }
            __syncthreads();                            // rbd/rbi reuse fence for the next row
        }
    }
}

// Generic-sub fallback (runtime sub, register tiles spill -- correct but slower; hit only for
// catalog dims outside the dispatch table below).
__global__ void rwkv7_pq_argmin_kernel_generic(const float* __restrict__ u, const float* __restrict__ cb,
                                               int64_t* __restrict__ out, const int nrows,
                                               const int ncent, const int sub) {
    extern __shared__ float sm[];
    float* srows = sm;
    float* rbd = sm + PQA_ROWS * sub;
    int* rbi = (int*)(rbd + 32);
    const int64_t row0 = (int64_t)blockIdx.x * PQA_ROWS;
    const int tid = threadIdx.x, nthreads = blockDim.x;
    const int nrow_here = (row0 + PQA_ROWS <= nrows) ? PQA_ROWS : (int)(nrows - row0);
    for (int j = tid; j < nrow_here * sub; j += nthreads) srows[j] = u[row0 * sub + j];
    __syncthreads();
    float bd[PQA_ROWS]; int bi[PQA_ROWS];
    for (int r = 0; r < PQA_ROWS; r++) { bd[r] = INFINITY; bi[r] = 0x7fffffff; }
    for (int c = tid; c < ncent; c += nthreads) {
        const float* cg = cb + (int64_t)c * sub;
        for (int r = 0; r < nrow_here; r++) {
            const float* srow = srows + r * sub;
            float d = 0.f;
            for (int j = 0; j < sub; j++) { float diff = srow[j] - cg[j]; d += diff * diff; }
            if (d < bd[r]) { bd[r] = d; bi[r] = c; }
        }
    }
    for (int r = 0; r < nrow_here; r++) {
        float d = bd[r]; int i = bi[r];
        for (int o = 16; o > 0; o >>= 1) {
            float od = __shfl_down_sync(FULL_MASK, d, o);
            int oi = __shfl_down_sync(FULL_MASK, i, o);
            if (od < d || (od == d && oi < i)) { d = od; i = oi; }
        }
        if ((tid & 31) == 0) { rbd[tid >> 5] = d; rbi[tid >> 5] = i; }
        __syncthreads();
        if (tid == 0) {
            const int nwarps = (nthreads + 31) / 32;
            float fd = rbd[0]; int fi = rbi[0];
            for (int w = 1; w < nwarps; w++)
                if (rbd[w] < fd || (rbd[w] == fd && rbi[w] < fi)) { fd = rbd[w]; fi = rbi[w]; }
            out[row0 + r] = (fi == 0x7fffffff) ? 0 : (int64_t)fi;
        }
        __syncthreads();
    }
}

at::Tensor rwkv7_pq_argmin_cuda(const at::Tensor& u_NS, const at::Tensor& cb_CS) {
    TORCH_CHECK(u_NS.device().type() == at::DeviceType::CUDA && cb_CS.device().type() == at::DeviceType::CUDA,
                "rwkv7_pq_argmin: CUDA tensors required");
    TORCH_CHECK(u_NS.scalar_type() == at::kFloat && cb_CS.scalar_type() == at::kFloat,
                "rwkv7_pq_argmin: float32 required");
    TORCH_CHECK(u_NS.dim() == 2 && cb_CS.dim() == 2 && u_NS.size(1) == cb_CS.size(1),
                "rwkv7_pq_argmin: [N,sub] x [ncent,sub] required");
    const at::Tensor u = u_NS.contiguous();
    const at::Tensor cb = cb_CS.contiguous();
    const int64_t n = u.size(0);
    TORCH_CHECK(n <= 0x7fffffffLL, "rwkv7_pq_argmin: too many rows");
    at::Tensor out = torch::empty({n}, u.options().dtype(torch::kInt64));
    if (n == 0) return out;
    const int threads = 256;
    const unsigned int blocks = (unsigned int)((n + PQA_ROWS - 1) / PQA_ROWS);
    const int ncent = (int)cb.size(0), sub = (int)cb.size(1);
    const float* up = u.data_ptr<float>();
    const float* cp = cb.data_ptr<float>();
    int64_t* op = out.data_ptr<int64_t>();
    switch (sub) {  // compile-time SUB keeps the per-thread tiles in registers (see kernel note)
        case 8:  rwkv7_pq_argmin_kernel<8><<<blocks, threads>>>(up, cp, op, (int)n, ncent); break;
        case 16: rwkv7_pq_argmin_kernel<16><<<blocks, threads>>>(up, cp, op, (int)n, ncent); break;
        case 32: rwkv7_pq_argmin_kernel<32><<<blocks, threads>>>(up, cp, op, (int)n, ncent); break;
        default: {
            const size_t smem = (size_t)PQA_ROWS * sub * sizeof(float)
                                + 32 * (sizeof(float) + sizeof(int));
            rwkv7_pq_argmin_kernel_generic<<<blocks, threads, smem>>>(up, cp, op, (int)n, ncent, sub);
        }
    }
    return out;
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
    m.impl("rwkv7_wkv_qat_forward_float", &rwkv7_wkv_qat_forward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_wkv_qat_backward_float", &rwkv7_wkv_qat_backward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_lr_trunc_test_float", &rwkv7_lr_trunc_test_cuda<float>);
    m.impl("rwkv7_wkv_qat_lr_forward_float", &rwkv7_wkv_qat_lr_forward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_wkv_qat_lr_backward_float", &rwkv7_wkv_qat_lr_backward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_set_pq_codebook", &rwkv7_set_pq_codebook_cuda);
    m.impl("rwkv7_set_norm_quant", &rwkv7_set_norm_quant_cuda);
    m.impl("rwkv7_set_pq_learn", &rwkv7_set_pq_learn_cuda);
    m.impl("rwkv7_pq_cb_grad_zero", &rwkv7_pq_cb_grad_zero_cuda);
    m.impl("rwkv7_pq_cb_grad_get", &rwkv7_pq_cb_grad_get_cuda);
    m.impl("rwkv7_pq_argmin", &rwkv7_pq_argmin_cuda);
}
}