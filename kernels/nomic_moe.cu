#include "common.cuh"

namespace gemv_config {
    constexpr int BLOCK_SIZE = 256;
    constexpr int OUTPUTS_PER_BLOCK = 64;
    constexpr int TILE_K = 32;
    constexpr int THREADS_PER_OUTPUT = BLOCK_SIZE / OUTPUTS_PER_BLOCK;  // 4 threads per output
}

__global__ void __launch_bounds__(256)
nomic_gate_gemv_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    half* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemv_config;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int num_tokens = expert_end - expert_start;

    if (num_tokens == 0) return;

    const int token_idx = blockIdx.y;
    if (token_idx >= num_tokens) return;

    const int block_n = blockIdx.x * OUTPUTS_PER_BLOCK;
    if (block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int token_id = sorted_token_ids[expert_start + token_idx];

    const int my_output = tid / THREADS_PER_OUTPUT;
    const int my_lane = tid % THREADS_PER_OUTPUT;

    const half* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const half* input_row = input + (size_t)token_id * hidden_dim;

    extern __shared__ char smem[];
    float* s_partial = reinterpret_cast<float*>(smem);

    float sum = 0.0f;

    const int global_n = block_n + my_output;
    if (global_n < intermediate_dim) {
        for (int k = my_lane; k < hidden_dim; k += THREADS_PER_OUTPUT) {
            sum += __half2float(input_row[k]) * __half2float(gate_w[k * intermediate_dim + global_n]);
        }
    }

    s_partial[tid] = sum;
    __syncthreads();

    // Reduction
    if (my_lane == 0 && my_output < OUTPUTS_PER_BLOCK) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < THREADS_PER_OUTPUT; i++) {
            total += s_partial[my_output * THREADS_PER_OUTPUT + i];
        }

        int global_n = block_n + my_output;
        if (global_n < intermediate_dim) {
            float result = apply_activation(total, activation_type);
            intermediate[(expert_start + token_idx) * intermediate_dim + global_n] = __float2half(result);
        }
    }
}

/*
 * Up projection GEMV kernel (transposed weights) - with parallel reduction
 * Grid: (ceil(hidden_dim / OUTPUTS_PER_BLOCK), num_tokens_for_expert, num_experts)
 * Block: (256)
 */
__global__ void __launch_bounds__(256)
nomic_up_gemv_kernel(
    const half* __restrict__ intermediate,
    const half* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    half* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemv_config;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int num_tokens = expert_end - expert_start;

    if (num_tokens == 0) return;

    const int token_idx = blockIdx.y;
    if (token_idx >= num_tokens) return;

    const int block_n = blockIdx.x * OUTPUTS_PER_BLOCK;
    if (block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int token_id = sorted_token_ids[expert_start + token_idx];
    const float routing_weight = sorted_weights[expert_start + token_idx];

    const int my_output = tid / THREADS_PER_OUTPUT;
    const int my_lane = tid % THREADS_PER_OUTPUT;

    const half* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const half* inter_row = intermediate + (size_t)(expert_start + token_idx) * intermediate_dim;

    extern __shared__ char smem[];
    float* s_partial = reinterpret_cast<float*>(smem);

    float sum = 0.0f;

    const int global_n = block_n + my_output;
    if (global_n < hidden_dim) {
        // up.T[k, n] = up[n, k] = up_w[n * intermediate_dim + k]
        // This is contiguous access for each n!
        const half* up_row = up_w + global_n * intermediate_dim;
        for (int k = my_lane; k < intermediate_dim; k += THREADS_PER_OUTPUT) {
            sum += __half2float(inter_row[k]) * __half2float(up_row[k]);
        }
    }

    s_partial[tid] = sum;
    __syncthreads();

    // Reduction
    if (my_lane == 0 && my_output < OUTPUTS_PER_BLOCK) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < THREADS_PER_OUTPUT; i++) {
            total += s_partial[my_output * THREADS_PER_OUTPUT + i];
        }

        int global_n = block_n + my_output;
        if (global_n < hidden_dim) {
            half result = __float2half(total * routing_weight);
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = result;
            } else {
                atomic_add_half(&output[token_id * hidden_dim + global_n], result);
            }
        }
    }
}

// ============================================================================
// Small GEMM Kernels - For seq_len 8-64
// ============================================================================

/*
 * Gate projection GEMM kernel (small tiles)
 */
__global__ void __launch_bounds__(gemm_small::THREADS)
nomic_gate_gemm_small_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    half* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemm_small;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    half* s_input = reinterpret_cast<half*>(smem);
    half* s_gate = s_input + SMEM_A;

    const half* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);

    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        // Load input tile
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = load_float4(&input[token_id * hidden_dim + global_k]);
            } else if (global_m < M) {
                half* ptr = reinterpret_cast<half*>(&val);
                int token_id = sorted_token_ids[expert_start + global_m];
                for (int j = 0; j < 8 && global_k + j < hidden_dim; j++) {
                    ptr[j] = input[token_id * hidden_dim + global_k + j];
                }
            }
            store_float4(&s_input[m * BLOCK_K + kk], val);
        }

        // Load gate weight tile
        for (int i = tid; i < BLOCK_K * BLOCK_N / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                val = load_float4(&gate_w[global_k * intermediate_dim + global_n]);
            }
            store_float4(&s_gate[kk * BLOCK_N + n], val);
        }
        __syncthreads();

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_input[warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_gate[kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    // Store results with activation
    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float val = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2half(val);
        }
    }
}

/*
 * Up projection GEMM kernel (small tiles, transposed weights)
 * Computes: output += routing_weight * (intermediate @ up.T)
 */
__global__ void __launch_bounds__(gemm_small::THREADS)
nomic_up_gemm_small_kernel(
    const half* __restrict__ intermediate,
    const half* __restrict__ up_weights,      // [num_experts, hidden_dim, intermediate_dim]
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    half* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemm_small;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    half* s_inter = reinterpret_cast<half*>(smem);
    half* s_up = s_inter + SMEM_A;
    int* s_token_ids = reinterpret_cast<int*>(s_up + SMEM_B);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    // up_weights: [num_experts, hidden_dim, intermediate_dim]
    // We compute intermediate @ up.T where up.T has shape [intermediate_dim, hidden_dim]
    const half* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    // Load token IDs and routing weights
    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);
    __syncthreads();

    for (int k = 0; k < intermediate_dim; k += BLOCK_K) {
        // Load intermediate tile
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < intermediate_dim) {
                val = load_float4(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
            store_float4(&s_inter[m * BLOCK_K + kk], val);
        }

        // Load transposed up weights: up.T[k, n] = up[n, k]
        // up_w layout: [hidden_dim, intermediate_dim], we want [intermediate_dim, hidden_dim]
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            half val = __float2half(0.0f);
            if (global_k < intermediate_dim && global_n < hidden_dim) {
                // up.T[global_k, global_n] = up[global_n, global_k]
                val = up_w[global_n * intermediate_dim + global_k];
            }
            s_up[kk * BLOCK_N + n] = val;
        }
        __syncthreads();

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_inter[warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_up[kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    // Store results with routing weight scaling
    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2half(val);
            } else {
                atomic_add_half(&output[token_id * hidden_dim + global_n], __float2half(val));
            }
        }
    }
}

// ============================================================================
// Large GEMM Kernels - For seq_len > 64
// ============================================================================

/*
 * Gate projection GEMM kernel (large tiles)
 */
__global__ void __launch_bounds__(gemm_large::THREADS)
nomic_gate_gemm_large_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    half* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemm_large;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    half* s_input = reinterpret_cast<half*>(smem);
    half* s_gate = s_input + SMEM_A;

    const half* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    // 2x4 WMMA tiles per warp
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2][4];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fill_fragment(frag_c[mi][ni], 0.0f);
        }
    }

    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        // Load input tile
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = load_float4(&input[token_id * hidden_dim + global_k]);
            }
            store_float4(&s_input[m * BLOCK_K + kk], val);
        }

        // Load gate weight tile
        for (int i = tid; i < BLOCK_K * BLOCK_N / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                val = load_float4(&gate_w[global_k * intermediate_dim + global_n]);
            }
            store_float4(&s_gate[kk * BLOCK_N + n], val);
        }
        __syncthreads();

        const int warp_row = warp_m * WARP_TILE_M;
        const int warp_col = warp_n * WARP_TILE_N;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int a_row = warp_row + mi * WMMA_M;
                wmma::load_matrix_sync(frag_a[mi], &s_input[a_row * BLOCK_K + kk], BLOCK_K);
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_gate[kk * BLOCK_N + b_col], BLOCK_N);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    wmma::mma_sync(frag_c[mi][ni], frag_a[mi], frag_b[ni], frag_c[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    // Store results with activation
    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = warp_row + mi * WMMA_M;
            int out_col = warp_col + ni * WMMA_N;
            wmma::store_matrix_sync(&s_out[out_row * BLOCK_N + out_col], frag_c[mi][ni], BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float val = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2half(val);
        }
    }
}

#ifndef NO_BF16_KERNEL
__global__ void __launch_bounds__(gemm_large::THREADS)
nomic_up_gemm_large_kernel(
    const half* __restrict__ intermediate,
    const half* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    half* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemm_large;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    half* s_inter = reinterpret_cast<half*>(smem);
    half* s_up = s_inter + SMEM_A;
    int* s_token_ids = reinterpret_cast<int*>(s_up + SMEM_B);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const half* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    // Load token IDs and routing weights
    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2][4];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fill_fragment(frag_c[mi][ni], 0.0f);
        }
    }
    __syncthreads();

    for (int k = 0; k < intermediate_dim; k += BLOCK_K) {
        // Load intermediate tile
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < intermediate_dim) {
                val = load_float4(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
            store_float4(&s_inter[m * BLOCK_K + kk], val);
        }

        // Load transposed up weights
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            half val = __float2half(0.0f);
            if (global_k < intermediate_dim && global_n < hidden_dim) {
                val = up_w[global_n * intermediate_dim + global_k];
            }
            s_up[kk * BLOCK_N + n] = val;
        }
        __syncthreads();

        const int warp_row = warp_m * WARP_TILE_M;
        const int warp_col = warp_n * WARP_TILE_N;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int a_row = warp_row + mi * WMMA_M;
                wmma::load_matrix_sync(frag_a[mi], &s_inter[a_row * BLOCK_K + kk], BLOCK_K);
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_up[kk * BLOCK_N + b_col], BLOCK_N);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    wmma::mma_sync(frag_c[mi][ni], frag_a[mi], frag_b[ni], frag_c[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    // Store results
    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = warp_row + mi * WMMA_M;
            int out_col = warp_col + ni * WMMA_N;
            wmma::store_matrix_sync(&s_out[out_row * BLOCK_N + out_col], frag_c[mi][ni], BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2half(val);
            } else {
                atomic_add_half(&output[token_id * hidden_dim + global_n], __float2half(val));
            }
        }
    }
}
#endif

#ifndef NO_BF16_KERNEL
__global__ void __launch_bounds__(gemm_small::THREADS)
nomic_gate_gemm_small_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemm_small;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_gate = s_input + SMEM_A;

    const __nv_bfloat16* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);

    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        // Load input tile
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = load_float4_bf16(&input[token_id * hidden_dim + global_k]);
            } else if (global_m < M) {
                __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(&val);
                int token_id = sorted_token_ids[expert_start + global_m];
                for (int j = 0; j < 8 && global_k + j < hidden_dim; j++) {
                    ptr[j] = input[token_id * hidden_dim + global_k + j];
                }
            }
            store_float4_bf16(&s_input[m * BLOCK_K + kk], val);
        }

        // Load gate weight tile
        for (int i = tid; i < BLOCK_K * BLOCK_N / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                val = load_float4_bf16(&gate_w[global_k * intermediate_dim + global_n]);
            }
            store_float4_bf16(&s_gate[kk * BLOCK_N + n], val);
        }
        __syncthreads();

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_input[warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_gate[kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    // Store results with activation
    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float val = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2bfloat16(val);
        }
    }
}

__global__ void __launch_bounds__(gemm_small::THREADS)
nomic_up_gemm_small_bf16_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const __nv_bfloat16* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemm_small;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    __nv_bfloat16* s_inter = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_up = s_inter + SMEM_A;
    int* s_token_ids = reinterpret_cast<int*>(s_up + SMEM_B);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const __nv_bfloat16* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    // Load token IDs and routing weights
    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2];

    wmma::fill_fragment(frag_c[0], 0.0f);
    wmma::fill_fragment(frag_c[1], 0.0f);
    __syncthreads();

    for (int k = 0; k < intermediate_dim; k += BLOCK_K) {
        // Load intermediate tile
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < intermediate_dim) {
                val = load_float4_bf16(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
            store_float4_bf16(&s_inter[m * BLOCK_K + kk], val);
        }

        // Load transposed up weights
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_k < intermediate_dim && global_n < hidden_dim) {
                val = up_w[global_n * intermediate_dim + global_k];
            }
            s_up[kk * BLOCK_N + n] = val;
        }
        __syncthreads();

        const int warp_row = warp_m * WMMA_M;
        const int warp_col = warp_n * WMMA_N * 2;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &s_inter[warp_row * BLOCK_K + kk], BLOCK_K);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_up[kk * BLOCK_N + b_col], BLOCK_N);
                wmma::mma_sync(frag_c[ni], frag_a, frag_b[ni], frag_c[ni]);
            }
        }
        __syncthreads();
    }

    // Store results
    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N * 2;

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int out_col = warp_col + ni * WMMA_N;
        wmma::store_matrix_sync(&s_out[warp_row * BLOCK_N + out_col], frag_c[ni], BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2bfloat16(val);
            } else {
                atomic_add_bf16(&output[token_id * hidden_dim + global_n], __float2bfloat16(val));
            }
        }
    }
}

/*
 * Gate projection GEMM kernel (large tiles) - BF16
 */
__global__ void __launch_bounds__(gemm_large::THREADS)
nomic_gate_gemm_large_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    using namespace gemm_large;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_gate = s_input + SMEM_A;

    const __nv_bfloat16* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2][4];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fill_fragment(frag_c[mi][ni], 0.0f);
        }
    }

    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        // Load input tile
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = load_float4_bf16(&input[token_id * hidden_dim + global_k]);
            }
            store_float4_bf16(&s_input[m * BLOCK_K + kk], val);
        }

        // Load gate weight tile
        for (int i = tid; i < BLOCK_K * BLOCK_N / 8; i += THREADS) {
            int kk = i / (BLOCK_N / 8);
            int n = (i % (BLOCK_N / 8)) * 8;
            int global_k = k + kk;
            int global_n = block_n + n;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_k < hidden_dim && global_n + 7 < intermediate_dim) {
                val = load_float4_bf16(&gate_w[global_k * intermediate_dim + global_n]);
            }
            store_float4_bf16(&s_gate[kk * BLOCK_N + n], val);
        }
        __syncthreads();

        const int warp_row = warp_m * WARP_TILE_M;
        const int warp_col = warp_n * WARP_TILE_N;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int a_row = warp_row + mi * WMMA_M;
                wmma::load_matrix_sync(frag_a[mi], &s_input[a_row * BLOCK_K + kk], BLOCK_K);
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_gate[kk * BLOCK_N + b_col], BLOCK_N);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    wmma::mma_sync(frag_c[mi][ni], frag_a[mi], frag_b[ni], frag_c[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    // Store results with activation
    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = warp_row + mi * WMMA_M;
            int out_col = warp_col + ni * WMMA_N;
            wmma::store_matrix_sync(&s_out[out_row * BLOCK_N + out_col], frag_c[mi][ni], BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < intermediate_dim) {
            float val = apply_activation(s_out[m * BLOCK_N + n], activation_type);
            intermediate[(expert_start + global_m) * intermediate_dim + global_n] = __float2bfloat16(val);
        }
    }
}

/*
 * Up projection GEMM kernel (large tiles, transposed weights) - BF16
 */
__global__ void __launch_bounds__(gemm_large::THREADS)
nomic_up_gemm_large_bf16_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const __nv_bfloat16* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int top_k
) {
    using namespace gemm_large;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ char smem[];
    __nv_bfloat16* s_inter = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_up = s_inter + SMEM_A;
    int* s_token_ids = reinterpret_cast<int*>(s_up + SMEM_B);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const __nv_bfloat16* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    // Load token IDs and routing weights
    for (int i = tid; i < BLOCK_M; i += THREADS) {
        int global_m = block_m + i;
        if (global_m < M) {
            s_token_ids[i] = sorted_token_ids[expert_start + global_m];
            s_routing[i] = sorted_weights[expert_start + global_m];
        } else {
            s_token_ids[i] = 0;
            s_routing[i] = 0.0f;
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2][4];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fill_fragment(frag_c[mi][ni], 0.0f);
        }
    }
    __syncthreads();

    for (int k = 0; k < intermediate_dim; k += BLOCK_K) {
        // Load intermediate tile
        for (int i = tid; i < BLOCK_M * BLOCK_K / 8; i += THREADS) {
            int m = i / (BLOCK_K / 8);
            int kk = (i % (BLOCK_K / 8)) * 8;
            int global_m = block_m + m;
            int global_k = k + kk;

            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_m < M && global_k + 7 < intermediate_dim) {
                val = load_float4_bf16(&intermediate[(expert_start + global_m) * intermediate_dim + global_k]);
            }
            store_float4_bf16(&s_inter[m * BLOCK_K + kk], val);
        }

        // Load transposed up weights
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_k < intermediate_dim && global_n < hidden_dim) {
                val = up_w[global_n * intermediate_dim + global_k];
            }
            s_up[kk * BLOCK_N + n] = val;
        }
        __syncthreads();

        const int warp_row = warp_m * WARP_TILE_M;
        const int warp_col = warp_n * WARP_TILE_N;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int a_row = warp_row + mi * WMMA_M;
                wmma::load_matrix_sync(frag_a[mi], &s_inter[a_row * BLOCK_K + kk], BLOCK_K);
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_col = warp_col + ni * WMMA_N;
                wmma::load_matrix_sync(frag_b[ni], &s_up[kk * BLOCK_N + b_col], BLOCK_N);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    wmma::mma_sync(frag_c[mi][ni], frag_a[mi], frag_b[ni], frag_c[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    // Store results
    float* s_out = reinterpret_cast<float*>(smem);

    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = warp_row + mi * WMMA_M;
            int out_col = warp_col + ni * WMMA_N;
            wmma::store_matrix_sync(&s_out[out_row * BLOCK_N + out_col], frag_c[mi][ni], BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        int global_m = block_m + m;
        int global_n = block_n + n;

        if (global_m < M && global_n < hidden_dim) {
            int token_id = s_token_ids[m];
            float weight = s_routing[m];
            float val = s_out[m * BLOCK_N + n] * weight;
            if (top_k == 1) {
                output[token_id * hidden_dim + global_n] = __float2bfloat16(val);
            } else {
                atomic_add_bf16(&output[token_id * hidden_dim + global_n], __float2bfloat16(val));
            }
        }
    }
}

// ============================================================================
// Kernel Launch API
// ============================================================================

/*
 * Launch Nomic MoE BF16 kernels with automatic kernel selection
 */
extern "C" void nomic_moe_forward_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weights,
    const __nv_bfloat16* up_weights,
    const int* sorted_token_ids,
    const float* sorted_weights,
    const int* expert_offsets,
    __nv_bfloat16* intermediate,
    __nv_bfloat16* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int max_tokens_per_expert,
    int top_k,
    int activation_type,
    cudaStream_t stream
) {
    // Select kernel based on tokens per expert
    if (max_tokens_per_expert <= thresholds::SMALL_GEMM_MAX_TOKENS) {
        // Small GEMM path
        using namespace gemm_small;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        size_t gate_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(__nv_bfloat16),
                               (size_t)SMEM_C * sizeof(float));

        size_t up_smem = (SMEM_A + SMEM_B) * sizeof(__nv_bfloat16) + BLOCK_M * (sizeof(int) + sizeof(float));
        up_smem = max(up_smem, SMEM_C * sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_small_bf16_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_small_bf16_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
    } else {
        // Large GEMM path
        using namespace gemm_large;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        size_t gate_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(__nv_bfloat16),
                               (size_t)SMEM_C * sizeof(float));

        size_t up_smem = (SMEM_A + SMEM_B) * sizeof(__nv_bfloat16) + BLOCK_M * (sizeof(int) + sizeof(float));
        up_smem = max(up_smem, SMEM_C * sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_large_bf16_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_large_bf16_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
    }
}

#endif // NO_BF16_KERNEL

/*
 * Launch Nomic MoE kernels with automatic kernel selection
 */
extern "C" void nomic_moe_forward(
    const half* input,
    const half* gate_weights,
    const half* up_weights,
    const int* sorted_token_ids,
    const float* sorted_weights,
    const int* expert_offsets,
    half* intermediate,
    half* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int max_tokens_per_expert,
    int top_k,
    int activation_type,
    cudaStream_t stream
) {
    // Select kernel based on tokens per expert
    if (max_tokens_per_expert <= thresholds::GEMV_MAX_TOKENS) {
        // GEMV path - reduced block count with parallel reduction
        using namespace gemv_config;
        int n_blocks_inter = (intermediate_dim + OUTPUTS_PER_BLOCK - 1) / OUTPUTS_PER_BLOCK;
        int n_blocks_out = (hidden_dim + OUTPUTS_PER_BLOCK - 1) / OUTPUTS_PER_BLOCK;

        // Shared memory for partial sums
        size_t gate_smem = BLOCK_SIZE * sizeof(float);
        size_t up_smem = BLOCK_SIZE * sizeof(float);

        dim3 grid_gate(n_blocks_inter, max_tokens_per_expert, num_experts);
        dim3 block(BLOCK_SIZE);

        nomic_gate_gemv_kernel<<<grid_gate, block, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_blocks_out, max_tokens_per_expert, num_experts);
        nomic_up_gemv_kernel<<<grid_up, block, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
    } else if (max_tokens_per_expert <= thresholds::SMALL_GEMM_MAX_TOKENS) {
        // Small GEMM path
        using namespace gemm_small;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        size_t gate_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(half),
                               (size_t)SMEM_C * sizeof(float));

        size_t up_smem = (SMEM_A + SMEM_B) * sizeof(half) + BLOCK_M * (sizeof(int) + sizeof(float));
        up_smem = max(up_smem, SMEM_C * sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_small_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_small_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
    } else {
        // Large GEMM path
        using namespace gemm_large;
        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        size_t gate_smem = max((size_t)(SMEM_A + SMEM_B) * sizeof(half),
                               (size_t)SMEM_C * sizeof(float));

        size_t up_smem = (SMEM_A + SMEM_B) * sizeof(half) + BLOCK_M * (sizeof(int) + sizeof(float));
        up_smem = max(up_smem, SMEM_C * sizeof(float));

        dim3 grid_gate(n_tiles_inter, m_tiles, num_experts);
        nomic_gate_gemm_large_kernel<<<grid_gate, THREADS, gate_smem, stream>>>(
            input, gate_weights, sorted_token_ids, expert_offsets,
            intermediate, hidden_dim, intermediate_dim, activation_type
        );

        dim3 grid_up(n_tiles_out, m_tiles, num_experts);
        nomic_up_gemm_large_kernel<<<grid_up, THREADS, up_smem, stream>>>(
            intermediate, up_weights, sorted_token_ids, sorted_weights, expert_offsets,
            output, hidden_dim, intermediate_dim, top_k
        );
    }
}
