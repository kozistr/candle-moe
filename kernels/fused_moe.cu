#include "common.cuh"
#include "preprocessing.cuh"

extern "C" void qwen3_moe_forward(
    const half* input,
    const half* gate_weights,
    const half* up_weights,
    const half* down_weights,
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
);

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
);

#ifndef NO_BF16_KERNEL
extern "C" void qwen3_moe_forward_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weights,
    const __nv_bfloat16* up_weights,
    const __nv_bfloat16* down_weights,
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
);

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
);
#endif

__global__ void convert_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

__global__ void convert_f32_to_bf16_kernel(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

void launch_bf16_to_f32(const __nv_bfloat16* input, float* output, size_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    convert_bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

void launch_f32_to_bf16(const float* input, __nv_bfloat16* output, size_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    convert_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

namespace fp32_config {
    // Large tiles for better memory efficiency
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 32;
    constexpr int THREADS = 256;

    // Thread mapping: each thread handles 4x4 outputs
    constexpr int THREAD_M = 4;
    constexpr int THREAD_N = 4;
    constexpr int THREADS_M = BLOCK_M / THREAD_M;  // 16
    constexpr int THREADS_N = BLOCK_N / THREAD_N;  // 16
}

__global__ void __launch_bounds__(256)
moe_gate_up_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gate_weights,
    const float* __restrict__ up_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_offsets,
    float* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type,
    bool has_up
) {
    using namespace fp32_config;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= intermediate_dim) return;

    const int tid = threadIdx.x;
    const int thread_m = tid / THREADS_N;  // 0-15
    const int thread_n = tid % THREADS_N;  // 0-15

    extern __shared__ char smem[];
    float* s_input = reinterpret_cast<float*>(smem);
    float* s_gate = s_input + BLOCK_M * BLOCK_K;
    float* s_up = s_gate + BLOCK_K * BLOCK_N;

    const float* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const float* up_w = has_up ? (up_weights + (size_t)expert_id * hidden_dim * intermediate_dim) : nullptr;

    // Each thread accumulates THREAD_M x THREAD_N outputs
    float acc_gate[THREAD_M][THREAD_N] = {{0.0f}};
    float acc_up[THREAD_M][THREAD_N] = {{0.0f}};

    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        // Cooperatively load input tile [BLOCK_M, BLOCK_K]
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += THREADS) {
            int m = i / BLOCK_K;
            int kk = i % BLOCK_K;
            int global_m = block_m + m;
            int global_k = k + kk;

            float val = 0.0f;
            if (global_m < M && global_k < hidden_dim) {
                int token_id = sorted_token_ids[expert_start + global_m];
                val = input[token_id * hidden_dim + global_k];
            }
            s_input[m * BLOCK_K + kk] = val;
        }

        // Cooperatively load weight tiles [BLOCK_K, BLOCK_N]
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            float gval = 0.0f;
            float uval = 0.0f;
            if (global_k < hidden_dim && global_n < intermediate_dim) {
                gval = gate_w[global_k * intermediate_dim + global_n];
                if (has_up) {
                    uval = up_w[global_k * intermediate_dim + global_n];
                }
            }
            s_gate[kk * BLOCK_N + n] = gval;
            if (has_up) {
                s_up[kk * BLOCK_N + n] = uval;
            }
        }
        __syncthreads();

        // Compute - each thread handles THREAD_M x THREAD_N output tile
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            // Load THREAD_M input values
            float a[THREAD_M];
            #pragma unroll
            for (int mi = 0; mi < THREAD_M; ++mi) {
                int row = thread_m * THREAD_M + mi;
                a[mi] = s_input[row * BLOCK_K + kk];
            }

            // Load THREAD_N weight values and accumulate
            #pragma unroll
            for (int ni = 0; ni < THREAD_N; ++ni) {
                int col = thread_n * THREAD_N + ni;
                float g = s_gate[kk * BLOCK_N + col];
                float u = has_up ? s_up[kk * BLOCK_N + col] : 0.0f;

                #pragma unroll
                for (int mi = 0; mi < THREAD_M; ++mi) {
                    acc_gate[mi][ni] += a[mi] * g;
                    if (has_up) {
                        acc_up[mi][ni] += a[mi] * u;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int mi = 0; mi < THREAD_M; ++mi) {
        int global_m = block_m + thread_m * THREAD_M + mi;
        if (global_m < M) {
            #pragma unroll
            for (int ni = 0; ni < THREAD_N; ++ni) {
                int global_n = block_n + thread_n * THREAD_N + ni;
                if (global_n < intermediate_dim) {
                    float result;
                    if (has_up) {
                        result = apply_activation(acc_gate[mi][ni], activation_type) * acc_up[mi][ni];
                    } else {
                        result = apply_activation(acc_gate[mi][ni], activation_type);
                    }
                    intermediate[(expert_start + global_m) * intermediate_dim + global_n] = result;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256)
moe_down_f32_kernel(
    const float* __restrict__ intermediate,
    const float* __restrict__ down_weights,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ expert_offsets,
    float* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    bool is_transposed,
    int top_k
) {
    using namespace fp32_config;

    const int expert_id = blockIdx.z;
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int M = expert_end - expert_start;

    if (M == 0) return;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    if (block_m >= M || block_n >= hidden_dim) return;

    const int tid = threadIdx.x;
    const int thread_m = tid / THREADS_N;
    const int thread_n = tid % THREADS_N;

    extern __shared__ char smem[];
    float* s_inter = reinterpret_cast<float*>(smem);
    float* s_down = s_inter + BLOCK_M * BLOCK_K;
    int* s_token_ids = reinterpret_cast<int*>(s_down + BLOCK_K * BLOCK_N);
    float* s_routing = reinterpret_cast<float*>(s_token_ids + BLOCK_M);

    const float* down_w = down_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

    // Load token IDs and routing weights for this tile
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

    float acc[THREAD_M][THREAD_N] = {{0.0f}};
    __syncthreads();

    for (int k = 0; k < intermediate_dim; k += BLOCK_K) {
        // Load intermediate tile
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += THREADS) {
            int m = i / BLOCK_K;
            int kk = i % BLOCK_K;
            int global_m = block_m + m;
            int global_k = k + kk;

            float val = 0.0f;
            if (global_m < M && global_k < intermediate_dim) {
                val = intermediate[(expert_start + global_m) * intermediate_dim + global_k];
            }
            s_inter[m * BLOCK_K + kk] = val;
        }

        // Load weights - handle transposed case
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_k = k + kk;
            int global_n = block_n + n;

            float val = 0.0f;
            if (global_k < intermediate_dim && global_n < hidden_dim) {
                if (is_transposed) {
                    val = down_w[global_n * intermediate_dim + global_k];
                } else {
                    val = down_w[global_k * hidden_dim + global_n];
                }
            }
            s_down[kk * BLOCK_N + n] = val;
        }
        __syncthreads();

        // Compute
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            float a[THREAD_M];
            #pragma unroll
            for (int mi = 0; mi < THREAD_M; ++mi) {
                int row = thread_m * THREAD_M + mi;
                a[mi] = s_inter[row * BLOCK_K + kk];
            }

            #pragma unroll
            for (int ni = 0; ni < THREAD_N; ++ni) {
                int col = thread_n * THREAD_N + ni;
                float w = s_down[kk * BLOCK_N + col];

                #pragma unroll
                for (int mi = 0; mi < THREAD_M; ++mi) {
                    acc[mi][ni] += a[mi] * w;
                }
            }
        }
        __syncthreads();
    }

    // Store results with routing weight scaling
    #pragma unroll
    for (int mi = 0; mi < THREAD_M; ++mi) {
        int local_m = thread_m * THREAD_M + mi;
        int global_m = block_m + local_m;
        if (global_m < M) {
            int token_id = s_token_ids[local_m];
            float weight = s_routing[local_m];

            #pragma unroll
            for (int ni = 0; ni < THREAD_N; ++ni) {
                int global_n = block_n + thread_n * THREAD_N + ni;
                if (global_n < hidden_dim) {
                    float val = acc[mi][ni] * weight;
                    if (top_k == 1) {
                        // Single expert per token - direct store, no atomicAdd needed
                        output[token_id * hidden_dim + global_n] = val;
                    } else {
                        // Multiple experts per token - need atomic accumulation
                        atomicAdd(&output[token_id * hidden_dim + global_n], val);
                    }
                }
            }
        }
    }
}

extern "C" {

void fused_moe(
    void* input,
    void* gate_weights,
    void* up_weights,
    void* down_weights,
    void* routing_weights,
    void* expert_indices,
    void* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int num_selected_experts,
    int activation_type,
    uint32_t moe_type,
    uint32_t dtype,
    void* stream_ptr
) {
    cudaStream_t stream = stream_ptr ? reinterpret_cast<cudaStream_t>(stream_ptr) : 0;

    const bool is_qwen3 = (moe_type == 0);
    const bool is_fp16 = (dtype == 0);
    const bool is_bf16 = (dtype == 1);
    const bool is_fp32 = (dtype == 2);
    const bool use_fp32_compute = is_fp32;
    const int total = num_tokens * num_selected_experts;

    size_t compute_elem_size = is_fp32 ? sizeof(float) : sizeof(half);
    size_t align = 256;

    size_t offset_size = ((num_experts + 2) * sizeof(int) + align - 1) & ~(align - 1);
    size_t sorted_ids_size = (total * sizeof(int) + align - 1) & ~(align - 1);
    size_t sorted_weights_size = (total * sizeof(float) + align - 1) & ~(align - 1);
    size_t intermediate_size = ((size_t)total * intermediate_dim * compute_elem_size + align - 1) & ~(align - 1);

    size_t total_workspace = offset_size + sorted_ids_size + sorted_weights_size + intermediate_size;

    char* d_workspace;
    cudaMallocAsync(&d_workspace, total_workspace, stream);

    size_t ws_offset = 0;
    int* d_expert_offsets = reinterpret_cast<int*>(d_workspace + ws_offset);
    ws_offset += offset_size;

    int* d_sorted_token_ids = reinterpret_cast<int*>(d_workspace + ws_offset);
    ws_offset += sorted_ids_size;

    float* d_sorted_weights = reinterpret_cast<float*>(d_workspace + ws_offset);
    ws_offset += sorted_weights_size;

    void* d_intermediate = reinterpret_cast<void*>(d_workspace + ws_offset);
    ws_offset += intermediate_size;

    launch_preprocessing(
        reinterpret_cast<const uint32_t*>(expert_indices),
        reinterpret_cast<const float*>(routing_weights),
        d_expert_offsets,
        d_sorted_token_ids,
        d_sorted_weights,
        num_tokens,
        num_experts,
        num_selected_experts,
        stream
    );

    // Determine max_tokens_per_expert for kernel grid sizing.
    // Use total as upper bound when it fits in grid.y (guaranteed correct, no sync).
    // For very large batches, sync to get exact value to avoid grid overflow.
    int max_tokens_per_expert;
    // CUDA grid.y limit is 65535. GEMM kernels use tiles so effective limit is higher.
    // For GEMV (seq <= 8): grid.y = max_tokens_per_expert directly
    // For GEMM: grid.y = ceil(max_tokens_per_expert / BLOCK_M), BLOCK_M >= 32
    // Use conservative limit for GEMV path compatibility.
    const int grid_y_safe_limit = 2097152;  // 65535 * 32 (smallest BLOCK_M)

    if (total <= grid_y_safe_limit) {
        // Use total as upper bound - guaranteed correct, no sync needed
        max_tokens_per_expert = total;
    } else {
        // For large batches, sync to get exact value
        cudaMemcpyAsync(&max_tokens_per_expert, &d_expert_offsets[num_experts + 1],
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        max_tokens_per_expert = max(1, max_tokens_per_expert);
    }

    if (is_fp16) {
        // FP16 path - use tensor core optimized kernels
        if (is_qwen3) {
            qwen3_moe_forward(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<const half*>(gate_weights),
                reinterpret_cast<const half*>(up_weights),
                reinterpret_cast<const half*>(down_weights),
                d_sorted_token_ids,
                d_sorted_weights,
                d_expert_offsets,
                reinterpret_cast<half*>(d_intermediate),
                reinterpret_cast<half*>(output),
                num_tokens,
                hidden_dim,
                intermediate_dim,
                num_experts,
                max_tokens_per_expert,
                num_selected_experts,  // top_k
                activation_type,
                stream
            );
        } else {
            nomic_moe_forward(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<const half*>(gate_weights),
                reinterpret_cast<const half*>(up_weights),
                d_sorted_token_ids,
                d_sorted_weights,
                d_expert_offsets,
                reinterpret_cast<half*>(d_intermediate),
                reinterpret_cast<half*>(output),
                num_tokens,
                hidden_dim,
                intermediate_dim,
                num_experts,
                max_tokens_per_expert,
                num_selected_experts,  // top_k
                activation_type,
                stream
            );
        }
    } else if (is_bf16) {
#ifndef NO_BF16_KERNEL
        // BF16 path - use native BF16 tensor core kernels (SM80+)
        if (is_qwen3) {
            qwen3_moe_forward_bf16(
                reinterpret_cast<const __nv_bfloat16*>(input),
                reinterpret_cast<const __nv_bfloat16*>(gate_weights),
                reinterpret_cast<const __nv_bfloat16*>(up_weights),
                reinterpret_cast<const __nv_bfloat16*>(down_weights),
                d_sorted_token_ids,
                d_sorted_weights,
                d_expert_offsets,
                reinterpret_cast<__nv_bfloat16*>(d_intermediate),
                reinterpret_cast<__nv_bfloat16*>(output),
                num_tokens,
                hidden_dim,
                intermediate_dim,
                num_experts,
                max_tokens_per_expert,
                num_selected_experts,  // top_k
                activation_type,
                stream
            );
        } else {
            nomic_moe_forward_bf16(
                reinterpret_cast<const __nv_bfloat16*>(input),
                reinterpret_cast<const __nv_bfloat16*>(gate_weights),
                reinterpret_cast<const __nv_bfloat16*>(up_weights),
                d_sorted_token_ids,
                d_sorted_weights,
                d_expert_offsets,
                reinterpret_cast<__nv_bfloat16*>(d_intermediate),
                reinterpret_cast<__nv_bfloat16*>(output),
                num_tokens,
                hidden_dim,
                intermediate_dim,
                num_experts,
                max_tokens_per_expert,
                num_selected_experts,  // top_k
                activation_type,
                stream
            );
        }
#else
#endif
    } else {
        // FP32 path - optimized with larger tiles
        using namespace fp32_config;
        const float* compute_input = reinterpret_cast<const float*>(input);
        const float* compute_gate = reinterpret_cast<const float*>(gate_weights);
        const float* compute_up = reinterpret_cast<const float*>(up_weights);
        const float* compute_down = reinterpret_cast<const float*>(down_weights);
        float* compute_output = reinterpret_cast<float*>(output);

        int m_tiles = (max_tokens_per_expert + BLOCK_M - 1) / BLOCK_M;
        int n_tiles_inter = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        int n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        size_t gate_up_smem = BLOCK_M * BLOCK_K * sizeof(float) + 2 * BLOCK_K * BLOCK_N * sizeof(float);
        size_t down_smem = BLOCK_M * BLOCK_K * sizeof(float) + BLOCK_K * BLOCK_N * sizeof(float) +
                        BLOCK_M * sizeof(int) + BLOCK_M * sizeof(float);

        if (is_qwen3) {
            // Qwen3: gate-up fused, then down
            dim3 grid_gate_up(n_tiles_inter, m_tiles, num_experts);
            moe_gate_up_f32_kernel<<<grid_gate_up, THREADS, gate_up_smem, stream>>>(
                compute_input,
                compute_gate,
                compute_up,
                d_sorted_token_ids,
                d_expert_offsets,
                reinterpret_cast<float*>(d_intermediate),
                hidden_dim,
                intermediate_dim,
                activation_type,
                true  // has_up
            );

            dim3 grid_down(n_tiles_out, m_tiles, num_experts);
            moe_down_f32_kernel<<<grid_down, THREADS, down_smem, stream>>>(
                reinterpret_cast<const float*>(d_intermediate),
                compute_down,
                d_sorted_token_ids,
                d_sorted_weights,
                d_expert_offsets,
                compute_output,
                hidden_dim,
                intermediate_dim,
                false,
                num_selected_experts
            );
        } else {
            // Nomic: gate-up fused, then down
            size_t gate_only_smem = BLOCK_M * BLOCK_K * sizeof(float) + BLOCK_K * BLOCK_N * sizeof(float);

            dim3 grid_gate_up(n_tiles_inter, m_tiles, num_experts);
            moe_gate_up_f32_kernel<<<grid_gate_up, THREADS, gate_only_smem, stream>>>(
                compute_input,
                compute_gate,
                compute_up,
                d_sorted_token_ids,
                d_expert_offsets,
                reinterpret_cast<float*>(d_intermediate),
                hidden_dim,
                intermediate_dim,
                activation_type,
                false
            );

            dim3 grid_up(n_tiles_out, m_tiles, num_experts);
            moe_down_f32_kernel<<<grid_up, THREADS, down_smem, stream>>>(
                reinterpret_cast<const float*>(d_intermediate),
                compute_up,
                d_sorted_token_ids,
                d_sorted_weights,
                d_expert_offsets,
                compute_output,
                hidden_dim,
                intermediate_dim,
                num_selected_experts,
                true
            );
        }
    }

    cudaFreeAsync(d_workspace, stream);
}

}
