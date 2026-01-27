#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#ifndef USE_ROCM
    #include <cub/cub.cuh>
#else
    #include <hipcub/hipcub.hpp>
#endif

#include <mma.h>
using namespace nvcuda;

// ============================================================================
// Optimized Tile Configuration
// Two-level tiling: Block handles [BLOCK_M, full N], internally tiled
// ============================================================================
#define BLOCK_M 64       // Tokens per block for large batches
#define BLOCK_M_SMALL 16 // Tokens per block for small batches
#define BLOCK_K 32       // K reduction tile
#define BLOCK_N 128      // Output elements per N-tile
#define THREADS 256
#define WARP_SIZE 32
#define NUM_WARPS (THREADS / WARP_SIZE)

// Tensor core tile sizes (for SM70+)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Elements per thread for register blocking (large tiles)
#define THREAD_M 8
#define THREAD_N 4

// Elements per thread for small tiles (BLOCK_M_SMALL=16, BLOCK_N=128)
// 16 * 128 / 256 = 8 elements per thread
#define THREAD_M_SMALL 2
#define THREAD_N_SMALL 4

// Threshold for using small tile kernel (tokens per expert)
// Balance between parallelism (3D kernel) and data reuse (2D kernel)
// Nomic (no down proj): 3D kernel works well up to higher thresholds
// Qwen3 (with down proj): 2D kernel better for larger batches due to intermediate buffer reuse
#define SMALL_BATCH_THRESHOLD_NOMIC 1024
#define SMALL_BATCH_THRESHOLD_QWEN 384

// ============================================================================
// Type conversion helpers
// ============================================================================
template<typename T> __device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float(float val) { return val; }
template<> __device__ __forceinline__ float to_float(half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T> __device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float(float val) { return val; }
template<> __device__ __forceinline__ half from_float(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

// Vectorized load helpers
__device__ __forceinline__ void load_float4(const float* ptr, float& a, float& b, float& c, float& d) {
    float4 v = *reinterpret_cast<const float4*>(ptr);
    a = v.x; b = v.y; c = v.z; d = v.w;
}

__device__ __forceinline__ void load_half4(const half* ptr, float& a, float& b, float& c, float& d) {
    float2 v = *reinterpret_cast<const float2*>(ptr);  // 4 halfs = 2 floats
    half2 h0 = *reinterpret_cast<const half2*>(&v.x);
    half2 h1 = *reinterpret_cast<const half2*>(&v.y);
    a = __half2float(h0.x); b = __half2float(h0.y);
    c = __half2float(h1.x); d = __half2float(h1.y);
}

__device__ __forceinline__ void load_bf16_4(const __nv_bfloat16* ptr, float& a, float& b, float& c, float& d) {
    float2 v = *reinterpret_cast<const float2*>(ptr);
    __nv_bfloat162 h0 = *reinterpret_cast<const __nv_bfloat162*>(&v.x);
    __nv_bfloat162 h1 = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
    a = __bfloat162float(h0.x); b = __bfloat162float(h0.y);
    c = __bfloat162float(h1.x); d = __bfloat162float(h1.y);
}

template<typename T>
__device__ __forceinline__ void load_vec4(const T* ptr, float& a, float& b, float& c, float& d);

template<>
__device__ __forceinline__ void load_vec4(const float* ptr, float& a, float& b, float& c, float& d) {
    load_float4(ptr, a, b, c, d);
}

template<>
__device__ __forceinline__ void load_vec4(const half* ptr, float& a, float& b, float& c, float& d) {
    load_half4(ptr, a, b, c, d);
}

template<>
__device__ __forceinline__ void load_vec4(const __nv_bfloat16* ptr, float& a, float& b, float& c, float& d) {
    load_bf16_4(ptr, a, b, c, d);
}

// ============================================================================
// Activation functions
// ============================================================================
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float apply_act(float x, int type) {
    if (type == 0) return silu(x);
    if (type == 1) return gelu(x);
    return fmaxf(0.0f, x);
}

// ============================================================================
// Atomic add for different types
// ============================================================================
__device__ __forceinline__ void atomic_add_f(float* addr, float val) {
    atomicAdd(addr, val);
}

#if __CUDA_ARCH__ >= 700
__device__ __forceinline__ void atomic_add_f(half* addr, float val) {
    atomicAdd(addr, __float2half(val));
}
#else
__device__ __forceinline__ void atomic_add_f(half* addr, float val) {
    unsigned int* base = (unsigned int*)((size_t)addr & ~2ULL);
    bool is_high = ((size_t)addr & 2);
    unsigned int old_val = *base, assumed, new_val;
    do {
        assumed = old_val;
        unsigned short lo = assumed & 0xFFFF, hi = assumed >> 16;
        half* target = is_high ? (half*)&hi : (half*)&lo;
        *target = __float2half(__half2float(*target) + val);
        new_val = lo | (hi << 16);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}
#endif

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void atomic_add_f(__nv_bfloat16* addr, float val) {
    atomicAdd(addr, __float2bfloat16(val));
}
#else
__device__ __forceinline__ void atomic_add_f(__nv_bfloat16* addr, float val) {
    unsigned int* base = (unsigned int*)((size_t)addr & ~2ULL);
    bool is_high = ((size_t)addr & 2);
    unsigned int old_val = *base, assumed, new_val;
    do {
        assumed = old_val;
        unsigned short lo = assumed & 0xFFFF, hi = assumed >> 16;
        __nv_bfloat16* target = is_high ? (__nv_bfloat16*)&hi : (__nv_bfloat16*)&lo;
        *target = __float2bfloat16(__bfloat162float(*target) + val);
        new_val = lo | (hi << 16);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}
#endif

// ============================================================================
// Optimized Gate-Up Kernel with Two-Level Tiling and Vectorized Loads
// Grid: (num_m_tiles, num_experts)  -- NOTE: only 2D, we handle N internally!
// Each block computes [BLOCK_M, intermediate_dim] by iterating over N tiles
// Input is loaded ONCE per K-tile, reused across all N tiles
// ============================================================================
template<typename T, bool HAS_UP>
__global__ void moe_gate_up_optimized_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gate_weights,
    const T* __restrict__ up_weights,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    float* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    const int expert_id = blockIdx.y;
    const int m_tile_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Expert token range
    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int num_tokens_expert = expert_end - expert_start;

    const int m_start = m_tile_idx * BLOCK_M;
    if (m_start >= num_tokens_expert) return;

    const int tile_m = min(BLOCK_M, num_tokens_expert - m_start);

    // Shared memory layout with padding for bank conflict avoidance
    // Use +1 padding to avoid bank conflicts (stride % 32 = 1)
    __shared__ float smem_input[BLOCK_M][BLOCK_K + 1];      // Input tile
    __shared__ float smem_gate[BLOCK_K][BLOCK_N + 1];       // Gate weights tile
    __shared__ float smem_up[BLOCK_K][BLOCK_N + 1];         // Up weights tile
    __shared__ int smem_token_ids[BLOCK_M];

    // Load token IDs once
    for (int i = tid; i < BLOCK_M; i += THREADS) {
        if (i < tile_m) {
            smem_token_ids[i] = sorted_token_ids[expert_start + m_start + i];
        }
    }
    __syncthreads();

    // Expert weight pointers
    const T* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const T* up_w = HAS_UP ? (up_weights + (size_t)expert_id * hidden_dim * intermediate_dim) : nullptr;

    // Process N tiles - this is the key optimization: iterate over N within the block
    for (int n_start = 0; n_start < intermediate_dim; n_start += BLOCK_N) {
        const int tile_n = min(BLOCK_N, intermediate_dim - n_start);

        // Register accumulators for this N-tile
        // Each thread handles THREAD_M rows and THREAD_N columns
        float gate_acc[THREAD_M][THREAD_N];
        float up_acc[THREAD_M][THREAD_N];

        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                gate_acc[i][j] = 0.0f;
                up_acc[i][j] = 0.0f;
            }
        }

        // Thread mapping for output tile
        // We have THREADS=256, BLOCK_M=64, BLOCK_N=128
        // Each thread computes THREAD_M=8 x THREAD_N=4 = 32 elements
        // Total coverage: 256 * 32 / (64*128) = 1.0 (exact coverage)
        const int thread_row = (tid / (BLOCK_N / THREAD_N)) * THREAD_M;  // 0, 8, 16, ...
        const int thread_col = (tid % (BLOCK_N / THREAD_N)) * THREAD_N;  // 0, 4, 8, ...

        // Iterate over K dimension
        for (int k_start = 0; k_start < hidden_dim; k_start += BLOCK_K) {
            const int tile_k = min(BLOCK_K, hidden_dim - k_start);

            // Cooperative load: input tile [BLOCK_M, BLOCK_K] using vectorized loads
            for (int idx = tid; idx < BLOCK_M * BLOCK_K / 4; idx += THREADS) {
                const int m = (idx * 4) / BLOCK_K;
                const int k = (idx * 4) % BLOCK_K;
                if (m < tile_m && k + 3 < tile_k) {
                    const int token_id = smem_token_ids[m];
                    float v0, v1, v2, v3;
                    load_vec4(input + token_id * hidden_dim + k_start + k, v0, v1, v2, v3);
                    smem_input[m][k] = v0;
                    smem_input[m][k+1] = v1;
                    smem_input[m][k+2] = v2;
                    smem_input[m][k+3] = v3;
                } else if (m < tile_m) {
                    // Handle boundary - scalar loads
                    for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                        if (k + kk < tile_k) {
                            const int token_id = smem_token_ids[m];
                            smem_input[m][k + kk] = to_float(input[token_id * hidden_dim + k_start + k + kk]);
                        } else {
                            smem_input[m][k + kk] = 0.0f;
                        }
                    }
                } else {
                    for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                        smem_input[m][k + kk] = 0.0f;
                    }
                }
            }

            // Cooperative load: gate weights [BLOCK_K, BLOCK_N] using vectorized loads
            for (int idx = tid; idx < BLOCK_K * BLOCK_N / 4; idx += THREADS) {
                const int k = (idx * 4) / BLOCK_N;
                const int n = (idx * 4) % BLOCK_N;
                if (k < tile_k && n + 3 < tile_n) {
                    float v0, v1, v2, v3;
                    load_vec4(gate_w + (k_start + k) * intermediate_dim + n_start + n, v0, v1, v2, v3);
                    smem_gate[k][n] = v0;
                    smem_gate[k][n+1] = v1;
                    smem_gate[k][n+2] = v2;
                    smem_gate[k][n+3] = v3;
                } else {
                    for (int nn = 0; nn < 4 && n + nn < BLOCK_N; nn++) {
                        if (k < tile_k && n + nn < tile_n) {
                            smem_gate[k][n + nn] = to_float(gate_w[(k_start + k) * intermediate_dim + n_start + n + nn]);
                        } else {
                            smem_gate[k][n + nn] = 0.0f;
                        }
                    }
                }
            }

            // Cooperative load: up weights [BLOCK_K, BLOCK_N]
            if constexpr (HAS_UP) {
                for (int idx = tid; idx < BLOCK_K * BLOCK_N / 4; idx += THREADS) {
                    const int k = (idx * 4) / BLOCK_N;
                    const int n = (idx * 4) % BLOCK_N;
                    if (k < tile_k && n + 3 < tile_n) {
                        float v0, v1, v2, v3;
                        load_vec4(up_w + (k_start + k) * intermediate_dim + n_start + n, v0, v1, v2, v3);
                        smem_up[k][n] = v0;
                        smem_up[k][n+1] = v1;
                        smem_up[k][n+2] = v2;
                        smem_up[k][n+3] = v3;
                    } else {
                        for (int nn = 0; nn < 4 && n + nn < BLOCK_N; nn++) {
                            if (k < tile_k && n + nn < tile_n) {
                                smem_up[k][n + nn] = to_float(up_w[(k_start + k) * intermediate_dim + n_start + n + nn]);
                            } else {
                                smem_up[k][n + nn] = 0.0f;
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // Compute: each thread processes its THREAD_M x THREAD_N tile
            #pragma unroll
            for (int k = 0; k < BLOCK_K; k++) {
                // Load input values for this thread's rows
                float inp[THREAD_M];
                #pragma unroll
                for (int i = 0; i < THREAD_M; i++) {
                    inp[i] = smem_input[thread_row + i][k];
                }

                // Accumulate gate and up projections
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    float g = smem_gate[k][thread_col + j];
                    float u = HAS_UP ? smem_up[k][thread_col + j] : 0.0f;
                    #pragma unroll
                    for (int i = 0; i < THREAD_M; i++) {
                        gate_acc[i][j] += inp[i] * g;
                        if constexpr (HAS_UP) {
                            up_acc[i][j] += inp[i] * u;
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Write results with activation
        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            const int m = thread_row + i;
            if (m < tile_m) {
                const int global_idx = expert_start + m_start + m;
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    const int n = thread_col + j;
                    if (n < tile_n) {
                        float activated = apply_act(gate_acc[i][j], activation_type);
                        if constexpr (HAS_UP) {
                            activated *= up_acc[i][j];
                        }
                        intermediate[global_idx * intermediate_dim + n_start + n] = activated;
                    }
                }
            }
        }
        __syncthreads();  // Ensure all writes complete before reusing smem
    }
}

// ============================================================================
// Optimized Output Kernel with Two-Level Tiling
// Grid: (num_m_tiles, num_experts)
// ============================================================================
template<typename T, bool IS_DOWN>
__global__ void moe_output_optimized_kernel(
    const float* __restrict__ intermediate,
    const T* __restrict__ proj_weights,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_routing_weights,
    T* __restrict__ output,
    int hidden_dim,
    int intermediate_dim
) {
    const int expert_id = blockIdx.y;
    const int m_tile_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int num_tokens_expert = expert_end - expert_start;

    const int m_start = m_tile_idx * BLOCK_M;
    if (m_start >= num_tokens_expert) return;

    const int tile_m = min(BLOCK_M, num_tokens_expert - m_start);

    // Shared memory with +1 padding to avoid bank conflicts
    __shared__ float smem_inter[BLOCK_M][BLOCK_K + 1];
    __shared__ float smem_proj[BLOCK_K][BLOCK_N + 1];
    __shared__ int smem_token_ids[BLOCK_M];
    __shared__ float smem_routing[BLOCK_M];

    // Load token IDs and routing weights once
    for (int i = tid; i < BLOCK_M; i += THREADS) {
        if (i < tile_m) {
            const int global_idx = expert_start + m_start + i;
            smem_token_ids[i] = sorted_token_ids[global_idx];
            smem_routing[i] = sorted_routing_weights[global_idx];
        }
    }
    __syncthreads();

    // For Nomic (IS_DOWN=false): weights are [num_experts, hidden_dim, intermediate_dim]
    // Accessing as [intermediate_dim, hidden_dim] requires transpose
    // For Qwen3 (IS_DOWN=true): weights are [num_experts, intermediate_dim, hidden_dim]
    const T* proj_w = proj_weights + (size_t)expert_id *
        (IS_DOWN ? (intermediate_dim * hidden_dim) : (hidden_dim * intermediate_dim));

    // Thread mapping
    const int thread_row = (tid / (BLOCK_N / THREAD_N)) * THREAD_M;
    const int thread_col = (tid % (BLOCK_N / THREAD_N)) * THREAD_N;

    // Process N tiles (N = hidden_dim for output)
    for (int n_start = 0; n_start < hidden_dim; n_start += BLOCK_N) {
        const int tile_n = min(BLOCK_N, hidden_dim - n_start);

        // Register accumulators
        float out_acc[THREAD_M][THREAD_N];
        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                out_acc[i][j] = 0.0f;
            }
        }

        // Iterate over K (intermediate_dim)
        for (int k_start = 0; k_start < intermediate_dim; k_start += BLOCK_K) {
            const int tile_k = min(BLOCK_K, intermediate_dim - k_start);

            // Load intermediate tile [BLOCK_M, BLOCK_K]
            for (int idx = tid; idx < BLOCK_M * BLOCK_K / 4; idx += THREADS) {
                const int m = (idx * 4) / BLOCK_K;
                const int k = (idx * 4) % BLOCK_K;
                if (m < tile_m && k + 3 < tile_k) {
                    const int global_idx = expert_start + m_start + m;
                    // Intermediate is contiguous FP32, use float4
                    float4 v = *reinterpret_cast<const float4*>(
                        intermediate + global_idx * intermediate_dim + k_start + k);
                    smem_inter[m][k] = v.x;
                    smem_inter[m][k+1] = v.y;
                    smem_inter[m][k+2] = v.z;
                    smem_inter[m][k+3] = v.w;
                } else {
                    for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                        if (m < tile_m && k + kk < tile_k) {
                            const int global_idx = expert_start + m_start + m;
                            smem_inter[m][k + kk] = intermediate[global_idx * intermediate_dim + k_start + k + kk];
                        } else {
                            smem_inter[m][k + kk] = 0.0f;
                        }
                    }
                }
            }

            // Load projection weights [BLOCK_K, BLOCK_N]
            // IS_DOWN: down_weights[expert][k][n] where k is intermediate, n is hidden
            // !IS_DOWN: up_weights[expert][n][k] - need coalesced transposed load
            if constexpr (IS_DOWN) {
                // Qwen3 case: weights[k][n] - load 4 consecutive n values (coalesced)
                for (int idx = tid; idx < BLOCK_K * BLOCK_N / 4; idx += THREADS) {
                    const int k = (idx * 4) / BLOCK_N;
                    const int n = (idx * 4) % BLOCK_N;
                    if (k < tile_k && n + 3 < tile_n) {
                        float v0, v1, v2, v3;
                        load_vec4(proj_w + (k_start + k) * hidden_dim + n_start + n, v0, v1, v2, v3);
                        smem_proj[k][n] = v0;
                        smem_proj[k][n+1] = v1;
                        smem_proj[k][n+2] = v2;
                        smem_proj[k][n+3] = v3;
                    } else {
                        for (int nn = 0; nn < 4 && n + nn < BLOCK_N; nn++) {
                            if (k < tile_k && n + nn < tile_n) {
                                smem_proj[k][n + nn] = to_float(proj_w[(k_start + k) * hidden_dim + n_start + n + nn]);
                            } else {
                                smem_proj[k][n + nn] = 0.0f;
                            }
                        }
                    }
                }
            } else {
                // Nomic case: weights are [hidden_dim, intermediate_dim] = [N, K]
                // We need smem_proj[k][n] = weights[n][k] = weights[n * intermediate_dim + k]
                // For COALESCED access: consecutive threads access consecutive k values
                // So we iterate with k as the fast-changing dimension
                for (int idx = tid; idx < BLOCK_K * BLOCK_N / 4; idx += THREADS) {
                    const int n = (idx * 4) / BLOCK_K;   // n changes slowly
                    const int k = (idx * 4) % BLOCK_K;   // k changes fast (coalesced!)

                    if (n < tile_n && k + 3 < tile_k) {
                        // Load 4 consecutive k values (coalesced memory access!)
                        // weights[n][k:k+4] = weights[n * intermediate_dim + k : k+4]
                        float v0, v1, v2, v3;
                        load_vec4(proj_w + (n_start + n) * intermediate_dim + k_start + k, v0, v1, v2, v3);
                        // Write to transposed positions in smem
                        smem_proj[k][n] = v0;
                        smem_proj[k+1][n] = v1;
                        smem_proj[k+2][n] = v2;
                        smem_proj[k+3][n] = v3;
                    } else {
                        for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                            if (n < tile_n && k + kk < tile_k) {
                                smem_proj[k + kk][n] = to_float(proj_w[(n_start + n) * intermediate_dim + k_start + k + kk]);
                            } else {
                                smem_proj[k + kk][n] = 0.0f;
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // Compute
            #pragma unroll
            for (int k = 0; k < BLOCK_K; k++) {
                float inter[THREAD_M];
                #pragma unroll
                for (int i = 0; i < THREAD_M; i++) {
                    inter[i] = smem_inter[thread_row + i][k];
                }

                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    float w = smem_proj[k][thread_col + j];
                    #pragma unroll
                    for (int i = 0; i < THREAD_M; i++) {
                        out_acc[i][j] += inter[i] * w;
                    }
                }
            }
            __syncthreads();
        }

        // Atomic add to output with routing weight
        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            const int m = thread_row + i;
            if (m < tile_m) {
                const int token_id = smem_token_ids[m];
                const float routing_w = smem_routing[m];
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    const int n = thread_col + j;
                    if (n < tile_n) {
                        atomic_add_f(output + token_id * hidden_dim + n_start + n, out_acc[i][j] * routing_w);
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// FUSED Single-Pass Kernel - No Intermediate Buffer!
// Each block computes full MoE output for BLOCK_M_FUSED tokens
// Intermediate stays in registers/shared memory, never written to global
// Key: Process one intermediate tile at a time, immediately use for output
// ============================================================================
#define BLOCK_M_FUSED 16     // Tokens per block (small to fit intermediate in smem)
#define TILE_I 64            // Intermediate tile size
#define TILE_N_OUT 64        // Output tile size
#define THREADS_FUSED 256

template<typename T, bool HAS_DOWN>
__global__ void moe_fused_single_pass_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gate_weights,
    const T* __restrict__ up_weights,
    const T* __restrict__ down_weights,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_routing_weights,
    T* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    const int expert_id = blockIdx.y;
    const int m_tile_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int num_tokens_expert = expert_end - expert_start;

    const int m_start = m_tile_idx * BLOCK_M_FUSED;
    if (m_start >= num_tokens_expert) return;

    const int tile_m = min(BLOCK_M_FUSED, num_tokens_expert - m_start);

    // Shared memory - carefully sized to fit
    // smem_input: 16 * 33 * 4 = 2.1 KB
    // smem_gate: 32 * 65 * 4 = 8.3 KB
    // smem_up: 32 * 65 * 4 = 8.3 KB
    // smem_gate_acc: 16 * 65 * 4 = 4.2 KB
    // smem_up_acc: 16 * 65 * 4 = 4.2 KB
    // smem_down: 64 * 65 * 4 = 16.6 KB
    // Total: ~44 KB (fits in 48KB default smem)
    __shared__ float smem_input[BLOCK_M_FUSED][BLOCK_K + 1];
    __shared__ float smem_gate[BLOCK_K][TILE_I + 1];
    __shared__ float smem_up[BLOCK_K][TILE_I + 1];
    __shared__ float smem_gate_acc[BLOCK_M_FUSED][TILE_I + 1];  // Gate accumulator
    __shared__ float smem_up_acc[BLOCK_M_FUSED][TILE_I + 1];    // Up accumulator
    __shared__ float smem_down[TILE_I][TILE_N_OUT + 1];
    __shared__ int smem_token_ids[BLOCK_M_FUSED];
    __shared__ float smem_routing[BLOCK_M_FUSED];

    // Load token info once
    if (tid < BLOCK_M_FUSED && tid < tile_m) {
        const int global_idx = expert_start + m_start + tid;
        smem_token_ids[tid] = sorted_token_ids[global_idx];
        smem_routing[tid] = sorted_routing_weights[global_idx];
    }
    __syncthreads();

    const T* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const T* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const T* down_w = HAS_DOWN ? (down_weights + (size_t)expert_id * intermediate_dim * hidden_dim) : nullptr;

    // Process intermediate in tiles
    for (int i_start = 0; i_start < intermediate_dim; i_start += TILE_I) {
        const int tile_i = min(TILE_I, intermediate_dim - i_start);

        // Clear accumulators
        for (int idx = tid; idx < BLOCK_M_FUSED * TILE_I; idx += THREADS_FUSED) {
            const int m = idx / TILE_I;
            const int i = idx % TILE_I;
            smem_gate_acc[m][i] = 0.0f;
            smem_up_acc[m][i] = 0.0f;
        }
        __syncthreads();

        // Compute gate and up projections for this intermediate tile
        for (int k_start = 0; k_start < hidden_dim; k_start += BLOCK_K) {
            const int tile_k = min(BLOCK_K, hidden_dim - k_start);

            // Load input [BLOCK_M_FUSED, BLOCK_K] with vectorized loads
            for (int idx = tid; idx < BLOCK_M_FUSED * BLOCK_K / 4; idx += THREADS_FUSED) {
                const int m = (idx * 4) / BLOCK_K;
                const int k = (idx * 4) % BLOCK_K;
                if (m < tile_m && k + 3 < tile_k) {
                    const int token_id = smem_token_ids[m];
                    float v0, v1, v2, v3;
                    load_vec4(input + token_id * hidden_dim + k_start + k, v0, v1, v2, v3);
                    smem_input[m][k] = v0;
                    smem_input[m][k+1] = v1;
                    smem_input[m][k+2] = v2;
                    smem_input[m][k+3] = v3;
                } else if (m < tile_m) {
                    for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                        if (k + kk < tile_k) {
                            const int token_id = smem_token_ids[m];
                            smem_input[m][k + kk] = to_float(input[token_id * hidden_dim + k_start + k + kk]);
                        } else {
                            smem_input[m][k + kk] = 0.0f;
                        }
                    }
                } else {
                    for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                        smem_input[m][k + kk] = 0.0f;
                    }
                }
            }

            // Load gate weights [BLOCK_K, TILE_I] with vectorized loads
            for (int idx = tid; idx < BLOCK_K * TILE_I / 4; idx += THREADS_FUSED) {
                const int k = (idx * 4) / TILE_I;
                const int i = (idx * 4) % TILE_I;
                if (k < tile_k && i + 3 < tile_i) {
                    float v0, v1, v2, v3;
                    load_vec4(gate_w + (k_start + k) * intermediate_dim + i_start + i, v0, v1, v2, v3);
                    smem_gate[k][i] = v0;
                    smem_gate[k][i+1] = v1;
                    smem_gate[k][i+2] = v2;
                    smem_gate[k][i+3] = v3;
                } else {
                    for (int ii = 0; ii < 4 && i + ii < TILE_I; ii++) {
                        if (k < tile_k && i + ii < tile_i) {
                            smem_gate[k][i + ii] = to_float(gate_w[(k_start + k) * intermediate_dim + i_start + i + ii]);
                        } else {
                            smem_gate[k][i + ii] = 0.0f;
                        }
                    }
                }
            }

            // Load up weights [BLOCK_K, TILE_I] with vectorized loads
            if constexpr (HAS_DOWN) {
                for (int idx = tid; idx < BLOCK_K * TILE_I / 4; idx += THREADS_FUSED) {
                    const int k = (idx * 4) / TILE_I;
                    const int i = (idx * 4) % TILE_I;
                    if (k < tile_k && i + 3 < tile_i) {
                        float v0, v1, v2, v3;
                        load_vec4(up_w + (k_start + k) * intermediate_dim + i_start + i, v0, v1, v2, v3);
                        smem_up[k][i] = v0;
                        smem_up[k][i+1] = v1;
                        smem_up[k][i+2] = v2;
                        smem_up[k][i+3] = v3;
                    } else {
                        for (int ii = 0; ii < 4 && i + ii < TILE_I; ii++) {
                            if (k < tile_k && i + ii < tile_i) {
                                smem_up[k][i + ii] = to_float(up_w[(k_start + k) * intermediate_dim + i_start + i + ii]);
                            } else {
                                smem_up[k][i + ii] = 0.0f;
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // Accumulate gate and up with unrolled inner loop
            for (int idx = tid; idx < BLOCK_M_FUSED * TILE_I; idx += THREADS_FUSED) {
                const int m = idx / TILE_I;
                const int i = idx % TILE_I;
                if (m < tile_m && i < tile_i) {
                    float gate_sum = 0.0f, up_sum = 0.0f;
                    #pragma unroll 8
                    for (int k = 0; k < BLOCK_K; k++) {
                        float inp = smem_input[m][k];
                        gate_sum += inp * smem_gate[k][i];
                        if constexpr (HAS_DOWN) {
                            up_sum += inp * smem_up[k][i];
                        }
                    }
                    smem_gate_acc[m][i] += gate_sum;
                    if constexpr (HAS_DOWN) {
                        smem_up_acc[m][i] += up_sum;
                    }
                }
            }
            __syncthreads();
        }

        // Apply activation and combine: intermediate = act(gate) * up
        for (int idx = tid; idx < BLOCK_M_FUSED * TILE_I; idx += THREADS_FUSED) {
            const int m = idx / TILE_I;
            const int i = idx % TILE_I;
            if (m < tile_m && i < tile_i) {
                float activated = apply_act(smem_gate_acc[m][i], activation_type);
                if constexpr (HAS_DOWN) {
                    smem_gate_acc[m][i] = activated * smem_up_acc[m][i];  // Reuse gate_acc for combined result
                } else {
                    smem_gate_acc[m][i] = activated;
                }
            }
        }
        __syncthreads();

        // Now compute output projection for each output tile
        for (int n_start = 0; n_start < hidden_dim; n_start += TILE_N_OUT) {
            const int tile_n = min(TILE_N_OUT, hidden_dim - n_start);

            // Load down/output weights [TILE_I, TILE_N_OUT] with vectorized loads
            if constexpr (HAS_DOWN) {
                for (int idx = tid; idx < TILE_I * TILE_N_OUT / 4; idx += THREADS_FUSED) {
                    const int i = (idx * 4) / TILE_N_OUT;
                    const int n = (idx * 4) % TILE_N_OUT;
                    if (i < tile_i && n + 3 < tile_n) {
                        float v0, v1, v2, v3;
                        load_vec4(down_w + (i_start + i) * hidden_dim + n_start + n, v0, v1, v2, v3);
                        smem_down[i][n] = v0;
                        smem_down[i][n+1] = v1;
                        smem_down[i][n+2] = v2;
                        smem_down[i][n+3] = v3;
                    } else {
                        for (int nn = 0; nn < 4 && n + nn < TILE_N_OUT; nn++) {
                            if (i < tile_i && n + nn < tile_n) {
                                smem_down[i][n + nn] = to_float(down_w[(i_start + i) * hidden_dim + n_start + n + nn]);
                            } else {
                                smem_down[i][n + nn] = 0.0f;
                            }
                        }
                    }
                }
            } else {
                // Nomic: transposed access - load 4 consecutive i values (coalesced)
                for (int idx = tid; idx < TILE_I * TILE_N_OUT / 4; idx += THREADS_FUSED) {
                    const int n = (idx * 4) / TILE_I;
                    const int i = (idx * 4) % TILE_I;
                    if (n < tile_n && i + 3 < tile_i) {
                        float v0, v1, v2, v3;
                        load_vec4(up_w + (n_start + n) * intermediate_dim + i_start + i, v0, v1, v2, v3);
                        smem_down[i][n] = v0;
                        smem_down[i+1][n] = v1;
                        smem_down[i+2][n] = v2;
                        smem_down[i+3][n] = v3;
                    } else {
                        for (int ii = 0; ii < 4 && i + ii < TILE_I; ii++) {
                            if (n < tile_n && i + ii < tile_i) {
                                smem_down[i + ii][n] = to_float(up_w[(n_start + n) * intermediate_dim + i_start + i + ii]);
                            } else {
                                smem_down[i + ii][n] = 0.0f;
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // Compute output contribution and accumulate with atomic
            // Thread mapping: each thread handles multiple (m, n) pairs
            for (int idx = tid; idx < BLOCK_M_FUSED * TILE_N_OUT; idx += THREADS_FUSED) {
                const int m = idx / TILE_N_OUT;
                const int n = idx % TILE_N_OUT;
                if (m < tile_m && n < tile_n) {
                    float sum = 0.0f;
                    #pragma unroll 8
                    for (int i = 0; i < TILE_I; i++) {
                        sum += smem_gate_acc[m][i] * smem_down[i][n];
                    }
                    const int token_id = smem_token_ids[m];
                    const float routing_w = smem_routing[m];
                    atomic_add_f(output + token_id * hidden_dim + n_start + n, sum * routing_w);
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Small-Batch Gate-Up Kernel (3D grid for short sequences)
// Grid: (num_n_tiles, num_m_tiles, num_experts)
// Optimized for few tokens per expert - simpler structure, less overhead
// ============================================================================
template<typename T, bool HAS_UP>
__global__ void moe_gate_up_small_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gate_weights,
    const T* __restrict__ up_weights,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    float* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
    const int expert_id = blockIdx.z;
    const int m_tile_idx = blockIdx.y;
    const int n_tile_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int num_tokens_expert = expert_end - expert_start;

    const int m_start = m_tile_idx * BLOCK_M_SMALL;
    const int n_start = n_tile_idx * BLOCK_N;

    if (m_start >= num_tokens_expert) return;
    if (n_start >= intermediate_dim) return;

    const int tile_m = min(BLOCK_M_SMALL, num_tokens_expert - m_start);
    const int tile_n = min(BLOCK_N, intermediate_dim - n_start);

    // Smaller shared memory for small batches
    __shared__ float smem_input[BLOCK_M_SMALL][BLOCK_K + 1];
    __shared__ float smem_gate[BLOCK_K][BLOCK_N + 1];
    __shared__ float smem_up[BLOCK_K][BLOCK_N + 1];
    __shared__ int smem_token_ids[BLOCK_M_SMALL];

    // Load token IDs
    if (tid < BLOCK_M_SMALL && tid < tile_m) {
        smem_token_ids[tid] = sorted_token_ids[expert_start + m_start + tid];
    }
    __syncthreads();

    const T* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const T* up_w = HAS_UP ? (up_weights + (size_t)expert_id * hidden_dim * intermediate_dim) : nullptr;

    // Thread mapping: each thread computes THREAD_M_SMALL x THREAD_N_SMALL elements
    const int thread_row = (tid / (BLOCK_N / THREAD_N_SMALL)) * THREAD_M_SMALL;
    const int thread_col = (tid % (BLOCK_N / THREAD_N_SMALL)) * THREAD_N_SMALL;

    float gate_acc[THREAD_M_SMALL][THREAD_N_SMALL];
    float up_acc[THREAD_M_SMALL][THREAD_N_SMALL];

    #pragma unroll
    for (int i = 0; i < THREAD_M_SMALL; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N_SMALL; j++) {
            gate_acc[i][j] = 0.0f;
            up_acc[i][j] = 0.0f;
        }
    }

    // Iterate over K
    for (int k_start = 0; k_start < hidden_dim; k_start += BLOCK_K) {
        const int tile_k = min(BLOCK_K, hidden_dim - k_start);

        // Load input [BLOCK_M_SMALL, BLOCK_K] with vectorized loads
        for (int idx = tid; idx < BLOCK_M_SMALL * BLOCK_K / 4; idx += THREADS) {
            const int m = (idx * 4) / BLOCK_K;
            const int k = (idx * 4) % BLOCK_K;
            if (m < tile_m && k + 3 < tile_k) {
                const int token_id = smem_token_ids[m];
                float v0, v1, v2, v3;
                load_vec4(input + token_id * hidden_dim + k_start + k, v0, v1, v2, v3);
                smem_input[m][k] = v0;
                smem_input[m][k+1] = v1;
                smem_input[m][k+2] = v2;
                smem_input[m][k+3] = v3;
            } else if (m < tile_m) {
                for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                    if (k + kk < tile_k) {
                        const int token_id = smem_token_ids[m];
                        smem_input[m][k + kk] = to_float(input[token_id * hidden_dim + k_start + k + kk]);
                    } else {
                        smem_input[m][k + kk] = 0.0f;
                    }
                }
            } else {
                for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                    smem_input[m][k + kk] = 0.0f;
                }
            }
        }

        // Load gate weights [BLOCK_K, BLOCK_N] with vectorized loads
        for (int idx = tid; idx < BLOCK_K * BLOCK_N / 4; idx += THREADS) {
            const int k = (idx * 4) / BLOCK_N;
            const int n = (idx * 4) % BLOCK_N;
            if (k < tile_k && n + 3 < tile_n) {
                float v0, v1, v2, v3;
                load_vec4(gate_w + (k_start + k) * intermediate_dim + n_start + n, v0, v1, v2, v3);
                smem_gate[k][n] = v0;
                smem_gate[k][n+1] = v1;
                smem_gate[k][n+2] = v2;
                smem_gate[k][n+3] = v3;
            } else {
                for (int nn = 0; nn < 4 && n + nn < BLOCK_N; nn++) {
                    if (k < tile_k && n + nn < tile_n) {
                        smem_gate[k][n + nn] = to_float(gate_w[(k_start + k) * intermediate_dim + n_start + n + nn]);
                    } else {
                        smem_gate[k][n + nn] = 0.0f;
                    }
                }
            }
        }

        // Load up weights with vectorized loads
        if constexpr (HAS_UP) {
            for (int idx = tid; idx < BLOCK_K * BLOCK_N / 4; idx += THREADS) {
                const int k = (idx * 4) / BLOCK_N;
                const int n = (idx * 4) % BLOCK_N;
                if (k < tile_k && n + 3 < tile_n) {
                    float v0, v1, v2, v3;
                    load_vec4(up_w + (k_start + k) * intermediate_dim + n_start + n, v0, v1, v2, v3);
                    smem_up[k][n] = v0;
                    smem_up[k][n+1] = v1;
                    smem_up[k][n+2] = v2;
                    smem_up[k][n+3] = v3;
                } else {
                    for (int nn = 0; nn < 4 && n + nn < BLOCK_N; nn++) {
                        if (k < tile_k && n + nn < tile_n) {
                            smem_up[k][n + nn] = to_float(up_w[(k_start + k) * intermediate_dim + n_start + n + nn]);
                        } else {
                            smem_up[k][n + nn] = 0.0f;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            float inp[THREAD_M_SMALL];
            #pragma unroll
            for (int i = 0; i < THREAD_M_SMALL; i++) {
                inp[i] = smem_input[thread_row + i][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_N_SMALL; j++) {
                float g = smem_gate[k][thread_col + j];
                float u = HAS_UP ? smem_up[k][thread_col + j] : 0.0f;
                #pragma unroll
                for (int i = 0; i < THREAD_M_SMALL; i++) {
                    gate_acc[i][j] += inp[i] * g;
                    if constexpr (HAS_UP) {
                        up_acc[i][j] += inp[i] * u;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < THREAD_M_SMALL; i++) {
        const int m = thread_row + i;
        if (m < tile_m) {
            const int global_idx = expert_start + m_start + m;
            #pragma unroll
            for (int j = 0; j < THREAD_N_SMALL; j++) {
                const int n = thread_col + j;
                if (n < tile_n) {
                    float activated = apply_act(gate_acc[i][j], activation_type);
                    if constexpr (HAS_UP) {
                        activated *= up_acc[i][j];
                    }
                    intermediate[global_idx * intermediate_dim + n_start + n] = activated;
                }
            }
        }
    }
}

// ============================================================================
// Small-Batch Output Kernel (3D grid)
// ============================================================================
template<typename T, bool IS_DOWN>
__global__ void moe_output_small_kernel(
    const float* __restrict__ intermediate,
    const T* __restrict__ proj_weights,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_routing_weights,
    T* __restrict__ output,
    int hidden_dim,
    int intermediate_dim
) {
    const int expert_id = blockIdx.z;
    const int m_tile_idx = blockIdx.y;
    const int n_tile_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int num_tokens_expert = expert_end - expert_start;

    const int m_start = m_tile_idx * BLOCK_M_SMALL;
    const int n_start = n_tile_idx * BLOCK_N;

    if (m_start >= num_tokens_expert) return;
    if (n_start >= hidden_dim) return;

    const int tile_m = min(BLOCK_M_SMALL, num_tokens_expert - m_start);
    const int tile_n = min(BLOCK_N, hidden_dim - n_start);

    __shared__ float smem_inter[BLOCK_M_SMALL][BLOCK_K + 1];
    __shared__ float smem_proj[BLOCK_K][BLOCK_N + 1];
    __shared__ int smem_token_ids[BLOCK_M_SMALL];
    __shared__ float smem_routing[BLOCK_M_SMALL];

    if (tid < BLOCK_M_SMALL && tid < tile_m) {
        const int global_idx = expert_start + m_start + tid;
        smem_token_ids[tid] = sorted_token_ids[global_idx];
        smem_routing[tid] = sorted_routing_weights[global_idx];
    }
    __syncthreads();

    const T* proj_w = proj_weights + (size_t)expert_id *
        (IS_DOWN ? (intermediate_dim * hidden_dim) : (hidden_dim * intermediate_dim));

    const int thread_row = (tid / (BLOCK_N / THREAD_N_SMALL)) * THREAD_M_SMALL;
    const int thread_col = (tid % (BLOCK_N / THREAD_N_SMALL)) * THREAD_N_SMALL;

    float out_acc[THREAD_M_SMALL][THREAD_N_SMALL];
    #pragma unroll
    for (int i = 0; i < THREAD_M_SMALL; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N_SMALL; j++) {
            out_acc[i][j] = 0.0f;
        }
    }

    for (int k_start = 0; k_start < intermediate_dim; k_start += BLOCK_K) {
        const int tile_k = min(BLOCK_K, intermediate_dim - k_start);

        // Load intermediate with vectorized loads (FP32)
        for (int idx = tid; idx < BLOCK_M_SMALL * BLOCK_K / 4; idx += THREADS) {
            const int m = (idx * 4) / BLOCK_K;
            const int k = (idx * 4) % BLOCK_K;
            if (m < tile_m && k + 3 < tile_k) {
                const int global_idx = expert_start + m_start + m;
                float4 v = *reinterpret_cast<const float4*>(
                    intermediate + global_idx * intermediate_dim + k_start + k);
                smem_inter[m][k] = v.x;
                smem_inter[m][k+1] = v.y;
                smem_inter[m][k+2] = v.z;
                smem_inter[m][k+3] = v.w;
            } else {
                for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                    if (m < tile_m && k + kk < tile_k) {
                        const int global_idx = expert_start + m_start + m;
                        smem_inter[m][k + kk] = intermediate[global_idx * intermediate_dim + k_start + k + kk];
                    } else {
                        smem_inter[m][k + kk] = 0.0f;
                    }
                }
            }
        }

        // Load projection weights with vectorized coalesced access
        if constexpr (IS_DOWN) {
            for (int idx = tid; idx < BLOCK_K * BLOCK_N / 4; idx += THREADS) {
                const int k = (idx * 4) / BLOCK_N;
                const int n = (idx * 4) % BLOCK_N;
                if (k < tile_k && n + 3 < tile_n) {
                    float v0, v1, v2, v3;
                    load_vec4(proj_w + (k_start + k) * hidden_dim + n_start + n, v0, v1, v2, v3);
                    smem_proj[k][n] = v0;
                    smem_proj[k][n+1] = v1;
                    smem_proj[k][n+2] = v2;
                    smem_proj[k][n+3] = v3;
                } else {
                    for (int nn = 0; nn < 4 && n + nn < BLOCK_N; nn++) {
                        if (k < tile_k && n + nn < tile_n) {
                            smem_proj[k][n + nn] = to_float(proj_w[(k_start + k) * hidden_dim + n_start + n + nn]);
                        } else {
                            smem_proj[k][n + nn] = 0.0f;
                        }
                    }
                }
            }
        } else {
            // Nomic: coalesced load with transposed store (vectorized)
            for (int idx = tid; idx < BLOCK_K * BLOCK_N / 4; idx += THREADS) {
                const int n = (idx * 4) / BLOCK_K;
                const int k = (idx * 4) % BLOCK_K;
                if (n < tile_n && k + 3 < tile_k) {
                    float v0, v1, v2, v3;
                    load_vec4(proj_w + (n_start + n) * intermediate_dim + k_start + k, v0, v1, v2, v3);
                    smem_proj[k][n] = v0;
                    smem_proj[k+1][n] = v1;
                    smem_proj[k+2][n] = v2;
                    smem_proj[k+3][n] = v3;
                } else {
                    for (int kk = 0; kk < 4 && k + kk < BLOCK_K; kk++) {
                        if (n < tile_n && k + kk < tile_k) {
                            smem_proj[k + kk][n] = to_float(proj_w[(n_start + n) * intermediate_dim + k_start + k + kk]);
                        } else {
                            smem_proj[k + kk][n] = 0.0f;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            float inter[THREAD_M_SMALL];
            #pragma unroll
            for (int i = 0; i < THREAD_M_SMALL; i++) {
                inter[i] = smem_inter[thread_row + i][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_N_SMALL; j++) {
                float w = smem_proj[k][thread_col + j];
                #pragma unroll
                for (int i = 0; i < THREAD_M_SMALL; i++) {
                    out_acc[i][j] += inter[i] * w;
                }
            }
        }
        __syncthreads();
    }

    // Atomic add to output
    #pragma unroll
    for (int i = 0; i < THREAD_M_SMALL; i++) {
        const int m = thread_row + i;
        if (m < tile_m) {
            const int token_id = smem_token_ids[m];
            const float routing_w = smem_routing[m];
            #pragma unroll
            for (int j = 0; j < THREAD_N_SMALL; j++) {
                const int n = thread_col + j;
                if (n < tile_n) {
                    atomic_add_f(output + token_id * hidden_dim + n_start + n, out_acc[i][j] * routing_w);
                }
            }
        }
    }
}

// ============================================================================
// Tensor Core Configuration for MoE
// Uses 16x16x16 WMMA tiles for FP16 computation
// Uses extended shared memory (up to 100KB on A40/A100)
// ============================================================================
#define TC_BLOCK_M 64       // Tokens per block
#define TC_BLOCK_K 32       // K tile (must be multiple of WMMA_K=16)
#define TC_BLOCK_N 64       // N tile (must be multiple of WMMA_N=16)
#define TC_THREADS 256      // 8 warps

// Shared memory strides - must be multiple of 16 for WMMA alignment
#define TC_INPUT_STRIDE 48   // BLOCK_K + padding for alignment (32 + 16)
#define TC_WEIGHT_STRIDE 80  // BLOCK_N + padding for alignment (64 + 16)

// Shared memory sizes for tensor core kernels
constexpr size_t TC_GATE_UP_SMEM = TC_BLOCK_M * TC_INPUT_STRIDE * sizeof(half) +   // input: 6KB
                                   2 * TC_BLOCK_K * TC_WEIGHT_STRIDE * sizeof(half) + // gate+up weights: 10KB
                                   TC_BLOCK_M * sizeof(int) +                        // token_ids: 0.25KB
                                   2 * TC_BLOCK_M * TC_BLOCK_N * sizeof(float);      // gate+up output: 32KB
                                   // Total: ~49KB

constexpr size_t TC_OUTPUT_SMEM = TC_BLOCK_M * TC_INPUT_STRIDE * sizeof(half) +    // inter: 6KB
                                  TC_BLOCK_K * TC_WEIGHT_STRIDE * sizeof(half) +   // proj weights: 5KB
                                  TC_BLOCK_M * sizeof(int) +                       // token_ids: 0.25KB
                                  TC_BLOCK_M * sizeof(float) +                     // routing: 0.25KB
                                  TC_BLOCK_M * TC_BLOCK_N * sizeof(float);         // output: 16KB
                                  // Total: ~28KB

// ============================================================================
// Tensor Core Gate-Up Kernel (SM70+ with FP16)
// Uses WMMA for 16x16x16 matrix operations
// Each warp computes a 32x32 tile (2x2 WMMA tiles)
// ============================================================================
template<bool HAS_UP>
__global__ void moe_gate_up_tensor_core_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const half* __restrict__ up_weights,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    float* __restrict__ intermediate,
    int hidden_dim,
    int intermediate_dim,
    int activation_type
) {
#if __CUDA_ARCH__ >= 700
    const int expert_id = blockIdx.y;
    const int m_tile_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    (void)lane_id;
    (void)lane_id;

    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int num_tokens_expert = expert_end - expert_start;

    const int m_start = m_tile_idx * TC_BLOCK_M;
    if (m_start >= num_tokens_expert) return;

    const int tile_m = min(TC_BLOCK_M, num_tokens_expert - m_start);

    // Dynamic shared memory layout (~50KB total):
    // [0]: smem_input     - 64 x 48 x 2 = 6144 bytes
    // [1]: smem_gate      - 32 x 80 x 2 = 5120 bytes
    // [2]: smem_up        - 32 x 80 x 2 = 5120 bytes
    // [3]: smem_token_ids - 64 x 4 = 256 bytes
    // [4]: smem_gate_out  - 64 x 64 x 4 = 16384 bytes
    // [5]: smem_up_out    - 64 x 64 x 4 = 16384 bytes
    extern __shared__ char shared_mem[];

    half* smem_input = (half*)shared_mem;
    half* smem_gate = smem_input + TC_BLOCK_M * TC_INPUT_STRIDE;
    half* smem_up = smem_gate + TC_BLOCK_K * TC_WEIGHT_STRIDE;
    int* smem_token_ids = (int*)(smem_up + TC_BLOCK_K * TC_WEIGHT_STRIDE);
    float* smem_gate_out = (float*)(smem_token_ids + TC_BLOCK_M);
    float* smem_up_out = smem_gate_out + TC_BLOCK_M * TC_BLOCK_N;

    // Helper macros for 2D indexing
    #define SMEM_INPUT(m, k) smem_input[(m) * TC_INPUT_STRIDE + (k)]
    #define SMEM_GATE(k, n) smem_gate[(k) * TC_WEIGHT_STRIDE + (n)]
    #define SMEM_UP(k, n) smem_up[(k) * TC_WEIGHT_STRIDE + (n)]
    #define SMEM_GATE_OUT(m, n) smem_gate_out[(m) * TC_BLOCK_N + (n)]
    #define SMEM_UP_OUT(m, n) smem_up_out[(m) * TC_BLOCK_N + (n)]

    // Load token IDs once
    for (int i = tid; i < TC_BLOCK_M; i += TC_THREADS) {
        if (i < tile_m) {
            smem_token_ids[i] = sorted_token_ids[expert_start + m_start + i];
        }
    }
    __syncthreads();

    const half* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
    const half* up_w = HAS_UP ? (up_weights + (size_t)expert_id * hidden_dim * intermediate_dim) : nullptr;

    // Warp tile mapping: 8 warps in 2x4 arrangement
    // Each warp handles 2x1 WMMA tiles = 32x16 output region
    // Total coverage: (2 warps * 32) x (4 warps * 16) = 64 x 64
    const int warp_row = warp_id / 4;  // 0 or 1
    const int warp_col = warp_id % 4;  // 0, 1, 2, or 3
    const int warp_m_base = warp_row * 32;   // 0 or 32
    const int warp_n_base = warp_col * 16;   // 0, 16, 32, or 48

    // Process output in N tiles
    for (int n_start = 0; n_start < intermediate_dim; n_start += TC_BLOCK_N) {
        const int tile_n = min(TC_BLOCK_N, intermediate_dim - n_start);

        // WMMA fragments for this warp (2x1 tiles in M, 1 tile in N)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[2];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_gate_acc[2];
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_up_acc[2];

        // Initialize accumulators
        wmma::fill_fragment(frag_gate_acc[0], 0.0f);
        wmma::fill_fragment(frag_gate_acc[1], 0.0f);
        if constexpr (HAS_UP) {
            wmma::fill_fragment(frag_up_acc[0], 0.0f);
            wmma::fill_fragment(frag_up_acc[1], 0.0f);
        }

        // Iterate over K dimension
        for (int k_start = 0; k_start < hidden_dim; k_start += TC_BLOCK_K) {
            const int tile_k = min(TC_BLOCK_K, hidden_dim - k_start);

            // Cooperative load: input [TC_BLOCK_M, TC_BLOCK_K]
            for (int idx = tid; idx < TC_BLOCK_M * TC_BLOCK_K; idx += TC_THREADS) {
                const int m = idx / TC_BLOCK_K;
                const int k = idx % TC_BLOCK_K;
                if (m < tile_m && k < tile_k) {
                    const int token_id = smem_token_ids[m];
                    SMEM_INPUT(m, k) = input[token_id * hidden_dim + k_start + k];
                } else {
                    SMEM_INPUT(m, k) = __float2half(0.0f);
                }
            }

            // Cooperative load: gate weights [TC_BLOCK_K, TC_BLOCK_N]
            for (int idx = tid; idx < TC_BLOCK_K * TC_BLOCK_N; idx += TC_THREADS) {
                const int k = idx / TC_BLOCK_N;
                const int n = idx % TC_BLOCK_N;
                if (k < tile_k && n < tile_n) {
                    SMEM_GATE(k, n) = gate_w[(k_start + k) * intermediate_dim + n_start + n];
                } else {
                    SMEM_GATE(k, n) = __float2half(0.0f);
                }
            }

            // Cooperative load: up weights [TC_BLOCK_K, TC_BLOCK_N]
            if constexpr (HAS_UP) {
                for (int idx = tid; idx < TC_BLOCK_K * TC_BLOCK_N; idx += TC_THREADS) {
                    const int k = idx / TC_BLOCK_N;
                    const int n = idx % TC_BLOCK_N;
                    if (k < tile_k && n < tile_n) {
                        SMEM_UP(k, n) = up_w[(k_start + k) * intermediate_dim + n_start + n];
                    } else {
                        SMEM_UP(k, n) = __float2half(0.0f);
                    }
                }
            }
            __syncthreads();

            // WMMA computation: process K in WMMA_K chunks
            for (int kk = 0; kk < TC_BLOCK_K; kk += WMMA_K) {
                // Load input fragments for this warp's M tiles
                wmma::load_matrix_sync(frag_a[0], &SMEM_INPUT(warp_m_base, kk), TC_INPUT_STRIDE);
                wmma::load_matrix_sync(frag_a[1], &SMEM_INPUT(warp_m_base + 16, kk), TC_INPUT_STRIDE);

                // Load gate weight fragment and compute
                wmma::load_matrix_sync(frag_b, &SMEM_GATE(kk, warp_n_base), TC_WEIGHT_STRIDE);
                wmma::mma_sync(frag_gate_acc[0], frag_a[0], frag_b, frag_gate_acc[0]);
                wmma::mma_sync(frag_gate_acc[1], frag_a[1], frag_b, frag_gate_acc[1]);

                // Load up weight fragment and compute
                if constexpr (HAS_UP) {
                    wmma::load_matrix_sync(frag_b, &SMEM_UP(kk, warp_n_base), TC_WEIGHT_STRIDE);
                    wmma::mma_sync(frag_up_acc[0], frag_a[0], frag_b, frag_up_acc[0]);
                    wmma::mma_sync(frag_up_acc[1], frag_a[1], frag_b, frag_up_acc[1]);
                }
            }
            __syncthreads();
        }

        // Store WMMA fragments to shared memory tiles
        #pragma unroll
        for (int tile_row = 0; tile_row < 2; tile_row++) {
            const int m_base = warp_m_base + tile_row * 16;
            wmma::store_matrix_sync(
                &SMEM_GATE_OUT(m_base, warp_n_base),
                frag_gate_acc[tile_row],
                TC_BLOCK_N,
                wmma::mem_row_major);
            if constexpr (HAS_UP) {
                wmma::store_matrix_sync(
                    &SMEM_UP_OUT(m_base, warp_n_base),
                    frag_up_acc[tile_row],
                    TC_BLOCK_N,
                    wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Write results to global with activation
        for (int idx = tid; idx < TC_BLOCK_M * TC_BLOCK_N; idx += TC_THREADS) {
            const int m = idx / TC_BLOCK_N;
            const int n = idx % TC_BLOCK_N;
            if (m < tile_m && n < tile_n) {
                float activated = apply_act(SMEM_GATE_OUT(m, n), activation_type);
                if constexpr (HAS_UP) {
                    activated *= SMEM_UP_OUT(m, n);
                }
                const int global_idx = expert_start + m_start + m;
                intermediate[global_idx * intermediate_dim + n_start + n] = activated;
            }
        }
        __syncthreads();
    }

    #undef SMEM_INPUT
    #undef SMEM_GATE
    #undef SMEM_UP
    #undef SMEM_GATE_OUT
    #undef SMEM_UP_OUT
#endif
}

// ============================================================================
// Tensor Core Output Kernel (SM70+ with FP16)
// Computes: output += intermediate @ down_weights * routing_weight
// ============================================================================
template<bool IS_DOWN>
__global__ void moe_output_tensor_core_kernel(
    const float* __restrict__ intermediate,
    const half* __restrict__ proj_weights,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_routing_weights,
    half* __restrict__ output,
    int hidden_dim,
    int intermediate_dim
) {
#if __CUDA_ARCH__ >= 700
    const int expert_id = blockIdx.y;
    const int m_tile_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int expert_start = expert_offsets[expert_id];
    const int expert_end = expert_offsets[expert_id + 1];
    const int num_tokens_expert = expert_end - expert_start;

    const int m_start = m_tile_idx * TC_BLOCK_M;
    if (m_start >= num_tokens_expert) return;

    const int tile_m = min(TC_BLOCK_M, num_tokens_expert - m_start);

    // Dynamic shared memory layout (~22KB total):
    // [0]: smem_inter     - 64 x 48 x 2 = 6144 bytes
    // [1]: smem_proj      - 32 x 80 x 2 = 5120 bytes
    // [2]: smem_token_ids - 64 x 4 = 256 bytes
    // [3]: smem_routing   - 64 x 4 = 256 bytes
    // [4]: smem_out       - 64 x 64 x 4 = 16384 bytes
    extern __shared__ char shared_mem[];

    half* smem_inter = (half*)shared_mem;
    half* smem_proj = smem_inter + TC_BLOCK_M * TC_INPUT_STRIDE;
    int* smem_token_ids = (int*)(smem_proj + TC_BLOCK_K * TC_WEIGHT_STRIDE);
    float* smem_routing = (float*)(smem_token_ids + TC_BLOCK_M);
    float* smem_out = smem_routing + TC_BLOCK_M;

    #define SMEM_INTER(m, k) smem_inter[(m) * TC_INPUT_STRIDE + (k)]
    #define SMEM_PROJ(k, n) smem_proj[(k) * TC_WEIGHT_STRIDE + (n)]
    #define SMEM_OUT(m, n) smem_out[(m) * TC_BLOCK_N + (n)]

    // Load token IDs and routing weights
    for (int i = tid; i < TC_BLOCK_M; i += TC_THREADS) {
        if (i < tile_m) {
            const int global_idx = expert_start + m_start + i;
            smem_token_ids[i] = sorted_token_ids[global_idx];
            smem_routing[i] = sorted_routing_weights[global_idx];
        }
    }
    __syncthreads();

    const half* proj_w = proj_weights + (size_t)expert_id *
        (IS_DOWN ? (intermediate_dim * hidden_dim) : (hidden_dim * intermediate_dim));

    // Warp tile mapping (same as gate-up kernel)
    // 8 warps in 2x4 arrangement, each handles 32x16 output
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int warp_m_base = warp_row * 32;
    const int warp_n_base = warp_col * 16;

    // Process output in N tiles
    for (int n_start = 0; n_start < hidden_dim; n_start += TC_BLOCK_N) {
        const int tile_n = min(TC_BLOCK_N, hidden_dim - n_start);

        // WMMA fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[2];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[2];

        wmma::fill_fragment(frag_acc[0], 0.0f);
        wmma::fill_fragment(frag_acc[1], 0.0f);

        // Iterate over K dimension (intermediate_dim)
        for (int k_start = 0; k_start < intermediate_dim; k_start += TC_BLOCK_K) {
            const int tile_k = min(TC_BLOCK_K, intermediate_dim - k_start);

            // Cooperative load: intermediate (FP32 -> FP16 conversion)
            for (int idx = tid; idx < TC_BLOCK_M * TC_BLOCK_K; idx += TC_THREADS) {
                const int m = idx / TC_BLOCK_K;
                const int k = idx % TC_BLOCK_K;
                if (m < tile_m && k < tile_k) {
                    const int global_idx = expert_start + m_start + m;
                    SMEM_INTER(m, k) = __float2half(intermediate[global_idx * intermediate_dim + k_start + k]);
                } else {
                    SMEM_INTER(m, k) = __float2half(0.0f);
                }
            }

            // Cooperative load: projection weights
            if constexpr (IS_DOWN) {
                // Qwen3: weights[k][n]
                for (int idx = tid; idx < TC_BLOCK_K * TC_BLOCK_N; idx += TC_THREADS) {
                    const int k = idx / TC_BLOCK_N;
                    const int n = idx % TC_BLOCK_N;
                    if (k < tile_k && n < tile_n) {
                        SMEM_PROJ(k, n) = proj_w[(k_start + k) * hidden_dim + n_start + n];
                    } else {
                        SMEM_PROJ(k, n) = __float2half(0.0f);
                    }
                }
            } else {
                // Nomic: weights[n][k] - need transposed load
                for (int idx = tid; idx < TC_BLOCK_K * TC_BLOCK_N; idx += TC_THREADS) {
                    const int k = idx / TC_BLOCK_N;
                    const int n = idx % TC_BLOCK_N;
                    if (k < tile_k && n < tile_n) {
                        SMEM_PROJ(k, n) = proj_w[(n_start + n) * intermediate_dim + k_start + k];
                    } else {
                        SMEM_PROJ(k, n) = __float2half(0.0f);
                    }
                }
            }
            __syncthreads();

            // WMMA computation
            for (int kk = 0; kk < TC_BLOCK_K; kk += WMMA_K) {
                wmma::load_matrix_sync(frag_a[0], &SMEM_INTER(warp_m_base, kk), TC_INPUT_STRIDE);
                wmma::load_matrix_sync(frag_a[1], &SMEM_INTER(warp_m_base + 16, kk), TC_INPUT_STRIDE);

                wmma::load_matrix_sync(frag_b, &SMEM_PROJ(kk, warp_n_base), TC_WEIGHT_STRIDE);

                wmma::mma_sync(frag_acc[0], frag_a[0], frag_b, frag_acc[0]);
                wmma::mma_sync(frag_acc[1], frag_a[1], frag_b, frag_acc[1]);
            }
            __syncthreads();
        }

        // Store WMMA fragments to shared memory tiles
        #pragma unroll
        for (int tile_row = 0; tile_row < 2; tile_row++) {
            const int m_base = warp_m_base + tile_row * 16;
            wmma::store_matrix_sync(
                &SMEM_OUT(m_base, warp_n_base),
                frag_acc[tile_row],
                TC_BLOCK_N,
                wmma::mem_row_major);
        }
        __syncthreads();

        // Atomic add to output with routing weight
        for (int idx = tid; idx < TC_BLOCK_M * TC_BLOCK_N; idx += TC_THREADS) {
            const int m = idx / TC_BLOCK_N;
            const int n = idx % TC_BLOCK_N;
            if (m < tile_m && n < tile_n) {
                const int token_id = smem_token_ids[m];
                const float routing_w = smem_routing[m];
                const float val = SMEM_OUT(m, n) * routing_w;
                atomicAdd(output + token_id * hidden_dim + n_start + n, __float2half(val));
            }
        }
        __syncthreads();
    }

    #undef SMEM_INTER
    #undef SMEM_PROJ
    #undef SMEM_OUT
#endif
}

// ============================================================================
// Simple token-parallel kernel for small batches
// ============================================================================
template<typename T, bool HAS_DOWN>
__global__ void moe_token_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gate_weights,
    const T* __restrict__ up_weights,
    const T* __restrict__ down_weights,
    const float* __restrict__ routing_weights,
    const uint32_t* __restrict__ expert_indices,
    T* __restrict__ output,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int activation_type
) {
    extern __shared__ float smem[];
    float* smem_input = smem;
    float* smem_inter = smem + hidden_dim;

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    for (int i = tid; i < hidden_dim; i += THREADS) {
        smem_input[i] = to_float(input[token_idx * hidden_dim + i]);
    }
    __syncthreads();

    T* token_out = output + token_idx * hidden_dim;

    for (int e = 0; e < num_selected_experts; e++) {
        const int expert_id = expert_indices[token_idx * num_selected_experts + e];
        const float routing_w = routing_weights[token_idx * num_selected_experts + e];

        const T* gate_w = gate_weights + (size_t)expert_id * hidden_dim * intermediate_dim;
        const T* up_w = up_weights + (size_t)expert_id * hidden_dim * intermediate_dim;

        for (int i = tid; i < intermediate_dim; i += THREADS) {
            float gate_sum = 0.0f, up_sum = 0.0f;
            for (int k = 0; k < hidden_dim; k++) {
                float inp = smem_input[k];
                gate_sum += inp * to_float(gate_w[k * intermediate_dim + i]);
                if constexpr (HAS_DOWN) {
                    up_sum += inp * to_float(up_w[k * intermediate_dim + i]);
                }
            }
            float activated = apply_act(gate_sum, activation_type);
            smem_inter[i] = HAS_DOWN ? (activated * up_sum) : activated;
        }
        __syncthreads();

        if constexpr (HAS_DOWN) {
            const T* down_w = down_weights + (size_t)expert_id * intermediate_dim * hidden_dim;
            for (int i = tid; i < hidden_dim; i += THREADS) {
                float sum = 0.0f;
                for (int k = 0; k < intermediate_dim; k++) {
                    sum += smem_inter[k] * to_float(down_w[k * hidden_dim + i]);
                }
                float cur = to_float(token_out[i]);
                token_out[i] = from_float<T>(cur + sum * routing_w);
            }
        } else {
            for (int i = tid; i < hidden_dim; i += THREADS) {
                float sum = 0.0f;
                for (int k = 0; k < intermediate_dim; k++) {
                    sum += smem_inter[k] * to_float(up_w[i * intermediate_dim + k]);
                }
                float cur = to_float(token_out[i]);
                token_out[i] = from_float<T>(cur + sum * routing_w);
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Preprocessing kernels
// ============================================================================
__global__ void count_experts_kernel(
    const uint32_t* __restrict__ expert_indices,
    int* __restrict__ expert_counts,
    int total_assignments
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_assignments) {
        atomicAdd(&expert_counts[expert_indices[idx]], 1);
    }
}

__global__ void compute_offsets_kernel(
    int* __restrict__ counts,
    int* __restrict__ offsets,
    int num_experts
) {
    if (threadIdx.x >= num_experts) return;

    __shared__ int smem[256];
    smem[threadIdx.x] = counts[threadIdx.x];
    counts[threadIdx.x] = 0;
    __syncthreads();

    int sum = 0;
    for (int i = 0; i < threadIdx.x; i++) {
        sum += smem[i];
    }
    offsets[threadIdx.x] = sum;

    if (threadIdx.x == num_experts - 1) {
        offsets[num_experts] = sum + smem[threadIdx.x];
    }
}

__global__ void build_sorted_indices_kernel(
    const uint32_t* __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    const int* __restrict__ offsets,
    int* __restrict__ sorted_token_ids,
    float* __restrict__ sorted_routing_weights,
    int* __restrict__ counters,
    int num_tokens,
    int num_selected
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * num_selected) return;

    int token_id = idx / num_selected;
    int select_idx = idx % num_selected;
    int expert_id = expert_indices[idx];
    float weight = routing_weights[token_id * num_selected + select_idx];

    int pos = atomicAdd(&counters[expert_id], 1);
    int out_idx = offsets[expert_id] + pos;
    sorted_token_ids[out_idx] = token_id;
    sorted_routing_weights[out_idx] = weight;
}

// ============================================================================
// C Interface
// ============================================================================
extern "C" {

void moe_token_parallel(
    void* input, void* gate_weights, void* up_weights, void* down_weights,
    float* routing_weights, uint32_t* expert_indices, void* output,
    int num_tokens, int hidden_dim, int intermediate_dim,
    int num_selected_experts, int activation_type, uint32_t moe_type, uint32_t dtype,
    void* stream_ptr
) {
    const int smem_size = (hidden_dim + intermediate_dim) * sizeof(float);
    const bool has_down = (moe_type == 0);
    cudaStream_t stream = stream_ptr ? reinterpret_cast<cudaStream_t>(stream_ptr) : 0;

    #define LAUNCH(T) \
        if (has_down) { \
            moe_token_kernel<T, true><<<num_tokens, THREADS, smem_size, stream>>>( \
                (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                routing_weights, expert_indices, (T*)output, \
                hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
        } else { \
            moe_token_kernel<T, false><<<num_tokens, THREADS, smem_size, stream>>>( \
                (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                routing_weights, expert_indices, (T*)output, \
                hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
        }

    if (dtype == 0) { LAUNCH(half); }
    else if (dtype == 1) { LAUNCH(__nv_bfloat16); }
    else { LAUNCH(float); }
    #undef LAUNCH
}

void fused_moe(
    void* input, void* gate_weights, void* up_weights, void* down_weights,
    float* routing_weights, uint32_t* expert_indices, void* output,
    int num_tokens, int hidden_dim, int intermediate_dim,
    int num_experts, int num_selected_experts, int activation_type,
    uint32_t moe_type, uint32_t dtype,
    int* expert_counts, int* expert_offsets, int* token_ids,
    int* counters, float* sorted_routing_weights, float* intermediate_buffer,
    void* stream_ptr
) {
    // Use token-parallel kernel for very small batches (no preprocessing overhead)
    if (num_tokens < 16) {
        moe_token_parallel(input, gate_weights, up_weights, down_weights,
                          routing_weights, expert_indices, output,
                          num_tokens, hidden_dim, intermediate_dim,
                          num_selected_experts, activation_type, moe_type, dtype, stream_ptr);
        return;
    }

    cudaStream_t stream = stream_ptr ? reinterpret_cast<cudaStream_t>(stream_ptr) : 0;
    const int total_assignments = num_tokens * num_selected_experts;
    const bool has_down = (moe_type == 0);
    const int avg_tokens_per_expert = (total_assignments + num_experts - 1) / num_experts;

    // Calculate intermediate buffer size in bytes
    const size_t intermediate_size = (size_t)total_assignments * intermediate_dim * sizeof(float);

    // Threshold for using fused kernel: 32MB intermediate buffer
    // When intermediate is large, the fused kernel avoids the write-read roundtrip
    // But it re-reads input more times, so only use when intermediate dominates
    const size_t FUSED_THRESHOLD = 32 * 1024 * 1024;

    // Use fused kernel when:
    // 1. Intermediate buffer is large (> 32MB)
    // 2. intermediate_dim / TILE_I * input_size < intermediate_size * 2
    //    i.e., extra input reads < intermediate write + read
    const size_t input_elem_size = (dtype == 0 || dtype == 1) ? 2 : 4;
    const size_t input_size = (size_t)num_tokens * hidden_dim * input_elem_size;
    const int num_intermediate_tiles = (intermediate_dim + TILE_I - 1) / TILE_I;
    const size_t extra_input_reads = (size_t)(num_intermediate_tiles - 1) * input_size;

    // Fused kernel is beneficial when extra input reads < intermediate memory traffic
    const bool use_fused_kernel = (intermediate_size > FUSED_THRESHOLD) &&
                                  (extra_input_reads < intermediate_size * 2);

    // Zero counters
    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);
    cudaMemsetAsync(counters, 0, num_experts * sizeof(int), stream);

    // Preprocessing: count experts, compute offsets, build sorted indices
    int blocks = (total_assignments + 255) / 256;
    count_experts_kernel<<<blocks, 256, 0, stream>>>(expert_indices, expert_counts, total_assignments);

    // Compute exclusive offsets on GPU (supports num_experts > 256)
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, expert_counts, expert_offsets, num_experts, stream);

    void* temp_storage = nullptr;
    cudaMallocAsync(&temp_storage, scan_temp_bytes, stream);

    cub::DeviceScan::ExclusiveSum(temp_storage, scan_temp_bytes, expert_counts, expert_offsets, num_experts, stream);

    cudaMemcpyAsync(expert_offsets + num_experts, &total_assignments, sizeof(int), cudaMemcpyHostToDevice, stream);

    cudaFreeAsync(temp_storage, stream);

    if (avg_tokens_per_expert == 0) {
        return;
    }

    // Check if we can use tensor cores (FP16 only, dimensions aligned, SM70+)
    const bool dims_aligned = (hidden_dim % WMMA_K == 0) && (intermediate_dim % WMMA_N == 0);
    bool use_tensor_cores = (dtype == 0) && dims_aligned;
    if (use_tensor_cores) {
        int device = 0;
        int sm_major = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
        if (sm_major < 7) {
            use_tensor_cores = false;
        }
    }

    // Use estimated max for ALL paths to avoid costly D2H sync
    // 2x average + block size is a safe upper bound (handles uneven distribution)
    const int estimated_max_tokens = (avg_tokens_per_expert * 2) + BLOCK_M;

    build_sorted_indices_kernel<<<blocks, 256, 0, stream>>>(
        expert_indices, routing_weights, expert_offsets,
        token_ids, sorted_routing_weights, counters,
        num_tokens, num_selected_experts);

    // PRIORITY: Use tensor cores when available (faster than fused kernel)
    if (use_tensor_cores) {
        const int num_m_tiles = (estimated_max_tokens + TC_BLOCK_M - 1) / TC_BLOCK_M;
        dim3 grid(num_m_tiles, num_experts);

        // Request extended shared memory for tensor core kernels (~50KB)
        // This is a one-time setup per kernel function pointer
        static bool smem_configured = false;
        if (!smem_configured) {
            cudaFuncSetAttribute(moe_gate_up_tensor_core_kernel<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            cudaFuncSetAttribute(moe_gate_up_tensor_core_kernel<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            cudaFuncSetAttribute(moe_output_tensor_core_kernel<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            cudaFuncSetAttribute(moe_output_tensor_core_kernel<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            smem_configured = true;
        }

        if (has_down) {
            moe_gate_up_tensor_core_kernel<true><<<grid, TC_THREADS, TC_GATE_UP_SMEM, stream>>>(
                (const half*)input, (const half*)gate_weights, (const half*)up_weights,
                expert_offsets, token_ids, intermediate_buffer,
                hidden_dim, intermediate_dim, activation_type);

            moe_output_tensor_core_kernel<true><<<grid, TC_THREADS, TC_OUTPUT_SMEM, stream>>>(
                intermediate_buffer, (const half*)down_weights,
                expert_offsets, token_ids, sorted_routing_weights, (half*)output,
                hidden_dim, intermediate_dim);
        } else {
            moe_gate_up_tensor_core_kernel<false><<<grid, TC_THREADS, TC_GATE_UP_SMEM, stream>>>(
                (const half*)input, (const half*)gate_weights, (const half*)up_weights,
                expert_offsets, token_ids, intermediate_buffer,
                hidden_dim, intermediate_dim, activation_type);

            moe_output_tensor_core_kernel<false><<<grid, TC_THREADS, TC_OUTPUT_SMEM, stream>>>(
                intermediate_buffer, (const half*)up_weights,
                expert_offsets, token_ids, sorted_routing_weights, (half*)output,
                hidden_dim, intermediate_dim);
        }
        return;
    }

    if (use_fused_kernel) {
        // Fused single-pass kernel: no intermediate buffer needed
        const int num_m_tiles = (estimated_max_tokens + BLOCK_M_FUSED - 1) / BLOCK_M_FUSED;

        dim3 grid(num_m_tiles, num_experts);

        #define LAUNCH_FUSED(T) \
            if (has_down) { \
                moe_fused_single_pass_kernel<T, true><<<grid, THREADS_FUSED, 0, stream>>>( \
                    (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                    expert_offsets, token_ids, sorted_routing_weights, (T*)output, \
                    hidden_dim, intermediate_dim, activation_type); \
            } else { \
                moe_fused_single_pass_kernel<T, false><<<grid, THREADS_FUSED, 0, stream>>>( \
                    (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                    expert_offsets, token_ids, sorted_routing_weights, (T*)output, \
                    hidden_dim, intermediate_dim, activation_type); \
            }

        if (dtype == 0) { LAUNCH_FUSED(half); }
        else if (dtype == 1) { LAUNCH_FUSED(__nv_bfloat16); }
        else { LAUNCH_FUSED(float); }
        #undef LAUNCH_FUSED
        return;
    }

    // Decision: use small-batch 3D kernel or large-batch 2D kernel
    // Small batch: 3D grid (n_tiles, m_tiles, experts) - simpler, less overhead
    // Large batch: 2D grid (m_tiles, experts) with N-loop inside - better data reuse
    // Adaptive threshold: Qwen3 (with down projection) benefits from 2D kernel sooner
    // due to intermediate buffer reuse; Nomic can use 3D kernel for larger batches
    const int small_batch_threshold = has_down ? SMALL_BATCH_THRESHOLD_QWEN : SMALL_BATCH_THRESHOLD_NOMIC;
    const bool use_small_kernel = (avg_tokens_per_expert < small_batch_threshold);

    if (use_small_kernel) {
        // Small-batch kernel: 3D grid, smaller tiles
        const int num_m_tiles = (estimated_max_tokens + BLOCK_M_SMALL - 1) / BLOCK_M_SMALL;
        const int num_n_tiles_gate = (intermediate_dim + BLOCK_N - 1) / BLOCK_N;
        const int num_n_tiles_out = (hidden_dim + BLOCK_N - 1) / BLOCK_N;

        dim3 grid_gate(num_n_tiles_gate, num_m_tiles, num_experts);
        dim3 grid_out(num_n_tiles_out, num_m_tiles, num_experts);

        #define LAUNCH_SMALL(T) \
            if (has_down) { \
                moe_gate_up_small_kernel<T, true><<<grid_gate, THREADS, 0, stream>>>( \
                    (const T*)input, (const T*)gate_weights, (const T*)up_weights, \
                    expert_offsets, token_ids, intermediate_buffer, \
                    hidden_dim, intermediate_dim, activation_type); \
                moe_output_small_kernel<T, true><<<grid_out, THREADS, 0, stream>>>( \
                    intermediate_buffer, (const T*)down_weights, \
                    expert_offsets, token_ids, sorted_routing_weights, (T*)output, \
                    hidden_dim, intermediate_dim); \
            } else { \
                moe_gate_up_small_kernel<T, false><<<grid_gate, THREADS, 0, stream>>>( \
                    (const T*)input, (const T*)gate_weights, (const T*)up_weights, \
                    expert_offsets, token_ids, intermediate_buffer, \
                    hidden_dim, intermediate_dim, activation_type); \
                moe_output_small_kernel<T, false><<<grid_out, THREADS, 0, stream>>>( \
                    intermediate_buffer, (const T*)up_weights, \
                    expert_offsets, token_ids, sorted_routing_weights, (T*)output, \
                    hidden_dim, intermediate_dim); \
            }

        if (dtype == 0) { LAUNCH_SMALL(half); }
        else if (dtype == 1) { LAUNCH_SMALL(__nv_bfloat16); }
        else { LAUNCH_SMALL(float); }
        #undef LAUNCH_SMALL
    } else {
        // Large-batch kernel: 2D grid with N handled inside
        const int num_m_tiles = (estimated_max_tokens + BLOCK_M - 1) / BLOCK_M;

        dim3 grid(num_m_tiles, num_experts);

        #define LAUNCH_OPTIMIZED(T) \
            if (has_down) { \
                moe_gate_up_optimized_kernel<T, true><<<grid, THREADS, 0, stream>>>( \
                    (const T*)input, (const T*)gate_weights, (const T*)up_weights, \
                    expert_offsets, token_ids, intermediate_buffer, \
                    hidden_dim, intermediate_dim, activation_type); \
                moe_output_optimized_kernel<T, true><<<grid, THREADS, 0, stream>>>( \
                    intermediate_buffer, (const T*)down_weights, \
                    expert_offsets, token_ids, sorted_routing_weights, (T*)output, \
                    hidden_dim, intermediate_dim); \
            } else { \
                moe_gate_up_optimized_kernel<T, false><<<grid, THREADS, 0, stream>>>( \
                    (const T*)input, (const T*)gate_weights, (const T*)up_weights, \
                    expert_offsets, token_ids, intermediate_buffer, \
                    hidden_dim, intermediate_dim, activation_type); \
                moe_output_optimized_kernel<T, false><<<grid, THREADS, 0, stream>>>( \
                    intermediate_buffer, (const T*)up_weights, \
                    expert_offsets, token_ids, sorted_routing_weights, (T*)output, \
                    hidden_dim, intermediate_dim); \
            }

        if (dtype == 0) { LAUNCH_OPTIMIZED(half); }
        else if (dtype == 1) { LAUNCH_OPTIMIZED(__nv_bfloat16); }
        else { LAUNCH_OPTIMIZED(float); }
        #undef LAUNCH_OPTIMIZED
    }
}

} // extern "C"
