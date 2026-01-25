#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <algorithm>

// ============================================================================
// Constants
// ============================================================================
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define SMEM_PAD 1

// ============================================================================
// Device info cache
// ============================================================================
static int g_sm_count = 0;
static int g_max_smem = 0;

inline void ensure_device_info() {
    if (g_sm_count > 0) return;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    g_sm_count = props.multiProcessorCount;
    cudaDeviceGetAttribute(&g_max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (g_max_smem == 0) g_max_smem = props.sharedMemPerBlock;
}

// ============================================================================
// Activation functions
// ============================================================================
__device__ __forceinline__ float activation_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float activation_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float apply_activation(float x, int type) {
    if (type == 0) return activation_silu(x);
    if (type == 1) return activation_gelu(x);
    return fmaxf(0.0f, x);  // ReLU
}

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

template<typename T>
__device__ __forceinline__ float load_as_float(const T* ptr) {
    return to_float(__ldg(ptr));
}

// ============================================================================
// Atomic add - use native when available (SM60+ for half, SM80+ for bf16)
// ============================================================================
__device__ __forceinline__ void atomic_add(float* addr, float val) {
    atomicAdd(addr, val);
}

#if __CUDA_ARCH__ >= 700
// Native half atomicAdd available on Volta+ (SM70+)
__device__ __forceinline__ void atomic_add(half* addr, float val) {
    atomicAdd(addr, __float2half(val));
}
#else
__device__ __forceinline__ void atomic_add(half* addr, float val) {
    // Fallback CAS-based for older architectures
    size_t addr_int = (size_t)addr;
    unsigned int* base_addr = (unsigned int*)(addr_int & ~2ULL);
    bool is_high = (addr_int & 2);

    unsigned int old_val = *base_addr;
    unsigned int assumed, new_val;

    do {
        assumed = old_val;
        unsigned short lo = assumed & 0xFFFF;
        unsigned short hi = (assumed >> 16) & 0xFFFF;

        if (is_high) {
            half h = *reinterpret_cast<half*>(&hi);
            h = __float2half(__half2float(h) + val);
            hi = *reinterpret_cast<unsigned short*>(&h);
        } else {
            half h = *reinterpret_cast<half*>(&lo);
            h = __float2half(__half2float(h) + val);
            lo = *reinterpret_cast<unsigned short*>(&h);
        }

        new_val = lo | (hi << 16);
        old_val = atomicCAS(base_addr, assumed, new_val);
    } while (assumed != old_val);
}
#endif

#if __CUDA_ARCH__ >= 800
// Native bfloat16 atomicAdd available on Ampere+ (SM80+)
__device__ __forceinline__ void atomic_add(__nv_bfloat16* addr, float val) {
    atomicAdd(addr, __float2bfloat16(val));
}
#else
__device__ __forceinline__ void atomic_add(__nv_bfloat16* addr, float val) {
    // Fallback CAS-based for older architectures
    size_t addr_int = (size_t)addr;
    unsigned int* base_addr = (unsigned int*)(addr_int & ~2ULL);
    bool is_high = (addr_int & 2);

    unsigned int old_val = *base_addr;
    unsigned int assumed, new_val;

    do {
        assumed = old_val;
        unsigned short lo = assumed & 0xFFFF;
        unsigned short hi = (assumed >> 16) & 0xFFFF;

        if (is_high) {
            __nv_bfloat16 h = *reinterpret_cast<__nv_bfloat16*>(&hi);
            h = __float2bfloat16(__bfloat162float(h) + val);
            hi = *reinterpret_cast<unsigned short*>(&h);
        } else {
            __nv_bfloat16 h = *reinterpret_cast<__nv_bfloat16*>(&lo);
            h = __float2bfloat16(__bfloat162float(h) + val);
            lo = *reinterpret_cast<unsigned short*>(&h);
        }

        new_val = lo | (hi << 16);
        old_val = atomicCAS(base_addr, assumed, new_val);
    } while (assumed != old_val);
}
#endif

// ============================================================================
// Main MoE Kernel (Token-parallel) with register blocking
// One block per token, processes all selected experts
// ============================================================================
#define OUTPUTS_PER_THREAD 4

template<typename T, bool HAS_DOWN_WEIGHTS>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void moe_kernel(
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
    float* smem_inter = smem + hidden_dim + SMEM_PAD;

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    // Load input to shared memory
    const T* token_input = input + token_idx * hidden_dim;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        smem_input[i] = load_as_float(token_input + i);
    }
    __syncthreads();

    T* token_output = output + token_idx * hidden_dim;

    // Compute stride for output parallelism
    const int outputs_per_warp = OUTPUTS_PER_THREAD * WARP_SIZE;
    const int outputs_per_iter = num_warps * outputs_per_warp;

    // Process each selected expert
    for (int k = 0; k < num_selected_experts; k++) {
        const int expert_id = expert_indices[token_idx * num_selected_experts + k];
        const float routing_weight = routing_weights[token_idx * num_selected_experts + k];

        const T* gate_w = gate_weights + expert_id * hidden_dim * intermediate_dim;
        const T* up_w = up_weights + expert_id * hidden_dim * intermediate_dim;

        // Phase 1: Gate (and Up when needed) projection with register blocking
        if constexpr (HAS_DOWN_WEIGHTS) {
            for (int base_i = warp_id * outputs_per_warp; base_i < intermediate_dim; base_i += outputs_per_iter) {
                float gate_acc[OUTPUTS_PER_THREAD] = {0};
                float up_acc[OUTPUTS_PER_THREAD] = {0};

                #pragma unroll 4
                for (int j = 0; j < hidden_dim; j++) {
                    const float inp = smem_input[j];
                    const int row_off = j * intermediate_dim;

                    #pragma unroll
                    for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                        const int i = base_i + lane_id + r * WARP_SIZE;
                        if (i < intermediate_dim) {
                            gate_acc[r] += inp * load_as_float(gate_w + row_off + i);
                            up_acc[r] += inp * load_as_float(up_w + row_off + i);
                        }
                    }
                }

                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < intermediate_dim) {
                        float activated = apply_activation(gate_acc[r], activation_type);
                        smem_inter[i] = activated * up_acc[r];
                    }
                }
            }
        } else {
            for (int base_i = warp_id * outputs_per_warp; base_i < intermediate_dim; base_i += outputs_per_iter) {
                float gate_acc[OUTPUTS_PER_THREAD] = {0};

                #pragma unroll 4
                for (int j = 0; j < hidden_dim; j++) {
                    const float inp = smem_input[j];
                    const int row_off = j * intermediate_dim;

                    #pragma unroll
                    for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                        const int i = base_i + lane_id + r * WARP_SIZE;
                        if (i < intermediate_dim) {
                            gate_acc[r] += inp * load_as_float(gate_w + row_off + i);
                        }
                    }
                }

                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < intermediate_dim) {
                        smem_inter[i] = apply_activation(gate_acc[r], activation_type);
                    }
                }
            }
        }
        __syncthreads();

        // Phase 2: Down projection with register blocking
        if (HAS_DOWN_WEIGHTS) {
            const T* down_w = down_weights + expert_id * intermediate_dim * hidden_dim;

            for (int base_i = warp_id * outputs_per_warp; base_i < hidden_dim; base_i += outputs_per_iter) {
                float acc[OUTPUTS_PER_THREAD] = {0};

                #pragma unroll 4
                for (int j = 0; j < intermediate_dim; j++) {
                    const float inter = smem_inter[j];
                    const int row_off = j * hidden_dim;

                    #pragma unroll
                    for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                        const int i = base_i + lane_id + r * WARP_SIZE;
                        if (i < hidden_dim) {
                            acc[r] += inter * load_as_float(down_w + row_off + i);
                        }
                    }
                }

                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < hidden_dim) {
                        float current = to_float(token_output[i]);
                        token_output[i] = from_float<T>(current + acc[r] * routing_weight);
                    }
                }
            }
        } else {
            // Nomic: up_weights for output (different layout - need warp reduction)
            for (int i = warp_id; i < hidden_dim; i += num_warps) {
                float acc = 0.0f;

                #pragma unroll 4
                for (int j = lane_id; j < intermediate_dim; j += WARP_SIZE) {
                    acc += smem_inter[j] * load_as_float(up_w + i * intermediate_dim + j);
                }

                // Warp reduction
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    acc += __shfl_xor_sync(0xffffffff, acc, offset);
                }

                if (lane_id == 0) {
                    float current = to_float(token_output[i]);
                    token_output[i] = from_float<T>(current + acc * routing_weight);
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Expert-Parallel Kernel (For large batches) with register blocking
// One block per (token, expert) pair - better L2 cache utilization
// ============================================================================
template<typename T, bool HAS_DOWN_WEIGHTS>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void moe_expert_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gate_weights,
    const T* __restrict__ up_weights,
    const T* __restrict__ down_weights,
    const float* __restrict__ routing_weights,
    const int* __restrict__ token_ids,
    const int* __restrict__ select_ids,
    T* __restrict__ output,
    int expert_id,
    int num_tokens_for_expert,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int activation_type
) {
    extern __shared__ float smem[];
    float* smem_input = smem;
    float* smem_inter = smem + hidden_dim + SMEM_PAD;

    const int local_idx = blockIdx.x;
    if (local_idx >= num_tokens_for_expert) return;

    const int token_idx = token_ids[local_idx];
    const int select_idx = select_ids[local_idx];
    const float routing_weight = routing_weights[token_idx * num_selected_experts + select_idx];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    // Load input
    const T* token_input = input + token_idx * hidden_dim;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        smem_input[i] = load_as_float(token_input + i);
    }
    __syncthreads();

    const T* gate_w = gate_weights + expert_id * hidden_dim * intermediate_dim;
    const T* up_w = up_weights + expert_id * hidden_dim * intermediate_dim;

    const int outputs_per_warp = OUTPUTS_PER_THREAD * WARP_SIZE;
    const int outputs_per_iter = num_warps * outputs_per_warp;

    // Phase 1: Gate (and Up when needed) with register blocking
    if constexpr (HAS_DOWN_WEIGHTS) {
        for (int base_i = warp_id * outputs_per_warp; base_i < intermediate_dim; base_i += outputs_per_iter) {
            float gate_acc[OUTPUTS_PER_THREAD] = {0};
            float up_acc[OUTPUTS_PER_THREAD] = {0};

            #pragma unroll 4
            for (int j = 0; j < hidden_dim; j++) {
                const float inp = smem_input[j];
                const int row_off = j * intermediate_dim;

                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < intermediate_dim) {
                        gate_acc[r] += inp * load_as_float(gate_w + row_off + i);
                        up_acc[r] += inp * load_as_float(up_w + row_off + i);
                    }
                }
            }

            #pragma unroll
            for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                const int i = base_i + lane_id + r * WARP_SIZE;
                if (i < intermediate_dim) {
                    float activated = apply_activation(gate_acc[r], activation_type);
                    smem_inter[i] = activated * up_acc[r];
                }
            }
        }
    } else {
        for (int base_i = warp_id * outputs_per_warp; base_i < intermediate_dim; base_i += outputs_per_iter) {
            float gate_acc[OUTPUTS_PER_THREAD] = {0};

            #pragma unroll 4
            for (int j = 0; j < hidden_dim; j++) {
                const float inp = smem_input[j];
                const int row_off = j * intermediate_dim;

                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < intermediate_dim) {
                        gate_acc[r] += inp * load_as_float(gate_w + row_off + i);
                    }
                }
            }

            #pragma unroll
            for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                const int i = base_i + lane_id + r * WARP_SIZE;
                if (i < intermediate_dim) {
                    smem_inter[i] = apply_activation(gate_acc[r], activation_type);
                }
            }
        }
    }
    __syncthreads();

    // Phase 2: Down projection with register blocking + atomic add
    T* token_output = output + token_idx * hidden_dim;

    if (HAS_DOWN_WEIGHTS) {
        const T* down_w = down_weights + expert_id * intermediate_dim * hidden_dim;

        for (int base_i = warp_id * outputs_per_warp; base_i < hidden_dim; base_i += outputs_per_iter) {
            float acc[OUTPUTS_PER_THREAD] = {0};

            #pragma unroll 4
            for (int j = 0; j < intermediate_dim; j++) {
                const float inter = smem_inter[j];
                const int row_off = j * hidden_dim;

                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < hidden_dim) {
                        acc[r] += inter * load_as_float(down_w + row_off + i);
                    }
                }
            }

            #pragma unroll
            for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                const int i = base_i + lane_id + r * WARP_SIZE;
                if (i < hidden_dim) {
                    atomic_add(token_output + i, acc[r] * routing_weight);
                }
            }
        }
    } else {
        // Nomic: warp reduction needed
        for (int i = warp_id; i < hidden_dim; i += num_warps) {
            float acc = 0.0f;

            #pragma unroll 4
            for (int j = lane_id; j < intermediate_dim; j += WARP_SIZE) {
                acc += smem_inter[j] * load_as_float(up_w + i * intermediate_dim + j);
            }

            // Warp reduction
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                acc += __shfl_xor_sync(0xffffffff, acc, offset);
            }

            if (lane_id == 0) {
                atomic_add(token_output + i, acc * routing_weight);
            }
        }
    }
}

// ============================================================================
// Batched Expert Kernel - processes multiple tokens per block
// Loads weight tiles to shared memory for reuse across tokens
// ============================================================================
#define TILE_K 64
#define TOKENS_PER_BLOCK 4  // Keep at 4 for larger models, smem limited

template<typename T, bool HAS_DOWN_WEIGHTS>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void moe_batched_expert_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gate_weights,
    const T* __restrict__ up_weights,
    const T* __restrict__ down_weights,
    const float* __restrict__ routing_weights,
    const int* __restrict__ token_ids,
    const int* __restrict__ select_ids,
    T* __restrict__ output,
    int expert_id,
    int num_tokens_for_expert,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int activation_type
) {
    // Shared memory layout:
    // - Input buffer: TOKENS_PER_BLOCK * hidden_dim
    // - Weight tile: TILE_K * BLOCK_SIZE (for coalesced access)
    // - Intermediate: TOKENS_PER_BLOCK * intermediate_dim
    extern __shared__ float smem[];

    const int block_token_start = blockIdx.x * TOKENS_PER_BLOCK;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    // Pointers into shared memory
    float* smem_inputs = smem;  // [TOKENS_PER_BLOCK][hidden_dim]
    float* smem_inter = smem_inputs + TOKENS_PER_BLOCK * (hidden_dim + SMEM_PAD);  // [TOKENS_PER_BLOCK][intermediate_dim]

    const T* gate_w = gate_weights + expert_id * hidden_dim * intermediate_dim;
    const T* up_w = up_weights + expert_id * hidden_dim * intermediate_dim;

    // Load inputs for all tokens in this block
    #pragma unroll
    for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
        const int global_t = block_token_start + t;
        if (global_t < num_tokens_for_expert) {
            const int token_idx = token_ids[global_t];
            const T* token_input = input + token_idx * hidden_dim;
            float* smem_input_t = smem_inputs + t * (hidden_dim + SMEM_PAD);

            for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
                smem_input_t[i] = load_as_float(token_input + i);
            }
        }
    }
    __syncthreads();

    // Phase 1: Gate (and Up when needed) projection for all tokens
    const int outputs_per_warp = OUTPUTS_PER_THREAD * WARP_SIZE;
    const int outputs_per_iter = num_warps * outputs_per_warp;

    if constexpr (HAS_DOWN_WEIGHTS) {
        for (int base_i = warp_id * outputs_per_warp; base_i < intermediate_dim; base_i += outputs_per_iter) {
            float gate_acc[TOKENS_PER_BLOCK][OUTPUTS_PER_THREAD];
            float up_acc[TOKENS_PER_BLOCK][OUTPUTS_PER_THREAD];

            #pragma unroll
            for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    gate_acc[t][r] = 0.0f;
                    up_acc[t][r] = 0.0f;
                }
            }

            #pragma unroll 2
            for (int j = 0; j < hidden_dim; j++) {
                const int row_off = j * intermediate_dim;

                float gate_vals[OUTPUTS_PER_THREAD];
                float up_vals[OUTPUTS_PER_THREAD];

                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < intermediate_dim) {
                        gate_vals[r] = load_as_float(gate_w + row_off + i);
                        up_vals[r] = load_as_float(up_w + row_off + i);
                    }
                }

                #pragma unroll
                for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                    const int global_t = block_token_start + t;
                    if (global_t < num_tokens_for_expert) {
                        const float inp = smem_inputs[t * (hidden_dim + SMEM_PAD) + j];

                        #pragma unroll
                        for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                            const int i = base_i + lane_id + r * WARP_SIZE;
                            if (i < intermediate_dim) {
                                gate_acc[t][r] += inp * gate_vals[r];
                                up_acc[t][r] += inp * up_vals[r];
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                const int global_t = block_token_start + t;
                if (global_t < num_tokens_for_expert) {
                    float* smem_inter_t = smem_inter + t * (intermediate_dim + SMEM_PAD);

                    #pragma unroll
                    for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                        const int i = base_i + lane_id + r * WARP_SIZE;
                        if (i < intermediate_dim) {
                            float activated = apply_activation(gate_acc[t][r], activation_type);
                            smem_inter_t[i] = activated * up_acc[t][r];
                        }
                    }
                }
            }
        }
    } else {
        for (int base_i = warp_id * outputs_per_warp; base_i < intermediate_dim; base_i += outputs_per_iter) {
            float gate_acc[TOKENS_PER_BLOCK][OUTPUTS_PER_THREAD];

            #pragma unroll
            for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    gate_acc[t][r] = 0.0f;
                }
            }

            #pragma unroll 2
            for (int j = 0; j < hidden_dim; j++) {
                const int row_off = j * intermediate_dim;

                float gate_vals[OUTPUTS_PER_THREAD];

                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < intermediate_dim) {
                        gate_vals[r] = load_as_float(gate_w + row_off + i);
                    }
                }

                #pragma unroll
                for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                    const int global_t = block_token_start + t;
                    if (global_t < num_tokens_for_expert) {
                        const float inp = smem_inputs[t * (hidden_dim + SMEM_PAD) + j];

                        #pragma unroll
                        for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                            const int i = base_i + lane_id + r * WARP_SIZE;
                            if (i < intermediate_dim) {
                                gate_acc[t][r] += inp * gate_vals[r];
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                const int global_t = block_token_start + t;
                if (global_t < num_tokens_for_expert) {
                    float* smem_inter_t = smem_inter + t * (intermediate_dim + SMEM_PAD);

                    #pragma unroll
                    for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                        const int i = base_i + lane_id + r * WARP_SIZE;
                        if (i < intermediate_dim) {
                            smem_inter_t[i] = apply_activation(gate_acc[t][r], activation_type);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // Phase 2: Down projection
    if (HAS_DOWN_WEIGHTS) {
        const T* down_w = down_weights + expert_id * intermediate_dim * hidden_dim;

        for (int base_i = warp_id * outputs_per_warp; base_i < hidden_dim; base_i += outputs_per_iter) {
            float acc[TOKENS_PER_BLOCK][OUTPUTS_PER_THREAD];

            #pragma unroll
            for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    acc[t][r] = 0.0f;
                }
            }

            #pragma unroll 2
            for (int j = 0; j < intermediate_dim; j++) {
                const int row_off = j * hidden_dim;

                // Load weight values
                float down_vals[OUTPUTS_PER_THREAD];
                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < hidden_dim) {
                        down_vals[r] = load_as_float(down_w + row_off + i);
                    }
                }

                // Apply to all tokens
                #pragma unroll
                for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                    const int global_t = block_token_start + t;
                    if (global_t < num_tokens_for_expert) {
                        const float inter = smem_inter[t * (intermediate_dim + SMEM_PAD) + j];

                        #pragma unroll
                        for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                            const int i = base_i + lane_id + r * WARP_SIZE;
                            if (i < hidden_dim) {
                                acc[t][r] += inter * down_vals[r];
                            }
                        }
                    }
                }
            }

            // Write outputs with atomic add
            #pragma unroll
            for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                const int global_t = block_token_start + t;
                if (global_t < num_tokens_for_expert) {
                    const int token_idx = token_ids[global_t];
                    const int select_idx = select_ids[global_t];
                    const float routing_weight = routing_weights[token_idx * num_selected_experts + select_idx];
                    T* token_output = output + token_idx * hidden_dim;

                    #pragma unroll
                    for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                        const int i = base_i + lane_id + r * WARP_SIZE;
                        if (i < hidden_dim) {
                            atomic_add(token_output + i, acc[t][r] * routing_weight);
                        }
                    }
                }
            }
        }
    } else {
        // Nomic path - reuse up_w loads across tokens in this block.
        float routing_weight_arr[TOKENS_PER_BLOCK];
        T* token_output_arr[TOKENS_PER_BLOCK];
        float* smem_inter_arr[TOKENS_PER_BLOCK];
        bool active[TOKENS_PER_BLOCK];

        #pragma unroll
        for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
            const int global_t = block_token_start + t;
            active[t] = (global_t < num_tokens_for_expert);
            if (active[t]) {
                const int token_idx = token_ids[global_t];
                const int select_idx = select_ids[global_t];
                routing_weight_arr[t] = routing_weights[token_idx * num_selected_experts + select_idx];
                token_output_arr[t] = output + token_idx * hidden_dim;
                smem_inter_arr[t] = smem_inter + t * (intermediate_dim + SMEM_PAD);
            }
        }

        for (int i = warp_id; i < hidden_dim; i += num_warps) {
            float acc[TOKENS_PER_BLOCK];
            #pragma unroll
            for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                acc[t] = 0.0f;
            }

            for (int j = lane_id; j < intermediate_dim; j += WARP_SIZE) {
                const float up_val = load_as_float(up_w + i * intermediate_dim + j);
                #pragma unroll
                for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                    if (active[t]) {
                        acc[t] += smem_inter_arr[t][j] * up_val;
                    }
                }
            }

            // Warp reduction
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                #pragma unroll
                for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                    acc[t] += __shfl_xor_sync(0xffffffff, acc[t], offset);
                }
            }

            if (lane_id == 0) {
                #pragma unroll
                for (int t = 0; t < TOKENS_PER_BLOCK; t++) {
                    if (active[t]) {
                        atomic_add(token_output_arr[t] + i, acc[t] * routing_weight_arr[t]);
                    }
                }
            }
        }
    }
}

// ============================================================================
// MEGA KERNEL: All experts in a single launch, no host dispatch
// Each block finds its expert via binary search in offsets array
// TPB = Tokens Per Block (template parameter for compile-time optimization)
// ============================================================================
#define MAX_EXPERTS 256

template<typename T, bool HAS_DOWN_WEIGHTS, int TPB>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void moe_mega_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gate_weights,
    const T* __restrict__ up_weights,
    const T* __restrict__ down_weights,
    const float* __restrict__ routing_weights,
    const int* __restrict__ token_offsets,  // [num_experts + 1] - token offset per expert
    const int* __restrict__ block_offsets,  // [num_experts + 1] - block offset per expert
    const int* __restrict__ token_ids,
    const int* __restrict__ select_ids,
    T* __restrict__ output,
    int num_experts,
    int _unused,  // kept for ABI compatibility
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int activation_type
) {
    // Read actual total blocks from block_offsets[num_experts]
    const int total_blocks = block_offsets[num_experts];

    // Early exit if this block is beyond the actual total
    if (blockIdx.x >= total_blocks) return;

    // Binary search on block_offsets to find which expert this block belongs to
    int expert_id = 0;
    {
        int lo = 0, hi = num_experts;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (block_offsets[mid] <= blockIdx.x) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        expert_id = lo;
    }

    // Compute local block index within this expert
    const int local_block = blockIdx.x - block_offsets[expert_id];
    const int expert_start = token_offsets[expert_id];
    const int expert_end = token_offsets[expert_id + 1];
    const int local_start = local_block * TPB;

    extern __shared__ float smem[];
    float* smem_inputs = smem;
    float* smem_inter = smem_inputs + TPB * (hidden_dim + SMEM_PAD);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    const T* gate_w = gate_weights + expert_id * hidden_dim * intermediate_dim;
    const T* up_w = up_weights + expert_id * hidden_dim * intermediate_dim;

    // Load inputs for tokens in this block
    #pragma unroll
    for (int t = 0; t < TPB; t++) {
        const int global_idx = expert_start + local_start + t;
        if (global_idx < expert_end) {
            const int token_idx = token_ids[global_idx];
            const T* token_input = input + token_idx * hidden_dim;
            float* smem_input_t = smem_inputs + t * (hidden_dim + SMEM_PAD);

            for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
                smem_input_t[i] = load_as_float(token_input + i);
            }
        }
    }
    __syncthreads();

    // Phase 1: Gate-Up projection
    const int outputs_per_warp = OUTPUTS_PER_THREAD * WARP_SIZE;
    const int outputs_per_iter = num_warps * outputs_per_warp;

    for (int base_i = warp_id * outputs_per_warp; base_i < intermediate_dim; base_i += outputs_per_iter) {
        float gate_acc[TPB][OUTPUTS_PER_THREAD];
        float up_acc[TPB][OUTPUTS_PER_THREAD];

        #pragma unroll
        for (int t = 0; t < TPB; t++) {
            #pragma unroll
            for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                gate_acc[t][r] = 0.0f;
                up_acc[t][r] = 0.0f;
            }
        }

        #pragma unroll 2
        for (int j = 0; j < hidden_dim; j++) {
            const int row_off = j * intermediate_dim;

            float gate_vals[OUTPUTS_PER_THREAD];
            float up_vals[OUTPUTS_PER_THREAD];

            #pragma unroll
            for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                const int i = base_i + lane_id + r * WARP_SIZE;
                if (i < intermediate_dim) {
                    gate_vals[r] = load_as_float(gate_w + row_off + i);
                    up_vals[r] = HAS_DOWN_WEIGHTS ? load_as_float(up_w + row_off + i) : 0.0f;
                }
            }

            #pragma unroll
            for (int t = 0; t < TPB; t++) {
                const int global_idx = expert_start + local_start + t;
                if (global_idx < expert_end) {
                    const float inp = smem_inputs[t * (hidden_dim + SMEM_PAD) + j];

                    #pragma unroll
                    for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                        const int i = base_i + lane_id + r * WARP_SIZE;
                        if (i < intermediate_dim) {
                            gate_acc[t][r] += inp * gate_vals[r];
                            if (HAS_DOWN_WEIGHTS) up_acc[t][r] += inp * up_vals[r];
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < TPB; t++) {
            const int global_idx = expert_start + local_start + t;
            if (global_idx < expert_end) {
                float* smem_inter_t = smem_inter + t * (intermediate_dim + SMEM_PAD);

                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < intermediate_dim) {
                        float activated = apply_activation(gate_acc[t][r], activation_type);
                        smem_inter_t[i] = HAS_DOWN_WEIGHTS ? (activated * up_acc[t][r]) : activated;
                    }
                }
            }
        }
    }
    __syncthreads();

    // Phase 2: Down projection
    if (HAS_DOWN_WEIGHTS) {
        const T* down_w = down_weights + expert_id * intermediate_dim * hidden_dim;

        for (int base_i = warp_id * outputs_per_warp; base_i < hidden_dim; base_i += outputs_per_iter) {
            float acc[TPB][OUTPUTS_PER_THREAD];

            #pragma unroll
            for (int t = 0; t < TPB; t++) {
                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    acc[t][r] = 0.0f;
                }
            }

            #pragma unroll 2
            for (int j = 0; j < intermediate_dim; j++) {
                const int row_off = j * hidden_dim;

                float down_vals[OUTPUTS_PER_THREAD];
                #pragma unroll
                for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                    const int i = base_i + lane_id + r * WARP_SIZE;
                    if (i < hidden_dim) {
                        down_vals[r] = load_as_float(down_w + row_off + i);
                    }
                }

                #pragma unroll
                for (int t = 0; t < TPB; t++) {
                    const int global_idx = expert_start + local_start + t;
                    if (global_idx < expert_end) {
                        const float inter = smem_inter[t * (intermediate_dim + SMEM_PAD) + j];

                        #pragma unroll
                        for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                            const int i = base_i + lane_id + r * WARP_SIZE;
                            if (i < hidden_dim) {
                                acc[t][r] += inter * down_vals[r];
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (int t = 0; t < TPB; t++) {
                const int global_idx = expert_start + local_start + t;
                if (global_idx < expert_end) {
                    const int token_idx = token_ids[global_idx];
                    const int select_idx = select_ids[global_idx];
                    const float routing_weight = routing_weights[token_idx * num_selected_experts + select_idx];
                    T* token_output = output + token_idx * hidden_dim;

                    #pragma unroll
                    for (int r = 0; r < OUTPUTS_PER_THREAD; r++) {
                        const int i = base_i + lane_id + r * WARP_SIZE;
                        if (i < hidden_dim) {
                            atomic_add(token_output + i, acc[t][r] * routing_weight);
                        }
                    }
                }
            }
        }
    } else {
        // Nomic path: reuse up_w loads across tokens in this block.
        float routing_weight_arr[TPB];
        T* token_output_arr[TPB];
        float* smem_inter_arr[TPB];
        bool active[TPB];

        #pragma unroll
        for (int t = 0; t < TPB; t++) {
            const int global_idx = expert_start + local_start + t;
            active[t] = (global_idx < expert_end);
            if (active[t]) {
                const int token_idx = token_ids[global_idx];
                const int select_idx = select_ids[global_idx];
                routing_weight_arr[t] = routing_weights[token_idx * num_selected_experts + select_idx];
                token_output_arr[t] = output + token_idx * hidden_dim;
                smem_inter_arr[t] = smem_inter + t * (intermediate_dim + SMEM_PAD);
            }
        }

        for (int i = warp_id; i < hidden_dim; i += num_warps) {
            float acc[TPB];
            #pragma unroll
            for (int t = 0; t < TPB; t++) {
                acc[t] = 0.0f;
            }

            for (int j = lane_id; j < intermediate_dim; j += WARP_SIZE) {
                const float up_val = load_as_float(up_w + i * intermediate_dim + j);
                #pragma unroll
                for (int t = 0; t < TPB; t++) {
                    if (active[t]) {
                        acc[t] += smem_inter_arr[t][j] * up_val;
                    }
                }
            }

            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                #pragma unroll
                for (int t = 0; t < TPB; t++) {
                    acc[t] += __shfl_xor_sync(0xffffffff, acc[t], offset);
                }
            }

            if (lane_id == 0) {
                #pragma unroll
                for (int t = 0; t < TPB; t++) {
                    if (active[t]) {
                        atomic_add(token_output_arr[t] + i, acc[t] * routing_weight_arr[t]);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Preprocessing kernels for expert-parallel mode
// ============================================================================
__global__ void count_tokens_per_expert_kernel(
    const uint32_t* __restrict__ expert_indices,
    int* __restrict__ counts,
    int total_assignments
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_assignments) {
        atomicAdd(&counts[expert_indices[idx]], 1);
    }
}

// GPU-based exclusive prefix sum (fast for small num_experts <= 256)
// Computes both token offsets and block offsets
__global__ void exclusive_scan_kernel(
    const int* __restrict__ counts,
    int* __restrict__ token_offsets,  // [num_experts + 1] - cumsum of token counts
    int* __restrict__ block_offsets,  // [num_experts + 1] - cumsum of blocks per expert
    int num_experts,
    int tokens_per_block
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int token_sum = 0;
        int block_sum = 0;
        for (int i = 0; i < num_experts; i++) {
            token_offsets[i] = token_sum;
            block_offsets[i] = block_sum;
            token_sum += counts[i];
            // ceil(counts[i] / tokens_per_block)
            block_sum += (counts[i] + tokens_per_block - 1) / tokens_per_block;
        }
        token_offsets[num_experts] = token_sum;
        block_offsets[num_experts] = block_sum;
    }
}

__global__ void build_token_lists_kernel(
    const uint32_t* __restrict__ expert_indices,
    const int* __restrict__ offsets,
    int* __restrict__ token_ids_out,
    int* __restrict__ select_ids_out,
    int* __restrict__ counters,
    int num_tokens,
    int num_selected
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * num_selected) return;

    int token_idx = idx / num_selected;
    int select_idx = idx % num_selected;
    int expert_id = expert_indices[idx];

    int pos = atomicAdd(&counters[expert_id], 1);
    int out_idx = offsets[expert_id] + pos;
    token_ids_out[out_idx] = token_idx;
    select_ids_out[out_idx] = select_idx;
}

// ============================================================================
// Host-side helpers
// ============================================================================
static inline size_t dtype_size_bytes(uint32_t dtype) {
    return (dtype == 2) ? 4 : 2;
}

// ============================================================================
// C Interface
// ============================================================================
extern "C" {

// Standard token-parallel MoE (output must be pre-zeroed)
void fused_moe(
    void* input,
    void* gate_weights,
    void* up_weights,
    void* down_weights,
    float* routing_weights,
    uint32_t* expert_indices,
    void* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int activation_type,
    uint32_t moe_type,  // 0 = Qwen3 (has down), 1 = Nomic (no down)
    uint32_t dtype      // 0 = FP16, 1 = BF16, 2 = FP32
) {
    ensure_device_info();

    const int smem_size = (hidden_dim + SMEM_PAD + intermediate_dim + SMEM_PAD) * sizeof(float);
    if (smem_size > g_max_smem) return;

    const bool has_down = (moe_type == 0);
    cudaStream_t stream = 0;

    #define LAUNCH_KERNEL(T) \
        cudaFuncSetAttribute(has_down ? moe_kernel<T, true> : moe_kernel<T, false>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
        if (has_down) { \
            moe_kernel<T, true><<<num_tokens, BLOCK_SIZE, smem_size, stream>>>( \
                (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                routing_weights, expert_indices, (T*)output, \
                hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
        } else { \
            moe_kernel<T, false><<<num_tokens, BLOCK_SIZE, smem_size, stream>>>( \
                (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                routing_weights, expert_indices, (T*)output, \
                hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
        }

    if (dtype == 0) { LAUNCH_KERNEL(half); }
    else if (dtype == 1) { LAUNCH_KERNEL(__nv_bfloat16); }
    else { LAUNCH_KERNEL(float); }

    #undef LAUNCH_KERNEL
}

// Expert-parallel MoE (call once per expert)
// Automatically uses batched kernel for better weight reuse when enough tokens
void fused_moe_expert(
    void* input,
    void* gate_weights,
    void* up_weights,
    void* down_weights,
    float* routing_weights,
    int* token_ids,
    int* select_ids,
    void* output,
    int expert_id,
    int num_tokens_for_expert,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int activation_type,
    uint32_t moe_type,
    uint32_t dtype
) {
    if (num_tokens_for_expert == 0) return;

    ensure_device_info();

    const bool has_down = (moe_type == 0);
    cudaStream_t stream = 0;

    // Use batched kernel when we have enough tokens for weight reuse benefit
    const bool use_batched = (num_tokens_for_expert >= TOKENS_PER_BLOCK);

    if (use_batched) {
        // Batched kernel: multiple tokens per block share weight loads
        const int batched_smem = TOKENS_PER_BLOCK * (hidden_dim + SMEM_PAD + intermediate_dim + SMEM_PAD) * sizeof(float);
        if (batched_smem > g_max_smem) goto use_single;  // Fallback if not enough smem

        const int num_blocks = (num_tokens_for_expert + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;

        #define LAUNCH_BATCHED_KERNEL(T) \
            cudaFuncSetAttribute(has_down ? moe_batched_expert_kernel<T, true> : moe_batched_expert_kernel<T, false>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, batched_smem); \
            if (has_down) { \
                moe_batched_expert_kernel<T, true><<<num_blocks, BLOCK_SIZE, batched_smem, stream>>>( \
                    (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                    routing_weights, token_ids, select_ids, (T*)output, \
                    expert_id, num_tokens_for_expert, hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
            } else { \
                moe_batched_expert_kernel<T, false><<<num_blocks, BLOCK_SIZE, batched_smem, stream>>>( \
                    (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                    routing_weights, token_ids, select_ids, (T*)output, \
                    expert_id, num_tokens_for_expert, hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
            }

        if (dtype == 0) { LAUNCH_BATCHED_KERNEL(half); }
        else if (dtype == 1) { LAUNCH_BATCHED_KERNEL(__nv_bfloat16); }
        else { LAUNCH_BATCHED_KERNEL(float); }

        #undef LAUNCH_BATCHED_KERNEL
        return;
    }

use_single:
    // Single-token kernel: one token per block
    const int smem_size = (hidden_dim + SMEM_PAD + intermediate_dim + SMEM_PAD) * sizeof(float);
    if (smem_size > g_max_smem) return;

    #define LAUNCH_EXPERT_KERNEL(T) \
        cudaFuncSetAttribute(has_down ? moe_expert_kernel<T, true> : moe_expert_kernel<T, false>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
        if (has_down) { \
            moe_expert_kernel<T, true><<<num_tokens_for_expert, BLOCK_SIZE, smem_size, stream>>>( \
                (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                routing_weights, token_ids, select_ids, (T*)output, \
                expert_id, num_tokens_for_expert, hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
        } else { \
            moe_expert_kernel<T, false><<<num_tokens_for_expert, BLOCK_SIZE, smem_size, stream>>>( \
                (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                routing_weights, token_ids, select_ids, (T*)output, \
                expert_id, num_tokens_for_expert, hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
        }

    if (dtype == 0) { LAUNCH_EXPERT_KERNEL(half); }
    else if (dtype == 1) { LAUNCH_EXPERT_KERNEL(__nv_bfloat16); }
    else { LAUNCH_EXPERT_KERNEL(float); }

    #undef LAUNCH_EXPERT_KERNEL
}

// Helper: count tokens per expert
void moe_count_tokens(
    uint32_t* expert_indices,
    int* expert_counts,
    int num_tokens,
    int num_selected,
    int num_experts
) {
    int total = num_tokens * num_selected;
    int blocks = (total + 255) / 256;
    count_tokens_per_expert_kernel<<<blocks, 256>>>(expert_indices, expert_counts, total);
}

// Helper: build sorted token lists
void moe_build_indices(
    uint32_t* expert_indices,
    int* expert_offsets,
    int* token_ids_out,
    int* select_ids_out,
    int* counters,
    int num_tokens,
    int num_selected
) {
    int total = num_tokens * num_selected;
    int blocks = (total + 255) / 256;
    build_token_lists_kernel<<<blocks, 256>>>(
        expert_indices, expert_offsets, token_ids_out, select_ids_out,
        counters, num_tokens, num_selected);
}

// Query device info
void moe_get_device_info(int* sm_count, int* max_smem) {
    ensure_device_info();
    *sm_count = g_sm_count;
    *max_smem = g_max_smem;
}

// ==========================================================================
// AUTO MODE: Single mega-kernel launch, NO host synchronization
// All experts processed in one kernel using GPU-side dispatch
// ==========================================================================
void fused_moe_auto(
    void* input,
    void* gate_weights,
    void* up_weights,
    void* down_weights,
    float* routing_weights,
    uint32_t* expert_indices,
    void* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int num_selected_experts,
    int activation_type,
    uint32_t moe_type,
    uint32_t dtype,
    // Workspace (caller provides pre-allocated buffers)
    int* expert_counts,      // [num_experts]
    int* expert_offsets,     // [num_experts + 1] - token offsets per expert
    int* token_ids,          // [num_tokens * num_selected_experts]
    int* select_ids,         // [num_tokens * num_selected_experts]
    int* counters,           // [num_experts] - temp for atomic counters
    int* block_offsets       // [num_experts + 1] - block offsets per expert
) {
    // For small batches, use token-parallel (simpler, less overhead)
    if (num_tokens < 16) {
        fused_moe(input, gate_weights, up_weights, down_weights,
                  routing_weights, expert_indices, output,
                  num_tokens, hidden_dim, intermediate_dim,
                  num_selected_experts, activation_type, moe_type, dtype);
        return;
    }

    ensure_device_info();
    cudaStream_t stream = 0;

    // Step 1: Determine tokens per block based on shared memory constraints
    const int smem_per_token = (hidden_dim + SMEM_PAD + intermediate_dim + SMEM_PAD) * sizeof(float);
    int tokens_per_block = TOKENS_PER_BLOCK;
    while (tokens_per_block > 1 && tokens_per_block * smem_per_token > g_max_smem) {
        tokens_per_block /= 2;
    }

    const int mega_smem = tokens_per_block * smem_per_token;
    const bool has_down = (moe_type == 0);

    // If even 1 token doesn't fit, fall back to token-parallel
    if (mega_smem > g_max_smem) {
        fused_moe(input, gate_weights, up_weights, down_weights,
                  routing_weights, expert_indices, output,
                  num_tokens, hidden_dim, intermediate_dim,
                  num_selected_experts, activation_type, moe_type, dtype);
        return;
    }

    // Step 2: Counters are expected to be zeroed by the caller.

    // Step 3: Count tokens per expert
    int total_assignments = num_tokens * num_selected_experts;
    int count_blocks = (total_assignments + 255) / 256;
    count_tokens_per_expert_kernel<<<count_blocks, 256, 0, stream>>>(
        expert_indices, expert_counts, total_assignments);

    // Step 4: Compute both token offsets and block offsets (exclusive scan) - on GPU
    exclusive_scan_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, block_offsets, num_experts, tokens_per_block);

    // Step 5: Build token lists
    build_token_lists_kernel<<<count_blocks, 256, 0, stream>>>(
        expert_indices, expert_offsets, token_ids, select_ids,
        counters, num_tokens, num_selected_experts);

    // Step 6: Output is expected to be zeroed by the caller.

    // Step 7: Launch mega kernel
    // Upper bound for total_blocks: each expert contributes ceil(count/TPB) blocks
    // This is at most total_assignments/TPB + num_experts
    const int max_blocks = (total_assignments + tokens_per_block - 1) / tokens_per_block + num_experts;

    #define LAUNCH_MEGA_KERNEL_TPB(T, TPB) \
        cudaFuncSetAttribute(has_down ? moe_mega_kernel<T, true, TPB> : moe_mega_kernel<T, false, TPB>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, mega_smem); \
        if (has_down) { \
            moe_mega_kernel<T, true, TPB><<<max_blocks, BLOCK_SIZE, mega_smem, stream>>>( \
                (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                routing_weights, expert_offsets, block_offsets, token_ids, select_ids, (T*)output, \
                num_experts, 0, hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
        } else { \
            moe_mega_kernel<T, false, TPB><<<max_blocks, BLOCK_SIZE, mega_smem, stream>>>( \
                (const T*)input, (const T*)gate_weights, (const T*)up_weights, (const T*)down_weights, \
                routing_weights, expert_offsets, block_offsets, token_ids, select_ids, (T*)output, \
                num_experts, 0, hidden_dim, intermediate_dim, num_selected_experts, activation_type); \
        }

    #define LAUNCH_MEGA_KERNEL(T) \
        if (tokens_per_block >= 4) { LAUNCH_MEGA_KERNEL_TPB(T, 4); } \
        else if (tokens_per_block >= 2) { LAUNCH_MEGA_KERNEL_TPB(T, 2); } \
        else { LAUNCH_MEGA_KERNEL_TPB(T, 1); }

    if (dtype == 0) { LAUNCH_MEGA_KERNEL(half); }
    else if (dtype == 1) { LAUNCH_MEGA_KERNEL(__nv_bfloat16); }
    else { LAUNCH_MEGA_KERNEL(float); }

    #undef LAUNCH_MEGA_KERNEL
}

} // extern "C"
