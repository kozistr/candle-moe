#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// ============================================================================
// Architecture Detection
// ============================================================================
#if defined(__CUDA_ARCH__)
    #define CUDA_ARCH __CUDA_ARCH__
#else
    #define CUDA_ARCH 750  // Default to SM75 for host code
#endif

#define IS_SM80_OR_HIGHER (CUDA_ARCH >= 800)
#define IS_SM75_OR_HIGHER (CUDA_ARCH >= 750)

// ============================================================================
// Constants
// ============================================================================
#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK 32

// WMMA tile sizes (for tensor cores)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================================
// Kernel Configuration - GEMV (decode, seq_len <= 8)
// For very small batches, use warp-level reductions
// ============================================================================
namespace gemv {
    constexpr int BLOCK_SIZE = 256;      // 8 warps
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int K_UNROLL = 8;          // Unroll factor for K dimension
    constexpr int VECTOR_SIZE = 8;       // Load 8 halfs at once (float4)
}

// ============================================================================
// Kernel Configuration - Small GEMM (seq_len 8-64)
// Good occupancy, moderate tile size
// ============================================================================
namespace gemm_small {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 32;
    constexpr int THREADS = 128;         // 4 warps

    // Warp tiling: 2x2 warps, each warp handles 16x32 output
    constexpr int WARPS_M = 2;
    constexpr int WARPS_N = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARPS_M;  // 16
    constexpr int WARP_TILE_N = BLOCK_N / WARPS_N;  // 32

    // Shared memory sizes
    constexpr int SMEM_A = BLOCK_M * BLOCK_K;        // Input tile
    constexpr int SMEM_B = BLOCK_K * BLOCK_N;        // Weight tile
    constexpr int SMEM_C = BLOCK_M * BLOCK_N;        // Output tile (for store)
}

// ============================================================================
// Kernel Configuration - Medium GEMM (seq_len 64-256)
// Balance between occupancy and efficiency
// ============================================================================
namespace gemm_medium {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 32;
    constexpr int THREADS = 128;         // 4 warps

    // Warp tiling: 2x2 warps, each warp handles 32x32 output (2x2 WMMA tiles)
    constexpr int WARPS_M = 2;
    constexpr int WARPS_N = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARPS_M;  // 32
    constexpr int WARP_TILE_N = BLOCK_N / WARPS_N;  // 32

    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_K * BLOCK_N;
    constexpr int SMEM_C = BLOCK_M * BLOCK_N;
}

// ============================================================================
// Kernel Configuration - Large GEMM (seq_len > 256)
// Maximum throughput with large tiles
// ============================================================================
namespace gemm_large {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 32;
    constexpr int THREADS = 256;         // 8 warps

    // Warp tiling: 4x2 warps for better M coverage
    constexpr int WARPS_M = 4;
    constexpr int WARPS_N = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARPS_M;  // 32
    constexpr int WARP_TILE_N = BLOCK_N / WARPS_N;  // 64

    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_K * BLOCK_N;
    constexpr int SMEM_C = BLOCK_M * BLOCK_N;
}

// ============================================================================
// Thresholds for kernel selection
// ============================================================================
namespace thresholds {
    constexpr int GEMV_MAX_TOKENS = 0;           // Disabled - GEMM is faster due to better memory access
    constexpr int SMALL_GEMM_MAX_TOKENS = 64;    // Use small GEMM for <= 64 tokens
    constexpr int MEDIUM_GEMM_MAX_TOKENS = 256;  // Use medium GEMM for <= 256 tokens
    // Large GEMM for > 256 tokens
}

// ============================================================================
// Activation Functions
// ============================================================================
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float gelu_f32(float x) {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x3)));
}

__device__ __forceinline__ float relu_f32(float x) {
    return fmaxf(0.0f, x);
}

// Activation dispatch
__device__ __forceinline__ float apply_activation(float x, int act_type) {
    switch (act_type) {
        case 0: return silu_f32(x);
        case 1: return gelu_f32(x);
        case 2: return relu_f32(x);
        default: return silu_f32(x);
    }
}

// Half precision activations (compute in FP32, return half)
__device__ __forceinline__ half silu_half(half x) {
    float fx = __half2float(x);
    return __float2half(silu_f32(fx));
}

__device__ __forceinline__ half gelu_half(half x) {
    float fx = __half2float(x);
    return __float2half(gelu_f32(fx));
}

__device__ __forceinline__ half relu_half(half x) {
    float fx = __half2float(x);
    return __float2half(relu_f32(fx));
}

__device__ __forceinline__ half apply_activation_half(half x, int act_type) {
    float fx = __half2float(x);
    float result = apply_activation(fx, act_type);
    return __float2half(result);
}

// BF16 precision activations (compute in FP32, return bf16)
__device__ __forceinline__ __nv_bfloat16 silu_bf16(__nv_bfloat16 x) {
    float fx = __bfloat162float(x);
    return __float2bfloat16(silu_f32(fx));
}

__device__ __forceinline__ __nv_bfloat16 gelu_bf16(__nv_bfloat16 x) {
    float fx = __bfloat162float(x);
    return __float2bfloat16(gelu_f32(fx));
}

__device__ __forceinline__ __nv_bfloat16 relu_bf16(__nv_bfloat16 x) {
    float fx = __bfloat162float(x);
    return __float2bfloat16(relu_f32(fx));
}

__device__ __forceinline__ __nv_bfloat16 apply_activation_bf16(__nv_bfloat16 x, int act_type) {
    float fx = __bfloat162float(x);
    float result = apply_activation(fx, act_type);
    return __float2bfloat16(result);
}

// ============================================================================
// Vectorized Load/Store Helpers
// ============================================================================

// Load 8 half values (128 bits) as float4
__device__ __forceinline__ float4 load_float4(const half* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(half* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

// Load 4 half values (64 bits) as float2
__device__ __forceinline__ float2 load_float2(const half* ptr) {
    return *reinterpret_cast<const float2*>(ptr);
}

__device__ __forceinline__ void store_float2(half* ptr, float2 val) {
    *reinterpret_cast<float2*>(ptr) = val;
}

// Load 2 half values (32 bits) as half2
__device__ __forceinline__ half2 load_half2(const half* ptr) {
    return *reinterpret_cast<const half2*>(ptr);
}

__device__ __forceinline__ void store_half2(half* ptr, half2 val) {
    *reinterpret_cast<half2*>(ptr) = val;
}

// ============================================================================
// BF16 Vectorized Load/Store Helpers
// ============================================================================

// Load 8 bf16 values (128 bits) as float4
__device__ __forceinline__ float4 load_float4_bf16(const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4_bf16(__nv_bfloat16* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

// Load 4 bf16 values (64 bits) as float2
__device__ __forceinline__ float2 load_float2_bf16(const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const float2*>(ptr);
}

__device__ __forceinline__ void store_float2_bf16(__nv_bfloat16* ptr, float2 val) {
    *reinterpret_cast<float2*>(ptr) = val;
}

// Load 2 bf16 values (32 bits) as nv_bfloat162
__device__ __forceinline__ __nv_bfloat162 load_bf162(const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ __forceinline__ void store_bf162(__nv_bfloat16* ptr, __nv_bfloat162 val) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = val;
}

// ============================================================================
// Warp-level Reduction Utilities
// ============================================================================

// Warp reduce sum using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp reduce max using shuffle
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block reduce sum (assumes THREADS threads, multiple of WARP_SIZE)
template<int THREADS>
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[THREADS / WARP_SIZE];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        val = (lane < THREADS / WARP_SIZE) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;
}

// ============================================================================
// Atomic Operations for Output Accumulation
// ============================================================================

// Atomic add for half precision (native on SM70+)
__device__ __forceinline__ void atomic_add_half(half* addr, half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, val);
#else
    // Fallback for older architectures (should not be used for SM75+)
    // Uses CAS on aligned 32-bit word containing the target half
    unsigned int* addr_as_uint = (unsigned int*)((char*)addr - ((size_t)addr & 2));
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned int new_val = assumed;
        half* as_half = (half*)&new_val;
        if ((size_t)addr & 2) {
            as_half[1] = __hadd(as_half[1], val);
        } else {
            as_half[0] = __hadd(as_half[0], val);
        }
        old = atomicCAS(addr_as_uint, assumed, new_val);
    } while (assumed != old);
#endif
}

// Atomic add for bf16 (native on SM80+)
__device__ __forceinline__ void atomic_add_bf16(__nv_bfloat16* addr, __nv_bfloat16 val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, val);
#else
    // Fallback for older architectures using CAS on aligned 32-bit word
    unsigned int* addr_as_uint = (unsigned int*)((char*)addr - ((size_t)addr & 2));
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned int new_val = assumed;
        __nv_bfloat16* as_bf16 = (__nv_bfloat16*)&new_val;
        if ((size_t)addr & 2) {
            float f = __bfloat162float(as_bf16[1]) + __bfloat162float(val);
            as_bf16[1] = __float2bfloat16(f);
        } else {
            float f = __bfloat162float(as_bf16[0]) + __bfloat162float(val);
            as_bf16[0] = __float2bfloat16(f);
        }
        old = atomicCAS(addr_as_uint, assumed, new_val);
    } while (assumed != old);
#endif
}

// ============================================================================
// Memory Access Helpers
// ============================================================================

// Calculate shared memory bank-conflict-free index
// Adding stride to avoid bank conflicts (32 banks, 4 bytes each)
__device__ __forceinline__ int smem_index_no_conflict(int row, int col, int row_stride) {
    // Add padding to avoid bank conflicts
    return row * (row_stride + 1) + col;
}

// Calculate expert weight offset
// Weights layout: [num_experts, dim1, dim2]
__device__ __forceinline__ size_t expert_weight_offset(
    int expert_id,
    int dim1,
    int dim2
) {
    return (size_t)expert_id * dim1 * dim2;
}

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// Kernel Launch Configuration Helpers
// ============================================================================

struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t smem_size;
};

// Calculate grid dimensions for expert-parallel execution
inline KernelConfig get_expert_parallel_config(
    int num_tokens_per_expert,
    int output_dim,
    int block_m,
    int block_n,
    int threads,
    size_t smem_per_block,
    int num_experts
) {
    KernelConfig config;
    int m_tiles = (num_tokens_per_expert + block_m - 1) / block_m;
    int n_tiles = (output_dim + block_n - 1) / block_n;

    config.grid = dim3(n_tiles, m_tiles, num_experts);
    config.block = dim3(threads);
    config.smem_size = smem_per_block;

    return config;
}

// ============================================================================
// Data Type Utilities
// ============================================================================

// Convert dtype enum to size in bytes
inline size_t dtype_size(uint32_t dtype) {
    switch (dtype) {
        case 0: return sizeof(half);        // FP16
        case 1: return sizeof(__nv_bfloat16); // BF16
        case 2: return sizeof(float);       // FP32
        default: return sizeof(half);
    }
}
