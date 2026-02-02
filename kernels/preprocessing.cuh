#pragma once

#include "common.cuh"

__global__ void moe_preprocess_kernel(
    const uint32_t* __restrict__ expert_indices,  // [num_tokens, top_k]
    const float* __restrict__ routing_weights,    // [num_tokens, top_k]
    int* __restrict__ expert_offsets,             // [num_experts + 1]
    int* __restrict__ sorted_token_ids,           // [num_tokens * top_k]
    float* __restrict__ sorted_weights,           // [num_tokens * top_k]
    int num_tokens,
    int num_experts,
    int top_k
) {
    extern __shared__ char smem[];
    int* s_counts = (int*)smem;
    int* s_offsets = s_counts + num_experts;

    const int tid = threadIdx.x;
    const int total = num_tokens * top_k;

    // Initialize counts to zero
    for (int i = tid; i < num_experts; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();

    // Count tokens per expert using atomic add
    for (int i = tid; i < total; i += blockDim.x) {
        int expert_id = expert_indices[i];
        // Bounds check to prevent OOB writes from invalid expert indices
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&s_counts[expert_id], 1);
        }
    }
    __syncthreads();

    // Compute prefix sum for offsets and max tokens (single thread for small expert count)
    if (tid == 0) {
        int sum = 0;
        int max_count = 0;
        for (int i = 0; i < num_experts; i++) {
            expert_offsets[i] = sum;
            s_offsets[i] = sum;
            max_count = max(max_count, s_counts[i]);
            sum += s_counts[i];
        }
        expert_offsets[num_experts] = sum;
        // Store max in the slot after the last offset
        expert_offsets[num_experts + 1] = max_count;
    }
    __syncthreads();

    // Sort tokens by expert assignment
    for (int i = tid; i < total; i += blockDim.x) {
        int token_id = i / top_k;
        int expert_id = expert_indices[i];
        // Bounds check to prevent OOB writes from invalid expert indices
        if (expert_id < 0 || expert_id >= num_experts) continue;

        float weight = routing_weights[i];

        // Atomic increment to get position within expert's token list
        int pos = atomicAdd(&s_offsets[expert_id], 1);

        sorted_token_ids[pos] = token_id;
        sorted_weights[pos] = weight;
    }
}

__global__ void moe_preprocess_8experts_kernel(
    const uint32_t* __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    int* __restrict__ expert_offsets,
    int* __restrict__ sorted_token_ids,
    float* __restrict__ sorted_weights,
    int num_tokens,
    int top_k
) {
    constexpr int NUM_EXPERTS = 8;

    extern __shared__ char smem[];
    int* s_counts = (int*)smem;
    int* s_offsets = s_counts + NUM_EXPERTS;

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int total = num_tokens * top_k;

    // Initialize counts
    if (tid < NUM_EXPERTS) {
        s_counts[tid] = 0;
    }
    __syncthreads();

    // Count tokens per expert - each warp processes a chunk
    int local_counts[NUM_EXPERTS] = {0};

    for (int i = tid; i < total; i += blockDim.x) {
        int expert_id = expert_indices[i];
        // Bounds check to prevent OOB writes from invalid expert indices
        if (expert_id >= 0 && expert_id < NUM_EXPERTS) {
            local_counts[expert_id]++;
        }
    }

    // Reduce local counts within warp, then across warps
    #pragma unroll
    for (int e = 0; e < NUM_EXPERTS; e++) {
        // Warp reduce
        int count = local_counts[e];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            count += __shfl_xor_sync(0xffffffff, count, offset);
        }

        // First thread of each warp adds to shared memory
        if (lane == 0) {
            atomicAdd(&s_counts[e], count);
        }
    }
    __syncthreads();

    // Compute offsets and max tokens (single thread)
    if (tid == 0) {
        int sum = 0;
        int max_count = 0;
        for (int i = 0; i < NUM_EXPERTS; i++) {
            expert_offsets[i] = sum;
            s_offsets[i] = sum;
            max_count = max(max_count, s_counts[i]);
            sum += s_counts[i];
        }
        expert_offsets[NUM_EXPERTS] = sum;
        // Store max in the slot after the last offset (expert_offsets[NUM_EXPERTS + 1])
        expert_offsets[NUM_EXPERTS + 1] = max_count;
    }
    __syncthreads();

    // Sort tokens by expert
    for (int i = tid; i < total; i += blockDim.x) {
        int token_id = i / top_k;
        int expert_id = expert_indices[i];
        // Bounds check to prevent OOB writes from invalid expert indices
        if (expert_id < 0 || expert_id >= NUM_EXPERTS) continue;

        float weight = routing_weights[i];

        int pos = atomicAdd(&s_offsets[expert_id], 1);
        sorted_token_ids[pos] = token_id;
        sorted_weights[pos] = weight;
    }
}

__global__ void moe_count_tokens_kernel(
    const uint32_t* __restrict__ expert_indices,
    int* __restrict__ expert_counts,  // [num_experts] - atomically updated
    int num_tokens,
    int num_experts,
    int top_k
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * top_k;

    // Use shared memory for local counting within block
    extern __shared__ int s_local_counts[];

    const int lane = threadIdx.x;

    // Initialize shared memory
    for (int i = lane; i < num_experts; i += blockDim.x) {
        s_local_counts[i] = 0;
    }
    __syncthreads();

    // Count within block
    for (int i = tid; i < total; i += gridDim.x * blockDim.x) {
        int expert_id = expert_indices[i];
        // Bounds check to prevent OOB writes from invalid expert indices
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&s_local_counts[expert_id], 1);
        }
    }
    __syncthreads();

    // Add block counts to global counts
    for (int i = lane; i < num_experts; i += blockDim.x) {
        if (s_local_counts[i] > 0) {
            atomicAdd(&expert_counts[i], s_local_counts[i]);
        }
    }
}

__global__ void moe_compute_offsets_kernel(
    const int* __restrict__ expert_counts,
    int* __restrict__ expert_offsets,
    int num_experts
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        int max_count = 0;
        for (int i = 0; i < num_experts; i++) {
            expert_offsets[i] = sum;
            max_count = max(max_count, expert_counts[i]);
            sum += expert_counts[i];
        }
        expert_offsets[num_experts] = sum;
        expert_offsets[num_experts + 1] = max_count;
    }
}

__global__ void moe_sort_tokens_kernel(
    const uint32_t* __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    int* __restrict__ expert_write_offsets,  // [num_experts] - atomically updated
    int* __restrict__ sorted_token_ids,
    float* __restrict__ sorted_weights,
    int num_tokens,
    int num_experts,
    int top_k
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * top_k;

    for (int i = tid; i < total; i += gridDim.x * blockDim.x) {
        int token_id = i / top_k;
        int expert_id = expert_indices[i];
        // Bounds check to prevent OOB writes from invalid expert indices
        if (expert_id < 0 || expert_id >= num_experts) continue;

        float weight = routing_weights[i];

        int pos = atomicAdd(&expert_write_offsets[expert_id], 1);
        sorted_token_ids[pos] = token_id;
        sorted_weights[pos] = weight;
    }
}

__global__ void compute_max_tokens_kernel(
    const int* __restrict__ expert_offsets,
    int* __restrict__ max_tokens,
    int num_experts
) {
    __shared__ int s_max;

    if (threadIdx.x == 0) {
        s_max = 0;
    }
    __syncthreads();

    // Each thread checks a subset of experts
    int local_max = 0;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        int count = expert_offsets[i + 1] - expert_offsets[i];
        local_max = max(local_max, count);
    }

    // Reduce within block
    atomicMax(&s_max, local_max);
    __syncthreads();

    if (threadIdx.x == 0) {
        *max_tokens = s_max;
    }
}

inline void launch_compute_max_tokens(
    const int* expert_offsets,
    int* max_tokens,
    int num_experts,
    cudaStream_t stream
) {
    int threads = min(256, num_experts);
    compute_max_tokens_kernel<<<1, threads, 0, stream>>>(
        expert_offsets, max_tokens, num_experts
    );
}

struct PreprocessResult {
    int* expert_offsets;      // [num_experts + 1]
    int* sorted_token_ids;    // [num_tokens * top_k]
    float* sorted_weights;    // [num_tokens * top_k]
};

inline void launch_preprocessing(
    const uint32_t* expert_indices,
    const float* routing_weights,
    int* expert_offsets,
    int* sorted_token_ids,
    float* sorted_weights,
    int num_tokens,
    int num_experts,
    int top_k,
    cudaStream_t stream
) {
    const int total = num_tokens * top_k;

    // For typical MoE sizes, single-block preprocessing is sufficient
    // and avoids synchronization overhead
    if (total <= 65536) {
        // Single block approach
        int threads = min(512, (total + 31) / 32 * 32);
        threads = max(threads, 64);  // Minimum 64 threads
        size_t smem_size = 2 * num_experts * sizeof(int);

        if (num_experts == 8) {
            moe_preprocess_8experts_kernel<<<1, threads, smem_size, stream>>>(
                expert_indices,
                routing_weights,
                expert_offsets,
                sorted_token_ids,
                sorted_weights,
                num_tokens,
                top_k
            );
        } else {
            moe_preprocess_kernel<<<1, threads, smem_size, stream>>>(
                expert_indices,
                routing_weights,
                expert_offsets,
                sorted_token_ids,
                sorted_weights,
                num_tokens,
                num_experts,
                top_k
            );
        }
    } else {
        // Multi-block approach for very large token counts
        // This requires workspace for intermediate counts

        // Allocate temporary storage for write offsets
        int* d_write_offsets;
        cudaMallocAsync(&d_write_offsets, num_experts * sizeof(int), stream);
        cudaMemsetAsync(d_write_offsets, 0, num_experts * sizeof(int), stream);

        // Clear expert offsets first (used as counts)
        cudaMemsetAsync(expert_offsets, 0, (num_experts + 1) * sizeof(int), stream);

        // Count tokens per expert
        int count_threads = 256;
        int count_blocks = min((total + count_threads - 1) / count_threads, 128);
        size_t count_smem = num_experts * sizeof(int);

        moe_count_tokens_kernel<<<count_blocks, count_threads, count_smem, stream>>>(
            expert_indices,
            expert_offsets,  // Reuse as counts
            num_tokens,
            num_experts,
            top_k
        );

        // Compute offsets (inclusive scan)
        moe_compute_offsets_kernel<<<1, 1, 0, stream>>>(
            expert_offsets,
            expert_offsets,
            num_experts
        );

        // Copy offsets to write offsets for sorting
        cudaMemcpyAsync(d_write_offsets, expert_offsets,
                       num_experts * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        // Sort tokens
        int sort_threads = 256;
        int sort_blocks = min((total + sort_threads - 1) / sort_threads, 256);

        moe_sort_tokens_kernel<<<sort_blocks, sort_threads, 0, stream>>>(
            expert_indices,
            routing_weights,
            d_write_offsets,
            sorted_token_ids,
            sorted_weights,
            num_tokens,
            num_experts,
            top_k
        );

        cudaFreeAsync(d_write_offsets, stream);
    }
}
