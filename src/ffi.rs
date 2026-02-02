use core::ffi::{c_int, c_void};

unsafe extern "C" {
    /// Top-K softmax kernel
    ///
    /// Arguments:
    /// - gating_output: [num_tokens, num_experts] - Router logits
    /// - topk_weight: [num_tokens, topk] - Output weights (modified in-place)
    /// - topk_indices: [num_tokens, topk] - Selected expert indices
    /// - token_expert_indices: [num_tokens, topk] - Token-expert mapping
    /// - num_experts: Number of experts (must be power of 2, <= 256)
    /// - num_tokens: Number of input tokens
    /// - topk: Number of experts to select per token
    /// - stream: CUDA stream handle
    pub(crate) fn topk_softmax(
        gating_output: *const c_void,
        topk_weight: *mut c_void,
        topk_indices: *const c_void,
        token_expert_indices: *const c_void,
        num_experts: c_int,
        num_tokens: c_int,
        topk: c_int,
        stream: *mut c_void,
    );

    /// Fused Mixture of Experts kernel
    ///
    /// Arguments:
    /// - input: [num_tokens, hidden_dim] - Input activations
    /// - gate_weights: [num_experts, hidden_dim, intermediate_dim] - Gate projection weights
    /// - up_weights: [num_experts, hidden_dim, intermediate_dim] - Up projection weights
    /// - down_weights: [num_experts, intermediate_dim, hidden_dim] - Down projection weights (Qwen3)
    /// - routing_weights: [num_tokens, num_selected_experts] - Routing weights (f32)
    /// - expert_indices: [num_tokens, num_selected_experts] - Selected expert indices (u32)
    /// - output: [num_tokens, hidden_dim] - Output activations (must be zero-initialized)
    /// - num_tokens: Number of input tokens
    /// - hidden_dim: Hidden dimension size
    /// - intermediate_dim: Intermediate (FFN) dimension size
    /// - num_experts: Total number of experts
    /// - num_selected_experts: Number of experts selected per token (top-k)
    /// - activation_type: 0=SiLU, 1=GELU, 2=ReLU
    /// - moe_type: 0=Qwen3 (gate-up-down), 1=Nomic (gate-up only)
    /// - dtype: 0=FP16, 1=BF16, 2=FP32
    /// - stream: CUDA stream handle
    pub(crate) fn fused_moe(
        input: *const c_void,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        down_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: c_int,
        hidden_dim: c_int,
        intermediate_dim: c_int,
        num_experts: c_int,
        num_selected_experts: c_int,
        activation_type: c_int,
        moe_type: u32,
        dtype: u32,
        stream: *mut c_void,
    );
}
