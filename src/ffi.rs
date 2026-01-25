use core::ffi::{c_int, c_void};

unsafe extern "C" {
    pub(crate) fn topk_softmax(
        gating_output: *const c_void,
        topk_weight: *mut c_void,
        topk_indices: *const c_void,
        token_expert_indices: *const c_void,

        num_experts: c_int,
        num_tokens: c_int,
        topk: c_int,
    );

    pub(crate) fn fused_moe(
        input: *const c_void,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        down_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        num_selected_experts: i32,
        activation_type: i32,
        moe_type: u32,
        dtype: u32,
    );

    #[allow(dead_code)]
    pub(crate) fn optimized_fused_moe(
        input: *const c_void,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        down_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        num_experts: i32,
        activation_type: i32,
        dtype: u32,
    );
}
