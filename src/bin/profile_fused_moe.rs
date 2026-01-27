use candle_moe::candle;

use candle::{DType, Device, Result, Tensor};
use std::os::raw::c_char;

// =============================
// NVTX FFI
// =============================
#[link(name = "nvToolsExt")]
unsafe extern "C" {
    fn nvtxRangePushA(msg: *const c_char) -> i32;
    fn nvtxRangePop() -> i32;
}

fn forward_moe_router(
    weights: &Tensor,
    seq_len: usize,
    top_k: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let topk_weight = Tensor::zeros((seq_len, top_k), DType::F32, device)?;
    let topk_indices = Tensor::zeros((seq_len, top_k), DType::U32, device)?;
    let token_expert_indices = Tensor::zeros((seq_len, top_k), DType::U32, device)?;

    candle_moe::apply_topk_softmax_inplace(
        weights,
        &topk_weight,
        &topk_indices,
        &token_expert_indices,
    )?;

    Ok((topk_weight, topk_indices))
}

fn main() -> Result<()> {
    std::thread::sleep(std::time::Duration::from_secs(1));

    let dtype = DType::F16;
    let seq_len = 16384usize;
    let hidden_size = 1024usize;
    let intermidiate_size = hidden_size * 4;
    let num_experts = 8;
    let top_k = 2;
    let iters = 1000; // enough to get stable averages in Nsight

    let device = Device::new_cuda(0)?;
    println!("Using device: {:?}", device);

    // --- Setup tensors ---
    let hidden_states =
        Tensor::randn(0.0f32, 1.0, (seq_len, hidden_size), &device)?.to_dtype(dtype)?;
    let weights =
        Tensor::randn(0.0f32, 1.0, (seq_len, num_experts), &device)?.to_dtype(DType::F32)?;
    let gate_weights = Tensor::randn(
        0.0,
        1.0,
        (num_experts, hidden_size, intermidiate_size),
        &device,
    )?
    .to_dtype(dtype)?;
    let up_weights = Tensor::randn(
        0.0,
        1.0,
        (num_experts, hidden_size, intermidiate_size),
        &device,
    )?
    .to_dtype(dtype)?;
    let down_weights = Tensor::randn(
        0.0,
        1.0,
        (num_experts, intermidiate_size, hidden_size),
        &device,
    )?
    .to_dtype(dtype)?;
    let (scores, indices) = forward_moe_router(&weights, seq_len, top_k, &device)?;

    let fused_moe = candle_moe::FusedMoE {
        num_experts: gate_weights.dim(0)?,
        num_selected_experts: top_k,
        activation: candle_moe::Activation::Silu,
    };

    // Warmup
    for _ in 0..iters {
        let _ = fused_moe.forward(
            &hidden_states,
            &gate_weights,
            &up_weights,
            Some(&down_weights),
            &scores,
            &indices,
            0_u32, // Qwen3 MoE (with down projection) - uses expert kernel
        )?;
    }
    device.synchronize()?;

    println!(
        "Profiling fused MoE: shape=({}, {}), num_experts={}, dtype={:?} iters={}",
        seq_len, hidden_size, num_experts, dtype, iters
    );

    // --- Profiling region with NVTX range per call ---
    for _ in 0..iters {
        unsafe {
            nvtxRangePushA(b"fused_moe_f16\0".as_ptr() as *const c_char);
        }

        let _ = fused_moe.forward(
            &hidden_states,
            &gate_weights,
            &up_weights,
            Some(&down_weights),
            &scores,
            &indices,
            0_u32, // Qwen3 MoE (with down projection) - uses expert kernel
        )?;

        unsafe {
            nvtxRangePop();
        }
    }
    device.synchronize()?;

    Ok(())
}
