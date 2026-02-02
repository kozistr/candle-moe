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

fn profile_qwen3_with_config(
    device: &Device,
    seq_len: usize,
    hidden_size: usize,
    top_k: usize,
    dtype: DType,
    iters: usize,
) -> Result<()> {
    let intermediate_size = hidden_size * 4;
    let num_experts: usize = 8;

    println!(
        "\n=== Qwen3 MoE (with down projection) ===\n\
         seq_len={}, hidden_size={}, intermediate_size={}, num_experts={}, top_k={}, dtype={:?}",
        seq_len, hidden_size, intermediate_size, num_experts, top_k, dtype
    );

    let hidden_states =
        Tensor::randn(0.0f32, 1.0, (seq_len, hidden_size), device)?.to_dtype(dtype)?;
    let weights =
        Tensor::randn(0.0f32, 1.0, (seq_len, num_experts), device)?.to_dtype(DType::F32)?;
    let gate_weights = Tensor::randn(
        0.0,
        1.0,
        (num_experts, hidden_size, intermediate_size),
        device,
    )?
    .to_dtype(dtype)?;
    let up_weights = Tensor::randn(
        0.0,
        1.0,
        (num_experts, hidden_size, intermediate_size),
        device,
    )?
    .to_dtype(dtype)?;
    let down_weights = Tensor::randn(
        0.0,
        1.0,
        (num_experts, intermediate_size, hidden_size),
        device,
    )?
    .to_dtype(dtype)?;
    let (scores, indices) = forward_moe_router(&weights, seq_len, top_k, device)?;

    let fused_moe = candle_moe::FusedMoE {
        num_experts: gate_weights.dim(0)?,
        num_selected_experts: top_k,
        activation: candle_moe::Activation::Silu,
    };

    // Warmup
    for _ in 0..11 {
        let _ = fused_moe.forward(
            &hidden_states,
            &gate_weights,
            &up_weights,
            Some(&down_weights),
            &scores,
            &indices,
            0_u32,
        )?;
    }
    device.synchronize()?;

    let dtype_str = match dtype {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "f32",
    };
    let nvtx_label = format!(
        "qwen3_{}x{}_topk{}_{}\0",
        seq_len, hidden_size, top_k, dtype_str
    );

    for _ in 0..iters {
        unsafe {
            nvtxRangePushA(nvtx_label.as_ptr() as *const c_char);
        }

        let _ = fused_moe.forward(
            &hidden_states,
            &gate_weights,
            &up_weights,
            Some(&down_weights),
            &scores,
            &indices,
            0_u32,
        )?;

        unsafe {
            nvtxRangePop();
        }
    }
    device.synchronize()?;

    println!("Completed {} iterations", iters);
    Ok(())
}

fn profile_qwen3(device: &Device, seq_len: usize, hidden_size: usize, iters: usize) -> Result<()> {
    profile_qwen3_with_config(device, seq_len, hidden_size, 2, DType::F16, iters)
}

fn profile_qwen3_topk1(
    device: &Device,
    seq_len: usize,
    hidden_size: usize,
    iters: usize,
) -> Result<()> {
    profile_qwen3_with_config(device, seq_len, hidden_size, 1, DType::F16, iters)
}

fn profile_qwen3_bf16(
    device: &Device,
    seq_len: usize,
    hidden_size: usize,
    iters: usize,
) -> Result<()> {
    profile_qwen3_with_config(device, seq_len, hidden_size, 2, DType::BF16, iters)
}

fn profile_nomic_with_config(
    device: &Device,
    seq_len: usize,
    hidden_size: usize,
    top_k: usize,
    dtype: DType,
    iters: usize,
) -> Result<()> {
    let intermediate_size = hidden_size * 4;
    let num_experts = 8;

    println!(
        "\n=== Nomic MoE (no down projection) ===\n\
         seq_len={}, hidden_size={}, intermediate_size={}, num_experts={}, top_k={}, dtype={:?}",
        seq_len, hidden_size, intermediate_size, num_experts, top_k, dtype
    );

    // --- Setup tensors ---
    let hidden_states =
        Tensor::randn(0.0f32, 1.0, (seq_len, hidden_size), device)?.to_dtype(dtype)?;
    let weights =
        Tensor::randn(0.0f32, 1.0, (seq_len, num_experts), device)?.to_dtype(DType::F32)?;
    // Nomic: gate and up weights have same shape
    let gate_weights = Tensor::randn(
        0.0,
        1.0,
        (num_experts, hidden_size, intermediate_size),
        device,
    )?
    .to_dtype(dtype)?;
    let up_weights = Tensor::randn(
        0.0,
        1.0,
        (num_experts, hidden_size, intermediate_size),
        device,
    )?
    .to_dtype(dtype)?;
    let (scores, indices) = forward_moe_router(&weights, seq_len, top_k, device)?;

    let fused_moe = candle_moe::FusedMoE {
        num_experts: gate_weights.dim(0)?,
        num_selected_experts: top_k,
        activation: candle_moe::Activation::Silu,
    };

    // Warmup
    for _ in 0..11 {
        let _ = fused_moe.forward(
            &hidden_states,
            &gate_weights,
            &up_weights,
            None, // No down weights for Nomic
            &scores,
            &indices,
            1_u32, // Nomic MoE
        )?;
    }
    device.synchronize()?;

    // --- Profiling region ---
    let dtype_str = match dtype {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "f32",
    };
    let nvtx_label = format!(
        "nomic_{}x{}_topk{}_{}\0",
        seq_len, hidden_size, top_k, dtype_str
    );
    for _ in 0..iters {
        unsafe {
            nvtxRangePushA(nvtx_label.as_ptr() as *const c_char);
        }

        let _ = fused_moe.forward(
            &hidden_states,
            &gate_weights,
            &up_weights,
            None, // No down weights for Nomic
            &scores,
            &indices,
            1_u32, // Nomic MoE
        )?;

        unsafe {
            nvtxRangePop();
        }
    }
    device.synchronize()?;

    println!("Completed {} iterations", iters);
    Ok(())
}

fn profile_nomic(device: &Device, seq_len: usize, hidden_size: usize, iters: usize) -> Result<()> {
    profile_nomic_with_config(device, seq_len, hidden_size, 2, DType::F16, iters)
}

fn profile_nomic_topk1(
    device: &Device,
    seq_len: usize,
    hidden_size: usize,
    iters: usize,
) -> Result<()> {
    profile_nomic_with_config(device, seq_len, hidden_size, 1, DType::F16, iters)
}

fn profile_nomic_bf16(
    device: &Device,
    seq_len: usize,
    hidden_size: usize,
    iters: usize,
) -> Result<()> {
    profile_nomic_with_config(device, seq_len, hidden_size, 2, DType::BF16, iters)
}

fn main() -> Result<()> {
    std::thread::sleep(std::time::Duration::from_secs(1));

    let device = Device::new_cuda(0)?;
    println!("Using device: {:?}", device);

    let iters = 1024;

    // =====================
    // Nomic cases (top_k=2, F16)
    // =====================

    // Small batch - tests the small batch threshold fix
    profile_nomic(&device, 16, 768, iters)?;
    profile_nomic(&device, 32, 768, iters)?;

    // Long sequence - tests tensor core performance
    profile_nomic(&device, 8192, 768, iters)?;

    // Very long sequence
    profile_nomic(&device, 32768, 768, iters)?;

    // Larger Nomic models
    profile_nomic(&device, 8192, 1024, iters)?;
    profile_nomic(&device, 8192, 2048, iters)?;

    // =====================
    // Nomic top_k=1 (tests memset-free optimization)
    // =====================
    profile_nomic_topk1(&device, 8192, 768, iters)?;
    profile_nomic_topk1(&device, 32768, 768, iters)?;

    // =====================
    // Nomic BF16
    // =====================
    profile_nomic_bf16(&device, 8192, 768, iters)?;
    profile_nomic_bf16(&device, 32768, 768, iters)?;

    // =====================
    // Qwen3 cases (top_k=2, F16)
    // =====================

    // Qwen3 with 768 hidden_dim (same as Nomic for fair comparison)
    profile_qwen3(&device, 16, 768, iters)?;
    profile_qwen3(&device, 32, 768, iters)?;
    profile_qwen3(&device, 8192, 768, iters)?;
    profile_qwen3(&device, 32768, 768, iters)?;

    // Standard Qwen3 case
    profile_qwen3(&device, 8192, 1024, iters)?;
    profile_qwen3(&device, 16384, 1024, iters)?;
    profile_qwen3(&device, 16384, 4096, iters)?;

    // =====================
    // Qwen3 top_k=1 (tests memset-free optimization)
    // =====================
    profile_qwen3_topk1(&device, 8192, 768, iters)?;
    profile_qwen3_topk1(&device, 32768, 768, iters)?;

    // =====================
    // Qwen3 BF16
    // =====================
    profile_qwen3_bf16(&device, 8192, 768, iters)?;
    profile_qwen3_bf16(&device, 32768, 768, iters)?;

    // =====================
    // Qwen3-8B-Embedding scale (hidden_dim=4096)
    // =====================
    profile_qwen3_with_config(&device, 32, 4096, 2, DType::F16, iters)?;
    profile_qwen3_with_config(&device, 32768, 4096, 2, DType::F16, iters)?;
    profile_qwen3_with_config(&device, 32, 4096, 2, DType::BF16, iters)?;
    profile_qwen3_with_config(&device, 32768, 4096, 2, DType::BF16, iters)?;

    println!("\n=== Profiling complete ===");
    Ok(())
}
