use anyhow::Result;
use candle::{D, DType, Device, IndexOp, Tensor};
use candle_transformers::models::deepseek2::{BincountOp, NonZeroOp};

#[allow(dead_code)]
fn to_vec2_round(t: Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|row| {
            row.iter()
                .map(|val| (val * b).round() / b)
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();
    Ok(t)
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

fn forward_moe_mlp(x: &Tensor, w1: &Tensor, w2: &Tensor, expert_idx: usize) -> Result<Tensor> {
    let expert_w1 = w1.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;
    let expert_w2 = w2.narrow(0, expert_idx, 1)?.squeeze(0)?;

    let x = x.broadcast_matmul(&expert_w1)?;
    let x = x.silu()?;

    Ok(x.broadcast_matmul(&expert_w2)?)
}

fn forward_moe_expert(
    hidden_states: &Tensor,
    gate: &Tensor,
    up: &Tensor,
    scores: &Tensor,
    indices: &Tensor,
    hidden_size: usize,
    num_experts: usize,
) -> Result<Tensor> {
    let hidden_states = hidden_states.reshape(((), hidden_size))?;

    let mut out = Tensor::zeros_like(&hidden_states)?;

    let counts = indices.flatten_all()?.bincount(num_experts as u32)?;

    for (expert_idx, &count) in counts.iter().enumerate().take(num_experts) {
        if count == 0u32 {
            continue;
        }

        let idx_top = indices.eq(expert_idx as f64)?.nonzero()?.t()?;
        let idx = &idx_top.i(0)?.contiguous()?;
        let top = &idx_top.i(1)?.contiguous()?;

        let expert_out =
            forward_moe_mlp(&hidden_states.index_select(idx, 0)?, gate, up, expert_idx)?
                .broadcast_mul(
                    &scores
                        .index_select(idx, 0)?
                        .gather(&top.unsqueeze(1)?, 1)?
                        .squeeze(1)?
                        .unsqueeze(D::Minus1)?
                        .to_dtype(hidden_states.dtype())?,
                )?;

        out = out.index_add(idx, &expert_out, 0)?;
    }

    Ok(out)
}

#[test]
fn fused_moe() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let dtype = DType::F32;
    let n_embed = 16;
    let n_inner = n_embed * 4;
    let seq_len = 16;
    let num_experts = 8;
    let top_k = 2;

    let hidden_states = Tensor::randn(0.0, 1.0, (seq_len, n_embed), &device)?.to_dtype(dtype)?;
    let weights = Tensor::randn(0.0, 1.0, (seq_len, num_experts), &device)?.to_dtype(DType::F32)?;

    let (scores, indices) = forward_moe_router(&weights, seq_len, top_k, &device)?;

    let gate_weights =
        Tensor::randn(0.0, 1.0, (num_experts, n_embed, n_inner), &device)?.to_dtype(dtype)?;
    let up_weights =
        Tensor::randn(0.0, 1.0, (num_experts, n_embed, n_inner), &device)?.to_dtype(dtype)?;

    let fused_moe = candle_moe::FusedMoE {
        num_experts: gate_weights.dim(0)?,
        num_selected_experts: top_k,
        activation: candle_moe::Activation::Silu,
    };

    let naive_moe_output = forward_moe_expert(
        &hidden_states,
        &gate_weights.permute((0, 2, 1))?,
        &up_weights.permute((0, 2, 1))?,
        &scores,
        &indices,
        n_embed,
        num_experts,
    )?;

    let fused_moe_output = fused_moe.forward(
        &hidden_states,
        &gate_weights,
        &up_weights,
        None,
        &scores,
        &indices,
        1_u32,
    )?;

    assert_eq!(
        to_vec2_round(naive_moe_output, 3)?,
        to_vec2_round(fused_moe_output, 3)?,
    );

    Ok(())
}
