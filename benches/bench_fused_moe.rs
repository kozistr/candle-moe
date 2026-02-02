use candle::{D, DType, Device, IndexOp, Result, Tensor};
use candle_transformers::models::deepseek2::{BincountOp, NonZeroOp};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

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

// Qwen3 style: gate-up-down with separate projections
fn forward_moe_mlp_qwen3(
    x: &Tensor,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
    expert_idx: usize,
) -> Result<Tensor> {
    let expert_gate = gate.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;
    let expert_up = up.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;
    let expert_down = down.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;

    let gate_out = x.broadcast_matmul(&expert_gate)?.silu()?;
    let up_out = x.broadcast_matmul(&expert_up)?;
    let intermediate = (gate_out * up_out)?;

    Ok(intermediate.broadcast_matmul(&expert_down)?)
}

fn forward_moe_expert_qwen3(
    hidden_states: &Tensor,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
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

        let expert_out = forward_moe_mlp_qwen3(
            &hidden_states.index_select(idx, 0)?,
            gate,
            up,
            down,
            expert_idx,
        )?
        .broadcast_mul(
            &scores
                .index_select(idx, 0)
                .unwrap()
                .gather(&top.unsqueeze(1)?, 1)
                .unwrap()
                .squeeze(1)?
                .unsqueeze(D::Minus1)
                .unwrap()
                .to_dtype(hidden_states.dtype())?,
        )?;

        out = out.index_add(idx, &expert_out, 0)?;
    }

    Ok(out)
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
                        .index_select(idx, 0)
                        .unwrap()
                        .gather(&top.unsqueeze(1)?, 1)
                        .unwrap()
                        .squeeze(1)?
                        .unsqueeze(D::Minus1)
                        .unwrap()
                        .to_dtype(hidden_states.dtype())?,
                )?;

        out = out.index_add(idx, &expert_out, 0)?;
    }

    Ok(out)
}

fn setup_tensors(
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
    n_embed: usize,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, candle_moe::FusedMoE)> {
    let device = Device::new_cuda(0)?;

    let n_inner = n_embed * 4;

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

    Ok((
        hidden_states,
        gate_weights,
        up_weights,
        scores,
        indices,
        fused_moe,
    ))
}

// Setup tensors for Qwen3 style MoE (with down projection)
fn setup_tensors_qwen3(
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
    n_embed: usize,
    dtype: DType,
) -> Result<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    candle_moe::FusedMoE,
)> {
    let device = Device::new_cuda(0)?;

    let n_inner = n_embed * 4;

    let hidden_states = Tensor::randn(0.0, 1.0, (seq_len, n_embed), &device)?.to_dtype(dtype)?;
    let weights = Tensor::randn(0.0, 1.0, (seq_len, num_experts), &device)?.to_dtype(DType::F32)?;

    let (scores, indices) = forward_moe_router(&weights, seq_len, top_k, &device)?;

    let gate_weights =
        Tensor::randn(0.0, 1.0, (num_experts, n_embed, n_inner), &device)?.to_dtype(dtype)?;
    let up_weights =
        Tensor::randn(0.0, 1.0, (num_experts, n_embed, n_inner), &device)?.to_dtype(dtype)?;
    let down_weights =
        Tensor::randn(0.0, 1.0, (num_experts, n_inner, n_embed), &device)?.to_dtype(dtype)?;

    let fused_moe = candle_moe::FusedMoE {
        num_experts: gate_weights.dim(0)?,
        num_selected_experts: top_k,
        activation: candle_moe::Activation::Silu,
    };

    Ok((
        hidden_states,
        gate_weights,
        up_weights,
        down_weights,
        scores,
        indices,
        fused_moe,
    ))
}

// Setup tensors for Qwen3 style MoE (with down projection)
fn setup_tensors_qwen3(
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
    n_embed: usize,
    dtype: DType,
) -> Result<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    candle_moe::FusedMoE,
)> {
    let device = Device::new_cuda(0)?;

    let n_inner = n_embed * 4;

    let hidden_states = Tensor::randn(0.0, 1.0, (seq_len, n_embed), &device)?.to_dtype(dtype)?;
    let weights = Tensor::randn(0.0, 1.0, (seq_len, num_experts), &device)?.to_dtype(DType::F32)?;

    let (scores, indices) = forward_moe_router(&weights, seq_len, top_k, &device)?;

    let gate_weights =
        Tensor::randn(0.0, 1.0, (num_experts, n_embed, n_inner), &device)?.to_dtype(dtype)?;
    let up_weights =
        Tensor::randn(0.0, 1.0, (num_experts, n_embed, n_inner), &device)?.to_dtype(dtype)?;
    let down_weights =
        Tensor::randn(0.0, 1.0, (num_experts, n_inner, n_embed), &device)?.to_dtype(dtype)?;

    let fused_moe = candle_moe::FusedMoE {
        num_experts: gate_weights.dim(0)?,
        num_selected_experts: top_k,
        activation: candle_moe::Activation::Silu,
    };

    Ok((
        hidden_states,
        gate_weights,
        up_weights,
        down_weights,
        scores,
        indices,
        fused_moe,
    ))
}

fn run_benchmark(
    c: &mut Criterion,
    group_name: &str,
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
    n_embed: usize,
    dtype: DType,
) {
    let (hidden_states, gate_weights, up_weights, scores, indices, fused_moe) =
        match setup_tensors(seq_len, num_experts, top_k, n_embed, dtype) {
            Ok(t) => t,
            Err(e) => {
                println!(
                    "Failed to setup tensors for group {}, skipping benchmark: {:?}",
                    group_name, e
                );
                return;
            }
        };

    let mut group = c.benchmark_group(group_name);
    group.sample_size(500);
    group.warm_up_time(std::time::Duration::from_millis(1500));
    group.measurement_time(std::time::Duration::from_millis(15000));

    // native warmup
    let _ = forward_moe_expert(
        &hidden_states,
        &gate_weights.permute((0, 2, 1)).unwrap(),
        &up_weights.permute((0, 2, 1)).unwrap(),
        &scores,
        &indices,
        n_embed,
        num_experts,
    )
    .unwrap();

    group.bench_function("native", |b| {
        b.iter(|| {
            let result = black_box(
                forward_moe_expert(
                    &hidden_states,
                    &gate_weights.permute((0, 2, 1)).unwrap(),
                    &up_weights.permute((0, 2, 1)).unwrap(),
                    &scores,
                    &indices,
                    n_embed,
                    num_experts,
                )
                .unwrap(),
            );
            result.device().synchronize().unwrap();
        })
    });

    // custom warmup
    let _ = fused_moe
        .forward(
            &hidden_states,
            &gate_weights,
            &up_weights,
            None,
            &scores,
            &indices,
            1_u32,
        )
        .unwrap();

    group.bench_function("custom", |b| {
        b.iter(|| {
            let fused_moe_output = black_box(
                fused_moe
                    .forward(
                        &hidden_states,
                        &gate_weights,
                        &up_weights,
                        None,
                        &scores,
                        &indices,
                        1_u32,
                    )
                    .unwrap(),
            );
            fused_moe_output.device().synchronize().unwrap();
        })
    });

    group.finish();

    // --- Manual Summary ---
    println!("\n--- Summary for {} ---", group_name);

    // Helper to run a few iterations and get an average time
    let measure = |f: &mut dyn FnMut()| {
        let mut durations = Vec::new();

        // Warmup
        for _ in 0..5 {
            f();
        }

        // Measurement
        for _ in 0..100 {
            let start = std::time::Instant::now();
            f();
            durations.push(start.elapsed());
        }

        let avg_duration = durations.iter().sum::<std::time::Duration>() / durations.len() as u32;
        avg_duration
    };

    let mut native = || {
        let moe_output: Tensor = forward_moe_expert(
            &hidden_states,
            &gate_weights.permute((0, 2, 1)).unwrap(),
            &up_weights.permute((0, 2, 1)).unwrap(),
            &scores,
            &indices,
            n_embed,
            num_experts,
        )
        .unwrap();
        moe_output.device().synchronize().unwrap();
    };

    let mut custom = || {
        let fused_moe_output = fused_moe
            .forward(
                &hidden_states,
                &gate_weights,
                &up_weights,
                None,
                &scores,
                &indices,
                1_u32,
            )
            .unwrap();
        fused_moe_output.device().synchronize().unwrap();
    };

    let native_dur = measure(&mut native);
    let custom_dur = measure(&mut custom);

    let speedup = native_dur.as_secs_f64() / custom_dur.as_secs_f64();
    println!(
        "Native: {:>10.3?} | Custom: {:>10.3?} | Speedup: {:.2}x",
        native_dur, custom_dur, speedup
    );

    println!("-----------------------------------\n");
}

// Benchmark for Qwen3 style MoE (with down projection)
fn run_benchmark_qwen3(
    c: &mut Criterion,
    group_name: &str,
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
    n_embed: usize,
    dtype: DType,
) {
    let (hidden_states, gate_weights, up_weights, down_weights, scores, indices, fused_moe) =
        match setup_tensors_qwen3(seq_len, num_experts, top_k, n_embed, dtype) {
            Ok(t) => t,
            Err(e) => {
                println!(
                    "Failed to setup tensors for group {}, skipping benchmark: {:?}",
                    group_name, e
                );
                return;
            }
        };

    let mut group = c.benchmark_group(group_name);
    group.sample_size(500);
    group.warm_up_time(std::time::Duration::from_millis(1500));
    group.measurement_time(std::time::Duration::from_millis(15000));

    // custom warmup (Qwen3 style with down projection, moe_type=0)
    let _ = fused_moe
        .forward(
            &hidden_states,
            &gate_weights,
            &up_weights,
            Some(&down_weights),
            &scores,
            &indices,
            0_u32, // Qwen3 MoE
        )
        .unwrap();

    group.bench_function("custom", |b| {
        b.iter(|| {
            let fused_moe_output = black_box(
                fused_moe
                    .forward(
                        &hidden_states,
                        &gate_weights,
                        &up_weights,
                        Some(&down_weights),
                        &scores,
                        &indices,
                        0_u32, // Qwen3 MoE
                    )
                    .unwrap(),
            );
            fused_moe_output.device().synchronize().unwrap();
        })
    });

    group.finish();

    // --- Manual Summary ---
    println!("\n--- Summary for {} ---", group_name);

    let measure = |f: &mut dyn FnMut()| {
        let mut durations = Vec::new();
        for _ in 0..5 {
            f();
        }
        for _ in 0..100 {
            let start = std::time::Instant::now();
            f();
            durations.push(start.elapsed());
        }
        durations.iter().sum::<std::time::Duration>() / durations.len() as u32
    };

    let mut native_qwen3 = || {
        let moe_output: Tensor = forward_moe_expert_qwen3(
            &hidden_states,
            &gate_weights.permute((0, 2, 1)).unwrap(),
            &up_weights.permute((0, 2, 1)).unwrap(),
            &down_weights.permute((0, 2, 1)).unwrap(),
            &scores,
            &indices,
            n_embed,
            num_experts,
        )
        .unwrap();
        moe_output.device().synchronize().unwrap();
    };

    let mut custom_qwen3 = || {
        let fused_moe_output = fused_moe
            .forward(
                &hidden_states,
                &gate_weights,
                &up_weights,
                Some(&down_weights),
                &scores,
                &indices,
                0_u32, // Qwen3 MoE
            )
            .unwrap();
        fused_moe_output.device().synchronize().unwrap();
    };

    let native_dur = measure(&mut native_qwen3);
    let custom_dur = measure(&mut custom_qwen3);
    let speedup = native_dur.as_secs_f64() / custom_dur.as_secs_f64();

    println!(
        "Native: {:>10.3?} | Custom: {:>10.3?} | Speedup: {:.2}x",
        native_dur, custom_dur, speedup
    );
    println!("-----------------------------------\n");
}

fn bench_fused_moe(c: &mut Criterion) {
    run_benchmark(c, "nomic_moe_tiny_seq_f32", 16, 8, 2, 768, DType::F32);
    run_benchmark(c, "nomic_moe_short_seq_f32", 32, 8, 2, 768, DType::F32);
    run_benchmark(c, "nomic_moe_mid_seq_f32", 512, 8, 2, 768, DType::F32);
    run_benchmark(c, "nomic_moe_long_seq_f32", 8192, 8, 2, 768, DType::F32);
    run_benchmark(
        c,
        "nomic_moe_very_long_seq_f32",
        32768,
        8,
        2,
        256,
        DType::F32,
    );
    run_benchmark(c, "nomic_moe_tiny_seq_f16", 16, 8, 2, 768, DType::F16);
    run_benchmark(c, "nomic_moe_long_seq_f16", 8192, 8, 2, 768, DType::F16);
    run_benchmark(
        c,
        "nomic_moe_very_long_seq_f16",
        32768,
        8,
        2,
        768,
        DType::F16,
    );

    // top_k=1 benchmarks (tests memset-free optimization with direct STORE)
    run_benchmark(c, "nomic_moe_tiny_seq_topk1_f16", 16, 8, 1, 768, DType::F16);
    run_benchmark(
        c,
        "nomic_moe_long_seq_topk1_f16",
        8192,
        8,
        1,
        768,
        DType::F16,
    );
    run_benchmark(
        c,
        "nomic_moe_very_long_seq_topk1_f16",
        32768,
        8,
        1,
        768,
        DType::F16,
    );

    // BF16 benchmarks
    run_benchmark(c, "nomic_moe_tiny_seq_bf16", 16, 8, 2, 768, DType::BF16);
    run_benchmark(c, "nomic_moe_long_seq_bf16", 8192, 8, 2, 768, DType::BF16);
    run_benchmark(
        c,
        "nomic_moe_very_long_seq_bf16",
        32768,
        8,
        2,
        768,
        DType::BF16,
    );

    // top_k=1 with BF16
    run_benchmark(
        c,
        "nomic_moe_long_seq_topk1_bf16",
        8192,
        8,
        1,
        768,
        DType::BF16,
    );

    // Qwen3-8B-Embedding model (hidden_dim=4096) - uses Qwen3 style with down projection
    run_benchmark_qwen3(c, "qwen3_8b_emb_short_seq_f16", 32, 8, 2, 4096, DType::F16);
    run_benchmark_qwen3(
        c,
        "qwen3_8b_emb_very_long_seq_f16",
        32768,
        8,
        2,
        4096,
        DType::F16,
    );
    run_benchmark_qwen3(
        c,
        "qwen3_8b_emb_short_seq_bf16",
        32,
        8,
        2,
        4096,
        DType::BF16,
    );
    run_benchmark_qwen3(
        c,
        "qwen3_8b_emb_very_long_seq_bf16",
        32768,
        8,
        2,
        4096,
        DType::BF16,
    );
}

criterion_group!(benches, bench_fused_moe);
criterion_main!(benches);
