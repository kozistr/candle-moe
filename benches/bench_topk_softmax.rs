use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::deepseek2::{TopKLastDimOp, TopKOutput};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn setup_tensors(
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let device = Device::new_cuda(0)?;

    let weights = Tensor::randn(0.0, 1.0, (seq_len, num_experts), &device)?.to_dtype(DType::F32)?;

    let topk_weight = Tensor::zeros((seq_len, top_k), DType::F32, &device)?;
    let topk_indices = Tensor::zeros((seq_len, top_k), DType::U32, &device)?;
    let token_expert_indices = Tensor::zeros((seq_len, top_k), DType::U32, &device)?;

    Ok((weights, topk_weight, topk_indices, token_expert_indices))
}

fn run_benchmark(
    c: &mut Criterion,
    group_name: &str,
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
) {
    let (weights, topk_weight, topk_indices, token_expert_indices) =
        match setup_tensors(seq_len, num_experts, top_k) {
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
    group.sample_size(1000);
    group.warm_up_time(std::time::Duration::from_millis(1500));
    group.measurement_time(std::time::Duration::from_millis(10000));

    // native warmup
    let softmax_weights = candle_nn::ops::softmax_last_dim(&weights).unwrap();
    let TopKOutput {
        values: _,
        indices: _,
    } = softmax_weights.topk(top_k).unwrap();

    group.bench_function("native_f32", |b| {
        b.iter(|| {
            let result = black_box(
                candle_nn::ops::softmax_last_dim(&weights)
                    .unwrap()
                    .topk(top_k)
                    .unwrap(),
            );
            result.values.device().synchronize().unwrap();
        })
    });

    // custom warmup
    candle_moe::apply_topk_softmax_inplace(
        &weights,
        &topk_weight,
        &topk_indices,
        &token_expert_indices,
    )
    .unwrap();

    group.bench_function("custom_f32", |b| {
        b.iter(|| {
            black_box(
                candle_moe::apply_topk_softmax_inplace(
                    &weights,
                    &topk_weight,
                    &topk_indices,
                    &token_expert_indices,
                )
                .unwrap(),
            );
            topk_weight.device().synchronize().unwrap();
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

    let mut native_f32 = || {
        let topk_output = candle_nn::ops::softmax_last_dim(&weights)
            .unwrap()
            .topk(top_k)
            .unwrap();
        topk_output.values.device().synchronize().unwrap();
    };

    let mut custom_f32 = || {
        candle_moe::apply_topk_softmax_inplace(
            &weights,
            &topk_weight,
            &topk_indices,
            &token_expert_indices,
        )
        .unwrap();
        topk_weight.device().synchronize().unwrap();
    };

    let native_f32_dur = measure(&mut native_f32);
    let custom_f32_dur = measure(&mut custom_f32);

    let f32_speedup = native_f32_dur.as_secs_f64() / custom_f32_dur.as_secs_f64();
    println!(
        "F32: Native: {:>10.3?} | Custom: {:>10.3?} | Speedup: {:.2}x",
        native_f32_dur, custom_f32_dur, f32_speedup
    );

    println!("-----------------------------------\n");
}

fn bench_topk_softmax(c: &mut Criterion) {
    run_benchmark(c, "topk_softmax_short_seq", 32, 8, 2);
    run_benchmark(c, "topk_softmax_mid_seq", 512, 8, 2);
    run_benchmark(c, "topk_softmax_long_seq", 8192, 8, 2);
    run_benchmark(c, "topk_softmax_very_long_seq", 32768, 8, 2);
}

criterion_group!(benches, bench_topk_softmax);
criterion_main!(benches);
