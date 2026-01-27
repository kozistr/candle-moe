# candle-moe

Fast CUDA `fused MoE` for [Candle](https://github.com/huggingface/candle) backend.

## Requirements

* SM 7.0+ (Volta+) GPU (e.g. A40)
* `Candle 0.9+`

## Benchmark

vs `candle 0.9.1` native kernels / `topk-softmax kernel`

| seq_len | num_experts | top k | candle 0.9 | candle-moe | speed-up |
|  :---:  | :---:       | :---: |  :---:     | :---:      | :---:    |
| 32      | 8           | 2     | 26.013 µs  | 7.968 µs   | 3.26x    |
| 512     | 8           | 2     | 25.829 µs  | 7.888 µs   | 3.27x    |
| 8192    | 8           | 2     | 46.106 µs  | 8.262 µs   | 5.58x    |
| 32768   | 8           | 2     | 100.683 µs | 9.743 µs   | 10.33x   |

Benchmarks run on A40 GPU

vs `candle 0.9.1` native kernels / `fused MoE kernel`

| moe type | fp   | seq_len | hidden_dim | num_experts | top k | candle 0.9 | candle-moe | speed-up |
|-:-:--| -:-:--|-:-:-----|-:-:--------|-:-:---------|-:-:---|-:-:--------|-:-:--------|-:-:------|
| nomic | f32 | 32      | 768        | 8           | 2     | 1.350 ms   | 185.830 µs | 7.27x    |
| nomic | f32 | 512     | 768        | 8           | 2     | 1.823 ms   | 363.393 µs | 5.02x    |
| nomic | f32 | 8192    | 768        | 8           | 2     | 11.645 ms  | 10.513 ms   | 1.11x    |
| nomic | f32 | 32768   | 768        | 8           | 2     | 43.338 ms  | 39.506 ms  | 1.10x    |
| nomic | f16 | 8192    | 768        | 8           | 2     | 9.492 ms   | 3.883 ms   | 2.44x    |
| nomic | f16 | 32768   | 768        | 8           | 2     | 41.201 ms  | 14.279 ms  | 2.89x    |
| qwen3 | f32 | 32      | 768        | 8           | 2     | 1.455 ms   | 214.840 µs | 6.77x    |
| qwen3 | f32 | 512     | 768        | 8           | 2     | 1.665 ms   | 489.990 µs | 3.40x    |
| qwen3 | f32 | 8192    | 768        | 8           | 2     | 12.479 ms  | 13.355 ms   | 0.93x    |
| qwen3 | f32 | 32768   | 768        | 8           | 2     | 48.655 ms  | 50.697 ms  | 0.96x    |
| qwen3 | f16 | 8192    | 768        | 8           | 2     | 10.592 ms   | 3.831 ms   | 2.76x    |
| qwen3 | f16 | 32768   | 768        | 8           | 2     | 40.856 ms  | 14.702 ms  | 2.78x    |

Benchmarks run on A40 GPU

## Usage

Add to your `Cargo.toml`.

```toml
[dependencies]
candle-moe = { git = "https://github.com/kozistr/candle-moe", rev = "990ac1f42248dd441c51c9b5bcb73c5b77c03f99" }
candle-core = { version = "0.9", features = ["cuda"] }
```

```rust
let topk_weight = Tensor::zeros((seq_len, self.top_k), DType::F32, device)?;
let topk_indices = Tensor::zeros((seq_len, self.top_k), DType::U32, device)?;
let token_expert_indices = Tensor::zeros((seq_len, self.top_k), DType::U32, device)?;

candle_moe::apply_topk_softmax_inplace(
    &weights,
    &topk_weight,
    &topk_indices,
    &token_expert_indices,
)?;

... 

let num_experts = 32;
let top_k = 2;
let moe_act = match activation {
    HiddenAct::Silu => candle_moe::Activation::Silu,
    HiddenAct::Gelu => candle_moe::Activation::Gelu,
    HiddenAct::Relu => candle_moe::Activation::Relu,
    _ => candle::bail!("not supported activation type"),
};

let fused_moe = candle_moe::FusedMoE::new(num_experts, top_k, moe_act);

...

let mut out = self.fused_moe.forward(
    &hidden_states,
    &self.gate_weight,
    &self.up_weight,
    None,
    &top_weights,
    &top_experts,
    1_u32, // Nomic MoE
)?;
```

## Run

### Profile

```bash
$ cargo build --release --bin profile_fused_moe && nsys profile -t cuda,osrt --stats=true --force-overwrite true -o nsys_moe ./target/release/profile_fused_moe
```

### Bench

```bash
cargo bench --bench bench_fused_moe
```

### Test

```bash
cargo test
```
