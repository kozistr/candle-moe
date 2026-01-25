# candle-moe

Fast CUDA `fused MoE` for [Candle](https://github.com/huggingface/candle) backend.

## Benchmark

vs `candle 0.9.1` native kernels / `topk-softmax kernel`

| seq_len | num_experts | top k | candle 0.9 | candle-moe | speed-up |
|  :---:  | :---:       | :---: |  :---:     | :---:      | :---:    |
| 32      | 8           | 2     | 26.013 µs  | 7.968 µs   | 3.26x    |
| 512     | 8           | 2     | 25.829 µs  | 7.888 µs   | 3.27x    |
| 8192    | 8           | 2     | 42.234 µs  | 8.205 µs   | 5.15x    |
| 32768   | 8           | 2     | 100.442 µs | 9.801 µs   | 10.25x   |

Benchmarks run on A40 GPU

vs `candle 0.9.1` native kernels / `fused MoE kernel`

|  fp   | seq_len | num_experts | top k | candle 0.9 | candle-moe | speed-up |
| :---: |  :---:  | :---:       | :---: |  :---:     | :---:      | :---:    |
| fp32  | 32      | 8           | 2     | 1.872 ms   | 230.790 µs | 8.11x    |
| fp32  | 512     | 8           | 2     | 1.968 ms   | 488.412 µs | 4.03x    |
| fp32  | 8192    | 8           | 2     | 12.620 ms  | 5.361 ms   | 2.35x    |
| fp32  | 32768   | 8           | 2     | 43.642 ms  | 21.152 ms  | 2.06x    |
| fp16  | 8192    | 8           | 2     | 9.914 ms   | 7.265 ms   | 1.36x    |
| fp16  | 32768   | 8           | 2     | 38.065 ms  | 26.295 ms  | 1.45x    |

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
