# candle-moe

Fast CUDA `fused MoE` for [Candle](https://github.com/huggingface/candle) backend.

## Benchmark

vs `candle 0.9.1` native kernels / `topk-softmax kernel`

| seq_len | num_experts | top k | candle 0.9 | candle-moe | speed-up |
|  :---:  | :---:       | :---: |  :---:     | :---:      | :---:    |
| 32      | 8           | 2     | 466.459 µs | 28.358 µs  | 16.45x   |
| 512     | 8           | 2     | 127.168 µs | 26.034 µs  | 4.88x    |
| 8192    | 8           | 2     | 497.787 µs | 29.097 µs  | 17.11x   |
| 32768   | 8           | 2     | 625.476 µs | 30.650 µs  | 20.41x   |

Benchmarks run on GTX 1060 6GB

vs `candle 0.9.1` native kernels / `fused MoE kernel`

|  fp   | seq_len | num_experts | top k | candle 0.9 | candle-moe | speed-up |
| :---: |  :---:  | :---:       | :---: |  :---:     | :---:      | :---:    |
| fp32  | 32      | 8           | 2     | 3.772 ms   | 36.839 µs  | 102.40x  |
| fp32  | 512     | 8           | 2     | 4.058 ms   | 39.022 µs  | 103.99x  |
| fp32  | 8192    | 8           | 2     | 21.342 ms  | 76.588 µs  | 278.66x  |
| fp32  | 32768   | 8           | 2     | 72.280 ms  | 256.842 µs | 281.42x  |
| fp16  | 8192    | 8           | 2     | 21.342 ms  | 76.588 µs  | 278.66x  |
| fp16  | 32768   | 8           | 2     | 72.280 ms  | 256.842 µs | 281.42x  |

Benchmarks run on GTX 1060 6GB

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

let fused_moe = candle_moe::FusedMoeForward::new(num_experts, top_k, moe_act);

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
