# candle-moe

Fast CUDA `fused MoE` for [Candle](https://github.com/huggingface/candle) backend.

## Benchmark

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

## Feature Flags

This crate supports both older and newer versions of candle/cudarc:

|  Feature  |      Candle Version     | cudarc Version | Default |
|   :---:   |          :---:          |       :---:    |  :---:  |
| `cuda-12` | 0.9+                    | 0.16+          |   ✅    |
| `cuda-11` | pre-0.9 (0.6, 0.7, 0.8) | pre-0.16       |         |
