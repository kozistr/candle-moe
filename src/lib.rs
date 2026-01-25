pub mod ffi;

pub use candle;
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{DType, Result, Storage, Tensor};
use half::{bf16, f16};
use std::ptr;

pub fn apply_topk_softmax_<
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
>(
    gating_output: &Tensor,
    topk_weight: &Tensor,
    topk_indices: &Tensor,
    token_expert_indices: &Tensor,
) -> Result<()> {
    let (g, g_l) = gating_output.storage_and_layout();
    let g: &candle::CudaStorage = match &*g {
        Storage::Cuda(g) => g,
        _ => candle::bail!("gating_output must be a cuda tensor"),
    };

    let (w, w_l) = topk_weight.storage_and_layout();
    let w = match &*w {
        Storage::Cuda(w) => w,
        _ => candle::bail!("topk_weight must be a cuda tensor"),
    };

    let (i, i_l) = topk_indices.storage_and_layout();
    let i = match &*i {
        Storage::Cuda(i) => i,
        _ => candle::bail!("topk_indices must be a cuda tensor"),
    };

    let (ei, ei_l) = token_expert_indices.storage_and_layout();
    let ei: &candle::CudaStorage = match &*ei {
        Storage::Cuda(ei) => ei,
        _ => candle::bail!("token_expert_indices must be a cuda tensor"),
    };

    let device = g.device();
    let stream = device.cuda_stream();
    let stream_ref = stream.as_ref();

    let g_rank = g_l.stride().len();
    let w_rank = w_l.stride().len();
    let i_rank = i_l.stride().len();
    let ei_rank = ei_l.stride().len();

    if g_rank != 2 || w_rank != 2 || i_rank != 2 || ei_rank != 2 {
        candle::bail!(
            "apply_topk_softmax_inplace expects input tensors of rank 2 (w: {w_l:?}, i: {i_l:?}, ei: {ei_l:?}, g: {g_l:?})"
        )
    }

    // Get cuda slices for all tensors
    let g = g.as_cuda_slice::<T>()?;
    let w = w.as_cuda_slice::<T>()?;
    let i = i.as_cuda_slice::<u32>()?;
    let ei = ei.as_cuda_slice::<u32>()?;

    // Get cuda views for all tensors
    let g = g.slice(g_l.start_offset()..);
    let w = w.slice(w_l.start_offset()..);
    let i = i.slice(i_l.start_offset()..);
    let ei = ei.slice(ei_l.start_offset()..);

    let (num_tokens, top_k) = w_l.shape().dims2()?;
    let (_, num_experts) = g_l.shape().dims2()?;

    let is_pow2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if !is_pow2 || num_experts > 256 {
        candle::bail!(
            "num_experts should be power of 2 and smaller than 256 (num_experts: {num_experts:?})"
        )
    }

    if (num_tokens, top_k) != i_l.shape().dims2()? {
        candle::bail!(
            "shape mismatch topk_indices {:?}, expected {:?}",
            i_l.shape(),
            (num_tokens, top_k)
        )
    }

    if (num_tokens, top_k) != ei_l.shape().dims2()? {
        candle::bail!(
            "shape mismatch token_expert_indices {:?}, expected {:?}",
            ei_l.shape(),
            (num_tokens, top_k)
        )
    }

    let (gate_devptr, _) = g.device_ptr(stream_ref);
    let gate_ptr = gate_devptr as usize as *const core::ffi::c_void;

    let (weight_devptr, _) = w.device_ptr(stream_ref);
    let weight_ptr = weight_devptr as usize as *mut core::ffi::c_void;

    let (indices_devptr, _) = i.device_ptr(stream_ref);
    let indices_ptr = indices_devptr as usize as *const core::ffi::c_void;

    let (expert_indices_devptr, _) = ei.device_ptr(stream_ref);
    let expert_indices_ptr = expert_indices_devptr as usize as *const core::ffi::c_void;

    unsafe {
        ffi::topk_softmax(
            gate_ptr,
            weight_ptr,
            indices_ptr,
            expert_indices_ptr,
            num_experts as i32,
            num_tokens as i32,
            top_k as i32,
        )
    }

    Ok(())
}

pub fn apply_topk_softmax_inplace(
    gating_output: &Tensor,
    topk_weight: &Tensor,
    topk_indices: &Tensor,
    token_expert_indices: &Tensor,
) -> Result<()> {
    match topk_weight.dtype() {
        DType::F16 => apply_topk_softmax_::<f16>(
            gating_output,
            topk_weight,
            topk_indices,
            token_expert_indices,
        ),
        DType::BF16 => apply_topk_softmax_::<bf16>(
            gating_output,
            topk_weight,
            topk_indices,
            token_expert_indices,
        ),
        DType::F32 => apply_topk_softmax_::<f32>(
            gating_output,
            topk_weight,
            topk_indices,
            token_expert_indices,
        ),
        dt => {
            candle::bail!(
                "apply_topk_softmax_inplace is only supported for f32, f16 and bf16 ({dt:?})"
            )
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Silu,
    Gelu,
    Relu,
}

impl Activation {
    fn to_int(self) -> i32 {
        match self {
            Activation::Silu => 0,
            Activation::Gelu => 1,
            Activation::Relu => 2,
        }
    }
}

fn fused_moe_internal_dtype(dtype: DType) -> Result<u32> {
    let internal_type: u32 = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => candle::bail!("dtype {dtype:?} is not supported"),
    };
    Ok(internal_type)
}

#[allow(clippy::too_many_arguments)]
fn validate_moe_inputs(
    input: &Tensor,
    gate_weights: &Tensor,
    up_weights: &Tensor,
    down_weights: Option<&Tensor>,
    routing_weights: &Tensor,
    expert_indices: &Tensor,
    num_experts: usize,
    num_selected_experts: usize,
    moe_type: u32,
) -> Result<(usize, usize)> {
    let (num_tokens, hidden_dim) = input.dims2()?;
    let (ne_g, hd_g, id_g) = gate_weights.dims3()?;
    let (ne_u, hd_u, id_u) = up_weights.dims3()?;
    let (ne_d, id_d, hd_d) = if let Some(dw) = down_weights {
        dw.dims3()?
    } else {
        (num_experts, id_u, hd_u)
    };
    let (nt, nse) = routing_weights.dims2()?;
    let (nt2, nse2) = expert_indices.dims2()?;

    if ne_g != num_experts || ne_u != num_experts || ne_d != num_experts {
        candle::bail!("Number of experts mismatch");
    }
    if hd_g != hidden_dim || hd_u != hidden_dim {
        candle::bail!("Hidden dimension mismatch for gate/up weights");
    }
    if hd_d != hidden_dim {
        candle::bail!(
            "Hidden dimension mismatch for down weights (expected {}, got {})",
            hidden_dim,
            hd_d
        );
    }
    if id_g != id_u || id_u != id_d {
        candle::bail!(
            "Intermediate dimension mismatch (gate: {}, up: {}, down: {})",
            id_g,
            id_u,
            id_d
        );
    }

    if nt != num_tokens || nt2 != num_tokens {
        candle::bail!("Number of tokens mismatch");
    }

    if nse != num_selected_experts || nse2 != num_selected_experts {
        candle::bail!("Number of selected experts mismatch");
    }

    if moe_type > 1 {
        candle::bail!("moe_type must be one of 0 or 1");
    }

    Ok((num_tokens, hidden_dim))
}

pub struct FusedMoE {
    pub num_experts: usize,
    pub num_selected_experts: usize,
    pub activation: Activation,
}

impl FusedMoE {
    pub fn new(num_experts: usize, num_selected_experts: usize, activation: Activation) -> Self {
        Self {
            num_experts,
            num_selected_experts,
            activation,
        }
    }

    /// Performs fused MoE forward pass
    /// Args:
    /// - input: [num_tokens, hidden_dim]
    /// - gate_weights: [num_experts, hidden_dim, intermediate_dim]
    /// - up_weights: [num_experts, hidden_dim, intermediate_dim]
    /// - down_weights: [num_experts, intermediate_dim, hidden_dim]
    /// - routing_weights: [num_tokens, num_selected_experts]
    /// - expert_indices: [num_tokens, num_selected_experts]
    /// - moe_type: qwen3: 0, nomic: 1
    ///
    /// Returns:
    /// - output tensor: [num_tokens, hidden_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input: &Tensor,
        gate_weights: &Tensor,
        up_weights: &Tensor,
        down_weights: Option<&Tensor>,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
        moe_type: u32,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = validate_moe_inputs(
            input,
            gate_weights,
            up_weights,
            down_weights,
            routing_weights,
            expert_indices,
            self.num_experts,
            self.num_selected_experts,
            moe_type,
        )?;

        let dtype = input.dtype();
        let output = Tensor::zeros((num_tokens, hidden_dim), dtype, input.device())?;

        _ = match dtype {
            DType::F16 => self.cuda_fwd::<f16>(
                input,
                gate_weights,
                up_weights,
                down_weights,
                routing_weights,
                expert_indices,
                moe_type,
                &output,
            ),
            DType::BF16 => self.cuda_fwd::<bf16>(
                input,
                gate_weights,
                up_weights,
                down_weights,
                routing_weights,
                expert_indices,
                moe_type,
                &output,
            ),
            DType::F32 => self.cuda_fwd::<f32>(
                input,
                gate_weights,
                up_weights,
                down_weights,
                routing_weights,
                expert_indices,
                moe_type,
                &output,
            ),
            dt => {
                candle::bail!("FusedMoE is only supported for f16, bf16 and f32 ({dt:?})")
            }
        };

        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        input: &Tensor,
        gate_weights: &Tensor,
        up_weights: &Tensor,
        down_weights: Option<&Tensor>,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
        moe_type: u32,
        output: &Tensor,
    ) -> Result<()> {
        let (num_tokens, hidden_dim) = input.dims2()?;
        let (_, _, intermediate_dim) = gate_weights.dims3()?;

        // Get storage and layouts
        let (input_storage, input_layout) = input.storage_and_layout();
        let (gate_storage, gate_layout) = gate_weights.storage_and_layout();
        let (up_storage, up_layout) = up_weights.storage_and_layout();
        let (routing_storage, routing_layout) = routing_weights.storage_and_layout();
        let (indices_storage, indices_layout) = expert_indices.storage_and_layout();
        let (output_storage, output_layout) = output.storage_and_layout();

        // Extract CUDA storage
        let input_cuda = match &*input_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("input must be a cuda tensor"),
        };
        let gate_cuda = match &*gate_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("gate_weights must be a cuda tensor"),
        };
        let up_cuda = match &*up_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("up_weights must be a cuda tensor"),
        };
        let routing_cuda = match &*routing_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("routing_weights must be a cuda tensor"),
        };
        let indices_cuda = match &*indices_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("expert_indices must be a cuda tensor"),
        };
        let output_cuda = match &*output_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle::bail!("output must be a cuda tensor"),
        };

        let device = indices_cuda.device();
        let stream = device.cuda_stream();
        let stream_ref = stream.as_ref();

        let input_slice = input_cuda
            .as_cuda_slice::<T>()?
            .slice(input_layout.start_offset()..);
        let gate_slice = gate_cuda
            .as_cuda_slice::<T>()?
            .slice(gate_layout.start_offset()..);
        let up_slice = up_cuda
            .as_cuda_slice::<T>()?
            .slice(up_layout.start_offset()..);
        let routing_slice = routing_cuda
            .as_cuda_slice::<f32>()?
            .slice(routing_layout.start_offset()..);
        let indices_slice = indices_cuda
            .as_cuda_slice::<u32>()?
            .slice(indices_layout.start_offset()..);
        let output_slice = output_cuda
            .as_cuda_slice::<T>()?
            .slice(output_layout.start_offset()..);

        let (input_devptr, _) = input_slice.device_ptr(stream_ref);
        let input_ptr = input_devptr as usize as *const core::ffi::c_void;

        let (gate_devptr, _) = gate_slice.device_ptr(stream_ref);
        let gate_ptr = gate_devptr as usize as *const core::ffi::c_void;

        let (up_devptr, _) = up_slice.device_ptr(stream_ref);
        let up_ptr = up_devptr as usize as *const core::ffi::c_void;

        let (routing_devptr, _) = routing_slice.device_ptr(stream_ref);
        let routing_ptr = routing_devptr as usize as *const core::ffi::c_void;

        let (indices_devptr, _) = indices_slice.device_ptr(stream_ref);
        let indices_ptr = indices_devptr as usize as *const core::ffi::c_void;

        let (output_devptr, _) = output_slice.device_ptr(stream_ref);
        let output_ptr = output_devptr as usize as *mut core::ffi::c_void;

        let down_ptr = if let Some(dw) = down_weights {
            let (down_storage, down_layout) = dw.storage_and_layout();

            let down_cuda = match &*down_storage {
                Storage::Cuda(cuda_storage) => cuda_storage,
                _ => candle::bail!("down_weights must be a cuda tensor"),
            };

            let down_slice = down_cuda
                .as_cuda_slice::<T>()?
                .slice(down_layout.start_offset()..);

            let (down_devptr, _) = down_slice.device_ptr(stream_ref);
            down_devptr as usize as *const core::ffi::c_void
        } else {
            ptr::null()
        };

        let dtype = fused_moe_internal_dtype(input.dtype())?;

        // Use expert-parallel kernel for large batches (much faster due to weight reuse)
        if num_tokens >= 16 {
            // Allocate workspace buffers for expert-parallel processing.
            // Using U32 which maps to 32-bit integers expected by C interface.
            let total_assignments = num_tokens * self.num_selected_experts;
            let cuda_dev = input_cuda.device();

            let expert_counts = cuda_dev.alloc_zeros::<u32>(self.num_experts)?;
            let counters = cuda_dev.alloc_zeros::<u32>(self.num_experts)?;
            let expert_offsets = unsafe { cuda_dev.alloc::<u32>(self.num_experts + 1)? };
            let block_offsets = unsafe { cuda_dev.alloc::<u32>(self.num_experts + 1)? };
            let token_ids = unsafe { cuda_dev.alloc::<u32>(total_assignments)? };
            let select_ids = unsafe { cuda_dev.alloc::<u32>(total_assignments)? };

            let (counts_devptr, _) = expert_counts.device_ptr(stream_ref);
            let (offsets_devptr, _) = expert_offsets.device_ptr(stream_ref);
            let (tids_devptr, _) = token_ids.device_ptr(stream_ref);
            let (sids_devptr, _) = select_ids.device_ptr(stream_ref);
            let (ctrs_devptr, _) = counters.device_ptr(stream_ref);
            let (blkoffs_devptr, _) = block_offsets.device_ptr(stream_ref);

            unsafe {
                ffi::fused_moe_auto(
                    input_ptr,
                    gate_ptr,
                    up_ptr,
                    down_ptr,
                    routing_ptr,
                    indices_ptr,
                    output_ptr,
                    num_tokens as i32,
                    hidden_dim as i32,
                    intermediate_dim as i32,
                    self.num_experts as i32,
                    self.num_selected_experts as i32,
                    self.activation.to_int(),
                    moe_type,
                    dtype,
                    counts_devptr as usize as *mut core::ffi::c_int,
                    offsets_devptr as usize as *mut core::ffi::c_int,
                    tids_devptr as usize as *mut core::ffi::c_int,
                    sids_devptr as usize as *mut core::ffi::c_int,
                    ctrs_devptr as usize as *mut core::ffi::c_int,
                    blkoffs_devptr as usize as *mut core::ffi::c_int,
                );
            }
        } else {
            // For small batches, use simple token-parallel kernel
            unsafe {
                ffi::fused_moe(
                    input_ptr,
                    gate_ptr,
                    up_ptr,
                    down_ptr,
                    routing_ptr,
                    indices_ptr,
                    output_ptr,
                    num_tokens as i32,
                    hidden_dim as i32,
                    intermediate_dim as i32,
                    self.num_selected_experts as i32,
                    self.activation.to_int(),
                    moe_type,
                    dtype,
                );
            }
        }

        Ok(())
    }
}
