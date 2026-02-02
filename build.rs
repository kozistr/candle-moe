// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::{Context, Result};
use std::path::PathBuf;

const KERNEL_FILES: [&str; 4] = [
    "kernels/topk_softmax.cu",
    "kernels/fused_moe.cu",
    "kernels/qwen3_moe.cu",
    "kernels/nomic_moe.cu",
];

const HEADER_FILES: [&str; 2] = ["kernels/common.cuh", "kernels/preprocessing.cuh"];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    for header_file in HEADER_FILES.iter() {
        println!("cargo:rerun-if-changed={header_file}");
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_MOE_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            let current_dir = std::env::current_dir()?;
            path.canonicalize().unwrap_or_else(|_| {
                panic!(
                    "Directory doesn't exists: {} (the current directory is {})",
                    &path.display(),
                    current_dir.display()
                )
            })
        }
    };

    let kernels: Vec<_> = KERNEL_FILES.iter().collect();
    let mut builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernels)
        .out_dir(build_dir.clone())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--ptxas-options=-v")
        .arg("--verbose");

    // Disable BF16 kernels for SM < 80 (pre-Ampere GPUs)
    // BF16 WMMA and certain BF16 intrinsics require SM 80+
    if let Ok(compute_cap) = std::env::var("CUDA_COMPUTE_CAP")
        && let Ok(cap) = compute_cap.parse::<u32>()
        && cap < 80
    {
        println!("cargo:warning=CUDA compute capability {cap} < 80, disabling BF16 kernels");
        builder = builder.arg("-DNO_BF16_KERNEL");
    }

    let target = std::env::var("TARGET").unwrap();

    let out_file = if target.contains("msvc") {
        build_dir.join("moe.lib")
    } else {
        build_dir.join("libmoe.a")
    };
    builder.build_lib(out_file);

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if target.contains("msvc") {
        // nothing to link to
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.contains("android") {
        println!("cargo:rustc-link-lib=dylib=c++_shared");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    Ok(())
}
