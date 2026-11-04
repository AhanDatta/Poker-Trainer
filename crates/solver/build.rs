//! Compiles the CUDA kernels in `cuda/kernels/` when the `gpu` feature is on;
//! does nothing otherwise (the default build stays pure Rust).
//!
//! Each `cuda/kernels/<name>.cu` becomes `$OUT_DIR/<name>.fatbin`, holding
//! machine code (SASS) for every architecture in `SOLVER_CUDA_ARCHS` (default
//! `89`, the RTX 40 series) plus PTX for the newest one. SASS loads on any
//! driver of the same CUDA major version; PTX is only a fallback for newer
//! GPUs, since a driver can't JIT PTX from a newer toolkit than itself.
//!
//! Environment:
//! * `NVCC`: path to nvcc (else `$CUDA_PATH/bin/nvcc`, else `nvcc` on PATH);
//! * `SOLVER_CUDA_ARCHS`: comma-separated compute capabilities, e.g. `86,89`;
//! * `SOLVER_GPU_CHECKED`: `1` keeps device bounds assertions in release
//!   builds, `0` drops them in debug builds (default: on unless release).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "gpu")]
    cuda::build();
}

#[cfg(feature = "gpu")]
mod cuda {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    pub fn build() {
        for var in ["NVCC", "CUDA_PATH", "SOLVER_CUDA_ARCHS", "SOLVER_GPU_CHECKED"] {
            println!("cargo:rerun-if-env-changed={var}");
        }
        let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
        let out = PathBuf::from(env::var("OUT_DIR").expect("set by cargo"));
        let include = manifest.join("cuda").join("include");
        let kernels = manifest.join("cuda").join("kernels");
        rerun_for_dir(&include);
        rerun_for_dir(&kernels);

        let nvcc = find_nvcc();
        let archs = archs();
        let checked = match env::var("SOLVER_GPU_CHECKED").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => env::var("PROFILE").as_deref() != Ok("release"),
        };
        println!("cargo:rustc-env=SOLVER_GPU_CHECKED_BUILD={}", u8::from(checked));
        let host_compiler = host_compiler_dir();

        let mut sources: Vec<PathBuf> = std::fs::read_dir(&kernels)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", kernels.display()))
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "cu"))
            .collect();
        sources.sort();
        for src in &sources {
            let stem = src.file_stem().and_then(|s| s.to_str()).expect("UTF-8 kernel file name");
            let dst = out.join(format!("{stem}.fatbin"));
            let mut cmd = Command::new(&nvcc);
            cmd.arg("-std=c++20")
                .arg("--fatbin")
                .args(["-Werror", "all-warnings"])
                // IEEE division/sqrt and denormals: the CPU-parity tests and
                // the reciprocal guard depend on them.
                .args(["-prec-div=true", "-prec-sqrt=true", "-ftz=false"])
                .arg("-lineinfo")
                .arg("-I")
                .arg(&include);
            for (i, a) in archs.iter().enumerate() {
                let code =
                    if i + 1 == archs.len() { format!("[sm_{a},compute_{a}]") } else { format!("sm_{a}") };
                cmd.arg("-gencode").arg(format!("arch=compute_{a},code={code}"));
            }
            if checked {
                cmd.arg("-DCCCL_ENABLE_DEVICE_ASSERTIONS");
            }
            if let Some(dir) = &host_compiler {
                cmd.arg("-ccbin").arg(dir);
            }
            if cfg!(windows) || env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
                // libcu++ requires MSVC's standard-conforming preprocessor.
                cmd.args(["-Xcompiler", "/Zc:preprocessor"]);
            }
            cmd.arg("-o").arg(&dst).arg(src);
            let output = cmd.output().unwrap_or_else(|e| {
                panic!("failed to run {} (is the CUDA Toolkit installed?): {e}", nvcc.display())
            });
            if !output.status.success() {
                panic!(
                    "nvcc failed on {}:\n{}\n{}",
                    src.display(),
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }

    fn rerun_for_dir(dir: &Path) {
        println!("cargo:rerun-if-changed={}", dir.display());
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    rerun_for_dir(&p);
                } else {
                    println!("cargo:rerun-if-changed={}", p.display());
                }
            }
        }
    }

    fn find_nvcc() -> PathBuf {
        if let Some(p) = env::var_os("NVCC") {
            return PathBuf::from(p);
        }
        if let Some(root) = env::var_os("CUDA_PATH") {
            let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
            let p = PathBuf::from(root).join("bin").join(exe);
            if p.exists() {
                return p;
            }
        }
        PathBuf::from("nvcc")
    }

    fn archs() -> Vec<u32> {
        let spec = env::var("SOLVER_CUDA_ARCHS").unwrap_or_else(|_| "89".to_string());
        let mut v: Vec<u32> = spec
            .split(',')
            .map(|s| s.trim().parse().unwrap_or_else(|_| panic!("bad SOLVER_CUDA_ARCHS entry {s:?}")))
            .collect();
        v.sort_unstable();
        v.dedup();
        assert!(!v.is_empty(), "SOLVER_CUDA_ARCHS is empty");
        v
    }

    /// Directory of MSVC's cl.exe (nvcc's host compiler on Windows), which is
    /// usually not on PATH outside a Developer Command Prompt.
    fn host_compiler_dir() -> Option<PathBuf> {
        let target = env::var("TARGET").ok()?;
        if !target.contains("msvc") {
            return None;
        }
        let tool = cc::windows_registry::find_tool(&target, "cl.exe")?;
        tool.path().parent().map(Path::to_path_buf)
    }
}
