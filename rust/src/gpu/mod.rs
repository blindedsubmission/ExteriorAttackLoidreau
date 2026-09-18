//! wgpu backend: the block-aware circuit H-step and the block-Wiedemann
//! sequence dots, on GPU.
//!
//! # Design
//!
//! The GPU runs exactly the arithmetic the CPU engines run -- the same
//! grouped substitute-and-discard schedule, the same per-block
//! forward/weight/embed/transpose pipeline -- expressed as one WGSL
//! compute shader with an op switch (see `shaders/mf.wgsl`).  Field
//! elements are four u32 limbs regardless of m (bits above m stay zero),
//! which is what gives the uniform m <= 128 support; the carry-less
//! multiply is shift-and-add (`wmul`, m iterations of `mulx`), matching
//! the reference `gramq.comp` convention.  Because addition is XOR, the
//! tree reduction in the dots kernel is bit-identical to the CPU's
//! sequential loop -- the CPU and GPU engines produce the same kernel
//! vectors at equal seed (tested).
//!
//! What stays on the host: setup (pivoting, Frobenius shifts, schedule
//! construction), Berlekamp-Massey / the block annihilator solve, and
//! the Las Vegas `W p == 0` gate.  Per H-step the host uploads the block
//! vector (N*B elements), the device runs broadcast -> r stage
//! dispatches -> weight -> embed -> r transposed stage dispatches ->
//! reduce (one command buffer, ~2r+4 dispatches), and the host reads
//! back N*B elements plus the B*B dots matrix of the step.  All other
//! state (schedule, coefficients, weights, U) is resident on the device.
//!
//! The scalar engine uses the same backend with B = 1 (dots computed on
//! the host from the downloaded vector -- one scalar per step -- which
//! keeps the scalar path's bit-compatibility with the reference CPU
//! engine trivially auditable).

//! Module layout: `params` (dispatch sizing + parameter blocks),
//! `device` (the wgpu context and the step/reconstruction machinery),
//! `solvers` (the two block-Wiedemann drivers), and the tests.

mod device;
mod params;
mod solvers;

#[cfg(test)]
mod tests;

pub use device::GpuEngine;
// tests address ndispatch_pub through the module namespace
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use params::ndispatch_pub;
pub use solvers::{block_wiedemann_gpu, block_wiedemann_gpu_pmb};

/// Errors of the GPU backend.
#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    Device(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no suitable wgpu adapter found"),
            GpuError::Device(s) => write!(f, "wgpu device error: {s}"),
        }
    }
}
impl std::error::Error for GpuError {}

