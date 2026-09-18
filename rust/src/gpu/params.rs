//! Dispatch sizing, the per-dispatch parameter block, and its staging.

use crate::operator::Operator;

/// Dynamic-offset alignment required by wgpu/WebGPU for uniform buffer
/// bindings; one Params block per dispatch lives at each stride.
pub(super) const PARAM_STRIDE: u64 = 256;
/// Per-step dispatches: dots, broadcast, r forward stages, weight, embed,
/// r transposed stages, reduce = 2r + 5 (one spare slot for safety).
#[allow(dead_code)] // documented pass-count helper
pub(crate) const fn ndispatch_pub(r: u32) -> u32 {
    ndispatch(r)
}

pub(super) const fn ndispatch(r: u32) -> u32 {
    2 * r + 6
}
/// Krylov steps recorded in one chunk (device-resident state between
/// them; the step-to-step handoff is an encoder-level buffer copy).
pub(super) const MAX_CHUNK: usize = 32;

/// Cap on COMPUTE PASSES per queue submission (empirical, wgpu 30 on
/// Mesa/ANV): a 672-pass submission silently corrupts -- dots and final
/// vector mismatch single-step results, with no error surfaced -- while
/// 336- and 480-pass submissions are bit-identical.  Per-encoder pass
/// limits with silent failure are a
/// known cross-platform wgpu hazard (Metal freezes at 683 passes,
/// gfx-rs/wgpu#8047); wgpu does not validate pass counts.  Submissions
/// are split below this cap AND the host polls between submissions,
/// which also serializes the per-chunk param-blob rewrites (their
/// dynamic offsets restart at 0 each submission).
pub(crate) const MAX_SUB_PASSES: usize = 384;

/// Steps per submission for a pipeline of `passes_per_step` dispatches.
pub(crate) fn sub_chunk_size(passes_per_step: usize) -> usize {
    (MAX_SUB_PASSES / passes_per_step.max(1)).max(1)
}

pub(super) fn workgroups(items: u64) -> u32 {
    items.div_ceil(64).min(65535) as u32
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Params {
    pub(super) op: u32,
    pub(super) n: u32,
    pub(super) k_coords: u32,
    pub(super) s: u32,
    pub(super) kr: u32,
    pub(super) bs: u32,
    pub(super) m: u32,
    pub(super) m_limb: u32,
    pub(super) m_bit: u32,
    pub(super) g: [u32; 4],
    pub(super) e0: u32,
    pub(super) e1: u32,
    pub(super) base: u32,
    pub(super) obase: u32,
    pub(super) v_off: u32,
    pub(super) u_off: u32,
    pub(super) y_off: u32,
    pub(super) t_off: u32,
    pub(super) a_off: u32,
    pub(super) d_off: u32,
    pub(super) coef_off: u32,
    pub(super) w_off: u32,
    /// op 8: candidate count of the reconstruction batch (Z stride)
    pub(super) ncand: u32,
    /// op 8: element offset of the `A_j` coefficient blocks in coef
    pub(super) ac_off: u32,
}


pub(super) fn base_params(op: &Operator, bs: usize) -> Params {
    Params {
        op: 0,
        n: op.n as u32,
        k_coords: op.k_coords as u32,
        s: op.s as u32,
        kr: (op.circuit.k * op.circuit.r) as u32,
        bs: bs as u32,
        m: op.field.m,
        m_limb: 0,
        m_bit: 0,
        g: [0; 4],
        e0: 0,
        e1: 0,
        base: 0,
        obase: 0,
        v_off: 0,
        u_off: 0,
        y_off: 0,
        t_off: 0,
        a_off: 0,
        d_off: 0,
        coef_off: 0,
        w_off: 0,
        ncand: 0,
        ac_off: 0,
    }
}

