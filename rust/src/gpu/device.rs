//! The wgpu device context and the step/reconstruction machinery.

use super::params::{
    base_params, ndispatch, sub_chunk_size, workgroups, Params, MAX_CHUNK,
    PARAM_STRIDE,
};
use super::GpuError;
use crate::field::{Fe, Field};
use crate::operator::Operator;
use pollster::block_on;
use wgpu::util::DeviceExt;

const WGSL: &str = include_str!("../../shaders/mf.wgsl");

/// Timestamp-attribution state: one query pair per dispatch of
/// the FIRST step of a chunk; resolved and read back with the chunk.
struct TsCtx {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    /// nanoseconds per timestamp tick
    period_ns: f32,
}

struct GpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    // holds the schedule/offset/vector buffers for the engine lifetime
    #[allow(dead_code)]
    sched_buf: wgpu::Buffer,
    coef_buf: wgpu::Buffer,
    vec_buf: wgpu::Buffer,
    #[allow(dead_code)]
    offs_buf: wgpu::Buffer,
    /// total u32 words in `vec_buf`
    #[allow(dead_code)] // plane count kept alongside the buffers it describes
    vec_words: u32,
    /// element offsets inside vec
    v_off: u32,
    u_off: u32,
    y_off: u32,
    t_off: u32,
    a_off: u32,
    d_off: u32,
    coef_off: u32,
    w_off: u32,
    /// element offset of the reconstruction `A_j` blocks in coef
    ac_off: u32,
    ts: Option<TsCtx>,
    /// stage group ranges of the fwd/trn tables (host copy)
    fwd_stages: Vec<(u32, u32)>,
    trn_stages: Vec<(u32, u32)>,
    sched_words_fwd: u32,
    n_groups: u32,
}

fn limbs4(x: Fe) -> [u32; 4] {
    Field::to_limbs(x)
}

/// The GPU half of a Wiedemann run: owns the operator's device state and
/// executes H-steps (and the dots of the block sequence) on the GPU.
pub struct GpuEngine {
    ctx: GpuCtx,
    pub op: Operator,
    pub bs: usize,
    /// per-dispatch GPU durations (µs) of the last chunk's first step,
    /// in `step_list` order (empty when timestamps are unsupported)
    pub last_step_ts_us: Vec<f32>,
    /// test hook: stop the debug run after weight+embed
    #[cfg(test)]
    pub(crate) debug_through_embed: bool,
}

impl GpuEngine {
    /// Create the GPU context and upload the operator's static state.
    pub fn new(op: Operator, bs: usize) -> Result<GpuEngine, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapters = block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        // prefer a real GPU over software (llvmpipe) fallbacks
        let adapter = adapters
            .iter()
            .find(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
            .or_else(|| adapters.first())
            .ok_or(GpuError::NoAdapter)?
            .clone();
        let info = adapter.get_info();
        // per-op GPU attribution (bench): timestamp queries when the
        // adapter exposes them, gracefully skipped otherwise
        let ts_features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let want_ts = adapter.features().contains(ts_features);
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("matrixfree"),
            required_features: if want_ts { ts_features } else { wgpu::Features::empty() },
            // 5 storage bindings (sched, coef, vec, offs, params): the
            // downlevel default allows only 4; every desktop GPU exposes
            // at least 8
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 8,
                ..wgpu::Limits::downlevel_defaults()
            },
            ..Default::default()
        }))
        .map_err(|e| GpuError::Device(e.to_string()))?;
        eprintln!(
            "gpu: {} ({:?}, backend {:?})",
            info.name, info.device_type, info.backend
        );

        let n = op.n as u32;
        let rr = op.k_coords as u32;
        let s = op.s as u32;
        let bs32 = bs as u32;
        let kr = (op.circuit.k * op.circuit.r) as u32;

        // schedule tables: fwd words then trn words; offs: fwd group
        // offsets (+1 sentinel) then trn group offsets
        let fwd = &op.circuit.fwd;
        let trn = &op.circuit.trn;
        let mut sched = Vec::with_capacity(fwd.words.len() + trn.words.len());
        sched.extend_from_slice(&fwd.words);
        sched.extend_from_slice(&trn.words);
        let mut offs = Vec::with_capacity(fwd.group_off.len() + trn.group_off.len());
        offs.extend_from_slice(&fwd.group_off);
        offs.extend_from_slice(&trn.group_off);
        let n_fwd_groups = fwd.group_off.len() as u32 - 1;

        // coefficient buffer: s*kr D_i entries then s*R weights, 4 limbs
        // each, then the reconstruction A_j blocks: room for
        // bs*bs*(ell+1) elements with ell = c + 2B, c = ceil(N/B) -- the
        // block-Horner window of the block engine
        let coef_off = 0u32;
        let w_off = s * kr;
        let ac_off = s * kr + s * rr;
        let c0 = u64::from(n).div_ceil(u64::from(bs32));
        let ell = c0 + 2 * u64::from(bs32);
        let ac_words = u64::from(bs32) * u64::from(bs32) * (ell + 1);
        let mut coef = Vec::with_capacity(((s * kr + s * rr) as usize + ac_words as usize) * 4);
        for x in &op.di {
            coef.extend_from_slice(&limbs4(*x));
        }
        for x in &op.cw {
            coef.extend_from_slice(&limbs4(*x));
        }
        coef.resize(coef.len() + ac_words as usize * 4, 0);

        // vec layout (element offsets; 4 u32 words per element).  The dots
        // region holds one B*B matrix per chunk step so a whole chunk can
        // run in one submission before the host reads the scalars back.
        let v_off = 0u32;
        let u_off = v_off + n * bs32;
        let y_off = u_off + n * bs32;
        let t_off = y_off + s * n * bs32;
        let a_off = t_off + s * rr * bs32;
        let d_off = a_off + n * bs32;
        let vec_words = (d_off + MAX_CHUNK as u32 * bs32 * bs32) * 4;

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let mkbuf = |data: Option<&[u8]>, size: u32, add: wgpu::BufferUsages| {
            let desc = wgpu::BufferDescriptor {
                label: None,
                size: u64::from(size),
                usage: usage | add,
                mapped_at_creation: false,
            };
            match data {
                Some(bytes) => device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytes,
                    usage: usage | add,
                }),
                None => device.create_buffer(&desc),
            }
        };
        let sched_u32: Vec<u8> = sched.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let offs_u32: Vec<u8> = offs.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let coef_u32: Vec<u8> = coef.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let sched_buf = mkbuf(Some(&sched_u32), sched_u32.len() as u32, wgpu::BufferUsages::empty());
        let offs_buf = mkbuf(Some(&offs_u32), offs_u32.len() as u32, wgpu::BufferUsages::empty());
        let coef_buf = mkbuf(Some(&coef_u32), coef_u32.len() as u32, wgpu::BufferUsages::empty());
        let vec_buf = mkbuf(None, vec_words * 4, wgpu::BufferUsages::empty());
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            // one Params block per dispatch of a chunk, at the dynamic
            // offset alignment; the per-dispatch offset selects the block
            size: PARAM_STRIDE * u64::from(ndispatch(2 * op.circuit.r as u32))
                * (MAX_CHUNK as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mf"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    // read-only STORAGE rather than UNIFORM: the chunk's
                    // param blocks exceed the guaranteed 16 KiB uniform
                    // binding limit, and storage supports the same
                    // dynamic-offset mechanism
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mf-pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sched_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: coef_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vec_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: offs_buf.as_entire_binding() },
                // dynamic offsets select one PARAM_STRIDE-sized block per
                // dispatch, so the static binding range is a single block
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buf,
                        offset: 0,
                        size: std::num::NonZeroU64::new(PARAM_STRIDE),
                    }),
                },
            ],
        });

        let g = Field::to_limbs(op.field.g);
        // timestamp attribution: one query pair per dispatch of the first
        // chunk step (bench only; skip silently when unsupported)
        let ts = if want_ts {
            let count = ndispatch(2 * op.circuit.r as u32) * 2;
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("mf-ts"),
                ty: wgpu::QueryType::Timestamp,
                count,
            });
            let bytes = u64::from(count) * 8;
            let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mf-ts-resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mf-ts-read"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Some(TsCtx { query_set, resolve_buf, read_buf, period_ns: queue.get_timestamp_period() })
        } else {
            None
        };
        let ctx = GpuCtx {
            ts,
            device,
            queue,
            pipeline,
            bind_group,
            params_buf,
            sched_buf,
            coef_buf,
            vec_buf,
            offs_buf,
            vec_words,
            v_off,
            u_off,
            y_off,
            t_off,
            a_off,
            d_off,
            coef_off,
            w_off,
            ac_off,
            fwd_stages: fwd.stage_groups.clone(),
            trn_stages: trn.stage_groups.clone(),
            sched_words_fwd: fwd.words.len() as u32,
            n_groups: n_fwd_groups,
        };
        // stash the (m-dependent) constants once
        let mut params = base_params(&op, bs);
        params.g = g;
        params.m_limb = op.field.m / 32;
        params.m_bit = op.field.m % 32;
        ctx.queue.write_buffer(&ctx.params_buf, 0, bytemuck::bytes_of(&params));
        Ok(GpuEngine {
            ctx,
            op,
            bs,
            last_step_ts_us: Vec::new(),
            #[cfg(test)]
            debug_through_embed: false,
        })
    }

    /// Upload the U block (B columns, interleaved x*B + c).
    pub fn upload_u(&mut self, u: &[Fe]) {
        let mut bytes = Vec::with_capacity(u.len() * 16);
        for x in u {
            bytes.extend_from_slice(&limbs4(*x).map(u32::to_ne_bytes).concat());
        }
        let off = u64::from(self.ctx.u_off) * 16;
        self.ctx.queue.write_buffer(&self.ctx.vec_buf, off, &bytes);
    }

    /// One H-step on a block: y <- M y in place (host view), and the dot
    /// matrix S[c1][c2] = `U_col(c1)` . `Y_col(c2)` of the INPUT y returned.
    pub fn hstep_dots(&mut self, y: &mut [Fe]) -> Vec<Fe> {
        let mut dots = self.run_chunk(y, 1);
        dots.pop().expect("one chunk step")
    }

    #[allow(dead_code)] // single-dispatch helper for test/debug paths
    fn dispatch(&mut self, params: &Params, wg: u32, wgy: u32) {
        // single-dispatch helper for the test/debug paths: params in block 0
        self.ctx
            .queue
            .write_buffer(&self.ctx.params_buf, 0, bytemuck::bytes_of(params));
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ctx.pipeline);
            pass.set_bind_group(0, &self.ctx.bind_group, &[0]);
            pass.dispatch_workgroups(wg, wgy, 1);
        }
        let sub = enc.finish();
        self.ctx.queue.submit([sub]);
    }

    /// The ordered dispatch list of one H-step with its dots written to
    /// dot slot `d_slot`: dots, broadcast, r forward stages, weight,
    /// embed, r transposed stages, reduce.  Each entry is
    /// (params, workgroups.x, workgroups.y); the stage ops dispatch the
    /// stack blocks along y (s-parallel dispatch).
    fn step_list(&self, d_slot: u32) -> Vec<(Params, u32, u32)> {
        let bs = self.bs as u32;
        let n = self.op.n as u32;
        let mut params = base_params(&self.op, self.bs);
        params.g = Field::to_limbs(self.op.field.g);
        params.m_limb = self.op.field.m / 32;
        params.m_bit = self.op.field.m % 32;
        params.v_off = self.ctx.v_off;
        params.u_off = self.ctx.u_off;
        params.y_off = self.ctx.y_off;
        params.t_off = self.ctx.t_off;
        params.a_off = self.ctx.a_off;
        params.d_off = self.ctx.d_off + d_slot * bs * bs;
        params.coef_off = self.ctx.coef_off;
        params.w_off = self.ctx.w_off;
        let mut list = Vec::with_capacity(ndispatch(2 * self.op.circuit.r as u32) as usize);
        let (su, nu, bsu) = (self.op.s as u64, u64::from(n), u64::from(bs));
        let mut push = |op: u32, e0: u32, e1: u32, base: u32, obase: u32, wg: u32, wgy: u32| {
            let mut p = params;
            p.op = op;
            p.e0 = e0;
            p.e1 = e1;
            p.base = base;
            p.obase = obase;
            list.push((p, wg, wgy));
        };
        // dots of the input (before the step): one workgroup PER PAIR
        // (the shader indexes pairs by workgroup id, so this dispatch is
        // counted in workgroups, not threads)
        push(6, 0, 0, 0, 0, (bsu * bsu) as u32, 1);
        // broadcast
        push(0, 0, 0, 0, 0, workgroups(su * nu * bsu), 1);
        // forward stages (b = r-1 downto 0), stack blocks on grid.y
        for b in (0..self.op.circuit.r).rev() {
            let (e0, e1) = self.ctx.fwd_stages[b];
            push(1, e0, e1, 0, 0, workgroups(u64::from(e1 - e0) * bsu), su as u32);
        }
        // weight + embed
        push(3, 0, 0, 0, 0, workgroups(su * (self.op.k_coords as u64) * bsu), 1);
        push(4, 0, 0, 0, 0, workgroups(su * nu * bsu), 1);
        // transposed stages (b = 0..r-1), on the trn table, blocks on grid.y
        for b in 0..self.op.circuit.r {
            let (e0, e1) = self.ctx.trn_stages[b];
            push(2, e0, e1, self.ctx.sched_words_fwd, self.ctx.n_groups + 1, workgroups(u64::from(e1 - e0) * bsu), su as u32);
        }
        // reduce
        push(5, 0, 0, 0, 0, workgroups(nu * bsu), 1);
        // handoff for chunk chaining: copy the accumulator to the input
        // region (same-buffer copies are illegal in WebGPU, hence op 7)
        push(7, 0, 0, 0, 0, workgroups(nu * bsu * 4), 1);
        list
    }

    /// Run `steps` H-steps in ONE submission: upload y once, each step is
    /// one compute pass (all its dispatches ordered inside the pass, wgpu
    /// inserts the inter-dispatch barriers) followed by an encoder-level
    /// a -> v copy that hands the result to the next step; one poll and
    /// one readback of the final vector and all step dots at the end.
    /// Returns the per-step dot matrices (of each step's INPUT), exactly
    /// as `steps` calls of `hstep_dots` would.
    pub fn run_chunk(&mut self, y: &mut [Fe], steps: usize) -> Vec<Vec<Fe>> {
        let bs = self.bs;
        let n = self.op.n;
        assert_eq!(y.len(), n * bs);
        assert!((1..=MAX_CHUNK).contains(&steps), "chunk size");
        // upload y into the v region
        let mut bytes = Vec::with_capacity(y.len() * 16);
        for x in y.iter() {
            bytes.extend_from_slice(&limbs4(*x).map(u32::to_ne_bytes).concat());
        }
        self.ctx
            .queue
            .write_buffer(&self.ctx.vec_buf, u64::from(self.ctx.v_off) * 16, &bytes);

        // stage all param blocks: step k, dispatch j -> block k*nd + j
        let nd = ndispatch(2 * self.op.circuit.r as u32) as usize;
        let mut lists = Vec::with_capacity(steps);
        for d_slot in 0..steps {
            lists.push(self.step_list(d_slot as u32));
        }
        // split into sub-submissions under the pass cap; the op-7 handoff
        // chains the steps across submissions, all dots land in their d
        // slots, and one final readback serves the whole chunk.  A poll
        // between submissions keeps each submission's param blob (whose
        // dynamic offsets restart at 0) from racing the previous one.
        let per_step = lists[0].len();
        let cap = sub_chunk_size(per_step).min(MAX_CHUNK);
        let ts = self.ctx.ts.as_ref();
        let mut done = 0usize;
        while done < steps {
            let k = (steps - done).min(cap);
            let mut blob = vec![0u8; PARAM_STRIDE as usize * nd * k];
            for (q, list) in lists[done..done + k].iter().enumerate() {
                for (j, (p, _, _)) in list.iter().enumerate() {
                    let off = (q * nd + j) * PARAM_STRIDE as usize;
                    blob[off..off + std::mem::size_of::<Params>()]
                        .copy_from_slice(bytemuck::bytes_of(p));
                }
            }
            self.ctx
                .queue
                .write_buffer(&self.ctx.params_buf, 0, &blob);
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for (q, list) in lists[done..done + k].iter().enumerate() {
                for (j, (_, wgx, wgy)) in list.iter().enumerate() {
                    let off = ((q * nd + j) * PARAM_STRIDE as usize) as u32;
                    // one pass per dispatch: wgpu guarantees ordering and
                    // memory visibility between passes of one encoder (same-
                    // bind-group dispatches inside ONE pass are not ordered)
                    // timestamp attribution: only the chunk's FIRST step is
                    // instrumented (one query pair per dispatch)
                    let timestamp_writes = match (done + q, ts) {
                        (0, Some(t)) => Some(wgpu::ComputePassTimestampWrites {
                            query_set: &t.query_set,
                            beginning_of_pass_write_index: Some((2 * j) as u32),
                            end_of_pass_write_index: Some((2 * j + 1) as u32),
                        }),
                        _ => None,
                    };
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes,
                    });
                    pass.set_pipeline(&self.ctx.pipeline);
                    pass.set_bind_group(0, &self.ctx.bind_group, &[off]);
                    pass.dispatch_workgroups(*wgx, *wgy, 1);
                }
            }
            if done == 0 {
                if let Some(t) = ts {
                    let count = nd * 2;
                    enc.resolve_query_set(&t.query_set, 0..count as u32, &t.resolve_buf, 0);
                    enc.copy_buffer_to_buffer(&t.resolve_buf, 0, &t.read_buf, 0, count as u64 * 8);
                }
            }
            let sub = enc.finish();
            self.ctx.queue.submit([sub]);
            self.ctx
                .device
                .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(60)) })
                .expect("gpu wait");
            done += k;
        }
        let ab_bytes = (n * bs * 16) as u64;
        // read back the final vector and all chunk dots in one wait
        self.ctx
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(60)) })
            .expect("gpu wait");
        let acc_bytes = self.readback(u64::from(self.ctx.a_off) * 16, ab_bytes);
        for (x, chunk) in y.iter_mut().zip(acc_bytes.chunks_exact(16)) {
            let l = [
                u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                u32::from_ne_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                u32::from_ne_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
                u32::from_ne_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
            ];
            *x = Field::from_limbs(l);
        }
        self.last_step_ts_us.clear();
        if let Some(t) = &self.ctx.ts {
            let slice = t.read_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).expect("ts map channel");
            });
            self.ctx
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(5)),
                })
                .expect("ts map wait");
            rx.recv().expect("ts map rx").expect("ts map");
            let ticks: Vec<u64> = slice
                .get_mapped_range()
                .expect("ts mapped")
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            t.read_buf.unmap();
            for pair in ticks.chunks_exact(2) {
                let d = pair[1].wrapping_sub(pair[0]);
                self.last_step_ts_us.push(d as f32 * t.period_ns / 1000.0);
            }
        }
        let dot_bytes =
            self.readback(u64::from(self.ctx.d_off) * 16, (steps * bs * bs * 16) as u64);
        let mut dots = Vec::with_capacity(steps);
        for k in 0..steps {
            let base = k * bs * bs * 16;
            let mut s = Vec::with_capacity(bs * bs);
            for chunk in dot_bytes[base..base + bs * bs * 16].chunks_exact(16) {
                let l = [
                    u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    u32::from_ne_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                    u32::from_ne_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
                    u32::from_ne_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
                ];
                s.push(Field::from_limbs(l));
            }
            dots.push(s);
        }
        dots
    }


    /// Dispatch list of one reconstruction round at order `j`: an H-step
    /// on the ncand-column Z block (broadcast, stages, weight, embed,
    /// reduce, handoff -- no dots) followed by the op-8 accumulate
    /// Z ^= V0 `A_j`.  Pipeline ops run with block width ncand; op 8 carries
    /// the engine block stride for `V0/A_j`.  2r+7 dispatches, fits in the
    /// ndispatch(2r) param slots.
    fn recon_list(&self, j: u32, ncand: usize) -> Vec<(Params, u32, u32)> {
        let bs = self.bs as u32;
        let n = self.op.n as u32;
        let nb = ncand as u64;
        let mut params = base_params(&self.op, ncand);
        params.g = Field::to_limbs(self.op.field.g);
        params.m_limb = self.op.field.m / 32;
        params.m_bit = self.op.field.m % 32;
        params.v_off = self.ctx.v_off;
        params.u_off = self.ctx.u_off;
        params.y_off = self.ctx.y_off;
        params.t_off = self.ctx.t_off;
        params.a_off = self.ctx.a_off;
        params.d_off = self.ctx.d_off;
        params.coef_off = self.ctx.coef_off;
        params.w_off = self.ctx.w_off;
        params.ac_off = self.ctx.ac_off;
        params.ncand = ncand as u32;
        let mut list = Vec::with_capacity(ndispatch(2 * self.op.circuit.r as u32) as usize);
        let mut push = |op: u32, e0: u32, e1: u32, base: u32, obase: u32, wg: u32, wgy: u32| {
            let mut p = params;
            p.op = op;
            p.e0 = e0;
            p.e1 = e1;
            p.base = base;
            p.obase = obase;
            list.push((p, wg, wgy));
        };
        let s = self.op.s as u64;
        push(0, 0, 0, 0, 0, workgroups(s * u64::from(n) * nb), 1);
        for b in (0..self.op.circuit.r).rev() {
            let (e0, e1) = self.ctx.fwd_stages[b];
            push(1, e0, e1, 0, 0, workgroups(u64::from(e1 - e0) * nb), s as u32);
        }
        push(3, 0, 0, 0, 0, workgroups(s * (self.op.k_coords as u64) * nb), 1);
        push(4, 0, 0, 0, 0, workgroups(s * u64::from(n) * nb), 1);
        for b in 0..self.op.circuit.r {
            let (e0, e1) = self.ctx.trn_stages[b];
            push(2, e0, e1, self.ctx.sched_words_fwd, self.ctx.n_groups + 1, workgroups(u64::from(e1 - e0) * nb), s as u32);
        }
        push(5, 0, 0, 0, 0, workgroups(u64::from(n) * nb), 1);
        push(7, 0, 0, 0, 0, workgroups(u64::from(n) * nb * 4), 1);
        // accumulate round: engine block stride for V0/A_j, order in e0
        let mut p8 = params;
        p8.op = 8;
        p8.bs = bs;
        p8.e0 = j;
        list.push((p8, workgroups(u64::from(n) * nb), 1));
        list
    }

    /// GPU-resident batched block-Horner reconstruction:
    /// Z = sum_{j=0..=ell} M^j (V0 `A_j`) for a batch of `ncand` candidate
    /// columns, computed as ell+1 rounds of [H-step on Z, Z ^= V0 `A_j`]
    /// through the same one-pass-per-dispatch chunk machinery -- Z never
    /// leaves the device until the single final readback.
    ///
    /// `v0` is the PRISTINE start block (n*bs, engine stride; the Krylov
    /// mutates its own copy, this one must be the original); `a` holds the
    /// per-order coefficient blocks, layout a[j*bs*ncand + c*bs + bp].
    pub fn reconstruct(&mut self, v0: &[Fe], a: &[Fe], ell: usize, ncand: usize) -> Vec<Fe> {
        let (n, bs) = (self.op.n, self.bs);
        assert_eq!(v0.len(), n * bs, "v0 must be the pristine n*bs start block");
        assert_eq!(a.len(), bs * ncand * (ell + 1), "A_j block layout");
        assert!(ncand >= 1 && ncand <= bs, "candidate batch within engine block");
        self.upload_u(v0);
        let mut ac_bytes = Vec::with_capacity(a.len() * 16);
        for x in a {
            ac_bytes.extend_from_slice(&limbs4(*x).map(u32::to_ne_bytes).concat());
        }
        self.ctx
            .queue
            .write_buffer(&self.ctx.coef_buf, u64::from(self.ctx.ac_off) * 16, &ac_bytes);
        let zeros = vec![0u8; n * ncand * 16];
        self.ctx
            .queue
            .write_buffer(&self.ctx.vec_buf, u64::from(self.ctx.v_off) * 16, &zeros);
        let nd = ndispatch(2 * self.op.circuit.r as u32) as usize;
        let mut done = 0usize;
        let per_round = 2 * self.op.circuit.r + 7;
        let cap = sub_chunk_size(per_round).min(MAX_CHUNK);
        while done < ell + 1 {
            let k = (ell + 1 - done).min(cap);
            let mut blob = vec![0u8; PARAM_STRIDE as usize * nd * k];
            let mut lists = Vec::with_capacity(k);
            for q in 0..k {
                // Horner runs the orders j = ell..0
                let j = (ell - (done + q)) as u32;
                lists.push(self.recon_list(j, ncand));
            }
            for (q, list) in lists.iter().enumerate() {
                for (t, (p, _, _)) in list.iter().enumerate() {
                    let off = (q * nd + t) * PARAM_STRIDE as usize;
                    blob[off..off + std::mem::size_of::<Params>()]
                        .copy_from_slice(bytemuck::bytes_of(p));
                }
            }
            self.ctx
                .queue
                .write_buffer(&self.ctx.params_buf, 0, &blob);
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for (q, list) in lists.iter().enumerate() {
                for (t, (_, wgx, wgy)) in list.iter().enumerate() {
                    let off = ((q * nd + t) * PARAM_STRIDE as usize) as u32;
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.ctx.pipeline);
                    pass.set_bind_group(0, &self.ctx.bind_group, &[off]);
                    pass.dispatch_workgroups(*wgx, *wgy, 1);
                }
            }
            let sub = enc.finish();
            self.ctx.queue.submit([sub]);
            // complete before the next chunk's param blob overwrites the
            // blocks this submission is still reading (they restart at
            // offset 0 each chunk), and keep every submission short -- see
            // MAX_SUB_PASSES
            self.ctx
                .device
                .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(300)) })
                .expect("gpu wait");
            done += k;
        }
        let bytes = self.readback(u64::from(self.ctx.v_off) * 16, (n * ncand * 16) as u64);
        let mut z = Vec::with_capacity(n * ncand);
        for chunk in bytes.chunks_exact(16) {
            let l = [
                u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                u32::from_ne_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                u32::from_ne_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
                u32::from_ne_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
            ];
            z.push(Field::from_limbs(l));
        }
        z
    }


    /// Debug: run broadcast + forward stages only and return the y
    /// region (s buffers of N*B interleaved elements).
    #[cfg(test)]
    pub(crate) fn debug_forward_state(&mut self, y: &[Fe]) -> Vec<Fe> {
        let bs = self.bs;
        let n = self.op.n;
        let mut bytes = Vec::with_capacity(y.len() * 16);
        for x in y.iter() {
            bytes.extend_from_slice(&limbs4(*x).map(|w| w.to_ne_bytes()).concat());
        }
        self.ctx
            .queue
            .write_buffer(&self.ctx.vec_buf, self.ctx.v_off as u64 * 16, &bytes);
        let groups = |items: u64| items.div_ceil(64).min(65535) as u32;
        let mut params = base_params(&self.op, bs);
        params.g = Field::to_limbs(self.op.field.g);
        params.m_limb = self.op.field.m / 32;
        params.m_bit = self.op.field.m % 32;
        params.v_off = self.ctx.v_off;
        params.u_off = self.ctx.u_off;
        params.y_off = self.ctx.y_off;
        params.t_off = self.ctx.t_off;
        params.a_off = self.ctx.a_off;
        params.d_off = self.ctx.d_off;
        params.coef_off = self.ctx.coef_off;
        params.w_off = self.ctx.w_off;
        params.op = 0;
        self.dispatch(&params, groups((self.op.s * n * bs) as u64), 1);
        params.op = 1;
        params.base = 0;
        params.obase = 0;
        for b in (0..self.op.circuit.r).rev() {
            let (e0, e1) = self.ctx.fwd_stages[b];
            params.e0 = e0;
            params.e1 = e1;
            self.dispatch(&params, groups(((e1 - e0) * bs as u32) as u64), self.op.s as u32);
        }
        params.op = 3;
        self.dispatch(&params, groups((self.op.s * self.op.k_coords * bs) as u64), 1);
        params.op = 4;
        self.dispatch(&params, groups((self.op.s * n * bs) as u64), 1);
        if self.debug_through_embed {
            // transposed stages, then stop
            params.op = 2;
            params.base = self.ctx.sched_words_fwd;
            params.obase = self.ctx.n_groups + 1;
            for b in 0..self.op.circuit.r {
                let (e0, e1) = self.ctx.trn_stages[b];
                params.e0 = e0;
                params.e1 = e1;
                self.dispatch(&params, groups(((e1 - e0) * bs as u32) as u64), self.op.s as u32);
            }
        }
        let raw = self.readback(self.ctx.y_off as u64 * 16, (self.op.s * n * bs * 16) as u64);
        let mut out = Vec::with_capacity(self.op.s * n * bs);
        for chunk in raw.chunks_exact(16) {
            let l = [
                u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                u32::from_ne_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                u32::from_ne_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
                u32::from_ne_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
            ];
            out.push(Field::from_limbs(l));
        }
        out
    }

    fn readback(&mut self, offset: u64, size: u64) -> Vec<u8> {
        let vbuf = self.ctx.vec_buf.clone();
        self.readback_buf(&vbuf, offset, size)
    }

    fn readback_buf(&mut self, buf: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        let pad = (4 - (size % 4)) % 4;
        let size_padded = size + pad;
        let tmp = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_padded,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buf, offset, &tmp, 0, size);
        let sub = encoder.finish();
        self.ctx.queue.submit([sub]);
        let slice = tmp.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(5)) })
            .expect("gpu wait");
        rx.recv().unwrap().expect("map");
        let data = slice.get_mapped_range().expect("mapped range").to_vec();
        tmp.unmap();
        data[..size as usize].to_vec()
    }
}

