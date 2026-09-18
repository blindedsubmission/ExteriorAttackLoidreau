// Matrix-free exterior circuit, block-aware, GF(2^m) for m <= 128.
//
// One compute shader, one op-switch (same structure as the reference
// mf.comp, extended from 2 to 4 u32 limbs and from one vector to B
// interleaved block columns).  The host dispatches one workgroup per 64
// items; every thread decodes its item index and returns when out of
// range, so no grid-stride bookkeeping is needed.
//
// State layout inside `vec` (u32 words, 4 per field element):
//   v_off : current block vector Y, N x B interleaved  (element x*B + c)
//   u_off : projection block U,        N x B
//   y_off : per-stack-block buffers,   s buffers of N x B
//   t_off : per-stack-block outputs,   s buffers of R x B
//   a_off : accumulator,               N x B
//   d_off : dot matrix S,              B x B
//
//   op 0 broadcast : ybuf[i][x*B+c] = Y[x*B+c]
//   op 1 fwd stage : ybuf[i][key*B+c] ^= wmul(D_i[cidx], ybuf[i][other*B+c])
//   op 2 tr stage  : same on the transposed tables (key/src exchanged)
//   op 3 weight    : t[i][x*B+c] = wmul(cw[i*R+x], ybuf[i][x*B+c])  (x < R)
//   op 4 embed     : ybuf[i][x*B+c] = x < R ? t[i][x*B+c] : 0
//   op 5 reduce    : A[x*B+c] = XOR_i ybuf[i][x*B+c]
//   op 6 dots      : S[c1*B+c2] = XOR_x U[x*B+c1] * Y[x*B+c2]
//   op 8 accum     : Z[x*ncand+c] ^= sum_bp V0[x*B+bp] * A_j[c*B+bp]
//                    (block-Horner reconstruction round of order j; the
//                    M-step pipeline ops run at block width ncand while
//                    V0/A_j keep the engine block stride B, hence the
//                    separate ncand param)
//
// Race-freedom of op1/op2: within a stage, write targets and read targets
// occupy disjoint colex rank ranges (subsets of [0,j) vs subsets
// containing j); entries sharing a write target are pre-grouped on the
// host, and one thread computes the whole group sum and stores it once.
// Stack blocks are dispatched along grid.y (one workgroup row per block,
// Rung B): blocks touch disjoint ybuf regions, so cross-block
// synchronization is unnecessary and the race-freedom argument is the
// grouped-write-target one above, per block, unchanged.

struct Params {
    op: u32,
    n: u32,          // exterior coordinate count N
    r_coords: u32,   // per-block output count R
    s: u32,          // stack height
    kr: u32,         // k*r coefficients per block
    bs: u32,         // block size B
    m: u32,          // field degree
    m_limb: u32,     // limb of bit m (m/32); 4 means "shifted off"
    m_bit: u32,      // bit of bit m within its limb (m%32)
    g0: u32, g1: u32, g2: u32, g3: u32,  // modulus low part g, 4 limbs
    e0: u32, e1: u32,                    // stage group range [e0, e1)
    base: u32, obase: u32,               // schedule table word base / offs base
    v_off: u32, u_off: u32, y_off: u32,  // element offsets
    t_off: u32, a_off: u32, d_off: u32,
    coef_off: u32,                        // D_i tables start (element idx)
    w_off: u32,                           // weights start (element idx)
    ncand: u32,                           // op 8: candidate count (Z stride)
    ac_off: u32,                          // op 8: A_j blocks start (element idx)
};

@group(0) @binding(0) var<storage, read>       sched: array<u32>;
@group(0) @binding(1) var<storage, read>       coef: array<u32>;
@group(0) @binding(2) var<storage, read_write> vec: array<u32>;
@group(0) @binding(3) var<storage, read>       offs: array<u32>;
@group(0) @binding(4) var<storage, read>      p: Params;

fn ld4(off: u32) -> vec4<u32> {
    return vec4<u32>(vec[4u * off], vec[4u * off + 1u], vec[4u * off + 2u], vec[4u * off + 3u]);
}
fn st4(off: u32, x: vec4<u32>) {
    vec[4u * off] = x.x; vec[4u * off + 1u] = x.y;
    vec[4u * off + 2u] = x.z; vec[4u * off + 3u] = x.w;
}
fn bit_of(a: vec4<u32>, limb: u32, bit: u32) -> u32 {
    var w = a.x;
    if limb == 1u { w = a.y; }
    if limb == 2u { w = a.z; }
    if limb == 3u { w = a.w; }
    return (w >> bit) & 1u;
}

// a -> a * x mod f: 128-bit left shift; if bit m-1 was set, drop the
// shifted-in x^m bit (it falls off the limbs when m == 128) and XOR g.
fn mulx(a: vec4<u32>) -> vec4<u32> {
    let m1_limb = (p.m - 1u) / 32u;
    let m1_bit = (p.m - 1u) % 32u;
    let ovf = bit_of(a, m1_limb, m1_bit);
    // 128-bit left shift: limb i receives the carry bit of limb i-1
    var q = vec4<u32>(
        a.x << 1u,
        (a.y << 1u) | (a.x >> 31u),
        (a.z << 1u) | (a.y >> 31u),
        (a.w << 1u) | (a.z >> 31u)
    );
    if ovf == 1u {
        if p.m_limb == 0u { q.x = q.x ^ (1u << p.m_bit); }
        if p.m_limb == 1u { q.y = q.y ^ (1u << p.m_bit); }
        if p.m_limb == 2u { q.z = q.z ^ (1u << p.m_bit); }
        if p.m_limb == 3u { q.w = q.w ^ (1u << p.m_bit); }
        q = q ^ vec4<u32>(p.g0, p.g1, p.g2, p.g3);
    }
    return q;
}

// Carry-less multiply by shift-and-add: m iterations of mulx.
fn wmul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var acc = vec4<u32>(0u, 0u, 0u, 0u);
    var y = b;
    for (var i = 0u; i < p.m; i++) {
        let abit = bit_of(a, i / 32u, i % 32u);
        let mask = vec4<u32>(0u - abit);
        acc = acc ^ (y & mask);
        y = mulx(y);
    }
    return acc;
}

fn cld(i: u32, idx: u32) -> vec4<u32> {
    let base = p.coef_off + i * p.kr + idx;
    return vec4<u32>(coef[4u * base], coef[4u * base + 1u], coef[4u * base + 2u], coef[4u * base + 3u]);
}
fn cwld(i: u32, idx: u32) -> vec4<u32> {
    let base = p.w_off + i * p.r_coords + idx;
    return vec4<u32>(coef[4u * base], coef[4u * base + 1u], coef[4u * base + 2u], coef[4u * base + 3u]);
}
fn coefld(idx: u32) -> vec4<u32> {
    return vec4<u32>(coef[4u * idx], coef[4u * idx + 1u], coef[4u * idx + 2u], coef[4u * idx + 3u]);
}

var<workgroup> dot_shared: array<vec4<u32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
    let gid = gid3.x;
    if p.op == 0u {
        // broadcast: item = (i, element) over s*N*B
        let per = p.s * p.n * p.bs;
        if gid < per {
            let rem = gid % (p.n * p.bs);
            let i = (gid - rem) / (p.n * p.bs);
            st4(p.y_off + i * p.n * p.bs + rem, ld4(p.v_off + rem));
        }
    } else if p.op == 1u || p.op == 2u {
        // stage: item = (group, column) along x; stack block index = grid.y
        // (dispatch (groups((e1-e0)*B), s, 1)) -- s-parallel, no inner loop
        let c = gid % p.bs;
        let g = gid / p.bs;
        if g < (p.e1 - p.e0) {
            let i = wgid.y;
            let gg = p.e0 + g;
            let off = offs[p.obase + gg];
            let key = sched[p.base + off];
            let cnt = sched[p.base + off + 1u];
            let nb = p.n * p.bs;
            var acc = ld4(p.y_off + i * nb + key * p.bs + c);
            for (var j = 0u; j < cnt; j++) {
                let cidx = sched[p.base + off + 2u + 2u * j];
                let other = sched[p.base + off + 3u + 2u * j];
                acc = acc ^ wmul(cld(i, cidx), ld4(p.y_off + i * nb + other * p.bs + c));
            }
            st4(p.y_off + i * nb + key * p.bs + c, acc);
        }
    } else if p.op == 3u {
        // weight: item = (i, x, c) over s*R*B
        let per = p.s * p.r_coords * p.bs;
        if gid < per {
            let rem = gid % (p.r_coords * p.bs);
            let i = (gid - rem) / (p.r_coords * p.bs);
            st4(p.t_off + i * p.r_coords * p.bs + rem,
                wmul(cwld(i, rem / p.bs), ld4(p.y_off + i * p.n * p.bs + rem)));
        }
    } else if p.op == 4u {
        // embed: y = x < R ? t : 0 over s*N*B
        let per = p.s * p.n * p.bs;
        if gid < per {
            let rem = gid % (p.n * p.bs);
            let i = (gid - rem) / (p.n * p.bs);
            let x = rem / p.bs;
            let val = select(vec4<u32>(0u, 0u, 0u, 0u),
                             ld4(p.t_off + i * p.r_coords * p.bs + rem),
                             x < p.r_coords);
            st4(p.y_off + i * p.n * p.bs + rem, val);
        }
    } else if p.op == 5u {
        // reduce: A = XOR_i ybuf[i]
        if gid < p.n * p.bs {
            var acc = vec4<u32>(0u, 0u, 0u, 0u);
            for (var i = 0u; i < p.s; i++) {
                acc = acc ^ ld4(p.y_off + i * p.n * p.bs + gid);
            }
            st4(p.a_off + gid, acc);
        }
    } else if p.op == 7u {
        // handoff: copy the accumulator region to the input region so the
        // next step chains on-device (WebGPU forbids same-buffer
        // copy_buffer_to_buffer, so this is a plain word copy)
        let words = p.n * p.bs * 4u;
        if gid < words {
            vec[p.v_off * 4u + gid] = vec[p.a_off * 4u + gid];
        }
    } else if p.op == 8u {
        // reconstruction accumulate (Rung D): one round of the batched
        // block Horner, order j = p.e0.  Z (the v region at ncand stride)
        // ^= V0 (the u region, ENGINE block stride p.bs) times the A_j
        // coefficient block (coef tail at p.ac_off, layout
        // A[j*B*ncand + c*B + bp]).  Each thread owns one Z slot; V0/A
        // are read-only in this dispatch.
        let per = p.n * p.ncand;
        if gid < per {
            let c = gid % p.ncand;
            let x = gid / p.ncand;
            var acc = ld4(p.v_off + x * p.ncand + c);
            for (var bp = 0u; bp < p.bs; bp++) {
                let aa = coefld(p.ac_off + p.e0 * p.bs * p.ncand + c * p.bs + bp);
                let vv = ld4(p.u_off + x * p.bs + bp);
                acc = acc ^ wmul(vv, aa);
            }
            st4(p.v_off + x * p.ncand + c, acc);
        }
    } else if p.op == 6u {
        // dots: one workgroup per (c1, c2); 64-lane tree reduction.
        // XOR is associative, so the result is independent of the
        // reduction tree -- bit-identical to the CPU dot loop.
        let pair = wgid.x;                 // workgroup index = c1*B + c2
        let c1 = pair / p.bs;
        let c2 = pair % p.bs;
        var part = vec4<u32>(0u, 0u, 0u, 0u);
        var x = lid;
        let stride = 64u;
        while x < p.n {
            part = part ^ wmul(ld4(p.u_off + x * p.bs + c1), ld4(p.v_off + x * p.bs + c2));
            x = x + stride;
        }
        dot_shared[lid] = part;
        workgroupBarrier();
        var t = 32u;
        while t > 0u {
            if lid < t { dot_shared[lid] = dot_shared[lid] ^ dot_shared[lid + t]; }
            workgroupBarrier();
            t = t >> 1u;
        }
        if lid == 0u {
            st4(p.d_off + c1 * p.bs + c2, dot_shared[0]);
        }
    }
}
