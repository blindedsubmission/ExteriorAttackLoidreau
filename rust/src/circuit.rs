//! The substitute-and-discard circuit: matrix-free application of the
//! compound matrices `C_lam`([`I_k` | `D_i`]) and their transposes.
//!
//! # High-level intuition
//!
//! The exterior attack needs kernel vectors of the *stacked compound
//! matrix* E = (`C_lam`([`I_k` | `D_i`]))_i, an sR x N matrix.  Materializing it
//! costs multiple GB at attack scale.  The key observation (paper:
//! "A cheaper circuit: substitute and discard one coordinate") is that
//! `C_lam`([`I_k` | `D_i`]) = `C_lam(P)` `C_lam(Q_i)` where `Q_i` is a product of
//! k*r commuting transvections and `C_lam(P)` is a coordinate projection.
//! Applying the transvections one coordinate at a time, *discarding* every
//! exterior coordinate that contains an already-substituted basis vector,
//! applies the whole compound as r in-place stages on a single
//! length-N buffer:
//!
//! ```text
//! stage b (position j = k+b), for each lam-subset I of [0, j):
//!     x[I] ^= sum_{a in I, a < k} D_i[a][b] * x[(I \ {a}) | {j}]
//! ```
//!
//! running b = r-1 .. 0 forward, and b = 0 .. r-1 with src/dst exchanged
//! for the transpose.  Per block this costs
//! `T_and = k * (C(n-1, lam) - C(k-1, lam))` multiply-adds versus
//! `R * N` for a dense block matvec (a 200x saving at attack shapes), in
//! `O(N + kr)` memory.  Nothing outside the `D_i` table is stored.
//!
//! # Colex order is what makes it work
//!
//! Coordinates are indexed by the *colex* rank of the lambda-subset.  In
//! colex order the subsets of [0, j) are exactly ranks [0, C(j, lam)) and
//! subsets containing j sit at ranks >= C(j, lam): within a stage the
//! forward writes and the transposed writes each occupy one side of a
//! clean rank cut, disjoint from the reads of the same stage.  Stages
//! therefore update the buffer in place, and on the GPU every thread can
//! read while another thread writes -- provided entries that share a
//! write target are *grouped* so a single thread computes the whole sum
//! and stores it once (no atomics, no races; see `GroupedSchedule`).
//!
//! Conventions (shared with the project's reference implementations):
//! block i uses `D_i` = sigma^{-i}(D) (inverse Frobenius, matching
//! `stacked_compound`'s [-i] shift), coefficients are addressed as
//! `cidx = a*r + b` into a `k*r` table, and the kernel output is mapped
//! back to standard lex Pluecker coordinates by the caller.

use crate::field::{Fe, Field};
use crate::subset::{colex_rank, subsets_lex, Binomial};

/// Flat per-stage update entries of the circuit.
#[derive(Clone)]
pub struct Stage {
    /// Stage's substituted position j = k + b.
    #[allow(dead_code)] // documentary; the fused schedule encodes it implicitly
    pub j: u32,
    /// Entries: (dst colex rank, src colex rank, coefficient index a*r+b).
    pub dst: Vec<u32>,
    pub src: Vec<u32>,
    pub cidx: Vec<u32>,
}

/// Grouped (GPU) schedule for one stage: groups keyed by in-stage write
/// target.  Layout is a flat word array:
/// `[key, count, (cidx, other)*count] per group`, plus per-stage group
/// index ranges and a word-offset table with one trailing sentinel.
#[derive(Clone)]
pub struct GroupedStage {
    pub words: Vec<u32>,
    /// For stage index b: [first group, end group) into the global group
    /// numbering.
    pub stage_groups: Vec<(u32, u32)>,
    /// Word offset of each group's descriptor (len = total groups + 1).
    pub group_off: Vec<u32>,
}

impl GroupedStage {
    /// Build by grouping flat entries (dst, src, cidx) by write target.
    ///
    /// Entries are first stably sorted by key WITHIN each stage: for the
    /// forward circuit the entries of one subset are already consecutive,
    /// but in the transposed circuit the entries sharing a write target
    /// are scattered (same I, different a -> different S).  Without the
    /// sort those entries form several groups with the same key, and on
    /// the GPU two threads would race on y[key] -- precisely the
    /// "scatter race" of the reference implementation's bug list.  With
    /// it, every in-stage write target is owned by exactly one group and
    /// one thread computes its whole sum.
    pub fn from_flat(dst: &[u32], src: &[u32], cidx: &[u32], stage_starts: &[u32]) -> GroupedStage {
        let mut words = Vec::new();
        let mut group_off = vec![0u32];
        let mut stage_groups = Vec::new();
        for s in 0..stage_starts.len() - 1 {
            let lo = stage_starts[s] as usize;
            let hi = stage_starts[s + 1] as usize;
            let first_group = (group_off.len() - 1) as u32;
            // stable sort this stage's entries by write target
            let mut idx: Vec<usize> = (lo..hi).collect();
            idx.sort_by_key(|&x| dst[x]);
            let mut i = 0usize;
            while i < idx.len() {
                let key = dst[idx[i]];
                words.push(key);
                let count_pos = words.len();
                words.push(0);
                let mut cnt = 0u32;
                while i < idx.len() && dst[idx[i]] == key {
                    words.push(cidx[idx[i]]);
                    words.push(src[idx[i]]);
                    cnt += 1;
                    i += 1;
                }
                words[count_pos] = cnt;
                group_off.push(words.len() as u32);
            }
            stage_groups.push((first_group, (group_off.len() - 1) as u32));
        }
        GroupedStage { words, stage_groups, group_off }
    }

    /// Verify that expanding the groups reproduces the flat schedule
    /// entry-for-entry (cheap; run on every build).
    pub fn check_against(&self, dst: &[u32], src: &[u32], cidx: &[u32], stage_starts: &[u32]) {
        let mut out = Vec::with_capacity(dst.len());
        let n_groups = self.group_off.len() - 1;
        let mut gi = 0usize;
        for s in 0..stage_starts.len() - 1 {
            let (g0, g1) = self.stage_groups[s];
            assert_eq!(gi, g0 as usize, "stage group ranges must tile");
            for g in g0..g1 {
                let off = self.group_off[g as usize] as usize;
                let key = self.words[off];
                let cnt = self.words[off + 1];
                for q in 0..cnt {
                    out.push((key, self.words[off + 3 + 2 * q as usize], self.words[off + 2 + 2 * q as usize]));
                }
            }
            gi = g1 as usize;
        }
        assert_eq!(gi, n_groups, "all groups consumed");
        assert_eq!(out.len(), dst.len(), "entry count mismatch");
        // entry sets must agree per stage; grouped order within a group is
        // the flat order, so compare elementwise after per-stage sorting.
        let mut a: Vec<(u32, u32, u32)> = out.clone();
        let mut b: Vec<(u32, u32, u32)> = dst.iter().zip(src).zip(cidx).map(|((&d, &s), &c)| (d, s, c)).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "grouped schedule differs from flat schedule");
    }
}

/// The static circuit for one (n, k, lam): schedule tables independent of
/// the actual matrix D and of the stack height s.
#[derive(Clone)]
pub struct Circuit {
    pub n: usize,
    pub k: usize,
    pub lam: usize,
    pub r: usize,
    pub n_coords: u32,
    pub k_coords: u32,
    pub binom: Binomial,
    /// Stages in *build* order b = 0..r-1 (ascending).  The forward circuit
    /// runs them in reverse; the transpose in build order.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // read by the GPU schedule builder
    pub stages: Vec<Stage>,
    /// `stage_starts`[b] = index of stage b's first flat entry.
    pub stage_starts: Vec<u32>,
    /// Fused host-side schedule: every flat entry as an interleaved
    /// (cidx, src, dst) u32 triple, stages concatenated in build order
    /// (stage b = triples [`stage_starts`[b], `stage_starts`[b+1])).  One
    /// sequential array, one bounds check per entry -- the CPU H-step's
    /// hot loop; the grouped tables below stay the GPU's.
    pub flat: Vec<u32>,
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // the GPU builder's tables
    pub fwd: GroupedStage,
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // the GPU builder's tables
    pub trn: GroupedStage,
    /// Total multiply-adds per block application.
    pub t_and: usize,
}

impl Circuit {
    pub fn new(n: usize, k: usize, lam: usize) -> Circuit {
        assert!(lam >= 1 && k >= lam && n > k, "need 1 <= lam <= k < n");
        let r = n - k;
        let binom = Binomial::new(n + 1, lam);
        let n_coords = binom.c(n, lam) as u32;
        let k_coords = binom.c(k, lam) as u32;

        let mut stages = Vec::with_capacity(r);
        let mut stage_starts = vec![0u32];
        let mut t_and = 0usize;
        for b in 0..r {
            let j = k + b;
            let mut dst = Vec::new();
            let mut src = Vec::new();
            let mut cidx = Vec::new();
            for i in subsets_lex(j, lam) {
                let d = colex_rank(&i, &binom);
                for t in 0..lam {
                    if i[t] < k {
                        // S = (I \ {a}) | {j}, kept sorted
                        let mut s = i.clone();
                        s.remove(t);
                        s.push(j);
                        s.sort_unstable();
                        dst.push(d);
                        src.push(colex_rank(&s, &binom));
                        cidx.push((i[t] * r + b) as u32);
                    }
                }
            }
            t_and += dst.len();
            // cumulative flat-entry offsets: stage_starts[b]..stage_starts[b+1]
            // is stage b's entry range in the concatenated arrays below
            let prev = *stage_starts.last().unwrap();
            stage_starts.push(prev + dst.len() as u32);
            stages.push(Stage { j: j as u32, dst, src, cidx });
        }
        let all_dst: Vec<u32> = stages.iter().flat_map(|s| s.dst.iter().copied()).collect();
        let all_src: Vec<u32> = stages.iter().flat_map(|s| s.src.iter().copied()).collect();
        let all_cidx: Vec<u32> = stages.iter().flat_map(|s| s.cidx.iter().copied()).collect();
        // fused (cidx, src, dst) triples, stages in build order (one
        // sequential array for the CPU hot loops)
        let mut flat = Vec::with_capacity(all_dst.len() * 3);
        for st in &stages {
            for e in 0..st.dst.len() {
                flat.push(st.cidx[e]);
                flat.push(st.src[e]);
                flat.push(st.dst[e]);
            }
        }
        assert_eq!(flat.len(), all_dst.len() * 3, "fused schedule size");
        let fwd = GroupedStage::from_flat(&all_dst, &all_src, &all_cidx, &stage_starts);
        fwd.check_against(&all_dst, &all_src, &all_cidx, &stage_starts);
        let trn = GroupedStage::from_flat(&all_src, &all_dst, &all_cidx, &stage_starts);
        trn.check_against(&all_src, &all_dst, &all_cidx, &stage_starts);
        Circuit {
            n,
            k,
            lam,
            r,
            n_coords,
            k_coords,
            binom,
            stages,
            stage_starts,
            flat,
            fwd,
            trn,
            t_and,
        }
    }

    /// Apply `C_lam`([`I_k` | D]) forward: N coords in, R coords out.
    /// `y` must have length N; the first R entries receive the result.
    // contract: truncates y to R coords on return, hence the Vec
    #[allow(clippy::ptr_arg)]
    #[allow(dead_code)] // dense-stage API: exercised by tests and the GPU builder
    pub fn forward(&self, field: &Field, d: &[Fe], y: &mut Vec<Fe>) {
        for b in (0..self.r).rev() {
            let st = &self.stages[b];
            for e in 0..st.dst.len() {
                let c = d[st.cidx[e] as usize];
                y[st.dst[e] as usize] = field.add(y[st.dst[e] as usize], field.mul(c, y[st.src[e] as usize]));
            }
        }
        y.truncate(self.k_coords as usize);
    }

    /// Apply the transpose: R coords in (padded into a length-N buffer),
    /// N coords out.
    // contract: truncates y to R coords on return, hence the Vec
    #[allow(clippy::ptr_arg)]
    #[allow(dead_code)] // dense-stage API: exercised by tests and the GPU builder
    pub fn transpose(&self, field: &Field, d: &[Fe], y: &mut Vec<Fe>) {
        for b in 0..self.r {
            let st = &self.stages[b];
            for e in 0..st.dst.len() {
                let c = d[st.cidx[e] as usize];
                y[st.src[e] as usize] = field.add(y[st.src[e] as usize], field.mul(c, y[st.dst[e] as usize]));
            }
        }
    }

    /// The systematic matrix [`I_k` | D] as a dense row-major k x (k+r).
    #[allow(dead_code)] // dense-stage API: exercised by tests and the GPU builder
    pub fn systematic(field: &Field, k: usize, d: &[Fe]) -> Vec<Fe> {
        let r = d.len() / k;
        let w = k + r;
        let mut m = vec![field.zero(); k * w];
        for i in 0..k {
            m[i * w + i] = field.one();
        }
        for i in 0..k {
            for b in 0..r {
                m[i * w + k + b] = d[i * r + b];
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense;

    /// Deterministic pseudo-random field element.
    fn rnd_fe(f: &Field, sm: &mut u64) -> Fe {
        *sm ^= *sm << 13;
        *sm ^= *sm >> 7;
        *sm ^= *sm << 17;
        let lo = *sm;
        *sm ^= *sm << 13;
        *sm ^= *sm >> 7;
        *sm ^= *sm << 17;
        (lo & f.mask().0, *sm & f.mask().1)
    }

    #[test]
    fn circuit_matches_dense_compound() {
        // column identity: circuit(e_J) == column J of C_lam([I|D_i]) for
        // every J and several (inverse) Frobenius shifts, plus the bilinear
        // transpose roundtrip <Ex, z> == <x, E^T z>.
        for &(m, n, k, lam) in &[
            (7u32, 8usize, 5usize, 2usize),
            (16, 13, 9, 2),
            (16, 12, 9, 3),
            (8, 11, 7, 1),
            (16, 12, 8, 4),
        ] {
            let field = Field::new(m);
            let circ = Circuit::new(n, k, lam);
            let r = n - k;
            let mut sm = 0x51E9u64;
            for trial in 0..2 {
                let d: Vec<Fe> = (0..k * r).map(|_| rnd_fe(&field, &mut sm)).collect();
                for shift in 0..2.min(r) {
                    let di: Vec<Fe> = d.iter().map(|&x| field.frob(x, -(shift as i32))).collect();
                    let sys = Circuit::systematic(&field, k, &di);
                    let block = dense::compound_matrix(&field, &sys, k, k + r, lam);
                    let nc = circ.n_coords as usize;
                    let rc = circ.k_coords as usize;
                    // dense block: rows = lam-subsets of [0,k) (lex), cols =
                    // lam-subsets of [0,n) (lex).  The circuit runs in colex;
                    // loop the dense side in lex and translate by rank.
                    let rows = subsets_lex(k, lam);
                    let cols = subsets_lex(n, lam);
                    for (lj, jset) in cols.iter().enumerate() {
                        let mut e = vec![field.zero(); nc];
                        e[colex_rank(jset, &circ.binom) as usize] = field.one();
                        circ.forward(&field, &di, &mut e);
                        for (ls, sset) in rows.iter().enumerate() {
                            assert_eq!(
                                e[colex_rank(sset, &circ.binom) as usize],
                                block[ls * nc + lj],
                                "m={m} n={n} k={k} lam={lam} trial={trial} shift={shift} col={lj} row={ls}"
                            );
                        }
                    }
                    let _ = rc;
                }
                // transpose roundtrip
                let x: Vec<Fe> = (0..circ.n_coords).map(|_| rnd_fe(&field, &mut sm)).collect();
                let z: Vec<Fe> = (0..circ.k_coords).map(|_| rnd_fe(&field, &mut sm)).collect();
                let mut fx = x.clone();
                circ.forward(&field, &d, &mut fx);
                let mut ftz = vec![field.zero(); circ.n_coords as usize];
                ftz[..circ.k_coords as usize].copy_from_slice(&z);
                circ.transpose(&field, &d, &mut ftz);
                let lhs: Fe = fx.iter().zip(&z).fold(field.zero(), |acc, (&a, &b)| field.add(acc, field.mul(a, b)));
                let rhs: Fe = x.iter().zip(&ftz).fold(field.zero(), |acc, (&a, &b)| field.add(acc, field.mul(a, b)));
                assert_eq!(lhs, rhs, "roundtrip m={m} n={n} k={k} lam={lam}");
            }
        }
    }
}
