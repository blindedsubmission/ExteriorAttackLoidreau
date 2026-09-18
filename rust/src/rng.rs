//! Deterministic PRNG streams (xorshift64) shared by CPU and GPU paths.
//!
//! Every stochastic choice in the pipeline (Wiedemann start vectors,
//! conditioning weights, instance sampling) comes from one of these
//! streams so that a run is reproducible from its seed alone, and so the
//! CPU and GPU engines can be compared bit-for-bit at equal seed.

#[derive(Clone)]
pub struct Xorshift64 {
    pub s: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Xorshift64 {
        Xorshift64 { s: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1 }
    }

    /// Reference-engine seeding: `s = seed * 0x9E37... + 1`.  This exact
    /// stream is what makes the scalar engine bit-compatible with the
    /// development engines at equal seed (m <= 63).
    pub fn new_mf(seed: u64) -> Xorshift64 {
        Xorshift64 { s: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1) }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        self.s
    }

    /// Random field element of GF(2^m), m <= 128 (both words filled).
    pub fn next_fe(&mut self, mask: crate::field::Fe) -> crate::field::Fe {
        let lo = self.next_u64() & mask.0;
        let hi = self.next_u64() & mask.1;
        (lo, hi)
    }

    /// Random nonzero field element (redraw on zero).
    #[allow(dead_code)] // research/profiling helper
    pub fn next_nonzero_fe(&mut self, mask: crate::field::Fe) -> crate::field::Fe {
        loop {
            let x = self.next_fe(mask);
            if x != (0, 0) {
                return x;
            }
        }
    }
}

/// splitmix64 finalizer -- used where an independent stream with a long
/// period is preferable (weight generation in the reference engine).
#[derive(Clone)]
    #[allow(dead_code)] // research/profiling helper
pub struct SplitMix {
    pub s: u64,
}

impl SplitMix {
    #[allow(dead_code)] // research/profiling helper
    pub fn new(seed: u64) -> SplitMix {
        SplitMix { s: seed }
    }
    #[allow(dead_code)] // research/profiling helper
    pub fn next(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Xorshift64::new(7);
        let mut b = Xorshift64::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn nonzero_stream_never_yields_zero() {
        let mut r = Xorshift64::new(99);
        for _ in 0..1000 {
            let x = r.next_nonzero_fe((u64::MAX, u64::MAX));
            assert_ne!(x, (0, 0));
        }
    }
}
