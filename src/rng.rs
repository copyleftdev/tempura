//! Deterministic RNG infrastructure for reproducible annealing.
//!
//! All stochastic processes in Tempura flow through this trait.
//! Deterministic seeding guarantees: same seed → same sequence → same result.
//!
//! # Safety invariants (H-08)
//! - `next_u64` must be deterministic given the same seed
//! - `next_f64` must produce values in [0.0, 1.0)
//! - Different seeds must produce uncorrelated sequences
//! - Period must exceed 2^64 for practical chain lengths

/// Trait for deterministic pseudo-random number generators.
///
/// Implementations must be:
/// - Deterministic: same seed produces same sequence
/// - Fast: < 2ns per draw on modern hardware
/// - High-quality: pass `BigCrush` statistical test suite
pub trait Rng: Clone {
    /// Create a new RNG from the given seed.
    fn from_seed(seed: u64) -> Self;
    /// Generate the next 64-bit unsigned integer.
    fn next_u64(&mut self) -> u64;

    /// Uniform f64 in [0.0, 1.0) with 53-bit resolution.
    ///
    /// Uses the standard conversion: discard low 11 bits of u64,
    /// divide by 2^53. This avoids the subtle non-uniformity of
    /// naive `u64 as f64 / u64::MAX as f64`.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform f64 in [0.0, 1.0) returned as -ln(u), an Exp(1) variate.
    ///
    /// Used by log-domain acceptance (H-10c) to avoid `exp()` in the hot loop:
    ///   accept if -ln(u) > ΔE/T  ⟺  u < exp(-ΔE/T)
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn next_exp1(&mut self) -> f64 {
        -self.next_f64().max(f64::MIN_POSITIVE).ln()
    }
}

/// `SplitMix64` — used exclusively for seed scrambling.
///
/// Ensures that sequential user seeds (0, 1, 2, ...) produce uncorrelated
/// initial states in the primary RNG.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) const fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^ (state >> 31)
}

// ---------------------------------------------------------------------------
// Xoshiro256++ — Default RNG
// ---------------------------------------------------------------------------

/// Xoshiro256++ PRNG by Blackman & Vigna (2018).
///
/// - Period: 2^256 - 1
/// - Passes `BigCrush`, `PractRand`
/// - ~1ns per draw
/// - 32 bytes state
///
/// This is Tempura's default RNG, balancing quality, speed, and state size.
#[derive(Clone, Debug)]
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Rng for Xoshiro256PlusPlus {
    fn from_seed(seed: u64) -> Self {
        // Scramble user seed through splitmix64 to fill 256-bit state.
        // This ensures sequential seeds produce uncorrelated states.
        let mut z = seed;
        let s0 = {
            z = splitmix64(z);
            z
        };
        let s1 = {
            z = splitmix64(z);
            z
        };
        let s2 = {
            z = splitmix64(z);
            z
        };
        let s3 = {
            z = splitmix64(z);
            z
        };
        // Safety: splitmix64 of any non-degenerate seed will produce
        // at least one non-zero value. Belt-and-suspenders check:
        let s = [s0, s1, s2, s3];
        debug_assert!(s.iter().any(|&x| x != 0), "all-zero state is invalid");
        Self { s }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3])).rotate_left(23).wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }
}

// ---------------------------------------------------------------------------
// PCG-64 (XSL-RR variant)
// ---------------------------------------------------------------------------

/// PCG-XSL-RR-128/64 by Melissa O'Neill (2014).
///
/// - Period: 2^128
/// - Passes `BigCrush`, `PractRand`
/// - ~1ns per draw
/// - 16 bytes state + 8 bytes increment
///
/// Offers an alternative to Xoshiro256++ with different failure modes.
/// Used in H-08 to validate RNG independence.
#[derive(Clone, Debug)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Rng for Pcg64 {
    fn from_seed(seed: u64) -> Self {
        let s0 = splitmix64(seed);
        let s1 = splitmix64(s0);
        // Increment must be odd for full period
        let inc = (u128::from(s1) << 64 | u128::from(s0)) | 1;
        let mut rng = Self { state: 0, inc };
        // Advance state twice to escape zero neighborhood
        rng.state = rng.state.wrapping_add(inc);
        let _ = rng.next_u64();
        rng.state = rng.state.wrapping_add(u128::from(splitmix64(seed.wrapping_add(1))));
        let _ = rng.next_u64();
        rng
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let old_state = self.state;
        // LCG step
        self.state = old_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(self.inc);
        // XSL-RR output function
        let xsl = ((old_state >> 64) ^ old_state) as u64;
        let rot = (old_state >> 122) as u32;
        xsl.rotate_right(rot)
    }
}

// ---------------------------------------------------------------------------
// Default type alias
// ---------------------------------------------------------------------------

/// Default RNG for Tempura. Xoshiro256++ unless overridden.
pub type DefaultRng = Xoshiro256PlusPlus;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xoshiro_deterministic() {
        let mut a = Xoshiro256PlusPlus::from_seed(42);
        let mut b = Xoshiro256PlusPlus::from_seed(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn pcg_deterministic() {
        let mut a = Pcg64::from_seed(42);
        let mut b = Pcg64::from_seed(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn xoshiro_different_seeds_diverge() {
        let mut a = Xoshiro256PlusPlus::from_seed(0);
        let mut b = Xoshiro256PlusPlus::from_seed(1);
        let mut same = 0u32;
        for _ in 0..1000 {
            if a.next_u64() == b.next_u64() {
                same += 1;
            }
        }
        assert!(same < 5, "seeds 0 and 1 produced {} collisions", same);
    }

    #[test]
    fn f64_in_range() {
        let mut rng = Xoshiro256PlusPlus::from_seed(123);
        for _ in 0..100_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of range: {}", v);
        }
    }

    #[test]
    fn exp1_positive() {
        let mut rng = Xoshiro256PlusPlus::from_seed(456);
        for _ in 0..100_000 {
            let v = rng.next_exp1();
            assert!(v > 0.0, "exp1 must be positive: {}", v);
            assert!(v.is_finite(), "exp1 must be finite: {}", v);
        }
    }
}
