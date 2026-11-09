//! A small, fast random number generator (xoshiro256++ seeded through
//! splitmix64), so solves are reproducible from a seed on every platform.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    /// An independent generator for stream `stream` of `seed` (one per
    /// thread and epoch).
    pub fn stream(seed: u64, stream: u64) -> Rng {
        Rng::new(seed ^ stream.wrapping_mul(0xD6E8_FEB8_6659_FD93).rotate_left(29))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `0..n` (`n > 0`), without modulo bias.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        // Lemire's method: multiply-shift, rejecting the biased low products.
        let mut m = (self.next_u64() >> 32) * n as u64;
        if (m as u32) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u32) < threshold {
                m = (self.next_u64() >> 32) * n as u64;
            }
        }
        (m >> 32) as u32
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproducible_and_seed_dependent() {
        let draw = |seed| {
            let mut r = Rng::new(seed);
            (0..8).map(|_| r.next_u64()).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
        let mut a = Rng::stream(7, 0);
        let mut b = Rng::stream(7, 1);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn below_is_in_range_and_roughly_uniform() {
        let mut r = Rng::new(1);
        for n in [1u32, 2, 3, 7, 52, 1000] {
            let mut counts = vec![0u32; n as usize];
            let draws = 20_000 * n.min(60);
            for _ in 0..draws {
                counts[r.below(n) as usize] += 1;
            }
            let expected = draws as f64 / n as f64;
            for (i, &c) in counts.iter().enumerate() {
                assert!((c as f64 - expected).abs() < 6.0 * expected.sqrt() + 1.0, "n={n} i={i} c={c}");
            }
        }
    }

    #[test]
    fn unit_is_in_the_half_open_interval() {
        let mut r = Rng::new(3);
        let mut sum = 0.0f64;
        for _ in 0..100_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
            sum += u as f64;
        }
        assert!((sum / 100_000.0 - 0.5).abs() < 0.01);
    }
}
