//! Storage of cumulative regrets and strategy sums.
//!
//! Every decision node owns a contiguous `num_actions * num_hands` block in
//! two big arrays. In compressed mode the values are stored as 16-bit
//! integers with one `f32` scale per node (regrets: `i16`, strategy sums:
//! `u16`), halving memory and memory traffic.
//!
//! The solver visits each node from exactly one task at a time (the tree
//! structure guarantees disjointness), so blocks are handed out through raw
//! pointers instead of locks.

use std::cell::Cell;

/// A fixed-size buffer whose disjoint parts may be accessed from several
/// threads at once.
pub(crate) struct SharedBuf<T> {
    data: Box<[T]>,
    ptr: *mut T,
}

unsafe impl<T: Send> Send for SharedBuf<T> {}
unsafe impl<T: Send> Sync for SharedBuf<T> {}

impl<T: Copy + Default + Send + Sync> SharedBuf<T> {
    pub fn new(len: usize) -> SharedBuf<T> {
        // Written in parallel instead of `vec![0; len]`: a zeroed allocation
        // is mapped lazily, and since the solver reads each block before
        // writing it, every page would first map the shared zero page and
        // then take a copy-on-write fault with a TLB shootdown across all
        // cores (several seconds per flop solve). Writing first faults the
        // pages in directly, spread over the threads.
        use rayon::prelude::*;
        let mut v: Vec<T> = Vec::with_capacity(len);
        v.spare_capacity_mut().par_chunks_mut(1 << 16).for_each(|c| {
            for x in c {
                x.write(T::default());
            }
        });
        // SAFETY: all `len` elements were initialized above.
        unsafe { v.set_len(len) };
        let mut data = v.into_boxed_slice();
        let ptr = data.as_mut_ptr();
        SharedBuf { data, ptr }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// # Safety
    /// No other live mutable slice may overlap `off..off + len`.
    #[inline]
    pub unsafe fn get(&self, off: usize, len: usize) -> &[T] {
        debug_assert!(off + len <= self.data.len());
        std::slice::from_raw_parts(self.ptr.add(off), len)
    }

    /// # Safety
    /// No other live slice may overlap `off..off + len`.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub unsafe fn get_mut(&self, off: usize, len: usize) -> &mut [T] {
        debug_assert!(off + len <= self.data.len());
        std::slice::from_raw_parts_mut(self.ptr.add(off), len)
    }
}

/// Whether to store regrets/strategies as `f32` or compressed 16-bit values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageMode {
    F32,
    Compressed,
    /// Compressed when the `f32` tree would exceed 1 GB.
    Auto,
}

pub(crate) enum Storage {
    F32 {
        regrets: SharedBuf<f32>,
        strat: SharedBuf<f32>,
    },
    Compressed {
        regrets: SharedBuf<i16>,
        strat: SharedBuf<u16>,
        regret_scale: SharedBuf<f32>,
        strat_scale: SharedBuf<f32>,
    },
}

/// Location of a node's block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Block {
    pub offset: usize,
    pub len: usize,
    pub slot: usize,
}

thread_local! {
    static DECODE: Cell<Vec<f32>> = const { Cell::new(Vec::new()) };
}

/// `max(|x|)` over a slice, written so that it vectorizes.
#[inline]
pub(crate) fn max_abs(v: &[f32]) -> f32 {
    let mut m = [0f32; 8];
    let chunks = v.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            m[i] = m[i].max(c[i].abs());
        }
    }
    let mut r = m.iter().fold(0f32, |a, &b| a.max(b));
    for &x in rest {
        r = r.max(x.abs());
    }
    r
}

fn with_decode_buf<R>(len: usize, f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    let mut buf = DECODE.with(|c| c.take());
    // Every element is overwritten by the decoder: no need to zero.
    if buf.len() < len {
        buf.resize(len, 0.0);
    } else {
        buf.truncate(len);
    }
    let r = f(&mut buf);
    DECODE.with(|c| c.set(buf));
    r
}

impl Storage {
    pub fn new(len: usize, slots: usize, compressed: bool) -> Storage {
        if compressed {
            Storage::Compressed {
                regrets: SharedBuf::new(len),
                strat: SharedBuf::new(len),
                regret_scale: SharedBuf::new(slots),
                strat_scale: SharedBuf::new(slots),
            }
        } else {
            Storage::F32 { regrets: SharedBuf::new(len), strat: SharedBuf::new(len) }
        }
    }

    pub fn is_compressed(&self) -> bool {
        matches!(self, Storage::Compressed { .. })
    }

    /// Copies all values from a storage of the same shape (tests).
    #[cfg(test)]
    pub(crate) fn copy_from(&self, other: &Storage) {
        fn copy<T: Copy + Default + Send + Sync>(dst: &SharedBuf<T>, src: &SharedBuf<T>) {
            assert_eq!(dst.len(), src.len());
            // SAFETY: no other access is live during the copy.
            unsafe { dst.get_mut(0, dst.len()).copy_from_slice(src.get(0, src.len())) }
        }
        match (self, other) {
            (Storage::F32 { regrets, strat }, Storage::F32 { regrets: r2, strat: s2 }) => {
                copy(regrets, r2);
                copy(strat, s2);
            }
            (
                Storage::Compressed { regrets, strat, regret_scale, strat_scale },
                Storage::Compressed { regrets: r2, strat: s2, regret_scale: rs2, strat_scale: ss2 },
            ) => {
                copy(regrets, r2);
                copy(strat, s2);
                copy(regret_scale, rs2);
                copy(strat_scale, ss2);
            }
            _ => panic!("storage modes differ"),
        }
    }

    pub fn bytes(&self) -> u64 {
        match self {
            Storage::F32 { regrets, .. } => regrets.len() as u64 * 8,
            Storage::Compressed { regrets, regret_scale, .. } => {
                regrets.len() as u64 * 4 + regret_scale.len() as u64 * 8
            }
        }
    }

    /// Bytes per stored `(action, hand)` value pair.
    pub fn bytes_per_value(compressed: bool) -> u64 {
        if compressed {
            4
        } else {
            8
        }
    }

    /// Copies the node's cumulative regrets into `out`.
    ///
    /// # Safety
    /// The node's block must not be mutated concurrently.
    pub unsafe fn load_regrets(&self, b: Block, out: &mut [f32]) {
        match self {
            Storage::F32 { regrets, .. } => out.copy_from_slice(regrets.get(b.offset, b.len)),
            Storage::Compressed { regrets, regret_scale, .. } => {
                let scale = regret_scale.get(b.slot, 1)[0] / i16::MAX as f32;
                for (o, &q) in out.iter_mut().zip(regrets.get(b.offset, b.len)) {
                    *o = q as f32 * scale;
                }
            }
        }
    }

    /// Copies the node's strategy sums into `out`.
    ///
    /// # Safety
    /// The node's block must not be mutated concurrently.
    pub unsafe fn load_strat(&self, b: Block, out: &mut [f32]) {
        match self {
            Storage::F32 { strat, .. } => out.copy_from_slice(strat.get(b.offset, b.len)),
            Storage::Compressed { strat, strat_scale, .. } => {
                let scale = strat_scale.get(b.slot, 1)[0] / u16::MAX as f32;
                for (o, &q) in out.iter_mut().zip(strat.get(b.offset, b.len)) {
                    *o = q as f32 * scale;
                }
            }
        }
    }

    /// Applies `f` to the node's regrets (as `f32`) and stores the result.
    ///
    /// # Safety
    /// The caller must have exclusive access to the node's block.
    pub unsafe fn update_regrets(&self, b: Block, f: impl FnOnce(&mut [f32])) {
        match self {
            Storage::F32 { regrets, .. } => f(regrets.get_mut(b.offset, b.len)),
            Storage::Compressed { regrets, regret_scale, .. } => with_decode_buf(b.len, |buf| {
                let scale_slot = regret_scale.get_mut(b.slot, 1);
                let q = regrets.get_mut(b.offset, b.len);
                let scale = scale_slot[0] / i16::MAX as f32;
                for (o, &x) in buf.iter_mut().zip(q.iter()) {
                    *o = x as f32 * scale;
                }
                f(buf);
                let max = max_abs(buf);
                debug_assert!(max.is_finite(), "non-finite regrets");
                scale_slot[0] = max;
                if max > 0.0 {
                    let inv = i16::MAX as f32 / max;
                    // Round half away from zero without a libm call (keeps
                    // the loop vectorizable); `as` saturates.
                    for (x, &v) in q.iter_mut().zip(buf.iter()) {
                        let y = v * inv;
                        *x = (y + 0.5f32.copysign(y)) as i16;
                    }
                } else {
                    q.fill(0);
                }
            }),
        }
    }

    /// Applies `f` to the node's strategy sums (as `f32`) and stores the
    /// result. Values must stay non-negative.
    ///
    /// # Safety
    /// The caller must have exclusive access to the node's block.
    pub unsafe fn update_strat(&self, b: Block, f: impl FnOnce(&mut [f32])) {
        match self {
            Storage::F32 { strat, .. } => f(strat.get_mut(b.offset, b.len)),
            Storage::Compressed { strat, strat_scale, .. } => with_decode_buf(b.len, |buf| {
                let scale_slot = strat_scale.get_mut(b.slot, 1);
                let q = strat.get_mut(b.offset, b.len);
                let scale = scale_slot[0] / u16::MAX as f32;
                for (o, &x) in buf.iter_mut().zip(q.iter()) {
                    *o = x as f32 * scale;
                }
                f(buf);
                let max = max_abs(buf);
                debug_assert!(max.is_finite() && buf.iter().all(|&x| x >= -1e-6), "invalid strategy sums");
                scale_slot[0] = max;
                if max > 0.0 {
                    let inv = u16::MAX as f32 / max;
                    for (x, &v) in q.iter_mut().zip(buf.iter()) {
                        *x = (v.max(0.0) * inv + 0.5) as u16;
                    }
                } else {
                    q.fill(0);
                }
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_roundtrip() {
        for compressed in [false, true] {
            let s = Storage::new(8, 1, compressed);
            let b = Block { offset: 0, len: 8, slot: 0 };
            let vals = [1.0f32, -2.0, 0.5, 0.0, 3.0, -0.25, 7.0, -7.0];
            unsafe {
                s.update_regrets(b, |r| r.copy_from_slice(&vals));
                let mut out = [0f32; 8];
                s.load_regrets(b, &mut out);
                for (a, b) in out.iter().zip(vals) {
                    assert!((a - b).abs() <= 7.0 / 32767.0 + 1e-6, "{a} vs {b}");
                }
                s.update_strat(b, |r| r.copy_from_slice(&vals.map(f32::abs)));
                s.load_strat(b, &mut out);
                for (a, b) in out.iter().zip(vals) {
                    assert!((a - b.abs()).abs() <= 7.0 / 65535.0 + 1e-6);
                }
            }
        }
    }
}
