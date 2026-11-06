//! Encodings, limits and validation helpers shared by the heads-up and
//! three-player plans (and, where noted, by the kernels).

use super::{invalid, GpuError};
use std::ops::Range;

/// "No index" marker, shared with the kernels (`solver::kNone`).
pub(crate) const NONE: u32 = u32::MAX;

/// Cards in the deck.
pub(crate) const CARDS: usize = 52;

/// Largest normalized payoff and range weight the plan accepts: with at most
/// 1326 hands, every product the terminal kernels form stays below 1e22,
/// far inside f32's (and so double-single's) range of 3.4e38.
pub(crate) const MAX_PAYOFF: f64 = 1e12;
pub(crate) const MAX_WEIGHT: f32 = 1e6;

/// Most hands a player can have (every combo; `kMaxHands` in
/// multi_terminal.cu, whose showdowns keep a fixed number per thread).
pub(crate) const MAX_HANDS: u32 = 1326;

/// Most actions per decision node (the 16-bit kernels track alive children
/// in a 32-bit mask; `kMaxActions` in traverse_compressed.cu).
pub(crate) const MAX_ACTIONS: u32 = 32;

/// Discriminants of the kernels' `enum class Mode`.
pub(crate) const MODE_CFR: u32 = 0;
pub(crate) const MODE_BEST_RESPONSE: u32 = 1;
pub(crate) const MODE_AVERAGE: u32 = 2;

/// Dynamic shared memory the showdown kernel may use without an opt-in
/// (48 KiB minus a margin for its static CUB storage).
pub(crate) const SHOWDOWN_SHARED_LIMIT: u32 = 44 * 1024;

/// Bytes of dynamic shared memory the showdown kernel needs for opponent
/// lists of up to `max_opp` hands: two f64 prefix arrays and the f32 weights.
pub(crate) fn showdown_shared_bytes(max_opp: u32) -> u64 {
    (3 * u64::from(max_opp) + 1 + CARDS as u64) * 8 + u64::from(max_opp) * 4
}

/// `[start, start + count)` of some list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Span {
    pub start: u32,
    pub count: u32,
}

impl Span {
    pub(crate) fn range(self) -> Range<usize> {
        self.start as usize..self.start as usize + self.count as usize
    }

    pub(crate) fn end(self) -> u64 {
        u64::from(self.start) + u64::from(self.count)
    }

    pub(crate) fn of(start: usize, end: usize, what: &str) -> Result<Span, GpuError> {
        Ok(Span { start: to_u32(start, what)?, count: to_u32(end.saturating_sub(start), what)? })
    }
}

pub(crate) fn to_u32(x: usize, what: &str) -> Result<u32, GpuError> {
    u32::try_from(x).map_err(|_| invalid(format!("{what} ({x}) exceeds the kernels' u32 indexing")))
}

pub(crate) fn ensure(cond: bool, msg: impl FnOnce() -> String) -> Result<(), GpuError> {
    if cond {
        Ok(())
    } else {
        Err(GpuError::InvalidInput(msg()))
    }
}

/// `v[i]`, or an error naming `what`.
pub(crate) fn at<T: Copy>(v: &[T], i: usize, what: &str) -> Result<T, GpuError> {
    v.get(i).copied().ok_or_else(|| invalid(format!("{what}[{i}] out of range")))
}

/// `c1 | c2 << 8`.
pub(crate) fn pack_cards(c1: u8, c2: u8) -> u32 {
    u32::from(c1) | u32::from(c2) << 8
}

pub(crate) fn unpack_cards(packed: u32) -> (usize, usize) {
    ((packed & 0xFF) as usize, (packed >> 8 & 0xFF) as usize)
}

/// `off` is a 53-entry offset array starting at 0, non-decreasing and
/// ending at `total`.
pub(crate) fn check_offsets(off: &[u32], total: usize, what: &str) -> Result<(), GpuError> {
    ensure(off.len() == CARDS + 1, || format!("{what}: {} offsets", off.len()))?;
    ensure(off.first() == Some(&0) && off.windows(2).all(|w| w.first() <= w.get(1)), || {
        format!("{what}: offsets out of order")
    })?;
    ensure(off.last().is_some_and(|&e| e as usize == total), || {
        format!("{what}: offsets don't end at {total}")
    })
}
