//! Per-thread pool of `f32` buffers, so the recursive traversals don't hit
//! the allocator for every node visit.

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

thread_local! {
    static POOL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

/// A zero-initialized buffer borrowed from the current thread's pool and
/// returned to it on drop.
pub(crate) struct Buf(Vec<f32>);

impl Buf {
    #[allow(dead_code)]
    pub fn zeroed(len: usize) -> Buf {
        let mut v = POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
        v.clear();
        v.resize(len, 0.0);
        Buf(v)
    }

    /// A buffer of `len` elements with unspecified contents (for buffers that
    /// are completely overwritten before being read).
    pub fn uninit(len: usize) -> Buf {
        let mut v = POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
        if v.len() < len {
            v.resize(len, 0.0);
        } else {
            v.truncate(len);
        }
        Buf(v)
    }

    /// A buffer initialized with a copy of `src`.
    pub fn from_slice(src: &[f32]) -> Buf {
        let mut v = POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
        v.clear();
        v.extend_from_slice(src);
        Buf(v)
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.0);
        POOL.with(|p| p.borrow_mut().push(v));
    }
}

impl Deref for Buf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.0
    }
}

impl DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.0
    }
}
