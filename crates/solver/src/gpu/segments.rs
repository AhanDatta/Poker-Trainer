//! Splitting a tree too big for the device's scratch memory into
//! **segments**: the top of the tree (segment 0), and groups of **chunks**
//! (the subtrees under the root street's chance nodes: turn subtrees on a
//! flop, river subtrees on a turn), each group run in turn in one shared
//! scratch region. Shared by the heads-up and three-player plans, which
//! differ only in their reach-slot spaces (one per traverser heads-up, one
//! per traverser and opponent with three players) and in what their scratch
//! costs.
//!
//! Chunks are disjoint subtrees, so running them group by group computes
//! exactly what one pass over the whole tree computes. Value rows `[0,
//! top_rows)` belong to the top segment and the chunk roots, `[top_rows,
//! num_rows)` to the group running; each slot space's slots split the same
//! way (the plans lay those out per traverser).

use super::shared::{ensure, to_u32, NONE};
use super::{invalid, GpuError};
use std::ops::Range;

/// How much device memory the scratch buffers may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scratch {
    /// Never split the tree (tests).
    #[cfg_attr(not(test), allow(dead_code))]
    Unlimited,
    /// Split only if the plan plus the whole tree's scratch would need more
    /// than this many bytes, into groups that fit.
    Budget(u64),
    /// Always split (where the tree has chunks), with at most this many
    /// nodes per group (tests).
    #[cfg_attr(not(test), allow(dead_code))]
    ForceGroups { max_nodes: u32 },
}

/// The split of the tree into segments, and every node's value row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Segmentation {
    pub segment_of: Vec<u32>,
    pub num_segments: u32,
    pub row: Vec<u32>,
    pub top_rows: u32,
    pub num_rows: u32,
}

/// A tree's shape as the split sees it.
pub(crate) struct Shape<'a> {
    pub num_nodes: usize,
    /// Node `i`'s children (consecutive; children have larger indices).
    pub children: &'a dyn Fn(usize) -> Result<Range<usize>, GpuError>,
    /// The chunk roots: the children of the root street's chance nodes.
    pub roots: Vec<usize>,
    /// Per node, per slot space: the reach slots it hands out to its children.
    pub fresh: &'a dyn Fn(usize) -> Result<Vec<u64>, GpuError>,
    pub spaces: usize,
}

/// Splits `shape` under `scratch`; `bytes(rows, slots)` is the device memory
/// a pass with that many value rows and reach slots per space needs (plan
/// included).
pub(crate) fn segment(
    shape: &Shape<'_>,
    scratch: Scratch,
    bytes: &dyn Fn(u64, &[u64]) -> u64,
) -> Result<Segmentation, GpuError> {
    let nn = shape.num_nodes;
    let add = |acc: &mut Vec<u64>, i: usize| -> Result<(), GpuError> {
        let f = (shape.fresh)(i)?;
        ensure(f.len() == shape.spaces, || "slot spaces".into())?;
        for (a, x) in acc.iter_mut().zip(f) {
            *a += x;
        }
        Ok(())
    };
    let mut whole_slots = vec![1u64; shape.spaces];
    for i in 0..nn {
        add(&mut whole_slots, i)?;
    }
    let whole = || -> Result<Segmentation, GpuError> {
        let rows = (0..nn).map(|i| to_u32(i, "rows")).collect::<Result<Vec<u32>, _>>()?;
        Ok(Segmentation {
            segment_of: vec![0; nn],
            num_segments: 1,
            row: rows,
            top_rows: to_u32(nn, "rows")?,
            num_rows: to_u32(nn, "rows")?,
        })
    };
    let roots = &shape.roots;
    match scratch {
        Scratch::Unlimited => return whole(),
        Scratch::Budget(b) => {
            if roots.is_empty() || bytes(nn as u64, &whole_slots) <= b {
                return whole();
            }
        }
        Scratch::ForceGroups { .. } => {
            if roots.is_empty() {
                return whole();
            }
        }
    }

    // Each chunk's nodes (the subtree under its root).
    let mut chunk_of = vec![NONE; nn];
    let mut chunk_nodes: Vec<Vec<usize>> = Vec::with_capacity(roots.len());
    for (k, &root) in roots.iter().enumerate() {
        let k32 = to_u32(k, "chunks")?;
        let mut members = Vec::new();
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            let entry = chunk_of.get_mut(i).ok_or_else(|| invalid("chunk node out of range"))?;
            ensure(*entry == NONE, || format!("node {i} in two chunks"))?;
            *entry = k32;
            members.push(i);
            stack.extend((shape.children)(i)?);
        }
        members.sort_unstable();
        chunk_nodes.push(members);
    }

    // Top-region needs: rows for the top nodes and the chunk roots, slots
    // handed out by the top nodes.
    let top_nodes: Vec<usize> = (0..nn).filter(|&i| chunk_of.get(i) == Some(&NONE)).collect();
    let top_rows = (top_nodes.len() + roots.len()) as u64;
    let mut top_slots = vec![1u64; shape.spaces];
    for &i in &top_nodes {
        add(&mut top_slots, i)?;
    }
    // Per chunk: rows for its non-root nodes, slots its nodes hand out.
    let mut chunk_rows = Vec::with_capacity(roots.len());
    let mut chunk_slots = Vec::with_capacity(roots.len());
    for members in &chunk_nodes {
        chunk_rows.push(members.len() as u64 - 1);
        let mut s = vec![0u64; shape.spaces];
        for &i in members {
            add(&mut s, i)?;
        }
        chunk_slots.push(s);
    }
    let group_bytes = |rows: u64, slots: &[u64]| {
        let total: Vec<u64> = top_slots.iter().zip(slots).map(|(t, s)| t + s).collect();
        bytes(top_rows + rows, &total)
    };

    // Greedy grouping of consecutive chunks.
    let mut groups: Vec<Range<usize>> = Vec::new();
    let (mut rows, mut slots, mut members, mut start) = (0u64, vec![0u64; shape.spaces], 0u64, 0usize);
    for (k, (&cr, cs)) in chunk_rows.iter().zip(&chunk_slots).enumerate() {
        let size = cr + 1;
        let with: Vec<u64> = slots.iter().zip(cs).map(|(a, b)| a + b).collect();
        let fits_with = match scratch {
            Scratch::Budget(b) => group_bytes(rows + cr, &with) <= b,
            Scratch::ForceGroups { max_nodes } => members + size <= u64::from(max_nodes),
            Scratch::Unlimited => true,
        };
        if k > start && !fits_with {
            groups.push(start..k);
            (rows, members, start) = (0, 0, k);
            slots = vec![0; shape.spaces];
        }
        if let Scratch::Budget(b) = scratch {
            let needed = group_bytes(cr, cs);
            if needed > b {
                return Err(GpuError::OutOfMemory { needed, available: b });
            }
        }
        rows += cr;
        for (a, b) in slots.iter_mut().zip(cs) {
            *a += b;
        }
        members += size;
    }
    groups.push(start..roots.len());

    // Segments and rows.
    let mut segment_of = vec![0u32; nn];
    let mut row = vec![NONE; nn];
    let mut next_row = 0u32;
    for &i in top_nodes.iter().chain(roots) {
        *row.get_mut(i).ok_or_else(|| invalid("row out of range"))? = next_row;
        next_row += 1;
    }
    let top_rows = next_row;
    let mut num_rows = top_rows;
    for (g, chunks) in groups.iter().enumerate() {
        let seg = to_u32(g + 1, "segments")?;
        let mut cursor = top_rows;
        for k in chunks.clone() {
            let members = chunk_nodes.get(k).ok_or_else(|| invalid("chunk out of range"))?;
            let root = *roots.get(k).ok_or_else(|| invalid("chunk out of range"))?;
            for &i in members {
                *segment_of.get_mut(i).ok_or_else(|| invalid("node out of range"))? = seg;
                if i != root {
                    *row.get_mut(i).ok_or_else(|| invalid("row out of range"))? = cursor;
                    cursor = cursor.checked_add(1).ok_or_else(|| invalid("too many rows"))?;
                }
            }
        }
        num_rows = num_rows.max(cursor);
    }
    Ok(Segmentation { segment_of, num_segments: to_u32(groups.len() + 1, "segments")?, row, top_rows, num_rows })
}

/// Checks a segmentation against the tree: the root in the top segment,
/// segments changing only from the top to a group at a child of a chance
/// node (`is_chance`), value rows in the right region and unique where they
/// must be.
pub(crate) fn validate(
    seg: &Segmentation,
    num_nodes: usize,
    parent: &[u32],
    is_chance: &dyn Fn(usize) -> Result<bool, GpuError>,
) -> Result<(), GpuError> {
    let at = |v: &[u32], i: usize, what: &str| v.get(i).copied().ok_or_else(|| invalid(format!("{what}[{i}] out of range")));
    ensure(seg.segment_of.len() == num_nodes && seg.row.len() == num_nodes, || "segment and row arrays".into())?;
    ensure(seg.num_segments >= 1, || "no segments".into())?;
    ensure(seg.segment_of.first() == Some(&0), || "the root must be in the top segment".into())?;
    ensure(seg.segment_of.iter().all(|&s| s < seg.num_segments), || "segment out of range".into())?;
    ensure(seg.top_rows <= seg.num_rows, || "row regions".into())?;
    ensure(parent.len() == num_nodes, || "parent array".into())?;
    let mut top_used = vec![false; seg.top_rows as usize];
    let mut group_rows: Vec<(u32, u32)> = Vec::new();
    for i in 0..num_nodes {
        let s = at(&seg.segment_of, i, "segment_of")?;
        let row = at(&seg.row, i, "row")?;
        ensure(row < seg.num_rows, || format!("node {i}: row {row} out of range"))?;
        let pa = at(parent, i, "parent")?;
        let is_chunk_root = pa != NONE && at(&seg.segment_of, pa as usize, "segment_of")? != s;
        if is_chunk_root {
            ensure(at(&seg.segment_of, pa as usize, "segment_of")? == 0 && s >= 1, || {
                format!("node {i}: segments change other than below the top")
            })?;
            ensure(is_chance(pa as usize)?, || format!("chunk root {i}: parent isn't a chance node"))?;
        }
        if s == 0 || is_chunk_root {
            let used =
                top_used.get_mut(row as usize).ok_or_else(|| invalid(format!("node {i}: row not in the top region")))?;
            ensure(!*used, || format!("node {i}: top row {row} shared"))?;
            *used = true;
        } else {
            ensure(row >= seg.top_rows, || format!("node {i}: group node in the top row region"))?;
            group_rows.push((s, row));
        }
    }
    group_rows.sort_unstable();
    ensure(group_rows.windows(2).all(|w| w.first() != w.get(1)), || "a row is shared within a group".into())
}
