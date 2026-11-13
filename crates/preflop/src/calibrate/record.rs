//! The checkpoint file: one JSON line per finished `(spot, flop)` solve, so a
//! stopped calibration resumes where it was and the factors can be
//! recomputed from the raw sums.

use super::measure::{SeatSums, SolveInfo};
use super::spots::Spot;
use crate::realize::Context;
use holdem::NUM_CLASSES;
use serde::{Deserialize, Serialize};

/// One finished solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The spot's line (`Spot::line`).
    pub spot: String,
    /// The flop, e.g. `"As7d2c"`.
    pub flop: String,
    /// Seats of the players, out of position first.
    pub seats: [usize; 2],
    /// Their contexts (`Context::name`).
    pub contexts: [String; 2],
    pub reach: f64,
    /// Per player, per class: see [`SeatSums`].
    pub weight: [Vec<f64>; 2],
    pub value: [Vec<f64>; 2],
    pub check_down: [Vec<f64>; 2],
    pub combos: [Vec<u32>; 2],
    pub iterations: u32,
    pub exploitability_pct: f64,
    pub solved_on: String,
    #[serde(default)]
    pub gpu_error: Option<String>,
    pub seconds: f64,
}

impl Record {
    pub fn new(spot: &Spot, flop: &str, sums: &[SeatSums; 2], info: &SolveInfo) -> Record {
        Record {
            spot: spot.line.clone(),
            flop: flop.to_string(),
            seats: spot.seats,
            contexts: spot.contexts.map(|c| c.name()),
            reach: spot.reach,
            weight: [sums[0].weight.clone(), sums[1].weight.clone()],
            value: [sums[0].value.clone(), sums[1].value.clone()],
            check_down: [sums[0].check_down.clone(), sums[1].check_down.clone()],
            combos: [sums[0].combos.clone(), sums[1].combos.clone()],
            iterations: info.iterations,
            exploitability_pct: info.exploitability_pct,
            solved_on: info.solved_on.clone(),
            gpu_error: info.gpu_error.clone(),
            seconds: info.seconds,
        }
    }

    /// What identifies the solve: the spot's line and the flop.
    pub fn key(&self) -> (String, String) {
        (self.spot.clone(), self.flop.clone())
    }

    /// The players' contexts, or an error if a name is unknown.
    pub fn parsed_contexts(&self) -> Result<[Context; 2], String> {
        let parse =
            |name: &String| Context::parse_name(name).ok_or_else(|| format!("unknown context {name:?}"));
        Ok([parse(&self.contexts[0])?, parse(&self.contexts[1])?])
    }

    /// Checks what the aggregation relies on: known contexts and 169 finite
    /// sums per player, with weights not negative. (Check-down values can be
    /// slightly off, even negative, with sampled runouts, which mis-weight
    /// hands that hold a sampled card; the ratio of sums copes.)
    pub fn validate(&self) -> Result<(), String> {
        self.parsed_contexts()?;
        for p in 0..2 {
            let lengths =
                [self.weight[p].len(), self.value[p].len(), self.check_down[p].len(), self.combos[p].len()];
            if lengths.iter().any(|&n| n != NUM_CLASSES) {
                return Err(format!("player {p}: {lengths:?} values, expected {NUM_CLASSES} each"));
            }
            let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
            if !finite(&self.value[p]) || !finite(&self.weight[p]) || !finite(&self.check_down[p]) {
                return Err(format!("player {p}: a sum is not a number"));
            }
            if self.weight[p].iter().any(|&x| x < 0.0) {
                return Err(format!("player {p}: a weight is negative"));
            }
        }
        if !self.reach.is_finite() || self.reach < 0.0 {
            return Err(format!("reach {}", self.reach));
        }
        Ok(())
    }

    /// The record as one line of JSON (no newline).
    pub fn to_line(&self) -> String {
        serde_json::to_string(self).expect("records serialize")
    }
}

/// Reads a checkpoint. A last line without its newline is a write that was
/// cut off; it's dropped, and the second value says how many bytes of
/// `text` are complete (truncate the file to that before appending). Any
/// other bad line is an error.
pub fn parse_checkpoint(text: &str) -> Result<(Vec<Record>, usize), String> {
    let complete = text.rfind('\n').map_or(0, |i| i + 1);
    let mut out = Vec::new();
    for (i, line) in text[..complete].lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: Record =
            serde_json::from_str(line).map_err(|e| format!("checkpoint line {}: {e}", i + 1))?;
        record.validate().map_err(|e| format!("checkpoint line {}: {e}", i + 1))?;
        out.push(record);
    }
    Ok((out, complete))
}
