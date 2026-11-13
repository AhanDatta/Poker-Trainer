//! Calibrating the realization factors with the heads-up postflop solver.
//!
//! For a sample of heads-up pots that see a flop when everyone plays by the
//! built-in charts ([`spots`]), and a sample of flops for each, two postflop
//! solves are run with the chart ranges ([`measure`]): one with the real bet
//! sizes and one with checks only. A hand class's factor in a context is
//! what its combos win with betting over what they win checking down,
//! summed over the solves with each combo weighted by how often it gets
//! there ([`aggregate`]). Every finished solve goes to a checkpoint file
//! ([`record`]), so a long run can stop and resume.

pub mod aggregate;
pub mod measure;
pub mod record;
pub mod spots;

pub use aggregate::{aggregate, Calibration, Source};
pub use measure::{measure, Chance, PostflopSetup};
pub use record::{parse_checkpoint, Record};
pub use spots::{heads_up_flops, pick, Spot, SpotType};

use crate::rng::Rng;
use holdem::{Card, CardSet};
use std::collections::HashSet;

/// The first `n` flops of a spot's flop sequence: random, but the same for
/// the same `seed` and line, and a longer run starts with the same flops.
pub fn flops(seed: u64, line: &str, n: usize) -> Vec<[Card; 3]> {
    // FNV-1a of the line picks the spot's stream.
    let hash =
        line.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    let mut rng = Rng::stream(seed, hash);
    (0..n)
        .map(|_| {
            let mut used = CardSet::EMPTY;
            [(); 3].map(|_| loop {
                let card = Card::from_index(rng.below(52) as u8);
                if !used.contains(card) {
                    used.insert(card);
                    break card;
                }
            })
        })
        .collect()
}

/// A flop's name in records, e.g. `"As7d2c"`.
pub fn flop_name(flop: &[Card]) -> String {
    holdem::card::cards_to_string(flop).replace(' ', "")
}

/// One solve to run: a spot (an index into the spot list) on a flop.
#[derive(Clone, Debug, PartialEq)]
pub struct Job {
    pub spot: usize,
    pub flop: [Card; 3],
}

/// The solves for `flops_per_spot` flops of every spot, ordered so that a
/// run stopped early has covered every spot about equally: the first flop
/// of every spot, then the second, and so on.
pub fn plan(spots: &[Spot], flops_per_spot: usize, seed: u64) -> Vec<Job> {
    let per_spot: Vec<Vec<[Card; 3]>> = spots.iter().map(|s| flops(seed, &s.line, flops_per_spot)).collect();
    (0..flops_per_spot)
        .flat_map(|i| per_spot.iter().enumerate().map(move |(spot, f)| Job { spot, flop: f[i] }))
        .collect()
}

/// The jobs of `jobs` that `done` (finished records) doesn't have yet.
pub fn pending(spots: &[Spot], jobs: Vec<Job>, done: &[Record]) -> Vec<Job> {
    let done: HashSet<(String, String)> = done.iter().map(Record::key).collect();
    jobs.into_iter().filter(|j| !done.contains(&(spots[j.spot].line.clone(), flop_name(&j.flop)))).collect()
}

/// Runs `jobs` in order, handing each finished solve's record to `save`
/// (which writes the checkpoint; an error from it stops the run). A solve
/// that fails, or gives a record that doesn't validate, is reported to
/// `failed` and skipped. Returns how many records were saved.
pub fn run(
    spots: &[Spot],
    jobs: &[Job],
    setup: &PostflopSetup,
    mut save: impl FnMut(&Record) -> Result<(), String>,
    mut failed: impl FnMut(&Job, &str),
) -> Result<usize, String> {
    let mut saved = 0;
    for job in jobs {
        let spot = &spots[job.spot];
        let record = measure(spot, &job.flop, setup)
            .map(|(sums, info)| Record::new(spot, &flop_name(&job.flop), &sums, &info))
            .and_then(|r| r.validate().map(|()| r));
        match record {
            Ok(record) => {
                save(&record)?;
                saved += 1;
            }
            Err(why) => failed(job, &why),
        }
    }
    Ok(saved)
}

#[cfg(test)]
mod tests;
