//! Turning the checkpoint's sums into realization factors, with a standard
//! error per factor, and filling the contexts that heads-up solves don't
//! measure.

use super::record::Record;
use crate::realize::{Context, Position, PotKind, Realization, Role};
use holdem::NUM_CLASSES;
use std::fmt;

/// Where a context's factors come from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// Measured by this many solves.
    Measured(usize),
    /// Copied from the heads-up context with the same pot, role and
    /// position (multiway pots aren't measured).
    HeadsUp(Context),
    /// The average of the in-position and out-of-position contexts with the
    /// same pot and role.
    MiddleOf(Context, Context),
    /// Copied from the single-raised pot's caller with the same position
    /// (limped pots don't occur in the measured game).
    Limped(Context),
    /// Not measured and nothing to copy from: factor 1.
    Neutral,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::Measured(n) => write!(f, "measured by {n} solves"),
            Source::HeadsUp(c) => write!(f, "as [{}]", c.name()),
            Source::MiddleOf(a, b) => write!(f, "average of [{}] and [{}]", a.name(), b.name()),
            Source::Limped(c) => write!(f, "as [{}]", c.name()),
            Source::Neutral => write!(f, "not measured: 1"),
        }
    }
}

/// Measured realization factors.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub realization: Realization,
    /// Standard error of each measured factor (a jackknife over the solves,
    /// leaving out one `(spot, flop)` at a time); 0 where not measured or
    /// with a single solve.
    pub stderr: Vec<[f32; NUM_CLASSES]>,
    /// Per context, where its factors come from.
    pub source: Vec<Source>,
}

/// Per class, one solve's `Σ w·V` and `Σ w·V0`.
type SolveSums = ([f64; NUM_CLASSES], [f64; NUM_CLASSES]);

/// A ratio of sums, or `None` without a positive denominator.
fn ratio(num: f64, den: f64) -> Option<f64> {
    (den > 0.0 && num.is_finite()).then(|| (num / den).max(0.0))
}

/// The factors measured by `records`: for each context and class,
/// `Σ w·V / Σ w·V0` over the solves and combos, where `V` is a combo's value
/// with the real bet sizes, `V0` its check-down value and `w` its weight. A
/// class without a positive check-down sum in a measured context gets the
/// ratio of the context's other classes. Unmeasured contexts are filled as
/// [`Source`] describes.
pub fn aggregate(records: &[Record]) -> Result<Calibration, String> {
    let mut solves: Vec<Vec<SolveSums>> = vec![Vec::new(); Context::COUNT];
    for (i, r) in records.iter().enumerate() {
        r.validate().map_err(|e| format!("record {} ({} on {}): {e}", i + 1, r.spot, r.flop))?;
        for (p, ctx) in r.parsed_contexts()?.into_iter().enumerate() {
            let mut a = [0f64; NUM_CLASSES];
            let mut b = [0f64; NUM_CLASSES];
            for c in 0..NUM_CLASSES {
                a[c] = r.weight[p][c] * r.value[p][c];
                b[c] = r.weight[p][c] * r.check_down[p][c];
            }
            solves[ctx.index()].push((a, b));
        }
    }
    let mut realization = Realization::neutral();
    let mut stderr = vec![[0f32; NUM_CLASSES]; Context::COUNT];
    let mut source = vec![Source::Neutral; Context::COUNT];
    for (x, s) in solves.iter().enumerate().filter(|(_, s)| !s.is_empty()) {
        let value: [f64; NUM_CLASSES] = std::array::from_fn(|c| s.iter().map(|(a, _)| a[c]).sum());
        let check_down: [f64; NUM_CLASSES] = std::array::from_fn(|c| s.iter().map(|(_, b)| b[c]).sum());
        let with_check_down = (0..NUM_CLASSES).filter(|&c| check_down[c] > 0.0);
        let (a, b) = with_check_down.fold((0.0, 0.0), |(a, b), c| (a + value[c], b + check_down[c]));
        let mean = ratio(a, b).unwrap_or(1.0);
        let mut factors = [0f32; NUM_CLASSES];
        for c in 0..NUM_CLASSES {
            factors[c] = ratio(value[c], check_down[c]).unwrap_or(mean) as f32;
            stderr[x][c] = jackknife(s, c, value[c], check_down[c]) as f32;
        }
        realization.set(Context::from_index(x), factors);
        source[x] = Source::Measured(s.len());
    }
    fill_unmeasured(&mut realization, &mut source);
    Ok(Calibration { realization, stderr, source })
}

/// Jackknife standard error of the ratio `A / B` of class `c`, leaving out
/// one solve at a time (0 with fewer than two solves).
fn jackknife(solves: &[SolveSums], c: usize, a: f64, b: f64) -> f64 {
    let n = solves.len();
    let Some(full) = ratio(a, b) else { return 0.0 };
    if n < 2 {
        return 0.0;
    }
    let leave_out: Vec<f64> =
        solves.iter().map(|(ai, bi)| ratio(a - ai[c], b - bi[c]).unwrap_or(full)).collect();
    let mean = leave_out.iter().sum::<f64>() / n as f64;
    let var = leave_out.iter().map(|r| (r - mean).powi(2)).sum::<f64>() * (n - 1) as f64 / n as f64;
    var.sqrt()
}

/// Fills contexts without solves: heads-up middle positions (which don't
/// occur) as the average of in and out of position, multiway contexts as
/// their heads-up counterparts, limped pots as the single-raised pot's
/// caller. Measured contexts are kept.
fn fill_unmeasured(r: &mut Realization, source: &mut [Source]) {
    let unmeasured = |source: &[Source], ctx: Context| !matches!(source[ctx.index()], Source::Measured(_));
    let raised = [PotKind::Raised, PotKind::ThreeBet, PotKind::FourBet];
    for pot in raised {
        for role in [Role::Aggressor, Role::Caller] {
            let at = |position, multiway| Context { pot, role, position, multiway };
            let (ip, oop) = (at(Position::InPosition, false), at(Position::OutOfPosition, false));
            let middle = at(Position::Middle, false);
            if unmeasured(source, middle) {
                let (a, b) = (r.factors(ip), r.factors(oop));
                let avg: [f32; NUM_CLASSES] = std::array::from_fn(|c| (a[c] + b[c]) / 2.0);
                r.set(middle, avg);
                source[middle.index()] = Source::MiddleOf(ip, oop);
            }
            for position in [Position::InPosition, Position::OutOfPosition, Position::Middle] {
                let multi = at(position, true);
                if unmeasured(source, multi) {
                    r.set(multi, *r.factors(at(position, false)));
                    source[multi.index()] = Source::HeadsUp(at(position, false));
                }
            }
        }
    }
    for ctx in Context::all().filter(|c| c.pot == PotKind::Limped) {
        if unmeasured(source, ctx) {
            let from = Context { pot: PotKind::Raised, role: Role::Caller, ..ctx };
            r.set(ctx, *r.factors(from));
            source[ctx.index()] = Source::Limped(from);
        }
    }
}
