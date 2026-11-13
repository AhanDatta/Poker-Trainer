//! The text format of realization factors: a data file per game, written by
//! the calibration and read when solving.
//!
//! ```text
//! # comments start with '#'
//! includes_rake false
//! [raised aggressor ip headsup]
//! 1.0213 0.9876 ...   (169 factors in grid order, 13 per line: AA AKs ... 22)
//! ```
//!
//! Every one of the [`Context::COUNT`] contexts appears exactly once.

use super::{Context, Position, PotKind, Realization, Role};
use holdem::NUM_CLASSES;

const POTS: [(PotKind, &str); 4] = [
    (PotKind::Limped, "limped"),
    (PotKind::Raised, "raised"),
    (PotKind::ThreeBet, "3bet"),
    (PotKind::FourBet, "4bet"),
];
const ROLES: [(Role, &str); 2] = [(Role::Aggressor, "aggressor"), (Role::Caller, "caller")];
const POSITIONS: [(Position, &str); 3] =
    [(Position::InPosition, "ip"), (Position::OutOfPosition, "oop"), (Position::Middle, "middle")];
const WAYS: [(bool, &str); 2] = [(false, "headsup"), (true, "multiway")];

fn name_of<T: PartialEq + Copy>(table: &[(T, &'static str)], value: T) -> &'static str {
    table.iter().find(|(v, _)| *v == value).map(|(_, n)| *n).expect("every value is named")
}

fn value_of<T: Copy>(table: &[(T, &str)], name: &str) -> Option<T> {
    table.iter().find(|(_, n)| *n == name).map(|(v, _)| *v)
}

impl Context {
    /// The context's name in data files, e.g. `"raised aggressor ip headsup"`.
    pub fn name(self) -> String {
        format!(
            "{} {} {} {}",
            name_of(&POTS, self.pot),
            name_of(&ROLES, self.role),
            name_of(&POSITIONS, self.position),
            name_of(&WAYS, self.multiway)
        )
    }

    /// Reads a [`Context::name`].
    pub fn parse_name(name: &str) -> Option<Context> {
        let words: Vec<&str> = name.split_whitespace().collect();
        let [pot, role, position, ways] = words[..] else { return None };
        Some(Context {
            pot: value_of(&POTS, pot)?,
            role: value_of(&ROLES, role)?,
            position: value_of(&POSITIONS, position)?,
            multiway: value_of(&WAYS, ways)?,
        })
    }
}

impl Realization {
    /// Writes the factors in the format [`Realization::parse`] reads, after
    /// `header` (each of its lines becomes a comment).
    pub fn to_text(&self, header: &str) -> String {
        let mut out = String::new();
        for line in header.lines() {
            out.push_str(&if line.trim().is_empty() { "#\n".to_string() } else { format!("# {line}\n") });
        }
        out.push_str(&format!("includes_rake {}\n", self.includes_rake));
        for ctx in Context::all() {
            out.push_str(&format!("[{}]\n", ctx.name()));
            for row in self.factors(ctx).chunks(13) {
                let row: Vec<String> = row.iter().map(|f| format!("{f:.4}")).collect();
                out.push_str(&row.join(" "));
                out.push('\n');
            }
        }
        out
    }

    /// Reads factors written by [`Realization::to_text`]. Errors (with the
    /// line number) on anything else: unknown or repeated contexts, missing
    /// ones, a wrong number of factors, or factors that are negative or not
    /// numbers.
    pub fn parse(text: &str) -> Result<Realization, String> {
        let mut out = Realization::neutral();
        let mut includes_rake = None;
        let mut seen = [false; Context::COUNT];
        // The section being read: its context, first line and factors so far.
        let mut section: Option<(Context, usize, Vec<f32>)> = None;
        let finish = |section: Option<(Context, usize, Vec<f32>)>, out: &mut Realization| match section {
            Some((ctx, line, values)) if values.len() != NUM_CLASSES => Err(format!(
                "line {line}: [{}] has {} factors, expected {NUM_CLASSES}",
                ctx.name(),
                values.len()
            )),
            Some((ctx, _, values)) => {
                out.factors[ctx.index()] = values.try_into().expect("length checked");
                Ok(())
            }
            None => Ok(()),
        };
        for (i, raw) in text.lines().enumerate() {
            let n = i + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("includes_rake") {
                if includes_rake.is_some() || section.is_some() || seen.iter().any(|&s| s) {
                    return Err(format!("line {n}: includes_rake must come once, before the factors"));
                }
                includes_rake = Some(match rest.trim() {
                    "true" => true,
                    "false" => false,
                    other => {
                        return Err(format!("line {n}: includes_rake must be true or false, not {other:?}"))
                    }
                });
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                finish(section.take(), &mut out)?;
                let ctx =
                    Context::parse_name(name).ok_or_else(|| format!("line {n}: unknown context [{name}]"))?;
                if std::mem::replace(&mut seen[ctx.index()], true) {
                    return Err(format!("line {n}: [{name}] appears twice"));
                }
                section = Some((ctx, n, Vec::with_capacity(NUM_CLASSES)));
                continue;
            }
            let Some((_, _, values)) = section.as_mut() else {
                return Err(format!("line {n}: factors before any [context]"));
            };
            for word in line.split_whitespace() {
                let f: f32 = word.parse().map_err(|_| format!("line {n}: {word:?} is not a number"))?;
                if !f.is_finite() || f < 0.0 {
                    return Err(format!("line {n}: factor {word} must be finite and at least 0"));
                }
                values.push(f);
            }
        }
        finish(section, &mut out)?;
        out.includes_rake = includes_rake.ok_or("missing the includes_rake line")?;
        if let Some(missing) = seen.iter().position(|&s| !s) {
            return Err(format!("context [{}] is missing", Context::from_index(missing).name()));
        }
        Ok(out)
    }
}
