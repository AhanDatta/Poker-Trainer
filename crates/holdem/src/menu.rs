//! The bet-size menu.
//!
//! [`BetMenu::actions`] turns a betting [`Situation`] into the list of
//! actions a player may choose from. The live game *and* the solver's tree
//! builder both call this function, so a hand played at the table always stays
//! on the solved game tree.

use crate::card::ParseError;
use crate::game::{Action, Chips, Street};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// A bet or raise size.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum BetSize {
    /// Fraction of the pot (measured after calling), e.g. `0.75` for 75%.
    Pot(f64),
    /// Multiple of the current bet (raises only), e.g. `3.0` for "raise to 3x".
    Mult(f64),
    /// All-in.
    AllIn,
}

impl FromStr for BetSize {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let t = s.trim().to_ascii_lowercase();
        let err = || ParseError(format!("invalid bet size {s:?} (use e.g. \"75%\", \"3x\", \"allin\")"));
        if matches!(t.as_str(), "a" | "allin" | "all-in" | "all in" | "shove") {
            return Ok(BetSize::AllIn);
        }
        if let Some(p) = t.strip_suffix('%') {
            let v: f64 = p.trim().parse().map_err(|_| err())?;
            if v <= 0.0 {
                return Err(err());
            }
            return Ok(BetSize::Pot(v / 100.0));
        }
        if let Some(m) = t.strip_suffix('x') {
            let v: f64 = m.trim().parse().map_err(|_| err())?;
            if v <= 1.0 {
                return Err(err());
            }
            return Ok(BetSize::Mult(v));
        }
        Err(err())
    }
}

impl TryFrom<String> for BetSize {
    type Error = ParseError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<BetSize> for String {
    fn from(b: BetSize) -> String {
        b.to_string()
    }
}

impl fmt::Display for BetSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BetSize::Pot(p) => write!(f, "{}%", fmt_num(p * 100.0)),
            BetSize::Mult(m) => write!(f, "{}x", fmt_num(*m)),
            BetSize::AllIn => f.write_str("allin"),
        }
    }
}

fn fmt_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v}")
    }
}

/// Bet and raise sizes for one postflop street.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreetSizes {
    pub bet: Vec<BetSize>,
    pub raise: Vec<BetSize>,
}

impl StreetSizes {
    pub fn new(bet: &[&str], raise: &[&str]) -> StreetSizes {
        StreetSizes {
            bet: bet.iter().map(|s| s.parse().expect("valid bet size")).collect(),
            raise: raise.iter().map(|s| s.parse().expect("valid bet size")).collect(),
        }
    }
}

/// Postflop sizing rules.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PostflopMenu {
    pub flop: StreetSizes,
    pub turn: StreetSizes,
    pub river: StreetSizes,
    /// Maximum number of bets + raises on one street; once reached, only
    /// fold/call remain.
    pub max_raises: u32,
    /// Offer all-in as an extra option when shoving is at most this many pots
    /// (0 disables).
    pub add_allin_threshold: f64,
    /// A bet/raise becomes all-in when, after being called, the remaining
    /// stack would be at most this fraction of the pot.
    pub force_allin_threshold: f64,
    /// Merge sizes that are too close: a smaller size with pot ratio `y` is
    /// dropped next to a larger kept size `x` when `(1+x)/(1+y) < 1+threshold`.
    pub merge_threshold: f64,
}

impl Default for PostflopMenu {
    fn default() -> Self {
        PostflopMenu {
            flop: StreetSizes::new(&["33%", "75%"], &["3x"]),
            turn: StreetSizes::new(&["50%", "100%"], &["3x"]),
            river: StreetSizes::new(&["50%", "100%"], &["3x"]),
            max_raises: 3,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merge_threshold: 0.1,
        }
    }
}

impl PostflopMenu {
    pub fn street(&self, street: Street) -> &StreetSizes {
        match street {
            Street::Flop | Street::Preflop => &self.flop,
            Street::Turn => &self.turn,
            Street::River => &self.river,
        }
    }

    pub fn street_mut(&mut self, street: Street) -> &mut StreetSizes {
        match street {
            Street::Flop | Street::Preflop => &mut self.flop,
            Street::Turn => &mut self.turn,
            Street::River => &mut self.river,
        }
    }
}

/// Preflop sizing rules. Each raise level has exactly one size (plus all-in),
/// which keeps preflop charts simple: fold / call / raise / all-in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreflopMenu {
    /// First raise (open) when nobody limped, in big blinds.
    pub open_bb: f64,
    /// Raise over one limper, in big blinds.
    pub iso_bb: f64,
    /// Extra big blinds per additional limper.
    pub iso_per_limper_bb: f64,
    /// Re-raise sizes as multiples of the current bet: `[3-bet, 4-bet, ...]`.
    /// Beyond the listed levels only all-in is available.
    pub reraise_mult: Vec<f64>,
    /// Whether open-limping (calling the big blind) is allowed.
    pub allow_limp: bool,
}

impl Default for PreflopMenu {
    fn default() -> Self {
        PreflopMenu {
            open_bb: 2.5,
            iso_bb: 3.5,
            iso_per_limper_bb: 1.0,
            reraise_mult: vec![3.5, 2.3],
            allow_limp: true,
        }
    }
}

/// Complete bet-size menu.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BetMenu {
    pub preflop: PreflopMenu,
    pub postflop: PostflopMenu,
}

/// Everything the menu needs to know about the current decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Situation {
    pub street: Street,
    /// Total pot including all bets made on this street.
    pub pot: Chips,
    /// Chips needed to call (already capped by the player's stack).
    pub to_call: Chips,
    /// Highest street commitment ("raise-to" level) so far.
    pub current_bet: Chips,
    /// Largest meaningful raise-to amount: the player's all-in, capped by the
    /// largest amount any opponent could match.
    pub max_to: Chips,
    /// Smallest legal (full) raise-to amount.
    pub min_raise_to: Chips,
    /// Voluntary bets/raises so far on this street (the big blind isn't counted).
    pub num_raises: u32,
    /// Preflop limpers so far.
    pub limpers: u32,
    pub big_blind: Chips,
}

impl Situation {
    pub fn can_raise(&self) -> bool {
        self.max_to > self.current_bet
    }
}

impl BetMenu {
    /// The actions available in `sit`, ordered: fold, check/call, then
    /// raises by increasing amount.
    pub fn actions(&self, sit: &Situation) -> Vec<Action> {
        match sit.street {
            Street::Preflop => preflop_actions(sit, &self.preflop, &self.postflop),
            _ => postflop_actions(sit, &self.postflop),
        }
    }
}

/// Postflop action list (see [`BetMenu::actions`]).
pub fn postflop_actions(sit: &Situation, menu: &PostflopMenu) -> Vec<Action> {
    postflop_categorized(sit, menu).into_iter().map(|(a, _)| a).collect()
}

/// Kind of a postflop action for frequency statistics: what the player
/// chose among the menu's options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Category {
    Fold,
    Check,
    Call,
    /// A bet or raise from this menu size (when it isn't all-in).
    Size(BetSize),
    AllIn,
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Category::Fold => f.write_str("Fold"),
            Category::Check => f.write_str("Check"),
            Category::Call => f.write_str("Call"),
            Category::Size(s) => write!(f, "{s}"),
            Category::AllIn => f.write_str("All-in"),
        }
    }
}

/// Postflop actions, each with the [`Category`] it belongs to: a bet or
/// raise is labeled by the menu size that produced it, or all-in when it
/// puts the player (or the shortest opponent) all-in. When several sizes
/// round to the same amount the first one listed names it; a size merged
/// into a larger one disappears.
pub fn postflop_categorized(sit: &Situation, menu: &PostflopMenu) -> Vec<(Action, Category)> {
    let mut out = Vec::with_capacity(6);
    if sit.to_call > 0 {
        out.push((Action::Fold, Category::Fold));
        out.push((Action::Call, Category::Call));
    } else {
        out.push((Action::Check, Category::Check));
    }
    if !sit.can_raise() || sit.num_raises >= menu.max_raises {
        return out;
    }
    let sizes = menu.street(sit.street);
    let list = if sit.current_bet == 0 { &sizes.bet } else { &sizes.raise };
    let pot_after_call = (sit.pot + sit.to_call) as f64;
    // (raise-to, index of the size in `list`; `list.len()` = the extra all-in)
    let mut amounts = Vec::with_capacity(list.len() + 1);
    for (i, size) in list.iter().enumerate() {
        let to = match *size {
            BetSize::Pot(f) => sit.current_bet + (f * pot_after_call).round() as Chips,
            BetSize::Mult(m) => {
                if sit.current_bet == 0 {
                    continue;
                }
                (m * sit.current_bet as f64).round() as Chips
            }
            BetSize::AllIn => sit.max_to,
        };
        amounts.push((to, i));
    }
    if menu.add_allin_threshold > 0.0
        && ((sit.max_to - sit.current_bet) as f64) <= menu.add_allin_threshold * pot_after_call
    {
        amounts.push((sit.max_to, list.len()));
    }
    let amounts = finalize_tagged(sit, amounts, menu.force_allin_threshold, menu.merge_threshold);
    out.extend(amounts.into_iter().map(|(to, i)| {
        let cat = if to >= sit.max_to {
            Category::AllIn
        } else {
            list.get(i).map_or(Category::AllIn, |&s| Category::Size(s))
        };
        (Action::Raise(to), cat)
    }));
    out
}

/// Preflop action list (see [`BetMenu::actions`]).
pub fn preflop_actions(sit: &Situation, pre: &PreflopMenu, post: &PostflopMenu) -> Vec<Action> {
    let mut out = Vec::with_capacity(4);
    if sit.to_call > 0 {
        out.push(Action::Fold);
        // Facing only the blinds, calling is a limp.
        if sit.num_raises > 0 || pre.allow_limp {
            out.push(Action::Call);
        }
    } else {
        out.push(Action::Check);
    }
    if !sit.can_raise() {
        return out;
    }
    let bb = sit.big_blind as f64;
    let raise_to = if sit.num_raises == 0 {
        if sit.limpers == 0 {
            Some((pre.open_bb * bb).round() as Chips)
        } else {
            let extra = (sit.limpers - 1) as f64 * pre.iso_per_limper_bb;
            Some(((pre.iso_bb + extra) * bb).round() as Chips)
        }
    } else {
        pre.reraise_mult
            .get(sit.num_raises as usize - 1)
            .map(|m| (m * sit.current_bet as f64).round() as Chips)
    };
    let mut amounts: Vec<Chips> = raise_to.into_iter().collect();
    amounts.push(sit.max_to);
    let amounts = finalize_amounts(sit, amounts, post.force_allin_threshold, 0.0);
    out.extend(amounts.into_iter().map(Action::Raise));
    out
}

/// Clamps raise-to amounts to the legal interval, converts near-all-in sizes
/// to all-in, removes duplicates and merges sizes that are too close.
fn finalize_amounts(sit: &Situation, amounts: Vec<Chips>, force_allin: f64, merge: f64) -> Vec<Chips> {
    let tagged = amounts.into_iter().map(|a| (a, 0)).collect();
    finalize_tagged(sit, tagged, force_allin, merge).into_iter().map(|(a, _)| a).collect()
}

/// [`finalize_amounts`] for amounts tagged with where they came from: a
/// duplicate keeps the smallest tag, a merged size the tag of the size it
/// merged into.
fn finalize_tagged(
    sit: &Situation,
    amounts: Vec<(Chips, usize)>,
    force_allin: f64,
    merge: f64,
) -> Vec<(Chips, usize)> {
    let lo = sit.min_raise_to.min(sit.max_to);
    let pot_after_call = sit.pot + sit.to_call;
    let mut v: Vec<(Chips, usize)> = amounts
        .into_iter()
        .map(|(a, tag)| {
            let a = a.clamp(lo, sit.max_to);
            let new_pot = pot_after_call + 2 * (a - sit.current_bet);
            let threshold = (new_pot as f64 * force_allin).round() as Chips;
            if sit.max_to - a <= threshold {
                (sit.max_to, tag)
            } else {
                (a, tag)
            }
        })
        .filter(|&(a, _)| a > sit.current_bet)
        .collect();
    v.sort_unstable();
    v.dedup_by_key(|x| x.0);
    if merge > 0.0 && v.len() > 1 {
        let ratio = |a: Chips| (a - sit.current_bet) as f64 / pot_after_call as f64;
        let mut kept: Vec<(Chips, usize)> = Vec::with_capacity(v.len());
        for &a in v.iter().rev() {
            match kept.last() {
                Some(&k) if (1.0 + ratio(k.0)) / (1.0 + ratio(a.0)) < 1.0 + merge => {}
                _ => kept.push(a),
            }
        }
        kept.reverse();
        v = kept;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sit(street: Street, pot: Chips, to_call: Chips, current_bet: Chips, max_to: Chips) -> Situation {
        Situation {
            street,
            pot,
            to_call,
            current_bet,
            max_to,
            min_raise_to: if current_bet == 0 { 100 } else { current_bet * 2 },
            num_raises: if current_bet == 0 { 0 } else { 1 },
            limpers: 0,
            big_blind: 100,
        }
    }

    #[test]
    fn parse_sizes() {
        assert_eq!("75%".parse::<BetSize>().unwrap(), BetSize::Pot(0.75));
        assert_eq!("3x".parse::<BetSize>().unwrap(), BetSize::Mult(3.0));
        assert_eq!("allin".parse::<BetSize>().unwrap(), BetSize::AllIn);
        assert!("0x".parse::<BetSize>().is_err());
        assert_eq!(BetSize::Pot(0.33).to_string(), "33%");
    }

    #[test]
    fn flop_bets() {
        // 6bb pot, 97.5bb behind: check, 33%, 75%.
        let menu = BetMenu::default();
        let a = menu.actions(&sit(Street::Flop, 600, 0, 0, 9750));
        assert_eq!(a, vec![Action::Check, Action::Raise(198), Action::Raise(450)]);
    }

    #[test]
    fn facing_bet_raises() {
        let menu = BetMenu::default();
        // Facing a 450 bet into 600 (pot now 1050).
        let a = menu.actions(&sit(Street::Flop, 1050, 450, 450, 9750));
        assert_eq!(a, vec![Action::Fold, Action::Call, Action::Raise(1350)]);
    }

    #[test]
    fn small_spr_adds_and_forces_allin() {
        let menu = BetMenu::default();
        // River: pot 2000, 1500 behind -> 50% (1000) is forced all-in? After
        // calling 1000 the pot is 4000 and 500 remains: 500 <= 0.15*4000=600.
        let a = menu.actions(&sit(Street::River, 2000, 0, 0, 1500));
        assert_eq!(a, vec![Action::Check, Action::Raise(1500)]);
    }

    #[test]
    fn categories_name_the_menu_size() {
        let menu = BetMenu::default();
        let cats = |s: &Situation| -> Vec<String> {
            postflop_categorized(s, &menu.postflop).iter().map(|(_, c)| c.to_string()).collect()
        };
        assert_eq!(cats(&sit(Street::Flop, 600, 0, 0, 9750)), ["Check", "33%", "75%"]);
        assert_eq!(cats(&sit(Street::Flop, 1050, 450, 450, 9750)), ["Fold", "Call", "3x"]);
        // Short stack: 50% is forced all-in and merges with the added shove.
        assert_eq!(cats(&sit(Street::River, 2000, 0, 0, 1500)), ["Check", "All-in"]);
        // Shoving is offered next to the sizes when it is at most 1.5 pots.
        assert_eq!(cats(&sit(Street::Turn, 2000, 0, 0, 2900)), ["Check", "50%", "All-in"]);
    }

    #[test]
    fn raise_cap() {
        let menu = BetMenu::default();
        let mut s = sit(Street::Turn, 5000, 1000, 2000, 9000);
        s.num_raises = 3;
        assert_eq!(menu.actions(&s), vec![Action::Fold, Action::Call]);
    }

    #[test]
    fn preflop_open_and_three_bet() {
        let menu = BetMenu::default();
        // HU SB first in: 0.5bb to call.
        let s = Situation {
            street: Street::Preflop,
            pot: 150,
            to_call: 50,
            current_bet: 100,
            max_to: 10000,
            min_raise_to: 200,
            num_raises: 0,
            limpers: 0,
            big_blind: 100,
        };
        assert_eq!(
            menu.actions(&s),
            vec![Action::Fold, Action::Call, Action::Raise(250), Action::Raise(10000)]
        );
        // BB facing a 2.5bb open: 3-bet to 3.5x = 875.
        let s = Situation { pot: 350, to_call: 150, current_bet: 250, min_raise_to: 400, num_raises: 1, ..s };
        assert_eq!(
            menu.actions(&s),
            vec![Action::Fold, Action::Call, Action::Raise(875), Action::Raise(10000)]
        );
    }
}
