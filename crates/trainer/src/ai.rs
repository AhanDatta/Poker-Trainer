//! Heuristic AI opponents.
//!
//! Preflop the AI samples from chart frequencies adjusted by its
//! [`Profile`]. Postflop it estimates its equity against a model of each
//! opponent's range (starting from the chart range of their preflop line and
//! narrowed after every postflop action) and applies style-dependent
//! thresholds for value-betting, bluffing, calling and raising.

use crate::preflop::{chart_policy, Availability, ChartSet, Freqs, PreAction, PreDecision};
use crate::profiles::Profile;
use holdem::combo::{combo_cards, combo_mask, NUM_CLASSES, NUM_COMBOS};
use holdem::eval::EvalMask;
use holdem::{Action, ActionRecord, BetMenu, Card, CardSet, Chips, HandState, Street};
use rand::Rng;

/// Probabilities for the 169 classes at a preflop decision under a profile.
pub fn profile_policy(charts: &ChartSet, profile: &Profile, d: &PreDecision) -> Vec<Freqs> {
    let base = chart_policy(charts, d);
    profile.adjust_preflop(&base, &d.avail)
}

fn sample_freqs<R: Rng>(f: &Freqs, rng: &mut R) -> PreAction {
    let mut x: f32 = rng.random();
    for a in PreAction::ALL {
        let p = f[a.index()];
        if x < p {
            return a;
        }
        x -= p;
    }
    // Rounding: pick the most likely action.
    PreAction::ALL.into_iter().max_by(|a, b| f[a.index()].partial_cmp(&f[b.index()]).unwrap()).unwrap()
}

#[derive(Clone, Debug)]
pub struct AiPlayer {
    pub seat: usize,
    pub profile: Profile,
    /// Modeled range of every seat (unused for its own seat).
    pub reads: Vec<Vec<f32>>,
}

impl AiPlayer {
    pub fn new(seat: usize, profile: Profile, num_players: usize) -> AiPlayer {
        AiPlayer { seat, profile, reads: vec![vec![1.0; NUM_COMBOS]; num_players] }
    }

    /// Preflop decision.
    pub fn decide_preflop<R: Rng>(
        &self,
        charts: &ChartSet,
        decision: &PreDecision,
        hole: [Card; 2],
        rng: &mut R,
    ) -> PreAction {
        let class = holdem::HandClass::of_cards(hole[0], hole[1]).grid_index();
        debug_assert!(class < NUM_CLASSES);
        let policy = profile_policy(charts, &self.profile, decision);
        sample_freqs(&policy[class], rng)
    }

    /// Sets the read of `seat` (typically its preflop line range).
    pub fn set_read(&mut self, seat: usize, weights: Vec<f32>) {
        self.reads[seat] = weights;
    }

    /// Narrows the read of the acting seat after a postflop action.
    pub fn observe(&mut self, state_before: &HandState, rec: &ActionRecord) {
        if rec.player == self.seat || rec.street == Street::Preflop {
            return;
        }
        let read = &mut self.reads[rec.player];
        let board = &state_before.board;
        let dead = CardSet::from_cards(board);
        let scores = hand_scores(board, read, dead);
        let pot = state_before.pot().max(1) as f32;
        let mult = |s: f32| -> f32 {
            match rec.action {
                Action::Fold => 1.0,
                Action::Check => 1.0 - 0.55 * sigmoid((s - 0.82) / 0.06),
                Action::Call => 0.2 + 0.8 * sigmoid((s - 0.35) / 0.1),
                Action::Raise(_) => {
                    if rec.to_call_before == 0 {
                        let f = (rec.put_in as f32 / pot).min(1.5);
                        let t = 0.5 + 0.12 * f;
                        0.2 + 0.8 * sigmoid((s - t) / 0.08)
                    } else {
                        0.12 + 0.88 * sigmoid((s - 0.68) / 0.07)
                    }
                }
            }
        };
        for (w, &s) in read.iter_mut().zip(&scores) {
            if *w > 0.0 {
                *w *= mult(s);
            }
        }
    }

    /// Postflop decision among `actions` (menu actions).
    pub fn decide_postflop<R: Rng>(&self, state: &HandState, menu: &BetMenu, rng: &mut R) -> Action {
        let p = self.seat;
        let actions = state.legal_actions(menu);
        let sit = state.situation(p);
        let hole = state.hole[p];
        let raises: Vec<Chips> =
            actions.iter().filter_map(|a| if let Action::Raise(t) = a { Some(*t) } else { None }).collect();
        let opps: Vec<&[f32]> = state
            .players_in_hand()
            .into_iter()
            .filter(|&q| q != p)
            .map(|q| self.reads[q].as_slice())
            .collect();
        // `eq` is the chance of beating every opponent: right for pot odds,
        // but in multiway pots it understates hand strength (top pair is
        // well ahead of the 1/n fair share). The style thresholds are
        // heads-up equities, so compare them with a heads-up equivalent:
        // with independent opponents P(beat all) ≈ P(beat one)^n.
        let eq = equity_vs_ranges(hole, &state.board, &opps, 600, rng);
        let strength = if opps.len() > 1 { eq.powf(1.0 / opps.len() as f32) } else { eq };
        let draw = draw_strength(hole, &state.board);
        let prof = &self.profile;
        let street_factor = match state.street {
            Street::Flop => 1.0,
            Street::Turn => 0.8,
            _ => 0.6,
        };
        let size = |strength: f32, rng: &mut R| -> Chips {
            if raises.len() == 1 {
                return raises[0];
            }
            let x = (prof.size_bias + strength + rng.random_range(-0.5..0.5f32)).clamp(-1.0, 1.0);
            let idx = (((x + 1.0) / 2.0) * (raises.len() - 1) as f32).round() as usize;
            raises[idx.min(raises.len() - 1)]
        };

        if sit.to_call == 0 {
            if !raises.is_empty() {
                if strength >= prof.value_threshold {
                    if rng.random::<f32>() >= prof.slowplay && rng.random::<f32>() < prof.bet_freq {
                        let s = (strength - prof.value_threshold) * 3.0;
                        return Action::Raise(size(s, rng));
                    }
                    return Action::Check;
                }
                if strength >= prof.value_threshold - 0.12 && rng.random::<f32>() < prof.bet_freq * 0.25 {
                    return Action::Raise(size(-0.6, rng));
                }
                if strength < 0.42 {
                    let p_bluff = prof.bluff_freq * street_factor * (0.45 + draw);
                    if rng.random::<f32>() < p_bluff {
                        return Action::Raise(size(0.0, rng));
                    }
                }
            }
            return Action::Check;
        }

        let pot_after_call = (sit.pot + sit.to_call) as f32;
        let required = sit.to_call as f32 / pot_after_call;
        let all_in_call = sit.to_call >= state.stacks[p];
        let mut need = required * (1.0 - 0.45 * prof.call_down) + 0.03;
        if all_in_call {
            need += 0.05 * (1.0 - prof.call_down);
        }
        if !raises.is_empty()
            && strength >= prof.raise_threshold
            && rng.random::<f32>() < prof.bet_freq * (1.0 - prof.slowplay)
        {
            let s = (strength - prof.raise_threshold) * 4.0;
            return Action::Raise(size(s, rng));
        }
        if eq >= need {
            if !raises.is_empty()
                && draw > 0.3
                && rng.random::<f32>() < prof.bluff_freq * 0.25 * street_factor
            {
                return Action::Raise(raises[0]);
            }
            return Action::Call;
        }
        if !raises.is_empty() && rng.random::<f32>() < prof.bluff_freq * 0.12 * (0.3 + draw) * street_factor {
            return Action::Raise(raises[0]);
        }
        if eq >= need * 0.7 && rng.random::<f32>() < prof.call_down * 0.35 {
            return Action::Call;
        }
        Action::Fold
    }
}

/// Maps a preflop abstract action to the concrete menu action.
pub fn concrete_preflop(avail: &Availability, a: PreAction) -> Action {
    avail
        .concrete(a)
        .or_else(|| avail.concrete(PreAction::Passive))
        .or_else(|| avail.concrete(PreAction::Fold))
        .expect("some action is available")
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Monte-Carlo equity of `hole` against one hand from each of `opps` (combo
/// weights), with the board completed at random. Exact on the river against
/// a single opponent.
pub fn equity_vs_ranges<R: Rng>(
    hole: [Card; 2],
    board: &[Card],
    opps: &[&[f32]],
    samples: usize,
    rng: &mut R,
) -> f32 {
    let dead = CardSet::from_cards(board).with(hole[0]).with(hole[1]);
    // Per-opponent sampling tables.
    let tables: Vec<(Vec<usize>, Vec<f32>)> = opps
        .iter()
        .map(|w| {
            let mut combos = Vec::new();
            let mut cum = Vec::new();
            let mut total = 0f32;
            for (c, &x) in w.iter().enumerate() {
                if x > 0.0 && combo_mask(c) & dead.0 == 0 {
                    total += x;
                    combos.push(c);
                    cum.push(total);
                }
            }
            if combos.is_empty() {
                for c in 0..NUM_COMBOS {
                    if combo_mask(c) & dead.0 == 0 {
                        total += 1.0;
                        combos.push(c);
                        cum.push(total);
                    }
                }
            }
            (combos, cum)
        })
        .collect();
    if tables.is_empty() {
        return 1.0;
    }
    let board_mask = EvalMask::from_cards(board);
    if board.len() == 5 && tables.len() == 1 {
        let mine = board_mask.add(hole[0]).add(hole[1]).evaluate();
        let (combos, cum) = &tables[0];
        let (mut won, mut total) = (0f64, 0f64);
        let mut prev = 0f32;
        for (i, &c) in combos.iter().enumerate() {
            let w = (cum[i] - prev) as f64;
            prev = cum[i];
            let (a, b) = combo_cards(c);
            let theirs = board_mask.add(a).add(b).evaluate();
            total += w;
            won += w * if mine > theirs {
                1.0
            } else if mine == theirs {
                0.5
            } else {
                0.0
            };
        }
        return if total > 0.0 { (won / total) as f32 } else { 0.5 };
    }
    let missing = 5 - board.len();
    let mut won = 0f64;
    let mut count = 0usize;
    'sample: for _ in 0..samples {
        let mut used = dead;
        let mut opp_hands = Vec::with_capacity(tables.len());
        for (combos, cum) in &tables {
            let mut tries = 0;
            loop {
                let x = rng.random::<f32>() * cum[cum.len() - 1];
                let i = cum.partition_point(|&v| v <= x).min(combos.len() - 1);
                let c = combos[i];
                if combo_mask(c) & used.0 == 0 {
                    used.0 |= combo_mask(c);
                    opp_hands.push(combo_cards(c));
                    break;
                }
                tries += 1;
                if tries > 20 {
                    continue 'sample;
                }
            }
        }
        let mut bm = board_mask;
        for _ in 0..missing {
            loop {
                let c = Card::from_index(rng.random_range(0..52u8));
                if !used.contains(c) {
                    used.insert(c);
                    bm = bm.add(c);
                    break;
                }
            }
        }
        let mine = bm.add(hole[0]).add(hole[1]).evaluate();
        let mut best_opp = 0;
        let mut ties = 0;
        for &(a, b) in &opp_hands {
            let v = bm.add(a).add(b).evaluate();
            if v > best_opp {
                best_opp = v;
                ties = 0;
            }
            if v == mine {
                ties += 1;
            }
        }
        won += if mine > best_opp {
            1.0
        } else if mine == best_opp {
            1.0 / (ties as f64 + 1.0)
        } else {
            0.0
        };
        count += 1;
    }
    if count == 0 {
        0.5
    } else {
        (won / count as f64) as f32
    }
}

/// Draw potential in `0..1` (flush draws and straight draws; 0 on the river).
pub fn draw_strength(hole: [Card; 2], board: &[Card]) -> f32 {
    if board.len() >= 5 || board.len() < 3 {
        return 0.0;
    }
    let mut score: f32 = 0.0;
    // Flush draw: four of a suit including at least one hole card.
    for s in 0..4u8 {
        let on_board = board.iter().filter(|c| c.suit() == s).count();
        let in_hole = hole.iter().filter(|c| c.suit() == s).count();
        if in_hole > 0 && on_board + in_hole == 4 {
            score += if hole.iter().any(|c| c.suit() == s && c.rank() == 12) { 0.6 } else { 0.5 };
        }
    }
    // Straight draws: ranks that would complete a straight using a hole card.
    let ranks = |cards: &mut dyn Iterator<Item = &Card>| cards.fold(0u32, |m, c| m | (1 << c.rank()));
    let all = ranks(&mut board.iter().chain(hole.iter()));
    let board_only = ranks(&mut board.iter());
    let straight = |m: u32| -> bool {
        let w = (m << 1) | ((m >> 12) & 1); // shift up one and add the ace as bit 0 (wheel)
        let run = w & (w << 1) & (w << 2) & (w << 3) & (w << 4);
        run != 0
    };
    if !straight(all) {
        let outs = (0..13).filter(|&r| straight(all | (1 << r)) && !straight(board_only | (1 << r))).count();
        score += match outs {
            0 => 0.0,
            1 => 0.2,
            _ => 0.4,
        };
    }
    score.min(1.0)
}

/// Strength score in `0..1` of every combo of `range` on `board`: the
/// percentile of its made hand within the (weighted) range, boosted by
/// draws. Combos conflicting with `dead` score 0.
pub fn hand_scores(board: &[Card], range: &[f32], dead: CardSet) -> Vec<f32> {
    let bm = EvalMask::from_cards(board);
    let mut vals: Vec<(u32, usize)> = Vec::new();
    let mut scores = vec![0f32; NUM_COMBOS];
    for c in 0..NUM_COMBOS {
        if range[c] > 0.0 && combo_mask(c) & dead.0 == 0 {
            let (a, b) = combo_cards(c);
            vals.push((bm.add(a).add(b).evaluate(), c));
        }
    }
    vals.sort_unstable();
    let total: f32 = vals.iter().map(|&(_, c)| range[c]).sum();
    if total <= 0.0 {
        return scores;
    }
    let mut cum = 0f32;
    let mut i = 0;
    while i < vals.len() {
        let v = vals[i].0;
        let mut j = i;
        let mut group = 0f32;
        while j < vals.len() && vals[j].0 == v {
            group += range[vals[j].1];
            j += 1;
        }
        let pct = (cum + group * 0.5) / total;
        for k in i..j {
            let c = vals[k].1;
            let (a, b) = combo_cards(c);
            let d = draw_strength([a, b], board);
            scores[c] = (pct + d * 0.35 * (1.0 - pct)).min(1.0);
        }
        cum += group;
        i = j;
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use holdem::card::parse_cards;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn cards2(s: &str) -> [Card; 2] {
        let c = parse_cards(s).unwrap();
        [c[0], c[1]]
    }

    #[test]
    fn equity_sanity() {
        let mut rng = StdRng::seed_from_u64(5);
        let any = vec![1.0; NUM_COMBOS];
        let aa = equity_vs_ranges(cards2("AsAh"), &[], &[&any], 4000, &mut rng);
        assert!((aa - 0.85).abs() < 0.03, "{aa}");
        // River nuts vs anything.
        let board = parse_cards("2c7d9hTsJc").unwrap();
        let e = equity_vs_ranges(cards2("KdQd"), &board, &[&any], 0, &mut rng);
        assert!(e > 0.99, "{e}");
    }

    /// Reads after a 3-bet pot: AJ on Jc7d5d beats one such range ~66% of
    /// the time but both only ~45%.
    const READ: &str = "QQ+, TT-66, AK, AQ, KQs, 98s";

    /// Regression: in multiway pots the AI compared "chance to beat
    /// everyone" with heads-up thresholds, so even a Maniac checked top
    /// pair, top kicker.
    #[test]
    fn multiway_strong_hands_bet() {
        use holdem::{BetMenu, TableConfig};
        let hole = |s: &str| cards2(s);
        // 6-max, three players see a flop of Jc 7d 5d; seat 2 holds AJ.
        let mut st = HandState::new(
            TableConfig::new(6, 50, 100),
            0,
            vec![10_000; 6],
            vec![hole("2c3h"), hole("4c6h"), hole("AsJd"), hole("8c9h"), hole("TcTs"), hole("KcQh")],
        );
        let menu = BetMenu::default();
        // Button 0: HJ (4) and BTN (0) limp, the others fold, BB (2) checks.
        while let Some(p) = st.to_act.filter(|_| st.street == Street::Preflop) {
            let a = if [0, 2, 4].contains(&p) { Action::Call } else { Action::Fold };
            let a = if a == Action::Call && st.situation(p).to_call == 0 { Action::Check } else { a };
            st.apply(a).unwrap();
        }
        st.deal_board(&parse_cards("Jc7d5d").unwrap()).unwrap();
        let mut rng = StdRng::seed_from_u64(9);
        let mut bets = 0;
        for _ in 0..40 {
            // BB acts first on the flop (SB folded).
            let mut s = st.clone();
            while s.to_act != Some(2) {
                s.apply(Action::Check).unwrap();
            }
            let mut ai = AiPlayer::new(2, Profile::maniac(), 6);
            let read = holdem::Range::parse(READ).unwrap().into_weights();
            ai.reads[0] = read.clone();
            ai.reads[4] = read;
            if matches!(ai.decide_postflop(&s, &menu, &mut rng), Action::Raise(_)) {
                bets += 1;
            }
        }
        assert!(bets >= 30, "maniac bet top pair top kicker 3-way only {bets}/40 times");
    }

    #[test]
    fn draws() {
        let board = parse_cards("Ah7h2c").unwrap();
        assert!(draw_strength(cards2("KhQh"), &board) >= 0.5);
        assert_eq!(draw_strength(cards2("KcQd"), &board), 0.0);
        let board = parse_cards("9c8d2h").unwrap();
        assert!(draw_strength(cards2("Td7s"), &board) >= 0.4); // T987: open-ended
        assert!(draw_strength(cards2("JcTd"), &board) >= 0.2);
    }

    #[test]
    fn scores_order_hands() {
        let board = parse_cards("Kh7d2c").unwrap();
        let range = vec![1.0; NUM_COMBOS];
        let s = hand_scores(&board, &range, CardSet::from_cards(&board));
        let idx = |x: &str| {
            let c = parse_cards(x).unwrap();
            holdem::combo_index(c[0], c[1])
        };
        assert!(s[idx("KsKd")] > s[idx("AsKs")]);
        assert!(s[idx("AsKs")] > s[idx("QsJs")]);
        assert_eq!(s[idx("Kh2s")], 0.0); // conflicts with the board
    }
}
