//! Random deals: hole cards for every seat and a full board. Dealing real
//! cards makes card removal between any number of players exact, and a
//! sampled board is an unbiased stand-in for each hand's all-in equity.

use crate::rng::Rng;
use crate::spec::MAX_PLAYERS;
use holdem::combo::COMBO_CLASS;
use holdem::eval::EvalMask;
use holdem::{combo_index, Card, HandValue};

/// One deal, reduced to what the solver needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deal {
    /// Hand class (grid index) of each seat's hole cards.
    pub class: [u8; MAX_PLAYERS],
    /// Showdown value of each seat's hand on the dealt board.
    pub value: [HandValue; MAX_PLAYERS],
}

impl Deal {
    /// Deals `players` hands and a board from one deck.
    pub fn sample(players: usize, rng: &mut Rng) -> Deal {
        let cards = draw(2 * players + 5, rng);
        Deal::from_cards(players, &cards[..2 * players + 5])
    }

    /// The deal where seat `p` holds `cards[2p], cards[2p + 1]` and the
    /// board is the five cards after the hole cards.
    pub fn from_cards(players: usize, cards: &[Card]) -> Deal {
        assert!(players <= MAX_PLAYERS && cards.len() == 2 * players + 5);
        let board = EvalMask::from_cards(&cards[2 * players..]);
        let mut deal = Deal { class: [0; MAX_PLAYERS], value: [0; MAX_PLAYERS] };
        for p in 0..players {
            let (a, b) = (cards[2 * p], cards[2 * p + 1]);
            deal.class[p] = COMBO_CLASS[combo_index(a, b)];
            deal.value[p] = board.add(a).add(b).evaluate();
        }
        deal
    }

    /// The share of the pot seat `seat` wins at a showdown between the
    /// seats in `live` (a bit per seat): 1 over the number of best hands,
    /// or 0.
    #[inline]
    pub fn share(&self, live: u16, seat: usize) -> f32 {
        let mut best = 0;
        let mut ties = 0u32;
        let mut seats = live;
        while seats != 0 {
            let s = seats.trailing_zeros() as usize;
            seats &= seats - 1;
            match self.value[s].cmp(&best) {
                std::cmp::Ordering::Greater => (best, ties) = (self.value[s], 1),
                std::cmp::Ordering::Equal => ties += 1,
                std::cmp::Ordering::Less => {}
            }
        }
        if live & (1 << seat) != 0 && self.value[seat] == best {
            1.0 / ties as f32
        } else {
            0.0
        }
    }
}

/// The first `n` cards of a uniformly shuffled deck.
fn draw(n: usize, rng: &mut Rng) -> [Card; 52] {
    debug_assert!(n <= 52);
    let mut deck: [Card; 52] = std::array::from_fn(|i| Card::from_index(i as u8));
    for i in 0..n {
        let j = i + rng.below((52 - i) as u32) as usize;
        deck.swap(i, j);
    }
    deck
}

#[cfg(test)]
mod tests {
    use super::*;
    use holdem::{HandClass, HandState, TableConfig};

    fn cards(s: &str) -> Vec<Card> {
        s.split_whitespace().map(|c| c.parse().unwrap()).collect()
    }

    #[test]
    fn classes_and_shares_of_a_known_deal() {
        // Aces, kings and a hand that makes a flush with the board.
        let deal = Deal::from_cards(3, &cards("As Ah Ks Kd 7c 6c 2c 9c Jc 3s 4h"));
        let class = |p: usize| HandClass::from_grid_index(deal.class[p] as usize).to_string();
        assert_eq!([class(0), class(1), class(2)], ["AA", "KK", "76s"]);
        assert_eq!(deal.share(0b111, 2), 1.0);
        assert_eq!(deal.share(0b111, 0), 0.0);
        assert_eq!(deal.share(0b011, 0), 1.0);
        assert_eq!(deal.share(0b011, 2), 0.0, "a seat out of the hand wins nothing");
        // A board that plays for everyone splits the pot.
        let split = Deal::from_cards(2, &cards("2c 3d 2h 3s As Ks Qs Js Ts"));
        assert_eq!([split.share(0b11, 0), split.share(0b11, 1)], [0.5, 0.5]);
    }

    #[test]
    fn draws_are_distinct_and_uniform() {
        let mut rng = Rng::new(11);
        let mut first = [0u32; 52];
        for _ in 0..52_000 {
            let deck = draw(23, &mut rng);
            let mut seen = 0u64;
            for c in &deck[..23] {
                assert_eq!(seen & c.bit(), 0, "duplicate card");
                seen |= c.bit();
            }
            first[deck[22].index()] += 1;
        }
        for (i, &c) in first.iter().enumerate() {
            assert!((c as f64 - 1000.0).abs() < 200.0, "card {i} drawn {c} times in the last slot");
        }
    }

    /// Shares match the game engine's settlement of the same all-in hand.
    #[test]
    fn shares_match_the_engine_settlement() {
        let mut rng = Rng::new(5);
        for players in 2..=MAX_PLAYERS {
            for _ in 0..300 {
                let deck = draw(2 * players + 5, &mut rng);
                let deal = Deal::from_cards(players, &deck[..2 * players + 5]);
                let hole = (0..players).map(|p| [deck[2 * p], deck[2 * p + 1]]).collect();
                let config = TableConfig::new(players, 50, 100);
                // 2520 chips each divide evenly among any number of winners.
                let mut state = HandState::new(config, 0, vec![2520; players], hole);
                while let Some(p) = state.to_act {
                    let all_in = state.situation(p).max_to;
                    let action = if state.current_bet >= all_in {
                        holdem::Action::Call
                    } else {
                        holdem::Action::Raise(all_in)
                    };
                    state.apply(action).unwrap();
                }
                for street in [3, 1, 1] {
                    let at = 2 * players + state.board.len();
                    state.deal_board(&deck[at..at + street]).unwrap();
                }
                let settled = state.settle();
                let live = (1u16 << players) - 1;
                let pot = 2520.0 * players as f32;
                for p in 0..players {
                    let expected = settled.payouts[p] as f32;
                    assert!((deal.share(live, p) * pot - expected).abs() <= 1.0, "seat {p} of {players}");
                }
            }
        }
    }
}
