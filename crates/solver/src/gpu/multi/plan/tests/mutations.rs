//! Corrupted plans: every mutation must be rejected by `validate`.

use super::*;

/// Every mutation must be rejected.
fn assert_all_rejected(plan: &MultiPlan, mutations: Vec<(&str, Mutation)>) {
    for (what, mutate) in mutations {
        let mut bad = plan.clone();
        mutate(&mut bad);
        let r = bad.validate();
        eprintln!("MUT {what}: {r:?}");
        assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{what}: accepted");
    }
}

type Mutation = Box<dyn Fn(&mut MultiPlan)>;

fn find(plan: &MultiPlan, pred: impl Fn(usize) -> bool) -> usize {
    (0..plan.num_nodes as usize).find(|&i| pred(i)).unwrap()
}

#[test]
fn validate_rejects_corrupted_trees() {
    let cfg = spots().into_iter().find(|s| s.0 == "all-in turn").unwrap().1;
    let mut cfg = cfg;
    cfg.exact_removal = true;
    let m = MultiSolver::new(cfg).unwrap();
    let plan = plan_for(&m);
    plan.validate().unwrap();
    let parents = plan.parents().unwrap();
    let decision = find(&plan, |i| i > 0 && (plan.actor[i] as usize) < N);
    let other_decision = find(&plan, |i| i > decision && (plan.actor[i] as usize) < N);
    let chance = find(&plan, |i| plan.actor[i] == CHANCE);
    let dealt = plan.first_child[chance] as usize;
    let won = find(&plan, |i| plan.winner[i] != NONE);
    let showdown = find(&plan, |i| plan.actor[i] == NONE && plan.winner[i] == NONE);
    let below_decision = plan.first_child[0] as usize;
    // A batched leaf that isn't its parent's first child.
    let batched = find(&plan, |i| plan.scalar[i] == 0 && i != plan.first_child[parents[i] as usize] as usize);
    // A player in the hand at the chance node who doesn't act right after it.
    let free = (0..N).find(|&b| plan.folded[chance] >> b & 1 == 0 && b as u32 != plan.actor[dealt]).unwrap();
    let mutations: Vec<(&str, Mutation)> = vec![
        ("storage block out of range", Box::new(|p| p.offset[0] = p.storage_len)),
        ("children out of range", Box::new(|p| p.first_child[0] = p.num_nodes)),
        ("children before their parent", Box::new(move |p| p.first_child[decision] = 0)),
        ("two parents", Box::new(move |p| p.first_child[other_decision] = p.first_child[decision])),
        ("root actor changed", Box::new(|p| p.actor[0] = (p.actor[0] + 1) % 3)),
        ("root made a chance node", Box::new(|p| p.actor[0] = CHANCE)),
        ("actor out of range", Box::new(move |p| p.actor[decision] = 7)),
        ("actor folded", Box::new(move |p| p.folded[decision] |= 1 << p.actor[decision])),
        ("folded flags out of range", Box::new(|p| p.folded[0] = 8)),
        ("fold forgotten below a fold", Box::new(move |p| p.folded[won] = 0)),
        (
            "fold by a player not acting",
            Box::new(move |p| p.folded[below_decision] |= 1 << ((p.actor[0] + 2) % 3)),
        ),
        ("fold at a chance node", Box::new(move |p| p.folded[dealt] |= 1 << free)),
        ("wrong winner", Box::new(move |p| p.winner[won] = (p.winner[won] + 1) % 3)),
        ("winner out of range", Box::new(move |p| p.winner[won] = 3)),
        ("winner on a showdown", Box::new(move |p| p.winner[showdown] = 0)),
        ("winner on a decision node", Box::new(move |p| p.winner[decision] = 0)),
        ("showdown with one player", Box::new(move |p| p.folded[showdown] = 0b011)),
        ("terminal with children", Box::new(move |p| p.actions[showdown] = 1)),
        ("decision without children", Box::new(move |p| p.actions[decision] = 0)),
        ("too many actions", Box::new(|p| p.actions[0] = MAX_ACTIONS + 1)),
        ("scale slot shared", Box::new(move |p| p.scale_slot[other_decision] = p.scale_slot[decision])),
        ("scale slot out of range", Box::new(move |p| p.scale_slot[decision] = p.num_scale_slots)),
        ("scale slot on a chance node", Box::new(move |p| p.scale_slot[chance] = 0)),
        ("dealt card missing", Box::new(move |p| p.card[dealt] = NONE)),
        ("dealt card out of the deck", Box::new(move |p| p.card[dealt] = 52)),
        ("card dealt twice", Box::new(move |p| p.card[dealt + 1] = p.card[dealt])),
        ("card below a decision node", Box::new(move |p| p.card[below_decision] = 5)),
        ("card at the root", Box::new(|p| p.card[0] = 5)),
        ("zero chance factor", Box::new(move |p| p.chance_factor[chance] = 0.0)),
        ("non-finite chance factor", Box::new(move |p| p.chance_factor[chance] = f32::INFINITY)),
        ("scalar flag 2", Box::new(move |p| p.scalar[decision] = 2)),
        ("root batched", Box::new(|p| p.scalar[0] = 0)),
        (
            "chance child batched",
            Box::new(move |p| {
                for c in p.first_child[chance]..p.first_child[chance] + p.actions[chance] {
                    p.scalar[c as usize] = 0;
                }
            }),
        ),
        ("siblings with different rounding", Box::new(move |p| p.scalar[batched] = 1)),
        ("batched leaf with batching off", Box::new(|p| p.batch_river = false)),
        (
            "node array missing an entry",
            Box::new(|p| {
                p.winner.pop();
            }),
        ),
        ("no nodes", Box::new(|p| p.num_nodes = 0)),
    ];
    assert_all_rejected(&plan, mutations);
}

#[test]
fn validate_rejects_corrupted_schedules() {
    let cfg = spots().into_iter().find(|s| s.0 == "sampled turn").unwrap().1;
    let m = MultiSolver::new(cfg).unwrap();
    let plan = plan_for(&m);
    plan.validate().unwrap();
    // Player 1 traverses; its opponents are 0 and 2.
    let pp = &plan.players[1];
    let opp0 = pp.opp_nodes[0][1] as usize;
    let opp0_other = pp.opp_nodes[0][2] as usize;
    let opp1 = pp.opp_nodes[1][0] as usize;
    let chance = pp.chance_nodes[0] as usize;
    let own = pp.own_nodes[0] as usize;
    let child_of_opp0 = plan.first_child[opp0] as usize;
    // A decision node at which player 1 folded, and a node below it.
    let cut = *pp.fixed.iter().find(|&&f| plan.actor[f as usize] != NONE).unwrap() as usize;
    let below_cut = plan.first_child[cut] as usize;
    let updated = find(&plan, |i| pp.strategy_update[i] == 1);
    // Player 2's nodes are updated by player 0 unless player 0 folded.
    let not_updated = find(&plan, |i| pp.opp_nodes[1].contains(&(i as u32)) && pp.strategy_update[i] == 0);
    let actor_of_not_updated = plan.actor[not_updated] as usize;
    let mutations: Vec<(&str, Mutation)> = vec![
        (
            "slot out of range",
            Box::new(move |p| p.players[1].slot[0][child_of_opp0] = p.players[1].num_slots[0]),
        ),
        ("root not in slot 0", Box::new(|p| p.players[1].slot[1][0] = 1)),
        ("opponent child slot shifted", Box::new(move |p| p.players[1].child_slot[0][opp0] += 1)),
        ("second opponent's child slot shifted", Box::new(move |p| p.players[1].child_slot[1][opp1] += 1)),
        ("chance child slot shifted", Box::new(move |p| p.players[1].child_slot[1][chance] += 1)),
        (
            "chance node splits one opponent only",
            Box::new(move |p| p.players[1].child_slot[0][chance] = NONE),
        ),
        (
            "child slots at the root's slot",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                pp.child_slot[0][opp0] = 0;
                for (a, c) in (p.first_child[opp0]..p.first_child[opp0] + p.actions[opp0]).enumerate() {
                    pp.slot[0][c as usize] = a as u32;
                }
            }),
        ),
        (
            "other opponent's slot not passed through",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                pp.slot[1][child_of_opp0] = pp.num_slots[1] - 1;
            }),
        ),
        ("child slots at an own node", Box::new(move |p| p.players[1].child_slot[0][own] = 1)),
        (
            "slot handed out twice",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                let first = pp.child_slot[0][opp0].min(pp.child_slot[0][opp0_other]);
                for node in [opp0, opp0_other] {
                    pp.child_slot[0][node] = first;
                    for (a, c) in (p.first_child[node]..p.first_child[node] + p.actions[node]).enumerate() {
                        pp.slot[0][c as usize] = first + a as u32;
                    }
                }
            }),
        ),
        ("unreached node given a slot", Box::new(move |p| p.players[1].slot[0][below_cut] = 0)),
        ("cut-off node given child slots", Box::new(move |p| p.players[1].child_slot[0][cut] = 1)),
        ("reach buffer overflows u32 indexing", Box::new(|p| p.players[1].num_slots[0] = u32::MAX)),
        (
            "node scheduled twice",
            Box::new(|p| {
                let pp = &mut p.players[1];
                pp.own_nodes.push(pp.own_nodes[0]);
                pp.levels.last_mut().unwrap().own.count += 1;
            }),
        ),
        (
            "cut-off node scheduled",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                pp.own_nodes.push(cut as u32);
                pp.levels.last_mut().unwrap().own.count += 1;
            }),
        ),
        (
            "node in the wrong list",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                let a = pp.opp_nodes[0].iter().position(|&x| x as usize == opp0).unwrap();
                pp.opp_nodes[0][a] = own as u32;
            }),
        ),
        ("level not contiguous", Box::new(|p| p.players[1].levels[1].own.start += 1)),
        ("level past its list", Box::new(|p| p.players[1].levels.last_mut().unwrap().chance.count += 1)),
        (
            "empty level",
            Box::new(|p| {
                let pp = &mut p.players[1];
                let end = |v: &[u32]| Span { start: v.len() as u32, count: 0 };
                let lv = MultiLevel {
                    opp: [end(&pp.opp_nodes[0]), end(&pp.opp_nodes[1])],
                    own: end(&pp.own_nodes),
                    chance: end(&pp.chance_nodes),
                };
                pp.levels.push(lv);
            }),
        ),
        ("levels swapped", Box::new(|p| p.players[1].levels.swap(1, 2))),
        ("strategy update dropped", Box::new(move |p| p.players[1].strategy_update[updated] = 0)),
        (
            "strategy update in the wrong traversal",
            Box::new(move |p| p.players[1].strategy_update[not_updated] = 1),
        ),
        (
            "strategy update in the actor's own traversal",
            Box::new(move |p| p.players[actor_of_not_updated].strategy_update[not_updated] = 1),
        ),
        ("strategy update on a chance node", Box::new(move |p| p.players[1].strategy_update[chance] = 1)),
        ("strategy-update flag 2", Box::new(move |p| p.players[1].strategy_update[updated] = 2)),
        (
            "leaf missing",
            Box::new(|p| {
                let pp = &mut p.players[1];
                pp.fixed.pop();
                pp.fixed_payoff.pop();
            }),
        ),
        (
            "leaf listed twice",
            Box::new(|p| {
                let pp = &mut p.players[1];
                pp.fixed.push(pp.fixed[0]);
                pp.fixed_payoff.push(0.0);
            }),
        ),
        (
            "showdown listed as a fixed leaf",
            Box::new(|p| {
                let pp = &mut p.players[1];
                let s = pp.showdowns.remove(0);
                pp.showdown_table.remove(0);
                pp.win.remove(0);
                pp.lose.remove(0);
                pp.fixed.push(s);
                pp.fixed_payoff.push(0.0);
            }),
        ),
        (
            "fixed leaf listed as a showdown",
            Box::new(|p| {
                let pp = &mut p.players[1];
                let f = pp.fixed.remove(0);
                pp.fixed_payoff.remove(0);
                pp.showdowns.push(f);
                pp.showdown_table.push(0);
                pp.win.push(0.0);
                pp.lose.push(0.0);
            }),
        ),
        (
            "expanded node listed as a leaf",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                pp.fixed[0] = own as u32;
            }),
        ),
        (
            "fixed payoffs missing",
            Box::new(|p| {
                p.players[1].fixed_payoff.pop();
            }),
        ),
        ("non-finite payoff", Box::new(|p| p.players[1].win[0] = f64::NAN)),
        ("payoff too large", Box::new(|p| p.players[2].fixed_payoff[0] = 1e13)),
        ("showdown table out of range", Box::new(|p| p.players[0].showdown_table[0] = p.num_tables)),
        (
            "slot arrays short",
            Box::new(|p| {
                p.players[0].child_slot[1].pop();
            }),
        ),
    ];
    assert_all_rejected(&plan, mutations);
}

#[test]
fn validate_rejects_corrupted_hand_data() {
    let cfg = spots().into_iter().find(|s| s.0 == "river").unwrap().1;
    let m = MultiSolver::new(cfg).unwrap();
    let plan = plan_for(&m);
    plan.validate().unwrap();
    let view = &plan.players[0].opponents[1];
    let first_ranked = view.tables.ranges.iter().position(|&r| r != NONE).unwrap();
    // A card whose list is non-empty, and a hand of that opponent without it.
    let c = (0..CARDS).find(|&c| view.card_off[c] < view.card_off[c + 1]).unwrap();
    let entry = view.card_off[c] as usize;
    let without = (0..view.hands as usize)
        .find(|&j| {
            let (a, b) = unpack_cards(view.cards[j]);
            a != c && b != c
        })
        .unwrap() as u32;
    let mutations: Vec<(&str, Mutation)> = vec![
        ("bad hand", Box::new(|p| p.players[0].my_cards[0] = pack_cards(52, 1))),
        ("hand with a repeated card", Box::new(|p| p.players[0].my_cards[1] = pack_cards(7, 7))),
        ("hand counts disagree", Box::new(|p| p.players[1].hands += 1)),
        (
            "empty range",
            Box::new(|p| {
                p.players[1].hands = 0;
                p.players[1].my_cards.clear();
            }),
        ),
        ("node count disagrees with the arrays", Box::new(|p| p.num_nodes += 1)),
        ("wrong seat", Box::new(|p| p.players[0].opponents[0].player = 2)),
        ("seats swapped", Box::new(|p| p.players[2].opponents.swap(0, 1))),
        ("opponent's cards differ", Box::new(|p| p.players[0].opponents[1].cards.swap(0, 1))),
        (
            "identical combo out of range",
            Box::new(|p| {
                let v = &mut p.players[1].opponents[0];
                v.same[0] = v.hands;
            }),
        ),
        (
            "identical combos missing",
            Box::new(|p| {
                p.players[1].opponents[1].same.pop();
            }),
        ),
        ("negative weight", Box::new(|p| p.players[1].opponents[0].weights[0] = -1.0)),
        ("NaN weight", Box::new(|p| p.players[2].opponents[1].weights[0] = f32::NAN)),
        ("weight too large", Box::new(|p| p.players[0].opponents[0].weights[0] = 1e7)),
        (
            "weights missing",
            Box::new(|p| {
                p.players[0].opponents[0].weights.pop();
            }),
        ),
        (
            "card list entry out of range",
            Box::new(|p| {
                let v = &mut p.players[2].opponents[0];
                v.card_hands[0] = v.hands;
            }),
        ),
        (
            "card list entry without the card",
            Box::new(move |p| p.players[0].opponents[1].card_hands[entry] = without),
        ),
        ("card offsets out of order", Box::new(|p| p.players[0].opponents[0].card_off[1] = u32::MAX)),
        (
            "card offsets short",
            Box::new(|p| {
                p.players[0].opponents[0].card_off.pop();
            }),
        ),
        (
            "opponent order out of range",
            Box::new(|p| {
                let v = &mut p.players[0].opponents[1];
                v.tables.opp_order[0] = v.hands;
            }),
        ),
        ("card position out of range", Box::new(|p| p.players[1].opponents[1].tables.card_pos[0] = 60_000)),
        (
            "card rank past its list",
            Box::new(move |p| p.players[0].opponents[1].tables.ranks[first_ranked] = u32::MAX),
        ),
        (
            "strength range past the list",
            Box::new(move |p| p.players[0].opponents[1].tables.ranges[first_ranked] = 0xFFFE_0000),
        ),
        ("too much shared memory", Box::new(|p| p.players[2].opponents[1].tables.max_opp = 5000)),
        ("tables for another board count", Box::new(|p| p.num_tables += 1)),
    ];
    assert_all_rejected(&plan, mutations);
}
