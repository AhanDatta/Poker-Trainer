//! Drawing helpers shared by the screens: cards, action colors, bars and
//! the 13x13 hand grid (which scales its cells to the space it gets).

use holdem::combo::{combo_cards, HandClass, NUM_CLASSES};
use holdem::{Action, Card};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;
use std::cell::RefCell;

pub const ACCENT: Color = Color::Cyan;
/// Background of the part of a grid cell that isn't in the range.
pub const EMPTY_BG: Color = Color::Indexed(237);

/// Clickable things drawn on the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    /// A cell of the hand grid (grid index).
    Cell(usize),
    /// The i-th action of the current decision.
    Action(usize),
    /// A row of the spot editor.
    Field(usize),
    /// A graded decision on the review screen.
    Decision(usize),
}

/// Screen regions of clickable things, rebuilt on every draw.
#[derive(Default)]
pub struct HitMap(RefCell<Vec<(Rect, Hit)>>);

impl HitMap {
    pub fn clear(&self) {
        self.0.borrow_mut().clear();
    }
    pub fn add(&self, r: Rect, h: Hit) {
        self.0.borrow_mut().push((r, h));
    }
    pub fn at(&self, x: u16, y: u16) -> Option<Hit> {
        self.0
            .borrow()
            .iter()
            .rev()
            .find(|(r, _)| x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height)
            .map(|&(_, h)| h)
    }
}

pub fn suit_style(c: Card) -> (char, Color) {
    match c.suit() {
        0 => ('♣', Color::Green),
        1 => ('♦', Color::LightBlue),
        2 => ('♥', Color::LightRed),
        _ => ('♠', Color::White),
    }
}

pub fn card_span(c: Card) -> Span<'static> {
    let (sym, color) = suit_style(c);
    Span::styled(format!("{}{}", c.rank_char(), sym), Style::default().fg(color).add_modifier(Modifier::BOLD))
}

pub fn cards_spans(cards: &[Card]) -> Vec<Span<'static>> {
    let mut v = Vec::new();
    for (i, &c) in cards.iter().enumerate() {
        if i > 0 {
            v.push(Span::raw(" "));
        }
        v.push(card_span(c));
    }
    v
}

pub fn combo_spans(combo: usize) -> Vec<Span<'static>> {
    let (a, b) = combo_cards(combo);
    let (hi, lo) = if a.rank() >= b.rank() { (a, b) } else { (b, a) };
    vec![card_span(hi), card_span(lo)]
}

/// Cards drawn as 3-line boxes (`None` = face down / not dealt yet).
pub fn big_cards(cards: &[Option<Card>]) -> Vec<Line<'static>> {
    let mut rows = vec![Vec::new(), Vec::new(), Vec::new()];
    for (i, c) in cards.iter().enumerate() {
        if i > 0 {
            for r in rows.iter_mut() {
                r.push(Span::raw(" "));
            }
        }
        match c {
            Some(c) => {
                let (sym, color) = suit_style(*c);
                let st = Style::default().fg(color).add_modifier(Modifier::BOLD);
                let edge = Style::default().fg(Color::Gray);
                rows[0].push(Span::styled("╭───╮", edge));
                rows[1].push(Span::styled("│", edge));
                rows[1].push(Span::styled(format!("{}{} ", c.rank_char(), sym), st));
                rows[1].push(Span::styled("│", edge));
                rows[2].push(Span::styled("╰───╯", edge));
            }
            None => {
                let st = Style::default().fg(Color::DarkGray);
                rows[0].push(Span::styled("╭───╮", st));
                rows[1].push(Span::styled("│ · │", st));
                rows[2].push(Span::styled("╰───╯", st));
            }
        }
    }
    rows.into_iter().map(Line::from).collect()
}

/// Color of the `i`-th action in a node's action list.
pub fn action_color(actions: &[Action], i: usize) -> Color {
    match actions[i] {
        Action::Fold => Color::Indexed(67),
        Action::Check | Action::Call => Color::Green,
        Action::Raise(_) => {
            let raises: Vec<usize> =
                (0..actions.len()).filter(|&j| matches!(actions[j], Action::Raise(_))).collect();
            let k = raises.iter().position(|&j| j == i).unwrap_or(0);
            let palette = [Color::Yellow, Color::LightRed, Color::Red, Color::Magenta];
            if k + 1 == raises.len() && raises.len() > 1 {
                palette[(raises.len() - 1).min(3)]
            } else {
                palette[k.min(3)]
            }
        }
    }
}

pub fn bar(frac: f32, width: usize) -> String {
    let filled = (frac.clamp(0.0, 1.0) * width as f32).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

pub fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else if n == 0 {
        String::new()
    } else {
        s.chars().take(n - 1).collect::<String>() + "…"
    }
}

/// Splits `width` character cells among actions in proportion to `strategy`
/// (largest remainders), so a cell shows its mix rather than one action.
pub fn mix_slots(strategy: &[f32], width: usize) -> Vec<usize> {
    let total: f32 = strategy.iter().sum();
    if total <= 0.0 {
        return vec![0; strategy.len()];
    }
    let exact: Vec<f32> = strategy.iter().map(|&v| v / total * width as f32).collect();
    let mut slots: Vec<usize> = exact.iter().map(|&x| x.floor() as usize).collect();
    let mut left = width - slots.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..exact.len()).collect();
    order.sort_by(|&a, &b| (exact[b] - exact[b].floor()).total_cmp(&(exact[a] - exact[a].floor())));
    for &i in &order {
        if left == 0 {
            break;
        }
        slots[i] += 1;
        left -= 1;
    }
    slots
}

/// One cell of the hand grid.
#[derive(Clone, Debug, Default)]
pub struct GridCell {
    /// Share of the class in the range (0..=1): that much of the cell is
    /// colored. 0 = not in range.
    pub fill: f32,
    /// How the colored part is split (weights and colors, left to right).
    pub mix: Vec<(f32, Color)>,
    /// Second-row text (when cells are tall enough).
    pub sub: Option<String>,
}

/// Cell size for a grid drawn in `area` (including one title row and one
/// legend row).
pub fn grid_cell_size(area: Rect) -> (u16, u16) {
    let h = (area.height.saturating_sub(2) / 13).clamp(1, 3);
    let w = (area.width / 13).clamp(4, 3 * h + 4);
    (w, h)
}

/// Draws the 13x13 grid. A cell's colored width is its share of the range,
/// split by its mix; the rest is dark. `cursor` is highlighted.
pub fn draw_grid(
    f: &mut Frame,
    area: Rect,
    title: Line<'static>,
    cells: &[GridCell],
    legend: Vec<Span<'static>>,
    cursor: Option<usize>,
    hits: Option<&HitMap>,
) {
    debug_assert_eq!(cells.len(), NUM_CLASSES);
    let (cw, ch) = grid_cell_size(area);
    let mut lines = vec![title];
    for row in 0..13 {
        let mut rows: Vec<Vec<Span>> = vec![Vec::new(); ch as usize];
        for col in 0..13 {
            let g = row * 13 + col;
            let cell = &cells[g];
            let selected = cursor == Some(g);
            let label = HandClass::from_grid_index(g).to_string();
            // Background color of each column.
            let mut bg: Vec<Option<Color>> = vec![None; cw as usize];
            if cell.fill > 0.0 {
                let filled = ((cell.fill * cw as f32).round() as usize).clamp(1, cw as usize);
                let weights: Vec<f32> = cell.mix.iter().map(|m| m.0).collect();
                let slots = mix_slots(&weights, filled);
                let mut k = 0;
                for (i, &n) in slots.iter().enumerate() {
                    for _ in 0..n {
                        bg[k] = Some(cell.mix[i].1);
                        k += 1;
                    }
                }
                for b in bg.iter_mut().skip(k) {
                    *b = Some(EMPTY_BG);
                }
            }
            for (r, spans) in rows.iter_mut().enumerate() {
                let text: Vec<char> = match r {
                    0 => format!("{:<w$}", label, w = cw as usize).chars().collect(),
                    1 => format!("{:<w$}", cell.sub.clone().unwrap_or_default(), w = cw as usize)
                        .chars()
                        .take(cw as usize)
                        .collect(),
                    _ => vec![' '; cw as usize],
                };
                let mut x = 0;
                while x < cw as usize {
                    let b = bg[x];
                    let mut y = x;
                    while y < cw as usize && bg[y] == b {
                        y += 1;
                    }
                    let s: String = text[x..y].iter().collect();
                    let mut st = match b {
                        None => Style::default().fg(Color::DarkGray),
                        Some(c) if c == EMPTY_BG => Style::default().fg(Color::Gray).bg(c),
                        Some(c) => Style::default().fg(Color::Black).bg(c),
                    };
                    if selected {
                        st = st.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                        if b.is_none() || b == Some(EMPTY_BG) {
                            st = st.fg(Color::White).bg(Color::Indexed(240));
                        }
                    }
                    spans.push(Span::styled(s, st));
                    x = y;
                }
            }
            if let Some(h) = hits {
                let r = Rect {
                    x: area.x + col as u16 * cw,
                    y: area.y + 1 + row as u16 * ch,
                    width: cw,
                    height: ch,
                };
                if r.x + r.width <= area.x + area.width && r.y + r.height <= area.y + area.height {
                    h.add(r, Hit::Cell(g));
                }
            }
        }
        lines.extend(rows.into_iter().map(Line::from));
    }
    lines.push(Line::from(legend));
    f.render_widget(Paragraph::new(lines), area);
}

/// Legend spans: a colored square and a label per entry.
pub fn legend(entries: &[(String, Color)], max_label: usize) -> Vec<Span<'static>> {
    let mut v = Vec::new();
    for (l, c) in entries {
        v.push(Span::styled("  ", Style::default().bg(*c)));
        v.push(Span::raw(format!(" {}  ", truncate(l, max_label))));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_cells_show_mixes() {
        assert_eq!(mix_slots(&[1.0, 0.0], 4), vec![4, 0]);
        assert_eq!(mix_slots(&[0.75, 0.25], 4), vec![3, 1]);
        // A 14% bluff still shows up as one quarter of the cell.
        assert_eq!(mix_slots(&[0.86, 0.14], 4), vec![3, 1]);
        assert_eq!(mix_slots(&[0.5, 0.3, 0.2], 4), vec![2, 1, 1]);
        assert_eq!(mix_slots(&[0.0, 0.0], 4), vec![0, 0]);
        for s in [[0.1f32, 0.2, 0.7], [0.33, 0.33, 0.34], [0.05, 0.05, 0.9]] {
            assert_eq!(mix_slots(&s, 4).iter().sum::<usize>(), 4);
        }
    }

    #[test]
    fn grid_cells_grow_with_the_area() {
        let r = |w, h| Rect { x: 0, y: 0, width: w, height: h };
        assert_eq!(grid_cell_size(r(52, 15)), (4, 1));
        assert_eq!(grid_cell_size(r(30, 10)), (4, 1));
        assert_eq!(grid_cell_size(r(120, 30)), (9, 2));
        assert_eq!(grid_cell_size(r(200, 60)), (13, 3));
    }
}
