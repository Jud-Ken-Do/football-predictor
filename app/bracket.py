"""Centred knockout-bracket builder + Plotly renderer for the WC 2026 dashboard.

Pure logic (no Streamlit) so it is unit-testable. Two entry points:

  build_expected_bracket(match_data, pair_probs)  -> dict   (the predicted tree)
  bracket_figure(bracket, flags, ...)             -> go.Figure

The predicted bracket is the *most-likely path*: group finishers ranked by
expected points (played matches locked to actual results), the official FIFA
2026 R32 slot mapping filled via the backend's build_r32(), then each tie
resolved to the higher knockout-advance probability (P(win 90') + draw resolved
by relative strength, matching ko_winner). The Final sits dead centre with the
two halves fanning out to the R32 at the left and right edges.

This is a single deterministic route — distinct from the marginal per-team
advancement probabilities (the Monte-Carlo heatmap), which is the fuller picture.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import plotly.graph_objects as go

# The bracket structure + filler live with the predictor. Import it under the
# SAME module name the Streamlit app uses (predict_wc2026, via ROOT/scripts on
# sys.path) so it is not loaded twice under two names.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from football_predictor.data.wc2026 import GROUPS  # noqa: E402
from predict_wc2026 import (  # noqa: E402
    build_r32,
    R16_PAIRS,
    QF_PAIRS,
    SF_PAIRS,
)

# Vertical order (top→bottom) of the 16 R32 match indices on each half, derived
# from R16_PAIRS/QF_PAIRS/SF_PAIRS so connectors never cross. Left half feeds
# SF0 (→Final), right half feeds SF1.
_R32_ORDER_L = [1, 4, 0, 2, 10, 11, 8, 9]
_R32_ORDER_R = [3, 5, 6, 7, 13, 15, 12, 14]

# Compact labels for names too long for the cards.
_ABBR = {
    "Bosnia and Herzegovina": "Bosnia", "United States": "USA",
    "South Africa": "S. Africa", "South Korea": "S. Korea",
}

_ROUND_TITLES = {"r32": "Round of 32", "r16": "Round of 16", "qf": "Quarter-finals",
                 "sf": "Semi-finals", "final": "Final"}


# ── Bracket construction ──────────────────────────────────────────────────────

def _expected_qualifiers(match_data: list[dict]) -> dict:
    """Deterministic qualifiers from expected points (played matches locked).

    Returns {"1": {group: team}, "2": {group: team}, "3": {group: team}} in the
    exact shape build_r32() expects (the "3" map holds the 8 best third-placers).
    """
    agg = {t: {"team": t, "group": g, "pts": 0.0, "gd": 0.0, "gf": 0.0}
           for g, ts in GROUPS.items() for t in ts}

    for m in match_data:
        h, a = m.get("home_team"), m.get("away_team")
        if h not in agg or a not in agg:
            continue
        if m.get("played"):
            hg, ag = m["actual_home_goals"], m["actual_away_goals"]
            ph = 1.0 if hg > ag else 0.0
            pdr = 1.0 if hg == ag else 0.0
            pa = 1.0 if ag > hg else 0.0
            lh, la = float(hg), float(ag)
        else:
            ph = float(m.get("p_home", 1 / 3))
            pdr = float(m.get("p_draw", 1 / 3))
            pa = float(m.get("p_away", 1 / 3))
            lh, la = float(m.get("bp_lam_home", 1.3)), float(m.get("bp_lam_away", 1.1))
        agg[h]["pts"] += 3 * ph + pdr
        agg[a]["pts"] += 3 * pa + pdr
        agg[h]["gf"] += lh
        agg[a]["gf"] += la
        agg[h]["gd"] += lh - la
        agg[a]["gd"] += la - lh

    winners, runners, thirds = {}, {}, []
    for g, ts in GROUPS.items():
        ranked = sorted((agg[t] for t in ts),
                        key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
        winners[g] = ranked[0]["team"]
        runners[g] = ranked[1]["team"]
        thirds.append(ranked[2])
    thirds.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
    best8 = {t["group"]: t["team"] for t in thirds[:8]}
    return {"1": winners, "2": runners, "3": best8}


def _adv_prob(a: str, b: str, pair_probs: dict) -> float:
    """P(a advances past b) in a neutral knockout — win in 90' or via ET/pens,
    matching ko_winner: draws resolved by relative strength p_a/(p_a+p_b)."""
    p_h, p_d, p_a = pair_probs.get((a, b), (1 / 3, 1 / 3, 1 / 3))
    if not all(math.isfinite(x) for x in (p_h, p_d, p_a)):
        return 0.5
    denom = max(p_h + p_a, 1e-9)
    val = p_h + p_d * (p_h / denom)
    return val if math.isfinite(val) else 0.5


def _match(a: str, b: str, pair_probs: dict, children=None) -> dict:
    # Guard the backend's "Unknown" third-place fallback: a real team always
    # beats a phantom slot rather than letting "Unknown" advance on the 0.5 tie.
    a_bad = a in (None, "Unknown")
    b_bad = b in (None, "Unknown")
    if a_bad and not b_bad:
        return {"a": a, "b": b, "winner": b, "p": 1.0, "children": children}
    if b_bad and not a_bad:
        return {"a": a, "b": b, "winner": a, "p": 1.0, "children": children}
    pa = _adv_prob(a, b, pair_probs)
    winner = a if pa >= 0.5 else b
    return {"a": a, "b": b, "winner": winner, "p": max(pa, 1.0 - pa),
            "children": children}


def build_expected_bracket(match_data: list[dict], pair_probs: dict) -> dict:
    """Most-likely knockout tree. Returns dict of round → list[match] + champion."""
    quals = _expected_qualifiers(match_data)
    r32_pairs = build_r32(quals)

    r32 = [_match(a, b, pair_probs) for a, b in r32_pairs]
    r32_w = [m["winner"] for m in r32]

    r16 = [_match(r32_w[i], r32_w[j], pair_probs, children=(("r32", i), ("r32", j)))
           for i, j in R16_PAIRS]
    r16_w = [m["winner"] for m in r16]

    qf = [_match(r16_w[i], r16_w[j], pair_probs, children=(("r16", i), ("r16", j)))
          for i, j in QF_PAIRS]
    qf_w = [m["winner"] for m in qf]

    sf = [_match(qf_w[i], qf_w[j], pair_probs, children=(("qf", i), ("qf", j)))
          for i, j in SF_PAIRS]
    sf_w = [m["winner"] for m in sf]

    final = _match(sf_w[0], sf_w[1], pair_probs, children=(("sf", 0), ("sf", 1)))

    return {"r32": r32, "r16": r16, "qf": qf, "sf": sf, "final": [final],
            "champion": final["winner"], "champion_p": final["p"]}


# ── Layout + rendering ────────────────────────────────────────────────────────

# Column x by round. Pitch = 2.0 so cards (half-width 0.9 → 1.8 wide) clear each
# other with a 0.2 gap; the Final sits dead centre at x=8.
_X = {("r32", "L"): 0.0, ("r16", "L"): 2.0, ("qf", "L"): 4.0, ("sf", "L"): 6.0,
      ("r32", "R"): 16.0, ("r16", "R"): 14.0, ("qf", "R"): 12.0, ("sf", "R"): 10.0,
      ("final", "C"): 8.0}
_LEFT_R16 = {0, 1, 4, 5}   # R16_PAIRS indices feeding the left SF
_LEFT_QF = {0, 1}          # QF_PAIRS indices feeding the left SF


def _positions() -> dict:
    """(round, idx) -> (x, y). y of a parent is the mean of its children's y."""
    pos: dict = {}
    for p, k in enumerate(_R32_ORDER_L):
        pos[("r32", k)] = (_X[("r32", "L")], 7 - p)
    for p, k in enumerate(_R32_ORDER_R):
        pos[("r32", k)] = (_X[("r32", "R")], 7 - p)
    for idx, (i, j) in enumerate(R16_PAIRS):
        side = "L" if idx in _LEFT_R16 else "R"
        y = (pos[("r32", i)][1] + pos[("r32", j)][1]) / 2
        pos[("r16", idx)] = (_X[("r16", side)], y)
    for idx, (i, j) in enumerate(QF_PAIRS):
        side = "L" if idx in _LEFT_QF else "R"
        y = (pos[("r16", i)][1] + pos[("r16", j)][1]) / 2
        pos[("qf", idx)] = (_X[("qf", side)], y)
    for idx, (i, j) in enumerate(SF_PAIRS):
        side = "L" if idx == 0 else "R"
        y = (pos[("qf", i)][1] + pos[("qf", j)][1]) / 2
        pos[("sf", idx)] = (_X[("sf", side)], y)
    pos[("final", 0)] = (_X[("final", "C")], (pos[("sf", 0)][1] + pos[("sf", 1)][1]) / 2)
    return pos


def _label(team: str, flags: dict) -> str:
    name = _ABBR.get(team, team)
    if len(name) > 14:
        name = name[:13] + "…"
    fl = flags.get(team, "")
    return f"{fl} {name}".strip()


def bracket_figure(
    bracket: dict,
    flags: dict | None = None,
    col_win: str = "#4ade80",
    col_dim: str = "#a1a1aa",
    col_gold: str = "#fbbf24",
) -> go.Figure:
    """Render the centred bracket. Final in the middle; halves fan to the edges."""
    flags = flags or {}
    pos = _positions()
    cw, rh = 0.9, 0.34   # half-card-width, half-row-height (card height = 2*rh)

    fig = go.Figure()
    shapes, anns = [], []
    hov_x, hov_y, hov_t = [], [], []   # invisible hover points (one per match)

    def card_edges(x: float):
        return x - cw, x + cw

    def add_card(node: dict, x: float, y: float, is_final: bool = False):
        left, right = card_edges(x)
        top_y, bot_y = y + rh, y - rh
        winner = node["winner"]
        for team, slot_lo, slot_hi in [(node["a"], y, top_y), (node["b"], bot_y, y)]:
            is_w = team == winner
            shapes.append(dict(
                type="rect", x0=left, x1=right, y0=slot_lo, y1=slot_hi,
                line=dict(color=(col_gold if (is_final and is_w) else
                                 col_win if is_w else "rgba(255,255,255,0.12)"),
                          width=1.4 if is_w else 0.8),
                fillcolor=("rgba(251,191,36,0.16)" if (is_final and is_w)
                           else "rgba(74,222,128,0.14)" if is_w
                           else "rgba(255,255,255,0.02)"),
                layer="below",
            ))
            # Winner also marked non-colour: bold + a leading ▸ (colour-blind safe).
            lbl = _label(team, flags)
            txt = f"<b>▸ {lbl}</b>" if is_w else lbl
            anns.append(dict(
                x=(left + right) / 2, y=(slot_lo + slot_hi) / 2,
                text=txt, showarrow=False,
                font=dict(size=10.5,
                          color=("#fef3c7" if (is_final and is_w) else
                                 "#dcfce7" if is_w else col_dim)),
                xanchor="center", yanchor="middle",
            ))
        # win-probability tag on the card's top edge
        anns.append(dict(
            x=x, y=top_y + 0.14, text=f"{node['p']:.0%}",
            showarrow=False, font=dict(size=8.5, color="#a1a1aa"),
            xanchor="center", yanchor="bottom",
        ))
        # invisible hover point with the full matchup
        hov_x.append(x); hov_y.append(y)
        hov_t.append(f"{_label(node['a'], flags)}  vs  {_label(node['b'], flags)}"
                     f"<br>→ <b>{_label(winner, flags)}</b> advances ({node['p']:.0%})")

    def connect(child_xy, parent_xy, side):
        cx, cy = child_xy
        px, py = parent_xy
        cxe = cx + cw if side == "L" else cx - cw
        pxe = px - cw if side == "L" else px + cw
        mx = (cxe + pxe) / 2
        for (x0, y0, x1, y1) in [(cxe, cy, mx, cy), (mx, cy, mx, py), (mx, py, pxe, py)]:
            shapes.append(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                               line=dict(color="rgba(255,255,255,0.16)", width=1),
                               layer="below"))

    # Connectors first (so cards sit on top), then cards.
    for rnd in ("r16", "qf", "sf", "final"):
        for idx, node in enumerate(bracket[rnd]):
            parent = pos[(rnd, idx)]
            for (crnd, cidx) in node["children"]:
                child = pos[(crnd, cidx)]
                side = "L" if child[0] < parent[0] else "R"
                connect(child, parent, side)

    for rnd in ("r32", "r16", "qf", "sf", "final"):
        for idx, node in enumerate(bracket[rnd]):
            x, y = pos[(rnd, idx)]
            add_card(node, x, y, is_final=(rnd == "final"))

    # Champion banner above the Final.
    fx, fy = pos[("final", 0)]
    champ = bracket["champion"]
    anns.append(dict(
        x=fx, y=fy + rh + 0.85,
        text=f"🏆 {_label(champ, flags)}",
        showarrow=False, font=dict(size=15, color=col_gold),
        xanchor="center", yanchor="bottom",
    ))

    # Round headers along the top (one per column).
    header_y = 8.2
    for (rnd, side), x in _X.items():
        anns.append(dict(x=x, y=header_y, text=_ROUND_TITLES[rnd].upper(),
                         showarrow=False, font=dict(size=9, color="#52525b"),
                         xanchor="center", yanchor="bottom"))

    # Invisible hover layer.
    fig.add_trace(go.Scatter(
        x=hov_x, y=hov_y, mode="markers",
        marker=dict(size=26, color="rgba(0,0,0,0)"),
        hoverinfo="text", hovertext=hov_t, showlegend=False,
    ))

    fig.update_layout(
        shapes=shapes, annotations=anns,
        xaxis=dict(visible=False, range=[-1.2, 17.2]),
        yaxis=dict(visible=False, range=[-1.0, 9.2]),
        height=760, margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
    )
    return fig
