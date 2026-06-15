"""Tests for the centred knockout-bracket builder + renderer (app/bracket.py).

Covers the structural correctness of the predicted tree and the geometry of the
rendered figure (no overlapping cards, no backward connectors) — the layout bug
an adversarial review caught once.
"""
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))

import bracket as B  # noqa: E402
from football_predictor.data.wc2026 import GROUPS, GROUP_STAGE_SCHEDULE, normalise  # noqa: E402

_TEAMS = [t for g in GROUPS.values() for t in g]


def _synthetic_inputs():
    strength = {t: (len(_TEAMS) - i) / len(_TEAMS) for i, t in enumerate(_TEAMS)}

    def probs(a, b):
        da = strength[a] - strength[b]
        ph = 1 / (1 + math.exp(-3 * da))
        pd = max(0.22, 0.26 * (1 - abs(ph - 0.5)))
        ph *= (1 - pd)
        return ph, pd, 1 - ph - pd

    pair = {(a, b): probs(a, b) for a in _TEAMS for b in _TEAMS if a != b}
    md = [{
        "home_team": normalise(m["home_team"]), "away_team": normalise(m["away_team"]),
        "group": m["group"],
        **dict(zip(("p_home", "p_draw", "p_away"),
                   pair[(normalise(m["home_team"]), normalise(m["away_team"]))])),
        "bp_lam_home": 1.3, "bp_lam_away": 1.1, "played": False,
    } for m in GROUP_STAGE_SCHEDULE]
    return md, pair


def test_bracket_shape_and_distinct_teams():
    md, pair = _synthetic_inputs()
    bk = B.build_expected_bracket(md, pair)
    assert [len(bk[r]) for r in ("r32", "r16", "qf", "sf", "final")] == [16, 8, 4, 2, 1]
    r32_teams = {t for m in bk["r32"] for t in (m["a"], m["b"])}
    assert len(r32_teams) == 32                      # 12 winners + 12 runners + 8 thirds
    r16_teams = {t for m in bk["r16"] for t in (m["a"], m["b"])}
    assert r16_teams <= {m["winner"] for m in bk["r32"]}
    assert bk["champion"] in (bk["final"][0]["a"], bk["final"][0]["b"])


def test_played_matches_are_locked():
    md, pair = _synthetic_inputs()
    for m in md[:8]:
        m.update(played=True, actual_home_goals=2, actual_away_goals=0)
    bk = B.build_expected_bracket(md, pair)               # must not raise
    assert len({t for m in bk["r32"] for t in (m["a"], m["b"])}) == 32


def test_unknown_team_never_wins():
    md, pair = _synthetic_inputs()
    # Unknown vs a real team → real team advances with certainty.
    assert B._match("Unknown", "Brazil", pair)["winner"] == "Brazil"
    assert B._match("Brazil", "Unknown", pair)["winner"] == "Brazil"


def test_adv_prob_handles_nan_and_missing():
    assert B._adv_prob("A", "B", {}) == 0.5                       # missing key
    assert B._adv_prob("A", "B", {("A", "B"): (float("nan"), 0.3, 0.3)}) == 0.5


def test_figure_has_no_card_collisions_or_backward_connectors():
    md, pair = _synthetic_inputs()
    bk = B.build_expected_bracket(md, pair)
    pos = B._positions()
    cw, rh = 0.9, 0.34

    rects = []
    for rnd in ("r32", "r16", "qf", "sf", "final"):
        for idx in range(len(bk[rnd])):
            x, y = pos[(rnd, idx)]
            rects.append((x - cw, x + cw, y - rh, y + rh))

    def overlap(r1, r2):
        return not (r1[1] <= r2[0] or r2[1] <= r1[0] or r1[3] <= r2[2] or r2[3] <= r1[2])

    collisions = sum(overlap(rects[i], rects[j])
                     for i in range(len(rects)) for j in range(i + 1, len(rects)))
    assert collisions == 0

    backward = 0
    for rnd in ("r16", "qf", "sf", "final"):
        for idx, node in enumerate(bk[rnd]):
            px, py = pos[(rnd, idx)]
            for (cr, ci) in node["children"]:
                cx, cy = pos[(cr, ci)]
                if cx < px:                       # left half: parent edge must be right of child edge
                    if not (px - cw > cx + cw):
                        backward += 1
                else:                             # right half
                    if not (px + cw < cx - cw):
                        backward += 1
    assert backward == 0


def test_figure_renders():
    md, pair = _synthetic_inputs()
    fig = B.bracket_figure(B.build_expected_bracket(md, pair), flags={})
    d = fig.to_dict()
    assert len(d["layout"]["shapes"]) > 0
    assert len(d["data"][0]["x"]) == 31           # one hover point per match (16+8+4+2+1)
