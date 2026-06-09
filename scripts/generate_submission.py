"""Generate output.csv for the Rotate 2026 World Cup prediction competition.

Scoring:
  5 pts  — exact score
  3 pts  — right winner/draw + right goal difference (wrong exact score)
  2 pts  — right winner/draw only
  +3 pts — each team correctly tipped to advance from group (Round of 32)

Strategy: for each group, find the set of predicted scores across all 6 matches
that maximises total expected points = match points + advancement bonus.
Uses local search over top-k Poisson candidate scores per match.

Usage:
    python3.11 scripts/generate_submission.py
"""
import csv
import sys
import warnings
warnings.filterwarnings("ignore")

from math import exp, factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_wc2026 import train_model, predict_group_stage, load_actual_results, merge_actual_results

_ROOT = Path(__file__).resolve().parent.parent

_TEMPLATE_TO_SCHEDULE = {
    "Turkiye":    "Türkiye",
    "Cape Verde": "Cabo Verde",
    "Iran":       "IR Iran",
    "Curacao":    "Curaçao",
}


# ── Simulation helpers ────────────────────────────────────────────────────────

def _simulate_match(lam_h, lam_a, p_home, p_draw, p_away, n_sims, rng):
    lam_avg = (lam_h + lam_a) / 2
    outcomes = rng.choice(3, size=n_sims, p=[p_home, p_draw, p_away])
    h = np.zeros(n_sims, dtype=np.int32)
    a = np.zeros(n_sims, dtype=np.int32)

    mask = outcomes == 0
    if mask.any():
        gh = np.maximum(1, rng.poisson(lam_h, mask.sum()))
        h[mask] = gh
        a[mask] = np.minimum(gh - 1, rng.poisson(lam_a, mask.sum()))

    mask = outcomes == 1
    if mask.any():
        g = rng.poisson(lam_avg, mask.sum())
        h[mask] = a[mask] = g

    mask = outcomes == 2
    if mask.any():
        ga = np.maximum(1, rng.poisson(lam_a, mask.sum()))
        a[mask] = ga
        h[mask] = np.minimum(ga - 1, rng.poisson(lam_h, mask.sum()))

    return h, a


def _group_top2(scores_h, scores_a, match_pairs, n_teams=4):
    """Return boolean array of which teams finish top-2. Vectorised over sims."""
    single = scores_h.ndim == 1
    if single:
        scores_h = scores_h[np.newaxis]
        scores_a = scores_a[np.newaxis]

    n = scores_h.shape[0]
    pts = np.zeros((n, n_teams), dtype=np.int32)
    gd  = np.zeros((n, n_teams), dtype=np.int32)
    gf  = np.zeros((n, n_teams), dtype=np.int32)

    for m_idx, (hi, ai) in enumerate(match_pairs):
        h = scores_h[:, m_idx].astype(np.int32)
        a = scores_a[:, m_idx].astype(np.int32)
        hw = h > a; dw = h == a; aw = a > h
        pts[:, hi] += 3 * hw + dw
        pts[:, ai] += 3 * aw + dw
        gd[:, hi]  += h - a;  gd[:, ai]  += a - h
        gf[:, hi]  += h;      gf[:, ai]  += a

    # Sort: pts desc, gd desc, gf desc
    key = pts * 1_000_000 + (gd + 200) * 1_000 + gf
    ranks = np.argsort(-key, axis=1)
    advances = np.zeros((n, n_teams), dtype=bool)
    advances[np.arange(n)[:, None], ranks[:, :2]] = True

    return advances[0] if single else advances


def _match_pts(pred_h, pred_a, sim_h, sim_a):
    """Points for one predicted score vs simulated actuals. Vectorised."""
    pred_gd = int(pred_h) - int(pred_a)
    sim_gd  = sim_h.astype(np.int32) - sim_a.astype(np.int32)
    pred_out = np.sign(pred_gd)
    sim_out  = np.sign(sim_gd)
    right_out = sim_out == pred_out
    exact     = right_out & (sim_h == pred_h) & (sim_a == pred_a)
    right_gd  = right_out & (sim_gd == pred_gd) & ~exact
    return np.where(exact, 5, np.where(right_gd, 3, np.where(right_out, 2, 0))).astype(np.float32)


def _top_k_candidates(lam_h, lam_a, k=8, max_g=6):
    """Top-k scorelines by Poisson joint probability."""
    cands = []
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = (exp(-lam_h) * lam_h**h / factorial(h)) * \
                (exp(-lam_a) * lam_a**a / factorial(a))
            cands.append((h, a, p))
    cands.sort(key=lambda x: -x[2])
    return [(h, a) for h, a, _ in cands[:k]]


# ── Group-level optimiser ─────────────────────────────────────────────────────

def optimise_group_scores(group_matches, n_sims=20_000, k=8):
    """
    Find the predicted scores for all 6 matches in a group that maximise:

        E[total pts] = E[sum of match points] + E[advancement bonus]

    where E[advancement bonus] = 3 * sum_t P(team t predicted to advance)
                                       * P(team t actually advances)

    Played matches are locked to their actual score.
    Unplayed matches are optimised via local search over top-k candidates.
    """
    n = len(group_matches)
    rng = np.random.default_rng(42)

    # Build team list and match pairs
    teams: list[str] = []
    for m in group_matches:
        for t in (m["home_team"], m["away_team"]):
            if t not in teams:
                teams.append(t)
    match_pairs = [
        (teams.index(m["home_team"]), teams.index(m["away_team"]))
        for m in group_matches
    ]

    # Simulate outcomes (use actual score where played)
    sim_h = np.zeros((n_sims, n), dtype=np.int32)
    sim_a = np.zeros((n_sims, n), dtype=np.int32)
    for i, m in enumerate(group_matches):
        if m.get("played"):
            sim_h[:, i] = int(m["actual_home_goals"])
            sim_a[:, i] = int(m["actual_away_goals"])
        else:
            sim_h[:, i], sim_a[:, i] = _simulate_match(
                m["bp_lam_home"], m["bp_lam_away"],
                m["p_home"], m["p_draw"], m["p_away"],
                n_sims, rng,
            )

    # P(team advances) from simulations
    sim_advances = _group_top2(sim_h, sim_a, match_pairs)  # (n_sims, 4)
    p_advance = sim_advances.mean(axis=0)                  # (4,)

    # Candidate scores per match (single candidate if played)
    candidates: list[list[tuple[int, int]]] = []
    for m in group_matches:
        if m.get("played"):
            candidates.append([(int(m["actual_home_goals"]), int(m["actual_away_goals"]))])
        else:
            candidates.append(_top_k_candidates(m["bp_lam_home"], m["bp_lam_away"], k=k))

    k_per_match = [len(c) for c in candidates]

    # Precompute E[match pts] for each (match, candidate)
    match_ev = [
        np.array([_match_pts(ph, pa, sim_h[:, m], sim_a[:, m]).mean()
                  for ph, pa in candidates[m]])
        for m in range(n)
    ]

    def combo_ev(combo: list[int]) -> float:
        total_match = sum(match_ev[m][c] for m, c in enumerate(combo))
        pred_h = np.array([candidates[m][c][0] for m, c in enumerate(combo)], dtype=np.int32)
        pred_a = np.array([candidates[m][c][1] for m, c in enumerate(combo)], dtype=np.int32)
        pred_advances = _group_top2(pred_h, pred_a, match_pairs)  # (4,)
        adv = 3.0 * float(np.sum(pred_advances.astype(float) * p_advance))
        return total_match + adv

    # Start from per-match best
    combo = [int(np.argmax(match_ev[m])) for m in range(n)]
    best_ev = combo_ev(combo)

    # Local search: flip one match at a time until no improvement
    improved = True
    while improved:
        improved = False
        for m_idx in range(n):
            for c_idx in range(k_per_match[m_idx]):
                if c_idx == combo[m_idx]:
                    continue
                new_combo = combo.copy()
                new_combo[m_idx] = c_idx
                ev = combo_ev(new_combo)
                if ev > best_ev + 1e-9:
                    best_ev = ev
                    combo = new_combo
                    improved = True

    return [candidates[m][c] for m, c in enumerate(combo)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Training models (this takes ~60s)...")
    xgb, temp_cal, bp, ensemble, all_data, _ = train_model(quiet=False)

    print("\nPredicting all 72 group stage fixtures...")
    match_data = predict_group_stage(xgb, temp_cal, bp, ensemble, all_data, quiet=True)
    actual = load_actual_results()
    match_data = merge_actual_results(match_data, actual)

    # Group matches by group label
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in match_data:
        groups[m["group"]].append(m)

    print("Optimising scores per group (match pts + advancement bonus)...")
    score_lookup: dict[frozenset, tuple[str, str, int, int]] = {}
    for group_label in sorted(groups):
        group_matches = groups[group_label]
        best_scores = optimise_group_scores(group_matches)
        for m, (s_h, s_a) in zip(group_matches, best_scores):
            key = frozenset([m["home_team"], m["away_team"]])
            score_lookup[key] = (m["home_team"], m["away_team"], s_h, s_a)
        teams_in_group = sorted({m["home_team"] for m in group_matches} |
                                 {m["away_team"] for m in group_matches})
        print(f"  Group {group_label}: done")

    # Read template and fill scores
    template_path = _ROOT / "output_template 1.csv"
    with open(template_path, newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    unmatched = []
    for row in rows:
        t1 = _TEMPLATE_TO_SCHEDULE.get(row["team1"], row["team1"])
        t2 = _TEMPLATE_TO_SCHEDULE.get(row["team2"], row["team2"])
        key = frozenset([t1, t2])
        if key in score_lookup:
            home, away, s_h, s_a = score_lookup[key]
            s1, s2 = (s_h, s_a) if home == t1 else (s_a, s_h)
        else:
            s1, s2 = "", ""
            unmatched.append(f"{row['team1']} vs {row['team2']}")
        out_rows.append({
            "match_id": row["match_id"],
            "group":    row["group"],
            "team1":    row["team1"],
            "team2":    row["team2"],
            "score1":   s1,
            "score2":   s2,
        })

    out_path = _ROOT / "output" / "output.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match_id", "group", "team1", "team2", "score1", "score2"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved → {out_path}\n")

    if unmatched:
        print(f"WARNING: {len(unmatched)} fixtures not matched:")
        for u in unmatched:
            print(f"  - {u}")
        print()

    print(f"{'#':<4} {'Match':<34} {'Score':>5}")
    print("─" * 46)
    for row in out_rows:
        label = f"{row['team1']} vs {row['team2']}"
        score = f"{row['score1']}–{row['score2']}" if row["score1"] != "" else "?"
        print(f"{row['match_id']:<4} {label:<34} {score:>5}")


if __name__ == "__main__":
    main()
