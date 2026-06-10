"""Generate output.csv — v2 with cross-group 3rd-place advancement.

Identical to generate_submission.py except for one improvement:

  v1 estimates P(team advances) per group in isolation — only top-2 of that group.
  v2 estimates P(team advances to R32) via a global simulation of all 12 groups
     simultaneously, then picks the best 8 third-place teams per the official rule.

Why it matters: In WC 2026, 32 teams qualify — 12 group winners + 12 runners-up +
8 best third-place teams. A team tipped as top-2 in their group may have a higher
true P(advance) than the per-group sim shows because even if they slip to 3rd they
might still qualify as a best third. This makes the optimizer slightly more willing
to correctly tip borderline teams in competitive groups.

Usage:
    python3.11 scripts/generate_submission_v2.py
    python3.11 scripts/generate_submission_v2.py --sims 50000   # more global sims
"""
import csv
import sys
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
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


# ── Simulation helpers (same as v1) ───────────────────────────────────────────

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


def _group_standings(scores_h, scores_a, match_pairs, n_teams=4):
    """
    Returns (pts, gd, gf) arrays of shape (n_sims, n_teams) and rank array.
    ranks[:, 0] = index of 1st place team, etc.
    """
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

    key   = pts * 1_000_000 + (gd + 200) * 1_000 + gf
    ranks = np.argsort(-key, axis=1)          # shape (n_sims, n_teams)
    return pts, gd, gf, ranks


def _group_top2(scores_h, scores_a, match_pairs, n_teams=4):
    """Boolean (n_teams,) or (n_sims, n_teams) — top-2 finishers."""
    single = scores_h.ndim == 1
    if single:
        scores_h = scores_h[np.newaxis]
        scores_a = scores_a[np.newaxis]

    _, _, _, ranks = _group_standings(scores_h, scores_a, match_pairs, n_teams)
    n = scores_h.shape[0]
    advances = np.zeros((n, n_teams), dtype=bool)
    advances[np.arange(n)[:, None], ranks[:, :2]] = True

    return advances[0] if single else advances


def _match_pts(pred_h, pred_a, sim_h, sim_a):
    pred_gd = int(pred_h) - int(pred_a)
    sim_gd  = sim_h.astype(np.int32) - sim_a.astype(np.int32)
    pred_out = np.sign(pred_gd)
    sim_out  = np.sign(sim_gd)
    right_out = sim_out == pred_out
    exact     = right_out & (sim_h == pred_h) & (sim_a == pred_a)
    right_gd  = right_out & (sim_gd == pred_gd) & ~exact
    return np.where(exact, 5, np.where(right_gd, 3, np.where(right_out, 2, 0))).astype(np.float32)


def _top_k_candidates(lam_h, lam_a, k=8, max_g=6):
    cands = []
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = (exp(-lam_h) * lam_h**h / factorial(h)) * \
                (exp(-lam_a) * lam_a**a / factorial(a))
            cands.append((h, a, p))
    cands.sort(key=lambda x: -x[2])
    return [(h, a) for h, a, _ in cands[:k]]


# ── NEW: Global simulation for cross-group P(advance to R32) ─────────────────

def estimate_advance_probs(
    match_data: list[dict],
    n_sims: int = 30_000,
) -> dict[str, float]:
    """
    Simulate all 12 groups simultaneously and apply the best-8 third-place rule.
    Returns P(advance to R32) for every team — correctly accounting for 3rd-place pool.

    The best-8 thirds rule: after group stage, the 8 best-ranked third-place teams
    (by pts, then gd, then gf) also advance. This is cross-group: a 3rd-place team's
    chance depends on how strong the other groups' 3rd-place finishers are.
    """
    from football_predictor.data.wc2026 import GROUPS

    # Build per-group simulation data
    group_data: dict[str, dict] = {}
    for grp, group_teams in GROUPS.items():
        matches = [m for m in match_data if m["group"] == grp]
        teams = list(group_teams)  # canonical order for this group

        match_pairs = [
            (teams.index(m["home_team"]), teams.index(m["away_team"]))
            for m in matches
        ]
        group_data[grp] = {
            "teams": teams,
            "matches": matches,
            "match_pairs": match_pairs,
        }

    rng = np.random.default_rng(42)

    # advance_counts[team] = number of sims where team advances to R32
    advance_counts: dict[str, int] = {t: 0 for grp in GROUPS.values() for t in grp}

    # Simulate all groups in batches
    BATCH = 2000
    sims_done = 0

    while sims_done < n_sims:
        batch = min(BATCH, n_sims - sims_done)

        # For each group: simulate batch scores, get pts/gd/gf for 3rd-place teams
        # thirds_pool[sim_i] = list of (pts, gd, gf, team) for each group's 3rd-place team
        thirds_pool = [[] for _ in range(batch)]
        top2_mask: dict[str, np.ndarray] = {}   # grp → (batch, 4) bool

        for grp, gd in group_data.items():
            matches     = gd["matches"]
            teams       = gd["teams"]
            match_pairs = gd["match_pairs"]
            n_matches   = len(matches)
            n_teams     = len(teams)

            sim_h = np.zeros((batch, n_matches), dtype=np.int32)
            sim_a = np.zeros((batch, n_matches), dtype=np.int32)
            for i, m in enumerate(matches):
                if m.get("played"):
                    sim_h[:, i] = int(m["actual_home_goals"])
                    sim_a[:, i] = int(m["actual_away_goals"])
                else:
                    sim_h[:, i], sim_a[:, i] = _simulate_match(
                        m["bp_lam_home"], m["bp_lam_away"],
                        m["p_home"], m["p_draw"], m["p_away"],
                        batch, rng,
                    )

            pts, gd_arr, gf, ranks = _group_standings(sim_h, sim_a, match_pairs, n_teams)

            # Top-2 advance unconditionally
            advances = np.zeros((batch, n_teams), dtype=bool)
            advances[np.arange(batch)[:, None], ranks[:, :2]] = True
            top2_mask[grp] = advances

            # Collect 3rd-place team stats for cross-group pool
            third_idx = ranks[:, 2]  # shape (batch,)
            for sim_i in range(batch):
                ti = third_idx[sim_i]
                thirds_pool[sim_i].append((
                    int(pts[sim_i, ti]),
                    int(gd_arr[sim_i, ti]),
                    int(gf[sim_i, ti]),
                    grp,
                    teams[ti],
                ))

        # Apply best-8 thirds rule per sim
        for sim_i in range(batch):
            # Top-2 from each group advance unconditionally
            for grp, gd in group_data.items():
                for ti, team in enumerate(gd["teams"]):
                    if top2_mask[grp][sim_i, ti]:
                        advance_counts[team] += 1

            # Best 8 third-place teams across all 12 groups also advance
            thirds_sorted = sorted(
                thirds_pool[sim_i],
                key=lambda x: (-x[0], -x[1], -x[2])
            )
            for _pts, _gd, _gf, _grp, team in thirds_sorted[:8]:
                advance_counts[team] += 1

        sims_done += batch

    return {team: count / n_sims for team, count in advance_counts.items()}


# ── Group-level optimiser (v2: uses supplied p_advance) ──────────────────────

def optimise_group_scores(
    group_matches: list[dict],
    p_advance_global: dict[str, float],
    n_sims: int = 20_000,
    k: int = 8,
) -> list[tuple[int, int]]:
    """
    Same local search as v1, but uses p_advance_global (from cross-group simulation)
    instead of per-group simulation to estimate P(team actually advances to R32).
    """
    n = len(group_matches)
    rng = np.random.default_rng(42)

    teams: list[str] = []
    for m in group_matches:
        for t in (m["home_team"], m["away_team"]):
            if t not in teams:
                teams.append(t)

    match_pairs = [
        (teams.index(m["home_team"]), teams.index(m["away_team"]))
        for m in group_matches
    ]

    # P(team actually advances) from global simulation — this is the key difference
    p_advance = np.array([p_advance_global.get(t, 0.0) for t in teams])

    # Simulate group outcomes for E[match pts]
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

    candidates: list[list[tuple[int, int]]] = []
    for m in group_matches:
        if m.get("played"):
            candidates.append([(int(m["actual_home_goals"]), int(m["actual_away_goals"]))])
        else:
            candidates.append(_top_k_candidates(m["bp_lam_home"], m["bp_lam_away"], k=k))

    k_per_match = [len(c) for c in candidates]

    match_ev = [
        np.array([_match_pts(ph, pa, sim_h[:, m], sim_a[:, m]).mean()
                  for ph, pa in candidates[m]])
        for m in range(n)
    ]

    def combo_ev(combo: list[int]) -> float:
        total_match = sum(match_ev[m][c] for m, c in enumerate(combo))
        pred_h = np.array([candidates[m][c][0] for m, c in enumerate(combo)], dtype=np.int32)
        pred_a = np.array([candidates[m][c][1] for m, c in enumerate(combo)], dtype=np.int32)
        # Which teams does this combo tip as top-2?
        pred_advances = _group_top2(pred_h, pred_a, match_pairs)  # (4,) bool
        # Bonus: tipped team × P(they actually advance per global sim)
        adv = 3.0 * float(np.sum(pred_advances.astype(float) * p_advance))
        return total_match + adv

    combo = [int(np.argmax(match_ev[m])) for m in range(n)]
    best_ev = combo_ev(combo)

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

def main(n_global_sims: int = 30_000):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=n_global_sims,
                   help="Global simulation count for P(advance) estimation")
    args = p.parse_args()

    print("Training models (this takes ~60s)...")
    xgb, temp_cal, bp, ensemble, all_data, _ = train_model(quiet=False)

    print("\nPredicting all 72 group stage fixtures...")
    match_data = predict_group_stage(xgb, temp_cal, bp, ensemble, all_data, quiet=True)
    actual = load_actual_results()
    match_data = merge_actual_results(match_data, actual)

    print(f"\nRunning global simulation ({args.sims:,} sims) for cross-group P(advance)...")
    p_advance = estimate_advance_probs(match_data, n_sims=args.sims)

    # Show top/bottom P(advance) so we can sanity-check
    ranked = sorted(p_advance.items(), key=lambda x: -x[1])
    print("  Top 10 P(advance to R32):")
    for team, prob in ranked[:10]:
        print(f"    {team:30s}  {prob:.1%}")
    print("  Bottom 5:")
    for team, prob in ranked[-5:]:
        print(f"    {team:30s}  {prob:.1%}")

    # Optimise scores per group using global P(advance)
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in match_data:
        groups[m["group"]].append(m)

    print("\nOptimising scores per group (match pts + global advancement bonus)...")
    score_lookup: dict[frozenset, tuple[str, str, int, int]] = {}
    for group_label in sorted(groups):
        group_matches = groups[group_label]
        best_scores = optimise_group_scores(group_matches, p_advance)
        for m, (s_h, s_a) in zip(group_matches, best_scores):
            key = frozenset([m["home_team"], m["away_team"]])
            score_lookup[key] = (m["home_team"], m["away_team"], s_h, s_a)
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

    out_path = _ROOT / "output" / "output_v2.csv"
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
