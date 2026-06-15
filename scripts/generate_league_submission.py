"""Independent league-optimised submission — maximises EXPECTED MATCH POINTS.

The prediction league scores each fixture: exact = 5, right result + right goal
difference = 3, right result only = 2, wrong = 0. This script picks, for every
fixture *independently*, the scoreline with the highest expected points under
that rule, computed analytically from the model's Dixon-Coles score grid.

Difference vs output.csv (`generate_submission.py`): that one also folds in a
"teams through" advancement bonus, which pushes every pick toward a decisive
favourite win (hence its 1–0-heavy look). This one optimises pure match points,
so it predicts a draw (1–1) on the genuine coin-flips — exactly the games an
always-1–0 entry scores 0 on.

Writes SEPARATE files so you can A/B against your current entry:
    output/league_submission.csv   (same columns as output.csv — submittable)
    output/league_submission.txt   (readable: per-match pick, EV, draw count,
                                     and points scored so far vs an always-1–0 entry)

Usage:
    python3.11 scripts/generate_league_submission.py
    python3.11 scripts/generate_league_submission.py --name "Expected Value FC"
"""
import argparse
import csv
import sys
import warnings
warnings.filterwarnings("ignore")

from math import exp, factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_wc2026 import train_model, predict_group_stage, load_actual_results, merge_actual_results
from generate_submission import _TEMPLATE_TO_SCHEDULE

_ROOT = Path(__file__).resolve().parent.parent
_MAXG = 10           # goal grid ceiling for the probability matrix
_CANDS = range(0, 7)  # candidate scorelines 0..6 each side


# ── Dixon-Coles score grid + expected points ──────────────────────────────────

def _poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * lam ** k / factorial(k)


def _score_grid(lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    """P(home=h, away=a) over a 0..MAXG grid, Dixon-Coles low-score corrected."""
    ph = np.array([_poisson_pmf(h, lam_h) for h in range(_MAXG + 1)])
    pa = np.array([_poisson_pmf(a, lam_a) for a in range(_MAXG + 1)])
    grid = np.outer(ph, pa)
    # Dixon-Coles τ on the four low-score cells.
    grid[0, 0] *= 1.0 - lam_h * lam_a * rho
    grid[0, 1] *= 1.0 + lam_h * rho
    grid[1, 0] *= 1.0 + lam_a * rho
    grid[1, 1] *= 1.0 - rho
    grid = np.clip(grid, 1e-12, None)
    return grid / grid.sum()


def _points(pred_h: int, pred_a: int, act_h: int, act_a: int) -> int:
    """League scoring: exact 5, result+GD 3, result 2, wrong 0."""
    pred_sign = (pred_h > pred_a) - (pred_h < pred_a)
    act_sign = (act_h > act_a) - (act_h < act_a)
    if pred_sign != act_sign:
        return 0
    if pred_h == act_h and pred_a == act_a:
        return 5
    if (pred_h - pred_a) == (act_h - act_a):
        return 3
    return 2


def _best_scoreline(lam_h: float, lam_a: float, rho: float) -> tuple[int, int, float, float]:
    """Return (h, a, EV, runner_up_EV) maximising expected league points."""
    grid = _score_grid(lam_h, lam_a, rho)
    evs = []
    for ph in _CANDS:
        for pa in _CANDS:
            ev = 0.0
            for ah in range(_MAXG + 1):
                for aa in range(_MAXG + 1):
                    pts = _points(ph, pa, ah, aa)
                    if pts:
                        ev += grid[ah, aa] * pts
            evs.append((ev, ph, pa))
    evs.sort(reverse=True)
    best_ev, bh, ba = evs[0]
    return bh, ba, best_ev, evs[1][0]


def _favourite_1_0(m: dict) -> tuple[int, int]:
    """The always-1-0 baseline: 1-0 if home is the pre-match favourite, else 0-1."""
    ph = m.get("pred_p_home", m["p_home"])
    pa = m.get("pred_p_away", m["p_away"])
    return (1, 0) if ph >= pa else (0, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Expected Value FC", help="Entry name for the header")
    args = p.parse_args()

    print("=" * 56)
    print(f"  {args.name} — league match-points-optimal submission")
    print("=" * 56)
    print("\n[1/3] Training model...")
    xgb, temp_cal, bp, ensemble, all_data, _ = train_model(quiet=True)
    rho = float(getattr(bp, "_rho", 0.0) or 0.0)

    print("[2/3] Predicting 72 fixtures + optimising each scoreline...")
    match_data = merge_actual_results(
        predict_group_stage(xgb, temp_cal, bp, ensemble, all_data, quiet=True),
        load_actual_results(),
    )

    picks: dict[frozenset, tuple] = {}
    n_draw = 0
    total_ev = 0.0
    for m in match_data:
        bh, ba, ev, ru = _best_scoreline(m["bp_lam_home"], m["bp_lam_away"], rho)
        picks[frozenset((m["home_team"], m["away_team"]))] = (m["home_team"], m["away_team"], bh, ba, ev, ru)
        total_ev += ev
        if bh == ba:
            n_draw += 1

    # ── Points so far vs always-1-0, on already-played fixtures ────────────────
    played = [m for m in match_data if m.get("played")]
    ev_pts = base_pts = 0
    rows_played = []
    for m in played:
        key = frozenset((m["home_team"], m["away_team"]))
        _, _, bh, ba, _, _ = picks[key]
        ah, aa = int(m["actual_home_goals"]), int(m["actual_away_goals"])
        fh, fa = _favourite_1_0(m)
        pe, pb = _points(bh, ba, ah, aa), _points(fh, fa, ah, aa)
        ev_pts += pe
        base_pts += pb
        rows_played.append((m["home_team"], m["away_team"], ah, aa, bh, ba, pe, fh, fa, pb))

    # ── Write CSV in the submittable template format ──────────────────────────
    template_path = _ROOT / "templates" / "output_template.csv"
    with open(template_path, newline="") as f:
        rows = list(csv.DictReader(f))
    out_rows, unmatched = [], []
    for row in rows:
        t1 = _TEMPLATE_TO_SCHEDULE.get(row["team1"], row["team1"])
        t2 = _TEMPLATE_TO_SCHEDULE.get(row["team2"], row["team2"])
        key = frozenset((t1, t2))
        if key in picks:
            home, _, bh, ba, _, _ = picks[key]
            s1, s2 = (bh, ba) if home == t1 else (ba, bh)
        else:
            s1, s2 = "", ""
            unmatched.append(f"{row['team1']} vs {row['team2']}")
        out_rows.append({"match_id": row["match_id"], "group": row["group"],
                         "team1": row["team1"], "team2": row["team2"],
                         "score1": s1, "score2": s2})

    out_csv = _ROOT / "output" / "league_submission.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["match_id", "group", "team1", "team2", "score1", "score2"])
        w.writeheader(); w.writerows(out_rows)

    # ── Readable report ───────────────────────────────────────────────────────
    lines = [f"{args.name} — league match-points-optimal submission",
             "Scoring: exact=5, result+GD=3, result=2, wrong=0",
             f"Predicted draws: {n_draw}/72   |   total expected match points: {total_ev:.1f}", ""]
    if played:
        lines.append(f"POINTS ON {len(played)} PLAYED FIXTURES SO FAR:")
        lines.append(f"  This entry ({args.name}): {ev_pts} pts")
        lines.append(f"  Always-1-0 baseline:       {base_pts} pts")
        lines.append("")
        lines.append(f"  {'Match':<34} {'Result':>7}  {'EV pick':>8} {'pts':>3}   {'1-0 pick':>8} {'pts':>3}")
        for h, a, ah, aa, bh, ba, pe, fh, fa, pb in rows_played:
            lines.append(f"  {h+' vs '+a:<34} {f'{ah}-{aa}':>7}  {f'{bh}-{ba}':>8} {pe:>3}   {f'{fh}-{fa}':>8} {pb:>3}")
    report = "\n".join(lines)
    (_ROOT / "output" / "league_submission.txt").write_text(report + "\n")

    print("\n[3/3] Saved:")
    print(f"  → {out_csv}")
    print(f"  → {_ROOT / 'output' / 'league_submission.txt'}")
    print("\n" + report)
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} template fixtures unmatched.")


if __name__ == "__main__":
    main()
