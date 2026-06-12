"""Print the RAW most-likely scoreline per group fixture — straight from the
Poisson λ values, with NO submission optimizer (no 1-0 points constraint).

Shows: expected goals (λ), most-likely exact score, and top-3 scorelines.
Saves a sorted .txt report and a .csv to output/.
"""
import csv
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from math import exp, factorial

from predict_wc2026 import (
    train_model, predict_group_stage, load_actual_results, merge_actual_results,
)


def pmf(k, lam):
    return exp(-lam) * lam ** k / factorial(k)


def top_scorelines(lam_h, lam_a, k=3, max_g=6):
    cands = [
        (h, a, pmf(h, lam_h) * pmf(a, lam_a))
        for h in range(max_g + 1) for a in range(max_g + 1)
    ]
    cands.sort(key=lambda x: -x[2])
    return cands[:k]


print("Training model...")
xgb, temp_cal, bp, ensemble, all_data, _ = train_model(quiet=True)
match_data = predict_group_stage(xgb, temp_cal, bp, ensemble, all_data, quiet=True)
match_data = merge_actual_results(match_data, load_actual_results())
print()

# Sort by group, then by scheduled date
match_data = sorted(match_data, key=lambda m: (m["group"], m.get("date", "")))

lines: list[str] = []
csv_rows: list[dict] = []

lines.append("═" * 92)
lines.append("  WC 2026 — RAW MOST-LIKELY SCORELINES (no submission optimizer)")
lines.append("═" * 92)
lines.append(f"  {'Match':<46} {'xG':>11}   {'Top score':>9}   Top-3")
lines.append("  " + "─" * 88)

current = None
for m in match_data:
    if m["group"] != current:
        current = m["group"]
        lines.append(f"\n  ── Group {current} ──")
    h, a = m["home_team"], m["away_team"]
    lam_h, lam_a = m["bp_lam_home"], m["bp_lam_away"]
    top = top_scorelines(lam_h, lam_a)
    best_h, best_a, best_p = top[0]
    top3_str = "  ".join(f"{th}-{ta} ({tp:.0%})" for th, ta, tp in top)
    xg_str = f"{lam_h:.2f}-{lam_a:.2f}"
    match_str = f"{h} vs {a}"
    played = "  [played]" if m.get("played") else ""
    lines.append(f"  {match_str:<46} {xg_str:>11}   {best_h}-{best_a} ({best_p:>3.0%})   {top3_str}{played}")

    csv_rows.append({
        "group": m["group"],
        "date": m.get("date", ""),
        "home_team": h,
        "away_team": a,
        "lam_home": round(lam_h, 3),
        "lam_away": round(lam_a, 3),
        "score_home": best_h,
        "score_away": best_a,
        "score_prob": round(best_p, 4),
        "played": bool(m.get("played", False)),
    })

report = "\n".join(lines)
print(report)

out_dir = ROOT / "output"
out_dir.mkdir(exist_ok=True)
txt_path = out_dir / "raw_scorelines.txt"
csv_path = out_dir / "raw_scorelines.csv"

txt_path.write_text(report + "\n")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n  Saved → {txt_path}")
print(f"  Saved → {csv_path}")
