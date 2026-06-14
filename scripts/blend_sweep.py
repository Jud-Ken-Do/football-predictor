"""Sweep _MARKET_BLEND and show how it shifts one match's probabilities.
Trains once, then varies only the post-processing blend weight.
Records the chosen match's probabilities + top-10 advancement.

Usage:
    python3.11 scripts/blend_sweep.py
    python3.11 scripts/blend_sweep.py --match "Qatar vs Switzerland"
    python3.11 scripts/blend_sweep.py --blends 0.0,0.2,0.35,0.5 --sims 20000
"""
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--match", default="Algeria vs Austria",
                    help='Fixture to track, e.g. "Qatar vs Switzerland"')
parser.add_argument("--blends", default="0.0,0.12,0.2,0.3,0.4,0.5",
                    help="Comma-separated _MARKET_BLEND values to sweep")
parser.add_argument("--sims", type=int, default=30_000,
                    help="Monte Carlo sims for advancement probabilities")
parser.add_argument("--runs", type=int, default=1,
                    help="Runs per blend value (>1 to gauge MC noise)")
args = parser.parse_args()

BLENDS = [float(b) for b in args.blends.split(",")]
RUNS_PER_BLEND = args.runs
HOME_NAME, AWAY_NAME = (s.strip() for s in args.match.split(" vs "))

import numpy as np
import football_predictor.models.wc_context as ctx_mod
import football_predictor.features.wc2026_market as mkt_mod

from predict_wc2026 import train_model, predict_group_stage, load_actual_results, merge_actual_results
from generate_submission import estimate_advance_probs
from football_predictor.data.wc2026 import normalise

print("Training model once...")
xgb, temp_cal, bp, ensemble, all_data, _train_df = train_model(quiet=False)
actual = load_actual_results()
print("Done.\n")


def run_once(blend: float, seed_offset: int) -> tuple:
    ctx_mod._MARKET_BLEND = blend
    mkt_mod._load_cache.cache_clear()

    match_data = predict_group_stage(xgb, temp_cal, bp, ensemble, all_data, quiet=True)
    match_data = merge_actual_results(match_data, actual)

    # Pass the seed properly — the previous global monkey-patch of
    # np.random.default_rng ignored every caller's requested seed and would
    # silently break any other default_rng user in the process.
    p_advance = estimate_advance_probs(match_data, n_sims=args.sims, seed=42 + seed_offset)

    tracked = next((m for m in match_data
                    if normalise(m["home_team"]) == normalise(HOME_NAME)
                    and normalise(m["away_team"]) == normalise(AWAY_NAME)), None)
    if tracked is None:
        sys.exit(f"Fixture not found in group stage schedule: {HOME_NAME} vs {AWAY_NAME}")
    ph, pd_, pa = tracked["p_home"], tracked["p_draw"], tracked["p_away"]

    top10 = sorted(p_advance.items(), key=lambda x: -x[1])[:10]
    return ph, pd_, pa, top10


print(f"{'='*70}")
print(f"  MARKET BLEND SWEEP  ({', '.join(f'{b:.0%}' for b in BLENDS)})")
print(f"  Tracking: {HOME_NAME} vs {AWAY_NAME}")
print(f"{'='*70}\n")

for blend in BLENDS:
    print(f"── BLEND = {blend:.0%} ──────────────────────────────────────────────")
    for run in range(1, RUNS_PER_BLEND + 1):
        ph, pd_, pa, top10 = run_once(blend, seed_offset=(run - 1) * 7)
        print(f"  Run {run}:  {HOME_NAME} {ph:.1%} / Draw {pd_:.1%} / {AWAY_NAME} {pa:.1%}")
        if run == 1:
            print(f"  Top 10 advancement:")
            for team, prob in top10:
                print(f"    {team:<28} {prob:.1%}")
    print()
