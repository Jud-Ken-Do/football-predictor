"""Sweep _MARKET_BLEND from 0.12 to 0.20 in 0.02 steps, 2 runs each.
Trains once, then varies only the post-processing blend weight.
Records Algeria vs Austria probabilities + top-10 advancement.
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

BLENDS = [0.12, 0.14, 0.16, 0.18, 0.20]
RUNS_PER_BLEND = 2

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
    p_advance = estimate_advance_probs(match_data, n_sims=30_000, seed=42 + seed_offset)

    alg_match = next((m for m in match_data
                      if normalise(m["home_team"]) == normalise("Algeria")
                      and normalise(m["away_team"]) == normalise("Austria")), None)
    ph = alg_match["p_home"] if alg_match else 0.0
    pd_ = alg_match["p_draw"] if alg_match else 0.0
    pa  = alg_match["p_away"] if alg_match else 0.0

    top10 = sorted(p_advance.items(), key=lambda x: -x[1])[:10]
    return ph, pd_, pa, top10


print(f"{'='*70}")
print(f"  MARKET BLEND SWEEP  (12% → 20%, 2 runs each)")
print(f"{'='*70}\n")

for blend in BLENDS:
    print(f"── BLEND = {blend:.0%} ──────────────────────────────────────────────")
    for run in range(1, RUNS_PER_BLEND + 1):
        ph, pd_, pa, top10 = run_once(blend, seed_offset=(run - 1) * 7)
        print(f"  Run {run}:  Algeria {ph:.1%} / Draw {pd_:.1%} / Austria {pa:.1%}")
        if run == 1:
            print(f"  Top 10 advancement:")
            for team, prob in top10:
                print(f"    {team:<28} {prob:.1%}")
    print()
