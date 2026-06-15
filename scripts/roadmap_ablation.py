"""Ablation for roadmap R9 (form/SoS shrinkage) and R10 (EM-estimated μ/home-adv).

Toggles the constants flags and runs the WC backtest for each config, reporting
average log-loss vs baseline so we can keep/kill each per the project's standard
(must beat or match the recorded baseline; kept as correctness only if neutral).

    python3.11 scripts/roadmap_ablation.py            # WC 2014/2018/2022
    python3.11 scripts/roadmap_ablation.py --continental  # + 4 continental folds
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from football_predictor import constants as C
from football_predictor.features.kalman_strength import KalmanStrengthFeatures
import backtest as BT

CONFIGS = [
    ("baseline",  False, False),
    ("R9 shrink", True,  False),
    ("R10 mu/ha", False, True),
    ("R9+R10",    True,  True),
]


def _avg(years, fw):
    lls = []
    for y in years:
        m = BT.run_backtest(y, include_shap=False, friendly_weight=fw)
        if m:
            lls.append(m["log_loss"])
    return float(np.mean(lls)) if lls else float("nan"), lls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--continental", action="store_true")
    p.add_argument("--fw", type=float, default=0.8)
    args = p.parse_args()

    years = [2014, 2018, 2022]
    rows = []
    for name, shrink, em in CONFIGS:
        C.USE_FORM_SHRINKAGE = shrink
        C.USE_KALMAN_EM_MU_HA = em
        KalmanStrengthFeatures._EM_Q_CACHE.clear()   # avoid cross-config q reuse
        print(f"\n{'='*60}\n  CONFIG: {name}  (shrink={shrink}, em_mu_ha={em})\n{'='*60}")
        avg, lls = _avg(years, args.fw)
        cont = {}
        if args.continental:
            for cname, cfg in BT.CONTINENTAL_CONFIGS.items():
                m = BT.run_continental_backtest(cname, cfg, friendly_weight=args.fw)
                if m:
                    cont[cname] = m["log_loss"]
        rows.append((name, avg, lls, cont))

    # restore defaults
    C.USE_FORM_SHRINKAGE = False
    C.USE_KALMAN_EM_MU_HA = False

    base = rows[0][1]
    print(f"\n{'='*72}\n  ROADMAP ABLATION SUMMARY — avg WC log-loss (lower better)\n{'='*72}")
    print(f"  {'config':<12} {'WC avg':>9} {'Δ vs base':>11}   per-year")
    for name, avg, lls, cont in rows:
        d = avg - base
        verdict = "" if name == "baseline" else ("  ✅ better" if d < -1e-4 else "  ~ neutral" if abs(d) <= 1e-4 else "  ❌ worse")
        print(f"  {name:<12} {avg:>9.4f} {d:>+11.4f}   {[round(x,4) for x in lls]}{verdict}")
        if cont:
            wide = np.mean(lls + list(cont.values()))
            print(f"               wide(7-tourn) avg = {wide:.4f}   continental={ {k:round(v,4) for k,v in cont.items()} }")


if __name__ == "__main__":
    main()
