"""A/B backtest for new candidate feature modules (qualification, wc_pedigree).

Adds each module to DEFAULT_FEATURE_MODULES and runs the WC backtest, reporting
average log-loss vs baseline so we can keep/kill per the project standard (must
beat or match baseline; rejected if it regresses, like R9).

    python3.11 scripts/feature_ablation.py
    python3.11 scripts/feature_ablation.py --continental
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

_BASE = list(C.DEFAULT_FEATURE_MODULES)   # snapshot the deployed module list

CONFIGS = [
    ("baseline",   []),
    ("+qual",      ["qualification"]),
    ("+pedigree",  ["wc_pedigree"]),
    ("+both",      ["qualification", "wc_pedigree"]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--continental", action="store_true")
    p.add_argument("--fw", type=float, default=0.8)
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only these config names (e.g. baseline +qual)")
    args = p.parse_args()

    configs = [c for c in CONFIGS if args.only is None or c[0] in args.only]
    years = [2014, 2018, 2022]
    rows = []
    for name, extra in configs:
        C.DEFAULT_FEATURE_MODULES[:] = _BASE + extra      # mutate in place (shared ref)
        KalmanStrengthFeatures._EM_Q_CACHE.clear()
        print(f"\n{'='*60}\n  CONFIG: {name}  (+{extra})\n{'='*60}")
        lls = []
        for y in years:
            m = BT.run_backtest(y, include_shap=False, friendly_weight=args.fw)
            if m:
                lls.append(m["log_loss"])
        cont = {}
        if args.continental:
            for cname, cfg in BT.CONTINENTAL_CONFIGS.items():
                m = BT.run_continental_backtest(cname, cfg, friendly_weight=args.fw)
                if m:
                    cont[cname] = m["log_loss"]
        rows.append((name, float(np.mean(lls)) if lls else float("nan"), lls, cont))

    C.DEFAULT_FEATURE_MODULES[:] = _BASE   # restore

    base = rows[0][1]
    print(f"\n{'='*72}\n  FEATURE ABLATION SUMMARY — avg WC log-loss (lower better)\n{'='*72}")
    for name, avg, lls, cont in rows:
        d = avg - base
        verdict = "" if name == "baseline" else ("  ✅ better" if d < -1e-4 else "  ~ neutral" if abs(d) <= 1e-4 else "  ❌ worse")
        print(f"  {name:<12} {avg:>9.4f}  Δ {d:>+8.4f}  {[round(x,4) for x in lls]}{verdict}")
        if cont:
            wide = np.mean(lls + list(cont.values()))
            print(f"               wide(7) = {wide:.4f}  {{ {', '.join(f'{k}:{v:.3f}' for k,v in cont.items())} }}")


if __name__ == "__main__":
    main()
