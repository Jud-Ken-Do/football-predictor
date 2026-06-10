#!/usr/bin/env python3.11
"""Derive data-driven confederation strength offsets from historical Elo ratings.

Rather than hardcoding CONFEDERATION_STRENGTH offsets (UEFA=+50, CAF=-10 etc.),
this script runs a full Elo pass on historical data with NO starting offsets,
then computes:
  1. Mean Elo per confederation at end of each calendar year (trend)
  2. Cross-confederation head-to-head win rates vs. expected (calibration check)
  3. Recommended relative offsets (mean Elo deviations from global mean)

The script PRINTS recommendations — it does not modify constants.py automatically.
Review the output and update CONFEDERATION_STRENGTH manually if the data supports it.

Usage:
    python3.11 scripts/calibrate_confederations.py
    python3.11 scripts/calibrate_confederations.py --from-year 2000
    python3.11 scripts/calibrate_confederations.py --plot          # requires matplotlib
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from football_predictor.data.sources.international_results import fetch_training_data

# ── Confederation membership ───────────────────────────────────────────────────

_TEAM_CONFEDERATION: dict[str, str] = {
    # UEFA (Europe)
    "Albania": "UEFA", "Andorra": "UEFA", "Armenia": "UEFA", "Austria": "UEFA",
    "Azerbaijan": "UEFA", "Belarus": "UEFA", "Belgium": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Bulgaria": "UEFA", "Croatia": "UEFA", "Cyprus": "UEFA", "Czechia": "UEFA",
    "Czech Republic": "UEFA", "Denmark": "UEFA", "England": "UEFA", "Estonia": "UEFA",
    "Faroe Islands": "UEFA", "Finland": "UEFA", "France": "UEFA", "Georgia": "UEFA",
    "Germany": "UEFA", "Gibraltar": "UEFA", "Greece": "UEFA", "Hungary": "UEFA",
    "Iceland": "UEFA", "Ireland": "UEFA", "Israel": "UEFA", "Italy": "UEFA",
    "Kazakhstan": "UEFA", "Kosovo": "UEFA", "Latvia": "UEFA", "Liechtenstein": "UEFA",
    "Lithuania": "UEFA", "Luxembourg": "UEFA", "Malta": "UEFA", "Moldova": "UEFA",
    "Montenegro": "UEFA", "Netherlands": "UEFA", "North Macedonia": "UEFA", "Northern Ireland": "UEFA",
    "Norway": "UEFA", "Poland": "UEFA", "Portugal": "UEFA", "Romania": "UEFA",
    "Russia": "UEFA", "San Marino": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Spain": "UEFA", "Sweden": "UEFA",
    "Switzerland": "UEFA", "Türkiye": "UEFA", "Turkey": "UEFA", "Ukraine": "UEFA",
    "Wales": "UEFA",
    # CONMEBOL (South America)
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL", "Venezuela": "CONMEBOL",
    # CONCACAF (North/Central America & Caribbean)
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "Panama": "CONCACAF", "Trinidad and Tobago": "CONCACAF", "El Salvador": "CONCACAF",
    "Cuba": "CONCACAF", "Haiti": "CONCACAF", "Guatemala": "CONCACAF",
    "Curaçao": "CONCACAF", "Curacao": "CONCACAF", "Suriname": "CONCACAF",
    # AFC (Asia)
    "Australia": "AFC", "Bahrain": "AFC", "China PR": "AFC", "China": "AFC",
    "DR Congo": "CAF",  # misclassification guard
    "India": "AFC", "Indonesia": "AFC", "IR Iran": "AFC", "Iran": "AFC",
    "Iraq": "AFC", "Japan": "AFC", "Jordan": "AFC", "Kuwait": "AFC",
    "Kyrgyzstan": "AFC", "Lebanon": "AFC", "Malaysia": "AFC", "Myanmar": "AFC",
    "Oman": "AFC", "Palestine": "AFC", "Philippines": "AFC", "Qatar": "AFC",
    "Saudi Arabia": "AFC", "Singapore": "AFC", "South Korea": "AFC",
    "Syria": "AFC", "Tajikistan": "AFC", "Thailand": "AFC", "Uzbekistan": "AFC",
    "United Arab Emirates": "AFC", "Vietnam": "AFC", "Yemen": "AFC",
    # CAF (Africa)
    "Algeria": "CAF", "Angola": "CAF", "Benin": "CAF", "Burkina Faso": "CAF",
    "Cabo Verde": "CAF", "Cameroon": "CAF", "Cape Verde": "CAF",
    "Central African Republic": "CAF", "Comoros": "CAF", "Congo": "CAF",
    "Côte d'Ivoire": "CAF", "Ivory Coast": "CAF", "DR Congo": "CAF",
    "Egypt": "CAF", "Ethiopia": "CAF", "Gabon": "CAF", "Gambia": "CAF",
    "Ghana": "CAF", "Guinea": "CAF", "Guinea-Bissau": "CAF", "Kenya": "CAF",
    "Liberia": "CAF", "Libya": "CAF", "Madagascar": "CAF", "Malawi": "CAF",
    "Mali": "CAF", "Mauritania": "CAF", "Morocco": "CAF", "Mozambique": "CAF",
    "Namibia": "CAF", "Niger": "CAF", "Nigeria": "CAF", "Rwanda": "CAF",
    "São Tomé and Príncipe": "CAF", "Senegal": "CAF", "Sierra Leone": "CAF",
    "Somalia": "CAF", "South Africa": "CAF", "Sudan": "CAF", "Tanzania": "CAF",
    "Togo": "CAF", "Tunisia": "CAF", "Uganda": "CAF", "Zambia": "CAF", "Zimbabwe": "CAF",
    # OFC (Oceania)
    "New Zealand": "OFC", "Fiji": "OFC", "Solomon Islands": "OFC", "Vanuatu": "OFC",
    "Papua New Guinea": "OFC",
}


def _conf(team: str) -> str:
    return _TEAM_CONFEDERATION.get(team, "OTHER")


# ── Elo engine (no confederation offsets) ─────────────────────────────────────

_K = 40.0
_DEFAULT_ELO = 1500.0


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def run_elo_pass(df: pd.DataFrame) -> dict[str, float]:
    """Forward Elo pass with uniform starting ratings — no confederation bias."""
    ratings: dict[str, float] = {}
    df = df.sort_values("date").reset_index(drop=True)

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        mw = float(row.get("match_weight", 1.0))
        r_h = ratings.get(home, _DEFAULT_ELO)
        r_a = ratings.get(away, _DEFAULT_ELO)
        exp_h = _expected(r_h, r_a)

        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        ratings[home] = r_h + _K * mw * (s_h - exp_h)
        ratings[away] = r_a + _K * mw * ((1 - s_h) - (1 - exp_h))

    return ratings


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(df: pd.DataFrame, plot: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  CONFEDERATION CALIBRATION ANALYSIS")
    print("=" * 60)

    # Full Elo pass
    ratings = run_elo_pass(df)

    # Mean Elo per confederation
    conf_ratings: dict[str, list[float]] = defaultdict(list)
    for team, elo in ratings.items():
        c = _conf(team)
        if c != "OTHER":
            conf_ratings[c].append(elo)

    global_mean = np.mean(list(ratings.values()))
    print(f"\n  Global mean Elo (no offsets): {global_mean:.1f}")
    print(f"  {'Confederation':<12} {'N':>4} {'Mean Elo':>10} {'Offset vs global':>16}")
    print("  " + "-" * 46)

    conf_means: dict[str, float] = {}
    for conf in ["UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF", "OFC"]:
        elos = conf_ratings.get(conf, [])
        if elos:
            mean_elo = np.mean(elos)
            conf_means[conf] = mean_elo
            offset = mean_elo - global_mean
            print(f"  {conf:<12} {len(elos):>4} {mean_elo:>10.1f} {offset:>+16.1f}")

    # Cross-confederation H2H calibration
    print("\n  Cross-confederation H2H win rates vs. Elo expected:")
    print(f"  {'Match-up':<22} {'N':>5} {'Actual W%':>10} {'Elo Exp W%':>12} {'Gap':>8}")
    print("  " + "-" * 60)

    pairs = [
        ("UEFA", "CONMEBOL"), ("UEFA", "CONCACAF"), ("UEFA", "CAF"),
        ("UEFA", "AFC"), ("CONMEBOL", "CONCACAF"), ("CONMEBOL", "CAF"),
        ("CONMEBOL", "AFC"), ("CONCACAF", "CAF"), ("CAF", "AFC"),
    ]
    ratings_final = ratings
    for c1, c2 in pairs:
        cross = df[
            (df["home_team"].map(_conf) == c1) & (df["away_team"].map(_conf) == c2) |
            (df["home_team"].map(_conf) == c2) & (df["away_team"].map(_conf) == c1)
        ]
        if len(cross) < 5:
            continue
        wins_c1 = actual_exp = 0
        n = 0
        for _, row in cross.iterrows():
            h_conf = _conf(row["home_team"])
            is_c1_home = (h_conf == c1)
            c1_team = row["home_team"] if is_c1_home else row["away_team"]
            c2_team = row["away_team"] if is_c1_home else row["home_team"]
            hg, ag = row["home_goals"], row["away_goals"]
            c1_won = (hg > ag) if is_c1_home else (ag > hg)
            wins_c1 += int(c1_won)
            r1 = ratings_final.get(c1_team, _DEFAULT_ELO)
            r2 = ratings_final.get(c2_team, _DEFAULT_ELO)
            actual_exp += _expected(r1, r2)
            n += 1
        actual_wr = wins_c1 / n
        expected_wr = actual_exp / n
        gap = actual_wr - expected_wr
        label = f"{c1} vs {c2}"
        print(f"  {label:<22} {n:>5} {actual_wr:>10.3f} {expected_wr:>12.3f} {gap:>+8.3f}")

    # Recommended offsets (relative to CONCACAF=0 baseline, matching constants.py convention)
    print("\n  Recommended CONFEDERATION_STRENGTH offsets (relative to CONCACAF=0):")
    concacaf_mean = conf_means.get("CONCACAF", global_mean)
    print("  {")
    for conf in ["UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF", "OFC"]:
        if conf in conf_means:
            offset = round(conf_means[conf] - concacaf_mean)
            current = {"UEFA": 50, "CONMEBOL": 40, "CONCACAF": 0, "AFC": -10, "CAF": -10, "OFC": -30}.get(conf, 0)
            flag = "  ← change from " + str(current) if abs(offset - current) > 5 else ""
            print(f'    "{conf}": {offset},{flag}')
    print("  }")

    if plot:
        _plot_elo_history(df)


def _plot_elo_history(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    ratings: dict[str, float] = {}
    year_conf_means: dict[int, dict[str, float]] = {}

    for year, group in df.groupby("year"):
        run_elo_pass(group)  # incremental is complex; approximate with full pass per year
        conf_elos: dict[str, list] = defaultdict(list)
        for team, elo in run_elo_pass(df[df["year"] <= year]).items():
            c = _conf(team)
            if c != "OTHER":
                conf_elos[c].append(elo)
        year_conf_means[int(year)] = {c: np.mean(v) for c, v in conf_elos.items() if v}

    years = sorted(year_conf_means)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"UEFA": "blue", "CONMEBOL": "green", "CONCACAF": "orange",
               "AFC": "red", "CAF": "purple", "OFC": "gray"}
    for conf in ["UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF"]:
        ys = [year_conf_means[y].get(conf, np.nan) for y in years]
        ax.plot(years, ys, label=conf, color=colors.get(conf, "black"), linewidth=2)

    ax.set_title("Mean Elo per Confederation Over Time (no starting offsets)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Elo")
    ax.legend()
    ax.grid(alpha=0.3)
    out = ROOT / "output" / "confederation_elo_history.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Derive confederation strength offsets from data")
    parser.add_argument("--from-year", type=int, default=2000)
    parser.add_argument("--plot", action="store_true", help="Save Elo history plot")
    args = parser.parse_args()

    print(f"  Loading matches from {args.from_year}...")
    df = fetch_training_data(from_year=args.from_year)
    df["date"] = pd.to_datetime(df["date"])
    # Competitive only for calibration (friendlies add noise)
    comp = df["tournament"].str.lower().str.contains(
        "qualif|world cup|copa|euro|nations|africa|asian|gold cup", na=False
    )
    df = df[comp].reset_index(drop=True)
    print(f"  {len(df):,} competitive matches loaded.")

    analyse(df, plot=args.plot)


if __name__ == "__main__":
    main()
