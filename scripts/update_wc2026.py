#!/usr/bin/env python3
"""Record actual WC 2026 results and refresh remaining tournament odds.

Usage — record a result:
    python3 scripts/update_wc2026.py --result "Mexico vs South Africa" --score "2-1"
    python3 scripts/update_wc2026.py --result "France vs Uruguay" --score "3-0" --group A

Usage — just refresh predictions with already-recorded results:
    python3 scripts/update_wc2026.py --refresh

How it works:
  1. Appends the result to data/wc2026_actual_results.json
  2. The next call to predict_wc2026.py automatically reads this file, treats
     the played matches as known facts (bypasses Monte Carlo for those), and
     runs Monte Carlo only over the remaining fixtures.
  3. Kalman EKF receives the actual goals → strength estimates update in real time.

The update file format (JSON array of objects):
  [
    {"date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa",
     "home_goals": 2, "away_goals": 1, "group": "A"},
    ...
  ]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "data" / "wc2026_actual_results.json"
RESULTS_FILE.parent.mkdir(exist_ok=True)


def _load_results() -> list[dict]:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def _save_results(results: list[dict]) -> None:
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def _parse_match_string(s: str) -> tuple[str, str]:
    """Parse 'Team A vs Team B' → (home, away). Case-insensitive 'vs' separator."""
    parts = re.split(r"\s+vs\.?\s+", s, flags=re.IGNORECASE)
    if len(parts) != 2:
        print(f"Cannot parse match string: '{s}'. Expected 'Home Team vs Away Team'")
        sys.exit(1)
    return parts[0].strip(), parts[1].strip()


def _parse_score(s: str) -> tuple[int, int]:
    """Parse '2-1' → (2, 1)."""
    parts = s.strip().split("-")
    if len(parts) != 2:
        print(f"Cannot parse score: '{s}'. Expected 'goals_home-goals_away' e.g. '2-1'")
        sys.exit(1)
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        print(f"Score must be integers: '{s}'")
        sys.exit(1)


def _find_scheduled_fixture(home: str, away: str) -> tuple[dict | None, bool]:
    """Find the official fixture for a team pair.

    Returns (fixture, reversed) — reversed=True when the user typed the teams
    in the opposite orientation to the official schedule.
    """
    try:
        from football_predictor.data.wc2026 import GROUP_STAGE_SCHEDULE, normalise
        h, a = normalise(home), normalise(away)
        for m in GROUP_STAGE_SCHEDULE:
            mh, ma = normalise(m["home_team"]), normalise(m["away_team"])
            if (mh, ma) == (h, a):
                return m, False
            if (mh, ma) == (a, h):
                return m, True
    except Exception:
        pass
    return None, False


def cmd_record(args: argparse.Namespace) -> None:
    from football_predictor.data.wc2026 import normalise

    home, away = _parse_match_string(args.result)
    hg, ag = _parse_score(args.score)
    # Canonicalise so typed variants ("Iran", "USA") can't create phantom
    # teams in the rating systems.
    home, away = normalise(home), normalise(away)

    fixture, reversed_ = _find_scheduled_fixture(home, away)
    if fixture is None:
        print(f"  ⚠️  '{home} vs {away}' is not in the official WC 2026 group schedule.")
        confirm = input("  Record anyway (knockout/other)? (y/N): ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            return
    elif reversed_:
        # Store in the official orientation so merge/dedup keys always match.
        home, away = normalise(fixture["home_team"]), normalise(fixture["away_team"])
        hg, ag = ag, hg
        print(f"  (orientation swapped to match official schedule: {home} vs {away})")

    # Date: explicit flag > official schedule date > today.
    match_date = args.date or (fixture["date"] if fixture else str(date.today()))

    group = (args.group.upper() if args.group else
             (fixture["group"] if fixture else ""))

    entry: dict = {
        "date": match_date,
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "neutral": True,  # WC 2026 is all neutral venues
    }
    if group:
        entry["group"] = group

    results = _load_results()

    # Check for duplicate — same team pair in either orientation, any date
    # (a fixture only occurs once in the group stage).
    pair = {home, away}
    for r in results:
        if {normalise(r["home_team"]), normalise(r["away_team"])} == pair:
            print(f"  Result already recorded for {r['home_team']} vs {r['away_team']} on {r['date']}.")
            print(f"  Existing: {r['home_goals']}-{r['away_goals']}")
            overwrite = input("  Overwrite? (y/N): ").strip().lower()
            if overwrite != "y":
                print("  Aborted.")
                return
            results = [x for x in results if
                       {normalise(x["home_team"]), normalise(x["away_team"])} != pair]
            break

    results.append(entry)
    results.sort(key=lambda r: (r["date"], r["home_team"]))
    _save_results(results)

    result_str = "Win" if hg > ag else ("Draw" if hg == ag else "Loss")
    print(f"\n  Recorded: {home} {hg}-{ag} {away}  ({result_str})")
    print(f"  Saved to: {RESULTS_FILE}")
    print(f"  Total results on file: {len(results)}")
    print(f"\n  Run 'python3.11 scripts/predict_wc2026.py' to refresh predictions.")


def cmd_list(args: argparse.Namespace) -> None:
    results = _load_results()
    if not results:
        print("  No results recorded yet.")
        return
    print(f"\n  WC 2026 results on file ({len(results)} matches):")
    print(f"  {'Date':<12} {'Home':<22} {'Score':<8} {'Away':<22} {'Group'}")
    print("  " + "-" * 72)
    for r in sorted(results, key=lambda x: x["date"]):
        grp = r.get("group", "-")
        score = f"{r['home_goals']}-{r['away_goals']}"
        print(f"  {r['date']:<12} {r['home_team']:<22} {score:<8} {r['away_team']:<22} {grp}")


def cmd_refresh(args: argparse.Namespace) -> None:
    import subprocess
    predict_script = ROOT / "scripts" / "predict_wc2026.py"
    print("  Refreshing predictions with latest actual results...")
    subprocess.run([sys.executable, str(predict_script)], check=True)


def cmd_clear(args: argparse.Namespace) -> None:
    confirm = input(f"  Delete all recorded results from {RESULTS_FILE}? (y/N): ").strip().lower()
    if confirm == "y":
        _save_results([])
        print("  Cleared.")
    else:
        print("  Aborted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record WC 2026 actual results and refresh predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    p_record = sub.add_parser("record", help="Record a match result")
    p_record.add_argument("--result", required=True, help="e.g. 'Mexico vs South Africa'")
    p_record.add_argument("--score", required=True, help="e.g. '2-1'")
    p_record.add_argument("--group", default=None, help="Group letter e.g. A")
    p_record.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")

    sub.add_parser("list", help="List all recorded results")
    sub.add_parser("refresh", help="Run predict_wc2026.py with latest results")
    sub.add_parser("clear", help="Delete all recorded results")

    # Convenience: if --result and --score given at top level, treat as 'record'
    parser.add_argument("--result", default=None)
    parser.add_argument("--score", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--refresh", action="store_true")

    args = parser.parse_args()

    if args.cmd == "record" or (args.result and args.score):
        if not args.cmd:
            # top-level --result --score shorthand
            args.cmd = "record"
        cmd_record(args)
    elif args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "refresh" or args.refresh:
        cmd_refresh(args)
    elif args.cmd == "clear":
        cmd_clear(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
