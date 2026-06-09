from __future__ import annotations

import argparse
import json
import logging
import os

from football_predictor.constants import DEFAULT_FEATURE_MODULES, ModelType, SUPPORTED_COMPETITIONS
from football_predictor.data.sources.football_data_org import fetch_matches
from football_predictor.inference import predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict football match outcome probabilities.")
    parser.add_argument("home_team", type=str, help="Name of the home team.")
    parser.add_argument("away_team", type=str, help="Name of the away team.")
    parser.add_argument(
        "--competition", type=str, default="PL",
        choices=list(SUPPORTED_COMPETITIONS),
        help="Competition code (default: PL — Premier League).",
    )
    parser.add_argument(
        "--seasons", type=str, nargs="+", default=["2022", "2023"],
        help="Seasons to train on (default: 2022 2023).",
    )
    parser.add_argument(
        "--model", type=str, choices=[m.value for m in ModelType], default=ModelType.GRADIENT_BOOST.value,
        help="Model type to use.",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="football-data.org API key (or set FOOTBALL_DATA_ORG_API_KEY env var).",
    )
    parser.add_argument(
        "--modules", type=str, nargs="+", default=DEFAULT_FEATURE_MODULES,
        help="Feature modules to activate.",
    )

    args = parser.parse_args()

    matches = fetch_matches(args.competition, args.seasons, api_key=args.api_key)
    result = predict(
        args.home_team,
        args.away_team,
        matches,
        active_modules=args.modules,
        model_type=ModelType(args.model),
    )

    print(json.dumps({"home_team": args.home_team, "away_team": args.away_team, "probabilities": result}, indent=2))


if __name__ == "__main__":
    main()
