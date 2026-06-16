from football_predictor.features.base import FeatureModule
from football_predictor.features.form import FormFeatures
from football_predictor.features.rest import RestFeatures
from football_predictor.features.elo import EloFeatures
from football_predictor.features.glicko2 import Glicko2Features
from football_predictor.features.kalman_strength import KalmanStrengthFeatures
from football_predictor.features.sos import StrengthOfScheduleFeatures
from football_predictor.features.standings import StandingsFeatures
from football_predictor.features.h2h import H2HFeatures
from football_predictor.features.rankings import RankingsFeatures
from football_predictor.features.confederation import ConfederationFeatures
from football_predictor.features.tournament_stage import TournamentStageFeatures
from football_predictor.features.squad_strength import SquadStrengthFeatures
from football_predictor.features.injury import InjuryFeatures
from football_predictor.features.squad_wc2026 import SquadWC2026Features
from football_predictor.features.api_form import APIFormFeatures
from football_predictor.features.sofifa_ratings import SoFIFARatingsFeatures
from football_predictor.features.venue_wc2026 import VenueWC2026Features
from football_predictor.features.odds import OddsFeatures
from football_predictor.features.wc2026_market import WC2026MarketFeatures
from football_predictor.features.xg_form import XGFormFeatures
from football_predictor.features.transfermarkt import TransfermarktFeatures
from football_predictor.features.qualification import QualificationFeatures
from football_predictor.features.wc_pedigree import WCPedigreeFeatures

REGISTRY: dict[str, type[FeatureModule]] = {
    "form": FormFeatures,
    "rest": RestFeatures,
    "elo": EloFeatures,
    "glicko2": Glicko2Features,
    "kalman_strength": KalmanStrengthFeatures,
    "sos": StrengthOfScheduleFeatures,
    "standings": StandingsFeatures,
    "h2h": H2HFeatures,
    "rankings": RankingsFeatures,
    "confederation": ConfederationFeatures,
    "tournament_stage": TournamentStageFeatures,
    "squad_strength": SquadStrengthFeatures,
    "injury": InjuryFeatures,
    "squad_wc2026": SquadWC2026Features,
    "api_form": APIFormFeatures,
    "sofifa_ratings": SoFIFARatingsFeatures,
    "venue_wc2026": VenueWC2026Features,
    "odds": OddsFeatures,
    "wc2026_market": WC2026MarketFeatures,
    "xg_form": XGFormFeatures,
    "transfermarkt": TransfermarktFeatures,
    "qualification": QualificationFeatures,
    "wc_pedigree": WCPedigreeFeatures,
}

__all__ = [
    "FeatureModule", "REGISTRY",
    "FormFeatures", "EloFeatures", "Glicko2Features", "KalmanStrengthFeatures", "StrengthOfScheduleFeatures",
    "StandingsFeatures", "H2HFeatures", "RankingsFeatures", "ConfederationFeatures",
    "TournamentStageFeatures", "SquadStrengthFeatures", "InjuryFeatures",
    "SquadWC2026Features", "APIFormFeatures", "SoFIFARatingsFeatures",
    "VenueWC2026Features", "OddsFeatures", "XGFormFeatures", "TransfermarktFeatures",
]
