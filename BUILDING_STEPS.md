# Football Match Prediction System — Building Log

A narrative log of how this project came to be: the questions, the research, and the decisions that shaped the architecture.

---

## The Spark — Learning from Basic Pitch

The starting point was the Basic Pitch codebase (Spotify's audio-to-MIDI transcription library), which we had just been working in. The question came up naturally: could the same engineering discipline and architecture be applied to build a football match prediction model?

The answer was yes — with some important adaptations. The patterns that transfer directly are: a thin inference wrapper (a `Model` class with a `predict()` method), a clean separation between `predict()` and `predict_and_save()`, all constants centralised in one file, and tox for multi-environment testing. What doesn't transfer is the audio-specific machinery — windowing long signals into fixed-size chunks, supporting multiple serialisation formats (CoreML, TFLite, ONNX) for deployment across platforms. Football prediction inputs are small tabular feature vectors, so that complexity is unnecessary. The core pipeline simplifies to: feature vector → model → probability.

---

## Scope — What the System Should Be Able to Do

The ambition went beyond a simple model. The system should be grounded in current science — not just whatever happens to be convenient — which means doing real research into what approaches actually work in academic literature and production. It should integrate external data APIs, incorporate complementary models, support fine-tuning, and produce documentation clear enough for anyone to understand how it works.

The one honest constraint: heavy multi-day training runs can't be driven end-to-end in a single session. But everything else — research, architecture, implementation, evaluation, documentation — can be built and iterated on directly.

---

## The Key Design Decision — Modularity

The most important architectural choice, made before writing any code, was to treat the system as a **feature plugin registry**. The motivation was concrete: there are many signals that could improve predictions — FIFA/EA FC player ratings, real-time injury reports, squad fitness, weather at kickoff, betting market odds — but they vary in availability, quality, and relevance. A system that requires all of them at once is fragile. A system where each signal is an independent module that implements a simple interface (`fetch() → DataFrame`, `transform() → feature vector`) can start with three modules and grow to thirty without touching the core.

This maps directly onto what Basic Pitch got right: keep the inference core stable, make the inputs swappable.

---

## The Research — What Science Actually Says

A full research pass across academic literature (2022–2026), open-source production systems, and available data APIs produced the following picture.

### Models: Three Layers That Stack Well

**Layer 1 — Statistical Prior**

The original plan used Dixon-Coles (1997) as the statistical prior. After implementation, it was replaced for three reasons: (1) the multiplicative MLE form becomes ill-conditioned with 200+ international teams and sparse data — L-BFGS-B consistently hit iteration limits; (2) its home advantage parameter is irrelevant for neutral-venue WC prediction; (3) the low-score correction rho offers minimal benefit when team counts are high.

The replacement is a **Bayesian Hierarchical Poisson** model (Baio & Blangiardo 2010, *Journal of Applied Statistics*), using the log-linear additive form: `log(λ_h) = μ + att_h + def_a + home_adv`. Gaussian priors on attack/defence parameters act as L2 regularisation, shrinking sparse teams toward the mean. The corner constraint `Σ att = 0` ensures identifiability. L-BFGS-B with this parameterisation always converges. A `fit_mcmc()` variant using PyMC 5 is available for full posterior sampling (Rue & Salvesen 2000 style) when compute time allows.

The current academic gold standard extends the Poisson model into a **Bayesian state-space framework** (Koopman & Lit, *JRSS Series A*, 2015) where attack and defence strengths are tracked as dynamic latent states. This motivated the Extended Kalman Filter module (see below).

**Layer 2 — ML Refinement**

Gradient boosting (XGBoost, LightGBM, CatBoost) dominates both academic benchmarks and production systems. The best-known single-pipeline result is CatBoost combined with pi-ratings, achieving RPS 0.1925 and 55.82% accuracy on the M-COMP benchmark. Across all studies, the features that consistently matter most are: rolling form over the last 5 matches (the single most important feature by SHAP value in almost every study), pi-ratings or Elo differential, home advantage, rolling xG and xGA, bookmaker closing odds, EA FC player ratings (confirmed predictive in an ACM 2024 paper), and head-to-head record.

Neural networks are less competitive on football data — the tabular datasets are simply too small (around 380 EPL matches per season) for them to outperform gradient boosting without overfitting.

The deep learning state of the art is HIGFormer (Wang et al., accepted KDD 2025), a graph neural network that models player interactions as a heterogeneous graph combined with a transformer. It significantly outperforms prior methods but requires Wyscout or StatsBomb event-level data — it's the right thing to plug in later once that data is available.

**Layer 3 — Calibration**

This is the most underrated step. Research shows that a calibration-optimised model generates 69.86% higher simulated returns than an accuracy-optimised one. It is non-negotiable in any production-quality system.

The original plan used isotonic regression or Platt scaling. After implementation, **Temperature Scaling** (Guo et al., ICML 2017) was chosen instead: a single scalar T divides the log-probabilities before softmax. The reason: the calibration set for international football is small (~200 WC matches), and isotonic regression overfits badly on small datasets. Temperature Scaling has only one parameter (provably can't overfit) and is theoretically motivated — it's the correct transformation if the model's confidence is uniformly miscalibrated. In practice T ≈ 1.22, meaning XGBoost was overconfident by that factor.

**Layer 3b — Ensemble**

A scalar α ∈ [0,1] is learned to blend XGBoost (after temperature scaling) and BayesPoisson probabilities: `P = α·P_xgb + (1-α)·P_bp`. α is fitted by NLL minimisation on the calibration set. The model consistently learns α ≈ 0.60 — trusting XGBoost (data-driven, uses all 190 features) more than BayesPoisson (structural, uses only team identities and goal counts), but valuing the Poisson model's regularisation on sparse matchups.

### Features: What Actually Has Predictive Signal

The research confirmed that betting market odds are not just a comparison target — they are one of the top-three most informative features when included in an ML model directly. Markets are approximately but not fully efficient: an xG-based model demonstrated roughly 10% ROI against Bundesliga market odds over 11 seasons (Wilkens, 2026), confirming that process-based metrics capture signal the market doesn't fully price. The model learns to refine the market prior rather than predict from scratch. *(Odds not yet wired — future module.)*

EA FC / FIFA player ratings, while indirect, have been shown to carry real predictive power — they act as a compact encoding of squad quality that correlates with actual match performance across leagues and seasons. The EA FC 26 full attribute card (overall, pace, shooting, passing, defending, physicality) is now active via `sofifa_ratings.py` (33 features). The FIFA official WC 2026 squad list (club tier, average age, top-club ratio) is active via `squad_wc2026.py` (24 features).

**Rating systems added beyond Elo:**

*Glicko-2* (Glickman 2001) adds a reliability dimension (rating deviation φ) and a volatility term σ that the original Elo system lacks. Teams with few recent competitive matches get wider uncertainty intervals, which translates into softer win-probability estimates — important for international football where teams play only 10 matches/year.

*Extended Kalman Filter + RTS Smoother* (Koopman & Lit 2015) directly implements the Bayesian state-space model from the literature. State: `x = [att, def]` in log-goals space, evolving as a random walk. The Poisson observation model provides the Jacobian for the EKF update. The **Rauch-Tung-Striebel backward smoother** (Rauch et al. 1965) runs a second pass over historical data: it uses future matches to retroactively improve past strength estimates, giving XGBoost better training-time features. At prediction time only the causal forward pass is used — no data leakage. This is strictly better than Elvidge 2025's EKF variant, which uses Bradley-Terry binary outcomes rather than actual goal counts.

### Data: Free Sources That Cover Most Needs

The `soccerdata` Python package is the single best entry point — it wraps Club Elo, FBref (historical), Football-Data.co.uk, Sofascore, SoFIFA, Understat, and WhoScored into consistent Pandas DataFrames with local caching.

| Signal | Source | Cost |
|---|---|---|
| Results, standings, historical odds | football-data.org + football-data.co.uk | Free |
| xG, shot maps | Understat | Free (scraping) |
| Full event data (research) | StatsBomb Open Data | Free (non-commercial) |
| Elo ratings | Club Elo | Free |
| EA FC player ratings | SoFIFA via `soccerdata` | Free (scraping) |
| Live lineups, injuries | APIFootball | Free tier (100 req/day) |

---

## The Architecture

```
football-predictor/
│
├── constants.py              # all hyperparams, DEFAULT_FEATURE_MODULES list
├── inference.py              # Model class, predict(), predict_and_save()
│
├── features/                 # plugin system — each file = one signal source
│   ├── base.py               # FeatureModule interface (fetch + transform)
│   ├── elo.py                # classic Elo, K=40 intl / K=60 WC, neutral=no HA
│   ├── glicko2.py            # Glicko-2: rating + RD + volatility (Glickman 2001)
│   ├── kalman_strength.py    # EKF time-varying att/def + RTS backward smoother
│   ├── form.py               # rolling pts/goals/GD over 5/10/20 matches
│   ├── h2h.py                # head-to-head win rate, goals, last 10 meetings
│   ├── squad_strength.py     # 40-match rolling attack/defence from results
│   ├── confederation.py      # UEFA/CONMEBOL/CAF/AFC/CONCACAF strength encoding
│   ├── tournament_stage.py   # group/knockout, pressure multiplier, must-win flag
│   ├── rankings.py           # FIFA world ranking points (active)
│   ├── squad_wc2026.py       # FIFA official WC 2026 squad list (club tier, age)
│   ├── api_form.py           # last-10-match form from API-Football (pre-cached)
│   ├── sofifa_ratings.py     # EA FC 26: overall, pace, shooting, passing, etc.
│   ├── venue_wc2026.py       # partial home adv (MEX/USA/CAN), altitude, travel
│   │
│   ├── injury.py             # availability / fitness      ← stub, pass lists in
│   └── standings.py          # club standings              ← not used for WC
│
├── models/
│   ├── bayesian_poisson.py   # MAP + MCMC Bayesian Hierarchical Poisson
│   ├── gradient_boost.py     # XGBoost 3-class classifier
│   ├── calibration.py        # Temperature Scaling (single scalar T, Guo 2017)
│   ├── ensemble.py           # α-blend of XGB and BayesPoisson (NLL-fitted)
│   ├── dixon_coles.py        # legacy — kept for reference, not used in pipeline
│   └── gnn.py                # HIGFormer-style             ← later (needs event data)
│
├── data/
│   ├── sources/
│   │   ├── international_results.py   # martj42/international_results (47k+ matches)
│   │   └── football_data_org.py       # football-data.org REST API (EU leagues)
│   ├── wc2026.py             # groups, 72 fixtures, TEAM_NAME_MAP, normalise()
│   └── pipeline.py           # build_feature_matrix() — assembles feature modules
│
├── scripts/
│   ├── predict_wc2026.py     # main: train → predict → Monte Carlo → save output
│   ├── backtest.py           # evaluate on WC 2018 / 2022, SHAP, calibration plot
│   └── update_wc2026.py      # record actual results, refresh predictions live
│
├── evaluate/
│   └── metrics.py            # RPS, Brier score, log-loss, accuracy, ROI simulation
│
├── output/                   # prediction outputs, backtest reports, SHAP plots
└── tests/
```

---

## The World Cup Pivot

After the initial build, the target was clarified: **we're predicting the FIFA World Cup, not a domestic league**. This changes several assumptions and is worth documenting because it drives most of the subsequent architecture decisions.

### What's different about World Cup prediction

**Data volume.** International teams play roughly 10 matches per year versus 38 for a club side. A team that reaches the World Cup final will have played only ~60 competitive matches in the past 5 years. This means we need wider form windows (5/10/20 matches instead of 3/5/10), and we need to use *all* international fixtures — qualifiers, continental tournaments, Nations League — not just the World Cup itself, to have enough history to calibrate Elo ratings.

**No home advantage.** The World Cup is played at neutral venues. The Elo home advantage offset is set to 0 for international prediction.

**No league standings.** There's no league table to encode squad quality with. The primary quality signal switches to FIFA world rankings / Elo ratings computed from all international match history.

**Confederation effects.** UEFA and CONMEBOL teams historically overperform their Elo at World Cups; AFC and OFC teams underperform. This structural bias is worth encoding explicitly as a feature.

**Tournament stage.** Group-stage teams may rotate squads, tactically manage results, or protect leads differently than knockout-round teams. This changes expected goal rates and draw frequency in normal time.

**Data source.** The `martj42/international_results` GitHub dataset (47,000+ matches from 1872 to present, free CSV) replaces football-data.org as the primary training source. 8,306 competitive international matches from 2014 to 2021 are used for training; WC 2022 (64 matches) for evaluation.

### Feature modules added for World Cup

- `confederation.py` — encodes confederation membership and strength differential (UEFA/CONMEBOL vs AFC/CAF/OFC/CONCACAF)
- `tournament_stage.py` — encodes group stage vs knockout round, pressure multiplier, must-win flags
- `rankings.py` — FIFA world rankings (stubbed, pending reliable data source; Elo carries the same signal in the meantime)

### WC 2026 target

The 2026 World Cup (USA/Canada/Mexico) introduces a new format: 48 teams in 12 groups of 4, with a Round of 32 before the Round of 16. All group draw data and fixture generation logic lives in `data/wc2026.py`. Team name normalisation between the draw names and the `international_results` dataset names is tracked in `wc2026.TEAM_NAME_MAP`.

### Build Order

1. ✅ Project skeleton — constants, base interfaces, pyproject.toml, tox config
2. ✅ First data source — `football_data_org.py` (PL baseline)
3. ✅ First feature modules — `form.py`, `elo.py`, `standings.py`, `h2h.py`
4. ✅ Statistical prior — `dixon_coles.py` (with L-BFGS-B bounds fix)
5. ✅ Gradient boosting model — `gradient_boost.py` with XGBoost
6. ✅ Calibration layer — `calibration.py` (initially isotonic, upgraded to Temperature Scaling)
7. ✅ Evaluation harness — RPS, Brier score, log-loss, accuracy, ROI simulation
8. ✅ International data source — `international_results.py` (martj42, 47k matches)
9. ✅ World Cup feature modules — `confederation.py`, `tournament_stage.py`
10. ✅ WC 2026 fixture data — `data/wc2026.py` (12 groups, 72 fixtures, venue data)
11. ✅ FIFA rankings module — `rankings.py` (active)
12. ✅ EA FC / squad ratings — `sofifa_ratings.py` (EA FC 26, 33 features) + `squad_wc2026.py` (FIFA official squad list, 24 features)
13. ✅ Bayesian Hierarchical Poisson — `models/bayesian_poisson.py` (MAP + PyMC MCMC); replaced Dixon-Coles in active pipeline
14. ✅ Tournament simulator — Monte Carlo bracket simulation in `scripts/predict_wc2026.py` (50k sims, 48×48 pair-probability cache)
15. ✅ Glicko-2 rating system — `features/glicko2.py` (rating + deviation + volatility)
16. ✅ Extended Kalman Filter + RTS Smoother — `features/kalman_strength.py` (Koopman & Lit 2015 style; Joseph-form EKF + Rauch-Tung-Striebel backward pass)
17. ✅ Temperature Scaling — `models/calibration.py` (Guo et al. ICML 2017; replaced isotonic)
18. ✅ Ensemble model — `models/ensemble.py` (α-blend XGB + BayesPoisson, NLL-fitted, α≈0.60)
19. ✅ API-Football integration — `features/api_form.py` (pre-cached last-10-match form, api-sports.io)
20. ✅ Venue effects — `features/venue_wc2026.py` (partial home adv MEX/USA/CAN, altitude CDMX=2240m, travel burden by confederation)
21. ✅ Post-match live updating — `scripts/update_wc2026.py` + integration in `predict_wc2026.py`
22. ✅ Backtest — `scripts/backtest.py` (WC 2018/2022, log-loss, Brier, ECE, calibration plot, SHAP)
23. ⬜ GNN model — `models/gnn.py` (HIGFormer-style, requires Wyscout/StatsBomb event-level data)
24. ⬜ Betting odds module — `features/odds.py` (strongest single predictor; needs odds data source)
25. ⬜ Transfermarkt squad value — `features/transfermarkt.py` (confirmed predictive in Groll 2018)

---

## Key References

**Foundational models**
- Dixon & Coles (1997) — foundational Poisson goal model
- Baio & Blangiardo (2010) — Bayesian Hierarchical Poisson; *Journal of Applied Statistics* 37(2) — **active: `bayesian_poisson.py`**
- Karlis & Ntzoufras (2003) — bivariate Poisson (implemented in `footBayes`)
- Rue & Salvesen (2000) — MCMC for football; *JRSS Series C* — basis for `fit_mcmc()`

**Rating systems**
- Glickman, M.E. (2001) — Glicko-2 parameter system — **active: `glicko2.py`**
- Koopman, S.J. & Lit, R. (2015) — dynamic bivariate Poisson / state-space football; *JRSS Series A* 178(1) — **active: `kalman_strength.py`**
- Rauch, H.E., Tung, F. & Striebel, C.T. (1965) — RTS backward smoother; *AIAA Journal* 3(8) — **active: RTS pass in `kalman_strength.py`**
- Elvidge, S. (2025) — EKF Bradley-Terry for football rankings — informed EKF design (we use Poisson obs, not Bradley-Terry)
- Constantinou & Fenton (2013) — pi-ratings

**Machine learning & calibration**
- Guo, C. et al. (2017) — On calibration of modern neural networks; *ICML* — **active: `calibration.py` TemperatureScaling**
- Groll et al. (arXiv:2410.09068) — UEFA EURO 2024 stacked ensemble (confirmed ensemble α≈0.6 pattern)
- Lundberg, S. & Lee, S-I. (2017) — SHAP; *NeurIPS* — **active: `backtest.py`**

**Data & features**
- Wilkens (2026, SAGE) — xG vs. Bundesliga betting market; ~10% ROI over 11 seasons
- Bohner et al. (2015) — altitude effects on aerobic performance; *Sports Medicine* — basis for `venue_wc2026.py` altitude features

**Future**
- Wang et al. — HIGFormer (arXiv:2507.10626, KDD 2025) — graph neural network for football; requires event-level data
- Egidi, Karlis & Ntzoufras — *Predictive Modelling for Football Analytics* (Routledge, 2025)
- Bayesian state-space EPL model (*JRSS Series C*, Vol. 74, 2025)
