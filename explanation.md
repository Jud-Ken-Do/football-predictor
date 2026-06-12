# Occam's Folly — WC 2026 Group Stage Predictor

**Model:** XGBoost + Kalman EKF + Glicko-2 + Bayesian Poisson + Context-Adaptive Ensemble + Market Odds  
**Output:** Always 1–0 or 0–1

---

## What it's based on

We kept adding layers of sophistication — Kalman filters, Dixon-Coles corrections, Monte Carlo simulations — until Algeria made it out of Group J. At that point we stopped.

We also have a function called `_invert_over25` that does binary search to find what Poisson rate implies a given over-2.5 probability. The over-2.5 line. From Asian Handicap data. For a group stage prediction competition. At work. We are not sure when this stopped being funny.

---

## The five layers, explained without mercy

**1. XGBoost** — trained on ~49,000 international matches across ~120 features. These include: Glicko-2 ratings (Glickman 2001 — rating μ, deviation φ, volatility σ, Illinois algorithm update; if you know what that means, you are already having a worse time than you expected when you entered this competition), an Extended Kalman Filter (Koopman & Lit 2015) modelling time-varying attack and defence strength in log-goals space (process noise q tuned via the Shumway-Stoffer EM algorithm because the default wasn't good enough; trained on forward-only causal states — the RTS smoother uses future data and was not invited), rolling form over 5/10/20 matches, head-to-head records, strength-of-schedule, Transfermarkt squad valuations, FIFA rankings, xG where available (not for African teams — the data doesn't exist and FBref is Cloudflare-protected, so Senegal gets Elo), and confederation strength offsets derived from cross-confederation win rates because the mean-Elo approach was inflated by insular regional tournaments. Both Glicko-2 and the Kalman filter apply match-importance weights: World Cup matches count 1.5×, qualifiers and continental tournaments 1.0×, Nations League 0.7×, and friendlies 0.3× — a weight that is itself auto-tuned via grid search and cached, because apparently we couldn't decide how seriously to take friendlies either. Near-duplicate features are pruned at |r| > 0.95 before training. Hyperparameters are tuned via Optuna against WC 2018 and 2022 as held-out validation folds.

It predicts home win / draw / away win probabilities.

**2. Temperature Scaling** — a single number T that divides the log-probabilities to make the model less overconfident. This is the only component that a reasonable person would consider proportionate to the task.

**3. Bayesian Hierarchical Poisson** — a log-linear goal model (Baio & Blangiardo 2010) with a Dixon-Coles ρ correction for low-scoring cells, because 0–0 and 1–0 are slightly more common than a pure Poisson model predicts and we were not going to let that slide. The model's time-decay half-life is derived from the Kalman-tuned q, so both models share the same empirical assumption about how fast team strength drifts. This is called timescale consistency. It took a while.

**4. Context-adaptive ensemble** — rather than blending XGBoost and Poisson at a fixed 60/40, the blend weight α is a per-match sigmoid over odds availability, Kalman uncertainty, and H2H depth, trained by minimising negative log-likelihood with L2 regularisation. When none of those signals exist it falls back to a scalar. The current split is roughly 73/27 in favour of XGBoost.

**5. WC 2026 post-processing** — because the above wasn't enough:
   - *Venue*: Mexico, USA, and Canada get partial home advantage in their own stadiums. Mexico City gets an altitude adjustment for 2,240m. We checked whether any players are acclimatised.
   - *Market odds*: For all 72 fixtures we fetched pre-match Asian Handicap lines and Over/Under 2.5 from API-Football. The AH line gives expected goal difference; O/U 2.5 gives expected total goals. Together they define a market-implied Poisson model that we blend in at 20% weight. For the five matches where bookmakers couldn't agree on an AH line, we preserve the model's own directional signal and only anchor the total goals to the O/U. This is called "the AH fallback" and it has a unit test.
   - *Squad quality*: a small log-odds nudge from EA FC 26 ratings and last-10-match form from API-Football.
   - *Injuries*: star player absences weighted by squad market value. Rodrygo is out for Brazil (cruciate, back in September). Estêvão is also out until late July.

Then we run 50,000 Monte Carlo simulations, propagate Kalman posterior uncertainty through lognormal λ resampling, and pick the scoreline that maximises expected competition points — with the eight-best-third-place advancement bonus optimised jointly across all 12 groups simultaneously, because picking groups independently is suboptimal and we had already come this far. There is also a full MCMC posterior mode (PyMC 5, non-centered parameterisation, zero divergences) available via `--mcmc`. We did not use it for the submission because it takes five minutes and the point estimates don't move much. We built it anyway.

The output is always 1–0 or 0–1. No draws. We checked: for this scoring system (5/3/2/0 pts), predicting a draw is never the expected-value-maximising choice when P(draw) ≈ 27% and P(non-draw) ≈ 73%. The 2-pt base bonus for getting the direction right across the majority of outcomes always beats the 3-5 pts you'd get from draws on the minority. We ran the simulation both ways to be sure.

---

## Setup & Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Fast (MAP, ~60s)
python3 scripts/generate_submission.py

# Full posterior (MCMC on top of MAP, ~5 min — what we used)
python3 scripts/generate_submission.py --mcmc
```

The first run auto-tunes XGBoost hyperparameters via Optuna (WC 2018 & 2022 as validation folds) and the friendly match weight via grid search, then caches both to `data/xgb_tuned_params.json` and `data/tuned_params.json`. Every subsequent run loads from cache silently. You never need to think about this, which is more than can be said for everything else here.

Open `notebooks/tutorial.ipynb` for a step-by-step walkthrough.

---

## Updating predictions as results come in

```bash
# After each match
python3.11 scripts/update_wc2026.py --result "Mexico vs South Africa" --score "2-1" --group A

# Check what's been recorded
python3.11 scripts/update_wc2026.py list

# Refresh — locks played matches, updates Kalman states, re-simulates the rest
python3.11 scripts/update_wc2026.py --refresh
```

---

## File structure

Everything is here. We are not sorry.

| File | Purpose |
|---|---|
| `output/output.csv` | The actual submission. 72 scorelines. Mostly 1–0. |
| `scripts/generate_submission.py` | Regenerates output.csv. Runs in ~2 min. |
| `scripts/update_wc2026.py` | Feed in actual results; re-runs predictions with them locked in |
| `scripts/predict_wc2026.py` | Full prediction pipeline |
| `scripts/pipeline.py` | Orchestrator: fetch → load → features → train → predict → simulate |
| `notebooks/tutorial.ipynb` | Step-by-step walkthrough if you want to understand any of this |
| `data/wc2026_actual_results.json` | Actual results (empty; the tournament just started) |
| `data/wc2026_odds_cache.json` | Pre-match bookmaker odds, all 72 fixtures |
| `data/xgb_tuned_params.json` | Optuna-tuned XGBoost hyperparameters |
