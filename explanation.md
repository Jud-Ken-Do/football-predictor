# Occam's Folly — WC 2026 Group Stage Predictor

**Model:** XGBoost + Kalman EKF + Glicko-2 + Bayesian Poisson + Context-Adaptive Ensemble

## What it's based on

We kept adding layers of sophistication — Kalman filters, Dixon-Coles corrections, Monte Carlo simulations — until Algeria made it out of Group J. At that point we stopped.

The model stacks five layers, each added in good faith and with complete disregard for parsimony:

1. **XGBoost** trained on ~47,000 international matches (2010–2026) across 231 features: Glicko-2 ratings, an Extended Kalman Filter tracking time-varying attack/defence strength (process noise tuned via EM — yes, we tuned the noise), rolling form, H2H record, Transfermarkt squad values, FIFA rankings, xG data, and confederation strength offsets. Correlated features pruned before training.
2. **Temperature Scaling** — a single learned scalar on the XGBoost outputs. The one part of this model Occam would have approved of.
3. **Bayesian Hierarchical Poisson** (MAP) with Dixon-Coles ρ correction and a half-life derived from the Kalman-tuned q, so both models share the same assumption about how fast team strength drifts.
4. **Context-adaptive ensemble** blending XGBoost and Poisson via a per-match sigmoid weight responding to odds availability, Kalman uncertainty, and H2H depth.
5. **WC 2026 post-processing** — venue adjustments for host nations, altitude penalty for Mexico City, EA FC 26 squad quality nudge, injury-weighted market value penalty. This is the point at which a reasonable person would have stopped. We did not.

Rather than predicting the most likely scoreline, we run 30,000 Monte Carlo simulations and pick whichever scoreline maximises expected points under the 5/3/2/0 rubric, with the +3 advancement bonus optimised across all 12 groups simultaneously. This is called calibration.

---

## Setup & Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python3 scripts/generate_submission.py   # regenerates output/output.csv
```

The first time you run the full pipeline (`python3 scripts/pipeline.py`), it automatically tunes two parameters against historical World Cup data and caches the results — XGBoost hyperparameters via Optuna (using WC 2018 & 2022 as validation folds) and the friendly match weight via grid search. All subsequent runs, including `generate_submission.py`, load from cache. You never need to think about this.

Or open the notebook in `notebooks/` for a step-by-step walkthrough.

---

## Recording actual results (as the group stage plays out)

```bash
# Record a result
python3 scripts/update_wc2026.py --result "Mexico vs South Africa" --score "2-1" --group A

# List all recorded results
python3 scripts/update_wc2026.py list

# Refresh predictions with results locked in
python3 scripts/update_wc2026.py --refresh
```

Results are stored in `data/wc2026_actual_results.json`. The model locks played matches to actual scores, updates the Kalman EKF with the new goals data, and re-simulates only the remaining fixtures.

---

## File structure

Everything is here. We are not sorry.

| File | Purpose |
|---|---|
| `output/output.csv` | Competition submission — 72 predicted scorelines |
| `scripts/generate_submission.py` | Regenerate output.csv (Monte Carlo score optimiser) |
| `scripts/update_wc2026.py` | Record actual results + refresh predictions |
| `scripts/predict_wc2026.py` | Full prediction pipeline (called by above) |
| `notebooks/occams_folly.ipynb` | Interactive walkthrough |
| `data/wc2026_actual_results.json` | Actual results feed (empty until matches are played) |
