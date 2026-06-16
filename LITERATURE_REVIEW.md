# Literature Review — International / World Cup Match-Outcome Prediction

**Compiled:** 2026-06-16 via deep-research (101 agents, fan-out web search → fetch →
3-vote adversarial verification → cited synthesis). Every claim below was verified
(needed 2/3 verifier votes to be killed); vote tallies and confidence are shown per finding.

**Scope:** what the published literature says about our exact situation — benchmarks to
judge our log-loss against, whether the 0.70 market blend is justified, methods/signals we
may be missing, modelling upset-heavy tournaments, and dynamic vs static ratings.

> ⚠️ **Master caveat:** almost all the strongest evidence (dynamic-model superiority,
> odds-beat-models, gradient-boosting+ratings) comes from **dense European club/league
> data**, NOT sparse international football. The few genuinely WC-specific findings warn
> that World Cups are so unpredictable that very simple Elo models can match or beat far
> more complex ones. Treat club-only findings as **directional**, not proven for our data.

---

## Bottom line

1. **Our ~0.95–1.0 group-stage log-loss is competitive** and consistent with the
   published international/WC literature.
2. **Our 0.70 market-1X2 blend is validated** — odds are the hardest-to-beat single
   predictor — though the model still adds marginal value (odds edge is small, ~2.6%
   relative RPS on EURO 2024).
3. **The biggest lever is new signals, not model tweaks** — explicitly stated by Koopman
   & Lit (the authors our Kalman model already cites).
4. **World Cup upset performance (our weak WC 2022) is largely irreducible** — complexity
   does not reliably help at World Cups.
5. We already use the **right primary metric** (log-loss > RPS, per Wheatcroft).

---

## 1. Benchmarks — is ~0.95–1.0 log-loss any good?

**Finding [high, votes 3-0 / 3-0 / 2-1 / 3-0 / 2-1]:**
Best-in-class individual-match RPS for WC / international 1X2 prediction is **~0.186–0.198**,
log-loss **~0.84–1.00**. Our assumed ~0.19–0.20 RPS ceiling is **confirmed** (nuance:
0.19–0.20 is partly a *club* ceiling; the WC-specific floor is ~0.186).

- Robberechts & Davis (WC 2002–2014): best **Elo ordered-logit RPS 0.1860 / log-loss 0.9375**;
  bookmakers 0.1877; Groll RF 0.1870.
- 2018 WC: FiveThirtyEight and bookmakers both **RPS 0.1976, log-loss ~0.846**.
- Groll et al. EURO 2024: combined model RPS 0.2015, bookmakers 0.1973.
- Bunker/Yeung/Fujii (OISD / 2017 Soccer Prediction Challenge): CatBoost+pi-ratings best at RPS 0.1925.

**Our numbers in context** (group-stage log-loss):

| Tournament | Ours | Lit. best-in-class |
|---|---|---|
| WC 2014 | 0.947 | ~0.84–0.94 (WC floor log-loss ~0.9375) |
| WC 2018 | 0.988 | ~0.846 |
| WC 2022 | 1.069 | (upset-heavy; even SOTA struggles) |
| Continental | 0.74–1.06 | — |

Sources:
- https://lirias.kuleuven.be/retrieve/7527e8cb-f047-4ef5-8395-89edd5ddc792 (Robberechts & Davis)
- https://arxiv.org/pdf/2410.09068 (Groll et al., EURO 2024)
- https://arxiv.org/pdf/2403.07669 (Bunker, Yeung & Fujii survey)

---

## 2. Do bookmaker odds beat models? (validating the 0.70 blend)

**Finding [high, votes 2-1 / 2-1 / 3-0 / 3-0]:**
Bookmaker odds are the **strongest single predictor** and systematically out-predict
statistical/rating models. *"Betting odds prior to a match possess more information than the
result known after the match."*

- Wunderlich & Memmert (PLOS ONE 2018, ~15k club matches): odds have highest predictive
  quality, beating ELO-Odds at **p < 0.0001**.
- Leitner/Zeileis/Hornik: odds beat Elo for EURO 2008 (international).

**Finding [high, votes 3-0] — the international-specific number that matters:**
On **EURO 2024**, bookmakers beat all four models (LASSO, c-forest, XGBoost, combined) on
*every* metric — **RPS 0.1973 vs 0.2015; classif. rate 0.5179 vs 0.4923; ML 0.4047 vs
0.3994; 101/195 vs 96/195 correct** — but the **margin is small (~0.004 RPS, ~2.6% relative)**.

→ **A high but not absolute market-blend weight is justified.** Our model still adds value.
Note: WC 2026 group-stage fixtures have **no live pre-tournament odds**, so the blend is a
no-op there anyway; it matters for backtests/continental.

**Finding [high, votes 2-1]:** Even with only past results as input, **no score-driven model
beats the bookmaker** in a betting simulation; *"improvements should be directed towards
finding more and better explanatory variables"* — Koopman & Lit (2019), the authors our
Kalman model cites. (Caveat: six European leagues, not international.)

Sources:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5988281/
- https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0198668
- https://arxiv.org/pdf/2410.09068
- https://www.sciencedirect.com/science/article/abs/pii/S0169207018302048

---

## 3. Methods / signals we may be missing

**Finding [high, 3-0] — Poisson-ability covariate into the tree.**
Feeding a strength-rating estimate (Poisson ranking ability parameters) into a tree ensemble
as an additional covariate **substantially improves** WC prediction over either alone.
Groll, Ley, Schauberger & Van Eetvelde (WC 2018 RF paper): *"by combining the random forest
with the team ability parameters from the ranking methods as an additional covariate we can
improve the predictive power substantially"* — the hybrid was their final model.
→ We feed Elo/Glicko/Kalman but **not a Poisson-ability covariate** distinct from our
BayesPoisson ensemble member. Worth adding to XGBoost directly.
- https://arxiv.org/pdf/1806.03208

**Finding [high, 3-0 / 3-0] — SOTA WC-2022 hybrid (Groll/Zeileis).**
Blends three sources via a conditional-inference random forest trained on five prior WCs:
(a) bivariate-Poisson historic-ability model, (b) a **28-bookmaker consensus** (overround-
adjusted, logit-averaged), (c) current-status covariates (market value, FIFA rank, team
structure, population, GDP). Two actionable gaps vs us: **(1) LEARN the market-blend weight
per-match via a tree** instead of fixing 0.70; **(2) add socioeconomic/team-structure
covariates** (we already have market value via transfermarkt + FIFA rank).
- https://www.zeileis.org/news/fifa2022/

**Finding [high, 3-0] — Bayesian Bradley-Terry-Davidson (BTD) ranking.**
BTD-derived log-strength as a covariate in Poisson goal models, **evaluated on WC 2022 and
AFCON 2023** (Macri Demartino, Egidi & Torelli, Computational Statistics 2025). We have **no
BTD** implementation (only incidental Bradley-Terry mentions in `glicko2.py:11`, `sos.py:23`).
A candidate new rating feature **validated on international data**.
- https://arxiv.org/pdf/2405.10247

**Finding [medium, 3-0] — fuse odds inside the generative model.**
Hierarchical Bayesian Poisson where scoring rates are a **convex combination of historical-
data parameters and odds-derived parameters** (Egidi, Pauli & Torelli, Statistical Modelling
2018). An alternative to our post-hoc 0.70 1X2 blend.
⚠️ The related claim that this yields positive betting profit while raw odds lose was
**REFUTED (0-3)** — use only as a fusion-architecture idea, not a profit claim. Club leagues.
- https://arxiv.org/pdf/1802.08848

**Finding [medium, 2-1] — CatBoost + pi-ratings.**
*"Gradient-boosted tree models such as CatBoost, applied to soccer-specific ratings such as
pi-ratings, are currently the best-performing models on datasets containing only goals"*
(Bunker, Yeung & Fujii 2024). CatBoost+pi RPS 0.1925. Ideas: try CatBoost vs XGBoost; add
pi-ratings as a feature. (Club benchmark.)
- https://arxiv.org/pdf/2403.07669

---

## 4. Upset-heavy / high-parity tournaments

**Finding [high, 3-0 / 3-0]:** In upset-heavy World Cups, **very simple models can match or
beat complex ones.** A basic Elo ordered-logit (Elo diff + home adv) beat bookmakers, the
Groll RF, and goal-based ODM ratings across WC 2002–2014 (RPS 0.1860 vs bookmakers 0.1877,
RF 0.1870). Goal-based ODM ratings ranked **last**. *"The outcome of any match is
unpredictable enough to confound these sophisticated computer models."*
→ Directly relevant to our weak WC 2022: **complexity does not reliably help; well-calibrated
simple probabilities are the realistic ceiling.** (Caveat: ≤256 matches, tiny margins.)
- https://lirias.kuleuven.be/retrieve/7527e8cb-f047-4ef5-8395-89edd5ddc792

---

## 5. Dynamic / Kalman vs Elo/Glicko + recent SOTA

**Finding [medium, 3-0 / 3-0] — adaptive Bayesian time-varying strength.**
Period-specific commensurate priors + spike-and-slab shrinkage **outperform static and naive
discrete-time dynamic** goal models (Macri-Demartino, Egidi & Torelli, arXiv:2508.05891,
`footBayes` R pkg). Supports our time-varying EKF — but **club data only** (Bundesliga/EPL/
La Liga). Adaptive shrinkage is a possible EKF enhancement.
- https://arxiv.org/pdf/2508.05891

**Finding [high, 3-0] — score-driven dynamic goal models.**
Dynamic bivariate Poisson (best forecast precision) and dynamic Skellam/goal-difference
(best for betting) both beat benchmarks (Koopman & Lit 2019). Suggests a **Skellam (goal-
difference) head** as a betting-oriented alternative. (Six European leagues.)
- https://www.sciencedirect.com/science/article/abs/pii/S0169207018302048

**Finding [medium, 2-1] — GBDT on ratings is the goals-only SOTA.** See §3 CatBoost+pi.

---

## 6. Metric choice

**Finding [high, 3-0 / 3-0]:** The **logarithmic/ignorance score is preferable to RPS** for
probabilistic football forecasts; RPS's distance-sensitivity adds nothing toward the aims of
scoring rules (Wheatcroft, J. Quant. Analysis in Sports 2022; corroborated by penaltyblog
2025). → Our use of **log-loss as primary is already the better choice**; keep RPS only for
cross-paper comparability.
- https://arxiv.org/pdf/1908.08980

---

## Open questions (gaps the literature did NOT settle for us)

1. **Does our EKF actually beat Elo/Glicko for INTERNATIONAL football?** All dynamic-superiority
   evidence is club-only; the one WC-specific head-to-head favored *simple* Elo. Our EKF's
   marginal value on sparse international data is **unproven** → worth an ablation
   (Kalman-on vs Elo-only on the WC-3 backtest).
2. **What is the optimal / per-match LEARNED market-blend weight** vs our fixed 0.70?
   (Moot for WC 2026 group stage — no pre-tournament odds — but relevant for backtests.)
3. **Which NEW explanatory variables** most help sparse international prediction? Relative lift
   of squad/player-level, fatigue/travel, socioeconomic covariates is not isolated for WCs.
4. **Is there any method that beats well-calibrated probabilities in upset-heavy tournaments,**
   or is the unpredictability an irreducible ceiling (as Robberechts & Davis imply)?

---

## Recommended actions (ranked)

1. **Add BTD + Poisson-ability covariates to XGBoost** (§3 findings 1–2) — international-
   validated, cheap, gateable with `scripts/feature_ablation.py`. Highest ROI.
2. **Ablation: EKF vs Elo-only** on the WC-3 set (open question 1) — could simplify the stack.
3. **Keep prioritizing new signals** (xG for CAF/CONCACAF, squad/player-level) over model
   architecture — Koopman & Lit: signals beat models.
4. Optional: learn the market-blend weight (§3 finding 2); try CatBoost / pi-ratings (§3 finding 5).
5. **Do NOT** chase WC-2022-style upset performance with more model complexity — largely
   irreducible.

---

## Cross-check vs our own experiments

- **Odds-feature ablation (2026-06-16):** removing the XGB `odds` feature was neutral-to-
  slightly-worse (WC-3 1.0014 → 1.0038; wide-7 0.9470 → 0.9454). Consistent with the
  literature — odds carry real signal; keep the feature. The earlier R4 side-lead
  (1.0071→0.9995) did **not** reproduce; it was 2014-specific noise.

## All sources

- https://lirias.kuleuven.be/retrieve/7527e8cb-f047-4ef5-8395-89edd5ddc792
- https://arxiv.org/pdf/2410.09068
- https://arxiv.org/pdf/2403.07669
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5988281/
- https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0198668
- https://www.sciencedirect.com/science/article/abs/pii/S0169207018302048
- https://arxiv.org/pdf/1806.03208
- https://www.zeileis.org/news/fifa2022/
- https://arxiv.org/pdf/2405.10247
- https://arxiv.org/pdf/1802.08848
- https://arxiv.org/pdf/2508.05891
- https://arxiv.org/pdf/1908.08980
