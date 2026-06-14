# Missing Signals — Data Shopping List

Candidate signals **not** already in the model, prioritised, with sources and the
one rule that decides whether each is worth collecting. Deduped against what the
stack already has: Elo (MOV) / Glicko-2 / Kalman EKF strength, form, SoS, H2H,
squad market value, squad age, club tier, FIFA ranking, closing odds, partial xG
(UEFA/AFC/CONMEBOL), sofifa player ratings, injuries, venue/altitude/travel.

---

## The rule that governs everything here

> **A signal is only a real (trainable) feature if you can get its _historical_
> value at past match dates.** Snapshot-only signals (just today's 2026 value)
> cannot go into XGBoost — they would be covariate shift / leakage, which is
> exactly why `sofifa` and `squad_wc2026` are post-processing nudges, not
> training features. **When collecting data, find the _history_, not just the
> current number.** A signal you can't backtest is one our (thin) eval can't
> validate.

`Trainable` below = historical values obtainable → real XGB `FeatureModule`.
`Snapshot` = current-only → at best a `wc_context` post-processing nudge.

---

## Tier 0 — Free wins (no data hunt; computable from existing match history)

These are feature engineering on `international_results` — **no external data
needed**. Do these first; they're real trainable features and tell us whether
this _class_ of signal moves the metric at all before sourcing anything.

| Signal | Why it helps | Status |
|---|---|---|
| ~~**Rest days / congestion**~~ | days since last match + matches in last 30d → fatigue/freshness | ✅ **DONE 2026-06-14** (`features/rest.py`, kept: wide backtest 0.9512→0.9481) |
| **Qualification dominance** | pts %, goal diff, win rate in this team's qualifying campaign → "cruised in vs scraped in" | ✅ Trainable, free |
| **Penalty-shootout strength** | historical shootout W/L + conversion rate → fixes knockout coin-flips | ✅ Trainable, free |

## Tier 1 — High value, trainable, easy-ish to source

| Signal | Why | Source | Collect |
|---|---|---|---|
| **All-time WC pedigree** | appearances / titles / best-finish → tournament overperformance (GER/BRA/ARG/URU beat their ratings at WCs) | Wikipedia, RSSSF | ✅ inherently historical |
| **Manager tenure / stability** | months under current manager at match date → settled setups travel well | Transfermarkt, Wikipedia | ✅ "manager-since" dates per team |
| **GK shot-stopping** | (xG faced − goals conceded) rolling → outsized in low-scoring knockout games | derive where xG exists; FBref GK tables | ✅ where xG exists |

## Tier 2 — Biggest actual hole: fill the xG gap ✅ SUBSTANTIALLY DONE (2026-06-14)

**Closed via StatsBomb open-data** (`data/sources/statsbomb.py`), consumed as a
Kalman observation rather than an XGB feature (see CLAUDE.md → Kalman EKF; flagged
by `USE_XG_OBSERVATION=True`). 12 of the 13 previously-uncovered teams now carry
calibrated xG (AFCON 2023 → 9 CAF; Copa 2024 → MEX/USA/CAN; + WC 2022/2018, Euro
2020/2024). **Only New Zealand remains uncovered.** Ablation: 4/5 covered folds
improved, none worse (within-CI / consistency-based; CAF/CONCACAF gain for 2026 is
an extrapolation). Remaining open: NZ xG, and fresher 2025-26 qualifier xG (would
need a manual FBref pull — Cloudflare blocks automation).

Original gap (now mostly filled): Morocco, Senegal, Egypt, Algeria, Ghana, Ivory
Coast, Cape Verde, DR Congo, Tunisia, USA, Mexico, Canada — all covered; Panama,
Curaçao, Haiti were already covered by the Excel source.

- Sources: StatsBomb Open Data (free, some internationals), Understat, FBref
  (Cloudflare-blocked but scrapable with effort), or paid Opta/StatsBomb.
- Even partial CAF/CONCACAF coverage helps. Find **historical** match xG.

## Tier 3 — Snapshot-only (collect last; can't be validated well)

new.py-style signals. Worth it **only if collected historically**; otherwise
just another unvalidatable nudge.

| Signal | Why | Caveat |
|---|---|---|
| **Set-piece / aerial threat** (avg height, set-piece goal %) | stable team trait; matters in tight games | height ~static; set-piece % needs event data |
| **Squad cohesion** (caps of likely XI, % shared club, minutes together) | settled cores overperform | snapshot unless historical lineups assembled |
| **Ballon d'Or elite concentration** | top-end individual ceiling | overlaps existing `sofifa star_rating`; only worth it if collected **historically** (goes back decades) → then trainable |

---

## Recommended order

1. **Tier 0** — free, immediate, real trainable features. Build + backtest first.
2. **Tier 2** — xG gap; highest leverage.
3. **Tier 1** — pedigree, manager, GK.
4. **Tier 3** — only if historical.

## Honest caveat

The headline eval is 3 World Cups / 144 matches with ±0.1 CIs, and
`friendly_weight` was tuned on the scored years. Small per-feature effects will
be **indistinguishable from noise** on this eval. Strongly consider folding the
continental backtests (Copa/Euro/AFCON/Asian Cup) into the headline metric
*before* judging new signals — otherwise we can't tell signal from luck.

## Integration spec (so collected data drops straight into a FeatureModule)

Each new module extends `FeatureModule` (`football_predictor/features/base.py`):
- `fetch(competition, seasons) → DataFrame`
- `transform(match, data) → dict[str, float]`
- register in `features/__init__.py::REGISTRY`
- add to `DEFAULT_FEATURE_MODULES` (trainable) **or** `WC_CONTEXT_MODULES` (snapshot) in `constants.py`

Data files should key on **canonical team name** (`data/wc2026.py::normalise`) and,
for trainable signals, carry a **date** so the value can be looked up as-of each
historical match (no future leakage).
