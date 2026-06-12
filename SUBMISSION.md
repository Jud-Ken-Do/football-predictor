# Pre-Submission Checklist — Occam's Folly

## 1. Install dependencies
```bash
pip install -e ".[dev]"
```

## 2. Refresh live data (if not done today)
```bash
python3 scripts/pipeline.py --fetch
```

## 3. Record any played matches
```bash
python3 scripts/update_wc2026.py --result "Mexico vs South Africa" --score "2-1" --group A
python3 scripts/update_wc2026.py list   # verify
```
Skip if no matches have been played yet.

## 4. Run the full pipeline with MCMC
```bash
python3 scripts/pipeline.py --mcmc --sims 100000
```
~5–7 min. Trains full model stack including MCMC posterior, runs 100k simulations.

## 5. Generate output.csv
```bash
python3 scripts/generate_submission.py --sims 50000
```
Writes `output/output.csv`. ~2 min.

## 6. Verify
```bash
python3 -c "
import csv
rows = list(csv.DictReader(open('output/output.csv')))
assert len(rows) == 72, f'Expected 72 rows, got {len(rows)}'
assert all(r['score1'] != '' and r['score2'] != '' for r in rows), 'Missing scores'
print(f'OK — {len(rows)} rows, all scores filled')
"
```

## 7. Zip
```bash
zip -r "Occam's Folly.zip" \
  football_predictor/ \
  scripts/pipeline.py \
  scripts/generate_submission.py \
  scripts/predict_wc2026.py \
  scripts/update_wc2026.py \
  templates/ \
  data/ \
  FC26_20250921.csv \
  world_cup_2026_squads_fifa_2026-06-08.csv \
  world_cup_2026_squads_fifa_2026-06-08.md \
  output/output.csv \
  explanation.md \
  pyproject.toml \
  requirements.txt \
  -x "**/__pycache__/*" "**/*.pyc" "**/.DS_Store"
```

Submit `Occam's Folly.zip`.
