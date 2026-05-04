# FIFA World Cup 2026 — Match Predictor

A Streamlit dashboard that predicts every match of the 2026 FIFA World Cup
(Canada / Mexico / USA) using historical World Cup data and a Poisson model.

## Pages

- **Dashboard** — every fixture in chronological order with a probability bar
  (win / draw / loss), expected goals and the most-likely scoreline. Played
  matches show the actual scoreline alongside the prediction.
- **Match Predictor** — pick any two 2026 teams and inspect a full score-line
  probability grid plus head-to-head history.
- **Team Profiles** — historical record, recent form and goal timeline.
- **Tournament Simulator** — runs every group-stage match. Completed matches
  use the actual scoreline; future matches use the model.
- **Live Results** — upload a CSV of 2026 results. Tracks predicted vs actual
  scorelines, outcome accuracy, exact-score hits and average model confidence.
- **Methodology** — how the model works and how debutant teams are handled.

## Files required in repo root

```
app.py
requirements.txt
world-cup-Data-all-matches-2.csv
WCup_2026_4.2.5_en.xlsx
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Updating with live 2026 results

As games are played you can feed results back into the model:

1. Go to the **Live Results** page and download the pre-filled template
   (`results_2026_template.csv`).
2. Fill in `home_score` and `away_score` for each played match.
3. Save and upload the CSV via the **Upload results CSV** control in the
   sidebar — predictions for future matches refresh immediately.

The CSV schema is:

```
match_no,home_team,away_team,home_score,away_score
1,Mexico,South Africa,2,0
7,Brazil,Morocco,3,1
```

You can also commit the file to your repo as `results_2026.csv` and the app
will pick it up automatically on startup.

Use the **Use 2026 results in the model** sidebar toggle to switch between
live-updating predictions and the original pre-tournament forecast.

## Debutant-team strategies

Cape Verde, Curaçao, Jordan and Uzbekistan have no World Cup history.
Five fallback approaches are documented in the **Methodology** page —
the default is a tournament-baseline projection with a small penalty
(`debutant_strength = 0.85`). Swap in any other strategy by editing
`expected_goals()` in `app.py`.
