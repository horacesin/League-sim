# Football league simulator

A dependency-free Monte Carlo simulator. Give every team one goal-margin power rating, load a schedule, and get estimated season odds. Scores use a bivariate Poisson model with an optional draw adjustment.

## Ratings

Use a goal-margin power rating, normally from `0` (best) to `3` (weakest); negative values such as `-0.1` are valid. **Lower is better.** The supremacy midpoint is calculated directly as:

```text
away rating - home rating + 0.20 home advantage
```

For example, a home team rated `-0.1` against an away team rated `0.8` has an expected supremacy of `+1.10` goals. The model reports probabilities for these outcomes:

- **Supremacy:** expected `home goals - away goals`; it settles negative if the away team wins.
- **Goals:** expected `home goals + away goals`.

It gives percentages for home/draw/away, total goals (`0` through `7+`), and supremacy goal margins (`-3 or less` through `+3 or more`). The total-goals model starts from a league-wide 2.70-goal assumption. For very large supremacy ratings, it rises just enough to retain realistic, non-negative expected scores.

## Run it

```powershell
python league_simulator.py teams.example.json fixtures.example.csv --simulations 100000 --seed 42 --top-n 4 --relegation-n 3 --output odds.json
python league_simulator.py teams.example.json fixtures.example.csv --match "North City" Rovers
```

The first command returns each team's title, top-four, and relegation probabilities, plus average finishing position and points. The second gives match-outcome percentages and expected goals.

## Score distribution settings

The default model uses no score correlation and a `1.098` draw-inflation factor. It was fitted against the supplied supremacy/goals-to-1X2 examples. Set `--draw-inflation` or `--score-correlation` to test alternatives; additional examples would allow a stronger calibration.

## Your data

Copy `teams.example.json` and set one `rating` for each team. Make a CSV schedule with exactly these headers:

```csv
home,away
North City,Rovers
```

All team names in the schedule must exactly match the JSON names. Fixtures can be a full season or only the remaining matches, so you can use the same engine later with an interface that records results already played.

## UI direction

The engine is deliberately separated from the interface. A natural next step is a lightweight Streamlit app: editable ratings table, CSV upload/paste for fixtures, simulation count, sortable odds table and individual fixture quotes. It would require only a small UI layer; no model rewrite.
