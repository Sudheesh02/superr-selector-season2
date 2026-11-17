# SupeRR Selector Season 2 — Most Reliable Players Under Pressure

Reproducible ball-by-ball analytics pipeline for **Rajasthan Royals — SupeRR Selector Season 2 (Round 1)**.

## Methodology Highlights
- **Data Audit & Pre-Ball State (`01_data_audit_and_state.py`)**: Fixes `extras1_runs` (actual extra runs `1–5`) and `revised_overs` (stored in balls), builds zero-leakage pre-ball scoreboard states, computes shrunk venue scoring factors (`0.897–1.162`), and flags Elite matches (`367 IPL + 417 Full-Member T20I`).
- **Win-Probability Leverage Pressure Index (`02_pressure_and_ratings.py`)**: Fits a 2nd-innings Logistic Win Probability model (`ROC-AUC = 0.9031`, `Brier = 0.1263`), suppresses garbage time (`WP < 4%` or `> 96%`), and combines Match Leverage with RRR stress, collapse shock, and knockout stakes (`PI >= 45` captures `17.87%` of balls with `2.52x` 2nd-innings leverage and `+31.9%` wicket probability).
- **Context-Adjusted Impact & Empirical Bayes (`02_pressure_and_ratings.py`)**: Derives empirical Wicket Run Cost from the 20-over Run Expectancy matrix, applies Empirical Bayes shrinkage (`k = 120`), floored uplift vs. normal baseline, and match-by-match consistency (`150+ HP balls`, `300+ Normal balls`, `12+ HP matches`).
- **Sensitivity & Charts (`03_sensitivity_and_charts.py`)**: Stress-tests against all matches (`r = 0.995 / 0.997`), stricter `PI >= 50`, and chronological split-half (`2022–mid24` vs `mid24–26`), and generates `220 DPI` figures + Excel workbook.

## Reproduce Locally
1. Place the 6 official hackathon CSVs (`ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `teams.csv`, `playing_eleven.csv`, `competition_team_player_mapping.csv`) into a `./data` folder.
2. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   python run_all.py --data-dir ./data
   ```
