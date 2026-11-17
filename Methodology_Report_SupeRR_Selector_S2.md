# SupeRR Selector Season 2 (Round 1) — Methodology Report
**Identify the Most Reliable Players Under Pressure (2022–2026 IPL & T20I Data)**

---

## 1. Executive Summary & Submission Rankings

To identify the most reliable T20 batters and bowlers under pressure, we built an end-to-end, ball-by-ball analytics pipeline across **867 matches (200,999 deliveries)** from IPL (2022–2026) and T20 Internationals (2022–2026). Rather than relying on raw strike rates, economy rates, or arbitrary heuristics, we anchor pressure in **Empirical Win-Probability Leverage** (`AUC = 0.9031`, `Brier = 0.1263`) and evaluate performance via **Venue- & Context-Adjusted Net Run Impact per 100 Balls**, regularized with **Empirical Bayes Shrinkage ($k = 120$)** and **Match-by-Match Consistency**.

### Top 10 Batters Under Pressure (Exact Unstop Dropdown Names)
| Rank | Player Name | HP Balls | Qual. HP Innings | Normal SR | HP SR | EB Normal Impact/100b | EB HP Impact/100b | Positive HP Innings | Clutch Bat Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Shreyas Iyer** | 265 | 17 | 149.18 | 177.74 | +8.54 | +37.43 | 82.35% | **82.18** |
| **2** | **Heinrich Klaasen** | 216 | 15 | 165.36 | 174.54 | +20.91 | +28.65 | 80.00% | **70.34** |
| **3** | **Tim David** | 453 | 30 | 181.91 | 169.54 | +30.45 | +32.42 | 70.00% | **69.14** |
| **4** | **Towhid Hridoy** | 211 | 12 | 124.92 | 131.28 | -2.88 | +21.95 | 75.00% | **66.68** |
| **5** | **Andre Russell** | 255 | 21 | 175.10 | 163.53 | +23.82 | +26.40 | 76.19% | **66.42** |
| **6** | **Namboori Tilak Varma** | 542 | 34 | 144.10 | 143.17 | +10.25 | +23.23 | 70.59% | **64.52** |
| **7** | **Sikandar Raza** | 262 | 17 | 133.11 | 142.37 | +7.10 | +22.03 | 70.59% | **64.01** |
| **8** | **Mohammad Nabi** | 180 | 13 | 124.51 | 144.44 | -5.57 | +19.75 | 69.23% | **62.86** |
| **9** | **Daryl Mitchell** | 161 | 14 | 141.56 | 161.49 | +0.75 | +20.08 | 64.29% | **61.59** |
| **10** | **Suryakumar Yadav** | 342 | 26 | 166.08 | 160.53 | +29.14 | +22.56 | 69.23% | **59.55** |

### Top 10 Bowlers Under Pressure (Exact Unstop Dropdown Names)
| Rank | Player Name | HP Balls | Qual. HP Spells | Normal Econ | HP Econ | EB Normal Impact/100b | EB HP Impact/100b | Positive HP Spells | Clutch Bowl Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Sunil Narine** | 282 | 27 | 7.27 | 5.09 | +20.60 | +50.81 | 85.19% | **94.09** |
| **2** | **Varun Chakravarthy** | 459 | 41 | 8.51 | 6.26 | +18.21 | +31.80 | 75.61% | **75.73** |
| **3** | **Mohammad Nawaz** | 210 | 17 | 7.56 | 5.74 | +11.10 | +31.68 | 64.71% | **73.63** |
| **4** | **Mustafizur Rahman** | 331 | 32 | 8.35 | 6.60 | +4.31 | +24.30 | 78.12% | **72.67** |
| **5** | **Kuldeep Yadav** | 417 | 36 | 8.32 | 7.09 | +13.30 | +30.51 | 63.89% | **72.02** |
| **6** | **Sikandar Raza** | 171 | 14 | 7.47 | 5.40 | -4.29 | +28.93 | 57.14% | **71.04** |
| **7** | **Lockie Ferguson** | 171 | 17 | 9.52 | 6.42 | +2.75 | +22.33 | 76.47% | **70.80** |
| **8** | **Rashid Khan** | 468 | 40 | 7.62 | 6.73 | +18.21 | +25.31 | 65.00% | **67.12** |
| **9** | **Shaheen Afridi** | 196 | 22 | 8.32 | 7.74 | -0.22 | +19.79 | 68.18% | **66.69** |
| **10** | **Gudakesh Motie** | 222 | 16 | 8.94 | 6.57 | -4.83 | +23.60 | 56.25% | **66.31** |

---

## 2. Data Audit & Pre-Ball State Reconstruction

Before modeling, we audited all 6 raw CSV files and resolved two critical data-encoding quirks:
1. **Exact Extras Accounting (`extras1_runs` & `extras_2_runs`)**: `extras1_runs` stores the actual primary extra runs (e.g., 1–5 for wides, byes, and legbyes; 1 for no-balls; 5 for penalty runs), while `extras_2_runs` stores secondary extra runs (e.g., byes/legbyes off a no-ball). Thus `total_ball_runs = runs + extras1_runs + extras_2_runs`, and `bowler_runs` includes wides, no-balls, and "other" extras while excluding byes, legbyes, and penalty runs.
2. **DLS / Rain-Reduced Overs Stored in Balls**: `first_inn_revised_overs` and `second_inn_revised_overs` in `matches.csv` are stored in **balls** (e.g., `90`, `48`, `114`), not overs. Treating them directly as `max_balls` ensures accurate `balls_remaining` and Required Run Rate (`RRR`) in rain-affected matches.
3. **Zero-Leakage Pre-Ball State**: All scoreboard variables (`cum_runs_before`, `cum_wkts_before`, `balls_remaining_before`, `rrr_before`, `crr_before`, and `wkt_last_12b` collapse shock) are computed strictly prior to each delivery.

---

## 3. Pressure Definition & Empirical Validation

We define a unified **Pressure Index (`PI`, 0–100)** that captures match leverage and situational stress across both innings:

- **2nd Innings (Win-Probability Leverage + Chase Stress)**:
  - We fit a Logistic Regression Win Probability model on all 2nd-innings deliveries with known match outcomes using 6 pre-ball features (`rrr - venue_par_rpo`, `runs_needed / sqrt(balls_rem)`, `wickets_in_hand`, `log(1 + balls_rem)`, and interaction terms), achieving **ROC-AUC = 0.9031** and **Brier Score = 0.1263**.
  - **Ball Leverage** is defined as `WP(if 4 runs, 0 wkt) - WP(if 0 runs, 1 wkt)`.
  - Crucially, **"garbage time"** (`WP < 0.04` or `WP > 0.96`) is suppressed so hopeless chases and one-sided cakewalks do not inflate pressure stats.
  - `PI` combines Win-Prob Leverage (0–50 pts), RRR vs Venue-Par Stress (0–25 pts), Wicket Scarcity / Collapse Shock (0–15 pts), and Knockout/Playoff Stakes (0–10 pts).
- **1st Innings (Collapse + Par Deficit + Death Stakes)**:
  - Combines Wicket Scarcity & Recent 12-Ball Collapse Shock (0–38 pts), Scoring-Rate Deficit vs Venue-Adjusted Par & Competitive Death-Over Acceleration (0–37 pts), Innings Stage (0–15 pts), and Knockout/Playoff Stakes (0–10 pts).
- **Super Overs (`innings >= 3`)**: Assigned `PI = 100.0`.
- **High-Pressure Threshold (`PI >= 45`)**: Captures the top **17.87%** of deliveries (`35,920` of `200,999` balls), balanced across 1st and 2nd innings.

### Empirical Validation (Ordinary vs. High Pressure)
| Context Tier | Total Balls | Share of Dataset | Run Rate (RPO) | Wicket Prob (%) | Dot Ball (%) | Mean 2nd-Inn WP Leverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ordinary (`PI < 45`)** | 165,079 | 82.13% | 8.599 | 5.121% | 33.776% | 0.039 |
| **High Pressure (`PI >= 45`)** | 35,920 | 17.87% | 7.962 | 6.757% | 34.159% | **0.110 (2.82x)** |

Under High Pressure (`PI >= 45`), scoring drops by **0.64 RPO**, wicket probability spikes by **+31.9%** (`5.12%` → `6.76%`), and 2nd-innings Win-Probability Leverage is **2.82x higher** (`0.110` vs `0.039`).

---

## 4. Performance Measurement & Baseline Comparison

Raw strike rates and economy rates are confounded by phase (Powerplay vs. Death), wickets fallen, format (IPL vs. T20I), and pitch scoring environment. We therefore measure **Context- & Venue-Adjusted Net Run Impact**:

1. **Shrunk Venue Scoring Factor**: Each venue's innings-level runs-per-ball relative to format par is shrunk toward `1.0` (`k = 10` innings), ranging from `0.897` (sluggish pitches) to `1.162` (high-scoring venues).
2. **Empirical Wicket Run Cost (`WRC`)**: Derived directly from the 20-over empirical Run Expectancy Matrix `RE(over, wicket_tier)` as the marginal expected runs lost per wicket (clamped to `[3.5, 22.0]`), rather than a hardcoded linear guess.
3. **Ball-Level Net Run Impact**:
   - **Batter**: `(runs - exp_bat_runs * venue_factor) - WRC * (is_striker_out - exp_bat_out)`
   - **Bowler**: `(exp_bowl_runs * venue_factor - bowler_runs) + WRC * (is_bowler_wicket - exp_bowl_wkts)`
4. **Empirical Bayes Shrinkage ($k = 120$ balls)**: Both High-Pressure and Normal Impact per 100 balls are shrunk toward the qualified population mean to prevent small-sample outliers.
5. **Composite Clutch Score (Mean 50, SD 15)**:
   $$\text{Clutch Score} = 50 + 15 \times \left(0.60 \cdot z(\text{EB HP Impact}) + 0.15 \cdot z(\text{Uplift vs Par}) + 0.25 \cdot z(\text{Positive HP Match \%})\right)$$
   where `Uplift vs Par = EB_HP_Impact - max(0, EB_Normal_Impact)`, ensuring players are rewarded for rising above par under pressure without artificially rewarding weak normal-overs baselines.

---

## 5. Fair Comparisons, Sensitivity & Reliability

### Competition Tier & Eligibility
- **Elite Competition Filter**: To prevent Associate-nation or Emerging-team ("A" team) mismatches from inflating ratings for an IPL auction context, primary rankings evaluate the **784 Elite matches (182,064 balls = 90.6% of dataset)** comprising all 5 IPL seasons (2022–2026) and Full-Member vs. Full-Member T20Is.
- **Eligibility Thresholds**: Minimum **150 High-Pressure balls** (~25 overs), **300 Normal balls** (~50 overs), and **12 qualifying High-Pressure matches** (at least 5 HP balls in the match), yielding **42 qualified batters** and **67 qualified bowlers**.

### Sensitivity & Temporal Reliability Stress Tests
| Model Variant | HP Deliveries | Qual. Batters | Top-10 Bat Overlap | Spearman $r$ (Bat) | Qual. Bowlers | Top-10 Bowl Overlap | Spearman $r$ (Bowl) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Primary): Elite IPL+FM, `PI >= 45`** | 32,182 | 42 | 10 / 10 (Baseline) | 1.0000 | 67 | 10 / 10 (Baseline) | 1.0000 |
| **Model B: All Matches (incl. Associate/Emerging)** | 35,920 | 46 | 8 / 10 | 0.9948 | 70 | 9 / 10 | 0.9968 |
| **Model C: Stricter Threshold (`PI >= 50`, min 120 HP)** | 21,530 | 29 | 6 / 10 | 0.6889 | 50 | 5 / 10 | 0.7656 |
| **Model D: Temporal Split (`2022–mid24` vs `mid24–26`)** | 32,182 | 30 | Out-of-Sample | 0.1488 | 48 | Out-of-Sample | 0.4684 |

### Key Limitations
1. **Temporal Variance in Batting (`r = 0.1488`) vs. Bowling (`r = 0.4684`)**: Split-half out-of-sample correlation is substantially higher for bowlers than batters, reflecting the higher inherent variance of T20 high-pressure batting (single-ball dismissal risk) compared to 4-over bowling spells.
2. **Role Selection Effects**: Specialist death bowlers and lower-middle-order finishers accumulate more high-pressure deliveries than top-order openers or powerplay-only bowlers; minimum sample thresholds (`150 HP`, `300 Normal`) mitigate but do not eliminate role differences.
3. **1st-Innings Win Probability**: Because 1st-innings targets are not yet set, 1st-innings pressure relies on par-deficit and collapse proxies rather than direct chase Win-Probability Leverage.
