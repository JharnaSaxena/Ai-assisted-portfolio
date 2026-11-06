# AI-Assisted Portfolio Selection Using Neural Networks

CF End-Term group project. We test whether a neural network that predicts next-week stock returns can help build better portfolios than traditional methods, using Indian (NSE) stocks and a weekly walk-forward back-test.

> **Research question:** Does feeding neural-network return predictions into a Mean-Variance optimizer produce better risk-adjusted portfolios than (a) equal weighting and (b) Mean-Variance with historical-average returns? A well-supported "no, it does not" is a valid result.

## 1. The three strategies compared

| # | Strategy | Expected returns come from | Risk estimate |
| --- | --- | --- | --- |
| 1 | Equal weight | not used (1/N each) | not used |
| 2 | Traditional Mean-Variance | average of the last 52 weekly returns | covariance, last 52 weeks |
| 3 | **AI-assisted Mean-Variance** | **neural-network prediction for next week** | same covariance as #2 |

Strategies 2 and 3 differ **only** in where expected returns come from. The covariance method, risk-aversion parameter and constraints must be identical, so any difference is attributable to the neural network.

**Constraints (strategies 2 and 3):** weights sum to 100%, no short selling, at most 30% in any one stock. Rebalanced **weekly**.

## 2. The one rule that matters most: no data leakage

A decision made at the end of week *t* may use only information available at week *t*.

- Chronological split only, never random: **train 2017-01 → 2022-12 · validation 2023-01 → 2024-12 · test 2025-01 → 2026-09**.
- Scalers are fitted on **training rows only** (`Data/data_out/scaler_params_train_only.json`).
- Covariance and historical means at week *t* use only rows ≤ *t*.
- The test period is not used to choose models, features or hyper-parameters.
- The target (`target_ret_next`, the return of week *t+1*) is never an input.
- Performance of the portfolio chosen at *t* is measured on the actual return of week *t+1*.

## 3. Team structure and status

| Team | Responsibility | Gives | Status |
| --- | --- | --- | --- |
| A – Research \& Methodology | Literature review (Gu, Kelly \& Xiu 2020), research question, stock list | Methodology, stock list | \[ \] |
| **B – Data** | Weekly NSE dataset, features, target, splits, scaler | `Data/data_out/` files | **\[x\] built and audited** · pending: Team A sign-off on stock list |
| C – Traditional models | Equal weight, Mean-Variance, common optimizer function | Optimizer + weights | \[ \] |
| D – Neural network | Feedforward net, MAE/RMSE vs 52-week-average benchmark | Predicted returns | \[ \] |
| E – AI portfolio, back-test, presentation | Combines C and D, weekly back-test, results, report, demo | Results, final report | \[ \] |

Team members: `[A: names]` · `[B: names]` · `[C: names]` · `[D: names]` · `[E: names]`

## 4. Repository layout

```
Ai-assisted-portfolio/
├── README.md                      # this file: project overview and team interfaces
├── LICENSE
├── src/
│   └── build_dataset.py           # Team B: data pipeline (run first)
├── docs/
│   └── data_pipeline.md           # Team B README: conventions, data dictionary, viva questions
├── Data/
│   ├── data_raw/                  # cached daily prices (reproducible offline rebuilds)
│   └── data_out/                  # Team B outputs, consumed by C, D, E
│       ├── model_dataset.csv
│       ├── scaler_params_train_only.json
│       ├── weekly_prices.csv
│       ├── weekly_returns_pct.csv
│       ├── summary_stats_weekly_return.csv
│       ├── data_quality_report.txt
│       └── eda_*.png
├── optimizer/                     # Team C (planned)
├── nn/                            # Team D (planned)
├── backtest/                      # Team E (planned)
└── report/                        # literature review, final report, reflections
```

Folders marked *planned* do not exist yet. Adjust names as teams settle, but keep this file in sync.

## 5. How to run

```
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib yfinance        # add scipy / torch / scikit-learn as Teams C, D need them

python src/build_dataset.py --selftest             # must print three PASS lines
python src/build_dataset.py --raw-dir Data/data_raw --out-dir Data/data_out   # builds Data/data_out
```

- The data is **frozen at 2026-10-09** (`--asof`), so every teammate gets the same file.
- The daily CSVs already in `Data/data_raw/` are the cache from that run, so the command above works **offline**. Add `--refresh` to re-download from Yahoo, or `--source csv` to force the cached files and skip `yfinance` entirely.
- Rebuilding from the cached daily files reproduces every file in `Data/data_out/` byte for byte (checked; only the `source=` line of the quality report changes). It takes about 10 seconds.

Environment used for the dataset: Python 3.14.4, pandas 3.1.1, numpy 2.5.4, matplotlib 3.11.2, yfinance 1.8.0. Fix random seeds in every model script and record library versions in the report.

## 6. Data summary

- **Stocks (8 sectors):** RELIANCE, TCS, HDFCBANK, HINDUNILVR, SUNPHARMA, LT, MARUTI, BHARTIARTL (NSE), plus Nifty 50 as a market feature.
- **Source:** Yahoo Finance via `yfinance`, auto-adjusted for splits, bonus issues and dividends.
- **Weekly convention:** close of the last trading day of the week, labelled by that Friday. Volume is the weekly sum.
- **Rows:** 4,064 = 8 stocks × 508 weeks (313 train, 104 validation, 91 test per stock). No missing values: 0 weeks carried forward, 0 rows dropped.
- **Units:** all returns are in **percent** (1.00 means +1.00%).
- **One week flagged for review:** MARUTI, week ending 2020-04-10, +32.8%. Checked against the daily file (₹3,772.61 → ₹5,009.44 across that week): a real post-COVID rebound, not an unadjusted corporate action.

| Column | Meaning (known at end of week t) |
| --- | --- |
| `ret_1w` | Return in the latest completed week |
| `ret_4w` | Compounded return over the previous 4 weeks |
| `vol_8w` | Standard deviation of the last 8 weekly returns (sample, ddof=1) |
| `nifty_ret_1w` | Nifty 50 return in the latest week |
| `volume_ratio_8w` | Weekly volume ÷ average of the last 8 weeks (use this, not raw volume) |
| `target_ret_next` | Return in week t+1. **Prediction target only.** |

Full details, assumptions and checks are in `docs/data_pipeline.md`.
