# Team B – Data Collection and Processing

Output of this folder: a clean, documented weekly dataset for the CF End-Term project
("AI-Assisted Portfolio Selection Using Neural Networks"), used by Teams C, D and E.
Project-level overview and team interfaces: [`../README.md`](../README.md).

## 1. How to run (about 5 minutes)

```bash
pip install pandas numpy matplotlib yfinance
python src/build_dataset.py --selftest   # must print three PASS lines
python src/build_dataset.py --raw-dir Data/data_raw --out-dir Data/data_out   # builds everything into Data/data_out
```

Run the commands from the repository root; the script's default `--raw-dir`/`--out-dir` are relative to the
current directory, so the flags above are what point it at `Data/`.

- Raw daily files are cached in `Data/data_raw/` the first time, so later runs work offline and give identical
  results. The cache from the frozen run is already there.
- The data is **frozen at 9 Oct 2026** (`--asof`), so every teammate gets the same file.
- If Yahoo is blocked or you prefer NSE files: put one daily CSV per stock in `Data/data_raw/`
  (`TCS.csv`, `RELIANCE.csv`, …, `NIFTY50.csv`; columns `Date, Close, Volume`, NSE-style headers also work)
  and run `python src/build_dataset.py --source csv --raw-dir Data/data_raw --out-dir Data/data_out`.
  **NSE bhavcopy prices are not adjusted** for splits/bonus issues. If you use them, adjust first, or the
  returns around those dates will be wrong.
- **First thing after the real run: open `Data/data_out/data_quality_report.txt`** and check every week listed under
  "|return| > 25%". Each one is either a real move or an unadjusted corporate action. Fix or explain it.

## 2. Stocks (Team A confirms the final list)

| Ticker | Sector |
|---|---|
| RELIANCE.NS | Energy / Conglomerate |
| TCS.NS | IT Services |
| HDFCBANK.NS | Banking |
| HINDUNILVR.NS | FMCG |
| SUNPHARMA.NS | Pharma |
| LT.NS | Infrastructure / Capital Goods |
| MARUTI.NS | Automobiles |
| BHARTIARTL.NS | Telecom |

8 stocks, 8 different sectors, all with data well before 2016. With a 30% cap per stock the optimizer needs at
least 4 stocks, and 8 gives it room to work. To change the list, edit `STOCKS` at the top of the script.

## 3. Conventions (copy these into the report's Stage 2 section)

| Item | Choice |
|---|---|
| Data source | Yahoo Finance via `yfinance` (`.NS` tickers, `^NSEI` for Nifty 50), or user CSVs |
| Price | Adjusted close (splits, bonus issues and dividends), so returns are total-return style |
| Weekly price | Close of the **last trading day of the week**, labelled by that week's Friday (Thursday close if Friday is a holiday) |
| Weekly volume | Sum of daily volumes in the week |
| Return | `r_t = (P_t − P_{t−1}) / P_{t−1} × 100`, in **percent** |
| Missing week | Price carried forward at most 1 week, volume set to 0 for that week; anything longer is dropped and reported |
| Warm-up | Data downloaded from Dec 2015 so the first 2017 row has full 8-week and 52-week history |
| Split (by week t) | Train 2017-01 → 2022-12 · Validation 2023-01 → 2024-12 · Test 2025-01 → 2026-09 |

## 4. Variables (all known at the end of week t)

| Column | Meaning |
|---|---|
| `ret_1w` | Return in the latest completed week (required) |
| `ret_4w` | Compounded return over the previous 4 weeks, `P_t / P_{t−4} − 1`, not a sum of 4 returns (required) |
| `vol_8w` | Standard deviation (sample, ddof=1) of the last 8 weekly returns including week t (required) |
| `nifty_ret_1w` | Nifty 50 return in the latest week (optional) |
| `volume_ratio_8w` | Weekly volume ÷ average weekly volume of the last 8 weeks (optional). A ratio avoids raw volume being on a different scale for each stock |
| `target_ret_next` | **Return in week t+1. This is the prediction target and never an input.** |
| `weekly_volume`, `close` | Informational only; not features |

## 5. Files produced in `Data/data_out/` and who uses them

| File | Shape | For | How to use it |
|---|---|---|---|
| `model_dataset.csv` | one row per stock per week; has `split` column | **Team D** | Inputs = the 5 feature columns, target = `target_ret_next`. Never shuffle across the split boundaries |
| `scaler_params_train_only.json` | mean/std per feature from **train rows only** | **Team D** | `z = (x − mean) / std` on train, val and test alike. `apply_scaler()` in the script does this |
| `weekly_returns_pct.csv` | wide: week × stock (+ Nifty), full history | **Team C**, **Team E** | Covariance and historical means at week t: use only rows ≤ t (e.g. the last 52 rows ending at t). Realised return of the portfolio chosen at t = row t+1 |
| `weekly_prices.csv` | wide: week × stock (+ Nifty) | Team C, Team E | Same rules. Divide percent by 100 if the optimizer wants decimals |
| `summary_stats_weekly_return.csv`, `eda_*.png` | preliminary analysis | Report | Correlation and distribution charts use the training period only |
| `data_quality_report.txt` | log | Report/viva | Gaps filled, rows dropped, outlier weeks, rows per split |

Expected rows per stock: 313 train, 104 validation, 91 test (4,064 rows in total).

## 6. What was checked

- The brief's worked example reproduces exactly: +1.00% weekly, +3.01% four-week, +1.00% next-week target.
- Features at week t do not change when all later data is deleted (no look-ahead), and the target at t equals the return of t+1.
- On synthetic daily files (holidays, a missing week, mixed date formats, a fake unadjusted split): all weekly prices match an independent recalculation, a hand-computed row matches to 4 decimals, the splits do not overlap, the scaler equals the train-only statistics, and the fake split was flagged.
- **On the real data (run of 9 Oct 2026):** 4,064 rows kept, 0 rows dropped, 0 weeks carried forward, and exactly
  one week above the 25% flag — MARUTI, week ending 2020-04-10, +32.8%, confirmed against the daily file as a real
  post-COVID rebound. Rebuilding from the cached daily CSVs reproduces every output file byte for byte.
- **Still open:** an external NSE quote to confirm one price (TCS close on 2026-09-25, ₹2,070.99 in our data).

## 7. Likely viva questions for Team B

1. *Why weekly closes on the last trading day?* One consistent date for all stocks, and it matches the weekly rebalancing in the back-test.
2. *Why adjusted prices?* A split would look like a −50% return otherwise. Adjusted prices keep returns meaningful.
3. *Why is the 4-week return compounded?* Returns multiply; summing percentages understates or overstates it.
4. *Why split by time, not randomly?* A random split would let the model train on weeks after the ones it is tested on.
5. *Why fit the scaler on training data only?* Validation/test statistics would leak future information into the inputs.
6. *What happens to a week with no price?* Carried forward at most one week, logged in the quality report; longer gaps drop the row.
7. *Where could leakage still hide?* The target column (never feed it as an input) and rolling windows that extend past week t. Both are tested in `--selftest`.
