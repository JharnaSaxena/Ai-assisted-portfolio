#!/usr/bin/env python3
"""
Team B - Data pipeline
Project: AI-Assisted Portfolio Selection Using Neural Networks (CF End-Term)

What this script does
---------------------
1. Gets DAILY prices + volume for the chosen NSE stocks and the Nifty 50 index
   (Yahoo Finance via yfinance, or your own CSV files, e.g. from NSE).
2. Converts them to WEEKLY data with ONE convention for every stock:
   the close of the last trading day of the week, labelled by that week's Friday.
   Weekly volume = sum of the daily volumes in that week.
3. Builds the features (information known at the end of week t) and the target
   (the return in week t+1).
4. Splits the observations CHRONOLOGICALLY into train / validation / test.
5. Fits a scaler on TRAIN rows only and saves its parameters.
6. Writes a data-quality report, summary statistics and a few EDA charts.

Run
---
    pip install pandas numpy matplotlib yfinance
    python build_dataset.py --selftest        # checks the maths against the assessment example
    python build_dataset.py                   # downloads data, builds everything into ./data_out

    # No internet / Yahoo blocked? Put daily CSVs in ./data_raw (see README) and run:
    python build_dataset.py --source csv

All returns are in PERCENT (e.g. 1.00 means +1.00%), exactly like the assessment brief.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG  (Team A confirms the final stock list; change it here and re-run)
# =============================================================================
STOCKS = {  # Yahoo/NSE ticker -> sector
    "RELIANCE.NS": "Energy / Conglomerate",
    "TCS.NS": "IT Services",
    "HDFCBANK.NS": "Banking",
    "HINDUNILVR.NS": "FMCG",
    "SUNPHARMA.NS": "Pharma",
    "LT.NS": "Infrastructure / Capital Goods",
    "MARUTI.NS": "Automobiles",
    "BHARTIARTL.NS": "Telecom",
}
INDEX_TICKER = "^NSEI"          # Nifty 50
DOWNLOAD_START = "2015-12-01"   # buffer so that 2017 features and 52-week windows are complete
DEFAULT_ASOF = "2026-10-09"     # freeze date: every team gets identical data
SPLITS = {                      # by week_ending date t (the date the information is known)
    "train": ("2017-01-01", "2022-12-31"),
    "val": ("2023-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2026-09-30"),
}
FEATURES = ["ret_1w", "ret_4w", "vol_8w", "nifty_ret_1w", "volume_ratio_8w"]
TARGET = "target_ret_next"
MAX_FFILL_WEEKS = 1     # a missing weekly price is carried forward at most this many weeks
OUTLIER_PCT = 25.0      # |weekly return| above this is FLAGGED (not changed) for manual checking


def raw_name(ticker: str) -> str:
    return "NIFTY50" if ticker == INDEX_TICKER else ticker.replace(".NS", "")


# =============================================================================
# 1. LOADING DAILY DATA
# =============================================================================
def _pick(cols, options):
    low = {str(c).lower().strip(): c for c in cols}
    for o in options:
        if o in low:
            return low[o]
    raise KeyError(f"none of {options} found in columns {list(cols)}")


def _parse_dates(s: pd.Series) -> pd.DatetimeIndex:
    """ISO dates (2016-03-04) are parsed as year-month-day. Anything else (e.g. NSE's 04-Mar-2016 or
    04/03/2016) is read day-first. Never use dayfirst on ISO strings: it silently swaps day and month."""
    s = s.astype(str).str.strip()
    if s.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return pd.DatetimeIndex(pd.to_datetime(s.str[:10], format="%Y-%m-%d"))
    return pd.DatetimeIndex(pd.to_datetime(s, format="mixed", dayfirst=True)).normalize()


def load_csv(path: Path) -> pd.DataFrame:
    """Daily CSV with a date column, a close column and (optionally) a volume column.
    Accepts Yahoo-style and NSE-style headers. Use ADJUSTED closes (splits/bonus/dividends)."""
    df = pd.read_csv(path)
    dcol = _pick(df.columns, ["date", "timestamp"])
    ccol = _pick(df.columns, ["close", "adj close", "adj_close", "close price", "close_price"])
    try:
        vcol = _pick(df.columns, ["volume", "total traded quantity", "ttl_trd_qnty"])
    except KeyError:
        vcol = None

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

    out = pd.DataFrame({"Close": num(df[ccol])})
    out["Volume"] = num(df[vcol]) if vcol else np.nan
    out.index = _parse_dates(df[dcol])
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["Close"])


def load_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # imported here so --source csv works without it

    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)  # split+dividend adjusted
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the ticker or your connection.")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df[["Close", "Volume"]]


def get_daily(ticker: str, source: str, raw_dir: Path, asof: pd.Timestamp, refresh: bool) -> pd.DataFrame:
    cache = raw_dir / f"{raw_name(ticker)}.csv"
    if source == "csv" or (cache.exists() and not refresh):
        if not cache.exists():
            raise FileNotFoundError(f"Expected daily file {cache} (columns: Date, Close, Volume)")
        return load_csv(cache)
    df = load_yfinance(ticker, DOWNLOAD_START, (asof + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)  # cache the raw download so the run is reproducible offline
    return df


# =============================================================================
# 2. DAILY -> WEEKLY  (one convention for every stock and the index)
# =============================================================================
def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Week = Saturday..Friday, labelled by the Friday. Price = last available close in the week
    (so a Thursday close is used when Friday is a holiday). Volume = sum of daily volumes."""
    close = daily["Close"].resample("W-FRI").last()
    volume = daily["Volume"].resample("W-FRI").sum(min_count=1)
    return pd.DataFrame({"close": close, "volume": volume})


def build_panel(weekly: dict[str, pd.DataFrame], asof: pd.Timestamp, report: list[str]):
    start = min(w.index.min() for w in weekly.values())
    grid = pd.date_range(start, asof, freq="W-FRI")  # continuous weekly calendar: 1 row == 1 week
    prices = pd.DataFrame({t: w["close"].reindex(grid) for t, w in weekly.items()})
    volumes = pd.DataFrame({t: w["volume"].reindex(grid) for t, w in weekly.items()})

    missing_before = prices.isna() & (prices.ffill().notna())  # gaps AFTER a stock started trading
    prices = prices.ffill(limit=MAX_FFILL_WEEKS)
    filled = missing_before & prices.notna()
    volumes = volumes.mask(filled, 0.0)  # carried-forward price == no trading that week
    report.append("Missing-week handling (price carried forward max "
                  f"{MAX_FFILL_WEEKS} week, volume set to 0 for those weeks):")
    for t in prices.columns:
        still = int((missing_before[t] & prices[t].isna()).sum())
        report.append(f"  {t:<15} filled={int(filled[t].sum()):>3}   still_missing_after_fill={still:>3}")
    return prices, volumes


# =============================================================================
# 3. FEATURES AND TARGET  (everything at row t uses only data up to week t)
# =============================================================================
def make_features(prices: pd.Series, volume: pd.Series, nifty_prices: pd.Series) -> pd.DataFrame:
    ret = (prices / prices.shift(1) - 1) * 100                       # r_t = (P_t - P_{t-1}) / P_{t-1} * 100
    f = pd.DataFrame(index=prices.index)
    f["close"] = prices
    f["weekly_volume"] = volume
    f["ret_1w"] = ret                                                # most recent completed week
    f["ret_4w"] = (prices / prices.shift(4) - 1) * 100               # compounded 4-week return (not a sum)
    f["vol_8w"] = ret.rolling(8, min_periods=8).std()                # std of last 8 weekly returns incl. week t (ddof=1)
    f["nifty_ret_1w"] = (nifty_prices / nifty_prices.shift(1) - 1) * 100
    f["volume_ratio_8w"] = (volume / volume.rolling(8, min_periods=8).mean()).replace([np.inf, -np.inf], np.nan)
    f[TARGET] = ret.shift(-1)                                        # r_{t+1}: NOT known at week t, never an input
    return f


# =============================================================================
# 4. SPLIT + 5. TRAIN-ONLY SCALER
# =============================================================================
def assign_split(dates: pd.Series) -> pd.Series:
    conds = [(dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)) for a, b in SPLITS.values()]
    return pd.Series(np.select(conds, list(SPLITS.keys()), default="excluded"), index=dates.index)


def fit_scaler(train_df: pd.DataFrame) -> dict:
    """Mean/std from TRAIN rows only (pooled across stocks). Apply the same numbers to val and test."""
    return {
        "fit_on": "train rows only",
        "train_period": SPLITS["train"],
        "n_train_rows": int(len(train_df)),
        "features": {f: {"mean": float(train_df[f].mean()), "std": float(train_df[f].std(ddof=0))}
                     for f in FEATURES},
        "note": "z = (x - mean) / std. Target is NOT scaled here; Team D decides about the target.",
    }


def apply_scaler(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    for f, p in params["features"].items():
        out[f + "_z"] = (df[f] - p["mean"]) / p["std"]
    return out


# =============================================================================
# 6. EDA
# =============================================================================
def make_eda(long: pd.DataFrame, prices: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tickers = list(STOCKS)
    # 6a. rebased price history with the three periods shaded
    p = prices.loc[SPLITS["train"][0]:SPLITS["test"][1], tickers]
    reb = p / p.apply(lambda s: s.dropna().iloc[0]) * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    reb.plot(ax=ax, lw=1.3)
    ax.set_yscale("log")
    for (name, (a, b)), shade in zip(SPLITS.items(), ["#4C72B0", "#DD8452", "#55A868"]):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=shade, alpha=0.08)
        ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, ax.get_ylim()[1] * 0.93,
                name.upper(), ha="center", fontsize=9, color="#444")
    ax.set_title("Weekly adjusted close, rebased to 100 at the start of the training period (log scale)")
    ax.set_xlabel("")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out / "eda_1_prices_rebased.png", dpi=150)
    plt.close(fig)

    # 6b. correlation of weekly returns - TRAIN period only (no peeking at validation/test)
    wide = long[long.split == "train"].pivot(index="week_ending", columns="ticker", values="ret_1w")[tickers]
    corr = wide.corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tickers)), [t.replace(".NS", "") for t in tickers], rotation=45, ha="right")
    ax.set_yticks(range(len(tickers)), [t.replace(".NS", "") for t in tickers])
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr.values[i, j] > 0.6 else "#222", fontsize=8)
    ax.set_title("Correlation of weekly returns (training period)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "eda_2_return_correlation_train.png", dpi=150)
    plt.close(fig)

    # 6c. distribution of weekly returns per stock (training period)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([wide[t].dropna() for t in tickers], tick_labels=[t.replace(".NS", "") for t in tickers],
               showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("Weekly return (%)")
    ax.set_title("Weekly return distribution by stock (training period)")
    fig.tight_layout()
    fig.savefig(out / "eda_3_return_distribution_train.png", dpi=150)
    plt.close(fig)


# =============================================================================
# SELF-TEST: reproduces the worked example in the assessment + a look-ahead check
# =============================================================================
def selftest() -> None:
    # (a) Assessment illustration, information at 25 Sep 2026
    idx = pd.to_datetime(["2026-08-28", "2026-09-04", "2026-09-11", "2026-09-18", "2026-09-25", "2026-10-02"])
    # 4 Sep and 11 Sep are not given in the brief (it says intervening weeks exist); they do not
    # affect the three numbers checked below.
    p = pd.Series([1237.62, 1245.00, 1255.00, 1262.25, 1274.87, 1287.62], index=idx)
    f = make_features(p, pd.Series(1e6, index=idx), p)
    row = f.loc["2026-09-25"]
    assert round(row["ret_1w"], 2) == 1.00, row["ret_1w"]
    assert round(row["ret_4w"], 2) == 3.01, row["ret_4w"]
    assert round(row[TARGET], 2) == 1.00, row[TARGET]
    print("PASS  assessment example: ret_1w=+1.00%, ret_4w=+3.01%, next-week target=+1.00%")

    # (b) No look-ahead: features at week k must be identical when all later data is removed
    rng = np.random.default_rng(0)
    n = 200
    gidx = pd.date_range("2018-01-05", periods=n, freq="W-FRI")
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), index=gidx)
    vol = pd.Series(rng.integers(1e5, 1e6, n).astype(float), index=gidx)
    nif = pd.Series(1e4 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=gidx)
    full = make_features(price, vol, nif)
    for k in (30, 90, 150):
        cut = make_features(price.iloc[: k + 1], vol.iloc[: k + 1], nif.iloc[: k + 1])
        a = full.drop(columns=[TARGET]).iloc[k].to_numpy(dtype=float)
        b = cut.drop(columns=[TARGET]).iloc[k].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), f"look-ahead detected at row {k}"
    assert np.isclose(full[TARGET].iloc[50], full["ret_1w"].iloc[51])
    print("PASS  no look-ahead: features at week t do not change when future data is deleted")
    print("PASS  target at week t equals the return of week t+1")


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["yfinance", "csv"], default="yfinance")
    ap.add_argument("--raw-dir", default="data_raw", help="folder with (or to cache) the daily CSV files")
    ap.add_argument("--out-dir", default="data_out")
    ap.add_argument("--asof", default=DEFAULT_ASOF, help="freeze date; weeks ending after it are ignored")
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached CSVs exist")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    asof = pd.Timestamp(args.asof)
    raw_dir, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: list[str] = [f"DATA QUALITY REPORT   (source={args.source}, as-of={asof.date()})",
                         "Prices: " + ("Yahoo Finance auto-adjusted (splits + dividends)" if args.source == "yfinance"
                                       else "user-supplied CSV (must be adjusted for splits/bonus issues)"),
                         "Weekly convention: last trading-day close of the week, labelled by Friday; "
                         "volume = weekly sum.", ""]

    weekly = {}
    for t in [*STOCKS, INDEX_TICKER]:
        daily = get_daily(t, args.source, raw_dir, asof, args.refresh)
        weekly[t] = to_weekly(daily[daily.index <= asof])
        report.append(f"{t:<15} daily rows={len(daily):>5}  first={daily.index.min().date()}  last={daily.index.max().date()}")
    report.append("")

    prices, volumes = build_panel(weekly, asof, report)

    # features per stock
    frames = []
    for t in STOCKS:
        f = make_features(prices[t], volumes[t], prices[INDEX_TICKER])
        f.insert(0, "ticker", t)
        f.insert(1, "sector", STOCKS[t])
        frames.append(f.rename_axis("week_ending").reset_index())
    full = pd.concat(frames, ignore_index=True)

    # restrict to the project window and attach the split label
    win = full[(full.week_ending >= SPLITS["train"][0]) & (full.week_ending <= SPLITS["test"][1])].copy()
    win["split"] = assign_split(win["week_ending"])
    n_all = len(win)
    long = win.dropna(subset=FEATURES + [TARGET]).copy()
    report += ["", f"Rows in project window: {n_all};  dropped (missing feature/target): {n_all - len(long)};  kept: {len(long)}"]
    dropped = win[~win.index.isin(long.index)].groupby("ticker").size()
    for t, c in dropped.items():
        report.append(f"  dropped for {t}: {c}")

    # outlier flags (possible unadjusted split / bad tick) - flagged, never silently changed
    wide_ret = (prices / prices.shift(1) - 1) * 100
    flags = wide_ret[list(STOCKS)].stack()
    flags = flags[flags.abs() > OUTLIER_PCT]
    report += ["", f"Weeks with |return| > {OUTLIER_PCT:.0f}% (CHECK THESE MANUALLY - possible unadjusted corporate action):"]
    report += [f"  {d.date()} {t}: {v:+.1f}%" for (d, t), v in flags.items()] or ["  none"]

    # split counts
    report += ["", "Rows per split and stock:"]
    counts = long.groupby(["ticker", "split"]).size().unstack(fill_value=0)[list(SPLITS)]
    report += ["  " + line for line in counts.to_string().splitlines()]

    # scaler (train only)
    scaler = fit_scaler(long[long.split == "train"])
    (out / "scaler_params_train_only.json").write_text(json.dumps(scaler, indent=2))

    # write outputs
    cols = ["week_ending", "ticker", "sector", "split", "close", "weekly_volume", *FEATURES, TARGET]
    long = long[cols].sort_values(["ticker", "week_ending"])
    long.to_csv(out / "model_dataset.csv", index=False, float_format="%.6f")
    prices.round(4).rename_axis("week_ending").to_csv(out / "weekly_prices.csv")
    wide_ret.round(6).rename_axis("week_ending").to_csv(out / "weekly_returns_pct.csv")

    summ = (long.groupby(["ticker", "split"])["ret_1w"].agg(["count", "mean", "std", "min", "max"])
            .round(3).reset_index())
    summ.to_csv(out / "summary_stats_weekly_return.csv", index=False)
    make_eda(long, prices, out)
    (out / "data_quality_report.txt").write_text("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nDone. Files written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
