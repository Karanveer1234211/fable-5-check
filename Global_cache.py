#!/usr/bin/env python3
"""
MACRO CACHE — NIFTY 50 + INDIA VIX features for the model

Fetches NIFTY 50 and INDIA VIX daily data from Kite, computes derived macro
features, and saves a single parquet that gets joined into the panel during
training.

Run once per day BEFORE running Daily cache.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ───────────────────── Config ─────────────────────

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

WIN_DEFAULT_BASE = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache")
OUTPUT_PATH = WIN_DEFAULT_BASE / "macro_cache.parquet"

DEFAULT_START_DAYS_BACK = 8 * 365

NIFTY_SYMBOL = "NIFTY 50"
VIX_SYMBOL = "INDIA VIX"

NIFTY_TOKEN_OVERRIDE = 256265
VIX_TOKEN_OVERRIDE = 264969


def today_ist() -> dt.date:
    return dt.datetime.now(tz=IST).date()


# ─────────────────── Kite auth ────────────────────

def _token_file_path() -> str:
    env_path = os.environ.get("KITE_TOKEN_FILE")
    if env_path:
        return env_path

    default_win = r"C:\Users\karanvsi\PyCharmMiscProject\kite_token.json"
    if os.name == "nt" and os.path.exists(default_win):
        return default_win

    return os.path.join(os.path.dirname(__file__), "kite_token.json")


def _load_kite():
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise SystemExit("kiteconnect not installed. Run: pip install kiteconnect")

    token_file = _token_file_path()
    if not os.path.exists(token_file):
        raise SystemExit(f"Token file missing: {token_file}")

    with open(token_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    api_key = data.get("api_key")
    access_token = data.get("access_token")

    if not api_key or not access_token:
        raise SystemExit("api_key/access_token missing in token file")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    try:
        kite.timeout = 15.0
    except Exception:
        pass

    try:
        _ = kite.profile()
    except Exception as e:
        raise SystemExit(f"Kite auth failed: {e}")

    return kite


# ───────────── Fetch daily index data ─────────────

def _fetch_index_daily(
    kite,
    instrument_token: int,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:

    MAX_DAYS = 1999
    all_rows = []
    cur = start

    while cur <= end:
        chunk_end = min(end, cur + dt.timedelta(days=MAX_DAYS))
        start_dt = dt.datetime.combine(cur, dt.time(0, 0))
        end_dt = dt.datetime.combine(chunk_end, dt.time(23, 59))

        try:
            rows = kite.historical_data(
                instrument_token,
                from_date=start_dt,
                to_date=end_dt,
                interval="day",
                oi=False,
            )
            if not rows:
                rows = kite.historical_data(
                    instrument_token,
                    from_date=start_dt,
                    to_date=end_dt,
                    interval="day",
                )
            all_rows.extend(rows or [])
        except Exception:
            pass

        cur = chunk_end + dt.timedelta(days=1)

    if not all_rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(all_rows)

    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"date": "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(IST)

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    return (
        df[cols]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


# ───────────── Feature computation ────────────────

def compute_macro_features(
    nifty_df: pd.DataFrame,
    vix_df: pd.DataFrame,
) -> pd.DataFrame:

    if nifty_df.empty:
        raise RuntimeError("NIFTY dataframe empty")
    if vix_df.empty:
        raise RuntimeError("VIX dataframe empty")

    n = nifty_df[["timestamp", "close"]].copy()
    n["date"] = n["timestamp"].dt.normalize()
    n = (
        n[["date", "close"]]
        .rename(columns={"close": "nifty_close"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    n["M_nifty_ret"] = n["nifty_close"].pct_change() * 100
    n["M_nifty_ret_5d"] = (
        n["nifty_close"] / n["nifty_close"].shift(5) - 1
    ) * 100

    sma50 = n["nifty_close"].rolling(50, min_periods=10).mean()
    sma200 = n["nifty_close"].rolling(200, min_periods=50).mean()

    n["M_nifty_dist_sma50"] = (n["nifty_close"] - sma50) / sma50
    n["M_nifty_dist_sma200"] = (n["nifty_close"] - sma200) / sma200

    v = vix_df[["timestamp", "close"]].copy()
    v["date"] = v["timestamp"].dt.normalize()
    v = (
        v[["date", "close"]]
        .rename(columns={"close": "M_vix"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    v["M_vix_change"] = v["M_vix"].diff()
    mean60 = v["M_vix"].rolling(60, min_periods=20).mean()
    std60 = v["M_vix"].rolling(60, min_periods=20).std()
    v["M_vix_level_z60"] = (v["M_vix"] - mean60) / std60

    out = pd.merge(n, v, on="date", how="outer").sort_values("date")
    macro_cols = [c for c in out.columns if c.startswith("M_")]

    for c in macro_cols:
        out[c] = out[c].ffill()

    return out[["date"] + macro_cols].reset_index(drop=True)


# ───────────────────── Main ──────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build NIFTY 50 + INDIA VIX macro cache"
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--days-back",
        type=int,
        default=DEFAULT_START_DAYS_BACK,
    )
    args = parser.parse_args()

    end_date = today_ist()

    if not args.rebuild and OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        last_date = pd.to_datetime(existing["date"]).max().date()
        start_date = last_date - dt.timedelta(days=5)
    else:
        start_date = end_date - dt.timedelta(days=args.days_back)

    kite = _load_kite()

    nifty_df = _fetch_index_daily(kite, NIFTY_TOKEN_OVERRIDE, start_date, end_date)
    vix_df = _fetch_index_daily(kite, VIX_TOKEN_OVERRIDE, start_date, end_date)

    macro_df = compute_macro_features(nifty_df, vix_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    macro_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved macro cache: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()