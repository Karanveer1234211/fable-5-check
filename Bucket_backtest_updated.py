#!/usr/bin/env python3
"""
=============================================================================
BUCKET-WISE BACKTEST — daily cross-sectional model, ATR-bracket exits
=============================================================================
Takes the model's OOS predictions + the daily panel and answers the question
OOS/calibration cannot: net-of-cost, does the edge survive as real trades, and
WHERE (which prob-bucket x regime) does it concentrate.

Each signal -> enter NEXT day's open -> exit at the first of {TP, SL, timeout}
over a max hold, with TP/SL placed at k*ATR. Sweeps TP in {2,3,4,5} ATR x SL in
{2,3,4,5} ATR. Leak-free: the signal uses only info up to its date; ATR is as-of
the signal date; the entry is the next open (never the signal bar's close).
If a single day's range straddles both TP and SL, the SL is assumed to fill first
(pessimistic, never optimistic).

Reports, per (direction x prob-bucket x regime) and per TP/SL combo:
  n, win-rate, mean/median net (bps), profit factor, payoff, per-trade Sharpe,
  exit mix (%TP / %SL / %timeout), median time-to-TP, median time-to-SL,
  mean hold, median MAE, median MFE, and a trade-sequenced equity curve
  (cumulative net, max drawdown) for the headline long book.

  python bucket_backtest.py --self-test
  python bucket_backtest.py --preds predictions.parquet --panel panel_cache.parquet \
        --prob-col prob_top20_5d --regime-col stock_regime --cost-bps 20 \
        --horizon 5 --n-buckets 10 --tp-atr 2,3,4,5 --sl-atr 2,3,4,5 --out-dir bt_out
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# ATR (Wilder, as fraction of price) — computed per symbol, causal
# =============================================================================
def atr_fraction(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"].to_numpy("float64"), df["low"].to_numpy("float64"), df["close"].to_numpy("float64")
    pc = np.empty_like(c); pc[0] = c[0]; pc[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, atr / c, np.nan)


# =============================================================================
# Forward O/H/L/C windows (next H days, per symbol) — for entry + path
# =============================================================================
def build_forward(panel: pd.DataFrame, H: int):
    g = panel.groupby("symbol", sort=False)
    entry = g["open"].shift(-1).to_numpy("float64")           # enter next day's open
    Harr = np.full((len(panel), H), np.nan)
    Larr = np.full((len(panel), H), np.nan)
    Carr = np.full((len(panel), H), np.nan)
    Darr = np.empty((len(panel), H), dtype="datetime64[ns]")
    Darr[:] = np.datetime64("NaT")
    for hh in range(1, H + 1):
        Harr[:, hh - 1] = g["high"].shift(-hh).to_numpy("float64")
        Larr[:, hh - 1] = g["low"].shift(-hh).to_numpy("float64")
        Carr[:, hh - 1] = g["close"].shift(-hh).to_numpy("float64")
        Darr[:, hh - 1] = g["date"].shift(-hh).to_numpy("datetime64[ns]")
    return entry, Harr, Larr, Carr, Darr


# =============================================================================
# Vectorized ATR-bracket simulation (one direction, one TP/SL pair)
# =============================================================================
def simulate_bracket(entry, atrf, Harr, Larr, Carr, Darr, k_tp, k_sl, direction, cost_bps, H):
    n = entry.shape[0]
    cost = cost_bps / 1e4
    valid_days = (~np.isnan(Carr)).sum(1)                     # forward days available (>=1 to trade)
    tp_d, sl_d = k_tp * atrf, k_sl * atrf
    if direction > 0:
        TP = entry * (1 + tp_d); SL = entry * (1 - sl_d)
        tp_hit = Harr >= TP[:, None]; sl_hit = Larr <= SL[:, None]
        fav = Harr / entry[:, None] - 1.0; adv = Larr / entry[:, None] - 1.0
    else:
        TP = entry * (1 - tp_d); SL = entry * (1 + sl_d)
        tp_hit = Larr <= TP[:, None]; sl_hit = Harr >= SL[:, None]
        fav = 1.0 - Larr / entry[:, None]; adv = 1.0 - Harr / entry[:, None]
    tp_hit = np.nan_to_num(tp_hit, nan=False); sl_hit = np.nan_to_num(sl_hit, nan=False)

    def first(hitmat):
        any_ = hitmat.any(1)
        return np.where(any_, hitmat.argmax(1) + 1, 99)       # 1-indexed day, 99 = never
    f_tp, f_sl = first(tp_hit), first(sl_hit)

    reason = np.where((f_sl <= f_tp) & (f_sl < 99), "SL",     # pessimistic: SL wins ties
             np.where(f_tp < f_sl, "TP", "TIME"))
    exit_day = np.where(reason == "TIME", np.maximum(valid_days, 1), np.minimum(f_tp, f_sl))
    exit_day = np.clip(exit_day, 1, H)

    # exit price / return
    ret = np.full(n, np.nan)
    ret[reason == "TP"] = tp_d[reason == "TP"]                # +k_tp*ATR (in either direction's P&L)
    ret[reason == "SL"] = -sl_d[reason == "SL"]               # -k_sl*ATR
    tmask = reason == "TIME"
    last_idx = np.clip(valid_days - 1, 0, H - 1)
    close_last = Carr[np.arange(n), last_idx]
    with np.errstate(invalid="ignore"):
        ret[tmask] = direction * (close_last[tmask] / entry[tmask] - 1.0)
    net = ret - cost

    # MAE / MFE up to (and including) the exit day
    day_idx = np.arange(1, H + 1)[None, :]
    inwin = day_idx <= exit_day[:, None]
    mfe = np.where(inwin, fav, -np.inf); mfe = np.nanmax(np.where(np.isnan(mfe), -np.inf, mfe), 1)
    mae = np.where(inwin, adv, np.inf); mae = np.nanmin(np.where(np.isnan(mae), np.inf, mae), 1)

    t_tp = np.where(reason == "TP", f_tp, np.nan)
    t_sl = np.where(reason == "SL", f_sl, np.nan)
    exit_date = Darr[np.arange(n), np.clip(exit_day - 1, 0, H - 1)]

    tradeable = valid_days >= 1
    return dict(net=net, reason=reason, exit_day=exit_day.astype(float),
                mae=mae, mfe=mfe, t_tp=t_tp, t_sl=t_sl, exit_date=exit_date,
                tradeable=tradeable)


def _slow_one(entry, atrf, hi, lo, cl, k_tp, k_sl, direction, cost, H):
    """Reference single-trade bracket for the self-test."""
    tp_d, sl_d = k_tp * atrf, k_sl * atrf
    if direction > 0:
        TP, SL = entry * (1 + tp_d), entry * (1 - sl_d)
    else:
        TP, SL = entry * (1 - tp_d), entry * (1 + sl_d)
    for h in range(len(cl)):
        if np.isnan(cl[h]):
            last = h - 1; break
        hit_tp = (hi[h] >= TP) if direction > 0 else (lo[h] <= TP)
        hit_sl = (lo[h] <= SL) if direction > 0 else (hi[h] >= SL)
        if hit_sl:
            return -sl_d - cost, "SL", h + 1
        if hit_tp:
            return tp_d - cost, "TP", h + 1
        last = h
    return direction * (cl[last] / entry - 1.0) - cost, "TIME", last + 1


# =============================================================================
# METRICS for a group of trades
# =============================================================================
def metrics(net, reason, exit_day, mae, mfe, t_tp, t_sl) -> dict:
    n = net.size
    if n == 0:
        return {}
    wins, losses = net[net > 0], net[net < 0]
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf
    payoff = (wins.mean() / -losses.mean()) if losses.size and wins.size else np.nan
    sr = float(net.mean() / net.std(ddof=1)) if net.std(ddof=1) > 0 else 0.0
    return dict(
        n=int(n),
        win_rate=float((net > 0).mean()),
        mean_net_bps=float(net.mean() * 1e4),
        median_net_bps=float(np.median(net) * 1e4),
        profit_factor=float(pf),
        payoff=float(payoff),
        sharpe_per_trade=sr,
        pct_tp=float((reason == "TP").mean()),
        pct_sl=float((reason == "SL").mean()),
        pct_timeout=float((reason == "TIME").mean()),
        med_time_to_tp=float(np.nanmedian(t_tp)) if np.isfinite(np.nanmedian(t_tp)) else np.nan,
        med_time_to_sl=float(np.nanmedian(t_sl)) if np.isfinite(np.nanmedian(t_sl)) else np.nan,
        mean_hold_days=float(exit_day.mean()),
        med_mae_bps=float(np.median(mae) * 1e4),
        med_mfe_bps=float(np.median(mfe) * 1e4),
        avg_mae_bps=float(np.mean(mae) * 1e4),
        avg_mfe_bps=float(np.mean(mfe) * 1e4),
        p10_net_bps=float(np.percentile(net, 10) * 1e4),
        p25_net_bps=float(np.percentile(net, 25) * 1e4),
        p75_net_bps=float(np.percentile(net, 75) * 1e4),
        p90_net_bps=float(np.percentile(net, 90) * 1e4),
    )


def equity_curve(entry_date, net):
    """Trade-sequenced equity curve (grouped by ENTRY date, equal weight)."""
    s = pd.Series(net, index=pd.to_datetime(entry_date)).groupby(level=0).mean().sort_index()
    eq = s.cumsum()
    peak = eq.cummax()
    dd = (eq - peak)
    sr_daily = float(s.mean() / s.std(ddof=1) * np.sqrt(252)) if s.std(ddof=1) > 0 else 0.0
    return dict(cum_net_bps=float(eq.iloc[-1] * 1e4) if len(eq) else 0.0,
                max_drawdown_bps=float(dd.min() * 1e4) if len(dd) else 0.0,
                cohort_sharpe_ann=sr_daily, n_days=int(len(s))), eq


# =============================================================================
# DRIVER
# =============================================================================
def _norm_date(df):
    """Resolve the date column (timestamp or date), strip tz, floor to day."""
    col = "timestamp" if "timestamp" in df.columns else ("date" if "date" in df.columns else None)
    if col is None:
        raise SystemExit("Need a 'timestamp' or 'date' column.")
    s = pd.to_datetime(df[col], errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def run(preds, panel, cfg):
    # join predictions onto panel rows (key on day; panel uses 'timestamp')
    panel = panel.copy(); preds = preds.copy()
    panel["date"] = _norm_date(panel); preds["date"] = _norm_date(preds)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    # ATR fraction
    if cfg.atr_col and cfg.atr_col in panel.columns:
        af = panel[cfg.atr_col].to_numpy("float64")
        if not cfg.atr_is_pct:                                  # convert abs ATR -> fraction
            af = af / panel["close"].to_numpy("float64")
        panel["_atrf"] = af
    else:
        panel["_atrf"] = np.concatenate([atr_fraction(g) for _, g in panel.groupby("symbol", sort=False)])
    regcol = cfg.regime_col if (cfg.regime_col and cfg.regime_col in panel.columns) else None
    if regcol is None:
        panel["_regime"] = "all"; regcol = "_regime"

    entry, Harr, Larr, Carr, Darr = build_forward(panel, cfg.horizon)
    panel = panel.assign(_entry=entry)
    key = panel[["symbol", "date"]].copy()
    key["_row"] = np.arange(len(panel))
    m = preds.merge(key, on=["symbol", "date"], how="inner")
    if m.empty:
        raise SystemExit("No prediction rows matched the panel on (symbol,date).")
    # OOS-tail filter: keep only the last `oos_tail_frac` of dates (= the model's
    # test window; rows there were never in training, so they're genuinely OOS).
    if cfg.oos_tail_frac:
        ud = np.sort(pd.unique(panel["date"]))
        thr = ud[min(len(ud) - 1, int(len(ud) * (1 - cfg.oos_tail_frac)))]
        m = m[m["date"].to_numpy() >= thr]
        print(f"[bt] OOS-tail filter: dates >= {pd.Timestamp(thr).date()} -> {len(m):,} signal rows")
        if m.empty:
            raise SystemExit("No rows after OOS-tail filter.")
    abs_mode = cfg.prob_min is not None
    if abs_mode:
        edges = np.round(np.arange(cfg.prob_min, cfg.prob_max + 1e-9, cfg.prob_step), 4)
        m = m[m[cfg.prob_col] >= cfg.prob_min].copy()
        if m.empty:
            raise SystemExit(f"No rows with {cfg.prob_col} >= {cfg.prob_min}.")
        bid = np.clip(np.searchsorted(edges, m[cfg.prob_col].to_numpy(), side="right") - 1, 0, len(edges) - 2)
        m["_bucket"] = bid + 1
        bucket_label = {i + 1: f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)}
        print(f"[bt] absolute prob buckets (step {cfg.prob_step}) from {cfg.prob_min}: "
              f"{len(m):,} rows >= {cfg.prob_min}")
    else:
        m["_bucket"] = (m.groupby("date")[cfg.prob_col]
                        .transform(lambda s: pd.qcut(s.rank(method="first"), min(cfg.n_buckets, max(2, s.nunique())),
                                                     labels=False, duplicates="drop") + 1))
        bucket_label = {}
    rows = m["_row"].to_numpy()
    m["_regime"] = panel[regcol].to_numpy()[rows]
    sub_entry = entry[rows]; sub_atrf = panel["_atrf"].to_numpy()[rows]
    sH, sL, sC, sD = Harr[rows], Larr[rows], Carr[rows], Darr[rows]

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    present = sorted(int(b) for b in pd.unique(m["_bucket"].dropna()))
    n_buckets = present[-1] if present else 0
    top_b, bot_b = n_buckets, (present[0] if present else 1)
    if abs_mode:
        dir_specs = ((+1, "long", present),)                # long-only; lows filtered out
    else:
        dir_specs = ((+1, "long", range(1, n_buckets + 1)), (-1, "short", [bot_b]))

    bucket_rows, grid_rows = [], []
    headline_eq = None
    for k_tp, k_sl in itertools.product(cfg.tp_atr, cfg.sl_atr):
        for direction, dlabel, target_buckets in dir_specs:
            sim = simulate_bracket(sub_entry, sub_atrf, sH, sL, sC, sD, k_tp, k_sl, direction, cfg.cost_bps, cfg.horizon)
            ok = sim["tradeable"] & np.isfinite(sim["net"])
            for b in target_buckets:
                for rg in sorted(m["_regime"].dropna().unique().tolist()) + ["ALL"]:
                    sel = ok & (m["_bucket"].to_numpy() == b)
                    if rg != "ALL":
                        sel = sel & (m["_regime"].to_numpy() == rg)
                    idx = np.where(sel)[0]
                    if idx.size < cfg.min_trades:
                        continue
                    mt = metrics(sim["net"][idx], sim["reason"][idx], sim["exit_day"][idx],
                                 sim["mae"][idx], sim["mfe"][idx], sim["t_tp"][idx], sim["t_sl"][idx])
                    eqd, _ = equity_curve(sD[idx, 0], sim["net"][idx])
                    rowd = dict(tp_atr=k_tp, sl_atr=k_sl, direction=dlabel, bucket=int(b),
                                bucket_label=bucket_label.get(int(b), str(int(b))), regime=rg, **mt,
                                max_drawdown_bps=eqd["max_drawdown_bps"], cohort_sharpe_ann=eqd["cohort_sharpe_ann"])
                    bucket_rows.append(rowd)
                    # headline = top-bucket long, ALL regimes, default first TP/SL
                    if (direction > 0 and b == top_b and rg == "ALL"):
                        grid_rows.append(rowd)
                        if k_tp == cfg.tp_atr[0] and k_sl == cfg.sl_atr[0]:
                            eqm, eq = equity_curve(sD[idx, 0], sim["net"][idx])
                            headline_eq = (eqm, eq, k_tp, k_sl)

    bucket_df = pd.DataFrame(bucket_rows)
    grid_df = pd.DataFrame(grid_rows)
    bucket_df.to_csv(out_dir / "bucket_regime_summary.csv", index=False)
    grid_df.to_csv(out_dir / "tpsl_grid_top_long.csv", index=False)
    return bucket_df, grid_df, headline_eq, n_buckets


# =============================================================================
# SELF-TEST
# =============================================================================
def _synth_panel(seed=0, n_sym=40, n_days=400):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sym):
        price = 100.0
        skill = rng.normal(0, 1)                 # symbol's latent quality, drives prob & drift
        for d in range(n_days):
            mu = 0.0005 * np.tanh(skill)         # high-skill symbols drift up
            r = rng.normal(mu, 0.02)
            o = price
            cl = price * (1 + r)
            hi = max(o, cl) * (1 + abs(rng.normal(0, 0.006)))
            lo = min(o, cl) * (1 - abs(rng.normal(0, 0.006)))
            rows.append([f"S{s}", pd.Timestamp("2021-01-01") + pd.Timedelta(days=d), o, hi, lo, cl,
                         "bull_trend" if skill > 0 else "bear_trend", skill, r])
            price = cl
    df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close",
                                     "stock_regime", "_skill", "_r"])
    return df


def run_self_test() -> int:
    print("=" * 70); print("BUCKET BACKTEST — SELF-TEST"); print("=" * 70)

    # [1] vectorized bracket == slow reference (random paths, both directions)
    rng = np.random.default_rng(3)
    n, H = 500, 5
    entry = 100 + rng.normal(0, 5, n)
    atrf = np.abs(rng.normal(0.02, 0.005, n))
    Harr = entry[:, None] * (1 + np.abs(rng.normal(0, 0.03, (n, H))))
    Larr = entry[:, None] * (1 - np.abs(rng.normal(0, 0.03, (n, H))))
    Carr = entry[:, None] * (1 + rng.normal(0, 0.02, (n, H)))
    Darr = np.tile(np.arange(H).astype("datetime64[D]").astype("datetime64[ns]"), (n, 1))
    for direction in (+1, -1):
        sim = simulate_bracket(entry, atrf, Harr, Larr, Carr, Darr, 3, 2, direction, 10.0, H)
        for i in range(0, n, 7):
            rn, rr, rd = _slow_one(entry[i], atrf[i], Harr[i], Larr[i], Carr[i], 3, 2, direction, 10.0 / 1e4, H)
            assert abs(sim["net"][i] - rn) < 1e-9 and sim["reason"][i] == rr and sim["exit_day"][i] == rd, \
                f"mismatch dir={direction} i={i}: {sim['net'][i]:.6f}/{sim['reason'][i]}/{sim['exit_day'][i]} vs {rn:.6f}/{rr}/{rd}"
    print("[1] vectorized ATR-bracket == slow reference (long & short): PASS")

    # [2] pessimistic same-day straddle -> SL
    e = np.array([100.0]); a = np.array([0.01])
    Hs = np.array([[103.0, 100, 100, 100, 100]]); Ls = np.array([[97.5, 100, 100, 100, 100]])  # +3% and -2.5% same day
    Cs = np.array([[100.0, 100, 100, 100, 100]]); Ds = np.tile(np.arange(5).astype("datetime64[D]").astype("datetime64[ns]"), (1, 1))
    sim = simulate_bracket(e, a, Hs, Ls, Cs, Ds, 2, 2, +1, 0.0, 5)   # TP=+2%, SL=-2%, both hit day1
    assert sim["reason"][0] == "SL" and abs(sim["net"][0] + 0.02) < 1e-9, f"pessimism: {sim['reason'][0]},{sim['net'][0]}"
    print("[2] same-day TP+SL straddle resolves to SL (pessimistic): PASS")

    # [3] MAE/MFE + time-to-tp correct on a clean TP path
    Hs = np.array([[100.5, 101, 104, 100, 100]]); Ls = np.array([[99.5, 99, 99.2, 100, 100]])
    Cs = np.array([[100.2, 100.5, 103, 100, 100]])
    sim = simulate_bracket(e, a, Hs, Ls, Cs, Ds, 3, 5, +1, 0.0, 5)   # TP=+3% hit day3 (104>=103)
    assert sim["reason"][0] == "TP" and sim["exit_day"][0] == 3 and sim["t_tp"][0] == 3
    assert abs(sim["mfe"][0] - 0.04) < 1e-9 and abs(sim["mae"][0] + 0.01) < 1e-9   # MFE=+4% (104), MAE=-1% (99)
    print("[3] MAE/MFE + time-to-TP correct: PASS")

    # [4] full pipeline: high-prob bucket should net more than low-prob bucket
    panel = _synth_panel()
    # prob = noisy view of latent skill -> top bucket = genuinely better symbols
    preds = panel[["symbol", "date", "_skill"]].copy()
    preds["prob_top20_5d"] = 1 / (1 + np.exp(-(preds["_skill"] + np.random.default_rng(9).normal(0, 0.5, len(preds)))))
    cfg = Cfg(prob_col="prob_top20_5d", regime_col="stock_regime", cost_bps=5.0, horizon=5,
              n_buckets=5, tp_atr=[3], sl_atr=[3], min_trades=50, out_dir="/tmp/bt_selftest")
    bdf, gdf, eq, nb = run(preds, panel, cfg)
    longs = bdf[(bdf.direction == "long") & (bdf.regime == "ALL")].sort_values("bucket")
    top = longs[longs.bucket == nb]["mean_net_bps"].iloc[0]
    bot = longs[longs.bucket == 1]["mean_net_bps"].iloc[0]
    assert top > bot, f"top bucket ({top:.1f}) not > bottom ({bot:.1f})"
    print(f"[4] prob-bucket gradient holds: top {top:.1f}bps > bottom {bot:.1f}bps "
          f"(win-rate top {longs[longs.bucket==nb]['win_rate'].iloc[0]:.2f}): PASS")

    print("\nALL SELF-TESTS PASSED"); return 0


class Cfg:
    def __init__(self, **kw):
        self.preds = kw.get("preds"); self.panel = kw.get("panel")
        self.prob_col = kw.get("prob_col", "prob_top20_5d")
        self.regime_col = kw.get("regime_col", "stock_regime")
        self.atr_col = kw.get("atr_col"); self.atr_is_pct = kw.get("atr_is_pct", True)
        self.cost_bps = kw.get("cost_bps", 20.0); self.horizon = kw.get("horizon", 5)
        self.n_buckets = kw.get("n_buckets", 10)
        self.tp_atr = kw.get("tp_atr", [2, 3, 4, 5]); self.sl_atr = kw.get("sl_atr", [2, 3, 4, 5])
        self.min_trades = kw.get("min_trades", 100); self.out_dir = kw.get("out_dir", "bt_out")
        self.prob_min = kw.get("prob_min"); self.prob_step = kw.get("prob_step", 0.05)
        self.prob_max = kw.get("prob_max", 1.0); self.oos_tail_frac = kw.get("oos_tail_frac")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preds", type=str); ap.add_argument("--panel", type=str)
    ap.add_argument("--prob-col", default="prob_top20_5d")
    ap.add_argument("--regime-col", default="stock_regime")
    ap.add_argument("--atr-col", default=None)
    ap.add_argument("--atr-abs", action="store_true", help="ATR column is in price units, not a fraction")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n-buckets", type=int, default=10)
    ap.add_argument("--tp-atr", default="2,3,4,5")
    ap.add_argument("--sl-atr", default="2,3,4,5")
    ap.add_argument("--min-trades", type=int, default=100)
    ap.add_argument("--prob-min", type=float, default=None, help="absolute prob floor; enables fixed-width prob bins (e.g. 0.45)")
    ap.add_argument("--prob-step", type=float, default=0.05, help="width of absolute prob bins")
    ap.add_argument("--prob-max", type=float, default=1.0)
    ap.add_argument("--oos-tail-frac", type=float, default=None, help="keep only the last frac of dates (e.g. 0.10 = model test window)")
    ap.add_argument("--out-dir", default="bt_out")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(run_self_test())
    cfg = Cfg(prob_col=a.prob_col, regime_col=a.regime_col, atr_col=a.atr_col, atr_is_pct=not a.atr_abs,
              cost_bps=a.cost_bps, horizon=a.horizon, n_buckets=a.n_buckets,
              tp_atr=[float(x) for x in a.tp_atr.split(",")], sl_atr=[float(x) for x in a.sl_atr.split(",")],
              min_trades=a.min_trades, out_dir=a.out_dir,
              prob_min=a.prob_min, prob_step=a.prob_step, prob_max=a.prob_max, oos_tail_frac=a.oos_tail_frac)
    preds = pd.read_csv(a.preds) if str(a.preds).lower().endswith((".csv", ".tsv")) else pd.read_parquet(a.preds)
    pcols = ["symbol", "date", "timestamp", "open", "high", "low", "close"]
    for extra in ([a.regime_col] if a.regime_col else []) + ([a.atr_col] if a.atr_col else []):
        pcols.append(extra)
    try:
        import pyarrow.parquet as pq
        have = pq.read_schema(a.panel).names
        panel = pd.read_parquet(a.panel, columns=[c for c in dict.fromkeys(pcols) if c in have])
    except Exception:
        panel = pd.read_parquet(a.panel)
    bdf, gdf, eq, nb = run(preds, panel, cfg)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("\n=== prob-bucket gradient (LONG, ALL regimes, first TP/SL) ===")
    show = bdf[(bdf.direction == "long") & (bdf.regime == "ALL") &
               (bdf.tp_atr == cfg.tp_atr[0]) & (bdf.sl_atr == cfg.sl_atr[0])].sort_values("bucket")
    cols0 = (["bucket_label"] if "bucket_label" in show.columns else ["bucket"]) + \
            ["n", "win_rate", "mean_net_bps", "median_net_bps", "profit_factor",
             "pct_tp", "pct_sl", "pct_timeout", "med_time_to_tp", "med_time_to_sl", "med_mae_bps", "med_mfe_bps"]
    print(show[cols0].to_string(index=False))
    print(f"\n=== TP/SL grid (top bucket {nb} LONG, ALL regimes) ===")
    print(gdf[["tp_atr", "sl_atr", "n", "win_rate", "mean_net_bps", "profit_factor",
               "sharpe_per_trade", "pct_tp", "pct_sl", "mean_hold_days"]].to_string(index=False))
    print(f"\n=== top bucket x regime (LONG, first TP/SL) ===")
    tr = bdf[(bdf.direction == "long") & (bdf.bucket == nb) &
             (bdf.tp_atr == cfg.tp_atr[0]) & (bdf.sl_atr == cfg.sl_atr[0])]
    print(tr[["regime", "n", "win_rate", "mean_net_bps", "sharpe_per_trade", "pct_tp", "pct_sl"]].to_string(index=False))
    if eq:
        eqm, _, ktp, ksl = eq
        print(f"\n=== headline equity (top bucket LONG, TP{ktp}/SL{ksl} ATR) ===")
        print(f"cum_net {eqm['cum_net_bps']:.0f}bps | maxDD {eqm['max_drawdown_bps']:.0f}bps | "
              f"cohort Sharpe(ann) {eqm['cohort_sharpe_ann']:.2f} | {eqm['n_days']} entry-days")
    print(f"\nFull tables -> {cfg.out_dir}/bucket_regime_summary.csv , tpsl_grid_top_long.csv")


if __name__ == "__main__":
    main()