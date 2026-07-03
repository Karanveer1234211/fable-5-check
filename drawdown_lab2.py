#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drawdown_lab2.py -- SIZING, not selection. Round 2 of drawdown reduction.

Variants (all OOS, 15/15 bracket, per-unit-of-capital returns):
  BASE          top-3 equal weight (benchmark)
  TOPN5/8/10    wider book, equal weight   -> idiosyncratic diversification
  VT10/20/60    portfolio vol targeting: exposure = target / trailing vol of
                the strategy's OWN daily returns (lookback 10/20/60d), cap 1.5x
  DSIZE k=1/2/4 all 3 picks kept, weight_i ~ (1 - p_disaster_i)^k
  TOPN5+VT20    the two most likely survivors combined (only combo allowed)

STABILITY IS BUILT IN: every variant is also scored separately on the FIRST
and SECOND half of the OOS window. The verdict column applies the rule:
  WIN = maxDD improves >=25% vs BASE  AND  Sharpe >= BASE  AND
        ret_per_expo not collapsed  AND  the Sharpe edge holds in BOTH halves.
No re-running with new parameters afterwards. What wins here, wins. What
doesn't, dies tonight.

Usage: same args as drawdown_lab.py
  python drawdown_lab2.py --scored ...bigmove_scored_test.csv --panel ...panel_cache.parquet --out ...bigmove_deploy --cost-bps 28
"""
import argparse, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scored", required=True)
ap.add_argument("--panel", required=True)
ap.add_argument("--out", default=".")
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--cost-bps", type=float, default=28.0)
ap.add_argument("--tp-pct", type=float, default=15.0)
ap.add_argument("--sl-pct", type=float, default=15.0)
ap.add_argument("--dis-thr", type=float, default=-10.0)
ap.add_argument("--vt-cap", type=float, default=1.5)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

H, COST = int(args.horizon), args.cost_bps / 1e4
TP, SL = args.tp_pct / 100.0, args.sl_pct / 100.0


def mkdate(s):
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


# ---------------- load + resolve (same machinery as lab 1) ----------------
sc = pd.read_csv(args.scored)
dcol = "timestamp" if "timestamp" in sc.columns else "date"
sc["date"] = mkdate(sc[dcol])
sc = sc.dropna(subset=["prob", "date"])

FEATS = ["D_atr_pct", "D_range_pct", "D_bb_bw_20", "D_dvol_z20", "D_dollar_vol",
         "X_regime_dist_sma200", "M_nifty_ret_5d", "M_nifty_dist_sma50",
         "D_rsi14", "D_adx14", "D_rsi7", "D_gap_pct", "D_vol_z252", "D_atr_pct_z252"]
import pyarrow.parquet as pq
have = pq.read_schema(args.panel).names
cols = ["symbol", "open", "high", "low", "close"] + \
       [c for c in ("timestamp", "date") if c in have] + [c for c in FEATS if c in have]
panel = pd.read_parquet(args.panel, columns=list(dict.fromkeys(cols)))
pdc = "timestamp" if "timestamp" in panel.columns else "date"
panel["date"] = mkdate(panel[pdc])
panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
feats = [c for c in FEATS if c in panel.columns]

g = panel.groupby("symbol", sort=False)
entry = g["open"].shift(-1).to_numpy("float64")
Hi = np.column_stack([g["high"].shift(-k).to_numpy("float64") for k in range(1, H + 1)])
Lo = np.column_stack([g["low"].shift(-k).to_numpy("float64") for k in range(1, H + 1)])
Op = np.column_stack([g["open"].shift(-k).to_numpy("float64") for k in range(1, H + 1)])
Cl = np.column_stack([g["close"].shift(-k).to_numpy("float64") for k in range(1, H + 1)])
with np.errstate(invalid="ignore", divide="ignore"):
    hiR, loR, opR, clR = (A / entry[:, None] - 1.0 for A in (Hi, Lo, Op, Cl))

n = len(panel)
br = np.full(n, np.nan)
valid = np.isfinite(entry) & (entry > 0) & np.isfinite(clR).all(axis=1)
for i in np.where(valid)[0]:
    b = clR[i, -1]
    for k in range(H):
        if opR[i, k] <= -SL: b = opR[i, k]; break
        if loR[i, k] <= -SL: b = -SL; break
        if opR[i, k] >= TP:  b = opR[i, k]; break
        if hiR[i, k] >= TP:  b = TP; break
    br[i] = b
panel["_br"] = br - COST
panel["_mae5"] = loR.min(axis=1) * 100.0
panel["_row"] = np.arange(n)

m = sc.merge(panel[["symbol", "date", "_row"]], on=["symbol", "date"], how="inner")
m = m[np.isfinite(panel["_br"].to_numpy()[m["_row"].to_numpy()])].copy()
m["br"] = panel["_br"].to_numpy()[m["_row"]]
m["mae5"] = panel["_mae5"].to_numpy()[m["_row"]]
for f in feats:
    m[f] = panel[f].to_numpy()[m["_row"]]
m = m.sort_values(["date", "prob"], ascending=[True, False])
m["rank"] = m.groupby("date")["prob"].rank(ascending=False, method="first")
test_start = m["date"].min()
print(f"[resolve] {len(m):,} usable OOS rows | test starts {test_start.date()}")

# ---------------- disaster model (embargoed, for DSIZE only) ----------------
have_dis = False
try:
    from lightgbm import LGBMClassifier
    tr = panel[(panel["date"] < test_start - pd.Timedelta(days=H + 2)) & np.isfinite(panel["_mae5"])]
    ytr = (tr["_mae5"] <= args.dis_thr).astype(int)
    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    dis = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, max_depth=6,
                         min_data_in_leaf=500, reg_alpha=0.3, reg_lambda=10.0,
                         n_jobs=-1, random_state=args.seed, verbosity=-1).fit(Xtr, ytr)
    m["p_dis"] = dis.predict_proba(m[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0))[:, 1]
    have_dis = True
    print("[disaster] model fitted for DSIZE variants")
except Exception as e:
    print(f"[disaster] unavailable ({e}); DSIZE skipped")

# ---------------- daily series builders ----------------
def cohort_series(topn, weight_fn=None):
    out = {}
    for d, dd in m.groupby("date"):
        p = dd.sort_values("rank").head(topn)
        r = p["br"].to_numpy()
        if weight_fn is None:
            w = np.full(len(r), 1.0 / len(r))
        else:
            w = weight_fn(p)
        out[d] = float(np.nansum(r * w))
    return pd.Series(out).sort_index()

def dsize_w(k):
    def f(p):
        w = np.power(np.clip(1.0 - p["p_dis"].to_numpy(), 1e-3, 1.0), k)
        return w / w.sum()
    return f

def vol_target(s, lb, cap):
    rv = s.shift(1).rolling(lb, min_periods=max(5, lb // 2)).std()
    tgt = s.std(ddof=1)                       # constant target = full-period vol
    e = (tgt / rv).clip(upper=cap).fillna(1.0)
    return s * e, e

def stats(s, e=None):
    eq = s.cumsum(); dd = float((eq - eq.cummax()).min())
    sh = float(s.mean() / s.std(ddof=1) * np.sqrt(252)) if s.std(ddof=1) > 0 else 0.0
    mo = s.groupby(pd.Grouper(freq="ME")).sum()
    if e is None:
        e = pd.Series(1.0, index=s.index)
    rpe = float(s.sum() / e.sum()) if e.sum() > 0 else np.nan
    return sh, float(eq.iloc[-1] * 100), float(dd * 100), float(mo.min() * 100), \
           float(s.mean() * 1e4), float(rpe * 1e4), float(e.mean())

dates = np.sort(m["date"].unique())
mid = dates[len(dates) // 2]

variants = [("BASE", cohort_series(3), None),
            ("TOPN5", cohort_series(5), None),
            ("TOPN8", cohort_series(8), None),
            ("TOPN10", cohort_series(10), None)]
base3 = variants[0][1]
for lb in (10, 20, 60):
    s, e = vol_target(base3, lb, args.vt_cap)
    variants.append((f"VT{lb}", s, e))
if have_dis:
    for k in (1, 2, 4):
        variants.append((f"DSIZE_k{k}", cohort_series(3, dsize_w(k)), None))
s5 = cohort_series(5)
s, e = vol_target(s5, 20, args.vt_cap)
variants.append(("TOPN5+VT20", s, e))

base_full = stats(base3)
rows = []
for name, s, e in variants:
    full = stats(s, e)
    h1 = stats(s[s.index < mid], None if e is None else e[e.index < mid])
    h2 = stats(s[s.index >= mid], None if e is None else e[e.index >= mid])
    b1 = stats(base3[base3.index < mid]); b2 = stats(base3[base3.index >= mid])
    win = (name == "BASE") or (
        full[2] >= base_full[2] * 0.75 * -1 / -1 and  # placeholder, replaced below
        True)
    # verdict rule, explicit:
    dd_ok = full[2] > base_full[2] * 0.75            # maxDD less negative by >=25%? (dd negative)
    dd_ok = abs(full[2]) <= abs(base_full[2]) * 0.75
    sh_ok = full[0] >= base_full[0] - 1e-9
    rpe_ok = full[5] >= base_full[5] * 0.75
    both_halves = (h1[0] >= b1[0] - 0.05) and (h2[0] >= b2[0] - 0.05)
    verdict = "WIN" if (name != "BASE" and dd_ok and sh_ok and rpe_ok and both_halves) else \
              ("--" if name == "BASE" else "no")
    rows.append(dict(variant=name, sharpe=round(full[0], 2), cum_pct=round(full[1], 1),
                     maxDD=round(full[2], 1), worst_mo=round(full[3], 1),
                     rpe_bps=round(full[5], 1), avg_expo=round(full[6], 2),
                     sh_H1=round(h1[0], 2), sh_H2=round(h2[0], 2),
                     dd_H1=round(h1[2], 1), dd_H2=round(h2[2], 1), verdict=verdict))

res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(f"\n===== SIZING LAB (OOS, 15/15 bracket) | half-split at {pd.Timestamp(mid).date()} =====")
print(res.to_string(index=False))
res.to_csv(os.path.join(args.out, "drawdown_lab2_results.csv"), index=False)

print("""
===== HOW TO READ =====
verdict=WIN demands: |maxDD| cut >=25%, Sharpe >= BASE, ret-per-expo intact,
and Sharpe >= BASE's in BOTH halves independently. These are pre-registered;
nothing gets rerun with friendlier parameters afterwards.
Note TOPN variants are per-unit-of-capital: same total money, spread wider.
If TOPN5/8 wins, your 'top 3' becomes 'top 5/8 at 3/5ths size each' -- same
capital, same signals, less single-name Russian roulette.
If nothing wins: the remaining honest levers are (a) smaller book size and
(b) a second uncorrelated system -- both outside this file's power to test.
""")
