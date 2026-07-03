#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drawdown_lab.py -- kill the drawdown without killing the edge.

Tests, on the OOS slice, top-3/day with the 15/15 bracket (your live rule):

  BASE      top-3, equal weight, always full size          (the benchmark)
  REG50     exposure 0.5x when nifty_dist_sma50 < 0
  REG0      exposure 0.0x when nifty_dist_sma50 < 0        (fully flat)
  BREADTH   exposure 0.5x when universe median dist_sma200 < 0
  REGBRD    0.5x if either regime gauge negative, 0x if both
  IVOL      inverse-ATR weights within the day's 3 picks
  DISGATE   learned disaster model: walk down ranks, take first 3 names
            whose P(MAE<=-10% in 5d) is below the train-set 75th pct
  DISGATE+REG50   the two combined
  D1EXIT    post-entry: if day-1 close <= -5% from entry, exit day-2 open

Every variant is scored on the PORTFOLIO objective (what your equity feels):
  ann. Sharpe of daily cohort returns, max additive drawdown, cum return,
  worst month, % time at full size, per-trade disaster rate (MAE<=-10%),
  and return per unit of exposure (so 'just trade less' can't fake a win).

Judgement rule printed at the end. Single embargoed split for the disaster
model (train < test_start - 5d), same discipline as the main pipeline.

Usage:
  python drawdown_lab.py ^
    --scored "...\\bigmove_deploy\\bigmove_scored_test.csv" ^
    --panel  "...\\out_rank_single\\panel_cache.parquet" ^
    --out    "...\\bigmove_deploy" --cost-bps 28
"""
import argparse, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scored", required=True)
ap.add_argument("--panel", required=True)
ap.add_argument("--out", default=".")
ap.add_argument("--top-n", type=int, default=3)
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--cost-bps", type=float, default=28.0)
ap.add_argument("--tp-pct", type=float, default=15.0)
ap.add_argument("--sl-pct", type=float, default=15.0)
ap.add_argument("--dis-thr", type=float, default=-10.0, help="disaster = MAE <= this %% in 5d")
ap.add_argument("--dis-gate-q", type=float, default=0.75, help="skip names above this train quantile of p_disaster")
ap.add_argument("--d1-exit-pct", type=float, default=-5.0)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

H, N, COST = int(args.horizon), int(args.top_n), args.cost_bps / 1e4
TP, SL = args.tp_pct / 100.0, args.sl_pct / 100.0


def mkdate(s):
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


# ================= load & vectorized forward resolution =================
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
print(f"[load] scored {len(sc):,} rows | panel {len(panel):,} rows | features for disaster model: {feats}")

g = panel.groupby("symbol", sort=False)
entry = g["open"].shift(-1).to_numpy("float64")
Hi = np.column_stack([g["high"].shift(-k).to_numpy("float64") for k in range(1, H + 1)])
Lo = np.column_stack([g["low"].shift(-k).to_numpy("float64") for k in range(1, H + 1)])
Op = np.column_stack([g["open"].shift(-k).to_numpy("float64") for k in range(1, H + 1)])
Cl = np.column_stack([g["close"].shift(-k).to_numpy("float64") for k in range(1, H + 1)])

with np.errstate(invalid="ignore", divide="ignore"):
    hiR, loR, opR, clR = (A / entry[:, None] - 1.0 for A in (Hi, Lo, Op, Cl))

n = len(panel)
br = np.full(n, np.nan); d1x = np.full(n, np.nan)
valid = np.isfinite(entry) & (entry > 0) & np.isfinite(clR).all(axis=1)
idx = np.where(valid)[0]
D1 = args.d1_exit_pct / 100.0
for i in idx:                                # bracket + day1-exit paths, gap-aware
    b = clR[i, -1]; d = None
    for k in range(H):
        if opR[i, k] <= -SL: b = opR[i, k]; break
        if loR[i, k] <= -SL: b = -SL; break
        if opR[i, k] >= TP:  b = opR[i, k]; break
        if hiR[i, k] >= TP:  b = TP; break
    br[i] = b
    # D1EXIT: same bracket, but if day-1 close <= D1 and no exit on day1, out at day-2 open
    if clR[i, 0] <= D1:
        k = 0
        if opR[i, 0] <= -SL: d = opR[i, 0]
        elif loR[i, 0] <= -SL: d = -SL
        elif opR[i, 0] >= TP: d = opR[i, 0]
        elif hiR[i, 0] >= TP: d = TP
        else: d = opR[i, 1] if H > 1 and np.isfinite(opR[i, 1]) else clR[i, 0]
    d1x[i] = d if d is not None else b

panel["_br"] = br - COST
panel["_d1x"] = d1x - COST
panel["_mae5"] = loR.min(axis=1) * 100.0
panel["_row"] = np.arange(n)

# join scored OOS rows onto panel rows
m = sc.merge(panel[["symbol", "date", "_row"]], on=["symbol", "date"], how="inner")
m = m[np.isfinite(panel["_br"].to_numpy()[m["_row"].to_numpy()])].copy()
m["br"] = panel["_br"].to_numpy()[m["_row"]]
m["d1x"] = panel["_d1x"].to_numpy()[m["_row"]]
m["mae5"] = panel["_mae5"].to_numpy()[m["_row"]]
for f in feats:
    m[f] = panel[f].to_numpy()[m["_row"]]
m = m.sort_values(["date", "prob"], ascending=[True, False])
m["rank"] = m.groupby("date")["prob"].rank(ascending=False, method="first")
test_start = m["date"].min()
print(f"[resolve] {len(m):,} usable OOS rows | test starts {test_start.date()}")

# ================= per-day regime gauges =================
day = m.groupby("date").agg(nifty50=("M_nifty_dist_sma50", "median") if "M_nifty_dist_sma50" in m else ("prob", "size"),
                            brd=("X_regime_dist_sma200", "median") if "X_regime_dist_sma200" in m else ("prob", "size"))
has_nifty = "M_nifty_dist_sma50" in m.columns
has_brd = "X_regime_dist_sma200" in m.columns

# ================= disaster meta-model (embargoed) =================
p_dis_ok = False
try:
    from lightgbm import LGBMClassifier
    tr = panel[(panel["date"] < test_start - pd.Timedelta(days=H + 2)) & np.isfinite(panel["_mae5"])]
    tr = tr.dropna(subset=feats, how="all")
    ytr = (tr["_mae5"] <= args.dis_thr).astype(int)
    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    dis = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, max_depth=6,
                         min_data_in_leaf=500, reg_alpha=0.3, reg_lambda=10.0,
                         n_jobs=-1, random_state=args.seed, verbosity=-1).fit(Xtr, ytr)
    gate_cut = float(np.quantile(dis.predict_proba(Xtr)[:, 1], args.dis_gate_q))
    m["p_dis"] = dis.predict_proba(m[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0))[:, 1]
    from sklearn.metrics import roc_auc_score
    y_oos = (m["mae5"] <= args.dis_thr).astype(int)
    auc = roc_auc_score(y_oos, m["p_dis"]) if y_oos.nunique() > 1 else np.nan
    print(f"[disaster] base rate train {ytr.mean():.3f} | OOS AUC of disaster model = {auc:.3f} "
          f"| gate cutoff p>={gate_cut:.3f} ({args.dis_gate_q:.0%} train quantile)")
    p_dis_ok = True
except Exception as e:
    print(f"[disaster] model unavailable ({e}) -- DISGATE variants skipped")

# ================= build daily cohort returns per variant =================
def top_n_default(dd):
    return dd.head(N)

def top_n_disgate(dd):
    ok = dd[dd["p_dis"] < gate_cut]
    return ok.head(N) if len(ok) >= N else pd.concat([ok, dd[~dd.index.isin(ok.index)]]).head(N)

def daily_series(pick_fn, retcol, expo_fn=None, weight="eq"):
    out = {}
    expo_track = {}
    for d, dd in m.groupby("date"):
        picks = pick_fn(dd.sort_values("rank"))
        if len(picks) == 0:
            out[d] = 0.0; expo_track[d] = 0.0; continue
        r = picks[retcol].to_numpy()
        if weight == "ivol" and "D_atr_pct" in picks:
            w = 1.0 / np.clip(pd.to_numeric(picks["D_atr_pct"], errors="coerce").to_numpy(), 0.5, None)
            w = np.where(np.isfinite(w), w, 0.0); w = w / w.sum() if w.sum() > 0 else np.full(len(r), 1/len(r))
        else:
            w = np.full(len(r), 1.0 / len(r))
        e = 1.0 if expo_fn is None else expo_fn(d)
        out[d] = float(np.nansum(r * w)) * e
        expo_track[d] = e
    s = pd.Series(out).sort_index()
    return s, pd.Series(expo_track).sort_index()

def ex_reg50(d):  return 0.5 if has_nifty and day.loc[d, "nifty50"] < 0 else 1.0
def ex_reg0(d):   return 0.0 if has_nifty and day.loc[d, "nifty50"] < 0 else 1.0
def ex_brd(d):    return 0.5 if has_brd and day.loc[d, "brd"] < 0 else 1.0
def ex_regbrd(d):
    a = has_nifty and day.loc[d, "nifty50"] < 0
    b = has_brd and day.loc[d, "brd"] < 0
    return 0.0 if (a and b) else (0.5 if (a or b) else 1.0)

variants = [("BASE", top_n_default, "br", None, "eq"),
            ("IVOL", top_n_default, "br", None, "ivol"),
            ("D1EXIT", top_n_default, "d1x", None, "eq")]
if has_nifty:
    variants += [("REG50", top_n_default, "br", ex_reg50, "eq"),
                 ("REG0", top_n_default, "br", ex_reg0, "eq")]
if has_brd:
    variants += [("BREADTH", top_n_default, "br", ex_brd, "eq")]
if has_nifty and has_brd:
    variants += [("REGBRD", top_n_default, "br", ex_regbrd, "eq")]
if p_dis_ok:
    variants += [("DISGATE", top_n_disgate, "br", None, "eq")]
    if has_nifty:
        variants += [("DISGATE+REG50", top_n_disgate, "br", ex_reg50, "eq")]

rows = []
for name, pf, rc, ef, w in variants:
    s, e = daily_series(pf, rc, ef, w)
    eq = s.cumsum(); dd = float((eq - eq.cummax()).min())
    sh = float(s.mean() / s.std(ddof=1) * np.sqrt(252)) if s.std(ddof=1) > 0 else 0.0
    mo = s.groupby(pd.Grouper(freq="ME")).sum()
    active = s[e > 0]
    per_expo = float(s.sum() / e.sum()) if e.sum() > 0 else np.nan
    # disaster rate among taken trades
    taken = []
    for d, ddd in m.groupby("date"):
        taken.append(pf(ddd.sort_values("rank")))
    tk = pd.concat(taken)
    disr = float((tk["mae5"] <= args.dis_thr).mean())
    rows.append(dict(variant=name, sharpe=round(sh, 2), cum_pct=round(eq.iloc[-1] * 100, 1),
                     maxDD_pct=round(dd * 100, 1), worst_mo_pct=round(mo.min() * 100, 2),
                     mean_day_bps=round(s.mean() * 1e4, 1),
                     ret_per_expo_bps=round(per_expo * 1e4, 1),
                     avg_expo=round(float(e.mean()), 2),
                     disaster_rate=round(disr, 3)))

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n================= PORTFOLIO-LEVEL RESULTS (OOS, top-%d, 15/15 bracket) =================" % N)
print(res.to_string(index=False))
res.to_csv(os.path.join(args.out, "drawdown_lab_results.csv"), index=False)

print("""
================= HOW TO JUDGE =================
A variant WINS only if, vs BASE:  maxDD improves by >25%  AND  Sharpe does not
fall  AND  ret_per_expo does not collapse (that catches 'just trade less').
DISGATE is only real if the disaster model's OOS AUC printed above is > ~0.60;
below that it's noise and the gate is random.
D1EXIT: expect it to LOSE (your stop research says so) -- it's here so the
question dies with data, not opinion.
Whatever wins: run it again with --dis-gate-q 0.6 and 0.9 and check the
conclusion is STABLE across settings. If it flips, it's overfit. One winner,
frozen, then forward-validate. Do not stack three overlays because each looked
good alone -- that's how curve-fit portfolios are born.
""")
