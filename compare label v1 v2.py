#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_label_v1_v2.py -- ONE run, clear verdict: is the graded-MFE label (v2)
better than your live touch label (v1)?

It reuses YOUR pipeline's own functions (same panel load, same features, same
day-embargo split, same LGBM params) so the ONLY thing that differs between the
two models is the training target:

  v1  = LGBMClassifier on label_touch  (>= thr% MFE in H days)   <-- your live model
  v2  = LGBMRegressor  on mfe_Hd_pct   (predict the SIZE of the move)   <-- option B

Both are scored on the SAME out-of-sample test block. Both are then judged the
way you actually get paid: rank names each test day, take the top-K, and apply
the +15/-15 BRACKET. Whichever gives the better top-K bracket return wins.

This does NOT touch your pipeline or your live model. It trains throwaway models
in memory and prints a table. Nothing is deployed.

USAGE (same paths as your training command):
  python compare_label_v1_v2.py ^
    --panel "C:\\Users\\karanvsi\\Desktop\\Kite Connect\\out_rank_single\\panel_cache.parquet" ^
    --features "C:\\Users\\karanvsi\\Desktop\\Kite Connect\\out_rank_single\\features_train.json" ^
    --pipeline "C:\\Users\\karanvsi\\PyCharmMiscProject\\Big_move_pipeline_latest.py" ^
    --threshold-pct 5 --horizon 5 --test-frac 0.20 --cal-frac 0.10 --top-adv 1000
"""
import argparse, importlib.util, os, sys
import numpy as np
import pandas as pd


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location("bmp", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bracket_ret(entry_open, highs, lows, closes, tp, sl, cost):
    """+tp / -sl / horizon-close, walked day by day. SL first if both same day."""
    up = entry_open * (1 + tp); dn = entry_open * (1 - sl)
    for k in range(len(closes)):
        if lows[k] <= dn:
            return -sl - cost
        if highs[k] >= up:
            return tp - cost
    return closes[-1] / entry_open - 1.0 - cost


def topk_bracket_by_day(df_test, score_col, k, tp, sl, cost):
    """Each test day: rank by score, take top-k, mean bracket return.
    Returns (mean_bracket_ret, mean_close_ret, n_picks, tp_rate)."""
    picks = []
    for _, g in df_test.groupby("timestamp"):
        g = g.nlargest(k, score_col)
        picks.append(g)
    if not picks:
        return np.nan, np.nan, 0, np.nan
    P = pd.concat(picks)
    return (P["_bracket"].mean() * 100, P["_closeret"].mean() * 100,
            len(P), (P["_bracket_hit_tp"]).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--pipeline", required=True, help="path to Big_move_pipeline_latest.py")
    ap.add_argument("--symbol-col", default="symbol")
    ap.add_argument("--date-col", default="auto")
    ap.add_argument("--threshold-pct", type=float, default=5.0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--cal-frac", type=float, default=0.10)
    ap.add_argument("--embargo-days", type=int, default=-1)
    ap.add_argument("--top-adv", type=int, default=1000)
    ap.add_argument("--min-dollar-vol", type=float, default=0.0)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--tp", type=float, default=0.15)
    ap.add_argument("--sl", type=float, default=0.15)
    ap.add_argument("--cost", type=float, default=0.0028)
    ap.add_argument("--topk", default="3,5", help="comma list of K to report, e.g. 3,5")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    P = load_pipeline(args.pipeline)
    h = int(args.horizon); thr = float(args.threshold_pct)
    SYM = args.symbol_col
    from lightgbm import LGBMClassifier, LGBMRegressor

    print(f"[load] {args.panel}")
    panel = pd.read_parquet(args.panel)
    dcol = P.detect_date_col(panel) if args.date_col == "auto" else args.date_col
    panel["_day"] = P.normalize_days(panel[dcol])
    panel = P.liquid_filter(panel, SYM, args.top_adv, args.min_dollar_vol)

    feat_list, impute = P.load_features(args.features)
    feats = [f for f in feat_list if f in panel.columns]
    print(f"[features] {len(feats)}/{len(feat_list)} present")

    print(f"[targets] building {h}d MFE / close / mae ...")
    panel = P.add_forward_targets(panel, SYM, "_day", h)
    ret_col, mfe_col, mae_col = f"ret_{h}d_close_pct", f"mfe_{h}d_pct", f"mae_{h}d_pct"
    panel["label_touch"] = (panel[mfe_col] >= thr).astype("float")

    labeled = panel.dropna(subset=[mfe_col, ret_col, mae_col]).copy()

    def prep_X(frame):
        X = frame[feats].copy()
        for f in feats:
            if f in impute:
                X[f] = X[f].fillna(impute[f])
        return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = prep_X(labeled)
    embargo = h if args.embargo_days < 0 else args.embargo_days
    i_tr, i_ca, i_te = P.day_embargo_split(labeled["_day"].values, args.test_frac, args.cal_frac, embargo)
    print(f"[split] train={len(i_tr):,} cal={len(i_ca):,} test={len(i_te):,} embargo={embargo}d "
          f"test_start={pd.Timestamp(pd.Series(labeled['_day'].values[i_te]).min()).date()}")

    # -------- v1: touch classifier (your live setup) --------
    base_rate = float(labeled["label_touch"].values[i_tr].mean())
    spw = (1.0 - base_rate) / max(base_rate, 1e-6)
    params_clf = P.lgbm_params(args.seed, args.learning_rate, args.max_depth, args.n_estimators, spw)
    y_touch = labeled["label_touch"].values
    print("[v1] fitting touch classifier ...")
    clf = LGBMClassifier(**params_clf).fit(X.iloc[i_tr], y_touch[i_tr])
    s_v1 = clf.predict_proba(X.iloc[i_te])[:, 1]

    # -------- v2: graded MFE regressor (option B) --------
    params_reg = P.lgbm_params(args.seed, args.learning_rate, args.max_depth, args.n_estimators, 1.0)
    params_reg.pop("scale_pos_weight", None)          # not a regressor param
    params_reg["objective"] = "regression_l1"          # robust to fat MFE tail
    y_mfe = labeled[mfe_col].values
    print("[v2] fitting graded-MFE regressor ...")
    reg = LGBMRegressor(**params_reg).fit(X.iloc[i_tr], y_mfe[i_tr])
    s_v2 = reg.predict(X.iloc[i_te])

    # -------- assemble test frame with bracket outcome per pick --------
    # need OHLC path per (symbol, day) to walk the bracket. rebuild from panel.
    by = {}
    for s, g in panel.groupby(SYM, sort=False):
        g = g.sort_values("_day")
        by[s] = dict(day=g["_day"].values, o=g["open"].to_numpy(float),
                     hi=g["high"].to_numpy(float), lo=g["low"].to_numpy(float),
                     c=g["close"].to_numpy(float),
                     idx={pd.Timestamp(t): i for i, t in enumerate(g["_day"].values)})
    tp, sl, cost = args.tp, args.sl, args.cost
    syms = labeled[SYM].values[i_te]; days = labeled["_day"].values[i_te]
    br = np.full(len(i_te), np.nan); cr = np.full(len(i_te), np.nan); hittp = np.zeros(len(i_te))
    for j in range(len(i_te)):
        rec = by.get(syms[j]); d = pd.Timestamp(days[j])
        if rec is None or d not in rec["idx"]:
            continue
        i0 = rec["idx"][d]; fwd = range(i0 + 1, i0 + 1 + h)
        if i0 + h >= len(rec["c"]):
            continue
        e = rec["o"][i0 + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        hs = rec["hi"][i0 + 1:i0 + 1 + h]; ls = rec["lo"][i0 + 1:i0 + 1 + h]; cs = rec["c"][i0 + 1:i0 + 1 + h]
        up = e * (1 + tp); dn = e * (1 - sl); res = cs[-1] / e - 1.0 - cost; tphit = 0
        for kk in range(h):
            if ls[kk] <= dn:
                res = -sl - cost; break
            if hs[kk] >= up:
                res = tp - cost; tphit = 1; break
        br[j] = res; cr[j] = cs[-1] / e - 1.0 - cost; hittp[j] = tphit

    T = pd.DataFrame({"timestamp": days, SYM: syms, "v1": s_v1, "v2": s_v2,
                      "_bracket": br, "_closeret": cr, "_bracket_hit_tp": hittp}).dropna(subset=["_bracket"])

    print("\n" + "=" * 78)
    print(" LABEL BAKE-OFF  v1 (touch classifier)  vs  v2 (graded-MFE regressor)")
    print(f" judged on: rank each test day -> top-K -> +{tp:.0%}/-{sl:.0%} bracket, cost {cost*100:.2f}%")
    print("=" * 78)
    print(f"  {'selector':22}{'K':>3}{'n':>7}{'bracket%':>10}{'close%':>9}{'TP-rate%':>10}")
    verdict = {}
    for k in [int(x) for x in args.topk.split(",")]:
        for name, col in [("v1 touch prob", "v1"), ("v2 graded MFE", "v2")]:
            b, c, n, tpr = topk_bracket_by_day(T, col, k, tp, sl, cost)
            print(f"  {name:22}{k:>3}{n:>7}{b:>10.3f}{c:>9.3f}{tpr:>10.1f}")
            verdict[(name, k)] = b
        # head-to-head line
        d3 = verdict[("v2 graded MFE", k)] - verdict[("v1 touch prob", k)]
        tag = "v2 BETTER" if d3 > 0 else ("v1 better" if d3 < 0 else "tie")
        print(f"  {'-> v2 minus v1':22}{k:>3}{'':7}{d3:>+10.3f}   {tag}")
        print()

    print("READ: 'bracket%' is mean per-trade return of the top-K under YOUR exit rule.")
    print("Higher = better. v2 wins only if it beats v1 at BOTH K values by a clear margin.")
    print("This is ONE in-sample OOS window. A win here => train v2 for real and SHADOW it")
    print("against v1 on the live forward record before promoting. A tie/loss => keep v1.")


if __name__ == "__main__":
    main()
