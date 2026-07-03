#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
big_move_pipeline.py
====================
Deployable big-move (>=X% in N days) pipeline. Mirrors the way New_model operates:
it TRAINS, evaluates OUT-OF-SAMPLE, refits a deployable model on all data, then
SAVES everything you need to act and to backtest.

WHAT IT SAVES (into --out):
  bigmove_model.joblib        the deployable model (pickle-safe BigMoveModel wrapper:
                              carries feature_list + impute, so you just hand it a
                              panel and it scores). Refit on ALL labeled rows.
  bigmove_watchlist.csv       TODAY's actionable names: latest-date rows scored by the
                              deployable model, prob >= the top-decile cutoff (or your
                              --watchlist-min-prob / --watchlist-top-n), ranked.
  bigmove_scored_test.csv     OOS test-block scores ONLY (symbol,timestamp,prob,label,
                              ret_5d,mfe_5d). >>> USE THIS FOR BACKTESTS <<< it is the
                              only clean out-of-sample slice.
  bigmove_scored_panel.parquet  the FULL panel scored by the deployable model, with an
                              `in_test` flag + OHLC + labels + dollar_vol + decile.
                              For live reference / charting. NOTE: pre-test rows are
                              IN-SAMPLE here (the deploy model saw them) -- do not treat
                              the whole file as OOS; filter in_test==True or use the
                              bigmove_scored_test.csv above.
  bigmove_oos_report.json     base rate, ROC-AUC, PR-AUC, Brier, full decile table.
  bigmove_decile_report.csv   the decile table as CSV.
  bigmove_meta.json           every config value, feature list, train/test date bounds,
                              universe size, label definition, watchlist cutoff -- so any
                              backtest is fully reproducible.

DAILY USE (no retrain): once trained, score a fresh panel and refresh the watchlist:
  python big_move_pipeline.py --load-model "<out>\\bigmove_model.joblib" \
      --panel <fresh panel_cache.parquet> --features <features_train.json> --out <out>

TRAIN:
  python big_move_pipeline.py --panel panel_cache.parquet --features features_train.json \
      --out bigmove_deploy --threshold-pct 5 --horizon 5 --label-mode both --train-on touch \
      --test-frac 0.20 --top-adv 500 --scale-pos-weight 0 --refit-full

Run from a TERMINAL (the PyCharm green button passes no args).
"""

import argparse, json, os, sys, warnings, datetime as _dt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def log(m): print(m, flush=True)


# ============================================================ pickle-safe deployable model
class BigMoveModel:
    """Module-level (pickle-safe) wrapper. Carries the trained classifier, the exact
    ordered feature_list and the impute table, so scoring a raw panel is one call."""
    def __init__(self, clf, feature_list, impute, iso=None, config=None):
        self.clf = clf
        self.feature_list = list(feature_list)
        self.impute = dict(impute or {})
        self.iso = iso
        self.config = dict(config or {})

    def _prep(self, X):
        Xp = X.reindex(columns=self.feature_list)
        for f in self.feature_list:
            if f in self.impute:
                Xp[f] = Xp[f].fillna(self.impute[f])
        Xp = Xp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return Xp

    def predict_proba_1(self, X):
        p = self.clf.predict_proba(self._prep(X))[:, 1]
        if self.iso is not None:
            p = np.clip(self.iso.predict(p), 1e-6, 1.0 - 1e-6)
        return np.asarray(p, dtype=float)


# ============================================================ helpers
def detect_date_col(df):
    for c in ("timestamp", "date", "datetime"):
        if c in df.columns:
            return c
    raise SystemExit(f"[FATAL] no date column found in {list(df.columns)[:12]}")


def normalize_days(s):
    s = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize()


def add_forward_targets(df, sym_col, date_col, horizon):
    h = int(horizon)
    out = []
    for _, g in df.sort_values(date_col).groupby(sym_col, sort=False):
        g = g.copy()
        c = g["close"]
        g[f"ret_{h}d_close_pct"] = (c.shift(-h) / c - 1.0) * 100.0
        hi = pd.concat([g["high"].shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1)
        lo = pd.concat([g["low"].shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1)
        g[f"mfe_{h}d_pct"] = (hi / c - 1.0) * 100.0
        g[f"mae_{h}d_pct"] = (lo / c - 1.0) * 100.0
        out.append(g)
    return pd.concat(out, axis=0)


def day_embargo_split(day_series, test_frac, cal_frac, embargo_days):
    days = pd.Series(day_series).reset_index(drop=True)
    uniq = np.sort(days.unique())
    n = len(uniq)
    train_frac = 1.0 - test_frac - cal_frac
    if n < 5 or train_frac <= 0:
        raise SystemExit("[FATAL] not enough distinct days for the requested split.")
    cut_tr = int(train_frac * n)
    cut_cal = int((train_frac + cal_frac) * n)
    emb = max(0, int(embargo_days))

    def _slice(e):
        tr = set(uniq[: max(0, cut_tr - e)])
        ca = set(uniq[cut_tr: max(cut_tr, cut_cal - e)]) if cal_frac > 0 else set()
        te = set(uniq[cut_cal:])
        return tr, ca, te

    tr, ca, te = _slice(emb)
    if (not te or not tr or (cal_frac > 0 and not ca)) and emb > 0:
        tr, ca, te = _slice(0)
    dv = days.values
    return (np.where(np.isin(dv, list(tr)))[0],
            np.where(np.isin(dv, list(ca)))[0] if cal_frac > 0 else np.array([], dtype=int),
            np.where(np.isin(dv, list(te)))[0])


def lgbm_params(seed, lr, max_depth, n_estimators, scale_pos_weight):
    return dict(
        n_estimators=int(n_estimators), learning_rate=lr, num_leaves=31,
        max_depth=int(max_depth) if max_depth and max_depth > 0 else -1,
        feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
        min_data_in_leaf=500, min_gain_to_split=0.02, max_bin=255,
        reg_alpha=0.3, reg_lambda=10.0, extra_trees=True,
        scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=int(seed),
        verbosity=-1, feature_fraction_seed=int(seed), bagging_seed=int(seed),
        data_random_seed=int(seed),
    )


def decile_report(prob, label, ret5, mfe5, base_rate, cost_pct, n_buckets=10):
    df = pd.DataFrame({"prob": prob, "label": label, "ret5": ret5, "mfe5": mfe5})
    try:
        df["bk"] = pd.qcut(df["prob"], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        df["bk"] = pd.qcut(df["prob"].rank(method="first"), n_buckets, labels=False)
    rows = []
    for bk, g in df.groupby("bk"):
        hr = g["label"].mean()
        rows.append(dict(bucket=int(bk), n=int(len(g)),
                         avg_prob=round(float(g["prob"].mean()), 4),
                         hit_rate=round(float(hr), 4),
                         lift=round(float(hr / base_rate), 3) if base_rate > 0 else np.nan,
                         mean_ret5=round(float(g["ret5"].mean()), 3),
                         net_ret5=round(float(g["ret5"].mean()) - cost_pct, 3),
                         mean_mfe5=round(float(g["mfe5"].mean()), 3)))
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def load_features(path):
    j = json.load(open(path))
    feats = j["features"] if isinstance(j, dict) else list(j)
    impute = j.get("impute", {}) if isinstance(j, dict) else {}
    return feats, impute


def liquid_filter(panel, sym_col, top_adv, min_dollar_vol):
    if top_adv <= 0 and min_dollar_vol <= 0:
        return panel
    if "D_dollar_vol" in panel.columns:
        liq = panel.groupby(sym_col)["D_dollar_vol"].median()
    else:
        tmp = (panel["close"] * panel["volume"]).groupby(panel[sym_col]).median()
        liq = tmp
    keep = set(liq.index)
    if min_dollar_vol > 0:
        keep &= set(liq[liq >= min_dollar_vol].index)
    if top_adv > 0:
        keep &= set(liq.nlargest(top_adv).index)
    before = panel[sym_col].nunique()
    panel = panel[panel[sym_col].isin(keep)].copy()
    log(f"[liquid] kept {panel[sym_col].nunique()}/{before} symbols; rows now {len(panel):,}")
    return panel


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="Deployable big-move (>=X% in N days) pipeline.")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-model", default=None, help="score-only: load this BigMoveModel and refresh the watchlist")
    ap.add_argument("--symbol-col", default="symbol")
    ap.add_argument("--date-col", default="auto")
    ap.add_argument("--threshold-pct", type=float, default=5.0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--label-mode", choices=["touch", "close", "both"], default="touch")
    ap.add_argument("--train-on", choices=["touch", "close"], default="touch")
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--cal-frac", type=float, default=0.0)
    ap.add_argument("--embargo-days", type=int, default=-1)
    ap.add_argument("--top-adv", type=int, default=0)
    ap.add_argument("--min-dollar-vol", type=float, default=0.0)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--scale-pos-weight", type=float, default=1.0)
    ap.add_argument("--refit-full", action="store_true", default=True,
                    help="refit the saved/deploy model on ALL labeled rows (default on)")
    ap.add_argument("--no-refit-full", dest="refit_full", action="store_false")
    ap.add_argument("--watchlist-min-prob", type=float, default=-1.0,
                    help="watchlist = latest-date names with prob >= this (default: test top-decile cutoff)")
    ap.add_argument("--watchlist-top-n", type=int, default=0,
                    help="alternatively, take the top-N names by prob on the latest date")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import joblib
    h = int(args.horizon)
    thr = float(args.threshold_pct)
    cost_pct = 0.28  # for the net column in reports only

    log(f"[load] {args.panel}")
    panel = pd.read_parquet(args.panel)
    dcol = detect_date_col(panel) if args.date_col == "auto" else args.date_col
    panel["_day"] = normalize_days(panel[dcol])
    if args.start_date:
        panel = panel[panel["_day"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        panel = panel[panel["_day"] <= pd.Timestamp(args.end_date)]
    panel = liquid_filter(panel, args.symbol_col, args.top_adv, args.min_dollar_vol)

    feat_list, impute = load_features(args.features)
    feats = [f for f in feat_list if f in panel.columns]
    log(f"[features] {len(feats)}/{len(feat_list)} present")

    # ---------------------------------------------------------------- SCORE-ONLY (daily)
    if args.load_model:
        log(f"[score-only] loading {args.load_model}")
        model = joblib.load(args.load_model)
        panel["prob_bigmove"] = model.predict_proba_1(panel[feats])
        last_day = panel["_day"].max()
        wl = write_watchlist(panel, args, last_day, cutoff=args.watchlist_min_prob if args.watchlist_min_prob > 0 else None)
        log(f"[score-only] watchlist for {pd.Timestamp(last_day).date()} -> {len(wl)} names "
            f"-> {os.path.join(args.out, 'bigmove_watchlist.csv')}")
        return

    # ---------------------------------------------------------------- TRAIN
    log(f"[targets] building forward {h}d MFE / close-return per symbol ...")
    panel = add_forward_targets(panel, args.symbol_col, "_day", h)
    ret_col, mfe_col = f"ret_{h}d_close_pct", f"mfe_{h}d_pct"
    panel["label_touch"] = (panel[mfe_col] >= thr).astype("float")
    panel["label_close"] = (panel[ret_col] >= thr).astype("float")
    train_label = "label_" + (args.train_on if args.label_mode == "both" else args.label_mode)

    labeled = panel.dropna(subset=[mfe_col, ret_col]).copy()
    labeled[train_label] = labeled[train_label].astype(int)
    base_rate = float(labeled[train_label].mean())
    log(f"[label] train_on={train_label}  thr=+{thr}%  horizon={h}d  base_rate={base_rate:.4f}  rows={len(labeled):,}")

    def prep_X(frame):
        X = frame[feats].copy()
        for f in feats:
            if f in impute:
                X[f] = X[f].fillna(impute[f])
        return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = prep_X(labeled)
    y = labeled[train_label].values
    embargo = h if args.embargo_days < 0 else args.embargo_days
    i_tr, i_ca, i_te = day_embargo_split(labeled["_day"].values, args.test_frac, args.cal_frac, embargo)
    test_start = pd.Series(labeled["_day"].values[i_te]).min()
    log(f"[split] train={len(i_tr):,} cal={len(i_ca):,} test={len(i_te):,} embargo={embargo}d "
        f"test_start={pd.Timestamp(test_start).date()}")

    from lightgbm import LGBMClassifier
    spw = args.scale_pos_weight
    if spw <= 0:
        spw = (1.0 - base_rate) / max(base_rate, 1e-6)
    params = lgbm_params(args.seed, args.learning_rate, args.max_depth, args.n_estimators, spw)

    log("[train:eval] fitting eval model on train block ...")
    clf_eval = LGBMClassifier(**params).fit(X.iloc[i_tr], y[i_tr])
    iso = None
    if len(i_ca) > 0:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(clf_eval.predict_proba(X.iloc[i_ca])[:, 1], y[i_ca])
    p_te = clf_eval.predict_proba(X.iloc[i_te])[:, 1]
    if iso is not None:
        p_te = np.clip(iso.predict(p_te), 1e-6, 1 - 1e-6)

    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    yte = y[i_te]
    te_base = float(yte.mean())
    rep = decile_report(p_te, yte, labeled[ret_col].values[i_te], labeled[mfe_col].values[i_te],
                        te_base, cost_pct)
    metrics = dict(label=train_label, threshold_pct=thr, horizon=h, embargo_days=embargo,
                   rows_total=int(len(labeled)), rows_test=int(len(i_te)),
                   base_rate_full=round(base_rate, 4), base_rate_test=round(te_base, 4),
                   roc_auc=round(float(roc_auc_score(yte, p_te)), 4) if len(np.unique(yte)) > 1 else None,
                   pr_auc=round(float(average_precision_score(yte, p_te)), 4) if len(np.unique(yte)) > 1 else None,
                   brier=round(float(brier_score_loss(yte, p_te)), 5),
                   top_decile_prob_cutoff=round(float(np.quantile(p_te, 0.9)), 4),
                   decile_table=rep.to_dict("records"))

    # OOS scored test (the clean slice for backtests)
    scored_test = pd.DataFrame({
        args.symbol_col: labeled[args.symbol_col].values[i_te],
        "timestamp": labeled["_day"].values[i_te],
        "prob": p_te, "label": yte,
        ret_col: labeled[ret_col].values[i_te], mfe_col: labeled[mfe_col].values[i_te],
        "label_close": labeled["label_close"].values[i_te],
    })
    scored_test.to_csv(os.path.join(args.out, "bigmove_scored_test.csv"), index=False)
    rep.to_csv(os.path.join(args.out, "bigmove_decile_report.csv"), index=False)
    json.dump(metrics, open(os.path.join(args.out, "bigmove_oos_report.json"), "w"), indent=2, default=str)

    log("\n  OOS: base={:.4f}  ROC={}  PR-AUC={} (base {:.4f})  top-decile-cutoff={:.4f}".format(
        te_base, metrics["roc_auc"], metrics["pr_auc"], te_base, metrics["top_decile_prob_cutoff"]))
    log("  " + rep.to_string(index=False).replace("\n", "\n  "))

    # ---- deployable model: refit on ALL labeled rows (or reuse eval) ----
    if args.refit_full:
        log("[train:deploy] refitting on ALL labeled rows for deployment ...")
        clf_dep = LGBMClassifier(**params).fit(X, y)
    else:
        clf_dep = clf_eval
    model = BigMoveModel(clf_dep, feats, impute, iso=iso,
                         config=dict(threshold_pct=thr, horizon=h, label=train_label,
                                     top_adv=args.top_adv, scale_pos_weight=spw,
                                     trained_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                                     refit_full=bool(args.refit_full)))
    joblib.dump(model, os.path.join(args.out, "bigmove_model.joblib"))
    log(f"[save] bigmove_model.joblib")

    # ---- score FULL panel with deploy model (for reference / charting) ----
    panel["prob_bigmove"] = model.predict_proba_1(panel[feats])
    test_day_set = set(pd.Series(labeled["_day"].values[i_te]).unique())
    panel["in_test"] = panel["_day"].isin(test_day_set)
    try:
        panel["decile"] = pd.qcut(panel["prob_bigmove"], 10, labels=False, duplicates="drop")
    except ValueError:
        panel["decile"] = pd.qcut(panel["prob_bigmove"].rank(method="first"), 10, labels=False)
    keep_cols = [args.symbol_col, "_day", "open", "high", "low", "close", "prob_bigmove",
                 "in_test", "decile", "label_touch", "label_close", ret_col, mfe_col]
    if "D_dollar_vol" in panel.columns:
        keep_cols.append("D_dollar_vol")
    keep_cols = [c for c in keep_cols if c in panel.columns]
    sp = panel[keep_cols].rename(columns={"_day": "timestamp"})
    sp.to_parquet(os.path.join(args.out, "bigmove_scored_panel.parquet"), index=False)
    log(f"[save] bigmove_scored_panel.parquet ({len(sp):,} rows, in_test flag set)")

    # ---- watchlist (latest date) ----
    last_day = panel["_day"].max()
    cutoff = args.watchlist_min_prob if args.watchlist_min_prob > 0 else metrics["top_decile_prob_cutoff"]
    wl = write_watchlist(panel, args, last_day, cutoff=cutoff)
    log(f"[save] bigmove_watchlist.csv  ({len(wl)} names on {pd.Timestamp(last_day).date()}, "
        f"cutoff prob>={cutoff:.4f})")

    # ---- meta (reproducibility) ----
    meta = dict(created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(), panel=os.path.abspath(args.panel),
                features=os.path.abspath(args.features), n_features=len(feats),
                symbols=int(panel[args.symbol_col].nunique()),
                date_min=str(pd.Timestamp(panel["_day"].min()).date()),
                date_max=str(pd.Timestamp(panel["_day"].max()).date()),
                test_start=str(pd.Timestamp(test_start).date()),
                threshold_pct=thr, horizon=h, embargo_days=embargo, label=train_label,
                top_adv=args.top_adv, min_dollar_vol=args.min_dollar_vol,
                scale_pos_weight=spw, lgbm_params=params, refit_full=bool(args.refit_full),
                watchlist_cutoff=float(cutoff),
                files=dict(model="bigmove_model.joblib", watchlist="bigmove_watchlist.csv",
                           backtest_scores="bigmove_scored_test.csv (OOS only)",
                           scored_panel="bigmove_scored_panel.parquet (filter in_test for OOS)",
                           oos_report="bigmove_oos_report.json"))
    json.dump(meta, open(os.path.join(args.out, "bigmove_meta.json"), "w"), indent=2, default=str)
    log(f"[save] bigmove_meta.json")
    log(f"\n[done] all artifacts in {args.out}")
    log("  BACKTEST with bigmove_scored_test.csv (OOS). ACT on bigmove_watchlist.csv. "
        "Re-score daily with --load-model.")


def write_watchlist(panel, args, last_day, cutoff=None):
    sub = panel[panel["_day"] == last_day].copy()
    sub = sub.sort_values("prob_bigmove", ascending=False)
    if args.watchlist_top_n and args.watchlist_top_n > 0:
        wl = sub.head(args.watchlist_top_n)
    elif cutoff is not None:
        wl = sub[sub["prob_bigmove"] >= cutoff]
    else:
        wl = sub.head(50)
    wl = wl.copy()

    out = pd.DataFrame({
        args.symbol_col: wl[args.symbol_col].values,
        "date": pd.Timestamp(last_day).date(),
        "prob_bigmove": np.round(wl["prob_bigmove"].values, 4),
        "rank": np.arange(1, len(wl) + 1),
    })
    if "close" in wl.columns:
        out["close"] = wl["close"].values  # signal-day close = entry REFERENCE (real entry = next open, filled by resolver)

    # ---- pass-through DECISION-LAYER columns (model UNCHANGED; these are just model inputs echoed out) ----
    # output_name -> panel feature column.  Only those present are written.
    passthrough = {
        "atr_pct":          "D_atr_pct",           # PRIMARY volatility-brake input (absolute ATR%)
        "range_pct":        "D_range_pct",         # secondary volatility
        "bb_bw_20":         "D_bb_bw_20",          # secondary volatility
        "dvol_z20":         "D_dvol_z20",          # volume-spike (drawdown cousin)
        "dollar_vol":       "D_dollar_vol",        # liquidity
        "dist_sma200":      "X_regime_dist_sma200",# extension above 200DMA
        "nifty_ret_5d":     "M_nifty_ret_5d",      # MARKET regime (same for all names that day)
        "nifty_dist_sma50": "M_nifty_dist_sma50",  # MARKET regime
    }
    for nice, col in passthrough.items():
        if col in wl.columns:
            out[nice] = np.round(pd.to_numeric(wl[col], errors="coerce").values, 6)

    # ---- volatility-brake FLAG (NON-BINDING starting heuristic: top-quartile of today's list by atr_pct) ----
    # raw atr_pct is logged above, so you can re-tune to an absolute threshold later without re-running history.
    if "atr_pct" in out.columns and out["atr_pct"].notna().sum() >= 4:
        thr_q = float(np.nanquantile(out["atr_pct"].values, 0.75))
        out["vol_skip"] = (out["atr_pct"].values >= thr_q).astype(int)
    else:
        out["vol_skip"] = 0

    # ---- SAVE: latest (convenience, overwritten) + dated append-only (the FROZEN forward record) ----
    out.to_csv(os.path.join(args.out, "bigmove_watchlist.csv"), index=False)

    wdir = os.path.join(args.out, "watchlists")
    os.makedirs(wdir, exist_ok=True)
    dated = os.path.join(wdir, f"watchlist_{pd.Timestamp(last_day).date()}.csv")
    if os.path.exists(dated):
        log(f"[watchlist] {os.path.basename(dated)} already exists -- NOT overwriting "
            f"(protects the frozen forward record; delete it by hand to intentionally replace).")
    else:
        out.to_csv(dated, index=False)
        log(f"[watchlist] froze forward record -> watchlists/{os.path.basename(dated)}")
    return out


if __name__ == "__main__":
    main()