#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entry_filter_lab.py  --  does excluding certain ENTRY conditions from the daily
top pick-list reduce drawdown WITHOUT killing the edge?

DISCIPLINE (read this, it's the whole point):
  * Conditions are PRE-REGISTERED below in REGISTRY, decided from mechanism, NOT
    reverse-engineered from which trades lost. Outcome -> antecedent mining is
    banned here by construction: you edit REGISTRY before you look at results.
  * Every condition is POINT-IN-TIME: it uses only data <= pred_date (plus the
    entry-morning open for the gap test, which you observe before you fill). No
    lookahead -- unlike a stop, an entry filter cannot peek at the future.
  * Two verdicts per condition, and BOTH must clear:
      (1) STATISTICAL: does the excluded group separate on forward return?
          -> permutation p-value, respecting daily structure (within-day shuffle
             for name-level conditions, day-level shuffle for regime conditions),
          -> corrected for MULTIPLE TESTING (Bonferroni over the whole registry).
      (2) ECONOMIC: re-run the capital-constrained portfolio with the filter ON.
          Excluding a name frees its slot for the next-best pick, so the test is
          whether RETURN-PER-DRAWDOWN improves vs unfiltered -- NOT whether
          drawdown alone falls (a filter that cuts return and DD equally is just
          a stop in disguise and earns nothing).
  * A filter only "PASSES" if corrected-p < alpha AND ret/DD improves. Expect
    most to fail: the GBM already ingests dist_sma200 / bb_bw_20 / dvol_z20 /
    nifty cols, so it may already price these softly. A HARD filter must beat the
    model's SOFT weighting to be worth a hard exclusion.

Usage (Windows cmd, one line):
  python entry_filter_lab.py --panel "C:\\...\\bigmove_scored_panel.parquet"
  [--top-n 10] [--top-frac 0.10] [--horizon 5] [--slots 5]
  [--stop none|0.15] [--cost 0.28] [--perm 1000] [--alpha 0.05]
  [--start-capital 100000] [--out entry_filter]

NOTE: in-sample over one OOS window. Anything that passes here is a HYPOTHESIS
for the forward log, not a verdict.
"""
import argparse, os, sys
import numpy as np
import pandas as pd


# ============================ PRE-REGISTERED REGISTRY =========================
# Each condition EXCLUDES picks where expr(R) is True. Edit this list BEFORE you
# run. scope: "name" = per-stock (within-day permutation); "day" = market-wide
# (day-level permutation). Keep mechanisms honest -- every entry should have a
# one-line reason you could have written down in advance.
def build_registry():
    return [
        # --- SWING STRUCTURE (your idea: connect higher-lows / lower-highs) ---
        # EXCLUDE names NOT in the desired structure -> tests whether trading only
        # "solid structure" names beats trading the full top-N.
        dict(key="not_higher_lows",  scope="name", desc="exclude names NOT making higher lows",
             expr=lambda R: R["higher_lows"] == 0),
        dict(key="is_lower_highs",   scope="name", desc="exclude names making lower highs (downtrend structure)",
             expr=lambda R: R["lower_highs"] == 1),
        dict(key="not_uptrend_struct", scope="name", desc="exclude names not in clean uptrend structure (HL & not LH)",
             expr=lambda R: R["uptrend_struct"] == 0),
        # --- trend of the name (your idea: don't buy a name below its own trend) ---
        dict(key="below_ema20",      scope="name", desc="entry-day close < ema20",
             expr=lambda R: R["pred_close"] < R["ema20"]),
        dict(key="below_sma50",      scope="name", desc="close < sma50",
             expr=lambda R: R["pred_close"] < R["sma50"]),
        dict(key="below_sma200",     scope="name", desc="close < sma200",
             expr=lambda R: R["pred_close"] < R["sma200"]),
        # --- over-extension (move may already be spent: the 'opening spike' case) ---
        dict(key="ext_ema20_gt8",    scope="name", desc="close >8% above ema20",
             expr=lambda R: (R["pred_close"]/R["ema20"] - 1) > 0.08),
        dict(key="ext_ema20_gt12",   scope="name", desc="close >12% above ema20",
             expr=lambda R: (R["pred_close"]/R["ema20"] - 1) > 0.12),
        dict(key="runup20_gt15",     scope="name", desc="20-day run-up >15%",
             expr=lambda R: R["runup20"] > 0.15),
        dict(key="runup20_gt25",     scope="name", desc="20-day run-up >25%",
             expr=lambda R: R["runup20"] > 0.25),
        # --- gap at entry (you'd be buying the spike at the open) ---
        dict(key="gap_up_entry_gt3", scope="name", desc="entry open >3% above prev close",
             expr=lambda R: R["gap_entry"] > 0.03),
        dict(key="gap_up_entry_gt5", scope="name", desc="entry open >5% above prev close",
             expr=lambda R: R["gap_entry"] > 0.05),
        # --- volatility state (your existing vol_skip lens) ---
        dict(key="high_atr_topq",    scope="name", desc="top-ATR%% quartile among day's picks",
             expr=lambda R: R["atr_topq"] == 1),
        # --- market regime (edge supposedly concentrates in bear-trend -> test both) ---
        dict(key="regime_nifty_below50", scope="day", desc="exclude days NIFTY < its sma50",
             expr=lambda R: R["nifty_below_sma50"] == 1),
        dict(key="regime_nifty_above50", scope="day", desc="exclude days NIFTY > its sma50",
             expr=lambda R: R["nifty_below_sma50"] == 0),
    ]


# ============================ load + derive features ==========================
def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def load_panel(path):
    # read only what we need; the panel is large so subset columns at the source
    want = ["timestamp", "date", "dt", "symbol", "ticker", "tradingsymbol",
            "open", "high", "low", "close", "adj_close",
            "prob_bigmove", "prob", "probability", "score",
            "ema20", "ema_20", "atr_pct", "atrp",
            "nifty_dist_sma50", "in_test", "is_test", "oos", "test"]
    # read the schema cheaply (no row data) to see which of `want` actually exist
    import pyarrow.parquet as pq
    avail = pq.ParquetFile(path).schema.names
    cols = [c for c in avail if c.lower() in {w.lower() for w in want}]
    df = pd.read_parquet(path, columns=cols)

    ren = dict(timestamp=_pick(cols, "timestamp", "date", "dt"),
               symbol=_pick(cols, "symbol", "ticker", "tradingsymbol"),
               open=_pick(cols, "open"), high=_pick(cols, "high"),
               low=_pick(cols, "low"), close=_pick(cols, "close", "adj_close"),
               prob=_pick(cols, "prob_bigmove", "prob", "probability", "score"))
    miss = [k for k, v in ren.items() if v is None]
    if miss:
        sys.exit(f"[error] missing required columns {miss}. panel has: {avail}")
    df = df.rename(columns={v: k for k, v in ren.items()})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    try:
        if getattr(df["timestamp"].dt, "tz", None) is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df["timestamp"] = df["timestamp"].dt.normalize()
    for c in ["open", "high", "low", "close", "prob"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df = df.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close", "prob"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # in_test flag (already read in if present)
    src = _pick(cols, "in_test", "is_test", "oos", "test")
    if src and src in df.columns:
        df["in_test"] = df[src].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    else:
        df["in_test"] = False

    g = df.groupby("symbol", sort=False)
    # ema20 (use if present, else compute)
    ema = _pick(cols, "ema20", "ema_20")
    if ema:
        df["ema20"] = pd.to_numeric(df[ema], errors="coerce").astype("float32")
    else:
        df["ema20"] = g["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean()).astype("float32")
    # sma50 / sma200 / 20d run-up (all <= pred_date)
    df["sma50"]  = g["close"].transform(lambda s: s.rolling(50,  min_periods=30).mean()).astype("float32")
    df["sma200"] = g["close"].transform(lambda s: s.rolling(200, min_periods=120).mean()).astype("float32")
    df["runup20"] = (df["close"] / g["close"].shift(20) - 1).astype("float32")

    # ---- SWING STRUCTURE (mechanical, point-in-time) ---------------------------
    # A swing low at bar t = low[t] is the min of the +/-k window around it (fractal
    # pivot). Same for swing high. "Structure" = the sequence of the last confirmed
    # pivots as of the PREDICTION day (uses only bars <= pred_date; a pivot needs k
    # bars AFTER it to confirm, so the newest usable pivot is >=k bars old -> no
    # lookahead). This is the codeable version of "connecting higher lows / lower
    # highs" -- no hand-drawn lines, reproducible, gradeable.
    K = 3  # fractal half-window; a pivot is a local extreme over +/-K bars
    def _swing_series(gg):
        lo = gg["low"].to_numpy(float); hi = gg["high"].to_numpy(float); n = len(gg)
        is_lo = np.zeros(n, bool); is_hi = np.zeros(n, bool)
        for t in range(K, n - K):
            if lo[t] == lo[t-K:t+K+1].min(): is_lo[t] = True
            if hi[t] == hi[t-K:t+K+1].max(): is_hi[t] = True
        last2_lo = [np.nan, np.nan]; last2_hi = [np.nan, np.nan]
        hl = np.zeros(n, np.int8); lh = np.zeros(n, np.int8)
        confirm_lo = {p + K: lo[p] for p in range(n) if is_lo[p] and p + K < n}
        confirm_hi = {p + K: hi[p] for p in range(n) if is_hi[p] and p + K < n}
        for t in range(n):
            if t in confirm_lo: last2_lo = [last2_lo[1], confirm_lo[t]]
            if t in confirm_hi: last2_hi = [last2_hi[1], confirm_hi[t]]
            if np.isfinite(last2_lo[0]) and np.isfinite(last2_lo[1]):
                hl[t] = int(last2_lo[1] > last2_lo[0])
            if np.isfinite(last2_hi[0]) and np.isfinite(last2_hi[1]):
                lh[t] = int(last2_hi[1] < last2_hi[0])
        return hl, lh
    df["higher_lows"] = np.int8(0); df["lower_highs"] = np.int8(0)
    for _, idx in df.groupby("symbol", sort=False).groups.items():
        sub = df.loc[idx]
        hl, lh = _swing_series(sub)
        df.loc[idx, "higher_lows"] = hl
        df.loc[idx, "lower_highs"] = lh
    df["higher_lows"] = df["higher_lows"].astype("int8")
    df["lower_highs"] = df["lower_highs"].astype("int8")
    df["uptrend_struct"] = ((df["higher_lows"] == 1) & (df["lower_highs"] == 0)).astype("int8")

    # entry-morning gap: NEXT day's open vs this close (observed before you fill)
    df["next_open"] = g["open"].shift(-1).astype("float32")
    df["gap_entry"] = (df["next_open"] / df["close"] - 1).astype("float32")
    # atr_pct (optional)
    atr = _pick(cols, "atr_pct", "atrp")
    df["atr_pct"] = pd.to_numeric(df[atr], errors="coerce").astype("float32") if atr else np.float32(np.nan)
    # nifty regime (optional)
    nd = _pick(cols, "nifty_dist_sma50")
    df["nifty_below_sma50"] = (pd.to_numeric(df[nd], errors="coerce") < 0).astype("int8") if nd else np.int8(0)
    have_atr = atr is not None
    have_regime = nd is not None
    return df, have_atr, have_regime


def select_top(df, top_frac=None, top_n=None):
    if top_n is None and top_frac is None:
        top_n = 10
    out = []
    for d, gg in df.groupby("timestamp"):
        gg = gg.dropna(subset=["prob"])
        if gg.empty:
            continue
        if top_frac is not None:
            out.append(gg[gg["prob"] >= gg["prob"].quantile(1 - top_frac)])
        else:
            out.append(gg.nlargest(int(top_n), "prob"))
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


# ============================ forward outcome per pick ========================
def simulate(df, picks, horizon=5, stop_pct=None, cost_pct=0.28):
    """One fixed stop held constant (so the comparison isolates the ENTRY filter).
    stop_pct=None -> hold to horizon; else fixed touch stop at -stop_pct."""
    by = {}
    for s, gg in df.groupby("symbol", sort=False):
        gg = gg.sort_values("timestamp")
        by[s] = dict(dts=gg["timestamp"].values,
                     o=gg["open"].to_numpy(float), l=gg["low"].to_numpy(float),
                     c=gg["close"].to_numpy(float),
                     idx={pd.Timestamp(t): i for i, t in enumerate(gg["timestamp"].values)})
    cost = cost_pct / 100.0
    recs = []
    for _, p in picks.iterrows():
        s = p["symbol"]; pdte = pd.Timestamp(p["timestamp"]); rec = by.get(s)
        if rec is None or pdte not in rec["idx"]:
            continue
        i = rec["idx"][pdte]
        fwd = list(range(i + 1, min(i + 1 + horizon, len(rec["c"]))))
        if len(fwd) < horizon:
            continue
        entry = rec["o"][fwd[0]]
        if not np.isfinite(entry) or entry <= 0:
            continue
        l = rec["l"][fwd]; c = rec["c"][fwd]; dts = rec["dts"][fwd]
        # apply the (single) fixed stop
        if stop_pct is not None:
            lvl = entry * (1 - stop_pct)
            days = horizon; ret = c[-1] / entry - 1
            for k in range(horizon):
                if l[k] < lvl:
                    ret = min(lvl, rec["o"][fwd][k]) / entry - 1; days = k + 1; break
        else:
            days = horizon; ret = c[-1] / entry - 1
        ret_net = ret - cost
        marks = {pd.Timestamp(dts[k]): float(c[k] / entry - 1 - cost) for k in range(days - 1)}
        marks[pd.Timestamp(dts[days - 1])] = float(ret_net)
        recs.append(dict(
            symbol=s, pred_date=pdte, entry_date=pd.Timestamp(dts[0]),
            prob=float(p["prob"]), ret5_net=float(ret_net),
            mae=float(np.min(l[:days] / entry - 1)),
            exit=pd.Timestamp(dts[days - 1]), marks=marks,
            # carry the entry-state features needed by REGISTRY conditions
            pred_close=float(p["close"]), ema20=float(p["ema20"]),
            sma50=float(p["sma50"]), sma200=float(p["sma200"]),
            runup20=float(p["runup20"]), gap_entry=float(p["gap_entry"]),
            atr_pct=float(p["atr_pct"]), nifty_below_sma50=int(p["nifty_below_sma50"])))
    res = pd.DataFrame(recs)
    if res.empty:
        return res
    # top-ATR quartile flag computed WITHIN each day's picks (mirrors vol_skip)
    res["atr_topq"] = 0
    if res["atr_pct"].notna().any():
        thr = res.groupby("pred_date")["atr_pct"].transform(lambda s: s.quantile(0.75))
        res["atr_topq"] = (res["atr_pct"] >= thr).astype(int)
    return res


# ============================ permutation test ================================
def perm_pvalue(ret, excl, day_id, scope, n_perm=1000, rng=None):
    """Two-sided p for: mean(ret|kept) - mean(ret|excluded), under label exchange
    that respects daily structure. name-scope -> shuffle labels WITHIN each day;
    day-scope -> shuffle which DAYS are excluded. Returns (stat, p, n_excl)."""
    rng = rng or np.random.default_rng(0)
    ret = np.asarray(ret, float); excl = np.asarray(excl, bool)
    n_excl = int(excl.sum())
    if n_excl == 0 or n_excl == len(excl):
        return np.nan, np.nan, n_excl
    def stat_of(mask):
        a = ret[~mask]; b = ret[mask]
        return (a.mean() if len(a) else 0.0) - (b.mean() if len(b) else 0.0)
    obs = stat_of(excl)

    if scope == "day":
        days = np.unique(day_id)
        day_excl = {}
        for d in days:                                  # is this day an excluded day?
            day_excl[d] = bool(excl[day_id == d].any())
        ex_days = np.array([d for d in days if day_excl[d]])
        k = len(ex_days)
        cnt = 0
        for _ in range(n_perm):
            chosen = set(rng.choice(days, size=k, replace=False).tolist())
            m = np.array([day_id[j] in chosen for j in range(len(ret))])
            if abs(stat_of(m)) >= abs(obs):
                cnt += 1
        return obs, (cnt + 1) / (n_perm + 1), n_excl

    # name-scope: within-day shuffle, preserving each day's excluded count
    order = np.argsort(day_id, kind="stable")
    di = day_id[order]; ex = excl[order]
    bounds = np.searchsorted(di, np.unique(di), side="left").tolist() + [len(di)]
    seg = [(bounds[i], bounds[i + 1], int(ex[bounds[i]:bounds[i + 1]].sum()))
           for i in range(len(bounds) - 1)]
    cnt = 0
    for _ in range(n_perm):
        m = np.zeros(len(ret), bool)
        for a, b, k in seg:
            if k:
                pick = rng.choice(np.arange(a, b), size=k, replace=False)
                m[pick] = True
        m_orig = np.empty_like(m); m_orig[order] = m       # map back to original order
        if abs(stat_of(m_orig)) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1), n_excl


# ============================ portfolio (daily MTM) ===========================
def portfolio(res, mask_keep, slots=5, start_capital=100000.0):
    sub = res[mask_keep]
    recs = [dict(entry=r["entry_date"], exit=r["exit"], ret=float(r["ret5_net"]),
                 prob=float(r["prob"]), symbol=r["symbol"], marks=r["marks"])
            for _, r in sub.iterrows()]
    if not recs:
        return dict(cagr=np.nan, maxdd=np.nan, final_mult=np.nan, trades=0, ret_dd=np.nan)
    recs.sort(key=lambda x: (x["entry"], -x["prob"]))
    by_entry = {}
    for rr in recs:
        by_entry.setdefault(rr["entry"], []).append(rr)
    all_dates = sorted({d for rr in recs for d in rr["marks"].keys()})
    slot_base = [start_capital / slots] * slots
    slot_trade = [None] * slots
    eqc = []
    for d in all_dates:
        for s in range(slots):
            tr = slot_trade[s]
            if tr is not None and tr["exit"] == d:
                slot_base[s] *= (1 + tr["ret"]); slot_trade[s] = None
        cands = by_entry.get(d, [])
        held = {slot_trade[s]["symbol"] for s in range(slots) if slot_trade[s]}
        ci = 0
        for s in range(slots):
            if slot_trade[s] is None:
                while ci < len(cands) and cands[ci]["symbol"] in held:
                    ci += 1
                if ci >= len(cands):
                    break
                slot_trade[s] = cands[ci]; held.add(cands[ci]["symbol"]); ci += 1
        eq = sum(slot_base[s] if slot_trade[s] is None
                 else slot_base[s] * (1 + slot_trade[s]["marks"].get(d, 0.0))
                 for s in range(slots))
        eqc.append((d, eq))
    final = sum(slot_base)
    eq = pd.Series([e for _, e in eqc], index=[d for d, _ in eqc])
    if len(eq) < 2:
        return dict(cagr=np.nan, maxdd=np.nan, final_mult=final/start_capital,
                    trades=len(recs), ret_dd=np.nan)
    dd = (eq / eq.cummax() - 1).min()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    cagr = ((final / start_capital) ** (1 / yrs) - 1) * 100
    ret_dd = cagr / abs(dd * 100) if dd < 0 else np.nan
    return dict(cagr=cagr, maxdd=dd*100, final_mult=final/start_capital,
                trades=len(recs), ret_dd=ret_dd)


# ============================ main ===========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--top-frac", type=float, default=None)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--stop", default="none", help="'none' or a fraction like 0.15")
    ap.add_argument("--cost", type=float, default=0.28, help="round-trip cost %%")
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--start-capital", type=float, default=100000.0)
    ap.add_argument("--out", default="entry_filter")
    a = ap.parse_args()
    if not os.path.exists(a.panel):
        sys.exit(f"[error] panel not found: {a.panel}")
    stop_pct = None if str(a.stop).lower() == "none" else float(a.stop)

    print(f"[load] {a.panel}")
    df, have_atr, have_regime = load_panel(a.panel)
    print(f"[load] {len(df):,} rows | {df['symbol'].nunique()} symbols | "
          f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()} | "
          f"atr={'yes' if have_atr else 'NO'} | regime={'yes' if have_regime else 'NO'}")

    pool = df[df["in_test"]] if df["in_test"].any() else df
    if not df["in_test"].any():
        print("[warn] no in_test rows; using ALL history (results will be optimistic).")
    picks = select_top(pool, top_frac=a.top_frac, top_n=a.top_n)
    sel = f"top-{a.top_frac*100:.0f}%" if a.top_frac else f"top-{a.top_n or 10}"
    print(f"[pick] {len(picks):,} picks ({sel}/day)")
    res = simulate(df, picks, horizon=a.horizon, stop_pct=stop_pct, cost_pct=a.cost)
    if res.empty:
        sys.exit("[!] nothing resolved.")
    print(f"[sim ] {len(res):,} resolved  ({res['entry_date'].min().date()} -> "
          f"{res['entry_date'].max().date()}) | stop={a.stop} | cost={a.cost}%\n")

    registry = build_registry()
    # drop conditions whose inputs are absent
    reg = []
    for cdt in registry:
        if cdt["key"].startswith("high_atr") and not have_atr:
            continue
        if cdt["key"].startswith("regime") and not have_regime:
            continue
        reg.append(cdt)
    n_tests = len(reg)
    bonf = a.alpha / n_tests
    day_codes = res["pred_date"].astype("category").cat.codes.to_numpy()
    rng = np.random.default_rng(12345)

    # baseline portfolio (no filter)
    base = portfolio(res, np.ones(len(res), bool), a.slots, a.start_capital)

    rows = []
    for cdt in reg:
        excl = cdt["expr"](res).fillna(False).to_numpy(bool)
        stat, p, n_excl = perm_pvalue(res["ret5_net"].to_numpy(float), excl,
                                      day_codes, cdt["scope"], a.perm, rng)
        keep_mask = ~excl
        pf = portfolio(res, keep_mask, a.slots, a.start_capital)
        rows.append(dict(
            condition=cdt["key"], scope=cdt["scope"], desc=cdt["desc"],
            n_excl=n_excl, pct_excl=100.0*n_excl/len(res),
            mean_excl=res.loc[excl, "ret5_net"].mean()*100 if n_excl else np.nan,
            mean_kept=res.loc[keep_mask, "ret5_net"].mean()*100,
            sep=stat*100 if np.isfinite(stat) else np.nan,   # kept-minus-excluded, %%
            p_perm=p, p_bonf=min(p*n_tests, 1.0) if np.isfinite(p) else np.nan,
            cagr=pf["cagr"], maxdd=pf["maxdd"], ret_dd=pf["ret_dd"],
            d_ret_dd=(pf["ret_dd"]-base["ret_dd"]) if np.isfinite(pf["ret_dd"]) and np.isfinite(base["ret_dd"]) else np.nan))
    tab = pd.DataFrame(rows)
    tab["passes"] = (tab["p_perm"] < bonf) & (tab["d_ret_dd"] > 0)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("=" * 110)
    print(f" BASELINE (no filter):  CAGR {base['cagr']:+.1f}%   maxDD {base['maxdd']:.1f}%   "
          f"ret/DD {base['ret_dd']:.2f}   ({base['trades']:,} trades, {a.slots} slots, {sel}/day, stop={a.stop})")
    print("=" * 110)
    print(f" PRE-REGISTERED ENTRY FILTERS  --  {n_tests} tests, Bonferroni alpha = "
          f"{a.alpha}/{n_tests} = {bonf:.4f}")
    print("=" * 110)
    show = tab[["condition", "scope", "pct_excl", "mean_excl", "mean_kept",
                "sep", "p_perm", "p_bonf", "cagr", "maxdd", "ret_dd", "d_ret_dd", "passes"]]
    print(show.round({"pct_excl":1,"mean_excl":2,"mean_kept":2,"sep":3,"p_perm":4,
                      "p_bonf":3,"cagr":1,"maxdd":1,"ret_dd":2,"d_ret_dd":2}).to_string(index=False))

    winners = tab[tab["passes"]]
    print("\n" + "-" * 110)
    if len(winners):
        print(" PASSES (corrected-p < alpha AND ret/DD improves) -- forward-test these, don't trust them yet:")
        for _, r in winners.iterrows():
            print(f"   {r['condition']:20} excl {r['pct_excl']:.1f}% | "
                  f"sep {r['sep']:+.2f}% p_bonf {r['p_bonf']:.3f} | "
                  f"ret/DD {base['ret_dd']:.2f} -> {r['ret_dd']:.2f}")
    else:
        print(" PASSES: none. No pre-registered entry filter beat the model's soft weighting on")
        print(" a multiple-testing-corrected, risk-adjusted basis. The model likely already")
        print(" prices these. Don't hard-filter on any of them off this run.")
    print("-" * 110)

    tab.to_csv(f"{a.out}_results.csv", index=False)
    res.drop(columns=["marks"]).to_csv(f"{a.out}_picks.csv", index=False)
    print(f"\n[save] {a.out}_results.csv | {a.out}_picks.csv")
    print("\nNOTE: one in-sample OOS window; slippage/impact not modelled. A PASS here is a")
    print("hypothesis for the forward log. And remember: sizing is the bigger drawdown lever")
    print("than any of these filters -- a wide catastrophe stop + smaller per-name size does")
    print("more for survivable DD than excluding picks ever will.")


if __name__ == "__main__":
    main()
