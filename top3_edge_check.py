#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
top3_edge_check.py -- "Do I have an edge trading the top 3 picks per day?"

Answers with numbers, not vibes:
  [A] Top-3-per-day OOS performance, entry = NEXT session open, two exits:
        - hold5   : exit day-5 close (the model's native thesis)
        - br15    : +15%/-15% intrabar bracket, pessimistic straddle (your live rule)
      Net of --cost-bps. Mean/median per trade, hit rate, daily-cohort Sharpe,
      max drawdown, monthly breakdown, and the "drop best-5 trades" fragility test.
  [B] Permutation test #1: top-3 vs 2000 draws of a RANDOM 3 from the whole
      scored universe that day  -> does the MODEL beat random stock picking?
  [C] Permutation test #2 (the one that matters for you): top-3 vs 2000 draws
      of a random 3 from that day's TOP DECILE -> does rank 1-3 beat rank 4-N?
      If p is high here, trade any 3 watchlist names you like; rank adds nothing.
  [D] Bootstrap 95% CI on mean net per trade (block bootstrap by day).
  [E] If forward_validation.csv exists (your frozen live record), repeats [A]
      on rank<=3 rows there -- the only truly untouched evidence you own.

Usage:
  python top3_edge_check.py ^
    --scored "C:\\Users\\karanvsi\\PyCharmMiscProject\\bigmove_deploy\\bigmove_scored_test.csv" ^
    --panel  "C:\\Users\\karanvsi\\Desktop\\Kite Connect\\out_rank_single\\panel_cache.parquet" ^
    --out    "C:\\Users\\karanvsi\\PyCharmMiscProject\\bigmove_deploy" ^
    --cost-bps 28

Interpretation guide printed at the end. No file is written except a small
summary CSV (top3_edge_summary.csv) in --out.
"""
import argparse, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scored", required=True, help="bigmove_scored_test.csv (OOS slice)")
ap.add_argument("--panel", required=True, help="panel_cache.parquet (same one the model scores from)")
ap.add_argument("--out", default=".", help="dir for summary CSV; also searched for forward_validation.csv")
ap.add_argument("--top-n", type=int, default=3)
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--cost-bps", type=float, default=28.0)
ap.add_argument("--tp-pct", type=float, default=15.0)
ap.add_argument("--sl-pct", type=float, default=15.0)
ap.add_argument("--n-perm", type=int, default=2000)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

H, N, COST = int(args.horizon), int(args.top_n), args.cost_bps / 1e4
rng = np.random.default_rng(args.seed)


def mkdate(s):
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


# ---------------- load ----------------
sc = pd.read_csv(args.scored)
dcol = "timestamp" if "timestamp" in sc.columns else "date"
sc["date"] = mkdate(sc[dcol])
if "prob" not in sc.columns:
    raise SystemExit(f"no 'prob' column in {args.scored}: {sc.columns.tolist()}")
sc = sc.dropna(subset=["prob", "date"]).copy()
print(f"[load] scored OOS rows: {len(sc):,}  days: {sc['date'].nunique()}  "
      f"({sc['date'].min().date()} -> {sc['date'].max().date()})")

panel = pd.read_parquet(args.panel, columns=None)
pd_dcol = next(c for c in ("timestamp", "date", "datetime") if c in panel.columns)
panel["date"] = mkdate(panel[pd_dcol])
panel = panel[["symbol", "date", "open", "high", "low", "close"]].sort_values(["symbol", "date"])
by_sym = {s: g.reset_index(drop=True) for s, g in panel.groupby("symbol", sort=False)}

# ---------------- resolve every scored row: next-open entry, 5d path ----------------
# hold5: entry=next open, exit=day-H close.  br15: +tp/-sl intrabar, SL wins ties,
# gap-through filled at the OPEN (realistic, not at the level).
TP, SL = args.tp_pct / 100.0, args.sl_pct / 100.0
cache = {}
def resolve(sym, d):
    key = (sym, d)
    if key in cache:
        return cache[key]
    g = by_sym.get(sym)
    out = None
    if g is not None:
        pos = g.index[g["date"] == d]
        if len(pos):
            i = int(pos[0]); fut = g.iloc[i + 1: i + 1 + H]
            if len(fut) == H:
                e = float(fut["open"].iloc[0])
                if np.isfinite(e) and e > 0:
                    hi = fut["high"].values / e - 1.0
                    lo = fut["low"].values / e - 1.0
                    op = fut["open"].values / e - 1.0
                    cl = fut["close"].values / e - 1.0
                    hold5 = cl[-1]
                    br = cl[-1]
                    for k in range(H):
                        gap_dn = op[k] <= -SL
                        gap_up = op[k] >= TP
                        hit_sl = lo[k] <= -SL
                        hit_tp = hi[k] >= TP
                        if gap_dn:               # gapped through stop -> filled at open
                            br = op[k]; break
                        if hit_sl:               # pessimistic: SL before TP same bar
                            br = -SL; break
                        if gap_up:               # gapped through target -> open (better)
                            br = op[k]; break
                        if hit_tp:
                            br = TP; break
                    out = (hold5, br)
    cache[key] = out
    return out

res = [resolve(s, d) for s, d in zip(sc["symbol"], sc["date"])]
sc["hold5"] = [r[0] if r else np.nan for r in res]
sc["br15"] = [r[1] if r else np.nan for r in res]
sc = sc.dropna(subset=["hold5"]).copy()
sc["hold5_net"] = sc["hold5"] - COST
sc["br15_net"] = sc["br15"] - COST
print(f"[resolve] usable rows after next-open resolution: {len(sc):,}")

# ---------------- [A] top-N per day ----------------
sc = sc.sort_values(["date", "prob"], ascending=[True, False])
sc["rank"] = sc.groupby("date")["prob"].rank(ascending=False, method="first")
dec_cut = sc.groupby("date")["prob"].transform(lambda s: s.quantile(0.9))
sc["in_top_decile"] = sc["prob"] >= dec_cut

topN = sc[sc["rank"] <= N].copy()

def report(df, col, label):
    x = df[col].values
    daily = df.groupby("date")[col].mean()          # equal-weight cohort per day
    sh = daily.mean() / daily.std(ddof=1) * np.sqrt(252) if daily.std(ddof=1) > 0 else 0.0
    eq = daily.cumsum(); dd = (eq - eq.cummax()).min()
    srt = np.sort(x)[::-1]
    drop5 = srt[5:].mean() if len(srt) > 20 else np.nan
    print(f"\n  --- {label} ({col}) ---")
    print(f"  trades {len(x):4d} | days {daily.size:3d} | hit {(x>0).mean()*100:5.1f}% | "
          f"mean {x.mean()*100:+6.2f}% | median {np.median(x)*100:+6.2f}%")
    print(f"  daily-cohort Sharpe(ann) {sh:5.2f} | cum(additive) {eq.iloc[-1]*100:+7.1f}% | "
          f"maxDD {dd*100:+6.1f}% | mean w/o best-5 trades {drop5*100:+6.2f}%")
    mo = df.set_index("date")[col].groupby(pd.Grouper(freq="ME")).agg(["count", "mean"])
    mo = mo[mo["count"] > 0]
    print("  monthly mean%: " + "  ".join(f"{i.strftime('%Y-%m')}:{v*100:+5.2f}" for i, v in mo["mean"].items()))
    return dict(label=label, exit=col, n=len(x), hit=float((x > 0).mean()),
                mean=float(x.mean()), median=float(np.median(x)),
                sharpe_ann=float(sh), cum=float(eq.iloc[-1]), maxdd=float(dd),
                mean_wo_best5=float(drop5) if np.isfinite(drop5) else np.nan)

print("\n================ [A] TOP-%d PER DAY, OOS ================" % N)
rows = [report(topN, "hold5_net", f"top{N} hold-to-day5"),
        report(topN, "br15_net", f"top{N} bracket {args.tp_pct:.0f}/{args.sl_pct:.0f}")]

# ---------------- [B]/[C] permutation tests ----------------
def perm_test(pool_mask, label, col="hold5_net"):
    obs = topN[col].mean()
    days = topN["date"].unique()
    pools = {d: g[col].values for d, g in sc[pool_mask].groupby("date") if len(g) >= N}
    days = [d for d in days if d in pools]
    if len(days) < 10:
        print(f"  [{label}] not enough days with a pool -- skipped."); return np.nan
    sims = np.empty(args.n_perm)
    for j in range(args.n_perm):
        tot = 0.0; cnt = 0
        for d in days:
            p = pools[d]
            pick = p if len(p) <= N else p[rng.choice(len(p), N, replace=False)]
            tot += pick.sum(); cnt += len(pick)
        sims[j] = tot / cnt
    p = float((sims >= obs).mean())
    print(f"  [{label}] top{N} mean {obs*100:+.2f}%  vs random-{N} mean {sims.mean()*100:+.2f}% "
          f"(sd {sims.std()*100:.2f}) -> p = {p:.3f}")
    return p

print("\n================ [B]/[C] IS THE RANKING REAL? ================")
p_univ = perm_test(pd.Series(True, index=sc.index), "vs random 3 from WHOLE universe")
p_dec = perm_test(sc["in_top_decile"], "vs random 3 from TOP DECILE (rank test)")

# ---------------- [D] block bootstrap CI ----------------
daily = topN.groupby("date")["hold5_net"].mean().values
boots = np.array([rng.choice(daily, len(daily), replace=True).mean() for _ in range(5000)])
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"\n================ [D] BOOTSTRAP ================")
print(f"  mean daily top{N} cohort return {daily.mean()*100:+.3f}%  95% CI [{lo*100:+.3f}%, {hi*100:+.3f}%]"
      f"  -> {'CI EXCLUDES ZERO' if lo > 0 else 'CI INCLUDES ZERO (edge not proven at 95%)'}")

# ---------------- [E] frozen live record ----------------
fv_path = os.path.join(args.out, "forward_validation.csv")
if os.path.exists(fv_path):
    fv = pd.read_csv(fv_path)
    fv = fv[(fv.get("status") == "resolved")]
    if "rank" in fv.columns and len(fv):
        fv3 = fv[pd.to_numeric(fv["rank"], errors="coerce") <= N]
        r = pd.to_numeric(fv3["ret_5"], errors="coerce").dropna() - COST
        print(f"\n================ [E] FROZEN LIVE RECORD (rank<=%d) ================" % N)
        print(f"  n {len(r)} | hit {(r>0).mean()*100:.1f}% | mean {r.mean()*100:+.2f}% | median {r.median()*100:+.2f}%")
        if len(r) < 30:
            print("  (fewer than ~30 resolved -- directional only, keep it running)")
else:
    print(f"\n[E] no forward_validation.csv in {args.out} -- run resolve_forward.py to build the live record.")

pd.DataFrame(rows).assign(p_vs_universe=p_univ, p_vs_top_decile=p_dec,
                          boot_lo=lo, boot_hi=hi).to_csv(
    os.path.join(args.out, "top3_edge_summary.csv"), index=False)

print("""
================ HOW TO READ THIS ================
EDGE = YES needs ALL of:   mean net > 0 on BOTH exits,  bootstrap CI excluding 0,
                           p(vs universe) < 0.05,  and mean survives dropping best-5.
RANK 1-3 MATTERS only if:  p(vs top decile) < ~0.10. Otherwise any 3 watchlist
                           names are statistically the same -- pick the 3 you can
                           hold calmly (lower atr_pct) instead of rank.
Remember: this OOS block is ONE contiguous regime slice. The frozen forward
record [E] is the evidence that compounds -- weight it more every week.
""")
