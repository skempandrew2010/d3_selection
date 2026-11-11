"""Does the selection committee trust late-season meets more than early ones?
Uses the committee's revealed choices: every at-large team it took vs every region's first team left out.
For each pair, collect head-to-head and common-opponent evidence by when the meets were run (days before regionals),
and check how often each period's evidence agrees with the committee. Then backtest the common-opponents-in-field
selection with different time weightings.
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
S = sys.argv[1]; A = f"{S}/all"
src = open(f"{S}/scripts/team_adjust.py").read().split("# ---- out-of-sample strengths")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src); sys.argv = argv
H = pd.read_csv(f"{A}/selection_history.csv")
FULL = H.groupby(["gender", "season"]).region.nunique()
H = H[[FULL.get((g, s), 0) >= (10 if s >= 2021 else 8) for g, s in zip(H.gender, H.season)]].copy()
BINS = [("regional", 0, 0), ("last 3 weeks (conference)", 1, 21), ("3 to 5 weeks out", 22, 35), ("5+ weeks out", 36, 999)]
def binof(days):
    for name, lo, hi in BINS:
        if lo <= days <= hi: return name

def results(g, s):
    Rs = R[(R.gender == g) & (R.season == s)]; RD = Rs[Rs.rtype == "regional"].date.max()
    out = {}                                  # out[a][b] = list of (bin, sign a ahead of b)
    for rk, x in Rs[Rs.date <= RD].groupby("rk"):
        x = x.sort_values("place"); cnt = x.team.value_counts(); full = set(cnt[cnt >= 5].index)
        if len(full) < 2: continue
        y = x[x.team.isin(full)].copy(); y["k"] = y.groupby("team").cumcount() + 1; y = y[y.k <= 7]; y["tp"] = np.arange(1, len(y) + 1)
        sc = y[y.k <= 5].groupby("team").tp.sum().sort_values(); tt = list(sc.index)
        b_ = binof((RD - x.date.iloc[0]).days)
        for i, a in enumerate(tt):
            for b in tt[i + 1:]:
                out.setdefault(a, {}).setdefault(b, []).append((b_, 1)); out.setdefault(b, {}).setdefault(a, []).append((b_, -1))
    return out

def sgn(res, a, b, bins, wts=None):
    m = [(bn, v) for bn, v in res.get(a, {}).get(b, []) if bn in bins]
    if not m: return None
    tot = sum(v * (wts.get(bn, 1) if wts else 1) for bn, v in m)
    return int(np.sign(tot))

def common(res, a, b, opps, bins, wts=None):
    net = n = 0
    for o in opps:
        if o in (a, b): continue
        ra, rb = sgn(res, a, o, bins, wts), sgn(res, b, o, bins, wts)
        if ra is None or rb is None: continue
        net += ra - rb; n += 1
    return net, n

RES = {k: results(*k) for k in H.groupby(["gender", "season"]).groups}
# ---- 1) agreement by period on the committee's boundary choices
rows = []
for (g, s), d in H.groupby(["gender", "season"]):
    res = RES[(g, s)]; field = set(d[d.went].team); na = d[~d.auto]; sel = na[na.went]
    out = [r.sort_values("tplace").loc[lambda q: ~q.went].index[0] for _, r in na.groupby("region") if (~r.went).any()]
    for a in sel.index:
        for b in out:
            ta, tb = d.team[a], d.team[b]
            for name, _, _ in BINS:
                hh = sgn(res, ta, tb, {name})
                if hh: rows.append(dict(season=s, gender=g, kind="head-to-head", period=name, agree=hh > 0))
                if name != "regional":
                    cn, n = common(res, ta, tb, field, {name})
                    if cn: rows.append(dict(season=s, gender=g, kind="common opponents in field", period=name, agree=cn > 0, n=n))
AG = pd.DataFrame(rows)
order = [b[0] for b in BINS]
t = AG.groupby(["kind", "period"]).agree.agg(["mean", "size"]).reset_index()
t["period"] = pd.Categorical(t.period, order); t = t.sort_values(["kind", "period"])
print("HOW OFTEN EACH PERIOD'S EVIDENCE SIDES WITH THE COMMITTEE (selected team vs first team left out), 2015-2025")
print(t.assign(mean=lambda q: (q["mean"] * 100).round(0)).rename(columns={"mean": "agrees %", "size": "comparisons"}).to_string(index=False))
t2 = AG[AG.season >= 2021].groupby(["kind", "period"]).agree.agg(["mean", "size"]).reset_index()
t2["period"] = pd.Categorical(t2.period, order); t2 = t2.sort_values(["kind", "period"])
print("\n2021-2025 only"); print(t2.assign(mean=lambda q: (q["mean"] * 100).round(0)).to_string(index=False))
AG.to_csv(f"{A}/selection_timing_agreement.csv", index=False)

# ---- 2) backtest common-opponents-in-field selection with time weights
def run(d, res, n_at, wts, coef):
    pools = {rg: list(r.sort_values("tplace").index) for rg, r in d[~d.auto].groupby("region")}
    field = set(d[d.auto].team); picked = []; allb = set(order)
    for _ in range(n_at):
        board = [p[0] for p in pools.values() if p]; sc = {i: 0.0 for i in board}
        for x in range(len(board)):
            for y in range(x + 1, len(board)):
                a, b = board[x], board[y]; ta, tb = d.team[a], d.team[b]
                cn, n = common(res, ta, tb, field, allb, wts); hh = sgn(res, ta, tb, allb, wts) or 0
                z = coef[0] * cn / max(n, 1) + coef[1] * hh + coef[2] * (d.tplace[a] - d.tplace[b]); p = 1 / (1 + np.exp(-z))
                sc[a] += p; sc[b] += 1 - p
        best = max(board, key=lambda i: sc[i]); picked.append(best); field.add(d.team[best]); pools[d.region[best]].pop(0)
    return picked
COEF = (3.054, 1.939, -0.348)
WEIGHTS = {"all meets equal": None,
           "later counts more (regional 3, last 3 wks 2, 3-5 wks 1.5, earlier 1)": {"regional": 3, "last 3 weeks (conference)": 2, "3 to 5 weeks out": 1.5, "5+ weeks out": 1},
           "earlier counts more": {"regional": 1, "last 3 weeks (conference)": 1.5, "3 to 5 weeks out": 2, "5+ weeks out": 3},
           "drop meets 5+ weeks out": {"regional": 1, "last 3 weeks (conference)": 1, "3 to 5 weeks out": 1, "5+ weeks out": 0},
           "drop the regional": {"regional": 0, "last 3 weeks (conference)": 1, "3 to 5 weeks out": 1, "5+ weeks out": 1}}
bt = []
for (g, s), d in H[H.season >= 2022].groupby(["gender", "season"]):
    res = RES[(g, s)]; n = int((d.went & ~d.auto).sum()); act = set(d[d.went & ~d.auto].index)
    for name, w in WEIGHTS.items():
        p = run(d, res, n, w, COEF); bt.append(dict(gender=g, season=s, weighting=name, missed=n - len(set(p) & act), at_large=n))
B = pd.DataFrame(bt); B.to_csv(f"{A}/selection_timing_backtest.csv", index=False)
print("\nBACKTEST 2022-2025, common opponents in the field (176 at-large picks)")
print(B.groupby("weighting")[["at_large", "missed"]].sum().sort_values("missed").to_string())
