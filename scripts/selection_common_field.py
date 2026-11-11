"""No team strength: pick at-large teams from the board (best team left in each region) using results against common
opponents that are already in the field (automatic qualifiers + at-large picks so far).
Variants (held-out check on every season, nothing is fit except V4):
  V1 common opponents in the field, team vs team: each board pair compared on their net results vs shared field teams;
     ties go to head-to-head, then regional place. Pick = most comparison wins.
  V2 record vs the field: each board team's wins minus losses against teams already in the field. Ties: regional place.
  V3 V1 but common opponents among the field AND the other board teams (the manual's 'teams in the field and other
     potential at-large teams').
  V4 fitted team-vs-team model with only common-opponents-in-field, head-to-head and regional place (2021+ training).
"""
import sys, pickle, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
S = sys.argv[1]; A = f"{S}/all"
H = pd.read_csv(f"{A}/selection_history.csv")
FULL = H.groupby(["gender", "season"]).region.nunique()
H = H[[FULL.get((g, s), 0) >= (10 if s >= 2021 else 8) for g, s in zip(H.gender, H.season)]].copy()
W = pickle.load(open(f"{A}/h2h_matrices.pkl", "rb"))

def sgn(w, a, b):
    m = w.get(a, {}).get(b, [])
    return int(np.sign(sum(np.sign(x[1]) for x in m))) if m else None

def common_vs(w, a, b, opps):
    net = n = 0
    for o in opps:
        if o in (a, b): continue
        ra, rb = sgn(w, a, o), sgn(w, b, o)
        if ra is None or rb is None: continue
        net += ra - rb; n += 1
    return net, n

def run(d, w, n_at, mode, model=None):
    pools = {rg: list(r.sort_values("tplace").index) for rg, r in d[~d.auto].groupby("region")}
    field = set(d[d.auto].team); picked = []
    for _ in range(n_at):
        board = [p[0] for p in pools.values() if p]
        bt = {i: d.team[i] for i in board}
        if mode == "V2":
            def rec(i):
                r = [sgn(w, bt[i], o) for o in field]; r = [x for x in r if x is not None]
                return sum(x > 0 for x in r) - sum(x < 0 for x in r)
            best = max(board, key=lambda i: (rec(i), -d.tplace[i]))
        else:
            opps = field | (set(bt.values()) if mode == "V3" else set())
            score = {i: 0.0 for i in board}
            for x in range(len(board)):
                for y in range(x + 1, len(board)):
                    a, b = board[x], board[y]
                    cn, n = common_vs(w, bt[a], bt[b], opps); hh = sgn(w, bt[a], bt[b]) or 0
                    if mode == "V4":
                        f = np.array([cn / max(n, 1), hh, d.tplace[a] - d.tplace[b]])
                        p = 1 / (1 + np.exp(-(f @ model)))
                    else:
                        if cn != 0: p = float(cn > 0)
                        elif hh != 0: p = float(hh > 0)
                        elif d.tplace[a] != d.tplace[b]: p = float(d.tplace[a] < d.tplace[b])
                        else: p = 0.5
                    score[a] += p; score[b] += 1 - p
            best = max(board, key=lambda i: (score[i], -d.tplace[i]))
        picked.append(best); field.add(d.team[best]); pools[d.region[best]].pop(0)
    return picked

def fit_v4(seasons):
    """boundary pairs: each selected at-large team vs each region's first team left out, opponents = final field."""
    X, y = [], []
    for (g, s), d in H[H.season.isin(seasons)].groupby(["gender", "season"]):
        w = W[(g, s)]; field = set(d[d.went].team); na = d[~d.auto]; sel = na[na.went]
        out = [r.sort_values("tplace").loc[lambda q: ~q.went].index[0] for _, r in na.groupby("region") if (~r.went).any()]
        for a in sel.index:
            for b in out:
                cn, n = common_vs(w, d.team[a], d.team[b], field); hh = sgn(w, d.team[a], d.team[b]) or 0
                f = [cn / max(n, 1), hh, d.tplace[a] - d.tplace[b]]; X += [f, [-v for v in f]]; y += [1, 0]
    m = LogisticRegression(fit_intercept=False, max_iter=2000).fit(np.array(X), y)
    return m.coef_[0]

rows = []
for (g, s), d in H.groupby(["gender", "season"]):
    w = W[(g, s)]; n = int((d.went & ~d.auto).sum()); act = set(d[d.went & ~d.auto].index)
    for mode in ("V1", "V2", "V3", "V4"):
        model = fit_v4([x for x in H.season.unique() if x >= 2021 and x != s]) if mode == "V4" else None
        p = run(d, w, n, mode, model)
        rows.append(dict(gender=g, season=s, model=mode, at_large=n, missed=n - len(set(p) & act)))
    print(g, s, {r["model"]: r["missed"] for r in rows[-4:]}, flush=True)
O = pd.DataFrame(rows); O.to_csv(f"{A}/selection_common_field.csv", index=False)
print("\nALL SEASONS"); print(O.groupby("model")[["at_large", "missed"]].sum())
print("\n2022-2025"); print(O[O.season >= 2022].groupby("model")[["at_large", "missed"]].sum())
print("V4 weights (common net share, head-to-head, regional place):", fit_v4([x for x in H.season.unique() if x >= 2021]).round(3))
