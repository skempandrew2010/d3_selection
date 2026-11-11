"""With the 5-regional-runners rule: refit the common-opponents model, and test a strict rule order
(common opponents in the field, then head-to-head, then regional place). Held out by season, 2022-2025."""
import sys, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
S = sys.argv[1]
os.environ["AFILTER"] = "1"
src = open(f"{S}/scripts/selection_arule.py").read().split("# ---- 1) agreement")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src); sys.argv = argv
order = [b[0] for b in BINS]; allb = set(order)
def fit(seasons):
    X, y = [], []
    for (g, s), d in H[H.season.isin(seasons)].groupby(["gender", "season"]):
        res = RES[(g, s)]; field = set(d[d.went].team); na = d[~d.auto]; sel = na[na.went]
        out = [r.sort_values("tplace").loc[lambda q: ~q.went].index[0] for _, r in na.groupby("region") if (~r.went).any()]
        for a in sel.index:
            for b in out:
                cn, n = common(res, d.team[a], d.team[b], field, allb); hh = sgn(res, d.team[a], d.team[b], allb) or 0
                f = [cn / max(n, 1), hh, d.tplace[a] - d.tplace[b]]; X += [f, [-v for v in f]]; y += [1, 0]
    return LogisticRegression(fit_intercept=False, max_iter=2000).fit(np.array(X), y).coef_[0]
def run(d, res, n_at, coef=None):
    pools = {rg: list(r.sort_values("tplace").index) for rg, r in d[~d.auto].groupby("region")}
    field = set(d[d.auto].team); picked = []
    for _ in range(n_at):
        board = [p[0] for p in pools.values() if p]; sc = {i: 0.0 for i in board}
        for x in range(len(board)):
            for y in range(x + 1, len(board)):
                a, b = board[x], board[y]; ta, tb = d.team[a], d.team[b]
                cn, n = common(res, ta, tb, field, allb); hh = sgn(res, ta, tb, allb) or 0
                if coef is not None:
                    p = 1 / (1 + np.exp(-(coef[0] * cn / max(n, 1) + coef[1] * hh + coef[2] * (d.tplace[a] - d.tplace[b]))))
                else:
                    p = float(cn > 0) if cn else float(hh > 0) if hh else float(d.tplace[a] < d.tplace[b]) if d.tplace[a] != d.tplace[b] else 0.5
                sc[a] += p; sc[b] += 1 - p
        best = max(board, key=lambda i: (sc[i], -d.tplace[i])); picked.append(best); field.add(d.team[best]); pools[d.region[best]].pop(0)
    return picked
rows = []
for (g, s), d in H[H.season >= 2022].groupby(["gender", "season"]):
    res = RES[(g, s)]; n = int((d.went & ~d.auto).sum()); act = set(d[d.went & ~d.auto].index)
    c = fit([x for x in H.season.unique() if x >= 2021 and x != s])
    for name, p in (("refit weights", run(d, res, n, c)), ("strict rule order", run(d, res, n, None))):
        rows.append(dict(gender=g, season=s, model=name, missed=n - len(set(p) & act), at_large=n))
        for i in set(p) ^ act: rows[-1].setdefault("diff", []).append(f"{d.team[i]} ({d.region[i]} {d.tplace[i]}) {'in' if i in p else 'out'}")
    print(g, s, [(r["model"], r["missed"]) for r in rows[-2:]], flush=True)
O = pd.DataFrame(rows); print(O.groupby("model")[["at_large", "missed"]].sum())
print("final weights (2021-2025):", fit([x for x in H.season.unique() if x >= 2021]).round(3))
for r in O[O.model == "refit weights"].itertuples(): print(r.gender, r.season, getattr(r, "diff", ""))
