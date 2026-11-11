"""Deterministic selection systems vs what the committee actually did (2021-2025, five-regional-runners rule applied).
K1 Kolas-style: each round, the board team with the most wins over teams already in the field gets the next spot.
K2 same, net record (wins minus losses) vs the field.
Ties in both: head-to-head among the tied teams, then common opponents in the field, then regional place, then regional score."""
import sys, os
import numpy as np, pandas as pd
S = sys.argv[1]; os.environ["AFILTER"] = "1"
src = open(f"{S}/scripts/selection_arule.py").read().split("# ---- 1) agreement")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src); sys.argv = argv
allb = {b[0] for b in BINS}
def run(d, res, n_at, mode):
    pools = {rg: list(r.sort_values("tplace").index) for rg, r in d[~d.auto].groupby("region")}
    field = set(d[d.auto].team); picked = []
    for _ in range(n_at):
        board = [p[0] for p in pools.values() if p]
        def pts(i):
            r = [sgn(res, d.team[i], o, allb) for o in field]; r = [x for x in r if x is not None]
            w, l = sum(x > 0 for x in r), sum(x < 0 for x in r)
            return w if mode == "K1" else w - l
        P = {i: pts(i) for i in board}; top = max(P.values()); tied = [i for i in board if P[i] == top]
        if len(tied) > 1:          # tiebreak 1: head-to-head among tied teams
            hw = {i: sum((sgn(res, d.team[i], d.team[j], allb) or 0) for j in tied if j != i) for i in tied}
            m = max(hw.values()); tied = [i for i in tied if hw[i] == m]
        if len(tied) > 1:          # tiebreak 2: common opponents in the field
            cw = {i: sum(common(res, d.team[i], d.team[j], field, allb)[0] for j in tied if j != i) for i in tied}
            m = max(cw.values()); tied = [i for i in tied if cw[i] == m]
        best = min(tied, key=lambda i: (d.tplace[i], d.score[i]))
        picked.append(best); field.add(d.team[best]); pools[d.region[best]].pop(0)
    return picked
rows = []
for (g, s), d in H[H.season >= 2021].groupby(["gender", "season"]):
    res = RES[(g, s)]; n = int((d.went & ~d.auto).sum()); act = set(d[d.went & ~d.auto].index)
    for mode in ("K1", "K2"):
        p = run(d, res, n, mode)
        rows.append(dict(gender=g, season=s, system=mode, at_large=n, differs=n - len(set(p) & act),
                         swaps="; ".join(f"{d.team[i]} {'in' if i in p else 'out'}" for i in sorted(set(p) ^ act, key=lambda i: i not in p))))
O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
print(O.groupby("system")[["at_large", "differs"]].sum())
print(O[O.system == "K1"][["gender", "season", "differs", "swaps"]].to_string(index=False))
O.to_csv(f"{A}/selection_proposals.csv", index=False)
