"""Selection model tuned to the recent era of interregional racing (2022-2025).
Compares, on held-out 2022-2025 seasons:
  A  current team-score model (regional place, strength, H2H net vs top-5s) trained on all seasons
  B  same model trained only on 2021+ seasons
  C  head-to-head board model: every pair of teams on the board is compared directly using their own head-to-head and
     common-opponent results plus strength and regional place; the pick is the team that wins the most comparisons.
     Trained on the boundary pairs (selected vs first left out) of 2021+ seasons.
"""
import sys, pickle, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
S = sys.argv[1]; A = f"{S}/all"
src = open(f"{S}/scripts/selection_misses.py").read().split("rows, swaps = [], []")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src); sys.argv = argv
W = pickle.load(open(f"{A}/h2h_matrices.pkl", "rb"))
TEST = [2022, 2023, 2024, 2025]

def team_model(train_seasons, test):
    Xs, ys = [], []
    for (g2, s2), d in H[H.season.isin(train_seasons)].groupby(["gender", "season"]):
        X, y = boundary_pairs(d, COLS); Xs.append(X); ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys); sd = X.std(0) + 1e-9
    m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(X / sd, y)
    return pd.Series(test[COLS].values.astype(float) @ (m.coef_[0] / sd), index=test.index)

PF = ["d_place", "d_logrank", "d_net5", "h2h", "h2h_any", "common", "common_n"]
def pair_feats(d, w, a, b):
    hh, n = h2h(w, d.team[a], d.team[b]); cn, cnn = common(w, d.team[a], d.team[b])
    return [d.tplace[a] - d.tplace[b], np.log(d.strength_rank[a]) - np.log(d.strength_rank[b]), d.net_p5[a] - d.net_p5[b],
            np.sign(hh), float(n > 0), cn / max(cnn, 1), np.log1p(cnn)]
SYM = {"h2h_any", "common_n"}
def pair_training(seasons):
    X, y = [], []
    for (g, s), d in H[H.season.isin(seasons)].groupby(["gender", "season"]):
        w = W[(g, s)]; nonauto = d[~d.auto]; sel = nonauto[nonauto.went]
        out = [r.sort_values("tplace").loc[lambda q: ~q.went].index[0] for _, r in nonauto.groupby("region") if (~r.went).any()]
        for a in sel.index:
            for b in out:
                f = pair_feats(d, w, a, b); X.append(f); y.append(1)
                X.append([v if PF[k] in SYM else -v for k, v in enumerate(f)]); y.append(0)
    X = np.array(X, float); sd = X.std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=3000, fit_intercept=False).fit(X / sd, y)
    return m, sd

def fill_pairwise(d, w, n_at, m, sd):
    pools = {rg: list(r.sort_values("tplace").index) for rg, r in d[~d.auto].groupby("region")}
    picked = []
    for _ in range(n_at):
        board = [p[0] for p in pools.values() if p]
        score = {i: 0.0 for i in board}
        for x in range(len(board)):
            for y in range(x + 1, len(board)):
                a, b = board[x], board[y]
                p = m.predict_proba((np.array(pair_feats(d, w, a, b)) / sd)[None, :])[0, 1]
                score[a] += p; score[b] += 1 - p
        best = max(board, key=lambda i: score[i]); picked.append(best); pools[d.region[best]].pop(0)
    return picked

rows, detail = [], []
for (g, s), test in H[H.season.isin(TEST)].groupby(["gender", "season"]):
    n = int((test.went & ~test.auto).sum()); act = set(test[test.went & ~test.auto].index); w = W[(g, s)]
    res = {}
    res["A current, all seasons"] = fill(test, team_model([x for x in H.season.unique() if x != s], test), n)
    res["B current, 2021+ only"] = fill(test, team_model([x for x in H.season.unique() if x >= 2021 and x != s], test), n)
    m, sd = pair_training([x for x in H.season.unique() if x >= 2021 and x != s])
    res["C head-to-head board, 2021+"] = fill_pairwise(test, w, n, m, sd)
    m2, sd2 = pair_training([x for x in H.season.unique() if x != s])
    res["D head-to-head board, all seasons"] = fill_pairwise(test, w, n, m2, sd2)
    for k, p in res.items():
        rows.append(dict(gender=g, season=s, model=k, at_large=n, missed=n - len(set(p) & act)))
        for i in set(p) ^ act:
            detail.append(dict(model=k, gender=g, season=s, team=test.team[i], region=test.region[i], place=int(test.tplace[i]),
                               rank=int(test.strength_rank[i]), model_said="in" if i in p else "out"))
    print(g, s, {k: n - len(set(p) & act) for k, p in res.items()}, flush=True)
O = pd.DataFrame(rows); O.to_csv(f"{A}/selection_recent.csv", index=False); pd.DataFrame(detail).to_csv(f"{A}/selection_recent_misses.csv", index=False)
print(O.groupby("model")[["at_large", "missed"]].sum())
m, sd = pair_training([x for x in H.season.unique() if x >= 2021])
print("pairwise weights (per 1 SD):", dict(zip(PF, m.coef_[0].round(2))))
pickle.dump((m, sd), open(f"{A}/selection_pair_model.pkl", "wb"))
