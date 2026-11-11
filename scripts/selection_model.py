"""Learn how the DIII committee picks at-large teams, then test it season by season (leave one season out).
The committee works through a 'board': only the highest unselected team from each region can be picked next, because a team
cannot be chosen over a team that beat it at its regional (true in every season 2015-2025). So the model learns a score for
each candidate and fills the at-large spots greedily from the board. The score is fit as a pairwise ranking: every team
the committee took must outrank every 'first team left out' in each region.
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
S = sys.argv[1]; A = f"{S}/all"
H = pd.read_csv(f"{A}/selection_history.csv")
FULL = H.groupby(["gender", "season"]).region.nunique()
H = H[[FULL.get((g, s), 0) >= (10 if s >= 2021 else 8) for g, s in zip(H.gender, H.season)]].copy()    # complete regionals only
H["h2h_net_pool"] = H.h2h_w_pool - H.h2h_l_pool; H["h2h_net_auto"] = H.h2h_w_auto - H.h2h_l_auto
H["h2h_games_pool"] = H.h2h_w_pool + H.h2h_l_pool
H["log_strength_rank"] = np.log(H.strength_rank.fillna(400))
H["gap_ahead"] = H.gap_ahead.fillna(0); H["gap_behind"] = H.gap_behind.fillna(200).clip(upper=200)
# region strength: how many teams from the region rank in the national top 40 by strength
H["region_top40"] = H.groupby(["gender", "season", "region"]).strength_rank.transform(lambda v: (v <= 40).sum())
FEATS = {"regional place only": ["tplace"],
         "team strength only": ["log_strength_rank"],
         "committee criteria": ["tplace", "log_strength_rank", "h2h_net_pool", "h2h_net_auto", "gap_ahead", "gap_behind", "region_top40"]}

def boundary_pairs(d, cols):
    """selected at-large teams vs the first team left out in each region"""
    X, y = [], []
    nonauto = d[~d.auto]
    sel = nonauto[nonauto.went]
    out = []
    for rg, r in nonauto.groupby("region"):
        r = r.sort_values("tplace"); nxt = r[~r.went]
        if len(nxt): out.append(nxt.iloc[0])
    out = pd.DataFrame(out)
    for _, a in sel.iterrows():
        for _, b in out.iterrows():
            diff = a[cols].values.astype(float) - b[cols].values.astype(float)
            X += [diff, -diff]; y += [1, 0]
    return np.array(X), np.array(y)

def fill(d, score, n_at):
    """greedy board: each step, the best-scoring among the highest unselected team in every region."""
    pools = {rg: list(r.sort_values("tplace").index) for rg, r in d[~d.auto].groupby("region")}
    picked = []
    for _ in range(n_at):
        board = {rg: p[0] for rg, p in pools.items() if p}
        if not board: break
        rg = max(board, key=lambda k: score[board[k]])
        picked.append(board[rg]); pools[rg].pop(0)
    return picked

rows, preds = [], []
seasons = sorted(H.groupby(["gender", "season"]).groups)
for name, cols in FEATS.items():
    for (g, s) in seasons:
        test = H[(H.gender == g) & (H.season == s)]
        train = H[(H.season != s)]
        Xs, ys = [], []
        for (g2, s2), d in train.groupby(["gender", "season"]):
            X, y = boundary_pairs(d, cols); Xs.append(X); ys.append(y)
        X, y = np.vstack(Xs), np.concatenate(ys)
        mu, sd = X.std(0) * 0 + 0, X.std(0) + 1e-9
        m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(X / sd, y)
        score = pd.Series((test[cols].values / sd) @ m.coef_[0], index=test.index)
        n_at = int((test.went & ~test.auto).sum())
        picked = fill(test, score, n_at)
        actual = set(test[test.went & ~test.auto].index)
        right = len(set(picked) & actual)
        rows.append(dict(model=name, gender=g, season=s, at_large=n_at, correct=right, missed=n_at - right))
        if name == "committee criteria":
            for i in set(picked) ^ actual:
                preds.append(dict(gender=g, season=s, team=test.team[i], region=test.region[i], regional_place=int(test.tplace[i]),
                                  strength_rank=test.strength_rank[i], model_said="in" if i in picked else "out", committee="in" if i in actual else "out"))
O = pd.DataFrame(rows); O.to_csv(f"{A}/selection_backtest.csv", index=False); pd.DataFrame(preds).to_csv(f"{A}/selection_misses.csv", index=False)
pd.set_option("display.width", 200)
print(O.pivot_table(index=["gender", "season"], columns="model", values="missed").to_string())
print("\nTOTAL at-large picks", O.groupby("model").at_large.sum().iloc[0], "| missed:", O.groupby("model").missed.sum().to_dict())
# final coefficients on all seasons
Xs, ys = [], []
for (g2, s2), d in H.groupby(["gender", "season"]):
    X, y = boundary_pairs(d, FEATS["committee criteria"]); Xs.append(X); ys.append(y)
X, y = np.vstack(Xs), np.concatenate(ys); sd = X.std(0) + 1e-9
m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(X / sd, y)
coef = pd.Series(m.coef_[0], index=FEATS["committee criteria"]); print("\nstandardized weights:\n", coef.round(3).to_string())
pd.DataFrame({"coef": m.coef_[0], "sd": sd}, index=FEATS["committee criteria"]).to_csv(f"{A}/selection_weights.csv")
print(pd.DataFrame(preds).to_string(index=False))
