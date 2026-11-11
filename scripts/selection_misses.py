"""Why does the selection model miss at-large picks? Replays each season (weights fit without that season), lists every
wrong swap, and tests a committee-style rule that compares the teams on the board directly: head-to-head between them,
then common opponents, then regional point gap, then team strength.
"""
import sys, re, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
S = sys.argv[1]; A = f"{S}/all"
src = open(f"{S}/scripts/team_adjust.py").read().split("# ---- out-of-sample strengths")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src); sys.argv = argv
src2 = open(f"{S}/scripts/selection_model.py").read().split("rows, preds = [], []")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src2); sys.argv = argv
H["net_p5"] = H.h2h_w_p5 - H.h2h_l_p5
COLS = ["tplace", "log_strength_rank", "net_p5"]

def pairs_matrix(g, s):
    Rs = R[(R.gender == g) & (R.season == s)]; RD = Rs[Rs.rtype == "regional"].date.max()
    w = {}; meets = {}
    for rk, x in Rs[(Rs.date <= RD)].groupby("rk"):
        x = x.sort_values("place"); cnt = x.team.value_counts(); full = set(cnt[cnt >= 5].index)
        if len(full) < 2: continue
        y = x[x.team.isin(full)].copy(); y["k"] = y.groupby("team").cumcount() + 1; y = y[y.k <= 7]; y["tp"] = np.arange(1, len(y) + 1)
        sc = y[y.k <= 5].groupby("team").tp.sum().sort_values(); tt = list(sc.index); reg = x.rtype.iloc[0] == "regional"
        for i, a in enumerate(tt):
            for b in tt[i + 1:]:
                w.setdefault(a, {}).setdefault(b, []).append((x.meet.iloc[0], sc[b] - sc[a], reg))
                w.setdefault(b, {}).setdefault(a, []).append((x.meet.iloc[0], sc[a] - sc[b], reg))
    return w

def h2h(w, a, b):
    """net meets won by a over b (excluding the regional, which the board rule already uses)"""
    r = [np.sign(m[1]) for m in w.get(a, {}).get(b, []) if not m[2]]
    return int(sum(r)), len(r)

def common(w, a, b):
    oa, ob = set(w.get(a, {})), set(w.get(b, {}))
    net = 0; n = 0
    for o in (oa & ob) - {a, b}:
        ra = np.sign(sum(np.sign(m[1]) for m in w[a][o])); rb = np.sign(sum(np.sign(m[1]) for m in w[b][o]))
        net += int(ra - rb); n += 1
    return net, n

def fit(exclude):
    Xs, ys = [], []
    for (g2, s2), d in H.groupby(["gender", "season"]):
        if s2 == exclude: continue
        X, y = boundary_pairs(d, COLS); Xs.append(X); ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys); sd = X.std(0) + 1e-9
    m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(X / sd, y)
    return m.coef_[0] / sd

def fill_rule(d, w, n_at, mode):
    """board replay; mode 'copeland': each step, compare every pair on the board in criteria order and take the team with
    the most comparison wins (ties broken by team strength)."""
    pools = {rg: list(r.sort_values("tplace").index) for rg, r in d[~d.auto].groupby("region")}
    picked = []
    for _ in range(n_at):
        board = [p[0] for p in pools.values() if p]
        wins = {i: 0.0 for i in board}
        for x in range(len(board)):
            for y in range(x + 1, len(board)):
                a, b = board[x], board[y]; ta, tb = d.team[a], d.team[b]
                hh, n = h2h(w, ta, tb)
                if n and hh != 0: win = a if hh > 0 else b
                else:
                    cn, cnn = common(w, ta, tb)
                    if mode == "copeland" and cnn >= 3 and cn != 0: win = a if cn > 0 else b
                    else: win = a if d.strength_rank[a] < d.strength_rank[b] else b
                wins[win] += 1
        best = max(board, key=lambda i: (wins[i], -d.strength_rank[i]))
        picked.append(best); pools[d.region[best]].pop(0)
    return picked

rows, swaps = [], []
for (g, s), test in H.groupby(["gender", "season"]):
    w = pairs_matrix(g, s); coef = fit(s)
    sc = pd.Series(test[COLS].values.astype(float) @ coef, index=test.index)
    n = int((test.went & ~test.auto).sum()); act = set(test[test.went & ~test.auto].index)
    pm = fill(test, sc, n); pr = fill_rule(test, w, n, "copeland")
    rows.append(dict(gender=g, season=s, at_large=n, model_missed=n - len(set(pm) & act), rule_missed=n - len(set(pr) & act)))
    wrong_in = [i for i in pm if i not in act]; wrong_out = [i for i in act if i not in pm]
    for a in wrong_out:                      # committee took a, model did not
        for b in wrong_in:                   # model took b, committee did not
            ta, tb = test.team[a], test.team[b]; hh, nn = h2h(w, ta, tb); cn, cnn = common(w, ta, tb)
            swaps.append(dict(gender=g, season=s, committee_took=ta, c_region=test.region[a], c_place=int(test.tplace[a]), c_rank=int(test.strength_rank[a]),
                              model_took=tb, m_region=test.region[b], m_place=int(test.tplace[b]), m_rank=int(test.strength_rank[b]),
                              h2h=f"{(nn + hh)//2}-{(nn - hh)//2}" if nn else "never met", common=f"{cn:+d} of {cnn}",
                              c_gap_behind=test.gap_behind[a], m_gap_behind=test.gap_behind[b],
                              h2h_favors=("committee pick" if hh > 0 else "model pick" if hh < 0 else "even/none"),
                              common_favors=("committee pick" if cn > 0 else "model pick" if cn < 0 else "even/none"),
                              strength_favors=("committee pick" if test.strength_rank[a] < test.strength_rank[b] else "model pick")))
    print(g, s, rows[-1], flush=True)
O = pd.DataFrame(rows); SW = pd.DataFrame(swaps)
O.to_csv(f"{A}/selection_rule_test.csv", index=False); SW.to_csv(f"{A}/selection_swaps.csv", index=False)
print(O.sum(numeric_only=True))
print(SW.h2h_favors.value_counts().to_dict(), SW.common_favors.value_counts().to_dict(), SW.strength_favors.value_counts().to_dict())
pd.set_option("display.width", 250)
print(SW[["gender", "season", "committee_took", "c_region", "c_place", "c_rank", "model_took", "m_region", "m_place", "m_rank", "h2h", "common"]].to_string(index=False))
