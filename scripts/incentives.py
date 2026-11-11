"""How much would more interregional racing help selection? 2022-2025.
Part A: information gap on the committee's real decisions (each at-large pick vs each region's first team left out).
Part B: counterfactual schedules: add synthetic meets to the real schedule, simulate their results from season strength,
then run Kolas selection and compare with a strength oracle (the field you would pick if you knew every team's true level).
"""
import os, sys, re, pickle, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
S = sys.argv[1]; A = f"{S}/all"
src = open(f"{S}/scripts/team_adjust.py").read().split("# ---- program history")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src); sys.argv = argv
H = pd.read_csv(f"{A}/selection_history.csv"); H = H[H.season.between(2022, 2025)]
W = pickle.load(open(f"{A}/h2h_matrices.pkl", "rb"))
REGS = ["East", "Metro", "Niagara", "Mideast", "Mid-Atlantic", "South", "Great Lakes", "Midwest", "North", "West"]
mdate = R.drop_duplicates(["meet","season"]).set_index(["meet","season"]).date

def sg(w, a, b):
    m = w.get(a, {}).get(b, [])
    return int(np.sign(sum(np.sign(x[1]) for x in m))) if m else None

if os.environ.get("SKIPA"): pass
# ---------------- Part A
rowsA = []; wkrows = []
for (g, s), d in H.groupby(["gender", "season"]):
    w = W[(g, s)]; field = set(d[d.went].team); na = d[~d.auto]; sel = na[na.went]
    RD = R[(R.gender == g) & (R.season == s) & (R.rtype == "regional")].date.max()
    out = [r.sort_values("tplace").loc[lambda q: ~q.went].index[0] for _, r in na.groupby("region") if (~r.went).any()]
    for a in sel.index:
        for b in out:
            ta, tb = d.team[a], d.team[b]
            if d.region[a] == d.region[b]: continue
            direct = [m for m in w.get(ta, {}).get(tb, []) if not m[2]]
            com = [o for o in field - {ta, tb} if sg(w, ta, o) is not None and sg(w, tb, o) is not None]
            anyc = (set(w.get(ta, {})) & set(w.get(tb, {}))) - {ta, tb}
            rowsA.append(dict(g=g, s=s, direct=len(direct) > 0, common_field=len(com) > 0, n_common=len(com), any_common=len(anyc) > 0))
            for m in direct: wkrows.append(dict(g=g, s=s, weeks_out=(RD - mdate.get((m[0], s), RD)).days // 7))
    # who races out of region among each region's top 8
    top = d[d.tplace <= 8]
    for _, r in top.iterrows():
        opp = [o for o in w.get(r.team, {}) if o in set(top.team) and d.set_index("team").region.get(o) != r.region and any(not m[2] for m in w[r.team][o])]
        wkrows.append(dict(g=g, s=s, kind="top8", team=r.team, region=r.region, place=r.tplace, inter=len(opp) > 0))
GA = pd.DataFrame(rowsA); WK = pd.DataFrame(wkrows)
print("PART A boundary pairs (cross-region)", len(GA))
print(GA.groupby("s")[["direct", "common_field", "any_common"]].mean().round(3))
print("overall", GA[["direct", "common_field", "any_common"]].mean().round(3).to_dict(), "none at all", round((~GA.direct & ~GA.common_field).mean(), 3))
t8 = WK[WK.get("kind") == "top8"]
print("top-8 teams with an out-of-region top-8 opponent:", t8.groupby("s").inter.mean().round(3).to_dict())
print(t8.groupby("region").inter.mean().round(2).sort_values().to_dict())
dw = WK[WK.weeks_out.notna()] if "weeks_out" in WK else None
print("direct meetings by weeks before regionals:", dw.weeks_out.value_counts().sort_index().to_dict())
GA.to_csv(f"{A}/inc_partA.csv", index=False); t8.to_csv(f"{A}/inc_top8.csv", index=False)

# ---------------- Part B
HUBS3 = [["East", "Metro", "Niagara", "Mideast"], ["Mid-Atlantic", "South", "Great Lakes"], ["Midwest", "North", "West"]]
def rr_pairs(year):
    """round-robin rotation: 10 regions -> 5 pairs, different each year (circle method)"""
    r = REGS[:]; k = year % 9
    rest = r[1:]; rest = rest[k:] + rest[:k]; r = [r[0]] + rest
    return [[r[i], r[9 - i]] for i in range(5)]
DESIGNS = {
    "status quo": None,
    "rotating region pairs (2 regions per site)": lambda s: rr_pairs(s),
    "3 geographic hubs": lambda s: HUBS3,
    "2 halves (east 5 / west 5), rotating split": lambda s: [REGS[:5], REGS[5:]] if s % 2 else [REGS[0::2], REGS[1::2]],
    "one national preview meet": lambda s: [REGS],
    "bubble meet, 2 sites (east/west)": lambda s: [REGS[:5], REGS[5:]],
    "3 hubs + national preview (2 meetings)": lambda s: HUBS3 + [REGS],
    "national preview x2": lambda s: [REGS, REGS],
    "national preview x3": lambda s: [REGS, REGS, REGS],
}
import os; TEAM_SD = float(os.environ.get("SD", 1.3)); PART = float(os.environ.get("PART", 0.6)); LO, HI = map(int, os.environ.get("POOL", "1,8").split(",")); rng = np.random.default_rng(11)

def kolas(d, w, sgf):
    pools = {rg: list(r.sort_values("tplace").team) for rg, r in d[~d.auto].groupby("region")}
    place = dict(zip(d.team, d.tplace)); region = dict(zip(d.team, d.region)); srk = dict(zip(d.team, d.strength_rank))
    field = list(d[d.auto].team); n = int((d.went & ~d.auto).sum()); picks = []; how = []
    for _ in range(n):
        board = [p[0] for p in pools.values() if p]
        P = {t: sum(1 for o in field if sgf(t, o) == 1) for t in board}; top = max(P.values()); tied = [t for t in board if P[t] == top]; why = "points"
        if len(tied) > 1:
            hw = {t: sum((sgf(t, o) or 0) for o in tied if o != t) for t in tied}; m = max(hw.values())
            if sum(hw[t] == m for t in tied) < len(tied): why = "h2h"
            tied = [t for t in tied if hw[t] == m]
        if len(tied) > 1:
            def cn(a, b):
                v = 0
                for o in field:
                    if o in (a, b): continue
                    ra, rb = sgf(a, o), sgf(b, o)
                    if ra is None or rb is None: continue
                    v += ra - rb
                return v
            cw = {t: sum(cn(t, o) for o in tied if o != t) for t in tied}; m = max(cw.values())
            if sum(cw[t] == m for t in tied) < len(tied): why = "common"
            tied = [t for t in tied if cw[t] == m]
        if len(tied) > 1: why = "regional place"
        best = min(tied, key=lambda t: (place[t], srk[t])); picks.append(best); how.append(why); field.append(best); pools[region[best]].remove(best)
    return picks, how

def oracle(d):
    pools = {rg: list(r.sort_values("tplace").team) for rg, r in d[~d.auto].groupby("region")}
    st = dict(zip(d.team, d.strength)); region = dict(zip(d.team, d.region)); n = int((d.went & ~d.auto).sum()); picks = []
    for _ in range(n):
        board = [p[0] for p in pools.values() if p]; best = min(board, key=lambda t: st[t]); picks.append(best); pools[region[best]].remove(best)
    return picks

rowsB = []
for (g, s), d in H.groupby(["gender", "season"]):
    w = W[(g, s)]; orc = set(oracle(d)); n = int((d.went & ~d.auto).sum()); st = dict(zip(d.team, d.strength))
    comm = set(d[d.went & ~d.auto].team)
    for name, fn in DESIGNS.items():
        for rep in range(30 if fn else 1):
            extra = {}
            if fn:
                for site in fn(s):
                    pool = d[d.region.isin(site) & d.tplace.between(LO, HI)]
                    for P_ in (PART,):
                        go = [t for t in pool.team if rng.random() < P_]
                    perf = {t: st[t] + rng.normal(0, TEAM_SD) for t in go}
                    for a, b in itertools.combinations(go, 2):
                        v = 1 if perf[a] < perf[b] else -1
                        extra[(a, b)] = extra.get((a, b), 0) + v; extra[(b, a)] = extra.get((b, a), 0) - v
            def sgf(a, b, w=w, extra=extra):
                m = w.get(a, {}).get(b, []); v = sum(np.sign(x[1]) for x in m) + extra.get((a, b), 0)
                if not m and (a, b) not in extra: return None
                return int(np.sign(v))
            picks, how = kolas(d, w, sgf)
            rowsB.append(dict(g=g, s=s, design=name, rep=rep, n=n, oracle_match=len(set(picks) & orc), committee_match=len(set(picks) & comm),
                              blind=sum(h == "regional place" for h in how), tiebreak=sum(h != "points" for h in how)))
    print(g, s, flush=True)
B = pd.DataFrame(rowsB); B.to_csv(f"{A}/inc_partB.csv", index=False)
agg = B.groupby(["design", "g", "s"]).mean(numeric_only=True).groupby("design")[["n", "oracle_match", "blind", "tiebreak"]].sum()
agg["oracle_%"] = (agg.oracle_match / agg.n * 100).round(1); pd.set_option("display.width",200); agg["committee_match"]=B.groupby(["design","g","s"]).mean(numeric_only=True).groupby("design").committee_match.sum(); print(agg.sort_values("oracle_%").round(1).to_string())
# committee vs oracle for reference
cm = sum(len(set(d[d.went & ~d.auto].team) & set(oracle(d))) for _, d in H.groupby(["gender", "season"])); print("committee matches oracle:", cm, "of", int((H.went & ~H.auto).sum()))
