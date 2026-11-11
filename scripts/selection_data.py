"""Build the NCAA DIII cross country team-selection history (2015-2025) from TFRRS results.
For every regional: team places and scores. For every team: whether it went to nationals (5+ finishers there),
whether it was an automatic qualifier (regional winner; top 2 per region before 2021), and the resume features the
selection committee is told to use: regional place and point gaps, head-to-head and common-opponent results from
late-season meets (both teams with 5+ D3 finishers), plus a team strength rating.
Writes all/selection_history.csv.
"""
import sys, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
S = sys.argv[1]; A = f"{S}/all"
src = open(f"{S}/scripts/team_adjust.py").read().split("# ---- out-of-sample strengths")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src); sys.argv = argv          # R, nats_rows, rating_before, team_record
SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

def race_team_scores(x):
    """team score in one race: D3 finishers only, teams with 5+, top 5 score, 6-7 displace."""
    x = x.sort_values("place"); cnt = x.team.value_counts(); full = set(cnt[cnt >= 5].index)
    x = x[x.team.isin(full)].copy(); x["k"] = x.groupby("team").cumcount() + 1; x = x[x.k <= 7]
    x["tp"] = np.arange(1, len(x) + 1)
    sc = x[x.k <= 5].groupby("team").tp.sum(); six = x[x.k == 6].set_index("team").tp.reindex(sc.index).fillna(1e9)
    t = pd.DataFrame({"score": sc, "six": six}).sort_values(["score", "six"]); t["tplace"] = np.arange(1, len(t) + 1)
    return t

def region_name(meet):
    m = re.search(r"Division III (.+?) Region", str(meet)); return m.group(1) if m else str(meet)[:30]

rows = []
for g in ("m", "f"):
    for s in SEASONS:
        Rs = R[(R.gender == g) & (R.season == s)]
        nat = nats_rows(g, s); nat_teams = set(nat.team.value_counts()[lambda v: v >= 5].index); ND = nat.date.min()
        reg = Rs[Rs.rtype == "regional"]
        if reg.empty: continue
        RD = reg.date.max()
        n_auto = 1 if s >= 2021 else 2
        # team ratings before nationals selection (after regionals)
        ab = rating_before(g, s, RD + pd.Timedelta(days=1))
        # late-season team results (all countable races up to and including regionals)
        races = {}
        for rk, x in Rs[Rs.date <= RD].groupby("rk"):
            t = race_team_scores(x)
            if len(t) >= 2: races[rk] = (x.date.iloc[0], t)
        cand = []
        for mid, x in reg.groupby("meet_id"):
            x = x[x.rk == x.groupby("rk").size().idxmax()]
            t = race_team_scores(x).reset_index().rename(columns={"index": "team"})
            t["region"] = region_name(x.meet.iloc[0]); t["n_teams"] = len(t)
            t["gap_ahead"] = t.score - t.score.shift(1); t["gap_behind"] = t.score.shift(-1) - t.score
            t["auto"] = t.tplace <= n_auto
            cand.append(t)
        C = pd.concat(cand, ignore_index=True)
        C["went"] = C.team.isin(nat_teams)
        # team strength: mean of best 5 runner ratings on the team (lower = faster)
        tr = Rs[Rs.date <= RD].assign(r=lambda d: d.key.map(ab)).dropna(subset=["r"]).drop_duplicates("key")
        top5 = tr.sort_values("r").groupby("team").r.apply(lambda v: v.head(5).mean() if len(v) >= 5 else np.nan)
        C["strength"] = C.team.map(top5)
        C["strength_rank"] = C.strength.rank()
        # head to head among regional teams (late-season races)
        teams = set(C.team); auto = set(C[C.auto].team)
        w = {t: {} for t in teams}
        for rk, (d, t) in races.items():
            tt = [x for x in t.index if x in teams]
            for i, a in enumerate(tt):
                for b in tt[i + 1:]:
                    w[a][b] = w[a].get(b, 0) + 1; w[b][a] = w[b].get(a, 0) - 1        # a ahead of b
        def rec(team, pool):
            r = [np.sign(v) for o, v in w[team].items() if o in pool and o != team]
            return sum(x > 0 for x in r), sum(x < 0 for x in r)
        C["h2h_w_auto"], C["h2h_l_auto"] = zip(*[rec(t, auto) for t in C.team])
        pool_top = set(C[C.tplace <= 6].team)                                      # the realistic at-large pool
        C["h2h_w_pool"], C["h2h_l_pool"] = zip(*[rec(t, pool_top) for t in C.team])
        for kk in (3, 4, 5, 6, 7, 8, 10, 99):
            pk = set(C[C.tplace <= kk].team)
            C[f"h2h_w_p{kk}"], C[f"h2h_l_p{kk}"] = zip(*[rec(t, pk) for t in C.team])
        C["h2h_w_nat"], C["h2h_l_nat"] = zip(*[rec(t, nat_teams) for t in C.team])   # vs eventual field (reference only)
        C["gender"], C["season"] = g, s
        rows.append(C)
        sel = C[C.went & ~C.auto]
        viol = 0
        for rg, d in C.groupby("region"):
            d = d.sort_values("tplace"); went = d.went.values
            last = np.where(went)[0].max() if went.any() else -1
            viol += int((~went[:last + 1]).sum())
        print(g, s, "regions", C.region.nunique(), "auto", int(C.auto.sum()), "at-large", len(sel), "teams at nats", len(nat_teams),
              "| skipped-over teams (selected below a team that wasn't):", viol, flush=True)
H = pd.concat(rows, ignore_index=True)
H.to_csv(f"{A}/selection_history.csv", index=False)
print(H.groupby(["season", "gender"]).agg(at_large=("went", lambda v: int(v.sum()))).T.to_string())
