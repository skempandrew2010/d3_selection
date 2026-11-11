"""Race-by-race pair dataset for D3 runners in 5 conferences (NESCAC, UAA, MIAC, WIAC, American Rivers), men and women,
countable season only (Mon after 3rd Sat of Sept -> nationals), 2015-2019 & 2021-2025 (+2026 so far).

For every countable race with 2+ runners from those conferences, each runner's inputs come only from D3 runners'
results in EARLIER countable races that season. Every pair of conference runners in the race becomes one row:
A-minus-B inputs and whether A finished ahead.
"""
import re, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
S = sys.argv[1]; A = f"{S}/all"
CONF = {
 "NESCAC": {2015: 8105, 2016: 10724, 2017: 12764, 2018: 14154, 2019: 16670, 2021: 18919, 2022: 20620, 2023: 22832, 2024: 25200, 2025: 26599},
 "UAA": {2015: 9196, 2016: 11200, 2017: 13186, 2018: 14887, 2019: 16684, 2021: 18826, 2022: 21049, 2023: 22946, 2024: 24493, 2025: 27251},
 "MIAC": {2015: 9282, 2016: 11198, 2017: 13359, 2018: 14944, 2019: 16678, 2021: 19226, 2022: 20525, 2023: 23225, 2024: 25267, 2025: 26811},
 "WIAC": {2015: 9285, 2016: 10570, 2017: 13355, 2018: 14947, 2019: 16680, 2021: 19236, 2022: 20647, 2023: 23227, 2024: 25266, 2025: 27252},
 "American Rivers": {2015: 9268, 2016: 11194, 2017: 13353, 2018: 14943, 2019: 16675, 2021: 18728, 2022: 20523, 2023: 22316, 2024: 24926, 2025: 26377}}
CONF_MEET = {m: c for c, d in CONF.items() for m in d.values()}

R = pd.read_csv(f"{A}/results.csv", dtype={"athlete_id": str})
R["date"] = pd.to_datetime(R.date)
R = R[R.seconds.between(600, 3600)]
# results without an athlete link: borrow the athlete id seen for the same name + team + gender that season
_ids = R[R.athlete_id.fillna("") != ""].drop_duplicates(["season", "gender", "name", "team"]).set_index(["season", "gender", "name", "team"]).athlete_id
_miss = R.athlete_id.fillna("") == ""
R.loc[_miss, "athlete_id"] = [_ids.get((s, g, n, t), "") for s, g, n, t in zip(R.season[_miss], R.gender[_miss], R.name[_miss], R.team[_miss])]
R["key"] = R.athlete_id.fillna("").where(R.athlete_id.fillna("") != "", R.name + "|" + R.team)
R["rk"] = R.meet_id.astype(str) + "|" + R.gender + "|" + R.race
R = R.drop_duplicates(["rk", "key"])
ncaa = R.meet.str.match(r"^NCAA Division III", na=False)
R["rtype"] = np.where(ncaa & R.meet.str.contains("Region", na=False), "regional",
             np.where(ncaa & R.meet.str.contains("Championship", na=False), "nationals",
             np.where(R.meet_id.isin(CONF_MEET), "conference", "invitational")))
R.loc[R.rtype == "conference", "rtype"] = R.meet_id.map(CONF_MEET)
# D3 teams = teams at a D3 regional or nationals that season (per gender); conference teams = teams at that conference meet
D3 = set(zip(R[ncaa].season, R[ncaa].team_slug))
D3 |= {(2026, t) for (s, t) in D3 if s == 2025}          # 2026 regionals not run yet: carry 2025 D3 teams
R = R[[(s, t) in D3 for s, t in zip(R.season, R.team_slug)]].copy()
cm = R[R.meet_id.isin(CONF_MEET)]
TEAMCONF = {(s, t): CONF_MEET[m] for s, t, m in zip(cm.season, cm.team_slug, cm.meet_id)}
# 2026 has no conference meets yet: carry 2025 membership
for (s, t), c in list(TEAMCONF.items()):
    if s == 2025: TEAMCONF[(2026, t)] = c
R["conf"] = [TEAMCONF.get((s, t)) for s, t in zip(R.season, R.team_slug)]
R = R[R.groupby("rk").key.transform("size") >= 5]
import os as _os
if _os.environ.get("LONELY") == "1":   # drop races with no real D3 field: <15 D3 finishers, <3 D3 teams, or one team >50% of them
    _g = R.groupby("rk"); _n = _g.key.transform("size"); _t = _g.team.transform("nunique")
    _top = R.groupby(["rk", "team"]).key.transform("size") .groupby(R.rk).transform("max") / _n
    R = R[(_n >= 15) & (_t >= 3) & (_top <= 0.5)].copy()
R["frel"] = R.seconds / R.groupby("rk").seconds.transform("median") * 100       # vs the D3 runners in that race

L5 = pd.read_csv(f"{S}/out/d3_5000.csv", dtype={"athlete_id": str})
L5 = L5[~L5.altitude.astype(bool) & (L5.place <= 200)]
L5["r5"] = L5.seconds / L5.groupby(["gender", "season", "year"]).seconds.transform(lambda v: v.nsmallest(100).median()) * 100
T5 = L5.groupby(["gender", "year", "athlete_id"]).r5.min()
# last year's result at the same kind of championship (conference: same conference meet; regional: any regional; nationals)
champ = R[R.rtype != "invitational"]
PREV = {(g, s, rt, k): v for g, s, rt, k, v in zip(champ.gender, champ.season,
        np.where(champ.rtype.isin(list(CONF)), "conference", champ.rtype), champ.key, champ.frel)}

def wt(days): return np.where(days <= 14, 3.0, np.where(days <= 35, 1.5, 0.75))
rng = np.random.default_rng(0)
FEAT = ["rating", "last", "best", "races", "t5", "has_5k", "cls", "prev", "has_prev", "h2h", "h2h_n", "co", "co_n", "rating_mean"]
out = []
for (g, s), Rs in R.groupby(["gender", "season"]):
    tgt = Rs[Rs.conf.notna()]
    races = tgt.groupby("rk").filter(lambda x: len(x) >= 2)
    for D in sorted(races.date.unique()):
        prior = Rs[Rs.date < D].copy()
        if prior.empty: continue
        prior["w"] = wt((D - prior.date).dt.days)
        ab = prior.groupby("key").frel.mean()
        for _ in range(30):
            rc = (prior.frel - prior.key.map(ab)).groupby(prior.rk).mean()
            ab = ((prior.frel - prior.rk.map(rc)) * prior.w).groupby(prior.key).sum() / prior.w.groupby(prior.key).sum()
        prior["adj"] = prior.frel - prior.rk.map(rc)
        today = races[races.date == D]
        people = sorted(set(today.key) & set(prior.key))
        if len(people) < 2: continue
        idx = {k: i for i, k in enumerate(people)}; n = len(people)
        mine = prior[prior.key.isin(idx)].sort_values("date")
        P = pd.DataFrame({"key": people})
        P["rating"] = P.key.map(ab); P["last"] = P.key.map(mine.groupby("key").adj.last())
        P["best"] = P.key.map(mine.groupby("key").adj.min()); P["races"] = P.key.map(mine.groupby("key").size())
        P["t5"] = [T5.get((g, s, k), np.nan) for k in P.key]
        cls = today.drop_duplicates("key").set_index("key").cls.astype(str).str.extract(r"-(\d)")[0].astype(float)
        P["cls"] = P.key.map(cls)
        # head to head + one-degree common opponents from earlier races
        Hm, Hw, Hn = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        sub = prior[prior.rk.isin(mine.rk)]
        opp = {k: i for i, k in enumerate(sub.key.unique())}
        Sm, Wt = np.zeros((n, len(opp))), np.zeros((n, len(opp)))
        for rk, x in sub.groupby("rk", sort=False):
            me = x[x.key.isin(idx)]; ii = me.key.map(idx).values; w = x.w.iloc[0]
            mg = np.clip(me.frel.values[None, :] - me.frel.values[:, None], -10, 10)
            Hm[np.ix_(ii, ii)] += w * mg; Hw[np.ix_(ii, ii)] += w; Hn[np.ix_(ii, ii)] += 1
            cc = x.key.map(opp).values
            Sm[np.ix_(ii, cc)] += w * np.clip(x.frel.values[None, :] - me.frel.values[:, None], -15, 15); Wt[np.ix_(ii, cc)] += w
        np.fill_diagonal(Hw, 0); np.fill_diagonal(Hn, 0)
        H2H = np.divide(Hm, Hw, out=np.zeros_like(Hm), where=Hw > 0)
        E1 = np.divide(Sm, Wt, out=np.zeros_like(Sm), where=Wt > 0)
        for k, i in idx.items():
            if k in opp: Wt[i, opp[k]] = 0
        num, den, cnt = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)); has = Wt > 0
        for c in np.where(has.sum(0) >= 2)[0]:
            a = np.where(has[:, c])[0]; wa = Wt[a, c]
            hh = np.outer(wa, wa) / (wa[:, None] + wa[None, :])
            num[np.ix_(a, a)] += hh * (E1[a, c][:, None] - E1[a, c][None, :]); den[np.ix_(a, a)] += hh; cnt[np.ix_(a, a)] += 1
        np.fill_diagonal(den, 0); np.fill_diagonal(cnt, 0)
        CO = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        Pi = P.set_index("key")
        for rk, x in today.groupby("rk"):
            x = x[x.key.isin(idx)].drop_duplicates("key")
            if len(x) < 2: continue
            rt = x.rtype.iloc[0]; rtk = "conference" if rt in CONF else rt
            prevv = np.array([PREV.get((g, s - 1 if s != 2021 else 2019, rtk, k), np.nan) for k in x.key]) if rt != "invitational" else np.full(len(x), np.nan)
            ii = x.key.map(idx).values; m = len(x)
            iu, ju = np.triu_indices(m, 1)
            if rt == "invitational" and len(iu) > 3000:
                pick = rng.choice(len(iu), 3000, replace=False); iu, ju = iu[pick], ju[pick]
            a, b = ii[iu], ii[ju]
            v = {c: Pi[c].values.astype(float) for c in ["rating", "last", "best", "races", "t5", "cls"]}
            def d(c, lim, base=None):
                arr = v[c] if base is None else base
                ok = ~np.isnan(arr[a]) & ~np.isnan(arr[b])
                return np.clip(np.nan_to_num(arr[a] - arr[b]) * ok, -lim, lim)
            pv = np.full(n, np.nan); pv[ii] = prevv
            row = pd.DataFrame({
                "rating": d("rating", 10), "last": d("last", 10), "best": d("best", 10), "races": v["races"][a] - v["races"][b],
                "t5": d("t5", 8), "has_5k": (~np.isnan(v["t5"][a])).astype(float) - (~np.isnan(v["t5"][b])),
                "cls": d("cls", 4), "prev": d(None, 15, pv), "has_prev": (~np.isnan(pv[a])).astype(float) - (~np.isnan(pv[b])),
                "h2h": H2H[a, b], "h2h_n": Hn[a, b], "co": np.clip(CO[a, b], -10, 10), "co_n": np.log1p(cnt[a, b]),
                "rating_mean": (v["rating"][a] + v["rating"][b]) / 2})
            fr = x.frel.values
            row["y"] = (fr[iu] < fr[ju]).astype(np.int8)
            row["gender"], row["season"], row["date"], row["race"], row["rtype"] = g, s, D, rk, rt
            row["conf_a"], row["conf_b"] = x.conf.values[iu], x.conf.values[ju]
            out.append(row)
    print(g, s, "races done", flush=True)
PAIRS = pd.concat(out, ignore_index=True)
PAIRS.to_pickle(f"{A}/pairs.pkl")
print("pairs:", len(PAIRS)); print(PAIRS.groupby(["gender", "rtype"]).agg(pairs=("y", "size"), races=("race", "nunique")).to_string())
