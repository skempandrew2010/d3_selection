"""DIII Kolas standings after each weekend of the 2026 countable season.
For each cutoff (Sunday): projected regional order = team strength within region as of that date (regionals not run yet),
projected regional seven = top 7 runners by rating; results count only from meets where a team ran 5 of its projected seven.
Kolas selection: 10 projected regional winners, then 22 at-large picks from the board, each round the board team with the
most wins over teams already in the field; ties: head-to-head among tied, common opponents in the field, regional place.
Writes res26/kolas_weekly.json.
"""
import sys, re, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
S = sys.argv[1]
src = open(f"{S}/scripts/team_adjust.py").read().split("# ---- out-of-sample strengths")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src); sys.argv = argv
def region_name(meet):
    m = re.search(r"Division III (.+?) Region", str(meet)); return m.group(1) if m else None
CUTS = [pd.Timestamp(x) for x in ("2026-09-27", "2026-10-04", "2026-10-11", "2026-10-18", "2026-10-25", "2026-11-01")]
START = pd.Timestamp("2026-09-21")
OUT = {"cuts": [c.strftime("%Y-%m-%d") for c in CUTS], "genders": {}}

for g in ("m", "f"):
    R25 = R[(R.gender == g) & (R.season == 2025) & (R.rtype == "regional")]
    region_of = {t: region_name(m) for t, m in zip(R25.team, R25.meet)}
    Rs = R[(R.gender == g) & (R.season == 2026) & (R.date >= START)]
    teams_all = sorted({t for t in Rs.team.unique() if t in region_of})
    tid = {t: i for i, t in enumerate(teams_all)}
    weeks = []
    for cut in CUTS:
        X = Rs[Rs.date <= cut]
        ab = rating_before(g, 2026, cut + pd.Timedelta(days=1))
        p = X.drop_duplicates("key")[["key", "team"]].assign(r=lambda q: q.key.map(ab)).dropna(subset=["r"])
        p = p[p.team.isin(tid)].sort_values("r")
        cnt = p.groupby("team").size(); eligible = set(cnt[cnt >= 5].index)
        strength = p[p.team.isin(eligible)].groupby("team").r.apply(lambda v: v.head(5).mean())
        srank = strength.rank(method="first")
        seven = p.groupby("team").head(7).groupby("team").key.apply(set).to_dict()
        # results with the five-of-seven rule
        w = {}; meets_week = []
        for rk, x in X.groupby("rk"):
            x = x.sort_values("place"); c = x.team.value_counts(); full = set(c[c >= 5].index)
            ok = {t for t in full if len(set(x[x.team == t].key) & seven.get(t, set())) >= 5 and t in tid}
            if x.date.iloc[0] > cut - pd.Timedelta(days=6): meets_week.append((x.meet.iloc[0], len(ok)))
            if len(ok) < 2: continue
            y = x[x.team.isin(full)].copy(); y["k"] = y.groupby("team").cumcount() + 1; y = y[y.k <= 7]; y["tp"] = np.arange(1, len(y) + 1)
            sc = y[y.k <= 5].groupby("team").tp.sum().sort_values(); tt = [t for t in sc.index if t in ok]
            for i, a in enumerate(tt):
                for b in tt[i + 1:]:
                    w.setdefault(a, {}); w.setdefault(b, {}); w[a][b] = w[a].get(b, 0) + 1; w[b][a] = w[b].get(a, 0) - 1
        sg = lambda a, b: (int(np.sign(w[a][b])) if b in w.get(a, {}) and w[a][b] != 0 else (0 if b in w.get(a, {}) else None))
        regions = {}
        for t in sorted(eligible, key=lambda t: srank[t]): regions.setdefault(region_of[t], []).append(t)
        place = {t: k + 1 for rg, ts in regions.items() for k, t in enumerate(ts)}
        field = [ts[0] for ts in regions.values()]; auto = set(field)
        pools = {rg: ts[1:] for rg, ts in regions.items()}
        picks = []
        def pts(t, fld):
            return sum(1 for o in fld if o != t and sg(t, o) == 1)
        for _ in range(22):
            board = [ts[0] for ts in pools.values() if ts]
            P = {t: pts(t, field) for t in board}; top = max(P.values()); tied = [t for t in board if P[t] == top]
            why = "points"
            if len(tied) > 1:
                hw = {t: sum((sg(t, o) or 0) for o in tied if o != t) for t in tied}; m = max(hw.values())
                if len([t for t in tied if hw[t] == m]) < len(tied): why = "head-to-head"
                tied = [t for t in tied if hw[t] == m]
            if len(tied) > 1:
                def cnet(a, b):
                    n = 0
                    for o in field:
                        if o in (a, b): continue
                        ra, rb = sg(a, o), sg(b, o)
                        if ra is None or rb is None: continue
                        n += ra - rb
                    return n
                cw = {t: sum(cnet(t, o) for o in tied if o != t) for t in tied}; m = max(cw.values())
                if len([t for t in tied if cw[t] == m]) < len(tied): why = "common opponents"
                tied = [t for t in tied if cw[t] == m]
            if len(tied) > 1: why = "regional place"
            best = min(tied, key=lambda t: (place[t], srank[t]))
            picks.append((best, P[best], why, len(board))); field.append(best); pools[region_of[best]].remove(best)
        board = [ts[0] for ts in pools.values() if ts]
        out_list = sorted(board, key=lambda t: (-pts(t, field), place[t]))[:6]
        fset = set(field)
        rows = []
        for t in eligible:
            beat = [tid[o] for o in fset if o != t and sg(t, o) == 1]; lost = [tid[o] for o in fset if o != t and sg(t, o) == -1]
            status = "AQ" if t in auto else ("IN" if t in fset else ("OUT" if t in out_list else ""))
            rows.append([tid[t], region_of[t], place[t], int(srank[t]), len(beat), len(lost), status, beat, lost])
        weeks.append({"cut": cut.strftime("%Y-%m-%d"),
                      "auto": [tid[t] for t in auto],
                      "picks": [[tid[t], pt, why, nb] for t, pt, why, nb in picks],
                      "out": [[tid[t], pts(t, field)] for t in out_list],
                      "rows": rows,
                      "meets": [m for m, n in sorted(set(meets_week), key=lambda z: -z[1]) if n >= 3][:6]})
        print(g, cut.date(), "eligible", len(eligible), "picks", len(picks), "top meets", weeks[-1]["meets"][:3], flush=True)
    OUT["genders"][g] = {"teams": teams_all, "weeks": weeks}
json.dump(OUT, open(f"{S}/res26/kolas_weekly.json", "w"), separators=(",", ":"))
import os; print(os.path.getsize(f"{S}/res26/kolas_weekly.json"))
