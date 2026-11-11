"""Team-level adjustments at nationals: depth, team record, program history.

perf_i = strength_i + shift[team] + team_day[team] + own_day_i     (15% team luck, as calibrated)
shift[team] = g_depth * depth_z + g_record * record_z + g_hist * hist_z      (z-scored within each race; missing = 0)
  depth   = 5th-to-7th runner gap in model strength (bigger = thinner)
  record  = team-vs-team strength from countable races before nationals (top-5 average vs other nationals teams)
  history = program's average miss (actual - rating-based predicted place) at its previous 3 nationals
Strengths are out of sample: 2019/2021/2022 from forests trained 2015-2018; 2023-2025 from forests trained 2015-2022.
Gammas are fit to minimize squared team-place error; tested leave-one-season-out against gammas = 0.
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
S = sys.argv[1]
src = open(f"{S}/scripts/team_luck_nats.py").read().split("MODEL = ")[0]
argv, sys.argv = sys.argv, ["x", S]; exec(src); sys.argv = argv
A = f"{S}/all"
CAL = pd.read_csv(f"{A}/team_luck_calibration_nationals.csv").set_index("team_share")
RHO, SD = 0.15, float(CAL.loc[0.15, "total_sd"])
MODEL = "RF: all races + race type"

def nats_rows(g, s):
    x = R[(R.gender == g) & (R.season == s) & (R.rtype == "nationals")]
    return x[x.rk == x.groupby("rk").size().idxmax()]

def rating_before(g, s, D):
    prior = R[(R.gender == g) & (R.season == s) & (R.date < D)].copy(); prior["w"] = wt((D - prior.date).dt.days)
    ab = prior.groupby("key").frel.mean()
    for _ in range(30):
        rc = (prior.frel - prior.key.map(ab)).groupby(prior.rk).mean()
        ab = ((prior.frel - prior.rk.map(rc)) * prior.w).groupby(prior.key).sum() / prior.w.groupby(prior.key).sum()
    return ab

def score_order(perf, teams):
    _, tp, _, T = sim2(np.array([perf]), list(teams), 0.0, 0.0, 1, 0); return dict(zip(T, tp[0]))

# ---- program history: rating-only baseline miss at every nationals 2015-2025 ----
hist_rows = []
for g in ("m", "f"):
    for s in [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]:
        x = nats_rows(g, s); D = x.date.min()
        ab = rating_before(g, s, D); x = x.assign(r=x.key.map(ab)).dropna(subset=["r"])
        full = x.team.value_counts()[lambda v: v >= 5].index
        pred = score_order(x.r.values, x.team.values); act = score_order(x.frel.values, x.team.values)
        for t in full: hist_rows.append(dict(gender=g, season=s, team=t, base_miss=act[t] - pred[t]))
H = pd.DataFrame(hist_rows).sort_values(["gender", "team", "season"])
H["history"] = H.groupby(["gender", "team"]).base_miss.transform(lambda v: v.shift(1).rolling(3, min_periods=1).mean())

# ---- team record before nationals ----
def team_record(g, s, D, nteams):
    f = R[(R.gender == g) & (R.season == s) & (R.date < D) & R.team.isin(nteams)]
    rec = []
    for rk, x in f.groupby("rk"):
        t5 = x.sort_values("frel").groupby("team").frel.apply(lambda v: v.head(5).mean() if len(v) >= 5 else np.nan).dropna()
        ts = list(t5.index)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)): rec.append((ts[i], ts[j], t5[ts[j]] - t5[ts[i]]))
    if not rec: return {}
    teams = sorted({a for a, _, _ in rec} | {b for _, b, _ in rec}); k = {t: i for i, t in enumerate(teams)}
    Am = np.zeros((len(rec) + 1, len(teams))); y = np.zeros(len(rec) + 1)
    for n, (a, b, m) in enumerate(rec): Am[n, k[a]], Am[n, k[b]], y[n] = 1, -1, m
    Am[-1, :] = 1
    sol = np.linalg.lstsq(Am, y, rcond=None)[0]
    return {t: sol[k[t]] for t in teams}          # + = better

# ---- out-of-sample strengths + team features for 12 nationals races ----
RACES = []
for g in ("m", "f"):
    for train, tests in (([2015, 2016, 2017, 2018], [2019, 2021, 2022]), ([2015, 2016, 2017, 2018, 2019, 2021, 2022], [2023, 2024, 2025])):
        d, cols = train_data(MODEL, g, None, train)
        models = fit_bag(d, cols, train[-1])
        for s in tests:
            x = nats_rows(g, s)[["key", "team", "team_slug", "cls", "frel", "place"]]; D = nats_rows(g, s).date.min()
            F2, X, iu, ju = field_pairs_nats(g, s, D, x, "nationals")
            mus = strengths(models, cols, X, len(F2), iu, ju)
            teams = list(F2.team); T = sorted(set(teams)); full = [t for t in T if teams.count(t) >= 5]
            m = mus.mean(0)
            depth = {}
            for t in full:
                v = np.sort(m[np.array(teams) == t])
                depth[t] = (v[6] if len(v) >= 7 else v[-1] + 0.5) - v[4]
            rec = team_record(g, s, D, set(full))
            hs = H[(H.gender == g) & (H.season == s)].set_index("team").history
            act = score_order(F2.frel.values, teams)
            feat = pd.DataFrame({"team": full, "depth": [depth[t] for t in full], "record": [rec.get(t, np.nan) for t in full],
                                 "history": [hs.get(t, np.nan) for t in full], "actual": [act[t] for t in full]})
            for c in ("depth", "record", "history"):
                v = feat[c]; feat[c + "_z"] = ((v - v.mean()) / v.std()).fillna(0)
            RACES.append(dict(g=g, s=s, F2=F2, mus=mus, teams=teams, feat=feat))
            print(g, s, "ready:", len(full), "teams", flush=True)

def run(race, gam, n=300, seed=3, rho=RHO, sd=SD):
    f = race["feat"]; shift = {t: 0.0 for t in set(race["teams"])}
    for t, a, b, c in zip(f.team, f.depth_z, f.record_z, f.history_z):
        shift[t] = gam[0] * a - gam[1] * b + gam[2] * c          # thinner = slower, better record = faster, past over-ranking = slower
    mus = race["mus"] + np.array([shift[t] for t in race["teams"]])[None, :]
    _, tplace, _, T = sim2(mus, race["teams"], sd, rho, n, seed)
    idx = {t: i for i, t in enumerate(T)}
    tp = tplace[:, [idx[t] for t in f.team]]
    return tp, f.actual.values

def objective(gam, races):
    err = []
    for r in races:
        tp, act = run(r, gam, 200, 3)
        err += list((tp.mean(0) - act) ** 2)
    return np.mean(err)

def metrics(races, gam):
    out = []
    for r in races:
        tp, act = run(r, gam, 2000, 26)
        avg = tp.mean(0); win = (tp == 1).mean(0); top4 = (tp <= 4).mean(0)
        out.append(dict(rank_corr=spearmanr(avg, act).correlation, mae=np.mean(np.abs(avg - act)),
                        win_brier=np.mean((win - (act == 1)) ** 2), top4_brier=np.mean((top4 - (act <= 4)) ** 2),
                        champ_win=win[act == 1][0] * 100, top10_hit=len(set(np.argsort(avg)[:10]) & set(np.argsort(act)[:10]))))
    return pd.DataFrame(out).mean()

SIMPLEX = np.array([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]], dtype=float)
import pickle; pickle.dump([{k: v for k, v in r.items()} for r in RACES], open(f"{A}/team_adjust_races.pkl", "wb"))
seasons = sorted({r["s"] for r in RACES})
rows = []
for s in seasons:
    tr = [r for r in RACES if r["s"] != s]; te = [r for r in RACES if r["s"] == s]
    fit = minimize(objective, x0=np.zeros(3), args=(tr,), method="Nelder-Mead", options=dict(maxiter=150, xatol=0.005, fatol=0.02, initial_simplex=SIMPLEX))
    base, adj = metrics(te, np.zeros(3)), metrics(te, fit.x)
    rows.append(dict(season=s, g_depth=fit.x[0], g_record=fit.x[1], g_history=fit.x[2], **{f"base_{k}": v for k, v in base.items()},
                     **{f"adj_{k}": v for k, v in adj.items()}))
    print(rows[-1], flush=True)
CV = pd.DataFrame(rows); CV.to_csv(f"{A}/team_adjust_cv.csv", index=False)
full_fit = minimize(objective, x0=np.zeros(3), args=(RACES,), method="Nelder-Mead", options=dict(maxiter=200, xatol=0.005, fatol=0.02, initial_simplex=SIMPLEX))
pd.set_option("display.width", 220)
print("\nLEAVE-ONE-SEASON-OUT (held-out season, men + women averaged)")
show = CV[["season", "g_depth", "g_record", "g_history"] + [c for c in CV.columns if c.startswith(("base_", "adj_"))]]
print(show.round(3).to_string(index=False))
summary = pd.DataFrame({"current model": CV[[c for c in CV if c.startswith("base_")]].mean().values,
                        "+ depth, record, history": CV[[c for c in CV if c.startswith("adj_")]].mean().values},
                       index=[c[5:] for c in CV if c.startswith("base_")])
print("\nAVERAGE OVER HELD-OUT SEASONS"); print(summary.round(4).to_string())
print(f"\nfit on all 12 races: depth {full_fit.x[0]:.3f}, record {full_fit.x[1]:.3f}, history {full_fit.x[2]:.3f} (strength units per 1 SD; race-day SD is {SD:.2f})")
# univariate check: each adjustment alone, fit on all races (in-sample, for direction only)
for k, lab in enumerate(["depth", "record", "history"]):
    fx = minimize(lambda v: objective(np.eye(3)[k] * v[0], RACES), x0=[0.0], method="Nelder-Mead", options=dict(maxiter=40, initial_simplex=[[0.0], [0.3]]))
    print(f"  {lab} alone: gamma {fx.x[0]:.3f}, place MSE {fx.fun:.2f} vs {objective(np.zeros(3), RACES):.2f} with none")
summary.to_csv(f"{A}/team_adjust_summary.csv")
pd.Series(full_fit.x, index=["depth", "record", "history"]).to_csv(f"{A}/team_adjust_gammas.csv")
