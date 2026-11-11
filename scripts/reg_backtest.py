import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
S=sys.argv[1]
src=open(f"{S}/scripts/team_adjust.py").read().split("# ---- program history")[0]
argv,sys.argv=sys.argv,["x",S]; exec(src); sys.argv=argv
TSCALE={"m":1.67,"f":1.85}; TSD={"m":0.72,"f":0.74}; rng=np.random.default_rng(1); rows=[]
for g in "mf":
  for s in (2022,2023,2024,2025):
    Rs=R[(R.gender==g)&(R.season==s)]; X=Rs[Rs.rtype=="regional"]; D=X.date.min()
    for lead in (0,14,28):   # ratings as of regionals, 2 weeks before, 4 weeks before
      cut=D-pd.Timedelta(days=lead); ab=rating_before(g,s,cut); prior=Rs[Rs.date<cut]; nr=prior.groupby("key").size()
      sev=prior.drop_duplicates("key")[["key","team"]].assign(r=lambda q:q.key.map(ab)).dropna().sort_values("r")
      for rk,x in X.groupby("rk"):
        c=x.team.value_counts(); full=list(c[c>=5].index)
        y=x[x.team.isin(full)].sort_values("place").copy(); y["k"]=y.groupby("team").cumcount()+1; y=y[y.k<=7]; y["tp"]=np.arange(1,len(y)+1)
        act=y[y.k<=5].groupby("team").tp.sum().sort_values(); 
        tms=[t for t in act.index if (sev.team==t).sum()>=5]
        if len(tms)<3: continue
        rr=sev[sev.team.isin(tms)].groupby("team").head(7); r0=rr.r.values; ti=rr.team.map({t:i for i,t in enumerate(tms)}).values; nT=len(tms); N=1000
        unc=1.2/np.sqrt(rr.key.map(nr).fillna(1).values)
        perf=r0[None,:]+rng.normal(0,1,(N,len(r0)))*unc+rng.normal(0,TSD[g],(N,nT))[:,ti]+rng.standard_t(4,(N,len(r0)))*TSCALE[g]
        o=np.argsort(perf,1); tseq=ti[o]
        oh=np.zeros((N,len(r0),nT),np.int16); oh[np.arange(N)[:,None],np.arange(len(r0))[None,:],tseq]=1
        k=np.take_along_axis(oh.cumsum(1),tseq[:,:,None],2)[:,:,0]; tp=np.cumsum(k<=7,1)
        sc=np.zeros((N,nT)); m=k<=5; np.add.at(sc,(np.repeat(np.arange(N),len(r0))[m.ravel()],tseq[m]),tp[m])
        plc=np.argsort(np.argsort(sc,1),1)+1
        aplace={t:i+1 for i,t in enumerate([t for t in act.index if t in tms])}
        for i,t in enumerate(tms):
          rows.append(dict(g=g,s=s,lead=lead,p1=(plc[:,i]==1).mean(),p2=(plc[:,i]<=2).mean(),w1=aplace[t]==1,w2=aplace[t]<=2))
O=pd.DataFrame(rows)
for lead,d in O.groupby("lead"):
  d=d.assign(b=pd.cut(d.p1,[-.01,.05,.2,.4,.6,.8,1]))
  print("lead",lead,"days  brier win",round(((d.p1-d.w1)**2).mean(),4),"top2",round(((d.p2-d.w2)**2).mean(),4))
  print(d.groupby("b").agg(pred=("p1","mean"),actual=("w1","mean"),n=("w1","size")).round(3).to_string())
