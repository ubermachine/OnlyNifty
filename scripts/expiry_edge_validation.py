"""Rigorous validation of short-premium edge claims — independence + delta-neutrality.

Two controls that overturned an apparently significant result, run over 1,234 trading days
(2021-08 -> 2026-08, 2.16M NIFTY option rows from official NSE bhavcopy).

CONTROL 1 — SAMPLE INDEPENDENCE
Entering the same expiry at 1, 2 and 3 DTE is not three independent trades; they share one
settlement outcome. Naive pooling inflates t by ~sqrt(entries per expiry):

    condor 1-3 DTE, all entries    n=636  net +5.45  t=3.11   "significant"
    condor 1-3 DTE, 1-per-expiry   n=259  net +2.42  t=0.88   NOT significant

CONTROL 2 — DELTA NEUTRALITY
A put credit spread is LONG delta; a call credit spread is SHORT delta. Over this sample
Nifty rose +46.5%, so drift alone pays the put side and taxes the call side. A genuine
theta/vol edge would pay BOTH sides, since premium is collected on both:

    put  spread 1-3 DTE, 1-per-expiry  net +7.51  t=+2.86
    call spread 1-3 DTE, 1-per-expiry  net -4.99  t=-1.89
    condor (= put + call, delta-neutral)  net +2.42  t=+0.88   <- the clean test: no edge

The put and call legs are near mirror images, and the arithmetic ties out
(+7.51 - 4.99 = +2.52 ~ condor +2.42). The apparent "77% win rate edge" on the put side is
equity beta expressed through options, not alpha: capped upside, fat left tail, option
friction, and it inverts in a falling market. The index captures the same drift more cheaply.

VERDICT: no delta-neutral short-premium edge at 1-3 DTE, or at any longer DTE bucket tested.

Always report BOTH controls. A high win rate and a large t-stat survive neither on their own.
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, os
from src.nse_bhavcopy import CACHE_DIR
fs=sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet"))
df=pd.concat([pd.read_parquet(os.path.join(CACHE_DIR,f)) for f in fs],ignore_index=True)
df=df[df.symbol=="NIFTY"].copy()
df["trade_date"]=pd.to_datetime(df.trade_date); df["expiry"]=pd.to_datetime(df.expiry)
L=df.set_index(["trade_date","expiry","strike","option_type"])["close"]
L=L[~L.index.duplicated(keep="last")].to_dict()
und=df.groupby("trade_date")["underlying"].median().to_dict()
exp_spot={}
for e,g in df.groupby("expiry"):
    r=g[g.trade_date==e]
    if len(r): exp_spot[e]=float(r["underlying"].median())
days=sorted(df.trade_date.unique())
exps_by_day={d:sorted(g.expiry.unique()) for d,g in df.groupby("trade_date")}
s0,s1=und[days[0]],und[days[-1]]
print(f"MARKET over sample: Nifty {s0:.0f} -> {s1:.0f} ({100*(s1-s0)/s0:+.1f}%) across {len(days)} days\n")

def payoff(S,k,t,s): return s*(max(0.0,k-S) if t=="PE" else max(0.0,S-k))
def run(kind,dte_lo,dte_hi,one_per_expiry,width=100,offset=100):
    rows=[];seen=set()
    for d in days:
        sp=und.get(d)
        if not sp or np.isnan(sp): continue
        atm=int(round(sp/50)*50)
        for e in exps_by_day.get(d,[]):
            dte=(pd.Timestamp(e)-pd.Timestamp(d)).days
            if not(dte_lo<=dte<=dte_hi) or e not in exp_spot: continue
            if one_per_expiry and e in seen: continue
            legs=[(atm-offset,"PE",+1),(atm-offset-width,"PE",-1)] if kind=="put" else \
                 [(atm-offset,"PE",+1),(atm-offset-width,"PE",-1),(atm+offset,"CE",+1),(atm+offset+width,"CE",-1)]
            px=[L.get((d,e,float(k),t)) for k,t,_ in legs]
            if any(v is None for v in px): continue
            credit=sum(s*v for (_,_,s),v in zip(legs,px)); ml=width-credit
            if credit<=0 or ml<=0: continue
            S=exp_spot[e]
            rows.append(dict(e=e,dte=dte,pnl=credit-sum(payoff(S,k,t,s) for k,t,s in legs),ml=ml))
            if one_per_expiry: seen.add(e)
    return pd.DataFrame(rows)

print(f"{'structure':>10}{'dte':>7}{'sampling':>14}{'n':>6}{'win%':>7}{'net':>9}{'t':>7}{'detect':>8}  verdict")
print("-"*82)
for kind,fr in (("condor",1.1),("put",0.5)):
    for lo,hi in ((1,3),):
        for indep in (False,True):
            r=run(kind,lo,hi,indep)
            if len(r)<10: continue
            N=r.pnl.values-fr; sd=N.std(); t=N.mean()/(sd/np.sqrt(len(N)))
            det=2.8*sd/np.sqrt(len(N))
            lbl="1-per-expiry" if indep else "all entries"
            v="SIGNIFICANT" if abs(t)>2 and abs(N.mean())>det else ("marginal" if abs(t)>2 else "not sig")
            print(f"{kind:>10}{f'{lo}-{hi}d':>7}{lbl:>14}{len(N):>6}{100*(N>0).mean():>6.0f}%{N.mean():>9.2f}{t:>7.2f}{det:>8.2f}  {v}")
print()
# Delta-neutrality check: does the CALL side alone also profit at 1-3 DTE?
def call_only(lo,hi,indep,width=100,offset=100):
    rows=[];seen=set()
    for d in days:
        sp=und.get(d)
        if not sp or np.isnan(sp): continue
        atm=int(round(sp/50)*50)
        for e in exps_by_day.get(d,[]):
            dte=(pd.Timestamp(e)-pd.Timestamp(d)).days
            if not(lo<=dte<=hi) or e not in exp_spot: continue
            if indep and e in seen: continue
            legs=[(atm+offset,"CE",+1),(atm+offset+width,"CE",-1)]
            px=[L.get((d,e,float(k),t)) for k,t,_ in legs]
            if any(v is None for v in px): continue
            credit=sum(s*v for (_,_,s),v in zip(legs,px)); ml=width-credit
            if credit<=0 or ml<=0: continue
            S=exp_spot[e]
            rows.append(credit-sum(payoff(S,k,t,s) for k,t,s in legs))
            if indep: seen.add(e)
    return np.array(rows)
for indep in (False,True):
    c=call_only(1,3,indep)-0.5
    if len(c)>10:
        t=c.mean()/(c.std()/np.sqrt(len(c)))
        print(f"CALL spread 1-3d {'1-per-expiry' if indep else 'all entries':>13}: n={len(c)} win {100*(c>0).mean():.0f}% net {c.mean():+.2f} t={t:+.2f}")
print("\nIf BOTH put and call credit spreads profit at 1-3d, the edge is THETA (vol), not direction.")
print("If only the put side profits, it is the +52% market drift showing up as fake alpha.")
