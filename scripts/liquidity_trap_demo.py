"""Demonstrates the phantom-liquidity trap on real NSE bhavcopy — run before trusting any
options backtest built on settlement prices.

A delta-neutral iron condor, strikes placed at 1SD/2SD by expected move, held to expiry,
one entry per expiry, over 1,234 trading days:

    no liquidity filter            n=257  net +115.97/trade  t=9.32  "SIGNIFICANT"
    all legs volume > 0            n=257  net   +7.12/trade  t=0.60   not significant
    all legs vol>=100,  OI>=500    n=257  net   +6.60/trade  t=0.61   not significant
    all legs vol>=1000, OI>=5000   n=252  net  +10.05/trade  t=1.02   not significant

94% of the apparent edge was fictional. At 8-30 DTE the 2SD wings sit ~2,063 pts OTM, where
78.6% of trades had ZERO volume on at least one leg and only 21% had all four legs trade.
Bhavcopy still prints a `close` for those strikes — a theoretical settlement price, not a
fill. The "strategy" was buying protection nobody was selling, at prices nobody quoted, and
it produced a t-stat of 9 that would have lost money against a real order book.

Enforced in code by src.nse_bhavcopy.tradeable().
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, os
from src.nse_bhavcopy import CACHE_DIR
fs=sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet"))
df=pd.concat([pd.read_parquet(os.path.join(CACHE_DIR,f)) for f in fs],ignore_index=True)
df=df[df.symbol=="NIFTY"].copy()
df["trade_date"]=pd.to_datetime(df.trade_date); df["expiry"]=pd.to_datetime(df.expiry)
key=["trade_date","expiry","strike","option_type"]
P=df.set_index(key)["close"];P=P[~P.index.duplicated(keep="last")].to_dict()
V=df.set_index(key)["volume"];V=V[~V.index.duplicated(keep="last")].to_dict()
O=df.set_index(key)["oi"];O=O[~O.index.duplicated(keep="last")].to_dict()
avail={}
for k in P: avail.setdefault((k[0],k[1]),set()).add(k[2])
und=df.groupby("trade_date")["underlying"].median().to_dict()
es={}
for e,g in df.groupby("expiry"):
    r=g[g.trade_date==e]
    if len(r): es[e]=float(r["underlying"].median())
days=sorted(df.trade_date.unique()); ebd={d:sorted(g.expiry.unique()) for d,g in df.groupby("trade_date")}
def near(ks,x): return min(ks,key=lambda k:abs(k-x)) if ks else None
def po(S,k,t,s): return s*(max(0.0,k-S) if t=="PE" else max(0.0,S-k))
def run(lo,hi,ss,sw,min_vol,min_oi):
    rows=[];seen=set()
    for d in days:
        sp=und.get(d)
        if not sp or np.isnan(sp): continue
        atm=int(round(sp/50)*50)
        for e in ebd.get(d,[]):
            dte=(pd.Timestamp(e)-pd.Timestamp(d)).days
            if not(lo<=dte<=hi) or e in seen or e not in es: continue
            c=P.get((d,e,float(atm),"CE"));p=P.get((d,e,float(atm),"PE"))
            if not c or not p: continue
            iv=(c+p)/sp/np.sqrt(dte/365.0)*1.25; em=sp*iv*np.sqrt(dte/365.0)
            ks=avail.get((d,e),set())
            L4=[(near(ks,atm-ss*em),"PE",+1),(near(ks,atm-sw*em),"PE",-1),
                (near(ks,atm+ss*em),"CE",+1),(near(ks,atm+sw*em),"CE",-1)]
            if any(k is None for k,_,_ in L4): continue
            px=[P.get((d,e,float(k),t)) for k,t,_ in L4]
            if any(v is None for v in px): continue
            vols=[V.get((d,e,float(k),t),0) or 0 for k,t,_ in L4]
            ois=[O.get((d,e,float(k),t),0) or 0 for k,t,_ in L4]
            if min(vols)<min_vol or min(ois)<min_oi: continue      # TRADEABLE ONLY
            credit=sum(s*v for (_,_,s),v in zip(L4,px))
            width=max(L4[0][0]-L4[1][0], L4[3][0]-L4[2][0]); ml=width-credit
            if credit<=0 or ml<=0: continue
            S=es[e]
            rows.append(credit-sum(po(S,k,t,s) for k,t,s in L4)); seen.add(e)
    return np.array(rows)
print("SAME STRUCTURE, restricted to legs that ACTUALLY TRADED\n")
print(f"{'filter':>28}{'n':>5}{'win%':>7}{'net':>9}{'t':>7}{'detect':>8}  verdict")
print("-"*68)
for lbl,mv,mo in (("none (phantom prices ok)",0,0),("all legs volume>0",1,1),
                  ("all legs vol>=100, OI>=500",100,500),("all legs vol>=1000, OI>=5000",1000,5000)):
    a=run(8,30,1.0,2.0,mv,mo)
    if len(a)<8: print(f"{lbl:>28}{len(a):>5}   -- too few --"); continue
    N=a-1.1; sd=N.std(); t=N.mean()/(sd/np.sqrt(len(N))); det=2.8*sd/np.sqrt(len(N))
    v="SIGNIFICANT" if abs(t)>2 and abs(N.mean())>det else "not sig"
    print(f"{lbl:>28}{len(N):>5}{100*(N>0).mean():>6.0f}%{N.mean():>9.2f}{t:>7.2f}{det:>8.2f}  {v}")
