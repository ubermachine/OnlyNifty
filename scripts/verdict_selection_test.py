"""Does the desk verdict pick better-than-random moments? Tested with REAL positioning.

This is the question that matters for alpha, and it is not the same as "does the confluence
score rank trades". A verdict can be useful purely by SELECTING good moments even if its
score carries no ordering information.

Method: for every bar the verdict acts on, measure the forward 12-bar move in the signal's
own direction. Compare against 400 random-bar samples drawn with the SAME directional mix,
so the verdict's long/short bias is controlled for and only its TIMING is judged. Report
overlapping and non-overlapping (>=12 bars apart) separately, since overlapping forward
windows inflate significance.

RESULTS — synthetic option chain (confluence 27-57):
    all signals      n=303  hit 47.2%  mean -9.00 pts  t=-2.19
    non-overlapping  n=144  hit 50.7%  mean -9.78 pts  t=-1.69   1.5th pct vs random

RESULTS — REAL bhavcopy positioning (src/bhavcopy_context.py):
    all signals      n=128  hit 37.5%  mean -20.71 pts  t=-4.44  SIGNIFICANT
    non-overlapping  n= 67  hit 43.3%  mean -20.69 pts  t=-3.23  SIGNIFICANT
    random baseline  +0.49 pts (sd 7.40)  ->  observed sits at the 0.8th percentile

With real positioning the verdict's entry timing is SIGNIFICANTLY WORSE THAN RANDOM, after
controlling for its own directional bias. A related probe found no conditional structure:
corr(prior move already made, forward return) = -0.046 (t=-0.54), and the quartiles are
non-monotonic — so it is not "chasing exhausted moves", it simply selects poorly.

LIMITS, stated plainly: one 60-day window (intraday history is capped by the 5m feed),
n=67 independent observations, and OI-derived positioning is end-of-day. A negative result
is NOT a licence to fade the signal — that is a separate hypothesis needing out-of-sample
validation on data this test has already consumed.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os
from src.nse_bhavcopy import CACHE_DIR
from src.bhavcopy_context import build_context_from_bhavcopy
from src.data_engine import DataEngine
from src.edge_harness import EdgeTable
from src.strategy_rules import StrategyEngine, SignalType

df=DataEngine().fetch_yfinance_nifty(interval="5m",period="60d",max_cache_age_seconds=0)
close=df["close"].values; H=12
dates=sorted(set(df.index.strftime("%Y-%m-%d")))
ctx_by_day={}
for ds in dates:
    p=os.path.join(CACHE_DIR,f"{ds}.parquet")
    if not os.path.exists(p): continue
    day=pd.read_parquet(p); day=day[day.symbol=="NIFTY"]
    if day.empty: continue
    sp=float(day.underlying.median())
    c=build_context_from_bhavcopy(day,ds,sp)
    if c: ctx_by_day[ds]=c
print(f"REAL bhavcopy positioning context built for {len(ctx_by_day)}/{len(dates)} session days")
if ctx_by_day:
    k=sorted(ctx_by_day)[len(ctx_by_day)//2]; g=ctx_by_day[k]["gex_chart"]
    print(f"  sample {k}: put_wall {g['put_wall_strike']:.0f} | call_wall {g['call_wall_strike']:.0f} "
          f"| max_pain {g['zero_gex_strike']:.0f} | regime {g['net_dealer_regime']}")
    print(f"             PCR {ctx_by_day[k]['dir_flow']['pcr']:.2f} | strikes {len(ctx_by_day[k]['chain_df'])}\n")

eng=StrategyEngine(edge_table=EdgeTable()); rows=[]
for i in range(300,len(df)-H,3):
    ds=df.index[i].strftime("%Y-%m-%d"); ctx=ctx_by_day.get(ds)
    if ctx is None: continue
    sub=df.iloc[:i+1]
    s=eng.evaluate_bar(sub,current_idx=i,option_chain_df=ctx["chain_df"],options_context=ctx)
    if s.signal_type==SignalType.WAIT: continue
    d=1 if "LONG" in s.signal_type.value else -1
    rows.append(dict(i=i,dir=d,dirret=d*(close[i+H]-close[i])))
S=pd.DataFrame(rows)
print(f"DESK VERDICT with REAL positioning — signals fired: {len(S)}")
if len(S)<10: print("too few signals"); raise SystemExit
keep=[];last=-99
for _,r in S.iterrows():
    if r.i-last>=H: keep.append(r); last=r.i
N=pd.DataFrame(keep)
for lbl,X in (("all signals (overlapping)",S),("non-overlapping",N)):
    a=X.dirret.values; t=a.mean()/(a.std()/np.sqrt(len(a))) if a.std()>0 else 0
    print(f"  {lbl:<26} n={len(a):>4} hit={100*(a>0).mean():>5.1f}% mean={a.mean():>+7.2f} pts t={t:>+5.2f}  {'SIGNIFICANT' if abs(t)>2 else 'not sig'}")
rng=np.random.RandomState(42); base=[]
for _ in range(400):
    idx=rng.choice(np.arange(300,len(df)-H),size=len(N),replace=False)
    base.append(np.mean(N['dir'].values*(close[idx+H]-close[idx])))
base=np.array(base); obs=N.dirret.mean(); pct=100*(obs>base).mean()
print(f"\n  vs random baseline: random {base.mean():+.2f} (sd {base.std():.2f}) | observed {obs:+.2f} -> {pct:.1f}th pct")
print(f"  => {'BEATS random' if pct>95 else ('WORSE than random' if pct<5 else 'INDISTINGUISHABLE from random')}")
