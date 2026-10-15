"""Preparation risks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, norm

HERE=Path(__file__).resolve().parent; DATA=HERE.parent/'results'/'field_data'
AGES=(50.0,); HORIZONS=(100.0,250.0,500.0,1000.0)

def observations():
    """Use a fixed-horizon cohort; terminal retained episodes are censored.

    Eligibility is determined by entry time plus age plus horizon, for BOTH
    early-event and surviving episodes. Dropping only short-followed survivors
    would preferentially keep events and is deliberately avoided.
    """
    network=json.loads((DATA/'state_network.json').read_text())
    relaxed=json.loads((DATA/'relaxed_escape_scan.json').read_text())
    rows=[]
    for tr in network['trajectories']:
        states=tr['25']['states']
        if not states: continue
        # The cached accepted itinerary ends at its last observed retained
        # sample. Do not infer a transition from the end of that record.
        observation_end=min(float(network['parameters']['T']),
                            float(states[-1]['end'])+2.0)
        for j,state in enumerate(states):
            m=int(state['m'])
            if m not in (9,10,11) or j==0: continue
            duration=float(state['duration'])
            if duration < 50.0: continue
            event_observed=(j+1 < len(states))
            event_age=duration if event_observed else float('inf')
            for age in AGES:
                for horizon in HORIZONS:
                    # Same time-origin and end-of-observation gate for events
                    # and non-events, independent of their future event time.
                    if float(state['start'])+age+horizon > observation_end+1e-9:
                        continue
                    remaining=event_age-age
                    rows.append({'ensemble':'entered','trajectory':int(tr['seed']),
                                 'episode':f"{tr['seed']}:{j}",'m':m,'age':age,
                                 'horizon':horizon,'event':int(event_observed and remaining<=horizon),
                                 'event_observed':event_observed,
                                 'entry_time':float(state['start']),
                                 'observation_end':observation_end,
                                 'followup':min(remaining,horizon)})
    for r in relaxed:
        if float(r['noise'])!=.016 or int(r['m']) not in (9,10,11): continue
        event_observed=bool(r['event_observed'])
        event_time=float(r['first_passage_time']) if event_observed else float('inf')
        for age in AGES:
            if event_time < age: continue
            for horizon in HORIZONS:
                if age+horizon > float(r['horizon'])+1e-9: continue
                remaining=event_time-age
                rows.append({'ensemble':'relaxed','trajectory':int(r['seed']),
                             'episode':f"{r['m']}:{r['seed']}",'m':int(r['m']),'age':age,
                             'horizon':horizon,
                             'event':int(event_observed and remaining<=horizon),
                             'event_observed':event_observed,
                             'entry_time':0.0,'observation_end':float(r['horizon']),
                             'followup':min(remaining,horizon)})
    return rows

def effect(rows):
    out={}
    for e in ('entered','relaxed'):
        q=[r for r in rows if r['ensemble']==e]
        out[e]={'n':len(q),'events':sum(r['event'] for r in q),
                'risk':float(np.mean([r['event'] for r in q])) if q else np.nan,
                'trajectories':len({r['trajectory'] for r in q})}
    out['risk_difference']=out['entered']['risk']-out['relaxed']['risk']
    out['risk_ratio']=(out['entered']['risk']/out['relaxed']['risk']
                       if out['relaxed']['risk']>0 else None)
    return out

def clustered_bootstrap(rows,repeats=5000,seed=20261007):
    rng=np.random.default_rng(seed); vals=[]
    groups={e:sorted({r['trajectory'] for r in rows if r['ensemble']==e})
            for e in ('entered','relaxed')}
    for _ in range(repeats):
        sample=[]
        for e,seeds in groups.items():
            for rep,s in enumerate(rng.choice(seeds,len(seeds),replace=True)):
                sample += [{**r,'bootstrap_cluster':f'{e}:{rep}'} for r in rows
                           if r['ensemble']==e and r['trajectory']==s]
        v=effect(sample)['risk_difference']
        if np.isfinite(v): vals.append(v)
    return np.quantile(vals,[.025,.5,.975]).tolist()

def wilson_interval(events,n,alpha=.05):
    """Two-sided Wilson score interval for one binomial proportion."""
    z=float(norm.ppf(1-alpha/2)); p=events/n; den=1+z*z/n
    centre=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,centre-half),min(1.,centre+half)

def newcombe_difference_ci(e1,n1,e0,n0,alpha=.05):
    """Newcombe score interval (method 10) for p1-p0."""
    l1,u1=wilson_interval(e1,n1,alpha); l0,u0=wilson_interval(e0,n0,alpha)
    p1=e1/n1; p0=e0/n0; d=p1-p0
    lo=d-np.sqrt((p1-l1)**2+(u0-p0)**2)
    hi=d+np.sqrt((u1-p1)**2+(p0-l0)**2)
    return [float(max(-1.,lo)),float(min(1.,hi))]

def exact_one_sided_p(e1,n1,e0,n0):
    return float(fisher_exact([[e1,n1-e1],[e0,n0-e0]],alternative='greater').pvalue)

def main():
    rows=observations(); report=[]
    for m in (9,10,11):
        for age in AGES:
            for horizon in HORIZONS:
                q=[r for r in rows if r['m']==m and r['age']==age and r['horizon']==horizon]
                if not q: continue
                z=effect(q); a=z['entered']; b=z['relaxed']
                if a['n']==0 or b['n']==0: continue
                z.update({'m':m,'age':age,'horizon':horizon,
                          'newcombe_wilson_risk_difference_ci95':newcombe_difference_ci(a['events'],a['n'],b['events'],b['n']),
                          'fisher_exact_one_sided_p':exact_one_sided_p(a['events'],a['n'],b['events'],b['n']),
                          'cluster_bootstrap_risk_difference_ci95_descriptive':clustered_bootstrap(q)})
                report.append(z)
    pooled=[]
    for horizon in HORIZONS:
        q=[r for r in rows if r['horizon']==horizon]
        z=effect(q);z.update({'m':'stratified-pooled-descriptive','age':50.,'horizon':horizon,
                             'cluster_bootstrap_risk_difference_ci95_descriptive':clustered_bootstrap(q)})
        pooled.append(z)
    out={'estimand':'risk of a retained stripe-count episode ending in a subsequent accepted different-count episode within horizon, conditional on survival to age 50 and entry early enough for full administrative follow-up',
         'constant_hazard_assumption':False,
         'censoring_protocol':'Terminal retained episodes are right-censored, never counted as events. A common entry-time plus age plus horizon gate applies to early events and survivors. Endpoint is the cached final retained sample plus two time units, capped at T.',
         'multiple_comparisons':'Pointwise unadjusted intervals and p-values; m=9,10,11 and four horizons are not independent confirmatory discoveries.',
         'primary_interval':'Newcombe-Wilson score interval; unlike the empirical bootstrap it retains binomial uncertainty when a group has zero observed events',
         'rows':report,'pooled_descriptive':pooled}
    (DATA/'preparation_risks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
