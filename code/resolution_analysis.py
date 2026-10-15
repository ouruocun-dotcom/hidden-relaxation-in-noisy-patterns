"""Resolution analysis."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
HERE=Path(__file__).resolve().parent;DATA=HERE.parent/'results'/'field_data'

def stats(rows):
    branches=sum([r['branches'] for r in rows],[])
    down=sum(1 for r in rows for a,b in r['transitions'] if b<a)
    y=np.array([b['changed'] for b in branches],bool);x=np.array([b['chi'] for b in branches])
    auc=(float(mannwhitneyu(x[y],x[~y]).statistic/(sum(y)*sum(~y))) if np.any(y) and np.any(~y) else np.nan)
    return np.array([down/len(rows),np.mean(y) if len(y) else np.nan,auc],float)

def draw(rows,rng):
    seeds=sorted({r['seed'] for r in rows});chosen=rng.choice(seeds,len(seeds),replace=True)
    return [{**next(r for r in rows if r['seed']==s),'replicate':j} for j,s in enumerate(chosen)]

def main():
    raw=json.loads((DATA/'stochastic_convergence.json').read_text())['rows'];names=sorted({r['setting'] for r in raw})
    groups={n:[r for r in raw if r['setting']==n] for n in names};rng=np.random.default_rng(20261007)
    out={'metrics':['downward_transitions_per_trajectory','reactive_fraction','chi_auc'],'settings':{},'differences':{}}
    boots={}
    for n,q in groups.items():
        b=np.array([stats(draw(q,rng)) for _ in range(5000)]);boots[n]=b
        out['settings'][n]={'point':stats(q).tolist(),'ci95':np.nanquantile(b,[.025,.975],axis=0).T.tolist(),
                            'trajectories':len(q),'interventions':sum(len(r['branches']) for r in q)}
    for a,b in [('N384_dt0025','N512_dt0025'),('N768_dt0025','N512_dt0025'),
                ('N512_dt00125','N512_dt0025'),('N512_dt000625','N512_dt00125'),
                ('N512_dt000625','N512_dt0025')]:
        d=boots[a]-boots[b]
        out['differences'][f'{a}_minus_{b}']={'point':(stats(groups[a])-stats(groups[b])).tolist(),
          'ci95':np.nanquantile(d,[.025,.975],axis=0).T.tolist()}
    (DATA/'stochastic_convergence_analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
