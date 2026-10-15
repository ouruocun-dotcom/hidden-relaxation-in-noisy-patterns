"""Survival uncertainty."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;DATA=HERE.parent/'results'/'field_data';sys.path.insert(0,str(HERE))
from survival_models import fit_weibull

def main():
    rows=json.loads((DATA/'relaxed_escape_scan.json').read_text());rng=np.random.default_rng(20261007);out=[]
    for left in (0.,500.,1000.,1500.,1800.,2000.):
        w=fit_weibull(rows,left);e=fit_weibull(rows,left,fixed_p=1.);vals=[]
        for _ in range(500):
            q=[rows[i] for i in rng.integers(0,len(rows),len(rows))]
            wb=fit_weibull(q,left);eb=fit_weibull(q,left,fixed_p=1.)
            if wb['success'] and eb['success']:vals.append([wb['shape'],eb['aic']-wb['aic']])
        a=np.asarray(vals)
        out.append({'left':left,'risk_set':w['n'],'events':w['events'],'shape':w['shape'],
          'shape_bootstrap_ci95':np.quantile(a[:,0],[.025,.975]).tolist(),
          'aic_exponential_minus_weibull':e['aic']-w['aic'],
          'aic_difference_bootstrap_ci95':np.quantile(a[:,1],[.025,.975]).tolist(),
          'successful_bootstraps':len(a)})
        print(out[-1],flush=True)
    result={'truncation_times_prespecified_before_bootstrap':True,'bootstrap_unit':'complete simulated trajectory','rows':out}
    (DATA/'survival_uncertainty.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
