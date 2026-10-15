"""Coordinate comparison."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

HERE=Path(__file__).resolve().parent; DATA=HERE.parent/'results'/'field_data'
METRICS=('residual_band_2m','speed_mismatch','spacing_cv','spacing_q1',
         'slow_nonphase_amplitude','phase_q1_amplitude','orbit_distance')


def auc(rows,key,direction=1):
    p=[direction*r[key] for r in rows if r['changed'] and np.isfinite(r[key])]
    n=[direction*r[key] for r in rows if not r['changed'] and np.isfinite(r[key])]
    return float(mannwhitneyu(p,n).statistic/(len(p)*len(n))) if p and n else np.nan


def main():
    rows=json.loads((DATA/'physical_coordinate_features.json').read_text())
    labels={(r['seed'],r['source'],r['m'],float(r['captured_at'])):r['long_changed']
            for r in json.loads((DATA/'long_switch_off_results.json').read_text())}
    for r in rows: r['changed']=labels[(r['seed'],r['source'],r['m'],float(r['captured_at']))]
    # One prespecified state per distinct plateau: earliest (age 48) capture.
    first={}
    for r in sorted(rows,key=lambda q:q['capture_age']): first.setdefault(r['platform_id'],r)
    platform=list(first.values()); train=[r for r in platform if r['seed']<12]; test=[r for r in platform if r['seed']>=12]
    alltest=[r for r in rows if r['seed']>=12]
    directions={k:(1 if auc(train,k)>=.5 else -1) for k in METRICS}
    point={k:auc(test,k,directions[k]) for k in METRICS}; branch={k:auc(alltest,k,directions[k]) for k in METRICS}
    rng=np.random.default_rng(20261006); seeds=sorted({r['seed'] for r in test}); boot={k:[] for k in METRICS}
    deltas={k:[] for k in METRICS if k!='residual_band_2m'}
    for _ in range(10000):
        chosen=rng.choice(seeds,len(seeds),replace=True); sample=[]
        for j,s in enumerate(chosen): sample += [{**r,'replicate':j} for r in test if r['seed']==s]
        vals={k:auc(sample,k,directions[k]) for k in METRICS}
        for k,v in vals.items():
            if np.isfinite(v): boot[k].append(v)
        for k in deltas:
            if np.isfinite(vals[k]) and np.isfinite(vals['residual_band_2m']):
                deltas[k].append(vals['residual_band_2m']-vals[k])
    out={'unit':'earliest intervention per accepted plateau','n_platforms_total':len(platform),
         'n_validation_platforms':len(test),'n_validation_seeds':len(seeds),
         'directions_selected_on_training':directions,
         'validation_platform_auc':point,'validation_all_interventions_auc':branch,
         'cluster_bootstrap_ci95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()},
         'residual_minus_baseline_auc_ci95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in deltas.items()}}
    (DATA/'physical_predictor_comparison.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
