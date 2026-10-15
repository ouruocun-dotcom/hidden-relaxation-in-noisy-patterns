"""Parameter analysis."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prediction_models import choose_lambda,fit_predict

HERE=Path(__file__).resolve().parent;DATA=HERE.parent/'results'/'field_data'
def auc(rows,key):
    p=[r[key] for r in rows if r['changed']];n=[r[key] for r in rows if not r['changed']]
    return float(mannwhitneyu(p,n).statistic/(len(p)*len(n)))
def confusion(rows,key,t):
    y=np.array([r['changed'] for r in rows]);p=np.array([r[key]>=t for r in rows]);
    sen=float(np.mean(p[y])) if np.any(y) else np.nan;spe=float(np.mean(~p[~y])) if np.any(~y) else np.nan
    return {'sensitivity':sen,'specificity':spe,'balanced_accuracy':.5*(sen+spe),'n':len(rows),'events':int(y.sum())}
def bootstrap(rows,key,rng):
    seeds=sorted({(r.get('d12','base'),r['seed']) for r in rows});values=[]
    for _ in range(10000):
        chosen=rng.choice(len(seeds),len(seeds),replace=True);sample=[]
        for rep,j in enumerate(chosen):
            s=seeds[j];sample += [{**r,'replicate':rep} for r in rows if (r.get('d12','base'),r['seed'])==s]
        if any(r['changed'] for r in sample) and any(not r['changed'] for r in sample):values.append(auc(sample,key))
    return np.quantile(values,[.025,.975]).tolist()
def probabilistic(rows,p):
    y=np.array([r['changed'] for r in rows],bool);p=np.clip(np.asarray(p),1e-12,1-1e-12)
    return {'brier':float(np.mean((y-p)**2)),'log_loss':float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))}
def cluster_metric_ci(rows,p,threshold,rng,repeats=5000):
    seeds=sorted({(r.get('d12','base'),r['seed']) for r in rows});vals=[]
    for _ in range(repeats):
        ids=[]
        chosen=rng.choice(len(seeds),len(seeds),replace=True)
        for j in chosen:
            s=seeds[j];ids.extend(i for i,r in enumerate(rows) if (r.get('d12','base'),r['seed'])==s)
        q=[rows[i] for i in ids];y=np.array([r['changed'] for r in q],bool)
        if not np.any(y) or not np.any(~y):continue
        a=auc(q,'relative_speed_mismatch');pr=probabilistic(q,np.asarray(p)[ids]);cf=confusion(q,'relative_speed_mismatch',threshold)
        vals.append([a,pr['brier'],pr['log_loss'],cf['sensitivity'],cf['specificity'],cf['balanced_accuracy']])
    a=np.asarray(vals);names=('auc','brier','log_loss','sensitivity','specificity','balanced_accuracy')
    return {n:np.quantile(a[:,j],[.025,.975]).tolist() for j,n in enumerate(names)}
def main():
    base=json.loads((DATA/'physical_coordinate_features.json').read_text());labels=json.loads((DATA/'long_switch_off_results.json').read_text())
    lab={(r['seed'],r['source'],r['m'],float(r['captured_at'])):r['long_changed'] for r in labels}
    first={}
    for r in sorted(base,key=lambda z:z['capture_age']):
        r['changed']=lab[(r['seed'],r['source'],r['m'],float(r['captured_at']))];r['weight']=1.;first.setdefault(r['platform_id'],r)
    train=[r for r in first.values() if r['seed']<12];test=[r for r in first.values() if r['seed']>=12]
    values=sorted({r['relative_speed_mismatch'] for r in train});candidates=[(a+b)/2 for a,b in zip(values[:-1],values[1:])]
    threshold=max(candidates,key=lambda t:confusion(train,'relative_speed_mismatch',t)['balanced_accuracy'])
    raw=json.loads((DATA/'parameter_robustness.json').read_text());groups={};nw=np.load(DATA/'external_wave_speeds.npz')
    for p in (.18,.22):
        speeds={m:float(nw[f'speed_d12_{p:.2f}_m{m}']) for m in (9,10,11)}
        group=[]
        for x in raw:
            if x['d12']==p:
                for b in x['branches']:
                    mismatch=(abs(b['c_star']-speeds[b['m']]) if 'c_star' in b else b['speed_mismatch'])
                    group.append({**b,'d12':p,'weight':1.,'relative_speed_mismatch':mismatch/abs(speeds[b['m']])})
        groups[f'd12={p:.2f}']=group
    lam,_=choose_lambda(train,(False,('relative_speed_mismatch',)))
    rng=np.random.default_rng(20261006);out={'threshold_selected_on_base_training':threshold,'probability_model_ridge_lambda':lam,
      'coordinate':'chi=abs(c_star-c_m)/abs(c_m)',
      'reference_waves':'Newton-refined independently at each external parameter point; rms residual below 9e-15',
      'base_training':confusion(train,'relative_speed_mismatch',threshold),'base_locked_validation':confusion(test,'relative_speed_mismatch',threshold),'groups':{}}
    for name,rows in groups.items():
        pred,_,_=fit_predict(train,rows,(False,('relative_speed_mismatch',)),lam);prob=probabilistic(rows,pred)
        out['groups'][name]={'speed_auc':auc(rows,'relative_speed_mismatch'),'speed_auc_ci95':bootstrap(rows,'relative_speed_mismatch',rng),
          'compression_auc':auc(rows,'maximum_compression'),'compression_auc_ci95':bootstrap(rows,'maximum_compression',rng),
          'fixed_threshold_performance':confusion(rows,'relative_speed_mismatch',threshold),**prob,
          'cluster_metric_ci95':cluster_metric_ci(rows,pred,threshold,rng)}
    pooled=sum(groups.values(),[]);pp,_,_=fit_predict(train,pooled,(False,('relative_speed_mismatch',)),lam)
    out['pooled_external']={'speed_auc':auc(pooled,'relative_speed_mismatch'),'speed_auc_ci95':bootstrap(pooled,'relative_speed_mismatch',rng),
      'compression_auc':auc(pooled,'maximum_compression'),'fixed_threshold_performance':confusion(pooled,'relative_speed_mismatch',threshold),
      **probabilistic(pooled,pp),'cluster_metric_ci95':cluster_metric_ci(pooled,pp,threshold,rng)}
    (DATA/'parameter_robustness_analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
