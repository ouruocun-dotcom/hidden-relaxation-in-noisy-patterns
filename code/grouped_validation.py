"""Grouped validation."""
from __future__ import annotations
import json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
HERE=Path(__file__).resolve().parent;DATA=HERE.parent/'results'/'field_data';sys.path.insert(0,str(HERE))
from prediction_models import MODELS,prepare,choose_lambda,fit_predict,logloss,brier,auc
AUDIT_MODELS={k:MODELS[k] for k in ('m_only','m_age','m_phase','chi_only')}

def nested(rows,models=AUDIT_MODELS):
    ordered=[];pred={k:[] for k in models};folds=[]
    for seed in sorted({r['seed'] for r in rows}):
        tr=[r for r in rows if r['seed']!=seed];te=[r for r in rows if r['seed']==seed]
        folds.append({'seed':seed,'n':len(te),'events':sum(r['changed'] for r in te),
                      'platforms':len({r['platform_id'] for r in te})})
        ordered.extend(te)
        for name,spec in models.items():
            lam,_=choose_lambda(tr,spec);p,_,_=fit_predict(tr,te,spec,lam);pred[name].extend(p.tolist())
    return ordered,{k:np.array(v) for k,v in pred.items()},folds

def calibration(rows,p):
    y=np.array([r['changed'] for r in rows],float);w=np.array([r['weight'] for r in rows],float)
    x=logit(np.clip(p,1e-6,1-1e-6))
    def fun(q):
        eta=q[0]+q[1]*x;return np.sum(w*(np.logaddexp(0,eta)-y*eta))
    q=minimize(fun,[0.,1.],method='BFGS')
    return {'intercept':float(q.x[0]),'slope':float(q.x[1]),'success':bool(q.success)}

def metrics(rows,p):
    return {'brier':brier(rows,p),'log_loss':logloss(rows,p),'auc':auc(rows,p),
            'calibration':calibration(rows,p)}

def basic_metrics(rows,p):
    return {'brier':brier(rows,p),'log_loss':logloss(rows,p),'auc':auc(rows,p)}

def bootstrap(rows,pred,repeats=3000,seed=20261007):
    rng=np.random.default_rng(seed);seeds=sorted({r['seed'] for r in rows});vals={k:[] for k in pred};diff={k:[] for k in pred if k!='m_only'}
    for _ in range(repeats):
        chosen=rng.choice(seeds,len(seeds),replace=True);ids=[]
        for s in chosen:ids.extend(i for i,r in enumerate(rows) if r['seed']==s)
        sample=[rows[i] for i in ids];base=None
        for name,p in pred.items():
            try:m=basic_metrics(sample,p[ids])
            except ValueError:continue
            vals[name].append([m['brier'],m['log_loss'],m['auc']])
            if name=='m_only':base=m
        if base is not None:
            for name in diff:
                try:m=basic_metrics(sample,pred[name][ids])
                except ValueError:continue
                diff[name].append([base['brier']-m['brier'],base['log_loss']-m['log_loss'],m['auc']-base['auc']])
    labels=('brier','log_loss','auc');out={}
    for name,v in vals.items():
        a=np.asarray(v);out[name]={f'{lab}_ci95':np.quantile(a[:,j],[.025,.975]).tolist() for j,lab in enumerate(labels)}
    dout={}
    for name,v in diff.items():
        a=np.asarray(v);dout[name]={'brier_improvement_ci95':np.quantile(a[:,0],[.025,.975]).tolist(),
          'log_loss_improvement_ci95':np.quantile(a[:,1],[.025,.975]).tolist(),
          'auc_improvement_ci95':np.quantile(a[:,2],[.025,.975]).tolist()}
    return out,dout

def _full_pipeline_replicate(args):
    rows,seeds,draw_seed=args; rng=np.random.default_rng(draw_seed)
    models={k:AUDIT_MODELS[k] for k in ('m_only','chi_only')}
    chosen=rng.choice(seeds,len(seeds),replace=True)
    counts={s:int(np.sum(chosen==s)) for s in seeds}
    sample=[{**r,'weight':float(r['weight']*counts[r['seed']])}
            for r in rows if counts[r['seed']]>0]
    if len({r['seed'] for r in sample})<3:return None
    try: ordered,pred,_=nested(sample,models)
    except (ValueError,np.linalg.LinAlgError):return None
    base=basic_metrics(ordered,pred['m_only']);m=basic_metrics(ordered,pred['chi_only'])
    z=[base['brier']-m['brier'],base['log_loss']-m['log_loss'],m['auc']-base['auc']]
    return z if np.all(np.isfinite(z)) else None

def full_pipeline_cluster_bootstrap(rows,repeats=500,seed=20261008,workers=4):
    """Refit nested grouped CV after a trajectory-level weighted bootstrap."""
    seeds=sorted({r['seed'] for r in rows});seq=np.random.SeedSequence(seed).spawn(repeats)
    args=[(rows,seeds,int(s.generate_state(1)[0])) for s in seq]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        vals=[v for v in pool.map(_full_pipeline_replicate,args,chunksize=4) if v is not None]
    values={'chi_only':vals}
    out={}
    for name,v in values.items():
        a=np.asarray(v)
        out[name]={'successful_replicates':len(a),
          'brier_improvement_ci95':np.quantile(a[:,0],[.025,.975]).tolist(),
          'log_loss_improvement_ci95':np.quantile(a[:,1],[.025,.975]).tolist(),
          'auc_improvement_ci95':np.quantile(a[:,2],[.025,.975]).tolist()}
    return out

def analyse(rows):
    ordered,pred,folds=nested(rows);point={k:metrics(ordered,p) for k,p in pred.items()};ci,diff=bootstrap(ordered,pred)
    refit_ci=full_pipeline_cluster_bootstrap(rows)
    return {'n':len(ordered),'platforms':len({r['platform_id'] for r in ordered}),
            'trajectories':len({r['seed'] for r in ordered}),'folds':folds,'point':point,
            'cluster_bootstrap_ci95':ci,'paired_improvement_over_m_only':diff,
            'full_pipeline_cluster_bootstrap_improvement_over_m_only':refit_ci,
            'interval_interpretation':'full_pipeline intervals refit nested grouped cross-validation in every trajectory-level weighted bootstrap replicate; fixed-OOF intervals are retained as a sensitivity analysis',
            'out_of_fold_predictions':[{**{'seed':r['seed'],'platform_id':r['platform_id'],'m':r['m'],'changed':r['changed']},
              **{name:float(pred[name][i]) for name in pred}} for i,r in enumerate(ordered)],
            'ridge_grid':[0.,.01,.1,1.,10.,100.],
            'standardization':'training fold only','missing_capture':'no imputation; absent ages contribute no row'}

def main():
    rows=prepare()
    first={}
    for r in sorted(rows,key=lambda z:z['capture_age']):first.setdefault(r['platform_id'],{**r,'weight':1.})
    episode_result=analyse(list(first.values()))
    out={'primary_independent_episode_analysis':episode_result,
         'repeated_interventions_platform_weighted_sensitivity':json.loads((DATA/'reduced_model_competition.json').read_text())}
    (DATA/'grouped_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
