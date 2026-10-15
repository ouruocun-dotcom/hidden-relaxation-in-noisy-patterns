"""Prediction models."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import mannwhitneyu

HERE=Path(__file__).resolve().parent; DATA=HERE.parent/'results'/'field_data'
MODELS={'m_only':(True,()), 'm_age':(True,('log_age',)),
        'm_phase':(True,('maximum_compression',)),
        'm_age_phase':(True,('log_age','maximum_compression')),
        'chi_only':(False,('relative_speed_mismatch',)),
        'age_chi':(False,('log_age','relative_speed_mismatch')),
        'm_speed':(True,('relative_speed_mismatch',)),
        'm_age_speed':(True,('log_age','relative_speed_mismatch'))}
LAMBDAS=(0.,.01,.1,1.,10.,100.)

def prepare():
    rows=json.loads((DATA/'physical_coordinate_features.json').read_text())
    labels=json.loads((DATA/'long_switch_off_results.json').read_text())
    lab={(r['seed'],r['source'],r['m'],float(r['captured_at'])):r['long_changed'] for r in labels}
    counts={}
    for r in rows: counts[r['platform_id']]=counts.get(r['platform_id'],0)+1
    for r in rows:
        r['changed']=int(lab[(r['seed'],r['source'],r['m'],float(r['captured_at']))])
        r['log_age']=float(np.log(r['capture_age']/48.))
        r['weight']=1/counts[r['platform_id']]
    return rows

def design(train,test,include_m,extra):
    # Intercept plus m=10 and m=11 indicators; m=9 is the reference.
    def base(rows): return np.array([[1.,r['m']==10,r['m']==11] for r in rows],float)
    a,b=(base(train),base(test)) if include_m else (np.ones((len(train),1)),np.ones((len(test),1)))
    if extra:
        tr=np.array([[r[k] for k in extra] for r in train]);te=np.array([[r[k] for k in extra] for r in test])
        mu=tr.mean(0);sd=tr.std(0);sd[sd==0]=1
        a=np.c_[a,(tr-mu)/sd];b=np.c_[b,(te-mu)/sd]
    return a,b

def fit_predict(train,test,spec,lam):
    include_m,extra=spec;x,z=design(train,test,include_m,extra);y=np.array([r['changed'] for r in train]);w=np.array([r['weight'] for r in train]);w=w/w.mean()
    def objective(beta):
        eta=x@beta; nll=np.sum(w*(np.logaddexp(0,eta)-y*eta));return nll+.5*lam*np.sum(beta[1:]**2)
    def gradient(beta):
        g=x.T@(w*(expit(x@beta)-y));g[1:]+=lam*beta[1:];return g
    q=minimize(objective,np.zeros(x.shape[1]),jac=gradient,method='L-BFGS-B',
               options={'ftol':1e-14,'gtol':1e-9,'maxiter':2000})
    return expit(z@q.x),q.x,bool(q.success)

def logloss(rows,p):
    y=np.array([r['changed'] for r in rows]);w=np.array([r['weight'] for r in rows]);p=np.clip(p,1e-12,1-1e-12)
    return float(np.average(-(y*np.log(p)+(1-y)*np.log(1-p)),weights=w))
def brier(rows,p):
    y=np.array([r['changed'] for r in rows]);w=np.array([r['weight'] for r in rows]);return float(np.average((y-p)**2,weights=w))
def auc(rows,p):
    y=np.array([r['changed'] for r in rows]);return float(mannwhitneyu(p[y==1],p[y==0]).statistic/(sum(y==1)*sum(y==0)))
def calibration(rows,p):
    y=np.array([r['changed'] for r in rows]);w=np.array([r['weight'] for r in rows]);order=np.argsort(p); bins=np.array_split(order,3);points=[]
    for ids in bins: points.append({'predicted':float(np.average(p[ids],weights=w[ids])),'observed':float(np.average(y[ids],weights=w[ids])),'n':len(ids)})
    ece=float(sum(np.sum(w[ids])*abs(points[j]['predicted']-points[j]['observed']) for j,ids in enumerate(bins))/sum(w))
    return points,ece

def choose_lambda(rows,spec):
    seeds=sorted({r['seed'] for r in rows});scores={}
    for lam in LAMBDAS:
        total=0.;weight=0.
        for seed in seeds:
            tr=[r for r in rows if r['seed']!=seed];va=[r for r in rows if r['seed']==seed]
            if not va:continue
            p,_,_=fit_predict(tr,va,spec,lam);w=sum(r['weight'] for r in va);total+=w*logloss(va,p);weight+=w
        scores[lam]=total/weight
    return min(scores,key=scores.get),scores

def evaluate_split(train,test):
    out={};pred={}
    for name,spec in MODELS.items():
        lam,cv=choose_lambda(train,spec);p,beta,success=fit_predict(train,test,spec,lam);points,ece=calibration(test,p)
        pred[name]=p;out[name]={'lambda':lam,'inner_group_cv_log_loss':cv,'coefficients':beta.tolist(),'optimizer_success':success,
            'log_loss':logloss(test,p),'brier':brier(test,p),'auc':auc(test,p),'calibration_ece':ece,'calibration_tertiles':points}
    return out,pred

def main():
    rows=prepare();train=[r for r in rows if r['seed']<12];test=[r for r in rows if r['seed']>=12]
    locked,pred=evaluate_split(train,test)
    rng=np.random.default_rng(20261006);seeds=sorted({r['seed'] for r in test});boot={name:{'dll':[],'dbrier':[]} for name in MODELS if name!='m_only'}
    for _ in range(10000):
        chosen=rng.choice(seeds,len(seeds),replace=True);ids=[]
        for j,s in enumerate(chosen): ids.extend([i for i,r in enumerate(test) if r['seed']==s])
        sample=[test[i] for i in ids]
        base_ll=logloss(sample,pred['m_only'][ids]);base_br=brier(sample,pred['m_only'][ids])
        for name in boot:
            boot[name]['dll'].append(base_ll-logloss(sample,pred[name][ids]))
            boot[name]['dbrier'].append(base_br-brier(sample,pred[name][ids]))
    intervals={name:{'log_loss_improvement_ci95':np.quantile(v['dll'],[.025,.975]).tolist(),
                     'brier_improvement_ci95':np.quantile(v['dbrier'],[.025,.975]).tolist()} for name,v in boot.items()}
    # Nested leave-one-seed-out robustness on all trajectories.
    outer={name:[] for name in MODELS}; outer_rows=[]
    for seed in sorted({r['seed'] for r in rows}):
        tr=[r for r in rows if r['seed']!=seed];te=[r for r in rows if r['seed']==seed]
        if not te:continue
        outer_rows.extend(te)
        for name,spec in MODELS.items():
            lam,_=choose_lambda(tr,spec);p,_,_=fit_predict(tr,te,spec,lam);outer[name].extend(p.tolist())
    nested={name:{'log_loss':logloss(outer_rows,np.array(p)),'brier':brier(outer_rows,np.array(p)),'auc':auc(outer_rows,np.array(p))} for name,p in outer.items()}
    out={'design':'platform-weighted repeated interventions; split and resampling by noisy trajectory',
         'n_train_interventions':len(train),'n_test_interventions':len(test),
         'n_train_platforms':len({r['platform_id'] for r in train}),'n_test_platforms':len({r['platform_id'] for r in test}),
         'locked_validation':locked,'locked_cluster_bootstrap_improvements_over_m_only':intervals,'nested_leave_one_trajectory_out':nested}
    (DATA/'reduced_model_competition.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
