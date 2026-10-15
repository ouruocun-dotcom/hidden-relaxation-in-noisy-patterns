"""Survival models."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

HERE=Path(__file__).resolve().parent; DATA=HERE.parent/'results'/'field_data'; STATES=(9,10,11,12)


def design(rows):
    inv=np.array([1/r['noise'] for r in rows]); inv-=inv.mean()
    x=np.zeros((len(rows),5)); si={m:i for i,m in enumerate(STATES)}
    for i,r in enumerate(rows): x[i,si[r['m']]]=1
    x[:,-1]=inv
    t=np.array([r['first_passage_time'] if r['event_observed'] else r['horizon'] for r in rows])
    e=np.array([r['event_observed'] for r in rows],float)
    return x,t,e


def fit_weibull(rows,left=0.,fixed_p=None):
    x,t,e=design(rows); keep=t>left; x,t,e=x[keep],t[keep],e[keep]
    def nll(theta):
        beta=theta[:5]; p=fixed_p if fixed_p is not None else np.exp(theta[5]); eta=x@beta
        return float(np.sum(np.exp(np.clip(eta,-700,700))*(t**p-left**p)
                    -e*(eta+np.log(p)+(p-1)*np.log(t))))
    init=np.r_[[ -29.4,-27.8,-25.9,-23.7,-.088],1.1]
    if fixed_p is not None: init=init[:5]
    bounds=([(None,None)]*5 if fixed_p is not None else [(None,None)]*5+[(-3.,3.)])
    q=minimize(nll,init,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-13,'maxiter':5000})
    k=len(q.x); return {'left_truncation':left,'n':int(len(t)),'events':int(e.sum()),
        'log_likelihood':-float(q.fun),'aic':2*k+2*float(q.fun),'shape':fixed_p if fixed_p else float(np.exp(q.x[5])),
        'success':bool(q.success)}


def integrated_ramp(t,tau):
    return (t-3*tau*(1-np.exp(-t/tau))+1.5*tau*(1-np.exp(-2*t/tau))
            -(tau/3)*(1-np.exp(-3*t/tau)))


def fit_ramp(rows):
    x,t,e=design(rows)
    def nll(theta):
        eta=x@theta[:5]; tau=np.exp(theta[5]); hshape=(1-np.exp(-t/tau))**3
        return float(np.sum(np.exp(np.clip(eta,-700,700))*integrated_ramp(t,tau)
                            -e*(eta+np.log(np.maximum(hshape,1e-300)))))
    q=minimize(nll,np.r_[[-12.6,-11,-9.3,-7.8,-.059],np.log(1500)],method='L-BFGS-B',
               options={'ftol':1e-13,'maxiter':10000})
    return {'model':'saturating cubic hazard','tau':float(np.exp(q.x[5])),
            'log_likelihood':-float(q.fun),'aic':12+2*float(q.fun),'success':bool(q.success)}


def main():
    rows=json.loads((DATA/'relaxed_escape_scan.json').read_text())
    out={'earliest_event':min(r['first_passage_time'] for r in rows if r['event_observed']),
         'full_weibull':fit_weibull(rows),'full_exponential':fit_weibull(rows,fixed_p=1.),
         'saturating_hazard':fit_ramp(rows),'left_truncation':[]}
    for left in (500.,1000.,1500.,1800.,2000.):
        w=fit_weibull(rows,left); e=fit_weibull(rows,left,fixed_p=1.)
        out['left_truncation'].append({'left':left,'weibull':w,'exponential':e,
            'likelihood_ratio':2*(w['log_likelihood']-e['log_likelihood']),
            'lr_pvalue':float(chi2.sf(max(0,2*(w['log_likelihood']-e['log_likelihood'])),1))})
    (DATA/'survival_models.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
