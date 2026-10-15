"""Independent cache counts, confidence intervals, and separatrix check."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import norm
R=Path(__file__).resolve().parents[1]/'results'/'junction';s=json.loads((R/'results.json').read_text());checks=[]
for row in s['batches']:
 a=np.load(R/(row['batch']+'.npz'));ok=a['captured'];v=a['velocity'][ok];t=a['capture_time'][ok];y=a['outcome'];n=len(v);k=int(y.sum());z=norm.ppf(.975)
 # independently find score roots instead of using production closed form
 score=lambda p:(k-n*p)**2-z*z*n*p*(1-p)
 lo=brentq(score,0,k/n);hi=brentq(score,k/n,1)
 assert n==row['captured'] and k==row['continue'] and np.all((t>0)&(t<=s['horizon']))
 assert np.all(np.isnan(a['velocity'][~ok])) and np.all(np.isnan(a['capture_time'][~ok]))
 assert np.max(np.abs(np.array([lo,hi])-row['wilson95']))<1e-11
 assert np.array_equal(y,v>s['threshold_v'])
 checks.append({'batch':row['batch'],'n':n,'k':k,'wilson_roots':[lo,hi],'distance_to_threshold_min':float(np.min(abs(v-s['threshold_v'])))})
# Independent deterministic stable-manifold construction: integrate backwards from right saddle.
i=s['bias'];g=s['gamma'];a=np.arcsin(i);b=np.pi-a
lam=(-g-np.sqrt(g*g+4*np.cos(a)))/2
est=[]
for eps in [1e-5,1e-6,1e-7]:
 def hit(t,y):return y[0]-a
 hit.terminal=True;hit.direction=-1
 sol=solve_ivp(lambda t,y:[y[1],i-np.sin(y[0])-g*y[1]],[0,-100],[b-eps,-lam*eps],method='DOP853',events=hit,rtol=2e-12,atol=1e-13)
 val=float(sol.y_events[0][0,1]);est.append(val)
assert max(abs(v-s['threshold_v']) for v in est)<2e-8
x,y=s['batches'];d=x['fraction']-y['fraction'];se=np.sqrt(x['fraction']*(1-x['fraction'])/x['captured']+y['fraction']*(1-y['fraction'])/y['captured'])
out={'checks':checks,'backward_separatrix_thresholds':est,'difference_production_minus_fine':d,'normal95_descriptive':[d-1.95996398454*se,d+1.95996398454*se],'note':'independent-seed difference includes sampling variation, not a deterministic discretization bound'}
(R/'independent_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
