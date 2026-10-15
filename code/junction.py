"""RCSJ trajectories, noise-off continuation, and independent integrator comparison."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]/'results'/'junction'; ROOT.mkdir(parents=True,exist_ok=True)
BIAS=.15; GAMMA=.2; TEMP=.35; N=1024; HORIZON=1500.; FOLLOW=300.
A=np.arcsin(BIAS); SECTION=A+2*np.pi; SADDLE=np.pi-A

def field(t,y):return [y[1],BIAS-np.sin(y[0])-GAMMA*y[1]]
def continuation(v):
 def event(t,y):return y[0]-SADDLE
 event.direction=1;event.terminal=True
 sol=solve_ivp(field,[0,FOLLOW],[A,float(v)],method='DOP853',events=event,rtol=1e-10,atol=1e-12)
 return bool(len(sol.t_events[0]))
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-r,c+r]
def capture(dt,seed):
 rng=np.random.default_rng(seed);x=np.full(N,A);v=np.zeros(N);cv=np.full(N,np.nan);ct=np.full(N,np.nan)
 decay=np.exp(-GAMMA*dt);sigma=np.sqrt(TEMP*(1-decay**2))
 for step in range(round(HORIZON/dt)):
  active=np.isnan(cv)
  if not active.any():break
  ox=x.copy();ov=v.copy()
  v+=.5*dt*(BIAS-np.sin(x));x+=.5*dt*v
  v=decay*v+sigma*rng.standard_normal(N)
  x+=.5*dt*v;v+=.5*dt*(BIAS-np.sin(x))
  hit=active&(ox<SECTION)&(x>=SECTION)
  frac=(SECTION-ox[hit])/(x[hit]-ox[hit])
  cv[hit]=ov[hit]+frac*(v[hit]-ov[hit]);ct[hit]=(step+frac)*dt
 return cv,ct

def rk4(v,dt):
 # vectorized independent integrator, relative position at the shared section
 x=np.full(len(v),A);v=v.copy();hit=np.zeros(len(v),bool);max_positive=0.
 energy=lambda x,v:.5*v*v-np.cos(x)-BIAS*x
 for step in range(round(FOLLOW/dt)):
  e=energy(x,v)
  f=lambda x,v:BIAS-np.sin(x)-GAMMA*v
  a1=v;b1=f(x,v);a2=v+dt*b1/2;b2=f(x+dt*a1/2,v+dt*b1/2)
  a3=v+dt*b2/2;b3=f(x+dt*a2/2,v+dt*b2/2);a4=v+dt*b3;b4=f(x+dt*a3,v+dt*b3)
  x+=dt*(a1+2*a2+2*a3+a4)/6;v+=dt*(b1+2*b2+2*b3+b4)/6
  hit|=x>=SADDLE;max_positive=max(max_positive,float(np.max(energy(x,v)-e)))
 return hit,max_positive

def main():
 lo,hi=0.,4.
 for _ in range(45):
  mid=(lo+hi)/2
  if continuation(mid):hi=mid
  else:lo=mid
 barrier=2*np.sqrt(1-BIAS**2)-BIAS*(np.pi-2*A)
 summary={'bias':BIAS,'gamma':GAMMA,'temperature':TEMP,'paths_per_batch':N,'horizon':HORIZON,'followup':FOLLOW,'threshold_v':hi,'energy_bound_v':float(np.sqrt(2*barrier)),'barrier':float(barrier),'reset_outcome':continuation(0.),'batches':[]}
 for dt,seed,name in [(.01,2026101501,'production'),(.005,2026101502,'fine_repeat')]:
  cv,ct=capture(dt,seed);valid=np.isfinite(cv);vv=cv[valid]
  outcome=np.array([continuation(z) for z in vv]);check,e=rk4(vv,.0025);check2,e2=rk4(vv,.00125)
  assert np.array_equal(outcome,check) and np.array_equal(outcome,check2)
  assert np.array_equal(outcome,vv>hi)
  np.savez_compressed(ROOT/(name+'.npz'),velocity=cv,capture_time=ct,captured=valid,outcome=outcome)
  row={'batch':name,'dt':dt,'seed':seed,'captured':int(valid.sum()),'uncaptured':int((~valid).sum()),'continue':int(outcome.sum()),'fraction':float(outcome.mean()),'wilson95':wilson(outcome.sum(),len(vv)),'max_positive_energy_step':max(e,e2),'independent_RK4_disagreements':0}
  summary['batches'].append(row);print(json.dumps(row),flush=True)
 (ROOT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
