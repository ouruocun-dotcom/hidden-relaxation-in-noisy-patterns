"""Paired fixed-generator continuations from captured phase particles."""
from pathlib import Path
import json
import hashlib
import numpy as np
from preparation_gap import certificate
ROOT=Path(__file__).resolve().parents[1]/'results'/'junction'; ROOT.mkdir(parents=True,exist_ok=True)
BIAS=.15;GAMMA=.2;TEMP=.35
FINE_DT=.0025;H=np.array([1.,2.,5.,10.,20.]);ALPHA=.05;M=10
A=np.arcsin(BIAS);SADDLE=np.pi-A

def step(x,v,dt,z):
 v=v+.5*dt*(BIAS-np.sin(x));x=x+.5*dt*v
 a=np.exp(-GAMMA*dt);v=a*v+np.sqrt(TEMP*(1-a*a))*z
 x=x+.5*dt*v;v=v+.5*dt*(BIAS-np.sin(x))
 return x,v

def run(batch,seed):
 raw=(ROOT/(batch+'.npz')).read_bytes();inp=np.load(ROOT/(batch+'.npz'));valid=inp['captured'];vel=inp['velocity'][valid];n=len(vel)
 # rows retained and reset; independent units in columns, shared noise in each pair
 v0=np.stack([vel,np.zeros(n)]);xf=np.full((2,n),A);vf=v0.copy();xc=xf.copy();vc=v0.copy()
 tf=np.full((2,n),np.inf);tc=tf.copy();rng=np.random.default_rng(seed);a=np.exp(-GAMMA*FINE_DT)
 for k in range(round(H[-1]/(2*FINE_DT))):
  z1=rng.standard_normal(n);z2=rng.standard_normal(n)
  xf,vf=step(xf,vf,FINE_DT,z1)
  t=(2*k+1)*FINE_DT;hit=(xf>=SADDLE)&np.isinf(tf);tf[hit]=t
  xf,vf=step(xf,vf,FINE_DT,z2)
  t=(2*k+2)*FINE_DT;hit=(xf>=SADDLE)&np.isinf(tf);tf[hit]=t
  zc=(a*z1+z2)/np.sqrt(1+a*a)
  xc,vc=step(xc,vc,2*FINE_DT,zc);hit=(xc>=SADDLE)&np.isinf(tc);tc[hit]=t
 margin=np.sqrt(2*np.log(M/ALPHA)/n);rows=[]
 for horizon in H:
  y=tf<=horizon;coarse=tc<=horizon;d=y[0].astype(float)-y[1];gap=float(d.mean());L=gap-certificate(y[0].astype(int).tolist(),y[1].astype(int).tolist(),ALPHA,M,True)['radius']
  row={'horizon':float(horizon),'retained_events':int(y[0].sum()),'reset_events':int(y[1].sum()),'n':n,'gap':gap,'gap_lower_simultaneous':L,'minimax_error_lower_simultaneous':max(0,L)/2,'coarse_fine_disagree_retained':int(np.sum(coarse[0]!=y[0])),'coarse_fine_disagree_reset':int(np.sum(coarse[1]!=y[1])),'paired_discordant_retained_only':int(np.sum(y[0]&~y[1])),'paired_discordant_reset_only':int(np.sum(~y[0]&y[1]))}
  rows.append(row)
 np.savez_compressed(ROOT/(batch+'_noise_on.npz'),capture_path_indices=np.flatnonzero(valid),velocity=vel,fine_first_hit=tf,coarse_first_hit=tc,horizons=H)
 return {'batch':batch,'seed':seed,'input_sha256':hashlib.sha256(raw).hexdigest(),'hoeffding_margin':float(margin),'rows':rows}

def main():
 out={'bias':BIAS,'gamma':GAMMA,'temperature':TEMP,'fine_dt':FINE_DT,'coarse_dt':2*FINE_DT,'family_size':M,'alpha':ALPHA,'batches':[run('production',2026101503),run('fine_repeat',2026101504)],'scope':'Confidence is for fixed discrete-protocol conditional risks, including capture and future-noise sampling. No continuous-time numerical error certification.'}
 (ROOT/'noise_on_results.json').write_text(json.dumps(out,indent=2)+'\n')
 for b in out['batches']:print(b['batch'],json.dumps(b['rows'][2]),flush=True)
if __name__=='__main__':main()
