from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'results'/'junction';D=Path(__file__).resolve().parents[1]/'results'/'field_data';O=R.parent/'figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
blue='#2878a0';red='#c75238'
f=json.loads((D/'physical_coordinate_features.json').read_text());lab=json.loads((D/'long_switch_off_results.json').read_text());lookup={(r['seed'],r['index']):r for r in lab};fields=np.load(D/'capture_fields.npz');waves=np.load(D/'newton_traveling_waves.npz')
# Independently recompute the actual low-mode speed used by cached inference.
errs=[];all_rows=[]
for r in f:
 u=fields[f"fields_seed_{r['seed']}"][r['index']];n=u.shape[1];k=2*np.pi*np.fft.fftfreq(n,d=200/n);uh=np.fft.fft(u,axis=1)
 force=np.stack([-u[0]+u[0]**3-np.fft.ifft(-k*k*uh[0]).real+.2*u[1],-.2*u[0]+.1*u[1]])
 F=-k*k*np.fft.fft(force,axis=1);X=1j*k*uh;mask=(abs(np.fft.fftfreq(n)*n)<=2*r['m'])&(k!=0);w=1/k[mask]**2
 cs=-np.real(np.sum(np.conj(F[:,mask])*X[:,mask]*w))/np.sum(abs(X[:,mask])**2*w);cm=float(waves[f"speed_{r['m']}_N256"]);chi=abs(cs-cm)/abs(cm)
 errs.append(abs(chi-r['relative_speed_mismatch']));all_rows.append({**r,'outcome':lookup[(r['seed'],r['index'])]['long_changed']})
assert max(errs)<1e-10
first=[r for r in all_rows if r['capture_age']==48];assert len(first)==44 and sum(r['outcome'] for r in first)==20
# Illustrations: count10, closest to each outcome group's median chi; not validation.
selected=[]
for out in [False,True]:
 rows=[r for r in first if r['m']==10 and r['outcome']==out];med=np.median([r['relative_speed_mismatch'] for r in rows]);selected.append(min(rows,key=lambda r:abs(r['relative_speed_mismatch']-med)))
fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained',gridspec_kw={'width_ratios':[1.35,1]})
for j,(r,c) in enumerate(zip(selected,[blue,red])):
 u=fields[f"fields_seed_{r['seed']}"][r['index']][0];u=np.roll(u,-2*r['alignment_shift']);x=np.arange(len(u))*200/len(u)
 ax[0].plot(x,u+3*j,color=c,lw=1.3,label=f"10 stripes; speed mismatch {100*r['relative_speed_mismatch']:.0f}%")
 ax[0].text(198,3*j+1.35,'loses a stripe' if j else 'keeps 10 stripes',ha='right',color=c,fontsize=9)
ax[0].set(xlabel='Position',ylabel=r'Field $\phi$ (vertical offsets)',title='(a) Same count, different departures',xlim=(0,200),yticks=[],ylim=(-1.5,4.9));ax[0].legend(loc='center left',bbox_to_anchor=(.01,.5),fontsize=8)
for count in [9,10,11]:
 rows=[r for r in first if r['m']==count];rows=sorted(rows,key=lambda r:(r['seed'],r['index']));jit=np.linspace(-.12,.12,len(rows))
 for r,dx in zip(rows,jit):ax[1].scatter(count+dx,100*r['relative_speed_mismatch'],color=red if r['outcome'] else blue,marker='s' if r['outcome'] else 'o',s=25,alpha=.85)
ax[1].scatter([],[],c=blue,label='no further stripe loss',s=25);ax[1].scatter([],[],c=red,marker='s',label='further stripe loss',s=25)
ax[1].axhline(0,ls=':',color='.5');ax[1].set(xlabel='Stripe count',ylabel='Speed mismatch (%)',xticks=[9,10,11],title='(b) One capture per episode');ax[1].legend(fontsize=8)
fig.savefig(O/'speed_intuition.png',dpi=240)
(R/'speed_definition_check.json').write_text(json.dumps({'n':len(errs),'max_chi_recalculation_error':max(errs),'first_capture_n':len(first),'first_capture_events':20,'illustrative_pairs':selected,'selection':'count10; within each outcome select closest to median chi. Descriptive only; not an independent validation.'},indent=2)+'\n')
# Same-generator risks and their conservative model error certificate.
s=json.loads((R/'noise_on_results.json').read_text());fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for j,(b,style) in enumerate(zip(s['batches'],['-','--'])):
 rows=b['rows'];h=[r['horizon'] for r in rows];pa=[r['retained_events']/r['n'] for r in rows];pb=[r['reset_events']/r['n'] for r in rows]
 ax[0].plot(h,pa,style,color=red,marker='o' if j==0 else '^',label=('retain velocity' if j==0 else 'retain, independent batch'))
 ax[0].plot(h,pb,style,color=blue,marker='o' if j==0 else '^',label=('reset velocity' if j==0 else 'reset, independent batch'))
 lower=[100*r['minimax_error_lower_simultaneous'] for r in rows];ax[1].plot(h,lower,style,color='black' if j==0 else '.5',marker='o' if j==0 else '^',label=b['batch'].replace('_',' '))
ax[0].set(xlabel='Follow-up horizon H',ylabel='Fraction crossing next saddle',ylim=(-.015,.53),title='(a) Same noise, same position');ax[0].legend(fontsize=8)
ax[1].set(xlabel='Follow-up horizon H',ylabel='Lower error bound (percentage points)',ylim=(-.5,20),title='(b) Cost of one common prediction');ax[1].legend(fontsize=9)
ax[1].annotate('Simultaneous 95% sampling bound\nFixed discrete simulation protocol',xy=(.03,.98),xycoords='axes fraction',va='top',fontsize=8)
fig.savefig(O/'noise_on_diagnostic.png',dpi=240)
print('PASS: 126 low-mode speeds checked; two figures generated')
