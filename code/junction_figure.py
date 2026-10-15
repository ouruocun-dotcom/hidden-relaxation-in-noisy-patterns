from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1]/'results'/'junction'
s=json.loads((R/'results.json').read_text());d=np.load(R/'production.npz');v=d['velocity'][d['captured']];y=d['outcome'];a=np.arcsin(s['bias']);b=np.pi-a
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(12,3.65),layout='constrained');colors=['#2878a0','#cd593d']
for flag,c,label in [(False,colors[0],'retraps'),(True,colors[1],'continues')]:
 value=float(v[y==flag][np.argmin(abs(v[y==flag]-np.median(v[y==flag])))])
 sol=solve_ivp(lambda t,z:[z[1],s['bias']-np.sin(z[0])-s['gamma']*z[1]],[0,35],[a,value],t_eval=np.linspace(0,35,1000),rtol=1e-10,atol=1e-12)
 axs[0].plot(sol.t,(sol.y[0]-a)/(2*np.pi),color=c,label=label)
axs[0].axhline((b-a)/(2*np.pi),color='.4',ls=':',label='next saddle')
axs[0].plot([0,35],[0,0],color='.4',ls='--',lw=1,label='velocity reset')
axs[0].set(xlabel='Time after noise removal',ylabel=r'Displacement / $2\pi$',title='(a) Same position, different futures');axs[0].legend(fontsize=8,loc='upper right')
ins=axs[0].inset_axes([.53,.12,.43,.22]);xx=np.linspace(a-1,a+2*np.pi+1,400);ins.plot(xx,-np.cos(xx)-s['bias']*xx,color='.35');ins.scatter([a,b],[-np.cos(a)-s['bias']*a,-np.cos(b)-s['bias']*b],c=['black',colors[1]],s=12);ins.set_xticks([]);ins.set_yticks([])
bins=np.linspace(0,3.5,29)
for flag,c,label in [(False,colors[0],'retraps'),(True,colors[1],'continues')]:axs[1].hist(v[y==flag],bins=bins,color=c,label=label,alpha=.8)
axs[1].axvline(s['threshold_v'],color='black',label='$v_c$',lw=1.4);axs[1].axvline(s['energy_bound_v'],ls='--',color='.4',label='energy bound')
axs[1].set(xlabel='Velocity at capture',ylabel='Number of captures',title='(b) Hidden velocity separates outcomes');axs[1].legend(fontsize=8)
for j,row in enumerate(s['batches']):
 lo,hi=row['wilson95'];q=row['fraction'];axs[2].errorbar(j,q,yerr=[[q-lo],[hi-q]],fmt='o',capsize=4,color=colors[1]);axs[2].text(j,q+.045,f"{row['continue']}/{row['captured']}",ha='center',fontsize=9)
axs[2].scatter([0,1],[0,0],marker='s',color='.4',label='velocity reset')
axs[2].set(xticks=[0,1],xticklabels=['$h=0.01$','$h=0.005$'],ylabel='Noise-off continuation fraction',ylim=(-.025,.49),xlim=(-.6,1.6),title='(c) Independent timestep repeat');axs[2].legend(fontsize=8,loc='upper right')
fig.savefig(R.parent/'figures/josephson_intervention.png',dpi=240)
print('figure generated')
