"""Field figures."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; DATA=ROOT/'results'/'field_data'; OUT=ROOT/'results'/'figures'
OUT.mkdir(parents=True,exist_ok=True)
def load(name): return json.loads((DATA/name).read_text())


def spectrum_figure():
    z=np.load(DATA/'wave_spectrum.npz')
    data=[{'m':m,'N':384,'modes':[{'real':float(v.real),'imag':float(v.imag)} for v in z[f'eigenvalues_m{m}_N384']]} for m in (9,10,11,12)]
    fig,ax=plt.subplots(1,2,figsize=(7.1,3.0),constrained_layout=True)
    colors={9:'#4477AA',10:'#228833',11:'#EE7733',12:'#AA3377'}
    for q in data:
        if q['N']!=384: continue
        modes=q['modes'][3:15]
        ax[0].scatter([r['imag'] for r in modes],[r['real'] for r in modes],s=20,color=colors[q['m']],label=f"$m={q['m']}$")
        phase=q['modes'][7]; slow=q['modes'][3]
        ax[1].scatter(q['m'], -slow['real'],marker='s',s=35,color=colors[q['m']])
        ax[1].scatter(q['m'], -phase['real'],marker='o',s=35,color=colors[q['m']])
    ax[0].axhline(0,color='0.5',lw=.8); ax[0].set(xlabel=r'Im $\lambda$',ylabel=r'Re $\lambda$',ylim=(-3e-4,1e-5))
    ax[0].legend(frameon=False,ncol=2,fontsize=8)
    ax[1].set_yscale('log'); ax[1].set_xticks([9,10,11,12]); ax[1].set(xlabel='stripe count $m$',ylabel='decay rate')
    ax[1].scatter([],[],marker='s',color='0.3',label='slowest non-phase mode')
    ax[1].scatter([],[],marker='o',color='0.3',label=r'$q=1$ phase mode')
    ax[1].legend(frameon=False,fontsize=8)
    for s,a in zip('ab',ax): a.text(-.17,1.03,f'({s})',transform=a.transAxes,fontweight='bold')
    fig.savefig(OUT/'exact_spectrum_phase_modes.png',dpi=300); fig.savefig(OUT/'exact_spectrum_phase_modes.pdf')


def intervention_figure():
    hist=load('long_switch_off_history.json'); rows=load('long_switch_off_results.json')
    fig,ax=plt.subplots(1,2,figsize=(7.1,3.0),constrained_layout=True)
    t=np.r_[0,[r['time'] for r in hist]]; ch=np.r_[0,[r['changed'] for r in hist]]; mm=np.r_[np.nan,[r['survivor_peak_mismatch'] for r in hist]]
    ax[0].step(t,ch,where='post',color='#CC3311',label='persistent stripe loss')
    ax[0].step(t,mm,where='post',color='#4477AA',label='survivor peak mismatch')
    ax[0].set(xlabel='time after noise removal',ylabel='number of branches',xlim=(0,30000)); ax[0].legend(frameon=False,fontsize=8)
    ages=[48,100,200,400]; vals=[]; lo=[]; hi=[]; labels=[]
    for age in ages:
        g=[r for r in rows if r['capture_age']==age]; k=sum(r['long_changed'] for r in g); n=len(g); f=k/n
        vals.append(f); lo.append(0 if k==0 else beta.ppf(.025,k,n-k+1)); hi.append(1 if k==n else beta.ppf(.975,k+1,n-k)); labels.append(f'{k}/{n}')
    vals=np.array(vals);lo=np.array(lo);hi=np.array(hi)
    ax[1].errorbar(ages,vals,yerr=[vals-lo,hi-vals],fmt='o-',capsize=3,color='#6A3D9A')
    for x,y,s in zip(ages,vals,labels): ax[1].annotate(s,(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax[1].set(xlabel='plateau age at noise removal',ylabel='fraction with deterministic stripe loss',ylim=(0,.8))
    for s,a in zip('ab',ax): a.text(-.17,1.03,f'({s})',transform=a.transAxes,fontweight='bold')
    fig.savefig(OUT/'long_switch_off_audit.png',dpi=300); fig.savefig(OUT/'long_switch_off_audit.pdf')


def predictor_figure():
    feat=load('physical_coordinate_features.json'); result=load('long_switch_off_results.json'); comp=load('physical_predictor_comparison.json')
    lab={(r['seed'],r['source'],r['m'],float(r['captured_at'])):r['long_changed'] for r in result}
    for r in feat:r['changed']=lab[(r['seed'],r['source'],r['m'],float(r['captured_at']))]
    first={}
    for r in sorted(feat,key=lambda z:z['capture_age']): first.setdefault(r['platform_id'],r)
    test=[r for r in first.values() if r['seed']>=12]
    metrics=[('residual_band_2m',r'$R_{2m}$'),('speed_mismatch',r'$|c-c_m|$'),('spacing_cv','spacing CV'),('phase_q1_amplitude',r'linear $q=1$ phase')]
    fig,ax=plt.subplots(1,4,figsize=(7.1,2.8),constrained_layout=True,
                       gridspec_kw={'width_ratios':[1,1,1,2.0]})
    for j,(key,label) in enumerate(metrics[:3]):
        for changed,marker,color,off in [(False,'o','#4477AA',-.08),(True,'s','#CC3311',.08)]:
            y=[r[key] for r in test if r['changed']==changed]; ax[j].scatter(np.full(len(y),off),y,marker=marker,color=color,s=22,alpha=.75)
        ax[j].set_xticks([0],[label]); ax[j].tick_params(axis='x',labelrotation=15)
    ax[0].set_ylabel('instantaneous predictor')
    keys=[x[0] for x in metrics]; names=[x[1] for x in metrics]; auc=[comp['validation_platform_auc'][k] for k in keys]
    ci=np.array([comp['cluster_bootstrap_ci95'][k] for k in keys]); x=np.arange(len(keys))
    ax[3].bar(x,auc,color=['#332288','#117733','#DDCC77','#999999'],alpha=.85)
    ax[3].errorbar(x,auc,yerr=[np.array(auc)-ci[:,0],ci[:,1]-np.array(auc)],fmt='none',color='black',capsize=3)
    ax[3].axhline(.5,color='0.5',ls='--'); ax[3].set_xticks(x,names,rotation=25,ha='right'); ax[3].set(ylabel='validation AUC',ylim=(0,1.05))
    ax[0].text(-.45,1.03,'(a)',transform=ax[0].transAxes,fontweight='bold'); ax[3].text(-.2,1.03,'(b)',transform=ax[3].transAxes,fontweight='bold')
    fig.savefig(OUT/'physical_predictor_comparison.png',dpi=300); fig.savefig(OUT/'physical_predictor_comparison.pdf')


def transient_figure():
    a=load('survival_uncertainty.json')['rows'];cuts=[r['left'] for r in a]
    fig,ax=plt.subplots(1,2,figsize=(7.1,3.0),constrained_layout=True)
    shape=np.array([r['shape'] for r in a]);ci=np.array([r['shape_bootstrap_ci95'] for r in a])
    ax[0].errorbar(cuts,shape,yerr=[shape-ci[:,0],ci[:,1]-shape],fmt='o-',capsize=3,color='#AA3377');ax[0].axhline(1,color='.45',ls='--')
    ax[0].set(xlabel='left-truncation time',ylabel='Weibull shape')
    da=np.array([r['aic_exponential_minus_weibull'] for r in a]);dc=np.array([r['aic_difference_bootstrap_ci95'] for r in a])
    ax[1].errorbar(cuts,da,yerr=[da-dc[:,0],dc[:,1]-da],fmt='o-',capsize=3,color='#4477AA');ax[1].axhline(0,color='.45',ls='--')
    ax[1].set(xlabel='left-truncation time',ylabel=r'$\mathrm{AIC}_{exp}-\mathrm{AIC}_{Wei}$')
    for s,z in zip('ab',ax):z.text(-.17,1.03,f'({s})',transform=z.transAxes,fontweight='bold')
    fig.savefig(OUT/'left_truncation_audit.png',dpi=300); fig.savefig(OUT/'left_truncation_audit.pdf')


def closure_figure():
    audit=load('grouped_validation.json')['primary_independent_episode_analysis'];models=audit['point'];mci=audit['cluster_bootstrap_ci95']
    external=load('parameter_robustness_analysis.json')
    fig,ax=plt.subplots(1,2,figsize=(7.1,3.0),constrained_layout=True)
    order=['m_only','m_phase','chi_only']
    labels=[r'$m$',r'$m+$ compression',r'$\chi$']
    colors=['#999999','#DDCC77','#117733']
    x=np.arange(len(order))
    val=np.array([models[k]['brier'] for k in order]);ci=np.array([mci[k]['brier_ci95'] for k in order])
    ax[0].bar(x,val,color=colors,alpha=.9)
    ax[0].errorbar(x,val,yerr=[val-ci[:,0],ci[:,1]-val],fmt='none',color='black',capsize=3)
    ax[0].set_xticks(x,labels,rotation=18,ha='right')
    ax[0].set(ylabel='trajectory-held-out Brier score',ylim=(0,.32))
    groups=[external['groups']['d12=0.18'],external['groups']['d12=0.22']]
    glabels=[r'$D_{12}=0.18$',r'$D_{12}=0.22$']
    auc=[g['speed_auc'] for g in groups]
    ci=np.array([g['speed_auc_ci95'] for g in groups])
    xx=np.arange(2)
    ax[1].errorbar(xx,auc,yerr=[np.array(auc)-ci[:,0],ci[:,1]-np.array(auc)],
                   fmt='o',ms=6,capsize=3,color='#117733')
    ax[1].axhline(.5,color='0.45',ls='--',lw=.9)
    ax[1].set_xticks(xx,glabels,rotation=12,ha='right')
    ax[1].set(ylabel=r'AUC of $\chi=|c^*-c_m|/|c_m|$',ylim=(0,1.02))
    for i,g in enumerate(groups):
        ax[1].annotate(f"BA={g['fixed_threshold_performance']['balanced_accuracy']:.2f}",(i,auc[i]),
                       xytext=(0,8),textcoords='offset points',ha='center',fontsize=7)
    for s,a in zip('ab',ax): a.text(-.17,1.03,f'({s})',transform=a.transAxes,fontweight='bold')
    fig.savefig(OUT/'reduced_closure_robustness.png',dpi=300)
    fig.savefig(OUT/'reduced_closure_robustness.pdf')

def history_figure():
    d=load('preparation_risks.json')['rows'];q=[r for r in d if r['horizon']==100]
    fig,ax=plt.subplots(figsize=(4.8,3.1),constrained_layout=True);x=np.arange(3);w=.34
    er=np.array([r['entered']['risk'] for r in q]);rr=np.array([r['relaxed']['risk'] for r in q])
    ax.bar(x-w/2,er,w,label='newly entered',color='#CC3311');ax.bar(x+w/2,rr,w,label='prepared wave',color='#4477AA')
    ci=np.array([r['newcombe_wilson_risk_difference_ci95'] for r in q]);
    for i,r in enumerate(q):ax.text(i,er[i]+.035,f"RD {r['risk_difference']:.2f}\n[{ci[i,0]:.2f},{ci[i,1]:.2f}]",ha='center',fontsize=7)
    ax.set_xticks(x,[f"$m={r['m']}$" for r in q]);ax.set(ylabel='100-time-unit transition risk',ylim=(0,.85));ax.legend(frameon=False)
    fig.savefig(OUT/'finite_noise_history.png',dpi=300);fig.savefig(OUT/'finite_noise_history.pdf')

def convergence_figure():
    d=load('stochastic_convergence_analysis.json');order=['N384_dt0025','N512_dt0025','N768_dt0025','N512_dt00125','N512_dt000625']
    labels=['$N=384$\n$\\Delta t=.025$','$N=512$\n$\\Delta t=.025$','$N=768$\n$\\Delta t=.025$',
            '$N=512$\n$\\Delta t=.0125$','$N=512$\n$\\Delta t=.00625$']
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.8),constrained_layout=True);yl=['transitions / trajectory','reactive fraction',r'AUC of $\chi$']
    for j,a in enumerate(ax):
        v=np.array([d['settings'][k]['point'][j] for k in order]);ci=np.array([d['settings'][k]['ci95'][j] for k in order]);x=np.arange(len(order))
        a.errorbar(x,v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='o',capsize=3,color='#332288');a.set_xticks(x,labels,rotation=25,ha='right');a.set_ylabel(yl[j]);a.text(-.2,1.03,f'({"abc"[j]})',transform=a.transAxes,fontweight='bold')
        if j==2:a.axhline(.5,color='.5',ls='--')
    fig.savefig(OUT/'stochastic_convergence.png',dpi=300);fig.savefig(OUT/'stochastic_convergence.pdf')


def main():
    OUT.mkdir(exist_ok=True); plt.rcParams.update({'font.size':9,'axes.linewidth':.8})
    spectrum_figure(); intervention_figure(); predictor_figure(); transient_figure(); closure_figure(); history_figure(); convergence_figure()

if __name__=='__main__':main()
