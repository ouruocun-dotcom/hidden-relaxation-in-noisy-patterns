"""Field coordinates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample
from field_drift import drift,one_residual

HERE=Path(__file__).resolve().parent; DATA=HERE.parent/'results'/'field_data'; L=200.; N=256


def align(field,ref):
    score=np.fft.ifft(np.conj(np.fft.fft(ref[0]))*np.fft.fft(field[0])+
                       np.conj(np.fft.fft(ref[1]))*np.fft.fft(field[1])).real
    shift=int(np.argmax(score))
    return np.roll(field,-shift,axis=1),shift


def crossing_features(phi,m):
    x=np.arange(N)*L/N; nxt=np.roll(phi,-1); mask=(phi<0)&(nxt>=0); ids=np.flatnonzero(mask)
    if len(ids)!=m: return {k:np.nan for k in ('spacing_cv','spacing_q1','minimum_spacing_ratio','maximum_compression','spacing_laplacian_rms')}
    frac=-phi[ids]/(nxt[ids]-phi[ids]); pos=(x[ids]+frac*L/N)%L; pos.sort()
    spacing=np.diff(np.r_[pos,pos[0]+L]); cv=float(spacing.std()/spacing.mean())
    # First Fourier component of the displacement of crossings from a uniform lattice.
    disp=spacing-spacing.mean(); q1=float(abs(np.sum(disp*np.exp(-2j*np.pi*np.arange(m)/m)))/L)
    normalized=spacing/spacing.mean()
    return {'spacing_cv':cv,'spacing_q1':q1,
            'minimum_spacing_ratio':float(normalized.min()),
            'maximum_compression':float(1-normalized.min()),
            'spacing_laplacian_rms':float(np.sqrt(np.mean((np.roll(normalized,-1)-2*normalized+np.roll(normalized,1))**2)))}


def modal_coeff(delta,left,right,inds):
    values=[]
    for j in inds:
        den=np.vdot(left[:,j],right[:,j]); values.append(abs(np.vdot(left[:,j],delta)/den))
    return float(np.sqrt(np.mean(np.square(values))))


def phase_velocity_features(phi,psi,m,c_wave):
    fphi,_,ux,_,_=drift(phi,psi,L=L)
    nxt=np.roll(phi,-1); ids=np.flatnonzero((phi<0)&(nxt>=0))
    if len(ids)!=m:
        return {k:np.nan for k in ('phase_velocity_rms','maximum_spacing_contraction_rate','collision_propensity')}
    frac=-phi[ids]/(nxt[ids]-phi[ids]); pos=(ids+frac)*L/N
    interp=lambda a:(1-frac)*a[ids]+frac*a[(ids+1)%N]
    order=np.argsort(pos);pos=pos[order];spacing=np.diff(np.r_[pos,pos[0]+L])
    out={}
    for cutoff,label in ((None,''),(2*m,'_band_2m'),(4*m,'_band_4m')):
        ff,xx=fphi,ux
        if cutoff is not None:
            modes=np.abs(np.fft.fftfreq(N)*N);mask=modes<=cutoff
            ff=np.fft.ifft(np.fft.fft(fphi)*mask).real;xx=np.fft.ifft(np.fft.fft(ux)*mask).real
        velocities=(-interp(ff)/interp(xx))[order]
        contraction=velocities-np.roll(velocities,-1);positive=np.maximum(contraction,0)
        out[f'phase_velocity_rms{label}']=float(np.sqrt(np.mean((velocities-c_wave)**2)))
        out[f'maximum_spacing_contraction_rate{label}']=float(positive.max())
        out[f'collision_propensity{label}']=float(np.max(positive/spacing))
    return out


def main():
    idx=json.loads((DATA/'capture_fields_index.json').read_text()); fz=np.load(DATA/'capture_fields.npz')
    wz=np.load(DATA/'newton_traveling_waves.npz'); sz=np.load(DATA/'wave_spectrum.npz')
    rows=[]
    for row in idx:
        m=int(row['m']); field=fz[f"fields_seed_{row['seed']}"][row['index']]
        ff1,ff2,xx1,xx2,kk=drift(field[0],field[1])
        fitted_speed,fitted_residual,_=one_residual(ff1,ff2,xx1,xx2,kk,2*m)
        field=resample(field,N,axis=1)
        ref=np.stack([wz[f'phi_{m}_N256'],wz[f'psi_{m}_N256']]); aligned,shift=align(field,ref)
        delta=(aligned-ref).reshape(-1); left=sz[f'left_m{m}_N256']; right=sz[f'right_m{m}_N256']
        spacing=crossing_features(aligned[0],m)
        phase_velocity=phase_velocity_features(aligned[0],aligned[1],m,float(wz[f'speed_{m}_N256']))
        speed_mismatch=abs(fitted_speed-float(wz[f'speed_{m}_N256']))
        rows.append({**row,'platform_id':f"{row['seed']}:{m}:{row['entered_at']}",
          'alignment_shift':shift,**spacing,**phase_velocity,
          'speed_mismatch':speed_mismatch,
          'relative_speed_mismatch':speed_mismatch/abs(float(wz[f'speed_{m}_N256'])),
          'residual_band_2m':fitted_residual,
          # modes 3--6 are the two slow non-phase pairs; 7--8 are q=1 phase modes
          'slow_nonphase_amplitude':modal_coeff(delta,left,right,range(3,7)),
          'phase_q1_amplitude':modal_coeff(delta,left,right,range(7,9)),
          'orbit_distance':float(np.linalg.norm(delta)/np.sqrt(2*N))})
    (DATA/'physical_coordinate_features.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('rows',len(rows),'platforms',len({r['platform_id'] for r in rows}))

if __name__=='__main__': main()
