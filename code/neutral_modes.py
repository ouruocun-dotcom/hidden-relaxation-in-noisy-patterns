"""Neutral modes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;DATA=HERE.parent/'results'/'field_data'

def ov(a,b): return float(abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))
def subspace_singular(a,b):
    qa,_=np.linalg.qr(a);qb,_=np.linalg.qr(b)
    return np.linalg.svd(np.conj(qa).T@qb,compute_uv=False).real.tolist()
def main():
    z=np.load(DATA/'wave_spectrum.npz');waves=np.load(DATA/'newton_traveling_waves.npz');out=[]
    for N in (256,384):
      k=2*np.pi*np.fft.fftfreq(N,d=200/N)
      for m in (9,10,11,12):
        vals=z[f'eigenvalues_m{m}_N{N}'];right=z[f'right_m{m}_N{N}'];left=z[f'left_m{m}_N{N}']
        phi=waves[f'phi_{m}_N{N}'];psi=waves[f'psi_{m}_N{N}']
        dxphi=np.fft.ifft(1j*k*np.fft.fft(phi)).real;dxpsi=np.fft.ifft(1j*k*np.fft.fft(psi)).real
        tangent=np.r_[dxphi,dxpsi];ephi=np.r_[np.ones(N),np.zeros(N)];epsi=np.r_[np.zeros(N),np.ones(N)]
        neutral=[]
        for j in range(3):
            r=right[:,j];l=left[:,j]
            neutral.append({'eigenvalue':[float(vals[j].real),float(vals[j].imag)],
              'right_translation_overlap':ov(r,tangent),
              'right_mean_phi':float(abs(np.mean(r[:N]))/np.linalg.norm(r)),
              'right_mean_psi':float(abs(np.mean(r[N:]))/np.linalg.norm(r)),
              'left_constant_phi_overlap':ov(l,ephi),'left_constant_psi_overlap':ov(l,epsi)})
        vr=right[:,:3];vl=left[:,:3]
        const=np.column_stack([ephi,epsi]);means=np.array([[np.mean(vr[:N,j]),np.mean(vr[N:,j])] for j in range(3)]).T
        out.append({'m':m,'N':N,'operator':'unconstrained comoving PDE linearization',
                    'neutral_modes':neutral,
                    'neutral_subspace_translation_projection':float(np.linalg.norm(np.conj(np.linalg.qr(vr)[0]).T@(tangent/np.linalg.norm(tangent)))),
                    'left_constant_subspace_cosines':subspace_singular(vl,const),
                    'right_neutral_mean_map_singular_values':np.linalg.svd(means,compute_uv=False).real.tolist(),
                    'leading_fixed_mean_nontranslation_real':float(max(v.real for v in vals[3:])),
                    'interpretation':'one right translation mode; two neutral family directions paired with the two left constant conservation laws and excluded by fixed-mean constraints'})
    (DATA/'neutral_modes.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
