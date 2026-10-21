"""Compare legacy table and downloaded run_04 states with identical transport inputs."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np,pandas as pd,h5py
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tto_general_gmft import load_saddle,dynamic_structure_factor as general
from tto_structure_factor import row_mean_field,dynamic_structure_factor as scalar,eio_to_tto_reduced_momentum
from config import E_FERMI,ENERGY_BROADENING
spec=importlib.util.spec_from_file_location('scattering',root/'03_compute_scattering.py');sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
run=root.parent/'gMFT_transport/run_04'
with h5py.File(run/'data/surface_states_summary_nk24.hdf5') as f:
 k=np.squeeze(f['kf'][...]); en=np.squeeze(f['eigf'][...]);v=f['v_kf'][...];proj={i:f[f'proj_sl{i}'][...] for i in [1,2,3]}
weights=np.array([sum(sc.local_eio_weights(proj[j],np.ones(3)) for j in sc.EIO_NEIGHBORS[i]) for i in range(3)])
q=eio_to_tto_reduced_momentum(k[:,None,:]-k[None,:,:],10.154); w=en[None,:]-en[:,None]
vertex=1-np.clip(k@k.T/np.outer(np.linalg.norm(k,axis=1),np.linalg.norm(k,axis=1)),-1,1);np.fill_diagonal(vertex,0)
sw=np.sum(sum(abs(proj[i])**2 for i in [1,2,3]),axis=(1,2))
fermi=np.exp(-.5*((en-E_FERMI)/ENERGY_BROADENING)**2)/(np.sqrt(2*np.pi)*ENERGY_BROADENING)
def rho(d):
 t2=np.einsum('mij,mij->ij',d,weights);inv=np.sum(t2*vertex,axis=1)
 return 1/np.sum(fermi*abs(v[:,0])**2*sw/inv)
win=pd.read_csv(run/'winning_saddles.csv');out=pd.read_csv(run/'data/tto_transport_summary.csv')
rows=[]
for h in [.15,.2,.25]:
 for phi in [0,15,30]:
  t=pd.read_csv(root.parent/f'gMFT_in_plane_min/phi_{phi}/field_diagnostics_min_energy.csv')
  old=t[np.isclose(t.J_pm_over_J_zz,.15)&np.isclose(t.h_over_J_zz,h)&np.isclose(t.phi_degrees,phi)].iloc[0]
  chosen=win[np.isclose(win.h_over_J_zz,h/10)&(win.phi_degrees==phi)].iloc[0]
  mf=load_saddle(run/chosen.saddle_file)
  actual=out[np.isclose(out.h_over_J_zz,h/10)&(out.phi_degrees==phi)].iloc[0]
  kw=dict(j_zz_ev=.00017,eta_ev=.00002,nk=8)
  od=scalar(q,w,row_mean_field(old),**kw)['total']
  nd=general(q,w,mf,**kw)['total']
  same=dict(j_pm_ratio=float(mf['j_pm_ratio']),lambda_ratio=float(mf['lambda_ratio']),s_minus=mf['s_minus'],rho=float(mf['rho']),condensed=float(mf['rho'])>0,k_min=np.zeros(3))
  assert float(mf['rho'])==0
  sd=scalar(q,w,same,**kw)['total']
  hz=np.array([old[f'h_z_mu{i}'] for i in range(4)])
  hm=np.array([old[f'h_minus_real_mu{i}']+1j*old[f'h_minus_imag_mu{i}'] for i in range(4)])
  norm=np.maximum(np.sqrt(abs(hm)**2+hz**2),1e-12)
  oldm=row_mean_field(old)['s_minus'];oldz=np.array([old[f's_z_mu{i}'] for i in range(4)])
  residual=max(np.max(abs(.5*hm/norm-oldm)),np.max(abs(.5*hz/norm-oldz)))
  rows.append(dict(effective_field=h,phi=phi,old_residual=residual,old_rho=rho(od),new_recomputed_rho=rho(nd),new_saved_rho=actual.rho,scalar_general_relative_difference=np.max(abs(sd-nd))/np.max(abs(nd)),old_max_interface_transverse=max(abs(oldm[1:])),new_max_interface_transverse=max(abs(mf['s_minus'][1:]))))
r=pd.DataFrame(rows);print(r.to_string(index=False));r.to_csv(Path(__file__).with_name('gmft_reference_comparison.csv'),index=False)
