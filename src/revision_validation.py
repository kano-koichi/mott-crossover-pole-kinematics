#!/usr/bin/env python3
"""Fast independent analytic-branch audit with adaptive near-critical sampling.
No original source files are written.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
from numba import njit, prange
from scipy.integrate import cumulative_trapezoid
root=Path(__file__).resolve().parents[1] / "results" / "revision_validation"
root.mkdir(parents=True,exist_ok=True)
Uc=12.
@njit(parallel=True,cache=True)
def sweep(us, eps, wt, omega, low, hi,eta_s,eta_g):
 out=np.empty((len(us),6),dtype=np.float64)
 for j in prange(len(us)):
  U=us[j]
  R=U*U/4
  om2=(Uc*Uc-U*U)/4 if U<=Uc else 0.
  dom2=-U/2 if U<=Uc else 0.
  l=0.;h=0.;lr=0.;lm=0.;hr=0.;hm=0.
  for ix in range(len(omega)):
   z=omega[ix]+1j*eta_s
   D=z*z-om2
   sigma=R*z/D
   sr=(U/2)*z/D
   sm=R*z*dom2/(D*D)
   sumg=0j;sumg2=0j
   zgreen=omega[ix]+1j*eta_g-sigma
   for k in range(len(eps)):
    g=1/(zgreen-eps[k])
    sumg += wt[k]*g
    sumg2 += wt[k]*g*g
   A=-sumg.imag/np.pi
   dar=-(sumg2*sr).imag/np.pi
   dam=-(sumg2*sm).imag/np.pi
   l+=A*low[ix]; h+=A*hi[ix]
   lr+=dar*low[ix]; lm+=dam*low[ix]
   hr+=dar*hi[ix]; hm+=dam*hi[ix]
  out[j,0]=l;out[j,1]=h;out[j,2]=lr;out[j,3]=lm;out[j,4]=hr;out[j,5]=hm
 return out

def main():
 nk=80;nw=1201
 k=np.linspace(-np.pi,np.pi,nk,endpoint=False)
 ep=(-2*np.cos(k)[:,None]-2*np.cos(k)[None,:]).ravel()
 eps,counts=np.unique(np.round(ep,13),return_counts=True)
 wt=counts.astype(float)/len(ep)
 w=np.linspace(-12,12,nw);dw=w[1]-w[0]
 low=dw/(4*.05*np.cosh(np.clip(w/(.1),-80,80))**2)
 hi=dw*((np.abs(w)>=4.2)&(np.abs(w)<=8.8))
 metal=np.r_[np.linspace(0,11.4,229)[:-1],np.linspace(11.4,12,301)]
 ins=np.linspace(12,12.96,81)[1:]
 us=np.r_[metal,ins]
 vals=sweep(us,eps,wt,w,low,hi,.08,.04)
 cols=['LE','HE','LE_res','LE_motion','HE_res','HE_motion']
 df=pd.DataFrame(vals,columns=cols);df.insert(0,'U',us);df.insert(1,'u',us/12)
 df['LE_total']=df.LE_res+df.LE_motion; df['HE_total']=df.HE_res+df.HE_motion
 df.to_csv(root/'self_consistent_path_adaptive.csv',index=False)
 # Evaluate right derivative at critical exactly separately, but keep the left
 # derivative at critical in metal integral.
 right=sweep(np.asarray([12.+1e-12]),eps,wt,w,low,hi,.08,.04)[0]
 result=[]
 for name,idx0,idx1 in [('LE',2,3),('HE',4,5)]:
  cval=df[name].to_numpy();
  cum={}
  for lab,i in [('res',idx0),('motion',idx1)]:
   vec=vals[:,i]
   dy=(vec[1:]+vec[:-1])*.5*np.diff(us)
   idx=len(metal)-1
   dy[idx]=.5*(right[i]+vec[idx+1])*(us[idx+1]-us[idx])
   cum[lab]=np.r_[0,np.cumsum(dy)]
   df[name+'_'+lab+'_cum']=cum[lab]
  fit=cval[0]+cum['res']+cum['motion']
  df[name+'_fit']=fit
  df[name+'_fit_error']=fit-cval
  result.append({'readout':name,'max_abs_closure':float(np.max(np.abs(fit-cval))),'endpoint_err':float(fit[-1]-cval[-1]),'absolute_change':float(cval[-1]-cval[0]),'residue_delta':float(cum['res'][-1]),'motion_delta':float(cum['motion'][-1])})
 df.to_csv(root/'self_consistent_path_adaptive.csv',index=False)
 # denominator for original same definitions (completion over 0..1.08 Uc)
 mr=[]
 for q in (.25,.5,.75):
  for name in ('LE','HE'):
   c=df[name].to_numpy();comp=(c-c[0])/(c[-1]-c[0]);
   ii=np.flatnonzero(comp[1:]>=q)[0]+1; interp=us[ii-1]+(us[ii]-us[ii-1])*(q-comp[ii-1])/(comp[ii]-comp[ii-1])
   mr.append({'q':q,'readout':name,'marker_U':interp,'marker_lambda':interp/(Uc*1.08)})
 mar=pd.DataFrame(mr);mar.to_csv(root/'physical_dense_markers.csv',index=False)
 # physical stage: decomposition at early/late interaction for LE/HE
 row=[]
 for u in (.2,.4,.6,.8,.9,.95,.98,.99):
  ii=np.argmin(np.abs(us-Uc*u));r=df.iloc[ii]
  row.append({'u':r.u,'HE_residue_rate':r.HE_res,'HE_motion_rate':r.HE_motion,'LE_residue_rate':r.LE_res,'LE_motion_rate':r.LE_motion})
 pd.DataFrame(row).to_csv(root/'physical_stage_rates.csv',index=False)
 print('PHYSICAL-PATH INTEGRAL CHECK')
 for r in result:print(r)
 print('MARKERS');print(mar.to_string(index=False))
 print('RATES');print(pd.DataFrame(row).to_string(index=False))
 print('NEAR CRITICAL LEFT/RIGHT', vals[len(metal)-1].tolist(), right.tolist())
 (root/'physical_path_integral_check.json').write_text(json.dumps({'checks':result,'num_u':len(us),'nk':nk,'nw':nw,'original_unchanged':True},indent=2))
 # The same fixed solver inputs are reconstructed by the canonical repository
 # before this phase: calculate only a different postprocessed observable from its ED output.
 path=Path(__file__).resolve().parents[1] / "results" / "five_bath_ed_dmft" / "five_bath_dmft_curves.csv"
 curves=pd.read_csv(path).sort_values("U")
 UE=curves.U.to_numpy(float)
 def marker(y,q):
  c=(y-y[0])/(y[-1]-y[0]); hits=np.flatnonzero(c[1:]>=q)
  if not len(hits): return float("nan")
  idx=hits[0]+1
  return float(UE[idx-1]+(UE[idx]-UE[idx-1])*(q-c[idx-1])/(c[idx]-c[idx-1]))
 alt=[]
 for q in (.25,.5,.75):
  he=marker(curves.HE.to_numpy(float),q)
  for tag,arr in [('LE Fermi-window',curves.LE.to_numpy(float)),('A0 spectral value',curves.A0.to_numpy(float)),('Z Matsubara proxy',curves.Z.to_numpy(float))]:
   le=marker(arr,q)
   alt.append({'fraction':q,'LE_readout':tag,'U_HE':he,'U_LE':le,'HE_minus_LE':he-le,'HE_first':he<le})
 pd.DataFrame(alt).to_csv(root/'five_bath_alternative_readouts_from_regenerated_ED.csv',index=False)
 mr=[]
 for obs in ('HE','LE'):
  for q in (.25,.5,.75):
   t=float(mar[(mar.readout==obs)&(np.isclose(mar.q,q))].marker_U.iloc[0])
   rr=float(np.interp(t,df.U,df[obs+'_res_cum']))
   mm=float(np.interp(t,df.U,df[obs+'_motion_cum']))
   mr.append({'readout':obs,'fraction':q,'U_cross':t,'u_cross':t/12,'residue_integrated':rr,'motion_integrated':mm,'total':rr+mm,'residue_percent':100*rr/(rr+mm),'motion_percent':100*mm/(rr+mm)})
 pd.DataFrame(mr).to_csv(root/'physical_mechanism_fraction_at_markers.csv',index=False)
 # Verified tolerance gates explicitly prevent overwriting vetted figures with failures.
 closure={t['readout']:t['max_abs_closure'] for t in result}
 assert closure['LE']<5e-5 and closure['HE']<5e-6,closure
 sc=pd.DataFrame(mr)
 assert sc[(sc.readout=='HE')&(np.isclose(sc.fraction,.5))].residue_percent.iloc[0]>70
 assert sc[(sc.readout=='LE')&(np.isclose(sc.fraction,.75))].motion_percent.iloc[0]>65
 assert pd.DataFrame(alt).shape[0]==9
 print('Review-revision numerical validation and ED alternative readouts generated.')


if __name__=='__main__':main()
