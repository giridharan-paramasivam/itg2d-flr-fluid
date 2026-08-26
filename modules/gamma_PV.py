import numpy as np
import cupy as cp
import torch

def init_linmats(kx,ky,pars,n_hyper=1):
    # Initializing the linear matrices
    kapn,kapt,kapb,Gamma,D,H = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','Gamma','D','H']
    ]
    kpsq = kx**2 + ky**2

    sigk = ky>0
    Lk=sigk+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,0,0]=-Gamma*kapb*ky-1j*D*kpsq**n_hyper-1j*sigk*H/kpsq**2
    lm[:,0,1]=(kapn+kapt)*ky-Gamma*(kapn+kapt)*ky*kpsq
    lm[:,1,0]=-kapb*ky/Lk
    lm[:,1,1]=(kapn*ky-(kapn+kapt)*ky*kpsq)/Lk-1j*D*kpsq**n_hyper-1j*sigk*H/kpsq**2

    return lm

def linfreq(kx, ky, pars, n_hyper=1):
    lm = init_linmats(torch.from_numpy(kx), torch.from_numpy(ky), pars, n_hyper=n_hyper).cuda()
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

def gam_max(kx, ky, kapn, kapt, kapb, Gamma, D, H, n_hyper=1):
    if isinstance(ky, cp.ndarray):
        kx = kx.get()
        ky = ky.get()

    base_pars={'kapn':kapn,
        'kapt':kapt,
        'kapb':kapb,
        'Gamma':Gamma,
        'D':D,
        'H':H}

    om=linfreq(kx,ky,base_pars, n_hyper=n_hyper)
    gam=om.imag[:,0]
    return np.max(gam)
