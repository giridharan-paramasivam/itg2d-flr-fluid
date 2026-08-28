#%% Import libraries

import os
import numpy as np
import cupy as cp
import h5py as h5
from mlsarray.mlsarray import slicelist, init_kgrid
from mlsarray.mlsarray import irft2 as original_irft2, rft2 as original_rft2
from etdrk4cp.gsol import gsol, callbacks
from etdrk4cp.h5tools import save_data
from modules.gamma import gam_max   
from modules.basics import format_exp, round_to_nsig
from functools import partial
from time import time

#%% Parameters

Npx,Npy=1024,1024
Lx,Ly=64*np.pi,64*np.pi
kapt=2.0
kapn=0.2
kapb=0.02
n_hyper=3 #k^6

Nx,Ny=2*(Npx//3),2*(Npy//3)
sl=slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Ny/2)]
kx,ky=init_kgrid(sl,Lx,Ly)
kpsq=kx**2+ky**2
Nk=kx.size
dk=float(ky[0])
sigk=cp.sign(cp.abs(ky))
Lk=sigk+kpsq

nu=1e-1
H=4e-5 #1e-5,4e-5

dtshow=1.0
gammax=gam_max(kx,ky,kapn,kapt,kapb,nu,H,n_hyper)
dtstep,dtsavecb=round_to_nsig((512/Npx)*0.002/gammax,1),round_to_nsig(0.02/gammax,1)
t0,t1=0.0,round(50/gammax,0) 
wecontinue=False

output_dir = f"data/{Npx}/"
os.makedirs(output_dir, exist_ok=True)
fname = output_dir + f'out_hyper_kapt_{str(kapt).replace(".", "_")}_nu_{format_exp(nu)}_H_{format_exp(H)}.h5'
if not os.path.exists(fname):
    wecontinue=False

#%% Functions

irft2 = partial(original_irft2,sl=sl)
rft2 = partial(original_rft2,sl=sl)

def init_fields(kx,ky,w=10.0,A=1e-6):
    Phik=A*cp.exp(-kx**2/2/w**2-ky**2/2/w**2)*cp.exp(1j*2*np.pi*cp.random.rand(kx.size).reshape(kx.shape))
    Pk=A*cp.exp(-kx**2/2/w**2-ky**2/2/w**2)*cp.exp(1j*2*np.pi*cp.random.rand(kx.size).reshape(kx.shape))

    Phik[slbar]=0
    Pk[slbar]=0
    zk = cp.hstack((Phik, Pk))
    return zk

def fsavecb(t,y,flag):
    Phik = y[0,:]
    Pk = y[1,:]
    Omk = -kpsq*Phik
    vy = irft2(1j*kx*Phik) 
    Om = irft2(Omk)
    P = irft2(Pk)
    if flag=='fields':
        save_data(fl,'fields',ext_flag=True,Omk=Omk.get(),Pk=Pk.get(),t=t)
    elif flag=='zonal':
        vbar = cp.mean(vy,1)
        Ombar = cp.mean(Om,1)
        Pbar = cp.mean(P,1)
        save_data(fl,'zonal',ext_flag=True,vbar=vbar.get(),Ombar=Ombar.get(),Pbar=Pbar.get(),t=t)
    elif flag=='fluxes':
        vx = irft2(-1j*ky*Phik) #ExB flow: x comp
        wx = irft2(-1j*ky*Pk) #diamagnetic flow: x comp
        Q = cp.mean(P*vx,1)
        Qbox = cp.mean(Q)
        Rphi = cp.mean(vy*vx,1)
        Rd = cp.mean(vy*wx,1)
        save_data(fl,'fluxes',ext_flag=True,Q=Q.get(),Qbox=Qbox.get(),Rphi=Rphi.get(),Rd=Rd.get(),t=t)
    save_data(fl,'last',ext_flag=False,zk=y.get(),t=t,dt=r.hlast,tnexts=r.cbs.tnexts)

def fshowcb(t,y):
    Phik = y[0,:]
    Pk = y[1,:]
    vx = irft2(-1j*ky*Phik)
    P = irft2(Pk)
    Qbox = cp.mean(P*vx)
    Ktot = np.sum(kpsq*np.abs(Phik)**2)
    Kbar = np.sum((kx[slbar]*np.abs(Phik[slbar]))**2)
    print('t=',round(t,3),', ',round_to_nsig(time()-ct, 3),' secs elapsed.', end=' ')
    print(f'Kbar/Ktot={Kbar/Ktot*100:.3g}%, Qbox={Qbox.get():.3g}')

lm=np.zeros(kx.shape+(2,2),dtype=complex)

lm[:,0,0]=((-kapn+(kapn+kapt)*kpsq)*1j*ky/Lk-nu*kpsq**n_hyper-sigk*H/kpsq**2).get()
lm[:,0,1]=(kapb*1j*ky/Lk).get()
lm[:,1,0]=(-(kapn+kapt)*1j*ky).get()
lm[:,1,1]=(-nu*kpsq**n_hyper-sigk*H/kpsq**2).get()

def rhs_nl(t,y):
    # y has shape (2, Nk)
    dzkdt = cp.zeros_like(y)
    Phik = y[0,:]
    Pk = y[1,:]
    dPhikdt = dzkdt[0,:]
    dPkdt = dzkdt[1,:]
    
    dxphi = irft2(1j*kx*Phik)
    dyphi = irft2(1j*ky*Phik)
    dxP = irft2(1j*kx*Pk)
    dyP = irft2(1j*ky*Pk)
    nOmg = irft2(Lk*Phik)

    dPhikdt[:] += (1j*kx*rft2(dyphi*nOmg) - 1j*ky*rft2(dxphi*nOmg))/ Lk
    dPhikdt[:] += (kx**2*rft2(dxphi*dyP) - ky**2*rft2(dyphi*dxP) + kx*ky*rft2(dyphi*dyP - dxphi*dxP)) / Lk # dimagnetic nonlinearity

    dPkdt[:] += rft2(dyphi*dxP - dxphi*dyP)
    
    return dzkdt

#%% Run the simulation    

print(f'nu={nu}, kapn={kapn}, kapt={kapt}, kapb={kapb}')

if(wecontinue):
    fl=h5.File(fname,'r+',libver='latest')
    fl.swmr_mode = True
    zk = cp.array(fl['last/zk'][()])
    t0 = fl['last/t'][()]
    dtstep = fl['last/dt'][()]
    tnexts = fl['last/tnexts'][()]
else:
    fl = h5.File(fname,'w',libver='latest')
    fl.swmr_mode = True
    zk = init_fields(kx,ky)
    tnexts = None
    save_data(fl,'data',ext_flag=False,kx=kx.get(),ky=ky.get(),t0=t0,t1=t1)
    save_data(fl,'params',ext_flag=False,Npx=Npx,Npy=Npy,Lx=Lx,Ly=Ly,kapn=kapn,kapt=kapt,kapb=kapb,nu=nu,H=H,n_hyper=n_hyper,gammax=gammax)

ct=time()
fcbs=[fshowcb,partial(fsavecb,flag='fields'), partial(fsavecb,flag='zonal'), partial(fsavecb,flag='fluxes')]
dtcbs=[dtshow,10*dtsavecb,dtsavecb,dtsavecb]
cbs=callbacks(dtcbs,fcbs,tnexts)
r=gsol(rhs_nl, t0, zk, t1, lm, dtstep, callbacks=cbs, maxstep=1.0, tol=1e-8)
r.run()
fl.close()

