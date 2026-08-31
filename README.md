# itg2d-flr-fluid

[![DOI](https://zenodo.org/badge/1347527530.svg)](https://doi.org/10.5281/zenodo.22208948)

Pseudospectral solver for 2D fluid ITG with FLR nonlinearities.

Initialize the submodules with:

```bash
git submodule update --init --recursive
```

The code requires a CUDA-capable GPU, CuPy, and a CUDA-enabled PyTorch installation. The growth-rate calculation uses PyTorch on the GPU.

The [`requirements.txt`](requirements.txt) file contains the Python dependencies. The versions may need to be adjusted for different environments. Install them with:

```bash
python -m pip install -r requirements.txt
```

Run the simulations with:

```bash
python itg2d.py
python itg2d_hyper_etdrk4.py
python itg2d_PV_hyper_etdrk4.py
```

[`itg2d.py`](itg2d.py) solves the system with Laplacian viscosity using a CuPy implementation of SciPy's `DOP853` solver.

[`itg2d_hyper_etdrk4.py`](itg2d_hyper_etdrk4.py) and [`itg2d_PV_hyper_etdrk4.py`](itg2d_PV_hyper_etdrk4.py) solve the system with hyperviscosity using a CuPy implementation of the [etdrk4 scheme](https://doi.org/10.1137/S1064827502410633).

## Parameters

Simulation parameters are saved in the `params` group while `kx`, `ky`, `t0` and `t1` are saved in the `data` group.

`Npx`, `Npy` are the padded resolutions.

`Lx`, `Ly` are the box sizes normalized by the ion gyroradius.

`kapn` and `kapt` are the inverse density and temperature gradient length scales normalized by the ion gyroradius.

`kapb` is twice the magnetic field gradient length scale normalized by the ion gyroradius. Twice because it also includes the curvature, which equals the grad-B term in the electrostatic limit.

`n_hyper` is the hyperviscosity order. Thus, `n_hyper=3` corresponds to $\nabla^6$.

`Gamma` is the specific heat ratio. This is used only in [`itg2d_PV_hyper_etdrk4.py`](itg2d_PV_hyper_etdrk4.py).

`kx` and `ky` are the radial and poloidal (binormal) wavenumbers in the $k_y \geq 0$ domain, excluding $(k_x, k_y) = (0, 0)$. These are generated using the [`mlsarray`](mlsarray) submodule.

`nu` is the viscosity coefficient in [`itg2d.py`](itg2d.py) and the hyperviscosity coefficient in the other Python files.

`H` is the hypo-viscosity coefficient.

`gammax` is the maximum linear growth rate of the system. It is computed using the `gam_max` functions in [`modules/gamma.py`](modules/gamma.py#L31) and [`modules/gamma_PV.py`](modules/gamma_PV.py#L31), which use PyTorch to calculate the growth rates of the linearized system.

`dtstep` is the time step for the solver.

`dtshowcb` is the callback interval for the function that prints the progress of the solver.

`dtsavecb` is the callback interval for the functions that save the Fourier components of the vorticity and pressure, the radial zonal profiles, and the radial profile of fluxes.

`t0` is the initial time and `t1` is the time up to which the simulation is executed.

`wecontinue`, when set to `True`, allows the user to resume a simulation from where it was left off.

`fname` is the name of the HDF5 file that by default includes kapt, nu and H.

## Output

The `data` group stores `kx`, `ky`, `t0`, and `t1`. The `params` group stores the simulation parameters. The `fields`, `zonal`, and `fluxes` groups store diagnostic data, while `last` stores the latest state of the simulation for restarting.
