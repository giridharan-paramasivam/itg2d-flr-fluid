import numpy as np

def format_exp(d):
    """Formats a number in scientific notation for use in filenames."""
    dstr = f"{d:.1e}"
    base, exp = dstr.split("e")
    base = base.replace(".", "_")
    if "-" in exp:
        exp = exp.replace("-", "")
        prefix = "em"
    else:
        prefix = "e"
    exp = str(int(exp))
    return f"{base}_{prefix}{exp}"

def round_to_nsig(number, n):
    """Rounds a number to n significant figures."""
    if not np.isfinite(number): # Catches NaN, Inf, -Inf
        return number 
    if number == 0:
        return 0.0
    if n <= 0:
        raise ValueError("Number of significant figures (n) must be positive.")
    
    order_of_magnitude = np.floor(np.log10(np.abs(number)))
    decimals_to_round = int(n - 1 - order_of_magnitude)
    
    return np.round(number, decimals=decimals_to_round)

def irft2_g(uk, Nx, Npx, Npy):
    """Inverse real 2D Fourier transform with specific grid handling."""
    Nxh = int(Nx/2)
    u = np.zeros((Npx, int(Npy/2)+1), dtype=complex)
    u[:Nxh,:Nxh] = uk[:Nxh,:Nxh]
    u[-Nxh+1:,:Nxh] = uk[-Nxh+1:,:Nxh]
    return np.fft.irfft2(u, norm='forward')

def rft2_g(u, Nx, Ny):
    """Real 2D Fourier transform with specific grid handling."""
    Nxh = int(Nx/2)
    uk = np.zeros((Nx, int(Ny/2)+1), dtype=complex)
    yk = np.fft.rfft2(u, norm='forward')
    uk[:Nxh,:-1] = yk[:Nxh,:int(Ny/2)]
    uk[-1:-Nxh:-1,:-1] = yk[-1:-Nxh:-1,:int(Ny/2)]
    uk[0,0] = 0
    return uk

def irft_g(vk, Npx):
    """Inverse real 1D Fourier transform with specific grid handling."""
    Nxh = int(Npx/2)
    v = np.zeros(int(Npx/2)+1, dtype=complex)
    v[:Nxh] = vk[:Nxh]
    return np.fft.irfft(v, norm='forward')

def ubar(uk, Npx, Nx, Ny):
    """Returns the zonal mean of uk."""
    slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
    Nxh = int(Nx/2)
    vk = np.zeros(int(Npx/2)+1, dtype=complex)
    vk[1:Nxh] = uk[slbar]
    return np.fft.irfft(vk, norm='forward')
