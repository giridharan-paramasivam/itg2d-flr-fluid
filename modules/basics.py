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