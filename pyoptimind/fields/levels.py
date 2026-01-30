import numpy as np
import xarray as xr

R_DRYAIR = 287.05 # J/Kg K
def populate_mlfields(ds, keep_p_half : bool = False):
    """
    Compute 3D meteorological fields from hybrid coordinates.
    
    Calculates pressure, half-level pressure, air density, and layer pressure
    difference from IFS hybrid coordinate coefficients and surface pressure.
    
    Args:
        ds: xarray Dataset - Must contain sp, hybm, hyam, hybi, hyai, t
        keep_p_half: should p_half (needed for computing p at full levels) be kept?
    Returns:
        None (modifies dataset in place)
    
    Side Effects:
        Adds variables to dataset: p, p_half, rho_air, dp
    """
    
    minlev = int(ds.lev.min().values.item())
    maxlev = int(ds.lev.max().values.item())

    print(f"Computing 3D pressure, rho_air, and level pressure difference. minlev: {minlev}, maxlev: {maxlev}", flush=True)
    fullplevslice = slice(minlev-1, maxlev)
    halfplevslice = slice(minlev-1, maxlev+1)
    
    ds["p"] = (ds["sp"]*ds["hybm"].astype(np.float32).isel(nhym=fullplevslice) +
              ds["hyam"].astype(np.float32).isel(nhym=fullplevslice)).rename(nhym="lev")
    print("p done")
    
    ds["p_half"] = (ds["sp"]*ds["hybi"].astype(np.float32).isel(nhyi=halfplevslice) +
                   ds["hyai"].astype(np.float32).isel(nhyi=halfplevslice)).rename(nhyi="half_lev").assign_coords(
                   half_lev=np.arange(minlev, maxlev+2, dtype=float))
    print("p_half done", flush=True)
    
    ds["rho_air"] = ds["p"] / (np.float32(R_DRYAIR) * ds["t"])
    print("rho_air done", flush=True)
    
    ds["dp"] = ds["p_half"].drop_vars('half_lev').diff(dim="half_lev").rename(half_lev="lev")
    print("dp done", flush=True)
    
    if not keep_p_half:
        ds["p_half"] = None