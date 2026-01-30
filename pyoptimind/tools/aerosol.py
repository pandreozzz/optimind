import numpy as np
import xarray as xr

############################
# AEROSPECS CLASS
############################

from dataclasses import dataclass, field
@dataclass
class IFSAeroSpecs:
    name   : str
    median : np.ndarray = field(default_factory=lambda: np.ndarray(np.nan)) # median or geometric mean median = exp(mu) (rmod in IFS doc)
    shape  : np.ndarray = field(default_factory=lambda: np.ndarray(np.nan)) # geometric std shape = exp(sigma) (sigma in IFS doc)
    density: float = 1
    rmin   : float = 0.
    rmax   : float = np.inf
    fact   : np.ndarray = field(default_factory=lambda: np.ndarray(np.nan))  # for multimodal distributions fraction per each mode
    m_act  : float = 0. # activated mass fraction for hydrophilic specs
    nmodes : int = 1
    def __post_init__(self):
        for name, field_type in self.__annotations__.items():
            this_val = getattr(self, name)

            # make immutable
            if isinstance(this_val, list):
                setattr(self, name, tuple(this_val))
                this_val = getattr(self, name)
                
            if name in ["median", "shape", "fact"]:
                setattr(self, name, np.array(this_val, dtype=float))

            if name in ["density", "rmin", "rmax", "m_act"]:
                setattr(self, name, float(this_val))
                
            if not isinstance(getattr(self, name), field_type):
                raise TypeError(f"The field `{name}` was assigned by `{type(getattr(self, name))}` instead of `{field_type}`")
            
def preprocess_params(r_med, shape, mode_factors):
    # Force to be a rank > 0 array
    if np.array(r_med).shape == ():
        r_med_out = np.array([r_med]).copy()
    else:
        r_med_out = np.array(r_med).copy()
        
    if np.array(shape).shape == ():
        s_out = np.array([shape]).copy()
    else:
        s_out = np.array(shape).copy()

    n_modes = len(r_med_out)
    if len(s_out) != n_modes:
        raise ValueError(f"r_med({len(r_med_out)} and s({len(s_out)}) must have the same length!!")
    
    if mode_factors is None:
        mode_factors_out = np.ones((n_modes))/n_modes
    elif np.array(mode_factors).shape == ():
        mode_factors_out = np.array([mode_factors]).copy()
    else:
        mode_factors_out = np.array(mode_factors).copy()
        
    if len(mode_factors_out) != n_modes:
        raise ValueError(f"mode_factors({len(mode_factors_out)}) must be of length {n_modes}!!")

    if not np.isclose(mode_factors_out.sum(), 1.):
        print(f"Warning!! mode_factors sums to {mode_factors_out.sum()} != 1!")

    return r_med_out, s_out, mode_factors_out

def get_nccn_over_mcon(dens, rmed_in, shape_in, mode_factors_in = None,
                        rmin_n : float = 0, rmax_n : float = np.inf,
                        rmin_m : float = 0, rmax_m : float = np.inf):
    """
    dens : density of the aerosol species (Kg/m^3)
    rmed : mode radius of log-normal distribution
    shape: geometric std of log-normal distribution

    n_factors : number-proportion between modes if multi-modal
    
    returns
    -------
    nd  : number concentration per mass (cm^-3 Kg-1 m3)
    """

    r_med, shape, mode_factors = preprocess_params(rmed_in, shape_in, mode_factors_in)
    dens = np.array(dens)
    
    # integrate
    if rmin_n > 0 or rmax_n < np.inf:
        numer = lognorm_cdf(rmax_n, r_med, shape, mode_factors) -\
        lognorm_cdf(rmin_n, r_med, shape, mode_factors)
    else: # normalized
        numer = 1
        
    if rmin_m > 0 or rmax_m < np.inf:
        from scipy.integrate import quad
        integrals = lognorm_x3_cdf(rmax_m, r_med, shape, mode_factors) -\
        lognorm_x3_cdf(rmin_m, r_med, shape, mode_factors)
    else:
        # analytical formulas for statistical momenta
        integrals = np.array([np.exp(3*np.log(r)+9/2*np.log(s)**2)*f
                              for r,s,f in zip(r_med, shape, mode_factors)]).sum()
                             

    denom = (4/3*np.pi*dens)*integrals  
    
    return 1.e12*numer/denom

def lognorm_pdf(x_in, r_med_in, shape_in, mode_factors_in=None):
    """
    Normalized log-normal PDF
    """
    
    r_med, shape, mode_factors = preprocess_params(r_med_in, shape_in, mode_factors_in)
    x = np.array(x_in)
    si = np.log(shape) # sigma parameter
        
    if x.shape != () and len(x) > 1:
        x = x[None,:]
        r_med = r_med[:,None]
        si = si[:,None]
        mode_factors = mode_factors[:,None]

    with np.errstate(divide='ignore'):
        x_arg = x/r_med

    norm_fact = 1/(np.sqrt(2*np.pi)*si*x)

    return (norm_fact*np.exp(-np.log(x_arg)**2/(2*si**2))*mode_factors).sum(axis=0)

def lognorm_x3_pdf(x, r_med, shape, mode_factors=None):
    """
    x^3 n(x) where n is normalized lognormal PDF
    mode_factors are expressed for the modes of n(x)
    """
    return x**3*lognorm_pdf(x, r_med, shape, mode_factors)

def lognorm_cdf(x_in, r_med_in, shape_in, mode_factors_in=None):
    """
    int n(x) where n(x) is normalized lognormal PDF
    """
    from scipy.special import erf

    r_med, shape, mode_factors = preprocess_params(r_med_in, shape_in, mode_factors_in)
    x = np.array(x_in)
    si = np.log(shape) # sigma parameter
    
    if x.shape != () and len(x) > 1:
        x = x[None,:]
        r_med = r_med[:,None]
        si = si[:,None]
        mode_factors = mode_factors[:,None]

    with np.errstate(divide='ignore'):
        x_arg = x/r_med
        
    return (0.5*(1+ erf(np.log(x_arg)/(si*np.sqrt(2))))*mode_factors).sum(axis=0)

def lognorm_x3_cdf(x_in, r_med, shape, mode_factors=None):
    """
    Returns int x^3 n(x) where n(x) is normalized lognormal PDF
    mode_factors are expressed for the modes of n(x)
    """


    def go_integral(xsup):
        from scipy.integrate import quad
        return quad(lognorm_x3_pdf, 0, xsup, args=(r_med, shape, mode_factors))[0]
        

    x = np.array(x_in)
    if x.shape != () and len(x) > 1:
        result = np.array([go_integral(xsup) for xsup in x])
    else:
        result = go_integral(x)
    
        
    return result

def get_nccn_over_mcon_from_specs(species : IFSAeroSpecs,
                                  rmin_n : float = 0, rmax_n : float = np.inf,
                                  rmin_m : float = 0, rmax_m : float = np.inf
                                 ):

    rmin_n = np.clip(rmin_n, species.rmin, species.rmax)
    rmax_n = np.clip(rmax_n, species.rmin, species.rmax)
    rmin_m = np.clip(rmin_m, species.rmin, species.rmax)
    rmax_m = np.clip(rmax_m, species.rmin, species.rmax)

    if (rmin_n == rmax_n):
        return 0
    if (rmin_m == rmax_m):
        return np.inf
    
    return get_nccn_over_mcon(species.density, species.median, species.shape, species.fact,
                              rmin_n, rmax_n,
                              rmin_m, rmax_m
                             )           

def get_nccn_over_mcon_from_speclist(specs : dict) -> dict:
    """
    Docstring for get_nccn_over_mcon_from_speclist
    
    :param specs: Description
    :return: Description
    :rtype: dict[Any, Any]
    """

    return {name : get_nccn_over_mcon_from_specs(aerospec)
            for name,aerospec in specs.items()}

def compute_ccn_ifs(ws : xr.DataArray, lsm : xr.DataArray):
    """
    Args:
        Ws : xr.DataArray - absolute 10m wind speed 
        lsm : xr.DataArray - land-sea mask
    Returns:
        nd: same type as Ws/lsm - ccn/nd from ifs parametrization
    """

    landmask = lsm > 0.5
    wind_lt15 = ws < 15
    
    a_par = np.where(wind_lt15, 0.16, 0.13)
    b_par = np.where(wind_lt15, 1.45, 1.89)

    qa = np.exp(a_par*ws+b_par).clip(-np.inf, 327)
    
    c_par = xr.where(landmask, 2.21, 1.2)
    d_par = xr.where(landmask, 0.3,  0.5)



    na = 10**(c_par + d_par*np.log10(qa))
    
    nd = xr.where(landmask,
                  -2.10e-4*na**2 + 0.568*na - 27.9,
                  -1.15e-3*na**2 + 0.963*na + 5.30
                 )

    return nd