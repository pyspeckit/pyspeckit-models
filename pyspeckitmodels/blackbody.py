"""
============================
Simple black-body calculator
============================

Includes both wavelength and frequency blackbody functions.  Has flexible
units.  Also allows for a few varieties of modified blackbody.

"""
try:
    from numpy import exp
except ImportError:
    from math import exp

from pyspeckit.spectrum.units import frequency_dict,unitdict,wavelength_dict

def blackbody(nu,temperature, omega=1.0, units='cgs',frequency_units='Hz', normalize=max):
    # load constants in desired units
    h,k,c = unitdict[units]['h'],unitdict[units]['k'],unitdict[units]['c']

    # convert nu to Hz
    nu = nu * frequency_dict[frequency_units]

    I = 2*h*nu**3 / c**2 * (exp(h*nu/(k*temperature)) - 1)**-1

    if normalize and hasattr(I,'__len__'):
        if len(I) > 1:
            return I/normalize(I) * omega
        else:
            return I * omega
    else:
        return I * omega

def blackbody_wavelength(lam,temperature, omega=1.0, units='cgs',wavelength_units='Angstroms', normalize=max):
    """

    ergs / s / cm^2 / A / sr
    """
    # load constants in desired units
    h,k,c = unitdict[units]['h'],unitdict[units]['k'],unitdict[units]['c']

    # convert nu to Hz
    lam = lam * wavelength_dict[wavelength_units] / (1e-2 if units=='cgs' else 1)

    I = 2*h*c**2 / lam**5 * (exp(h*c/(k*temperature*lam)) - 1)**-1

    if normalize and hasattr(I,'__len__'):
        if len(I) > 1:
            return I/normalize(I) * omega
        else:
            return I * omega
    else:
        return I * omega

def modified_blackbody(nu,temperature,beta=1.75, logscale=0.0, logN=22,
        muh2=2.8, units='cgs',frequency_units='Hz', kappa0=4.0, nu0=505e9,
        normalize=max):
    """
    Snu =  2hnu^3 c^-2  (e^(hnu/kT) - 1)^-1  (1 - e^(-tau_nu) )
    Kappa0 and Nu0 are set as per http://arxiv.org/abs/1101.4654 which uses OH94 values.
    beta = 1.75 is a reasonable default for Herschel data
    N = 1e22 is the column density in cm^-2

    nu0 and nu must have same units!
    """
    h,k,c = unitdict[units]['h'],unitdict[units]['k'],unitdict[units]['c']
    mh = unitdict[units]['mh']

    kappanu = kappa0 * (nu/nu0)**beta
    # numpy apparently can't multiply floats and longs
    tau  = muh2 * mh * kappanu * 10.0**logN

    modification = (1.0 - exp(-1.0 * tau))

    I = blackbody(nu,temperature,units=units,frequency_units=frequency_units,normalize=normalize)*modification

    if normalize and hasattr(I,'__len__'):
        if len(I) > 1:
            return I/normalize(I) * 10.**logscale
        else:
            return I * 10.**logscale
    else:
        return I * 10.**logscale

def greybody(nu, temperature, beta, A=1.0, logscale=0.0,
        units='cgs', frequency_units='Hz', 
        kappa0=4.0, nu0=3000e9, normalize=max):
    h,k,c = unitdict[units]['h'],unitdict[units]['k'],unitdict[units]['c']
    mh = unitdict[units]['mh']

    modification = (1. - exp(-(nu/nu0)**beta))
    I = blackbody(nu,temperature,units=units,frequency_units=frequency_units,normalize=normalize)*modification

    if normalize and hasattr(I,'__len__'):
        if len(I) > 1:
            return I/normalize(I) * 10.**logscale
        else:
            return I * 10.**logscale
    else:
        return I * 10.**logscale

def modified_blackbody_wavelength(lam, temperature, beta=1.75, logscale=0.0,
        logN=22, muh2=2.8, units='cgs', wavelength_units='Angstroms',
        kappa0=4.0, nu0=3000e9, normalize=max):
    """
    Snu =  2hnu^3 c^-2  (e^(hnu/kT) - 1)^-1  (1 - e^(-tau_nu) )
    Kappa0 and Nu0 are set as per http://arxiv.org/abs/1101.4654 which uses OH94 values.
    beta = 1.75 is a reasonable default for Herschel data
    N = 1e22 is the column density in cm^-2

    nu0 and nu must have same units!
    """
    h,k,c = unitdict[units]['h'],unitdict[units]['k'],unitdict[units]['c']
    mh = unitdict[units]['mh']

    nu = c/(lam*wavelength_dict[wavelength_units]/wavelength_dict[unitdict[units]['length']])
    kappanu = kappa0 * (nu/nu0)**beta
    tau  = muh2 * mh * kappanu * 10.**logN

    modification = (1.0 - exp(-1.0 * tau))

    I = blackbody(nu,temperature,units=units,frequency_units='Hz',normalize=normalize)*modification

    if normalize and hasattr(I,'__len__'):
        if len(I) > 1:
            return I/normalize(I) * 10.**logscale
        else:
            return I * 10.**logscale
    else:
        return I * 10**logscale


try:
    import pyspeckit.mpfit as mpfit

    def fit_blackbody(xdata, flux, guesses=(0,0), err=None, blackbody_function=blackbody, quiet=True, **kwargs):
        """
        guesses = Temperature, Arbitrary Scale
        OR Temperature, Beta, Arbitrary Scale
        """

        def mpfitfun(x,y,err):
            if err is None:
                def f(p,fjac=None): return [0,(y-blackbody_function(x, *p, normalize=False, **kwargs))]
            else:
                def f(p,fjac=None): return [0,(y-blackbody_function(x, *p, normalize=False, **kwargs))/err]
            return f

        err = err if err is not None else flux*0.0 + 1.0

        mp = mpfit.mpfit(mpfitfun(xdata,flux,err), guesses, quiet=quiet)

        return mp

    def bb(xarr, temperature, omega, units='cgs'):
        """
        Blackbody model assuming input X-axis is a SpectroscopicAxis instance
        (default units are CGS)
        """
        xax = xarr.as_unit('Hz')
        return blackbody(xax, temperature, omega, units=units)

    blackbody_model = pyspeckit.models.model.SpectralModel(bb, 3, 
            shortvarnames=('T','\\Omega'),
            parnames=['temperature','omega'],
            fitunits='cm')
except:
    pass


