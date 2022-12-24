"""
Model the absorption spectrum of vibrational states of a species retrieved from
exomol
"""

import numpy as np
import requests
from astropy.io import ascii
from astropy import units as u
from astropy import constants
from pyspeckit import units
from astropy import table
try:
    import scipy.signal.fftconvolve as convolve
except ImportError:
    from numpy import convolve

from astroquery.vizier import Vizier
from astropy.table import Table

from tqdm.auto import tqdm

import os
path = os.path.split(os.path.realpath(__file__))[0]
# path = '/Users/adam/repos/pyspeckit-models/pyspeckitmodels/sio/'

# import requests
# from six import BytesIO
# import bz2
#
# resp = requests.get('https://exomol.com/db/SiO/28Si-16O/EBJT/28Si-16O__EBJT.trans.bz2')
# resp.raise_for_status()
# tablefile = bz2.BZ2File(BytesIO(resp.content))
# tbl = ascii.read(tablefile)


levels_fn = os.path.join(path, 'sio_levels.ecsv')
if os.path.exists(levels_fn):
    levels = Table.read(levels_fn)
else:
    levels = Vizier(row_limit=1e7).get_catalogs('J/MNRAS/434/1469/levels')[0]
    levels.write(levels_fn)
transitions_fn = os.path.join(path, 'sio_transitions.ecsv')
if os.path.exists(transitions_fn):
    transitions = Table.read(transitions_fn)
else:
    transitions = Vizier(row_limit=1e8).get_catalogs('J/MNRAS/434/1469/transit')[0]
    transitions.write(transitions_fn)


transitions_enhanced_fn = os.path.join(path, 'sio_transitions_enhanced.ecsv')
if os.path.exists(transitions_enhanced_fn):
    transitions = transitions_enhanced = Table.read(transitions_enhanced_fn)
else:
    print("Calculating wavelengths, etc.  This may take a while")
    eupper = np.zeros(len(transitions), dtype=np.float64) * u.cm**-1
    elower = np.zeros(len(transitions), dtype=np.float64) * u.cm**-1
    gupper = np.zeros(len(transitions), dtype=np.int16)
    glower = np.zeros(len(transitions), dtype=np.int16)
    wavelengths = np.zeros(len(transitions)) * u.um

    for isotope in np.unique(transitions['Mol']):
        levelmatches = levels['Mol'] == isotope
        transmatches = transitions['Mol'] == isotope
        sublevels = levels[levelmatches]
        subtrans = transitions[transmatches]
        # should already be sorted, but just to be sure
        sublevels.sort('i')
        # indices are the transition number minus one
        eupper[transmatches] = sublevels[subtrans['i1']-1]['E']
        elower[transmatches] = sublevels[subtrans['i0']-1]['E']
        gupper[transmatches] = sublevels[subtrans['i1']-1]['g']
        glower[transmatches] = sublevels[subtrans['i0']-1]['g']

    wavelengths = (eupper - elower).to(u.um, u.spectral())

    transitions_enhanced = transitions
    transitions_enhanced['eupper'] = eupper
    transitions_enhanced['elower'] = elower
    transitions_enhanced['gupper'] = gupper
    transitions_enhanced['glower'] = glower
    transitions_enhanced['wavelength'] = wavelengths
    transitions_enhanced.write(transitions_enhanced_fn)

partfunc_fn = os.path.join(path, 'sio_partfunc.ecsv')
if os.path.exists(partfunc_fn):
    partfunc = Table.read(partfunc_fn)
else:
    resp = requests.get('https://exomol.com/db/SiO/28Si-16O/EBJT/28Si-16O__EBJT.pf')
    resp.raise_for_status()
    partfunc = ascii.read(resp.text)
    partfunc.rename_column('col1', 'temperature')
    partfunc.rename_column('col2', 'Q')
    partfunc.write(partfunc_fn)


def tau_of_N(wavelength, column, tex=10*u.K, width=1.0*u.km/u.s,
             velocity=0.0*u.km/u.s,
             min_column=1e10,
             progressbar=tqdm,
             isotopomer=2816, unit_convention='cgs',
             width_units='km/s', velocity_units='km/s'):
    """
    Wavelength assumed to be an array

    Parameters
    ----------
    isotopomer : int
        Must be a four-digit integer.  Each digit refers to the last digit in C
        or O, e.g. for the standard 28Si16O, the number is 2816.
        30Si16O would be 3016.
    wavelength : quantity
        wavelength
    """
    # constants = units.unitdict[unit_convention]

    # not used prefactor = np.pi**0.5 * constants['e']**2 / constants['me'] / constants['c']

    wavelength = wavelength.to(u.cm, u.spectral())
    #wavelength_icm = wavelength.to(u.cm**-1, u.spectral())
    trans = transitions[transitions['Mol'] == isotopomer]
    OK_all_lines = ((trans['wavelength'] > wavelength.min()) &
                    (trans['wavelength'] < wavelength.max()))
    trans = trans[OK_all_lines]

    tau_total = np.zeros(wavelength.shape)
    column = u.Quantity(column, u.cm**-2)

    if isotopomer != 2816:
        raise NotImplementedError("Partition function only implemented for 28Si-16O; need to copy "
                                  "the coefficients from Barton+ 2013 table 4 for the others")
    partition_function = np.interp(tex.to(u.K).value, partfunc['temperature'], partfunc['Q'])

    for line in progressbar(trans):
        avalue = u.Quantity(line['A'], trans['A'].unit)
        wav = u.Quantity(line['wavelength'], trans['wavelength'].unit)
        nu = wav.to(u.Hz, u.spectral())
        # https://en.wikipedia.org/wiki/Einstein_coefficients#Oscillator_strengths
        # SI: oscstrength = avalue / ( 2 * np.pi * nu**2 * constants.e**2 / (constants.eps0**2 * constants.m_e * constants.c**3))
        gu = line['gupper']
        gl = line['glower']
        # oscstrength = (gu/gl * avalue / ((2 * np.pi * nu**2 * constants.e.esu**2 / (
        #     constants.m_e * constants.c**3)))).decompose()

        elower = u.Quantity(line['elower'], trans['elower'].unit)

        # from Yurchenko+ 2018 eqn 1
        nu_icm = nu.to(u.cm**-1, u.spectral())
        c2 = constants.h * constants.c / constants.k_B
        intensity = (gu * avalue / (8 * np.pi * constants.c * (nu_icm**2))
                     * np.exp(-elower * c2 / tex)
                     * (1-np.exp(-c2 * nu_icm / tex))
                     / partition_function).decompose()

        lambda_0 = velocity.to(u.um, u.doppler_optical(wav))
        width_lambda = width/constants.c * lambda_0
        width_icm = width/constants.c * lambda_0.to(u.cm**-1, u.spectral())

        lineprofile = np.sqrt(1/(2*np.pi)) / width_icm * np.exp(-(wavelength-lambda_0)**2/(2*width_lambda**2))
        #print((lineprofile.sum() * np.diff(wavelength_icm).mean()).decompose())

        # eqn 7
        crosssection = (intensity * lineprofile)

        tau_v = (crosssection * column).decompose()
        #print(f'max tau: {tau_v.max()}')
        tau_total += tau_v

    assert tau_total.shape == wavelength.shape
    return tau_total


def exomol_xsec(numin, numax, dnu, temperature):
    S = requests.Session()
    S.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    url = "https://exomol.com/xsec/28Si-16O/"
    resp = S.get(url)
    resp.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text)
    csrfmiddlewaretoken = soup.find('input', {'name': 'csrfmiddlewaretoken'}).attrs['value']

    resp2 = S.post(url, data={'dnu': dnu, 'numin': numin, 'numax': numax, 'T': temperature,
                              'csrfmiddlewaretoken': csrfmiddlewaretoken},
                   headers={'referer': 'https://exomol.com/xsec/28Si-16O/'})
    resp2.raise_for_status()
    # soup2 = BeautifulSoup(resp2.text)
    baseurl = 'https://exomol.com'
    sigmaurl = f'/results/28Si-16O_{int(numin)}-{int(numax)}_{temperature}K_{dnu:0.6f}.sigma'
    assert sigmaurl in resp2.text
    resp3 = S.get(baseurl + sigmaurl)
    resp3.raise_for_status()
    sigmas = np.array(list(map(float, resp3.text.split())))
    return sigmas


def test():
    numin, numax, dnu = 1.308467, 1.608467, 0.01
    tex = 500
    wavelengths = np.arange(numin, numax, dnu)*u.cm**-1

    S = requests.Session()
    S.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    url = "https://exomol.com/xsec/28Si-16O/"
    resp = S.get(url)
    resp.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text)
    csrfmiddlewaretoken = soup.find('input', {'name': 'csrfmiddlewaretoken'}).attrs['value']

    resp2 = S.post(url, data={'dnu': dnu, 'numin': numin, 'numax': numax, 'T': tex, 'csrfmiddlewaretoken': csrfmiddlewaretoken},
                   headers={'referer': 'https://exomol.com/xsec/28Si-16O/'})
    resp2.raise_for_status()
    # soup2 = BeautifulSoup(resp2.text)
    baseurl = 'https://exomol.com'
    sigmaurl = f'/results/28Si-16O_1-1_{tex}K_0.010000.sigma'
    assert sigmaurl in resp2.text
    resp3 = S.get(baseurl + sigmaurl)
    resp3.raise_for_status()
    sigmas = np.array(list(map(float, resp3.text.split())))

    column = 1e15*u.cm**-2
    tex = tex*u.K
    width = np.sqrt(constants.k_B * tex / (44*u.Da)).to(u.km/u.s)
    sigmas_calc = tau_of_N(wavelengths, column, tex=tex, width=width) / column

    # these don't agree
    # mine are higher by ~10^3
    # 1896.67847777
    # 100.0 K 5998.309367663244 cm2
    # 500.0 K 2682.3881261813913 cm2
    # 1000.0 K 1896.678477773368 cm2
    # 10000.0 K 186.93839582512095 cm2
    # so is this a problem with the partition function?
    import pylab as pl
    print(tex, sigmas_calc.max() / sigmas.max())
    pl.clf()
    pl.plot(wavelengths, sigmas, label='exomol')
    pl.plot(wavelengths, sigmas_calc, label='calculated')
    pl.legend(loc='best')

    return sigmas, sigmas_calc
