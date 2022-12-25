"""
Model the absorption spectrum of vibrational states of a species retrieved from
exomol
"""

import warnings
import numpy as np
import requests
from astropy.io import ascii
from astropy import units as u
from astropy import constants

# from astroquery.vizier import Vizier
from astropy.table import Table
from astroquery.linelists.cdms import CDMS

from tqdm.auto import tqdm

from six import BytesIO
import bz2

import os
path = os.path.split(os.path.realpath(__file__))[0]

# resp = requests.get('https://exomol.com/db/SiO/12C-16O/EBJT/28Si-16O__EBJT.trans.bz2')
# resp.raise_for_status()
# tablefile = bz2.BZ2File(BytesIO(resp.content))
# tbl = ascii.read(tablefile)


levels_fn = os.path.join(path, 'co_levels.ecsv')
if os.path.exists(levels_fn):
    levels = Table.read(levels_fn)
else:
    resp = requests.get('https://exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.states.bz2')
    resp.raise_for_status
    tablefile = bz2.BZ2File(BytesIO(resp.content))
    levels = ascii.read(tablefile)
    levels.rename_column('col1', 'i')
    levels.rename_column('col2', 'E')
    levels['E'].unit = u.cm**-1
    levels.rename_column('col3', 'g')
    levels.rename_column('col4', 'j')
    levels.rename_column('col5', 'v')
    levels.rename_column('col6', 'letter_e')
    levels.write(levels_fn)

transitions_fn = os.path.join(path, 'co_transitions.ecsv')
if os.path.exists(transitions_fn):
    transitions = Table.read(transitions_fn)
else:
    resp = requests.get('https://exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2')
    resp.raise_for_status
    tablefile = bz2.BZ2File(BytesIO(resp.content))
    transitions = ascii.read(tablefile)
    transitions.rename_column('col1', 'i0')
    transitions.rename_column('col2', 'i1')
    transitions.rename_column('col3', 'A')
    transitions['A'].unit = u.s**-1
    transitions.rename_column('col4', 'wavelength_')
    transitions['wavelength_'].unit = u.cm**-1
    transitions.write(transitions_fn)

transitions_enhanced_fn = os.path.join(path, 'co_transitions_enhanced.ecsv')
if os.path.exists(transitions_enhanced_fn):
    transitions = transitions_enhanced = Table.read(transitions_enhanced_fn)
else:
    print("Calculating wavelengths, etc.  This may take a while")
    eupper = np.zeros(len(transitions), dtype=np.float64) * u.cm**-1
    elower = np.zeros(len(transitions), dtype=np.float64) * u.cm**-1
    gupper = np.zeros(len(transitions), dtype=np.int16)
    glower = np.zeros(len(transitions), dtype=np.int16)
    wavelengths = np.zeros(len(transitions)) * u.um

    # should already be sorted, but just to be sure
    levels.sort('i')
    # indices are the transition number minus one
    eupper = levels[transitions['i0']-1]['E']
    elower = levels[transitions['i1']-1]['E']
    gupper = levels[transitions['i0']-1]['g']
    glower = levels[transitions['i1']-1]['g']
    assert np.all(eupper > elower)

    wavelengths = np.abs((eupper - elower).to(u.um, u.spectral()))

    transitions_enhanced = transitions
    transitions_enhanced['eupper'] = eupper
    transitions_enhanced['elower'] = elower
    transitions_enhanced['gupper'] = gupper
    transitions_enhanced['glower'] = glower
    transitions_enhanced['wavelength'] = wavelengths
    transitions_enhanced.write(transitions_enhanced_fn)

partfunc_fn = os.path.join(path, 'co_partfunc.ecsv')
if os.path.exists(partfunc_fn):
    partfunc = Table.read(partfunc_fn)
else:
    resp = requests.get('https://exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.pf')
    resp.raise_for_status()
    partfunc = ascii.read(resp.text)
    partfunc.rename_column('col1', 'temperature')
    partfunc.rename_column('col2', 'Q')
    partfunc.write(partfunc_fn)

def calc_intensities(wavelength, tex, temperature_threshold=100,
                     check_cdms=False):
    wavelength = wavelength.to(u.cm, u.spectral())

    trans = transitions
    OK_all_lines = ((trans['wavelength'] > wavelength.min()) &
                    (trans['wavelength'] < wavelength.max()))
    trans = trans[OK_all_lines]

    partition_function = np.interp(tex.to(u.K).value, partfunc['temperature'], partfunc['Q'])

    tex_to_icm = (tex * constants.k_B).to(u.cm**-1, u.spectral())

    intensities = []

    for line in (trans):
        elower = u.Quantity(line['elower'], trans['elower'].unit)
        if elower > tex_to_icm * temperature_threshold:
            continue

        wav = u.Quantity(line['wavelength'], trans['wavelength'].unit)

        avalue = u.Quantity(line['A'], trans['A'].unit)
        gu = line['gupper']

        # from Yurchenko+ 2018 eqn 1
        nu_icm = wav.to(u.cm**-1, u.spectral())
        c2 = constants.h * constants.c / constants.k_B
        intensity = (gu * avalue / (8 * np.pi * constants.c * (nu_icm**2) * partition_function)
                     * np.exp(-elower * c2 / tex)
                     * (1-np.exp(-c2 * nu_icm / tex))
                    ).decompose().to(u.cm)
        if check_cdms:
            try:
                cdmsq = CDMS.query_lines(wavelength.to(u.GHz, u.spectral()).min(),
                                         wavelength.to(u.GHz, u.spectral()).max(),
                                         min_strength=-500,
                                         molecule="028503 CO",
                                         temperature_for_intensity=tex.value)[0]
            except Exception:
                continue
            cdmslgint = 10**cdmsq['LGINT']
            cdmsint = cdmslgint * u.MHz * u.nm**2
            cdmsint_icm = (cdmsint / constants.c).to(intensity.unit)
            print(f"T={tex} Intensity: {intensity}.  CDMS intensity={cdmslgint} -> {cdmsint} -> {cdmsint_icm} Wavelength={wav.to(u.um)}, frequency={wav.to(u.GHz, u.spectral())}")
            print(f"CDMS / our intensity: {cdmsint_icm / intensity}")
        intensities.append(intensity)
    return intensities


def tau_of_N(wavelength, column, tex=10*u.K, width=1.0*u.km/u.s,
             velocity=0.0*u.km/u.s,
             min_column=1e10,
             progressbar=tqdm,
             norm_threshold=1e-4,
             temperature_threshold=100,
             isotopomer=1216,
            ):
    """
    Wavelength assumed to be an array

    Parameters
    ----------
    isotopomer : int
        Must be a four-digit integer.  Each digit refers to the last digit in C
        or O, e.g. for the standard 12C16O, the number is 1216.
        30Si16O would be 3016.
    wavelength : quantity
        wavelength
    """
    # constants = units.unitdict[unit_convention]

    # not used prefactor = np.pi**0.5 * constants['e']**2 / constants['me'] / constants['c']

    wavelength = wavelength.to(u.cm, u.spectral())
    wavenumber = wavelength.to(u.cm**-1, u.spectral())
    dx_icm = np.abs(wavelength[1].to(u.cm**-1, u.spectral())-wavelength[0].to(u.cm**-1, u.spectral()))
    dx = np.abs(wavelength[1] - wavelength[0]).to(u.um)

    trans = transitions
    OK_all_lines = ((trans['wavelength'] > wavelength.min()) &
                    (trans['wavelength'] < wavelength.max()))
    trans = trans[OK_all_lines]

    tau_total = np.zeros(wavelength.shape)
    column = u.Quantity(column, u.cm**-2)

    if isotopomer != 1216:
        raise NotImplementedError("Partition function only implemented for 12C-16O")
    partition_function = np.interp(tex.to(u.K).value, partfunc['temperature'], partfunc['Q'])
    # print(len(trans))
    # print("transition table: ", trans)
    # print()

    tex_to_icm = (tex * constants.k_B).to(u.cm**-1, u.spectral())

    for line in progressbar(trans):
        elower = u.Quantity(line['elower'], trans['elower'].unit)
        if elower > tex_to_icm * temperature_threshold:
            continue

        # print(f"Line: {line}")

        wav = u.Quantity(line['wavelength'], trans['wavelength'].unit)
        lambda_0 = velocity.to(u.um, u.doppler_optical(wav))
        width_lambda = (width/constants.c * lambda_0).to(u.um)
        width_icm = (width/constants.c * lambda_0.to(u.cm**-1, u.spectral())).to(u.cm**-1)
        # print("widths", width_lambda, width_icm, width, dx, dx_icm)
        # print(f"partition function: {partition_function}")

        # the line profile has to integrate to 1 and has to have units of cm (inverse frequency) because integral f(nu) dnu = 1 (unitless)

        # for reasons unknown, the Doppler profile adopted by exomol uses a Gaussian where the width is defined like this:
        # e^(-(x-x0)^2 / sigma^2)
        # where sigma = sqrt(k T / m)
        # but a 'normal' Gaussian has profile
        # e^(-(x-x0)^2 / (2 sigma^2))
        # and the theoretical line width is
        # sigma_v = sqrt(2 k T / m)
        # which is actually a factor of 4 off in the exponent?

        # integrates to 1
        lineprofile = np.sqrt(1/(np.pi)) / width_icm * np.exp(-(wavelength-lambda_0)**2/(width_lambda**2)) * dx_icm
        # print(f'lineprofile sum: {lineprofile.sum()}, lineprofile integral: {(lineprofile*dx_icm).sum().decompose()} dx={dx} dx_icm={dx_icm}.  Lineprofile max: {lineprofile.max()}')
        # print((lineprofile.sum() * np.diff(wavelength_icm).mean()).decompose())
        if lineprofile.sum() == 0:
            warnings.warn("Line profile is zero, skipping.")
            continue
        if lineprofile.sum() < 1-norm_threshold:
            warnings.warn(f"Line profile is not normalized.  Sum is {lineprofile.sum():0.2g}")
            lineprofile /= lineprofile.sum()


        avalue = u.Quantity(line['A'], trans['A'].unit)
        # https://en.wikipedia.org/wiki/Einstein_coefficients#Oscillator_strengths
        # SI: oscstrength = avalue / ( 2 * np.pi * nu**2 * constants.e**2 / (constants.eps0**2 * constants.m_e * constants.c**3))
        gu = line['gupper']


        # from Yurchenko+ 2018 eqn 1
        nu_icm = wav.to(u.cm**-1, u.spectral())
        c2 = constants.h * constants.c / constants.k_B
        intensity = (gu * avalue / (8 * np.pi * constants.c * (nu_icm**2) * partition_function)
                     * np.exp(-elower * c2 / tex)
                     * (1-np.exp(-c2 * nu_icm / tex))
                    ).decompose().to(u.cm)
        # print(wavelength.to(u.GHz, u.spectral()).min(), wavelength.to(u.GHz, u.spectral()).max())
        # cdmsq = CDMS.query_lines(wavelength.to(u.GHz, u.spectral()).min(),
        #                          wavelength.to(u.GHz, u.spectral()).max(),
        #                          min_strength=-500,
        #                          molecule="028503 CO",
        #                          temperature_for_intensity=tex.value)[0]
        # # print(cdmsq)
        # cdmslgint = 10**cdmsq['LGINT']
        # cdmsint = cdmslgint * u.MHz * u.nm**2
        # cdmsint_icm = (cdmsint / constants.c).to(intensity.unit)
        # print(f"Intensity: {intensity}.  CDMS intensity={cdmslgint} -> {cdmsint} -> {cdmsint_icm} Wavelength={wav.to(u.um)}, frequency={wav.to(u.GHz, u.spectral())}")
        # print(f"CDMS / our intensity: {cdmsint_icm / intensity}")



        # using this to hack the units "right" - trying to solve 6 order of magnitude disagreement...
        # eqn 7
        crosssection = (intensity * lineprofile) / dx_icm
        # print(f"Max cross-section: {crosssection.max().to(u.cm**2)}     =     {crosssection.max().to(u.barn)}")

        tau_v = (crosssection * column).decompose()
        # print(f'max tau: {tau_v.max()}')
        tau_total += tau_v

    return tau_total



def exomol_xsec(numin, numax, dnu, temperature, molecule='12C-16O'):
    S = requests.Session()
    S.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    url = f"https://exomol.com/xsec/{molecule}/"
    resp = S.get(url)
    resp.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, features='html5lib')
    csrfmiddlewaretoken = soup.find('input', {'name': 'csrfmiddlewaretoken'}).attrs['value']

    resp2 = S.post(url, data={'dnu': dnu, 'numin': numin, 'numax': numax, 'T': temperature,
                              'csrfmiddlewaretoken': csrfmiddlewaretoken},
                   headers={'referer': url})
    resp2.raise_for_status()
    # soup2 = BeautifulSoup(resp2.text, 'html5')
    baseurl = 'https://exomol.com'
    sigmaurl = f'/results/{molecule}_{int(numin)}-{int(numax)}_{temperature}K_{dnu:0.6f}.sigma'
    assert sigmaurl in resp2.text
    resp3 = S.get(baseurl + sigmaurl)
    resp3.raise_for_status()
    sigmas = np.array(list(map(float, resp3.text.split())))
    return sigmas


def test():
    numin, numax, dnu = 3.815033, 3.895033, 0.01
    tex = 300
    wavelengths = np.arange(numin, numax, dnu)*u.cm**-1

    sigmas = exomol_xsec(numin, numax, dnu, tex, molecule='12C-16O')

    column = 1e15*u.cm**-2
    tex = tex*u.K
    width = np.sqrt(constants.k_B * tex / (28*u.Da)).to(u.km/u.s)
    sigmas_calc = tau_of_N(wavelengths, column, tex=tex, width=width) / column

    import pylab as pl
    print(f"tex={tex}, max calc: {sigmas_calc.max()} max downloaded: {sigmas.max()} ratio={sigmas_calc.max() / sigmas.max()}")
    pl.clf()
    pl.plot(wavelengths, sigmas, label='exomol')
    pl.plot(wavelengths, sigmas_calc, label='calculated')

    wavelengths2 = np.linspace(numin, numax, 100000)*u.cm**-1
    sigmas_calc_2 = tau_of_N(wavelengths2, column, tex=tex, width=width) / column
    pl.plot(wavelengths2, sigmas_calc_2, label='calculated2')
    pl.semilogy()
    pl.ylim(1e-30, 1e-18)
    pl.legend(loc='best')
    pl.xlabel("Wavelength [cm$^{-1}$]")
    pl.ylabel("Cross Section [cm$^{-2}$]")

    return sigmas, sigmas_calc

def test2():
    numin, numax, dnu = 2100, 2190, 0.01
    tex = 100
    wavelengths = np.arange(numin, numax+dnu/2, dnu)*u.cm**-1

    sigmas = exomol_xsec(numin, numax, dnu, tex, molecule='12C-16O')

    column = 1e17*u.cm**-2
    tex = tex*u.K
    width = np.sqrt(constants.k_B * tex / (28*u.Da)).to(u.km/u.s)
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
    print(f"tex={tex}, max calc: {sigmas_calc.max()} max downloaded: {sigmas.max()} ratio={sigmas_calc.max() / sigmas.max()}")
    pl.clf()
    dnu = wavelengths[1]-wavelengths[0]
    pl.plot(wavelengths, sigmas/dnu, label='exomol')
    pl.plot(wavelengths, sigmas_calc/dnu, label='calculated')
    pl.xlabel("Wavelength [cm$^{-1}$]")
    pl.ylabel("Cross Section [cm$^{-2}$]")

    wavelengths2 = np.linspace(numin, numax, 100000)*u.cm**-1
    dnu = wavelengths2[1]-wavelengths2[0]
    sigmas_calc_2 = tau_of_N(wavelengths2, column, tex=tex, width=width) / column
    pl.plot(wavelengths2, sigmas_calc_2/dnu, label='calculated2')
    pl.semilogy()
    pl.ylim(1e-35, 1e-15)
    pl.legend(loc='best')

    return sigmas, sigmas_calc


def test_tloop():
    # numin, numax, dnu = 3.815033, 3.895033, 0.01
    # numin, numax, dnu = 2142.245, 2143.245, 0.01

    numin, numax, dnu = 2147.081139-0.01*11, 2147.081139+0.01*11, 0.01
    tems = np.linspace(100, 5000, 9)
    ratios = []
    ratios2 = []
    import pylab as pl
    fig = pl.figure(1)
    for ii, tex in enumerate(tems[::-1]):
        tex = int(tex)
        wavelengths = np.arange(numin, numax + dnu/2, dnu)*u.cm**-1

        sigmas = exomol_xsec(numin, numax, dnu, tex, molecule='12C-16O')

        column = 1e15*u.cm**-2
        tex = tex*u.K
        width = np.sqrt(constants.k_B * tex / (28*u.Da)).to(u.km/u.s)
        sigmas_calc = tau_of_N(wavelengths, column, tex=tex, width=width, progressbar=lambda x: x) / column

        ax = pl.subplot(3, 3, ii+1)
        ax.cla()
        pl.plot(wavelengths, sigmas, label='exomol')
        pl.plot(wavelengths, sigmas_calc, label='calculated')

        wavelengths2 = np.linspace(numin, numax, 100000)*u.cm**-1
        sigmas_calc_2 = tau_of_N(wavelengths2, column, tex=tex, width=width, progressbar=lambda x: x) / column
        pl.plot(wavelengths2, sigmas_calc_2, label='calculated2')
        pl.semilogy()
        ymin, ymax = pl.ylim()
        pl.ylim(1e-30, ymax)
        pl.legend(loc='best')
        pl.xlabel("Wavelength [cm$^{-1}$]")
        pl.ylabel("Cross Section [cm$^{-2}$]")
        pl.title(tex)

        print(f"tex={tex}, max calc: {sigmas_calc.max()} max downloaded: {sigmas.max()} ratio={sigmas_calc.max() / sigmas.max()}, {sigmas_calc_2.max()/sigmas.max()}")
        ratios.append((sigmas_calc.max() / sigmas.max()).decompose().value)
        ratios2.append((sigmas_calc_2.max() / sigmas.max()).decompose().value)

    fig = pl.figure(2)
    fig.clf()
    pl.semilogy(tems[::-1], ratios)
    pl.semilogy(tems[::-1], ratios2)
    return sigmas, sigmas_calc, ratios, ratios2
