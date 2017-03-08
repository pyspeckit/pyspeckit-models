"""
Simple Rousellot 2000 OH line model
"""

from astropy.io import ascii
import pyspeckit
import numpy as np

import os
path = os.path.split(os.path.realpath(__file__))[0]
ohlines = ascii.read(path+'/oh/rousselot2000.dat')


def oh_model(xarr, amplitude, width, velocity):
    """
    Simple Rousellot model; amplitudes are fixed relative to each other 
    """

    model = np.array(xarr * 0)
    xarr = xarr.as_unit('angstroms')

    lines_to_use = (ohlines['col1'] > xarr.min()) * (ohlines['col1'] < xarr.max())

    for wavelength, intensity in ohlines[lines_to_use]:
        lw = width / pyspeckit.units.speedoflight_kms * wavelength
        center = wavelength + wavelength * velocity / pyspeckit.units.speedoflight_kms
    
        model += amplitude * intensity * np.exp(-(xarr-center)**2 / (2.0*lw)**2)

    return model

def add_to_registry(sp):
    """
    Add the OH model to the Spectrum's fitter registry
    """
    # can't have absorption in recombination case
    OHemission = pyspeckit.models.model.SpectralModel(oh_model, 3, 
            shortvarnames=('A','\\sigma','\\Delta x'),
            parnames=['amplitude','width','velocity'],
            parlimited=[(True,False),(True,False),(False,False)], 
            parlimits=[(0,0), (0,0), (0,0)],
            fitunits='angstroms')

    sp.Registry.add_fitter('oh', OHemission,
            OHemission.npars, multisingle='multi')

