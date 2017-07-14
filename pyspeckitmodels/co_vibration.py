"""
Predict the CO fundamental/harmonic absorption for a given column density
"""

from astropy.io import ascii
import numpy as np
import pyspeckit
from pyspeckit import units
from astropy import table
try:
    import scipy.signal.fftconvolve as convolve
except ImportError:
    from numpy import convolve

try:
    import os
    path = os.path.split(os.path.realpath(__file__))[0]

    all_lines = ascii.read(path+'/co/co_parameters.txt')
    all_lines.add_column(table.Column(all_lines['wavenum']**-1,name='wavelength',))

    data = ["$^{1%s}$c$^{1%s}$o v=%i-%i j=%i-%i" % (
                                                str(line['isotopomer'])[0],
                                                str(line['isotopomer'])[1],
                                                line['vup'], line['vlo'],
                                                line['jlo']+1, line['jlo'])
                                                for line in all_lines]
    col = table.Column(np.array(data), name='names',)
    all_lines.add_column(col)

    all_lines.add_column(table.Column((all_lines['E_low'] *
                                       units.unitdict['cgs']['c']*
                                       units.unitdict['cgs']['h']/
                                       units.unitdict['cgs']['kb']),
                                      name='T_low',))
except Exception as ex:
    print(ex)
    raise

def tau_of_N(wavelength, column, tex=10, width=1.0, velocity=0.0,
             isotopomer=26, Be=57.63596828e9, unit_convention='cgs',
             width_units='km/s', velocity_units='km/s'):
    """
    Wavelength assumed to be an array cm

    Parameters
    ----------
    isotopomer : int
        Must be a two-digit integer.  Each digit refers to the last digit in C
        or O, e.g. for the standard 12C16O, the number is 26 = 1[2]CO 1[6]O.
        13C18O would be 38.
    wavelength : float
        wavelength in cm
    """
    constants = units.unitdict[unit_convention]

    prefactor = np.pi**0.5 * constants['e']**2 / constants['me'] / constants['c']

    all_iso = (all_lines['isotopomer']==isotopomer)
    OK_all_lines = (all_lines['wavelength']> wavelength.min()) * (all_lines['wavelength'] < wavelength.max()) * all_iso

    #model = np.ones(wavelength.shape)
    tau_total = np.zeros(wavelength.shape)

    nu = constants['c'] / wavelength


    # slow - can this be approximated?
    valid = all_lines[all_iso]['T_low'] < tex * 5
    vpartition_total = np.sum( [np.exp(-(line['vlo']+0.5) * constants['h'] / constants['kb'] / tex * constants['c'] * line['E_low']) 
        for line in all_lines[all_iso][valid]] )

    lines = all_lines[OK_all_lines]

    for line in lines:
        oscstrength = line['gf-val']
        wav = line['wavenum']**-1
        J = line['jlo']

        rotation_partition = (2*J+1) * np.exp(-(J*(J+1)*Be*constants['h']/(constants['kb']*tex))) / (1.0+(tex * constants['kb'] / (Be*constants['h']))**2)**0.5
        vibration_partition = np.exp(-(line['vlo']+0.5) * constants['h'] / constants['kb'] / tex * constants['c'] * line['E_low']) / vpartition_total
        column_i = column * rotation_partition * vibration_partition
        if column_i < 1e10:
            continue

        tau_0 = column_i*oscstrength*wav / (width*units.velocity_dict[width_units]/units.velocity_dict[constants['speed']])

        v_in_units = velocity*units.velocity_dict[velocity_units]/units.velocity_dict[constants['speed']]

        lambda_off = constants['c'] / ( constants['c']/wav * (1.0+v_in_units/constants['c'])**-1 ) 
        nu0 = constants['c'] / (lambda_off)

        u = constants['c'] * (nu0-nu)/nu0
        b = np.sqrt(2)*(width * units.velocity_dict[width_units]/units.velocity_dict[constants['speed']])

        #if fast:
        #    tau_total[np.argmin(np.abs(wavelength-wav+lambda_off))] += tau_0
        #else:
        tau_v = tau_0 * np.exp(-(u/b)**2)
        tau_total += tau_v
        #model *= np.exp(-tau_v)
        #print "matched line ",line," tau_0 = " ,tau_0, " wav = ",wav

        #if fast:
        #    wavelength_center = np.median(wavelength)
        #    kernel = np.exp( -(wavelength-wavelength_center)**2/(2*width / constants['c'] * wavelength_center) ) 

        #    convolve(tau_total, kernel, mode='same')

    return tau_total


try:
    import pyspeckit

    def modelspectrum(xarr, column, tex=10, width=1.0, velocity=0.0, units=None, **kwargs):
        """
        CO model absorption spectrum given an X-array and a total CO column

        Parameters
        ----------
        xarr : `pyspeckit.spectrum.units.SpectroscopicAxis`
            An X-axis instance (will be converted to cm)
        temperature : float
            Excitation temperature of the CO molecule
        """
        if units is None:
            xax = xarr.as_unit('cm')
        else:
            xax = pyspeckit.units.SpectroscopicAxis(xarr, units=units)

        tau = tau_of_N(xax, column, tex=tex, width=width, velocity=velocity, **kwargs)
        co_model = np.exp(-tau)
        return co_model

    absorption_model = pyspeckit.models.model.SpectralModel(modelspectrum, 4, 
            shortvarnames=('N','T','\\sigma','\\Delta x'),
            parnames=['column','temperature','width','velocity'],
            fitunits='cm')

except ImportError:
    pass

try:
    import blackbody

    def absorbed_blackbody(xarr, column, bbtemperature, omega, tex=10, width=1.0, velocity=0.0, extinction=False, units=None, **kwargs):
        """ 
        CO model absorption on a blackbody
        """
        if units is None:
            try:
                xax = xarr.as_unit('cm')
            except AttributeError:
                raise AttributeError("Must specify units if xarr is not a SpectroscopicAxis instance.")
        else:
            xax = pyspeckit.units.SpectroscopicAxis(xarr, units=units).as_unit('cm')

        bb = blackbody.blackbody_wavelength(xax, bbtemperature, wavelength_units='cm', omega=omega, normalize=False)
        co_tau = tau_of_N(xax, column, width=width, velocity=velocity, tex=tex, **kwargs)
        co_emi = blackbody.blackbody_wavelength(xax, tex, wavelength_units='cm', omega=omega, normalize=False) * (1.0-np.exp(-co_tau))

        model = bb*np.exp(-co_tau) + co_emi
        
        if extinction:
            # alpha=1.8 comes from Martin & Whittet 1990.  alpha=1.75 from Rieke and Lebofsky 1985
            Al = extinction * (xax/2.2e-4)**(-1.75)
            model *= np.exp(-Al)

        return model

    absorbed_blackbody_model = pyspeckit.models.model.SpectralModel(absorbed_blackbody, 5, 
            shortvarnames=('N','T','\\Omega','\\sigma','\\Delta x'),
            parnames=['column','temperature','omega','width','velocity'],
            fitunits='cm')
    absorbed_blackbody_model_texvar = pyspeckit.models.model.SpectralModel(absorbed_blackbody, 6, 
            shortvarnames=('N','T','\\Omega','T_{ex}','\\sigma','\\Delta x'),
            parnames=['column','temperature','omega','tex','width','velocity'],
            fitunits='cm')
    absorbed_blackbody_model_texvar_extinction = pyspeckit.models.model.SpectralModel(absorbed_blackbody, 7, 
            shortvarnames=('N','T','\\Omega','T_{ex}','\\sigma','\\Delta x','A_K'),
            parnames=['column','temperature','omega','tex','width','velocity','extinction'],
            parlimits=[(1e10,1e20),(2.73,100000),(0,1),(2.73,10000),(0,1000),(0,0),(0,100)],
            parlimited=[(True,True),(True,True),(True,True),(True,True),(True,True),(False,False),(True,True)],
            fitunits='cm')

except ImportError:
    pass

if __name__=="__main__":
    print("Predictions for G26.347307-0.41227641")
    print("Best-fit line @ 33.3 km/s (2.4 kpc) has T_A(13CO) = 0.63 K -> N(12CO)=5.13e16 (sigma=1.5 km/s)")
    print("Assuming tex=20")

    import blackbody

    x = np.linspace(2.30,2.47,600)
    x_superres = np.linspace(2.30,2.47,60000)
    tau_superres = tau_of_N(x_superres*1e-4, 5.13e16, 20, 1.5, 33.3)
    tau_atmospheric = tau_of_N(x_superres*1e-4, 5e14, 6000, 1.8, 33.3)

    cont = agpy.blackbody.blackbody_wavelength(x*1e4, 6000)
    cont_superres = agpy.blackbody.blackbody_wavelength(x_superres*1e4, 6000)

    averaged = np.concatenate([[(cont_superres*np.exp(-tau_superres))[i::100] for i in range(100)]]).mean(axis=0)
    atm_averaged = np.concatenate([[(cont_superres*np.exp(-tau_atmospheric))[i::100] for i in range(100)]]).mean(axis=0)

    from astropy.io import fits
    atmosphere = fits.getdata('/Users/adam/anaconda/envs/astropy35/h2fit/atran2000.fits')
    #atmosphere = fits.getdata('/Users/adam/agpy/h2fit_support/atran2000.fits')
    
    import pylab
    pylab.figure(1)
    pylab.clf()
    #pylab.plot(x_superres, cont_superres*np.exp(-tau_superres))
    pylab.plot(atmosphere[0],atmosphere[1])
    pylab.plot(x, atm_averaged)
    pylab.plot(x, averaged)
    pylab.gca().set_xlim(2.30,2.47)
    pylab.gca().set_ylim(averaged.min(),averaged.max())
    pylab.title("G26.347-0.412")
    pylab.savefig('/Users/adam/work/cepheid_distance/CO_absorption_predicted_spectrum_G26.347-0.412.png')

    print("Predictions for G26.419144-0.50988479")
    print("Best-fit line @ 19.2 km/s (1.6 kpc) has T_A(13CO) = 0.98 K -> N(12CO)=7.98e16 (sigma=0.32 km/s)")
    print("Assuming tex=20")

    # 26.316284,-0.40536589  d>0.8 kpc   112.03646,113.08685   rms=0.12 max=0.30 (at 19 km/s), A=0.31 v=33.4 w=2.9
    # 26.278794,-0.42970377  d>1.6 kpc   118.14556,109.12947   rms=0.09 max=0.30
    # 26.338947,-0.49258955  d>1.2 kpc   108.40267,98.904143   rms=0.11 max=0.31
    # 26.403965,-0.5400639   d>1.2 kpc   97.86309 ,91.184737   rms=0.21 max=0.58
    # 26.30095,-0.55934725   d>1.8 kpc   114.62647,88.049232   rms=0.17 max=0.29
    # 26.316013,-0.55298208  d>2.3 kpc   112.17268,89.084219   rms=0.23 max=0.64
    # 26.347307,-0.41227641  d>3.4 kpc   106.99591,111.96319   rms=0.09 max=0.26 (at 19km/s), A=0.70 v=33.3 w=1.75 
    # 26.419144,-0.50988479  d>2.6 kpc   95.374313,96.091909   A=0.59  v=19.5  w=0.48
    # 26.252724,-0.44042119  d>2.3 kpc   122.3905 ,107.3868    weak/nondetection rms=0.10 max=0.26
    # 26.268552,-0.58696603  d>3.0 kpc   119.91491,83.558373   weak/nondetection rms=0.24 (max 0.63)
    # 26.277175,-0.50593405  d>3.4 kpc   118.45517,96.734305   A=0.58  v=18.8  w=0.36
    # 26.290964,-0.51439111  d>2.6 kpc   116.21877,95.359173   A=0.54  v=19.2  w=0.56
    # 26.348893,-0.57856622  d>2.1 kpc   106.84555,84.924197   rms=0.19  (no detection)
