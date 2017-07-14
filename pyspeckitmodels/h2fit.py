import pyspeckit
from .support import readcol
import numpy as np

constants = pyspeckit.units.unitdict['cgs']

import os
tablepath = os.path.split(os.path.realpath(__file__))[0]+"/h2/"

def h2level_energy(V,J):
    """ Returns the theoretical level energy as a function of the
    vibrational (V) and rotational (J) state of the molecule. 
    in units of ergs
    
    Constants are from NIST: 
    http://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Units=SI&Mask=1000#Diatomic
    (see the bottom of the table)
    """

    We=4401.21
    Be=60.853
    WeXe=121.33 
    De=.0471
    Ae=3.062
    re=.74144

    return constants['h'] * constants['c'] * (We*(V+0.5) + Be*(J*(J+1)) - WeXe*(V+.5)**2 - De*J**2*(J+1)**2 - Ae*(V+.5)*(J+1)*J)

try:
    # read in rest energies before calling function
    resten = readcol(tablepath+'dalgarno1984_table5.txt',verbose=0)

    def restwl(vu,vl,ju,jl,calc=False):
        """ Uses energy levels measured by Dabrowski & Herzberg, Can J. Physics, 62,1639,1984 
        vu,vl - upper and lower vibrational states
        ju,jl - upper and lower rotational states 
        returns wavelength in microns
        online versions of this table:
        http://www.astronomy.ohio-state.edu/~depoy/research/observing/molhyd.htm
        http://www.jach.hawaii.edu/UKIRT/astronomy/calib/spec_cal/h2_s.html
        """
        if calc:
            return 1e4*h*c / (h2level_energy(vu,ju) - h2level_energy(vl,jl))
        else:
            if ju >= resten.shape[0] or vu >= resten.shape[1]:
                return 0
            dl = .01/(resten[ju][vu]-resten[jl][vl])
            return dl * 1e6
except IOError:
    print("Could not find dalgarno1984_table5.txt.  H2 rest energies not available.")

try:
    h2table = readcol(tablepath+'h2pars.txt',asStruct=True,skipafter=1)

    def aval(vu,ju,jl):
        """
        Lookup table for Einstein-A value as a function of 
        vibrational level, upper/lower J level
        Values from: http://www.jach.hawaii.edu/UKIRT/astronomy/calib/spec_cal/h2_s.html
        """
        if ju-jl==2:
            trans = 'S'
        elif ju-jl==-2:
            trans = 'O'
        elif ju-jl==0:
            trans = 'Q'
        else:
            raise ValueError("delta-J must be -2,0,2")

        if vu==0 and trans=='Q':
            # strongly forbidden
            return 0 

        wh = (h2table.jl==jl) * (h2table.trans==trans) * (h2table.vu==vu)
        if wh.sum() == 0:
            return 0
            #raise ValueError("No valid matches")

        return (h2table.aval[wh]*1e-7)[0]

    aval_vect = np.vectorize(aval)
except IOError:
    pass

try:
    # atran = pyfits.open('/Users/adam/observations/triplespec/Spextool2/data/atran2000.fits')
    atran = readcol(tablepath+'atran.txt')
    atran_wl = atran[:,0]*1e4
    atran_tr = atran[:,1]
    atran_arc = readcol(tablepath+'atran_arcturus.txt')
    ARCSORT = argsort(atran_arc[:,0])
    atran_arcwl = atran_arc[ARCSORT,0]*1e4
    atran_arctr = atran_arc[ARCSORT,1]
    def atmotrans(x):
        """ returns the atmospheric transmission at the given wavelength (in angstroms) """
        closest = argmin(abs(atran_wl-x))
        if atran_wl[closest] < x:
            m = (atran_tr[closest+1]-atran_tr[closest])/(atran_wl[closest+1]-atran_wl[closest])
            b = atran_tr[closest]
            y = m * (x-atran_wl[closest]) + b
        elif atran_wl[closest] > x:
            m = (atran_tr[closest]-atran_tr[closest-1])/(atran_wl[closest]-atran_wl[closest-1])
            b = atran_tr[closest-1]
            y = m * (x-atran_wl[closest-1]) + b
        else:
            y = atran_tr[closest]
        return y

    atmotrans_vect = np.vectorize(atmotrans)
except IOError:
    print("Could not find atran.txt.  atran atmospheric transition model will not be available")

def modelspec(wavelength,T,A,w,dx,op,Ak=0,extinction=False):
    """
    Generate a model H2 emission spectrum

    (incomplete - normalization is not obvious; it should be physically motivated but isn't right now)

    Parameters
    ----------
    wavelength : angstroms
    """
    model = wavelength * 0
    A=np.abs(A)
    # assume width, shift given in velocity:
    w = w*np.mean(wavelength)/(constants['c']/1e5)
    dx = dx*np.mean(wavelength)/(constants['c']/1e5)
    for v in range(1,6):
        for j in range(1,14):
            if (j % 2):  # ortho/para multiplier
                mult=op
            else: 
                mult=1
            # S branch
            wl = restwl(v,v-1,j,j-2) * 10**4
            model += A*mult*(2*j+1)*aval(v,j,j-2)*np.exp(-h2level_energy(v,j)/(constants['k']*T)) * np.exp( - ( wavelength - wl - dx )**2 / (2*w**2) )
            # Q branch
            wl = restwl(v,v-1,j,j) * 10**4
            model += A*mult*(2*(j)+1)*aval(v,j,j)*np.exp(-h2level_energy(v,j)/(constants['k']*T)) * np.exp( - ( wavelength - wl - dx )**2 / (2*w**2) )
    if extinction:
        # alpha=1.8 comes from Martin & Whittet 1990.  alpha=1.75 from Rieke and Lebofsky 1985
        Al = Ak * (wavelength/22000.0)**(-1.75)
        model *= np.exp(-Al)
    return model

h2_model = modelspec

def tau_of_N(microns, column, v0=0, temperature=20, width=1.0, velocity=0.0, orthopara=3):
    """
    Return the optical depth of an H2 line as a function of wavelength...
    (absorption)
    """
    grounden = h2level_energy(0,0)
    alllevelpop = np.sum([np.exp(-(h2level_energy(v,j)-grounden)/(k*temperature)) for v in range(0,6) for j in range(0,14)])
    tautotal = microns*0
    for vu in range(1,6):
        for j in range(0,14):
            if (j % 2):  # ortho/para multiplier
                mult=orthopara
            else: 
                mult=1
            # S branch
            wl = restwl(vu,v0,j+2,j) 
            offset = velocity/(c/1e5) * wl
            w = width/(c/1e5) * wl
            einA = aval(vu,j+2,j)
            oscstrength = constants.electronmass * c / pi / constants.electroncharge**2 * (wl/1e4)**2 / (8*pi) * einA
            column_i = column*(mult/(orthopara+1.)) * (2*j+1) * np.exp(-(h2level_energy(v0,j)-grounden)/(k*temperature)) / alllevelpop
            tau_0 = column_i*oscstrength*(wl/1e4) / (width*1e5)
            tau_nu = np.exp( - ( microns - wl - offset )**2 / (2*w**2) ) * tau_0
            tautotal += tau_nu
            
            # Q branch
            wl = restwl(vu,v0,j,j) 
            offset = velocity/(c/1e5) * wl
            w = width/(c/1e5) * wl
            einA = aval(vu,j,j)
            oscstrength = constants.electronmass * c / pi / constants.electroncharge**2 * (wl/1e4)**2 / (8*pi) * einA
            column_i = column*(mult/(orthopara+1.)) * (2*j+1) * np.exp(-(h2level_energy(v0,j)-grounden)/(k*temperature)) / alllevelpop
            tau_0 = column_i*oscstrength*(wl/1e4) / (width*1e5)
            tau_nu = np.exp( - ( microns - wl - offset )**2 / (2*w**2) ) * tau_0
            tautotal += tau_nu

    return tautotal

def modelpars(fixpars=[None,None,None,None,None],
        minpars=[200.0,0,0,-500.0,1.5],
        maxpars=[15000.0,None,1000.0,500.0,3.5],
        params=[2000,5e-9,31.1,-20.0,3],
        extinction=False,
        **kwargs):

    if len(kwargs) > 0:
        extinction=kwargs['extinction']

    limitedmin = map(testnone,minpars)
    limitedmax = map(testnone,maxpars)
    fixed = map(testnone,fixpars)
    minpars = map(nonetozero,minpars)
    maxpars = map(nonetozero,maxpars)
    fixpars = map(nonetozero,fixpars)

    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],'parname':"TEMPERATURE",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],'parname':"SCALE",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],'parname':"WIDTH",'error':0},
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],'parname':"SHIFT",'error':0},
                {'n':4,'value':params[4],'limits':[minpars[4],maxpars[4]],'limited':[limitedmin[4],limitedmax[4]],'fixed':fixed[4],'parname':"ORTHOtoPARA",'error':0}]

    if extinction:
        parinfo.append({'n':5,'value':1,'limits':[0,30],'limited':[1,1],'fixed':0,'parname':"AK",'error':0})

    return parinfo

