import numpy
import re

# define physical constants to high precision
h=6.626068e-27
c=2.99792e10
k=1.3806503e-16
e=4.803e-12

def h2level_energy(V,J):
    """ Returns the theoretical level energy as a function of the
    vibrational (V) and rotational (J) state of the molecule. 
    
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

    return h * c * (We*(V+0.5) + Be*(J*(J+1)) - WeXe*(V+.5)**2 - De*J**2*(J+1)**2 - Ae*(V+.5)*(J+1)*J)

# Dalgarno 1984 table
resten = numpy.array(
      [[     0.  ,   4161.14,   8086.93,  11782.36,  15250.31,  18491.92],
       [   118.5 ,   4273.75,   8193.81,  11883.51,  15345.81,  18581.71],
       [   354.35,   4497.82,   8406.29,  12084.66,  15535.7 ,  18760.28],
       [   705.54,   4831.41,   8722.7 ,  12384.14,  15818.27,  19026.01],
       [  1168.78,   5271.36,   9139.86,  12778.78,  16190.66,  19375.99],
       [  1740.21,   5813.95,   9654.15,  13265.27,  16649.48,  19807.03],
       [  2414.76,   6454.28,  10261.2 ,  13839.18,  17190.36,  20314.77],
       [  3187.57,   7187.44,  10955.68,  14495.46,  17808.76,  20894.94],
       [  4051.73,   8007.77,  11732.12,  15228.88,  18499.06,  21542.14],
       [  5001.97,   8908.28,  12584.8 ,  16033.83,  19256.43,  22251.21],
       [  6030.81,   9883.79,  13507.42,  16904.02,  20074.45,  23016.21],
       [  7132.03,  10927.12,  14493.58,  17833.83,  20947.48,  23831.68],
       [  8296.61,  12031.44,  15537.15,  18816.78,  21869.48,  24691.77],
       [  9523.82,  13191.06,  16632.1 ,  19847.08,  22834.61,  25590.22]])


def restwl(vu,vl,ju,jl):
    """ Uses energy levels measured by Dalgarno, Can J. Physics, 62,1639,1984 
    vu,vl - upper and lower vibrational states
    ju,jl - upper and lower rotational states 
    returns wavelength in microns"""
    if ju >= resten.shape[0] or vu >= resten.shape[1]:
        return 0
    dl = .01/(resten[ju][vu]-resten[jl][vl])
    return dl * 1e6

transdiff = {'S':2,'Q':0,'O':-2}

def linename_to_restwl(linename):
    """ Parse a line name of the form S(1) 1-0, Q(2) 2-1, or 0-0 S(0), etc."""
    upper,lower = re.compile('([0-9]*)-([0-9]*)').search(linename).groups()
    transtype,jl = re.compile('([SQO])\(([0-9]*)\)').search(linename).groups()
    ju = int(jl) + transdiff[transtype]
    rwl = restwl(int(upper),int(lower),ju,int(jl))
    return rwl

def linename_to_restwl_txt(linelistfile = '/Users/adam/work/IRAS05358/code/linelist.txt',outfile='/Users/adam/work/IRAS05358/code/newlinelist.txt'):

    lines = readcol.readcol(linelistfile,fsep='|',twod=False,dtype='S')
    outf = open(outfile,'w')

    for line in transpose(lines):
        name = line[0]
        jre = re.compile('\(([0-9]*)\)').search(name)
        if jre == None:
            print >>outf, "%10s|%10s" % (line[0],line[1])
            continue
        else:
            jl = int( jre.groups()[0] )
        if name[4] == 'S':
            ju = jl + 2
        elif name[4] == 'Q': 
            ju = jl
        elif name[4] == 'O':
            ju = jl - 2
        else:
            print >>outf, "%10s|%10s" % (line[0],line[1])
            continue
        vu = int( name[0] )
        vl = int( name[2] )
        rwl = restwl(vu,vl,ju,jl)
        if rwl == 0:
            rwl = float(line[1])
        print >>outf,"%10s|%10.8f" % (name,rwl)


def aval(v,ju,jl):
    """
    Lookup table for Einstein-A value as a function of 
    vibrational level, upper/lower J level
    Values from: http://www.jach.hawaii.edu/UKIRT/astronomy/calib/spec_cal/h2_s.html
    """
    if v==1:
        if jl==0 and ju-jl==2: return 2.53e-7 
        elif jl==1 and ju-jl==2: return 3.47e-7 
        elif jl==2 and ju-jl==2: return 3.98e-7 
        elif jl==3 and ju-jl==2: return 4.21e-7 
        elif jl==4 and ju-jl==2: return 4.19e-7 
        elif jl==5 and ju-jl==2: return 3.96e-7 
        elif jl==6 and ju-jl==2: return 3.54e-7 
        elif jl==7 and ju-jl==2: return 2.98e-7 
        elif jl==8 and ju-jl==2: return 2.34e-7 
        elif jl==9 and ju-jl==2: return 1.68e-7 
        elif jl==1 and ju-jl==0: return 4.29e-7 
        elif jl==2 and ju-jl==0: return 3.03e-7 
        elif jl==3 and ju-jl==0: return 2.78e-7 
        elif jl==4 and ju-jl==0: return 2.65e-7 
        else: return 0
    elif v==2:
        if jl==0 and ju-jl==2: return 3.68e-7 
        elif jl==1 and ju-jl==2: return 4.98e-7 
        elif jl==2 and ju-jl==2: return 5.60e-7 
        elif jl==3 and ju-jl==2: return 5.77e-7 
        elif jl==4 and ju-jl==2: return 5.57e-7 
        else: return 0
    elif v==3:
        if jl==0 and ju-jl==2: return 3.88e-7 
        elif jl==1 and ju-jl==2: return 5.14e-7 
        elif jl==2 and ju-jl==2: return 5.63e-7 
        elif jl==3 and ju-jl==2: return 5.63e-7 
        elif jl==4 and ju-jl==2: return 5.22e-7 
        elif jl==5 and ju-jl==2: return 4.50e-7 
        else: return 0
    else: return 0

aval_vect=numpy.vectorize(aval)


def modelspec(x,T,A,w,dx,op,Ak=0,extinction=False):
    model = x * 0
    A=abs(A)
    # assume width, shift given in velocity:
    w = w*mean(x)/3e5
    dx = dx*mean(x)/3e5
    for v in xrange(1,6):
        for j in xrange(1,14):
            if (j % 2):  # ortho/para multiplier
                mult=op
            else: 
                mult=1
            # S branch
            wl = restwl(v,v-1,j,j-2) * 10**4
            model += A*mult*(2*j+1)*aval(v,j,j-2)*exp(-h2level_energy(v,j)/(k*T)) * exp( - ( x - wl - dx )**2 / (2*w**2) )
            # Q branch
            wl = restwl(v,v-1,j,j) * 10**4
            model += A*mult*(2*(j)+1)*aval(v,j,j)*exp(-h2level_energy(v,j)/(k*T)) * exp( - ( x - wl - dx )**2 / (2*w**2) )
    if extinction:
        Al = Ak * (x/22000.0)**(-1.75)
        model *= exp(-Al)
    return model

def testnone(x):
    return int(x != None)

def nonetozero(x):
    if x == None: return 0
    else: return x

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

def twoTmodelspec(x,T1,A1,w1,dx1,op1,T2,A2,w2,dx2,op2):
    sumspec = modelspec(x,T1,A1,w1,dx1,op1) + modelspec(x,T2,A2,w2,dx2,op2)
    return sumspec

def twoTmodelpars(fixpars=[None,None,None,None,None,None,None,None,None,None],
        minpars=[200.0,0,0,-500.0,1.0,200.0,0,0,-500.0,1.0],
        maxpars=[15000.0,None,1000.0,500.0,5.0,15000.0,None,1000.0,500.0,5.0],
        params=[2000,5e-9,3.1,-2.0,3,2000,5e-9,3.1,-2.0,3]):

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
                {'n':4,'value':params[4],'limits':[minpars[4],maxpars[4]],'limited':[limitedmin[4],limitedmax[4]],'fixed':fixed[4],'parname':"ORTHOtoPARA",'error':0},
                {'n':5,'value':params[5],'limits':[minpars[5],maxpars[5]],'limited':[limitedmin[5],limitedmax[5]],'fixed':fixed[5],'parname':"TEMPERATURE",'error':0},
                {'n':6,'value':params[6],'limits':[minpars[6],maxpars[6]],'limited':[limitedmin[6],limitedmax[6]],'fixed':fixed[6],'parname':"SCALE",'error':0},
                {'n':7,'value':params[7],'limits':[minpars[7],maxpars[7]],'limited':[limitedmin[7],limitedmax[7]],'fixed':fixed[7],'parname':"WIDTH",'error':0},
                {'n':8,'value':params[8],'limits':[minpars[8],maxpars[8]],'limited':[limitedmin[8],limitedmax[8]],'fixed':fixed[8],'parname':"SHIFT",'error':0},
                {'n':9,'value':params[9],'limits':[minpars[9],maxpars[9]],'limited':[limitedmin[9],limitedmax[9]],'fixed':fixed[9],'parname':"ORTHOtoPARA",'error':0}]
    return parinfo

def modpar(parinfo,fieldnum,value=None,fixed=False,lowlim=None,uplim=None):
    parinfo[fieldnum]['fixed'] = fixed
    if value != None:
        parinfo[fieldnum]['value'] = value
    if lowlim != None:
        parinfo[fieldnum]['limits'][0] = lowlim
        parinfo[fieldnum]['limited'][0] = True
    if uplim != None:
        parinfo[fieldnum]['limits'][1] = uplim
        parinfo[fieldnum]['limited'][1] = True


