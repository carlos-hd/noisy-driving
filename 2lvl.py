import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check
#import colorednoise_new_norm as cn
import arbitrary_noise as an
from scipy.interpolate import interp1d
from numpy.fft import rfftfreq, fft
from tqdm import tqdm

plt.rcParams.update({'font.size': 22})


opts = Options(store_states=True, store_final_state=True, ntraj=200)
omega = 2.*np.pi*40

def twobasis():
    return np.array([basis(2, 0), basis(2, 1)], dtype=object)


from math import sqrt
from scipy.stats import norm

def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta**2*dt)

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def productstateZ(up_atom, down_atom, N):
    up, down = twobasis()
    return up
    #pbasis = np.full((2,2), down)
    #pbasis[up_atom] = down
    #pbasis[down_atom] = up
    
    #return tensor(pbasis)


def productstateX(m, j, N):
    up, down = twobasis()
    return (up + down).unit()
    #pbasis = np.full((2,2), down)
    #pbasis[m] = (up + down).unit()
    #pbasis[j] = (up + down).unit()
    #return tensor(pbasis)


def sigmap(m, N):
    up, down = twobasis()
    return up * down.dag()
    #oplist = np.full(N, identity(2))
    #oplist[m] = up * down.dag()
    #return tensor(oplist)


def sigmam(m, N):
    up, down = twobasis()
    return down*up.dag()
    #oplist = np.full(N, identity(2))
    #oplist[m] = down * up.dag()
    #return tensor(oplist)


def sigmaz(j, N):
    return  Qobj([[1, 0], [0, -1]])
    #oplist = np.full(N, identity(2))
    #oplist[j] = Qobj([[1, 0], [0, -1]])
    #return tensor(oplist)


def sigmax(j, N):
    return Qobj([[0, 1], [1, 0]])
    #oplist = np.full(N, identity(2))
    #oplist[j] = Qobj([[0, 1], [1, 0]])
    #return tensor(oplist)


def sigmay(j, N):
    return Qobj([[0, -1j], [1j, 0]])
    #oplist = np.full(N, identity(2))
    #oplist[j] = Qobj([[0, -1j], [1j, 0]])
    #return tensor(oplist)


def MagnetizationZ(N):
    sum = 0
    for j in range(0, N):
        sum += sigmaz(j, N)
    return sum / N


def MagnetizationX(N):
    sum = 0
    for j in range(0, N):
        sum += sigmax(j, N)
    return sum / N

def MagnetizationY(N):
    sum = 0
    for j in range(0, N):
        sum += sigmay(j, N)
    return sum / N


def H(J, N):
    H = 0
    for j in range(0, N):
        H += 1 * omega/2 * sigmaz(j, N) 
    return H


def H1(j0,N):
    H = 0
    for j in range(0, N):
        H += -j0 * (sigmap(j, N))
    return H


def H2(j0,N):
    H = 0
    for j in range(0, N):
        H += -j0 * (sigmam(j, N))
    return H

def L(Gamma, N):
    up, down = twobasis()
    L = np.sqrt(Gamma) * sigmam(j,N)#(down * up.dag()+up*down.dag())
    return L

def autocorr(x):
    y = np.correlate(x,x,'full')
    return y[y.size//2:]/(len(x)+1)
    #result = [pd.Series(x).autocorr(n) for n in range(len(x)-1)]
    #return result

def fourier_pos(t, y, window  = True):
    spectrum = abs(fft(np.hanning(len(y))*y))
    N = len(spectrum)//2
    dt = t[1]-t[0]
    fs = 1/dt #scan frequency--> needs to be halfed by nyquist
    freq = np.linspace(0, fs/2, N+1, endpoint=True)[1:] #only returns positive frequencies
    spectrum = spectrum[:N]*2/N
    return freq, spectrum

def ohmic_spectrum(w):
    if w == 0.0: # dephasing inducing noise
        return gamma1
    else: # relaxation inducing noise
        return gamma1 / 2 * (w / (2 * np.pi)) * (w > 0.0)


def rotmat(t):
    om = omega
    return Qobj([[np.cos(om*t), np.sin(om*t), 0],[-np.sin(om*t), np.cos(om*t), 0],[0,0,1]])

from numpy import sqrt, newaxis
from numpy.fft import irfft, rfftfreq, ifft
from numpy.random import normal
from numpy import sum as npsum

def psd_gaussian(psd, freq, fmin=0):

    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    
    # Make sure size is a list so we can iterate it and assign to it.
    size = len(psd)*2
    try:
        size = list(size)
    except TypeError:
        size = [size]
    
    # The number of samples in each time series
    samples = size[-1]
    
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples, d = freq[1]-freq[0])#*max(freq)*2
    
    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1./samples) # Low frequency cutoff
    ix   = npsum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(psd):
        psd[:ix] = psd[ix]
        #s_scale[:ix] = s_scale[ix]
    #if len(f)>len(psd):
    #    print(len(f), len(psd))
    s_scale = np.sqrt(psd)#(psd)**(1/2.)#s_scale**(-exponent/2.)
    
    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * sqrt(npsum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(psd) #len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]
    
    # Generate scaled random power + phase
    sr = normal(scale=s_scale, size=size)
    si = normal(scale=s_scale, size=size)
    
    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2): si[...,-1] = 0
    
    # Regardless of signal length, the DC component must be real
    si[...,0] = 0
    
    # Combine power + corrected phase to Fourier components
    s  = (sr + 1J * si)
    
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1)  * samples /2
    
    return y

def phase_corrected(size, omega_t, omega_0, dt):
    phi = np.zeros((size))
    phi = dt*(omega_t)
    return np.cumsum(phi)#*omega_0
def spectrum_new(w):
    if w == 0.0:
        return Gamma/np.pi
    else:
        return Gamma/np.pi#np.pi#(Gamma*np.pi+np.exp(w/Gamma))#*(np.heaviside(w-50,1))


N = 1
Gamma = 2*np.pi/80 #*np.sqrt(1.4142)  #will be sqrtd in Lindblad operator
#print(H(0, N))

j = 2.*np.pi*1 #Omega


#print(H1(j,N))
#print(H2(j,N))
##print('Initial state ....')
#print(productstateZ(0, 0, N))


timesteps = 2**13
endtime = 12
runs = 500

freqs = rfftfreq(timesteps, endtime/timesteps)#[:timesteps//2]

loaded = np.load('../Linewidth_measurements/probe_data_lock.npz')
probe_f = loaded['f']/j*1e-6
probe_psd = loaded['psd']
wn_psd = np.mean(probe_psd[-6:])
extrap = np.array(((probe_f[-1]+1,probe_f[-1]+2,probe_f[-1]+3, freqs[-1]),(wn_psd,wn_psd,wn_psd,wn_psd)))
f = np.concatenate(([0],probe_f,extrap[0,:]))
psd = np.concatenate(([probe_psd[0]],probe_psd, extrap[1,:]))
probe = interp1d(f, psd)

timesteps+=2 #for arbitrary noise, so calculated psd of noise (times samples) gives same amount of points as input psd 
#freq_array =  [(11,10), (15,10), (40,10), (18,10), (27,10), (61,10), (51,10), (105,10), (93, 10)]#[ (60,1), (60,10), (60,30), (60,70)] #
#freq_array =  [(11,10), (12,10), (13,10), (14,10), (15,10), (16,10), (17,10), (18,10), (19,10), (20,10)]
#freq_array = [(300,5), (300,50), (300,200), (300,600)]
freq_array = [(10,10), (15,10), (30,10)]# ,(300,300), (400,400), (500,500)]
freq_array = [(1,1),(4,4), (20,20), (100,100) ]
kf_strengths = [5]#, 10, 20, 30]
#freq_array = [(500,500)]
fig1, ax1 = plt.subplots(figsize = (15,10))
fig11, ax11 = plt.subplots(figsize = (15,10))
fig2, ax2 = plt.subplots(2,1,figsize = (20,15))
ix = 0
#for jj in freq_array:
for kf in kf_strengths:
    #if freq_array[-1][0]>=max(freqs):
    #    print(max(freqs))
    #    break
    t = np.linspace(0, endtime, timesteps)
    #freq_psd_init =(np.heaviside(freqs-(jj[0]-jj[1]),0)-np.heaviside(freqs-(jj[0]+jj[1]),0))*Gamma/np.pi/endtime/freqs**2 #+ 10*Gamma/np.pi/endtime/
    freq_psd_init = Gamma*np.sqrt(t[1])*np.pi/endtime/freqs**2# + kf*Gamma/np.pi/endtime/freqs**3
    #freq_psd_init = probe(freqs)*np.sqrt(t[1])/endtime/freqs**2   #probe_psd_failed :(
    expect_z = np.zeros((runs-1, timesteps))
    expect_x = expect_z.copy()
    psd_sim = np.zeros((runs-1, len(freqs)))
    for k in tqdm(range(runs)):
        t = np.linspace(0, endtime, timesteps)#[1:-1]#Perturbtime?
        #white_noise = np.random.normal(0, 1, timesteps)
        arbitrary_noise =  (psd_gaussian(freq_psd_init, freqs))
        noise = arbitrary_noise
        #brown = brownian(0, timesteps, t[1],Gamma/(t[1]))#arbitrary_noise
        
        #fnoise = cn.powerlaw_psd_gaussian(0, timesteps)*(Gamma/np.pi)#*(t[1])
        #noise = phase_corrected(timesteps, fnoise, omega, t[1])
        f, s = fourier_pos(t, autocorr(noise))
        if k == 0:
            noise = [0]*t
            continue
        noise -= noise[0]
        func1 = lambda t: np.exp(-1j * (t *omega + noise) ) + np.exp( 1j * ( t * omega + noise) )
        noisy_data1 = func1(t)*0.5 #0.5 for cosinus to exp transformation
        S1 = Cubic_Spline(t[0], t[-1], noisy_data1)
        
        #func1 = lambda t: -1j* np.exp(1j * (t * omega + noise-noise[0])) + 1j*np.exp( - 1j * (omega*t +noise-noise[0]))
        noisy_data1 = func1(t)*0.5 #0.5 for cosinus to exp transformation
        S2 = Cubic_Spline(t[0], t[-1], noisy_data1)
        
        #func2 = lambda t: np.array(noise) #- 1j*np.exp(1j * (omega*t + noise-noise[0])) + 1j* np.exp( - 1j * (t  * omega + noise-noise[0]) )
        #noisy_data2 = func2(t)
        #S2 = Cubic_Spline(t[0], t[-1], noisy_data2)
        
        
        Exps = [sigmap(j,N), MagnetizationZ(N), sigmap(j,N)*sigmap(j,N).dag()]
        
        opts = Options(store_states=True, store_final_state=True, ntraj=200)
        
        result2 = mesolve([H(0, N), [H1(j,N), S1], [H2(j,N), S2]], productstateZ(0, 0, N), t, [], Exps, options=opts)
        if k == 0: 
            result  = result2
            continue #therefore expect_z is one element smaller
        expect_z[k-1,:] = result2.expect[2] 
        expect_x[k-1,:] = result2.expect[0]
        psd_sim[k-1,:] = s
    #"""  
    func1 = lambda t:  np.exp(-1j * (t * omega)) +np.exp( 1j * (omega*t ))
    noisy_data3 = func1(t)*0.5
    S1 = Cubic_Spline(t[0], t[-1], noisy_data3)
    
    func2 = lambda t: np.exp(1j * (omega*t )) + np.exp( - 1j * (t  * omega ) )
    noisy_data2 = func2(t)*0.5
    S2 = Cubic_Spline(t[0], t[-1], noisy_data3)
    
    result3 = mesolve([H(0, N), [H1(j,N), S1], [H2(j,N), S2]], productstateZ(0, 0, N), t,[L(Gamma, N)], Exps,options=opts,progress_bar=True)
    """
    fig, ax = plt.subplots(2, 3)
    
    ax[0, 0].plot(t, np.imag(func1(t)))
    #ax[0, 0].plot(t, S1(t), lw=2)
    
    ax[0, 1].plot(t, np.imag(func2(t)))
    #ax[0, 1].plot(t, S2(t), lw=2)
    
    ax[0, 2].plot(t, np.imag(func2(t))+np.imag(func1(t)))
    times = t
    #ax[1, 1].plot(times, result2.expect[1], label="MagnetizationZ");
    #ax[1, 1].plot(times, result2.expect[0], label="MagnetizationX");
    ax[1, 1].plot(times, np.mean(expect_1, axis = 0), label = "MagnetizationZ - avg, white noise")
    ax[1, 1].plot(times, result3.expect[1], label ="Magnetization Z, gamma = "+str(Gamma))
    ax[1, 1].set_xlabel('Time [1/Omega]');
    ax[1, 1].set_ylabel('Magentization');
    ax[1,1].legend()
    
    plt.show()
    """
    psi0 = productstateZ(0, 0, N)
    a_ops = [[sigmax(j,N), spectrum_new]]#[-H1(j,N)*S1+H2(j,N)*S2]
    Exps = [MagnetizationX(N), MagnetizationY(N), sigmap(j, N).dag()*sigmap(j, N)]
    opts = Options(store_states=True, store_final_state=True, ntraj=200)
    nosigma = sigmam(1, N).dag()*sigmam(j, N)
    H_br = -0*omega*nosigma+j/2*sigmax(j,N)

    result1 = brmesolve(H_br, psi0, t, a_ops = a_ops, e_ops = Exps, use_secular = False)
    magnetizations = np.array(result1.expect)
    
    psi = np.zeros_like(magnetizations)
    for i in range(len(t)):
        psi[:,i] = (rotmat(t[i]).dag()*magnetizations[:,i])
        
    mean_x = np.mean(expect_x, axis = 0)
    var_x = np.sqrt(np.var(expect_x, axis = 0, ddof = 1))
    

    mean_z = np.mean(expect_z, axis = 0)
    var_z = np.sqrt(np.var(expect_z, axis = 0, ddof = 1))
    psd = np.mean(psd_sim, axis = 0)
    jj = kf
    
    c = 'C'+str(ix)
    ax1.plot(t, 1-mean_z, label = "".format(kf = kf), color = c)#central freq.= {c_f}, full width = {width}".format(c_f = jj[0], width = 2*jj[1]),color = c)    
    ax1.fill_between(t, 1-(mean_z-var_z), 1-(mean_z+var_z), alpha = 0.2)
    #ax1.plot(t, np.mean(expect_x, axis = 0), '--', label = "MagnetizationX - central freq.= {c_f}, width = {width}".format(c_f = jj[0], width = 2*jj[1]))#, color = c)
    #ax1.plot(t, result2.expect[2], label = "Magnetizaton Y")
    #ax2.plot(f, psd)
    #ax1.plot(t, psi[2], label = 'redfield', color='C2')
    
    ax1.set_xlabel('Time [1/Omega]');
    ax1.set_ylabel('Population ');
    ax1.set_xlim(-0.1,10.5)
    ix+=1
    
    ax11.plot(t[:100], noisy_data1[:100], 'o-', color =  c)
    ax11.plot(t[:100], noisy_data3[:100], color = 'k')
    ax11.set_xlabel('Time [1/Omega]')
    ax11.set_ylabel('E(t)/E$_0$')
    ax11.set_xlim(-t[1]/2, 2*np.pi/omega)
    #psd_avg = np.mean(psd_sim, axis = 0)
    ax2[0].plot(freqs, freq_psd_init*freqs**2, '--', label = 'desired PSD', lw = 2, color = c)
    #ax2[0].semilogy(freqs, psd*freqs**2, label = 'averaged PSD of noise', color = c)
    ax2[0].semilogy(freqs, psd*freqs**2, label = 'averaged PSD of noise', color = c)
    ax2[0].legend()
    ax2[0].set_xlabel('Fourier frequency')
    
    ax2[1].plot(t, noise, color = c)
    ax2[1].set_xlabel('Time [1/Omega]')
    ax2[1].set_ylabel('$\phi$ [rad.]')
#ax1[0].plot(t, result.expect[1], '--', label = 'no noise') #only for no noise thing
ax1.plot(t, 1-result3.expect[2], label ="Lindbald operators, gamma = $\Omega$/{gg}".format(gg = j/Gamma), color = 'k')
ax1.legend()
fig1.show()
#fig2.show()
"""
x  =np.mean(expect_x, axis = 0)*np.cos(-omega*t)-result2.expect[2]*np.sin(-omega* t)
y  =np.mean(expect_x, axis = 0)*np.sin(-omega*t)+result2.expect[2]*np.cos(-omega* t)
z = np.mean(expect_z, axis = 0)

bb = Bloch()
bb.add_points([x,y,z])
bb.show()
"""