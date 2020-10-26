import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check
import colorednoise_new_norm as cn
import arbitrary_noise as an
from numpy.fft import rfftfreq, fft
from tqdm import tqdm

plt.rcParams.update({'font.size': 18})


opts = Options(store_states=True, store_final_state=True, ntraj=200)
omega = 2.*np.pi*100

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
    L = np.sqrt(Gamma) * (down * up.dag()+up*down.dag())
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


N = 1
Gamma = 2*np.pi/100 #*np.sqrt(1.4142)  #will be sqrtd in Lindblad operator
#print(H(0, N))

j = 2.*np.pi*5 #Omega


#print(H1(j,N))
#print(H2(j,N))
##print('Initial state ....')
#print(productstateZ(0, 0, N))


timesteps = 2**13
endtime = 4
runs = 100

freqs = rfftfreq(timesteps, endtime/timesteps)#[:timesteps//2]

timesteps+=2 #for arbitrary noise, so calculated psd of noise (times samples) gives same amount of points as input psd 
#freq_array =  [(11,10), (15,10), (40,10), (18,10), (27,10), (61,10), (51,10), (105,10), (93, 10)]#[ (60,1), (60,10), (60,30), (60,70)] #
#freq_array =  [(11,10), (12,10), (13,10), (14,10), (15,10), (16,10), (17,10), (18,10), (19,10), (20,10)]
#freq_array = [(300,1), (300,5), (300,20), (300,50)]
fig1, ax1 = plt.subplots(2,1,figsize = (20,15))
#fig2, ax2 = plt.subplots(figsize = (20,15))
ix = 0
for jj in freq_array:
    #if freq_array[-1][0]>=max(freqs):
    #    print(max(freqs))
    #    break
    #freq_psd_init = Gamma*(np.heaviside(freqs-(jj[0]-jj[1]),0)-np.heaviside(freqs-(jj[0]+jj[1]),0))
    
    expect_z = np.zeros((runs-1, timesteps))
    expect_x = expect_z.copy()
    psd_sim = np.zeros((runs, len(freqs)))
    for k in tqdm(range(runs)):
        t = np.linspace(0, endtime, timesteps)#[1:-1]#Perturbtime?
        #white_noise = np.random.normal(0, 1, timesteps)
        #arbitrary_noise =  (an.psd_gaussian(freq_psd_init, freqs))*t[1]
        #noise = arbitrary_noise
        #brown = brownian(0, timesteps, t[1],Gamma/(t[1]))#arbitrary_noise
        #if jj[0] == 0:
            #print('rt noise')
            #noise = np.pi/2*np.sign(an.psd_gaussian(freq_psd_init, freqs))
            #noise = cn.powerlaw_psd_gaussian(1, timesteps)#np.random.normal(0, timesteps*0.0005*np.sqrt(np.sqrt(2)/2), timesteps)
        #f, s = fourier_pos(t, autocorr(noise))
        #noise = np.array(brown)
        noise = cn.powerlaw_psd_gaussian(2, timesteps)*(Gamma)*(t[1])
        if k == 0:
            noise = [0]*t
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
        
        
        Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N)]
        
        opts = Options(store_states=True, store_final_state=True, ntraj=200)
        
        result2 = mesolve([H(0, N), [H1(j,N), S1], [H2(j,N), S2]], productstateZ(0, 0, N), t, [], Exps, options=opts)
        if k == 0: 
            result  = result2
            continue #therefore expect_z is one element smaller
        expect_z[k-1,:] = result2.expect[1] 
        expect_x[k-1,:] = result2.expect[0]
        #psd_sim[k,:] = s
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
    mag_z = np.mean(expect_z, axis = 0)
    var_z = np.sqrt(np.var(expect_z, axis = 0, ddof = 1))
    
    c = 'C'+str(ix)
    ax1[0].plot(t, mag_z)#, label = "central freq.= {c_f}, full width = {width}".format(c_f = jj[0], width = 2*jj[1]))#,color = c)
    ax1[0].plot(t, result.expect[1], '--', label = 'no noise') #only for no noise thing
    ax1[0].fill_between(t, mag_z-var_z, mag_z+var_z, alpha = 0.2)
    #ax1.plot(t, np.mean(expect_x, axis = 0), '--', label = "MagnetizationX - central freq.= {c_f}, width = {width}".format(c_f = jj[0], width = 2*jj[1]))#, color = c)
    #ax1.plot(t, result2.expect[2], label = "Magnetizaton Y")
    ax1[0].plot(t, result3.expect[1], label ="with Lindbald operator, gamma = "+str(Gamma))
    ax1[0].legend()
    ax1[0].set_xlabel('Time [1/Omega]');
    ax1[0].set_ylabel('Magnetization Z');
    ix+=1
    
    ax1[1].plot(t[:30], noisy_data1[:30], 'o-')
    ax1[1].plot(t[:30], noisy_data3[:30])
    #psd_avg = np.mean(psd_sim, axis = 0)
    #ax2.plot(freqs, freq_psd_init, '--', label = 'desired PSD', color = c)
    #ax2.plot(freqs, psd_avg, label = 'averaged PSD of noise', color = c)
    #ax2.legend()

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