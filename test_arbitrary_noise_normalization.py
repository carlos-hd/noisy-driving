# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:23:05 2020

@author: QD
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, newaxis
from numpy.fft import irfft, rfftfreq, ifft
from numpy.random import normal
from numpy import sum as npsum

from numpy.fft import rfftfreq, rfft, fft
from tqdm import tqdm

def autocorr(x):
    y = np.correlate(x,x,'full')
    return y[y.size//2:]/(len(x)+1)

def fourier_pos(t, y, window  = True):
    spectrum = abs(fft(np.hanning(len(y))*y))
    N = len(spectrum)//2
    dt = t[1]-t[0]
    fs = 1/dt #scan frequency--> needs to be halfed by nyquist
    freq = np.linspace(0, fs/2, N+1, endpoint=True)[1:]
    spectrum = spectrum[:N]*2/N
    return freq, spectrum

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
    f = freqs#rfftfreq(samples)#*max(freq)*2
    
    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1./samples) # Low frequency cutoff
    ix   = npsum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(psd):
        psd[:ix] = psd[ix]
        #s_scale[:ix] = s_scale[ix]
    #if len(f)>len(psd):
    #    print(len(f), len(psd))
    s_scale = (np.sqrt(2)*psd)**(1/2.)#s_scale**(-exponent/2.)
    
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
    
    return f,y

timesteps = 2**13
endtime = 10
runs = 2000
Gamma = 2*np.pi/80
t = np.linspace(0, endtime, timesteps)

freqs = rfftfreq(timesteps,endtime/timesteps)#[:timesteps//2]
timesteps +=2
freq_array = [(100,150)]
jj = freq_array[0]
t = np.linspace(0, endtime, timesteps)
freq_psd_init =(np.heaviside(freqs-(jj[0]-jj[1]),0)-np.heaviside(freqs-(jj[0]+jj[1]),0))*Gamma/np.pi/endtime/freqs**2+10*Gamma/np.pi/endtime/freqs**3
expect_z = np.zeros((runs-1, timesteps))
expect_x = expect_z.copy()
psd_sim = np.zeros((runs-1, len(freqs)))
for k in tqdm(range(runs)):
    #[1:-1]#Perturbtime?
    #white_noise = np.random.normal(0, 1, timesteps)
    ff, arbitrary_noise =  psd_gaussian(freq_psd_init, freqs)
    noise = arbitrary_noise
    #brown = brownian(0, timesteps, t[1],Gamma/(t[1]))#arbitrary_noise
    noise-= noise[0]
    #noise = cn.powerlaw_psd_gaussian(2, timesteps)*(Gamma)*(t[1])
    #if k == 0:
    #    noise = [0]*t
    noise -= noise[0]
    f, s = fourier_pos(t, autocorr(noise))
    psd_sim[k-1,:] = s
#"""  

psd = np.mean(psd_sim, axis = 0)
fig2, ax2 = plt.subplots(2,1,figsize = (20,15))
ix = 0
c = 'C'+str(ix)

#psd_avg = np.mean(psd_sim, axis = 0)
ax2[0].plot(freqs, freq_psd_init*freqs**2, '--', label = 'desired PSD', color = c)
#ax2[0].semilogy(freqs, psd*freqs**2, label = 'averaged PSD of noise', color = c)
ax2[0].loglog(freqs, psd*freqs**2, label = 'averaged PSD of noise', color = c)
ax2[0].legend()
ax2[0].set_xlabel('Forier frequency')

ax2[1].plot(t, noise, color = c)
ax2[1].set_xlabel('Time [1/Omega]')
#ax2[0].set_ylim(0.5,1.5)