# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:01:48 2020

@author: QD
"""


import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check
#import colorednoise as cn
#import arbitrary_noise as an
from numpy.fft import rfftfreq, fft
from tqdm import tqdm

plt.rcParams.update({'font.size': 22})


opts = Options(store_states=True, store_final_state=True, ntraj=200)
omega = 2.*np.pi*40

def twobasis():
    return np.array([basis(2, 0), basis(2, 1)], dtype=object)

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
        H += -j0 * (sigmap())
    return H


def H2(j0,N):
    H = 0
    for j in range(0, N):
        H += -j0 * (sigmam())
    return H

def L(Gamma, N):
    up, down = twobasis()
    L = np.sqrt(Gamma) * (down * up.dag()+up*down.dag())
    return L

def ohmic_spectrum(w):
    if w == 0.0: # dephasing inducing noise
        return gamma1
    else: # relaxation inducing noise
        return gamma1  / 2 * (w / (2 * np.pi)) * (w > 0.0)

def spectrum(w):
    if w == 0.0:
        return Gamma/np.pi
    elif w == 1.5:
        return 10000000*Gamma
    else:
        return 10000000*Gamma/np.pi#+Gamma*10**3/w#np.pi#(Gamma*np.pi+np.exp(w/Gamma))#*(np.heaviside(w-50,1))


j = 2.*np.pi*1 #Omega
def rotmat(t):
    #return np.exp(-1j*omega*t)*sigmam()*sigmam().dag()+sigmap()*sigmap().dag()
    om = omega
    return Qobj([[np.cos(om*t), np.sin(om*t), 0],[-np.sin(om*t), np.cos(om*t), 0],[0,0,1]])
N = 1
Gamma = 2*np.pi/80#*np.sqrt(1.4142)
#print(H(0, N))
gamma1  = Gamma
j = 2.*np.pi*1 #Omega
psi0 = basis(2,1).unit()
#print(H1(j,N))
#print(H2(j,N))
##print('Initial state ....')
#print(productstateZ(0, 0, N))

#kappa = 0.2
spectra_cb =  (Gamma*np.pi, Gamma*np.pi)

timesteps = 2**14
endtime = 12

t = np.linspace(0, endtime, timesteps)

func1 = lambda t: 1* np.exp(-1j * omega * t)+1* np.exp(1j * omega * t)#2*np.cos(omega * t) #
noisy_data1 = func1(t)*0.5 #0.5 for cosinus to exp transformation
S1 = Cubic_Spline(t[0], t[-1], noisy_data1)

func2 = lambda t: 1 *np.exp( 1j * t * omega)+ 1* np.exp(-1j * omega * t)
noisy_data2 = func2 (t)*0.5
S2 = Cubic_Spline(t[0], t[-1], noisy_data2)

c_ops = [np.sqrt(Gamma)*sigmam()]
#c_ops = [np.sqrt(Gamma)* sigmam()]
aops = [[Gamma*sigmax(j,N), spectra_cb]]
a_ops = [[sigmax(j,N), spectrum]]#[-H1(j,N)*S1+H2(j,N)*S2]

Exps = [MagnetizationX(N), MagnetizationY(N), sigmap()*sigmap().dag()]

opts = Options(store_states=True, store_final_state=True, ntraj=200)

nosigma = sigmam().dag()*sigmam()
H_br = -0*omega*nosigma+j/2*sigmax(j,N)#H(0,N) #[H(0, N), [H1(j,N), S1], [H2(j,N), -S2]]
#result2 = brmesolve(H_br , psi0, t, a_ops = [Gamma*np.pi*1.4142* sigmax(j,N)], e_ops = Exps, spectra_cb = [spectrum], use_secular = False)
result3 = brmesolve(H_br, psi0, t, a_ops = a_ops, e_ops = Exps, use_secular = False)
result = mesolve([H(0, N), [H1(j,N), S1], [H2(j,N), S2]], psi0, t, c_ops, e_ops = Exps, options = opts )

#plot_expectation_values([result,  result3])

magnetizations = np.array(result3.expect)
fig, ax = plt.subplots(3,1,figsize= (15,15), sharex = True)
fig1, ax1 = plt.subplots(figsize=(15,10))
ax[0].plot(t, result.expect[0])
ax[1].plot(t, result.expect[1])
ax[2].plot(t, result.expect[2], label  = 'Lindblad ME')
#ax[2].legend()


psi = np.zeros_like(magnetizations)
for i in range(len(t)):
    psi[:,i] = (rotmat(t[i]).dag()*magnetizations[:,i])
    

ax[0].plot(t, -psi[0], label = 'Magnetization X')
ax[1].plot(t, -psi[1], label = 'Magnetization Y')
ax[2].plot(t, psi[2], label = 'pop. ground state')

ax[0].set_ylabel('Magentization X')
ax[1].set_ylabel('Magentization Y')
ax[2].set_ylabel('population')
ax[2].set_xlabel('Time [1/$\Omega$]')

ax1.plot(t, psi[2])
ax1.plot(t, result.expect[2], color = 'k')
ax1.set_ylabel('Population')
ax1.set_xlabel('Time [1/$\Omega$]')
ax1.set_xlim(-0.1,10.5)


#ax[3].plot(t, psi[2]-result.expect[2])
#ax[3].set_ylabel('diff BRME-ME')
# magnetizations = np.array(result3.expect)
# psi = np.zeros_like(magnetizations)
# for i in range(len(t)):
#     psi[:,i] = rotmat(t[i]).dag()*magnetizations[:,i]
    
# ax[0].plot(t, -psi[0], label = 'Magnetization X')
# ax[1].plot(t, -psi[1], label = 'Magnetization Y')
# ax[2].plot(t, psi[2], label = 'Magnetization Z')
    
plt.figure(figsize= (15,10))
rel_diff = psi[2]/result.expect[2]-1
plt.plot(t, rel_diff)
plt.ylim(-0.5,0.5)

#fre = np.linspace(0,10,100)
#plt.figure()
#plt.plot(fre, spectrum(fre))