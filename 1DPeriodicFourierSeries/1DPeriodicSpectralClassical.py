# %%

import numpy as np
from scipy import fft, integrate
import pyutils as pu

# %%

# Parameters

time = 2

C = 4
D = 1
S = -0.2
wx = 2
L = wx*np.pi

# number of qubits for physical space
nq = 8
nx = 2 ** nq

# %%

# Initialization

x = np.linspace(-L/2, L/2, num=nx, endpoint=False)

k_s = np.array([1, 3])
k_c = np.array([2, ])
u_init = np.zeros(x.size)
for k in k_s:
    u_init = u_init + np.sin(k*x)
for k in k_c:
    u_init = u_init + np.cos(k*x)

# %%

kx = fft.fftfreq(nx)*nx*2*np.pi/L
fx = fft.fft(u_init)

def rhs(t, y):
    return (-1.j*C*kx-D*np.square(kx)+S)*y

sol = integrate.solve_ivp(rhs, [0, time], fx, method='RK23', t_eval=np.linspace(0, time, num=21))

results = fft.ifft(sol.y, axis=0)

# %%

pc = pu.fig.PlotConfig()

fig, ax = pc.get_simple()

#ax.plot(x, u_init, '-k')

for i in range(0,3):
    idx = 2*i
    t = sol.t[idx]

    ut = np.zeros(x.size)
    for k in k_s:
        ut = ut + np.sin(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)
    for k in k_c:
        ut = ut + np.cos(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)
    ax.plot(x, ut, '-k', label='Analytic')

    ax.plot(x, results[:,idx].real, '-.r', label='Classical')

ax.legend(frameon=False)

ax.set_xlim([-L/2, L/2])

# %%

np.savez('1DPeriodicSpectralClassicalC{:g}D{:g}S{:g}wx{:g}nx{:g}'.format(C,D,S,wx,nq),
         x = x, t = sol.t, y = results)

# %%
