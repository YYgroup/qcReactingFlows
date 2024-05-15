# %%

import numpy as np
import pyutils as pu

# %%

C = 4
D = 1
S = -0.2
wx = 2
L = wx * np.pi

nqx = 8

nqy = 10
wy = 8

time = 0.3

#p_list = np.array([0.0003, 0.0001, 0.00001, 0.000001])
#label_list = ['Noisy 99.97\%', 'Noisy 99.99\%', 'Noisy 99.999\%', 'Noisy 99.9999\%']
p_list = np.array([0.0003, 0.0001, 0.00001])
label_list = ['Noisy 99.97\%', 'Noisy 99.99\%', 'Noisy 99.999\%']
ls_list = [':', '-.', '--']
c_list = ['b', 'orange', 'r']

# %%

# solution

k_s = np.array([1, 3])
k_c = np.array([2,])

# %%

nx = 2**nqx

x = np.linspace(-L/2, L/2, num=nx, endpoint=False)

# %%

data_Q = np.load('1DPeriodicSpectralQuantumC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,wx,nqx,wy,nqy))

# %%

pc = pu.fig.PlotConfig()

# %%

# plot profiles

fig, ax = pc.get_simple()

mspace = 32

#ut = np.zeros(x.size)
#for k in k_s:
#    ut = ut + np.sin(k*(x-C*time)) * np.exp((-D*np.square(k)+S)*time)
#for k in k_c:
#    ut = ut + np.cos(k*(x-C*time)) * np.exp((-D*np.square(k)+S)*time)
#ax.plot(x, ut, '-k', label='Analytic')

ax.plot(x, data_Q['y'][:,3], '-k', label='Ideal')

for i, p in enumerate(p_list):
    data_NS = np.load('1DPeriodicSpectralNoisyQuantumSingleC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}p{:g}t{:g}.npz'.format(C,D,S,wx,nqx,wy,nqy,p,time))
    ax.plot(x, data_NS['y'], 
            c=c_list[i], ls=ls_list[i], label=label_list[i])

ax.set_xlim([-L/2, L/2])
ax.set_ylim([-1.0, 1.0])

ax.set_xticks(np.linspace(-L/2, L/2, num=5))
ax.set_xticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])

ax.set_yticks(np.linspace(-1, 1, num=5))

ax.set_xlabel('$x$')
ax.set_ylabel('$\phi$')

#ax.legend(frameon=False, loc='lower left')
ax.legend(frameon=False, loc='upper right',
          ncols=2, columnspacing=0.5,
          handlelength=2.2, handletextpad=0.3)

# %%

fig.savefig('fig1DPeriodicSeriesNoisyAccuracywy{:g}ny{:g}.png'.format(wy, nqy, p), dpi=600)
fig.savefig('fig1DPeriodicSeriesNoisyAccuracywy{:g}ny{:g}.eps'.format(wy, nqy, p))

# %%
