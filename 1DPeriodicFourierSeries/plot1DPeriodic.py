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

# %%

# solution

k_s = np.array([1, 3])
k_c = np.array([2,])

# %%

nx = 2**nqx

x = np.linspace(-L/2, L/2, num=nx, endpoint=False)

# %%

data_Q = np.load('1DPeriodicSpectralQuantumC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,wx,nqx,wy,nqy))
data_D = np.load('1DPeriodicDiscretizedQuantumC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,wx,nqx,wy,nqy))
data_C = np.load('1DPeriodicSpectralClassicalC{:g}D{:g}S{:g}wx{:g}nx{:g}.npz'.format(C,D,S,wx,nqx))

# %%

pc = pu.fig.PlotConfig()

# %%

# plot profiles

fig, ax = pc.get_simple()

mspace = 32

for i in range(4):
    idx = 3 * i
    t = data_C['t'][idx]

    ut = np.zeros(x.size)
    for k in k_s:
        ut = ut + np.sin(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)
    for k in k_c:
        ut = ut + np.cos(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)
    ax.plot(x, ut, '-k')

    ax.plot(x, data_C['y'][:,idx].real, '--b')
    #ax.plot(x[::mspace], data_C['y'][::mspace,idx].real, linestyle='none', marker='s', mfc='none', mec='b')

    ax.plot(x, data_Q['y'][:,idx], '-.r')

    ax.plot(x, data_D['y'][:,idx], ':', c='tab:green')

ax.text(-0.6, -1.5, '$t=0$')
ax.text(-0.3, -1.15, '$t=0.3$')
ax.text( 0.55, -0.8, '$t=0.6$')
ax.text( 2.0, -0.6, '$t=0.9$')

ax.set_xlim([-L/2, L/2])
ax.set_ylim([-1.6, 2.2])

ax.set_xticks(np.linspace(-L/2, L/2, num=5))
ax.set_xticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])

ax.set_xlabel('$x$')
ax.set_ylabel('$\phi$')

ax.legend(['Analytical', 'Classical SP', 'Quantum SP', 'Quantum FD'], frameon=False)

# %%

fig.savefig('fig1DPeriodicSerieswy{:g}ny{:g}.png'.format(wy, nqy), dpi=600)
fig.savefig('fig1DPeriodicSerieswy{:g}ny{:g}.eps'.format(wy, nqy))

# %%

# process error

t_Q = data_Q['t']
L2_Q = np.zeros(t_Q.size)

for i, t in enumerate(t_Q):

    # analytic
    ut = np.zeros(x.size)
    for k in k_s:
        ut = ut + np.sin(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)
    for k in k_c:
        ut = ut + np.cos(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)

    L2_Q[i] = np.sqrt(np.sum(np.square(data_Q['y'][:,i]-ut)))

t_C = data_C['t']
L2_C = np.zeros(t_C.size)

for i, t in enumerate(t_C):

    # analytic
    ut = np.zeros(x.size)
    for k in k_s:
        ut = ut + np.sin(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)
    for k in k_c:
        ut = ut + np.cos(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)

    L2_C[i] = np.sqrt(np.sum(np.square(data_C['y'][:,i].real-ut)))

# %%

fig, ax = pc.get_simple()

ax.plot(t_C[1:], L2_C[1:], '--sb', label='Classical')
ax.plot(t_Q[1:], L2_Q[1:], '-.or', label='Quantum')

ax.set_yscale('log')

ax.legend(frameon=False)

# %%
