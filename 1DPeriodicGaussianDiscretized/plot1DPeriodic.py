# %%

import numpy as np
import pyutils as pu

# %%

L = 30
C = 10
D = 0.5
S = -1

nqx = 8
nqy = 10
wy = 8

mu = -10

# %%

nx = 2**nqx

x = np.linspace(-L/2, L/2, num=nx, endpoint=False)

# %%

data_QC = np.load('1DPeriodicDiscretizedSeparateC{:g}D{:g}S{:g}L{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,L,nqx,wy,nqy))
data_FD = np.load('1DPeriodicDiscretizedFiniteDifferenceC{:g}D{:g}S{:g}L{:g}nx{:g}.npz'.format(C,D,S,L,nqx))

# %%

pc = pu.fig.PlotConfig(subplot_ratio=0.8)

# %%

# plot profiles

fig, ax = pc.get_simple()

for i in range(5):
    t = data_QC['t'][5*i]

    ut = np.exp(-np.square(x-mu-C*t)/(1+4*D*t)) / np.sqrt(1+4*D*t) * np.exp(S*t)
    ax.plot(x, ut, '-k')

    ax.plot(x, data_FD['y'][:,5*i], '--b')

    ax.plot(x, data_QC['y'][:,5*i], '-.r')

    if i != 0:
        ax.text(5*i-10.5, ut.max()+0.03, '$t={:.1f}$'.format(t))

ax.text(-9, 0.95, '$t=0$')

ax.set_xlim([-L/2, L/2])
ax.set_ylim([-0.02, 1.02])

ax.set_xlabel('$x$')
ax.set_ylabel('$\phi$')

ax.legend(['Analytical', 'Classical FD', 'Quantum FD'], frameon=False)

# %%

fig.savefig('fig1DPeriodicGaussianFDwy{:g}ny{:g}.png'.format(wy, nqy), dpi=600)
fig.savefig('fig1DPeriodicGaussianFDwy{:g}ny{:g}.eps'.format(wy, nqy))

# %%

# process error

t_QC = data_QC['t']
L2_QC = np.zeros(t_QC.size)

for i, t in enumerate(t_QC):
    # analytic
    ut = np.exp(-np.square(x-mu-C*t)/(1+4*D*t)) / np.sqrt(1+4*D*t) * np.exp(S*t)

    L2_QC[i] = np.sqrt(np.sum(np.square(data_QC['y'][:,i]-ut))) / np.sqrt(np.sum(np.square(ut)))

t_FD = data_FD['t']
L2_FD = np.zeros(t_FD.size)

L2_QCFD = np.zeros(t_FD.size)

for i, t in enumerate(t_FD):
    # analytic
    ut = np.exp(-np.square(x-mu-C*t)/(1+4*D*t)) / np.sqrt(1+4*D*t) * np.exp(S*t)

    L2_FD[i] = np.sqrt(np.sum(np.square(data_FD['y'][:,i]-ut))) / np.sqrt(np.sum(np.square(ut)))

    L2_QCFD[i] = np.sqrt(np.sum(np.square(data_FD['y'][:,i]-data_QC['y'][:,i])))

# %%

fig, ax = pc.get_simple()

ax.plot(t_FD[1:], L2_FD[1:], '--sb', label='Classical')
ax.plot(t_QC[1:], L2_QC[1:], '-.or', label='Quantum')

ax.set_yscale('log')

ax.legend(frameon=False)

# %%
