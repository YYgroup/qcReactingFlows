# %%

import numpy as np
import pyutils as pu

# %%

C = 5
D = 0.01
S = -1
L = 30

nqx = 8
nqy = 10
wy = 16

# %%

nx = 2**nqx

x = np.linspace(-L/2, L/2, num=nx, endpoint=False)

# %%

data_QC = np.load('1DInOutErfQuantumC{:g}D{:g}S{:g}L{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,L,nqx,wy,nqy))
data_FD = np.load('1DInOutErfClassicalC{:g}D{:g}S{:g}L{:g}nx{:g}.npz'.format(C,D,S,L,nqx))

# %%

pc = pu.fig.PlotConfig(subplot_ratio=0.8)

# %%

# plot profiles

fig, ax = pc.get_simple()

for i in range(5):

    idx = 5*i
    t = data_QC['t'][idx]

    ax.plot(x[:-1], data_FD['y'][:-1,idx], '--b')

    ax.plot(x[:-1], data_QC['y'][:-1,idx], '-.r')

    ax.text(9, data_FD['y'][:-1,idx].max()+0.02, '$t={:.1f}$'.format(t))

ax.set_xlim([-L/2, L/2])
ax.set_ylim([-0.02, 1.1])

ax.set_xlabel('$x$')
ax.set_ylabel('$\phi$')

ax.legend(['Classical FD', 'Quantum FD'], frameon=False, loc='upper left')

# %%

fig.savefig('fig1DInOutErfProfileswy{:g}ny{:g}.png'.format(wy, nqy), dpi=400)
fig.savefig('fig1DInOutErfProfileswy{:g}ny{:g}.eps'.format(wy, nqy))

# %%
