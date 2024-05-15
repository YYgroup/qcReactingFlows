# %%

import numpy as np
import pyutils as pu

# %%

L = 30
C = 10
D = 0.5
S = -1

nqx = 8

nqy = [7, 8, 9, 10, 11]
wy = 8

nqt = 9
wyt = 8

mu = -10

# %%

nx = 2**nqx

x = np.linspace(-L/2, L/2, num=nx, endpoint=False)

# %%

pc = pu.fig.PlotConfig()

# %%

data_C = np.load('1DPeriodicDiscretizedFiniteDifferenceC{:g}D{:g}S{:g}L{:g}nx{:g}.npz'.format(C,D,S,L,nqx))

time = data_C['t']

err_C = np.zeros(time.size)
err_Q_nq = np.zeros((time.size,len(nqy)))

epsilon = 1e-323
#data_Q = np.load('1DPeriodicSpectralQuantumC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,wx,nqx,wy,nqy))

# %%

for i, t in enumerate(time):
    
    # analytic solution
    ut = np.exp(-np.square(x-mu-C*t)/(1+4*D*t)) / np.sqrt(1+4*D*t) * np.exp(S*t)

    ut_norm = np.sqrt(np.sum(np.square(ut)))

    # Classical
    err_C[i] = np.sqrt(np.sum(np.square((data_C['y'][:,i].real-ut)))) / ut_norm

    # Quantum
    for j, v in enumerate(nqy):
        data_Q = np.load('1DPeriodicDiscretizedSeparateC{:g}D{:g}S{:g}L{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,L,nqx,wyt,v))
        err_Q_nq[i,j] = np.sqrt(np.sum(np.square((data_Q['y'][:,i]-ut)))) / ut_norm

# %%

idx_t = [3, 6, 9]
idx_n = [0, 1, 2, 3, 4]

ls = ['--', ':', '-', '-.', '--']
lc = ['m', 'r', 'b', 'g', 'tab:orange']
lm = ['^', 'o', 's', 'D', 'v']

# %%

fig, ax = pc.get_simple()

ax.plot(time[1:], err_C[1:], '-.k', label='Classical')

for k, v in enumerate(idx_n):
    ax.plot(time[1:], err_Q_nq[1:,v],
            ls=ls[k], c=lc[k], marker=lm[k],
            label='$N_y={:g}$'.format(2**nqy[v]))

ax.set_yscale('log')

ax.set_xlim([0, 2.1])
ax.set_ylim([5e-3, 5e-1])

ax.legend(frameon=False, loc='upper left', ncols=2)

ax.set_xlabel('$t$')
ax.set_ylabel('$L_2$ norm')

# %%

fig.savefig('fig1DPeriodicGaussianDiscretizedErrorTimeNy.png', dpi=400)
fig.savefig('fig1DPeriodicGaussianDiscretizedErrorTimeNy.eps')

# %%

fig, ax = pc.get_simple()

for k, v in enumerate(idx_t):
    ax.plot(np.power(2, nqy), err_Q_nq[v,:],
            ls=ls[k], c=lc[k], marker=lm[k],
            label='$t={:.1f}$'.format(time[v]))

ax.plot([2e2, 4e3], [1e-1, 1e-1/400], '-.k')

ax.text(4e2, 5e-2, '$k=-2$')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim([4e1, 1e4])
ax.set_ylim([3e-6, 5e-1])

ax.set_xlabel('$N_y = 2^{n_y}$')
ax.set_ylabel('$L_2$ norm of relative error')

ax.legend(frameon=False, loc='lower left')

# %%

#fig.savefig('fig1DPeriodicFourierSpectralErrorParameterNy.png', dpi=400)
#fig.savefig('fig1DPeriodicFourierSpectralErrorParameterNy.eps')
