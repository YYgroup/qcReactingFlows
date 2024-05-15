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

nqy = [6, 7, 8, 9, 10, 11, 12]
wy = [4, 5, 6, 7, 8, 9, 10]

nqt = 9
wyt = 8

# %%

# solution

k_s = np.array([1, 3])
k_c = np.array([2,])

# %%

nx = 2**nqx

x = np.linspace(-L/2, L/2, num=nx, endpoint=False)

# %%

pc = pu.fig.PlotConfig()

# %%

data_C = np.load('1DPeriodicSpectralClassicalC{:g}D{:g}S{:g}wx{:g}nx{:g}.npz'.format(C,D,S,wx,nqx))

time = data_C['t'][:11]

err_C = np.zeros(time.size)
err_Q_nq = np.zeros((time.size,len(nqy)))
err_D_nq = np.zeros((time.size,len(nqy)))
err_Q_wy = np.zeros((time.size,len(wy)))

epsilon = 1e-323
#data_Q = np.load('1DPeriodicSpectralQuantumC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,wx,nqx,wy,nqy))

# %%

for i, t in enumerate(time):
    
    # analytic solution
    ut = np.zeros(x.size)
    for k in k_s:
        ut = ut + np.sin(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)
    for k in k_c:
        ut = ut + np.cos(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)

    ut_norm = np.sqrt(np.sum(np.square(ut)))

    # Classical
    err_C[i] = np.sqrt(np.sum(np.square((data_C['y'][:,i].real-ut)))) / ut_norm

    # Quantum
    for j, v in enumerate(nqy):
        data_Q = np.load('1DPeriodicSpectralQuantumC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,wx,nqx,wyt,v))
        err_Q_nq[i,j] = np.sqrt(np.sum(np.square((data_Q['y'][:,i]-ut)))) / ut_norm

        data_D = np.load('1DPeriodicDiscretizedQuantumC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,wx,nqx,wyt,v))
        err_D_nq[i,j] = np.sqrt(np.sum(np.square((data_D['y'][:,i]-ut)))) / ut_norm

    for j, v in enumerate(wy):
        data_Q = np.load('1DPeriodicSpectralQuantumC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}.npz'.format(C,D,S,wx,nqx,v,nqt))
        err_Q_wy[i,j] = np.sqrt(np.sum(np.square((data_Q['y'][:,i]-ut)))) / ut_norm

# %%

idx_t = [3, 6, 9]
idx_n = [2, 4, 6]
idx_w = [0, 2, 4]

ls = ['--', ':', '-']
lc = ['g', 'b', 'r']
lm = ['D', 's', 'o']

# %%

fig, ax = pc.get_simple()

for k, v in enumerate(idx_t):
    ax.plot(nqy, err_Q_nq[v,:],
            ls='-', c=lc[k], marker=lm[k], mfc='none',
            label='$t={:.1f}$ SP'.format(time[v]))

for k, v in enumerate(idx_t):
    ax.plot(nqy, err_D_nq[v,:],
            ls='-.', c=lc[k], marker=lm[k], mfc='none',
            label='$t={:.1f}$ FD'.format(time[v]))

#ax.plot([2e2, 4e3], [1e-1, 1e-1/400], '-.k')

#ax.text(4e2, 5e-2, '$k=-2$')

ax.set_yscale('log')

ax.set_xlim([5.8, 12.2])
ax.set_ylim([3e-6, 5e-1])

ax.set_xlabel('$n_p$')
#ax.set_ylabel('$L_2$ norm of relative error')
ax.set_ylabel('$\epsilon$')

ax.legend(frameon=False, loc='lower left', ncols=1)

# %%

fig.savefig('fig1DPeriodicSeriesErrorParameternp.png', dpi=600)
fig.savefig('fig1DPeriodicSeriesErrorParameternp.eps')

# %%

fig, ax = pc.get_simple()

ax.plot(time[1:], err_C[1:], '-.k', label='Classical')

for k, v in enumerate(idx_n):
    ax.plot(time[1:], err_Q_nq[1:,v],
            ls=ls[k], c=lc[k], marker=lm[k],
            label='$N_y={:g}$'.format(2**nqy[v]))

ax.set_yscale('log')

ax.set_xlim([0, 2.1])
ax.set_ylim([4e-6, 1])

ax.legend(frameon=False, loc='upper left', ncols=2)

ax.set_xlabel('$t$')
ax.set_ylabel('$L_2$ norm of relative error')

# %%

fig.savefig('fig1DPeriodicFourierSpectralErrorTimeNy.png', dpi=400)
fig.savefig('fig1DPeriodicFourierSpectralErrorTimeNy.eps')

# %%

fig, ax = pc.get_simple()

ax.plot(time[1:], err_C[1:], '-.k', label='Classical')

for k, v in enumerate(idx_w):
    ax.plot(time[1:], err_Q_wy[1:,v],
            ls=ls[k], c=lc[k], marker=lm[k],
            label='$L_y={:g}\pi$'.format(wy[v]))

ax.set_yscale('log')

ax.set_xlim([0, 2.1])
ax.set_ylim([2e-4, 1e1])

ax.legend(frameon=False, loc='upper left')

ax.set_xlabel('$t$')
ax.set_ylabel('$L_2$ norm of relative error')

# %%

fig.savefig('fig1DPeriodicFourierSpectralErrorTimeLy.png', dpi=400)
fig.savefig('fig1DPeriodicFourierSpectralErrorTimeLy.eps')

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

fig.savefig('fig1DPeriodicFourierSpectralErrorParameterNy.png', dpi=400)
fig.savefig('fig1DPeriodicFourierSpectralErrorParameterNy.eps')

# %%

fig, ax = pc.get_simple()

for k, v in enumerate(idx_t):
    ax.plot(wy, err_Q_wy[v,:],
            ls=ls[k], c=lc[k], marker=lm[k],
            label='$t={:.1f}$'.format(time[v]))

#ax.plot([2e2, 2e3, 6e3], [1e-0, 1e-2, 1e-2/9], '-.k')

#ax.text(4e2, 5e-1, '$k=-2$')

#ax.set_xscale('log')
#ax.set_yscale('log')

#ax.set_xlim([4e1, 1e4])
#ax.set_ylim([1e-5, 2e0])

#ax.set_xlabel('$N_y = 2^{n_y}$')
ax.set_ylabel('$L_2$ norm of error')

ax.legend(frameon=False, loc='lower left')

# %%
