# %%

import numpy as np
from scipy import special
from scipy.integrate import solve_ivp
import pyutils as pu

# %%

C = 5
D = 0.01
S = -1
L = 30

t = 2

# number of qubits for phi
nq = 8
nx = 2 ** nq

# %%

# initialization
# nx - 1 grid points
x = np.linspace(-L/2, L/2, num=nx, endpoint=False)

dx = L / nx

u_init = 0.5 * (special.erf(x)+1)
u_init[-1] = 1

# %%

# discretization

# convection

AC = np.zeros((nx, nx))

for i in range(nx-2):
    AC[i,i+1] = 1
    AC[i+1,i] = -1
AC[-2, -2] = 1

AC = AC * C / (2*dx)

# diffusion

AD = 2*np.eye(nx)
AD[-1, -1] = 0
for i in range(nx-2):
    AD[i,i+1] = -1
    AD[i+1,i] = -1
# zero gradient
AD[-2, -3] = -2

AD = AD * D / np.square(dx)

# reaction

AS = -S*np.eye(nx)
AS[-1, -1] = 0

# 

A = AC + AD + AS

def rhs(t, y):
    return np.matmul(-A, y)

# %%

sol = solve_ivp(rhs, [0, t], u_init, method='RK23', t_eval=np.linspace(0, t, num=21))

# %%

# plot

pc = pu.fig.PlotConfig()

fig, ax = pc.get_simple()

for i in range(0, 5):
    idx = 5*i

    time = sol.t[idx]

    ax.plot(x[:-1], sol.y[:-1,idx], label='$t={:g}$'.format(time))

ax.legend(frameon=False)

# %%

np.savez('1DInOutErfClassicalC{:g}D{:g}S{:g}L{:g}nx{:g}'.format(C,D,S,L,nq),
         x = x, t = sol.t, y = sol.y)

# %%
