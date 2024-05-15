# %%

import numpy as np
from scipy import fft, linalg, special, integrate
from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
import pyutils as pu

# %%

wx = 8
wy = 2
wp = 8

D = 0.1
S = -0.5

Lx = wx*np.pi
Ly = wy*np.pi

t = 1.5

nx = 8
Nx = 2 ** nx

ny = 6
Ny = 2 ** ny

nq = 8
Nq = 2 ** nq

# %%

x = np.linspace(-Lx/2, Lx/2, num=Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, num=Ny, endpoint=False)

dx = Lx/Nx
dy = Ly/Ny

# %%

vely = 2*np.cos(y) + 3
vel = np.outer(np.ones(Nx), vely)

phix = 0.5*(special.erf(x+6)+1)
phix_norm = np.sqrt(np.sum(np.square(phix)))
phix0 = phix / phix_norm

phiy = np.ones(Ny)
phiy_norm = np.sqrt(np.sum(np.square(phiy)))
phiy0 = phiy / phiy_norm

phi = np.outer(phix, np.ones(Ny))
phi_norm = np.sqrt(np.sum(np.square(phi)))
phi0 = phi.flatten('F') / phi_norm

# %%

# convection

ACx1D = np.zeros((Nx, Nx))
for i in range(Nx-2):
    ACx1D[i,i+1] = 1
    ACx1D[i+1,i] = -1
ACx1D[-2,-2] = 1

#ACx = np.kron(np.eye(Ny), ACx1D) / dx
ACx = np.kron(np.diag(vely), ACx1D) / dx

AC = ACx

# %%

# diffusion

ADx1D = 2*np.eye(Nx)
ADx1D[-1,-1] = 0
for i in range(Nx-2):
    ADx1D[i,i+1] = -1
    ADx1D[i+1,i] = -1
ADx1D[-2, -3] = -2

ADx = np.kron(np.eye(Ny), ADx1D) * D / np.square(dx)

ADy1D = 2*np.eye(Ny)
for i in range(Ny-1):
    ADy1D[i,i+1] = -1
    ADy1D[i+1,i] = -1
ADy1D[0,-1] = -1
ADy1D[-1,0] = -1
tmp = np.eye(Nx)
tmp[-1,-1] = 0
ADy = np.kron(ADy1D, tmp) * D / np.square(dx)

AD = ADx+ADy

# %%

# reaction

ASx1D = -S * np.eye(Nx)
ASx1D[-1,-1] = 0
AS = np.kron(np.eye(Ny), ASx1D)

# %%

A = AC + AD + AS

def rhs(t, y):
    return -np.matmul(A, y)

# %%

sol = integrate.solve_ivp(rhs, [0, t], phi.flatten('F'), method='RK23', t_eval=np.linspace(0,t,num=16))

# %%

pc = pu.fig.PlotConfig()

fig, ax = pc.get_simple()

#ax.contourf(x[:-1], y, phi.transpose()[:,:-1], levels=50)
ax.contour(x[:-1], y, sol.y[:,-6].reshape((Ny,Nx))[:,:-1], levels=[0.1, 0.2, 0.3, 0.5])
ax.contourf(x[:-1], y, sol.y[:,-6].reshape((Ny,Nx))[:,:-1], levels=50)

# %%

np.savez('2DInOutErfClassicalD{:g}wx{:g}wy{:g}nx{:g}ny{:g}'.format(D,wx,wy,nx,ny),
         x = x, y = y, t = sol.t, sol=sol.y)

# %%
