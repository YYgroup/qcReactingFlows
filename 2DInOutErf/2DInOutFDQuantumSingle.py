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
wp = 16

D = 0.1
S = -0.5

Lx = wx*np.pi
Ly = wy*np.pi

t = 1

nx = 7
Nx = 2 ** nx

ny = 5
Ny = 2 ** ny

nq = 9
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

# %%

A = np.matrix(A)
AH = A.H
H1 = (A+AH)/2
H2 = (A-AH)/2

# %%

# auxillary variable

shift = 0
alpha = 1
bzone = 0

p = np.linspace(-np.pi*(wp/2+shift), np.pi*(wp/2-shift), num=Nq, endpoint=False)

bottom = np.exp(-np.pi*(wp/2-shift))

v_init = np.zeros(Nq)
for i, vp in enumerate(p):
    if vp < -np.pi*(wp/2+shift-bzone) or vp > np.pi*(wp/2-shift-bzone):
        v_init[i] = 0
    elif vp < - np.pi * shift :
        tmp = np.exp(alpha*(vp+shift*(1+1/alpha)*np.pi))
        if tmp < bottom :
            tmp = bottom
        v_init[i] = tmp
    else:
        v_init[i] = np.exp(-vp)

fv = fft.fft(v_init, norm='forward')
kv = fft.fftfreq(Nq)*Nq*2/wp

# %%

backend = Aer.get_backend('statevector_simulator')

# %%

qx = QuantumRegister(nx, name='qx')
qy = QuantumRegister(ny, name='qy')

# %%

states = []

circ_init_y = QuantumCircuit(qy, name='Init')
circ_init_y.initialize(phiy0, qy)
circ_init_y_inst = circ_init_y.to_instruction()

for i, k in enumerate(kv):
    # initialization
    circ_init_x = QuantumCircuit(qx, name='Init')
    circ_init_x.initialize(phix0*fv[i]/np.abs(fv[i]), qx)
    circ_init_x_inst = circ_init_x.to_instruction()

    qc = QuantumCircuit(qx, qy)
    qc.append(circ_init_x_inst, qx)
    qc.append(circ_init_y_inst, qy)
    qc.barrier()

    M = 1.j*k*H1 - H2
    U = linalg.expm(M*t)
    qc.append(Operator(U), range(nx+ny))

    job = backend.run(transpile(qc, backend))
    state = job.result().get_statevector(qc)
    states.append(state)

# %%

idx = 0
soln = np.zeros(Nx*Ny, dtype='complex128')

for i, k in enumerate(kv):
    soln += states[i].data * np.abs(fv[i]) * np.exp(1.j*np.pi*k*wp/2*2*(Nq/2+idx)/Nq)

soln *= phi_norm
soln *= np.exp(p[int(Nq/2+idx)])

# %%

pc = pu.fig.PlotConfig()

fig, ax = pc.get_simple()

#ax.contourf(x[:-1], y, phi.transpose()[:,:-1], levels=50)
#ax.contourf(x[:-1], y, sol.y[:,-1].reshape((Ny,Nx))[:,:-1], levels=50)
#ax.contourf(x[:-1], y, soln.real.reshape((Ny,Nx))[:,:-1], levels=50)
ax.contour(x[:-1], y, soln.real.reshape((Ny,Nx))[:,:-1], levels=[0.2,])

# %%

np.savez('2DInOutFDQuantumD{:g}wx{:g}wy{:g}nx{:g}ny{:g}wp{:g}np{:g}t{:g}'.format(D,wx,wy,nx,ny,wp,nq,t),
         x, y, sol=soln.real)

# %%
