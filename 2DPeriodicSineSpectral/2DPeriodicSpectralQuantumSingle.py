# %%

import numpy as np
from scipy import fft, linalg, special
from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
import pyutils as pu

# %%

wx = 2
wy = 2
wp = 8

Cx = 2
Cy = 1
D = 1
S = -0.2

Lx = wx*np.pi
Ly = wy*np.pi

t = 1.5

nx = 5
Nx = 2 ** nx

ny = 5
Ny = 2 ** ny

nq = 8
Nq = 2 ** nq

# %%

x = np.linspace(-Lx/2, Lx/2, num=Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, num=Ny, endpoint=False)

kx = fft.fftfreq(Nx)*Nx*2/wx
ky = fft.fftfreq(Ny)*Ny*2/wy

k1x = np.outer(np.ones(Ny), kx).flatten()
k1y = np.outer(ky, np.ones(Nx)).flatten()

k2x = np.outer(np.ones(Ny), np.square(kx)).flatten()
k2y = np.outer(np.square(ky), np.ones(Nx)).flatten()

# %%

ux = np.sin(x)
uy = np.sin(y)

# %%

zx = np.outer(ux, np.ones(Ny))
zy = np.outer(np.ones(Nx), uy)
z = zx + zy

z_norm = np.sqrt(np.sum(np.square(z)))
z0 = z.flatten() / z_norm

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

qr = QuantumRegister(nx+ny, name='qr')

# %%

# QFT

circ_QFT_x = QFT(nx)
circ_QFT_y = QFT(ny)

circ_IQFT_x = QFT(nx).inverse()
circ_IQFT_y = QFT(ny).inverse()

# %%

states = []

for i, k in enumerate(kv):

    circ_init = QuantumCircuit(qr, name='init')
    circ_init.initialize(z0*fv[i]/np.abs(fv[i]), qr)
    circ_init_inst = circ_init.to_instruction()

    qc = QuantumCircuit(qr)
    qc.append(circ_init_inst, qr)
    qc.barrier()
    qc.append(circ_QFT_x, range(nx))
    qc.append(circ_QFT_y, range(nx,nx+ny))
    qc.barrier()

    H = np.diag( Cx*k1x + Cy*k1y + D*k*(k2x+k2y) - S*k )
    U = linalg.expm(1.j*H*t)
    qc.append(Operator(U), range(nx+ny))

    qc.barrier()
    qc.append(circ_IQFT_x, range(nx))
    qc.append(circ_IQFT_y, range(nx,nx+ny))
    qc.barrier()

    job = backend.run(transpile(qc, backend))
    state = job.result().get_statevector(qc)

    states.append(state)

# %%

idx = 0
soln = np.zeros(Nx*Ny, dtype='complex128')

for i, k in enumerate(kv):
    soln += states[i].data * np.abs(fv[i]) * np.exp(1.j*np.pi*k*wp/2*2*(Nq/2+idx)/Nq)

soln *= z_norm
soln *= np.exp(p[int(Nq/2+idx)])

# %%

pc = pu.fig.PlotConfig()

fig, ax = pc.get_simple()

#ax.contourf(x, y, z.transpose(), levels=50)
#ax.contourf(x, y, z0.reshape(Nx,Ny).transpose(), levels=50)
ax.contourf(x, y, soln.real.reshape(Nx,Ny).transpose(), levels=50)

# %%
