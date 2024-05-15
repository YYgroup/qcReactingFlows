# %%

import numpy as np
from scipy import fft, linalg, special
from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
import pyutils as pu

# %%

backend = Aer.get_backend('statevector_simulator')

# %%

C = 5
D = 0.01
S = -1
L = 30

time = np.linspace(0, 2, num=21)

# number of qubits for phi
nq = 8
qr = QuantumRegister(nq, name='qr')
nx = 2 ** nq

nqy = 10
ny = 2 ** nqy

results = np.zeros((nx, time.size))

# %%

# initialization

# nx - 1 grid points

x = np.linspace(-L/2, L/2, num=nx, endpoint=False)
dx = L / nx

u_init = np.zeros(nx)
g = 0.5*(special.erf(x) + 1)
u_init[:-1] = g[1:]
u_init[-1] = 1

u_init_norm = np.sqrt(np.sum(np.square(u_init)))
u0 = u_init / u_init_norm

# %%

shift = 0
alpha = 1
bzone = 0
wy = 16
y = np.linspace(-np.pi*(wy/2+shift), np.pi*(wy/2-shift), num=ny, endpoint=False)

bottom = np.exp(-np.pi*(wy/2-shift))

v_init = np.zeros(ny)
for i, vy in enumerate(y):
    if vy < -np.pi*(wy/2+shift-bzone) or vy > np.pi*(wy/2-shift-bzone):
        v_init[i] = 0
    elif vy < - np.pi * shift :
        tmp = np.exp(alpha*(vy+shift*(1+1/alpha)*np.pi))
        if tmp < bottom :
            tmp = bottom
        v_init[i] = tmp
    else:
        v_init[i] = np.exp(-vy)

v_init0 = np.zeros(ny)
v_init0[:int(ny/2)] = np.exp(y[:int(ny/2)])
v_init0[int(ny/2):] = np.exp(-y[int(ny/2):])
 
fv = fft.fft(v_init, norm='forward')
kv = fft.fftfreq(ny)*ny*2/wy

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

# %%

A = np.matrix(A)
AH = A.H
H1 = (A+AH)/2
H2 = (A-AH)/2

# %%

for j, t in enumerate(time):

    states = []
    for i, k in enumerate(kv):
        # initialization
        circ_init_u = QuantumCircuit(qr, name='Init')
        circ_init_u.initialize(u0*fv[i]/np.abs(fv[i]), qr)
        circ_init_u_inst = circ_init_u.to_instruction()

        qc = QuantumCircuit(qr)
        qc.append(circ_init_u_inst, qr)
        qc.barrier()

        M = 1.j*k*H1 - H2
        U = linalg.expm(M*t)
        qc.append(Operator(U), qr)

        job = backend.run(transpile(qc, backend))
        state = job.result().get_statevector(qc)
        states.append(state)

    # IFT
    idx = 0
    soln = np.zeros(nx, dtype='complex128')

    for i, k in enumerate(kv):
        soln += states[i].data * np.abs(fv[i]) * np.exp(1.j*np.pi*k*wy/2*2*(ny/2+idx)/ny)

    soln *= u_init_norm
    soln *= np.exp(y[int(ny/2+idx)])

    results[:,j] = soln.real

# %%

pc = pu.fig.PlotConfig()

fig, ax = pc.get_simple()

ax.plot(x[:-1], u_init[:-1], '-.b', label='$t=0.0$')

ax.plot(x[:-1], soln.real[:-1], '--r', label='$t=2.0$ Quantum')

ax.legend(frameon=False, loc='upper right')

ax.set_xlabel('$x$')
ax.set_ylabel('$\phi$')

ax.set_xlim([-L/2, L/2])

ax.set_xticks(np.linspace(-L/2, L/2, num=5))

# %%

np.savez('1DInOutErfQuantumC{:g}D{:g}S{:g}L{:g}nx{:g}wy{:g}ny{:g}'.format(C,D,S,L,nq,wy,nqy), 
         x = x, t = time, y = results)
# %%
