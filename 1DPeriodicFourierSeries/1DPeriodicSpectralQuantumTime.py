# %%

import numpy as np
from scipy import fft, linalg
from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
import pyutils as pu

# %%

backend = Aer.get_backend('statevector_simulator')

# %%

# Parameters

time = np.linspace(0, 2, num=21)

C = 4
D = 1
S = -0.2
wx = 2
L = wx*np.pi

# number of qubits for physical space
nq = 8
qr = QuantumRegister(nq, name='qr')
nx = 2 ** nq

# dimension of the auxillary variable
nqy = 9
ny = 2 ** nqy

wy = 10

results = np.zeros((nx, time.size))

# %%

# Initialization

x = np.linspace(-L/2, L/2, num=nx, endpoint=False)

k_s = np.array([1, 3])
k_c = np.array([2,])
u_init = np.zeros(x.size)
for k in k_s:
    u_init = u_init + np.sin(k*x)
for k in k_c:
    u_init = u_init + np.cos(k*x)

u_init_norm = np.sqrt(np.sum(np.square(u_init)))

u0 = u_init / u_init_norm

# %%

# auxillary variable

shift = 0
alpha = 10
bzone = 0

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

fv = fft.fft(v_init, norm='forward')
kv = fft.fftfreq(ny)*ny*2/wy

# %%

# QFT

circ_QFT = QFT(nq)
circ_IQFT = QFT(nq).inverse()

# %%

kx = fft.fftfreq(nx)*nx*2*np.pi/L

# %%

for j, t in enumerate(time):

    states = []

    for i, k in enumerate(kv):

        # initialization
        circ_init = QuantumCircuit(qr, name='Init_u')
        circ_init.initialize(u0*fv[i]/np.abs(fv[i]), qr)
        circ_init_inst = circ_init.to_instruction()

        qc = QuantumCircuit(qr)
        qc.append(circ_init_inst, qr)
        qc.barrier()
        qc.append(circ_QFT, qr)
        qc.barrier()
        # wavenumbers
        H = np.diag(C*kx+D*k*np.square(kx)-S*k)
        # Caution
        U = linalg.expm(1.j*H*t)
        #
        qc.append(Operator(U), qr)
        qc.barrier()
        qc.append(circ_IQFT, qr)
        qc.barrier()
        # run
        job = backend.run(transpile(qc, backend))
        state = job.result().get_statevector(qc)
        # store
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

#ax.plot(x, u_init, '-k')

ut = np.zeros(x.size)
for k in k_s:
    ut = ut + np.sin(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)
for k in k_c:
    ut = ut + np.cos(k*(x-C*t)) * np.exp((-D*np.square(k)+S)*t)
ax.plot(x, ut, '-k', label='Analytic')

for i in range(0, 5):
    ax.plot(x, results[:,5*i], '-.r', label='Quantum')

ax.set_xlim([-L/2, L/2])

# %%

np.savez('1DPeriodicSpectralQuantumC{:g}D{:g}S{:g}wx{:g}nx{:g}wy{:g}ny{:g}'.format(C,D,S,wx,nq,wy,nqy), 
         x = x, t = time, y = results)

# %%
