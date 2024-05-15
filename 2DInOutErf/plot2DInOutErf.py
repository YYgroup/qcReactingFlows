# %%

import numpy as np
from scipy import special
import pyutils as pu

# %%

C = 4
D = 0.1
S = -0.5

wx = 8
wy = 2
wp = 16

Lx = wx*np.pi
Ly = wy*np.pi

nx = 6
Nx = 2 ** nx

ny = 4
Ny = 2 ** ny

nq = 9
Nq = 2 ** nq

# %%

x = np.linspace(-Lx/2, Lx/2, num=Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, num=Ny, endpoint=False)

vely = 2*np.cos(y) + C
vel = np.outer(np.ones(Nx), vely)

phix = 0.5*(special.erf(x+5)+1)
phix_norm = np.sqrt(np.sum(np.square(phix)))
phix0 = phix / phix_norm

phiy = np.ones(Ny)
phiy_norm = np.sqrt(np.sum(np.square(phiy)))
phiy0 = phiy / phiy_norm

phi = np.outer(phix, np.ones(Ny))
phi_norm = np.sqrt(np.sum(np.square(phi)))
phi0 = phi.flatten('F') / phi_norm

# %%

xn = x[:-1]
yn = np.linspace(-Ly/2, Ly/2, num=Ny+1, endpoint=True)
XN, YN = np.meshgrid(xn, yn)

veln = np.outer(np.ones(Nx-1), 2*np.cos(yn)+C)

field = np.zeros((Ny+1, Nx-1))

# %%

#data_C = np.load('2DInOutErfClassicalD{:g}wx{:g}wy{:g}nx{:g}ny{:g}.npz'.format(D,wx,wy,nx,ny))
data_C = np.load('2DInOutErfUpwindClassicalC{:g}D{:g}wx{:g}wy{:g}nx{:g}ny{:g}.npz'.format(C,D,wx,wy,nx,ny))

t = 0.5
data_Q1 = np.load('2DInOutFDUpwindQuantumC{:g}D{:g}wx{:g}wy{:g}nx{:g}ny{:g}wp{:g}np{:g}t{:g}.npz'.format(C,D,wx,wy,nx,ny,wp,nq,t))

t = 1
data_Q2 = np.load('2DInOutFDUpwindQuantumC{:g}D{:g}wx{:g}wy{:g}nx{:g}ny{:g}wp{:g}np{:g}t{:g}.npz'.format(C,D,wx,wy,nx,ny,wp,nq,t))

# %%

pc = pu.fig.PlotConfig('PCI_single', nrow=3, subplot_ratio=0.25, 
                       space_height=0.1, 
                       margin_right=0.05, margin_top=0.7,
                       margin_left=0.9, margin_bottom=0.8)

fig = pc.get_fig()

ax0 = pc.get_axes(fig, i=0)

field[:-1,:] = phi.transpose()[:,:-1]
field[-1,:] = phi.transpose()[0,:-1]
p0 = ax0.contourf(xn, yn, field, 
                  cmap='coolwarm', levels=100, vmin=0, vmax=1)
#p0 = ax0.contourf(x[:-1], y, phi.transpose()[:,:-1], 
#                  cmap='coolwarm', levels=100, vmin=0, vmax=1)

ax0.contour(xn, yn, field, 
            levels=[0.5,], linestyles='-.', colors='k')
#ax0.contour(x[:-1], y, phi.transpose()[:,:-1], 
#            levels=[0.5,], linestyles='-.', colors='k')

ax0.quiver(XN[::2,6::20], YN[::2,6::20], 
           veln.transpose()[::2,6::20], 
           np.zeros(veln.transpose().shape)[::2,6::20])
#X, Y = np.meshgrid(x[:-1], y)
##ax0.quiver(X[::5,10::40], Y[::5,10::40], 
##           vel.transpose()[:,:-1][::5,10::40], 
##           np.zeros(vel.transpose()[:,:-1].shape)[::5,10::40])
#ax0.quiver(X[::2,6::20], Y[::2,6::20], 
#           vel.transpose()[:,:-1][::2,6::20], 
#           np.zeros(vel.transpose()[:,:-1].shape)[::2,6::20])

ax0.set_xticks(np.linspace(-10,10,num=5))
ax0.set_xticklabels([])
ax0.set_yticks(np.linspace(-2,2,num=3))

ax0.set_ylabel('$y$')
ax0.text(7.3,1.5,'$t=0$')
ax0.text(-11.5,1.5,r'$\bm{u}$')

ax1 = pc.get_axes(fig, i=1)

field[:-1,:] = data_Q1['sol'].reshape((Ny,Nx))[:,:-1]
field[-1,:] = data_Q1['sol'].reshape((Ny,Nx))[0,:-1]
ax1.contourf(xn, yn, field,
             cmap='coolwarm', levels=100, vmin=0, vmax=1)
#ax1.contourf(x[:-1], y, data_Q1['sol'].reshape((Ny,Nx))[:,:-1],
#             cmap='coolwarm', levels=100, vmin=0, vmax=1)

field[:-1,:] = data_C['sol'][:,5].reshape((Ny,Nx))[:,:-1]
field[-1,:] = data_C['sol'][:,5].reshape((Ny,Nx))[0,:-1]
ax1.contour(xn, yn, field,
            levels=[0.5,], linestyles='-.', colors='k')
#ax1.contour(x[:-1], y, data_C['sol'][:,5].reshape((Ny,Nx))[:,:-1],
#            levels=[0.5,], linestyles='-.', colors='k')

ax1.set_xticks(np.linspace(-10,10,num=5))
ax1.set_xticklabels([])
ax1.set_yticks(np.linspace(-2,2,num=3))

ax1.set_ylabel('$y$')
ax1.text(7.3,1.5,'$t=0.5$')

ax2 = pc.get_axes(fig, i=2)

field[:-1,:] = data_Q2['sol'].reshape((Ny,Nx))[:,:-1]
field[-1,:] = data_Q2['sol'].reshape((Ny,Nx))[0,:-1]
ax2.contourf(xn, yn, field,
             cmap='coolwarm', levels=100, vmin=0, vmax=1)
#ax2.contourf(x[:-1], y, data_Q2['sol'].reshape((Ny,Nx))[:,:-1],
#             cmap='coolwarm', levels=100, vmin=0, vmax=1)

field[:-1,:] = data_C['sol'][:,-6].reshape((Ny,Nx))[:,:-1]
field[-1,:] = data_C['sol'][:,-6].reshape((Ny,Nx))[0,:-1]
ax2.contour(xn, yn, field,
            levels=[0.5,], linestyles='-.', colors='k')
#ax2.contour(x[:-1], y, data_C['sol'][:,-6].reshape((Ny,Nx))[:,:-1],
#            levels=[0.5,], linestyles='-.', colors='k')

ax2.set_xticks(np.linspace(-10,10,num=5))
ax2.set_yticks(np.linspace(-2,2,num=3))

ax2.set_ylabel('$y$')
ax2.set_xlabel('$x$')
ax2.text(7.3,1.5,'$t=1.0$')

rect = (0.36, 0.9, 0.6, 0.02)
ax_c = fig.add_axes(rect)
cbar = fig.colorbar(p0, cax=ax_c, orientation='horizontal')

ax_c.xaxis.set_ticks_position('top')
ax_c.tick_params(direction='in')
ax_c.xaxis.set_ticks(np.linspace(0,1,num=6))

fig.text(0.28, 0.9, r'$\phi$')

# %%

fig.savefig('fig2DInOut.png', dpi=600)
fig.savefig('fig2DInOut.eps')

# %%
