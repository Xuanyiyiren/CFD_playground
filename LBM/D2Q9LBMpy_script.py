r"""
D2Q9 Lattice Boltzmann Method (LBM) von Kármán vortex street simulation

Main code reference: https://youtu.be/JFWqCQHg-Hs
Main theory reference: Lattice Boltzmann and Gas Kinetic Flux Solvers (https://doi.org/10.1142/11949)

---
Lattice Boltzmann Equation (LBE):
\partial_{t}f + \vec{\xi}\cdot \nabla f = \Omega[f,f]

After trapezoidal rule and rearrangement:
\bar f(\vec x + \vec{\xi}\Delta t, \vec{\xi}, t + \Delta t) - \bar f(\vec x, \vec{\xi}, t) = \frac{f^{\text{eq}}(\vec x, \vec{\xi}, t) - \bar f(\vec x, \vec{\xi}, t) }{\tau'}

---
D2Q9 Scheme:
Discrete velocities and weights:
veloI = [0, 0, 1, 0, -1, 1, 1, -1, -1]
veloJ = [0, 1, 0, -1, 0, 1, -1, -1, 1]
weights = [16, 4, 4, 4, 4, 1, 1, 1, 1] / 36

---
Algorithm Summary:
1. Compute equilibrium distribution
2. Collision step
3. Streaming step

---
Physical parameters:
Delta x = \tilde{\xi} Delta t
nu = ((tau' - 0.5) * Delta t * k_B * T / m)
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import shutil
from tqdm import tqdm

# Isothermal and constant kinematic viscosity
Nx = 400
Ny = 100
tau = .53  # numerical relaxation time

# D2Q9 model velocities and weights
Nv = 9  # Number of velocities in D2Q9 model
veloI = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])  # x-components of velocities
veloJ = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])  # y-components of velocities
weights = np.array([16, 4, 4, 4, 4, 1, 1, 1, 1]) / 36  # weights for D2Q9 model

# Equilibrium distribution function
def setfeq(rho, ux, uy):
    cdot = veloI * ux + veloJ * uy
    return rho * weights * (
        1 + 3 * cdot + 9/2 * cdot**2 - 3/2 * (ux**2 + uy**2)
    )

rho = 10.34
ux = 1.28 / rho
uy = 0 / rho
movingfeq = setfeq(rho, ux, uy)
staticfeq = setfeq(rho, 0, 0)

# Geometry: cylinder obstacle
radius = 13
cylinder = np.full((Nx, Ny), False)
center = (Nx // 4, Ny // 2)
for i in range(Nx):
    for j in range(Ny):
        if (i - center[0])**2 + (j - center[1])**2 < radius**2:
            cylinder[i, j] = True

# Initial conditions
boltzmannf = 0.01 * np.random.rand(Nx, Ny, Nv)
boltzmannf[:, :] += movingfeq
boltzmannf[cylinder] = staticfeq

# Prepare output directory
if os.path.exists('simufigs'):
    shutil.rmtree('simufigs')
os.mkdir('simufigs')

Nt = 10000
skip = 50
savefig = True

# Use ffmpeg -i frame_%04d.png -r 20 output.mp4 to generate video
from tqdm import trange

figlabel = 0
fig, ax = plt.subplots(figsize=(8, 3))
im = None

for it in trange(Nt, desc='LBM Simulation'):
    rho = np.sum(boltzmannf, axis=2)  # density
    ux = np.sum(boltzmannf * veloI, axis=2) / rho  # x-velocity
    uy = np.sum(boltzmannf * veloJ, axis=2) / rho  # y-velocity

    # Compute equilibrium distribution function
    boltzmannfeq = np.zeros_like(boltzmannf)
    for i in range(Nv):
        cdot = veloI[i] * ux + veloJ[i] * uy
        boltzmannfeq[:, :, i] = rho * weights[i] * (1 + 3 * cdot + 9/2 * cdot**2 - 3/2 * (ux**2 + uy**2))

    # Collision step
    boltzmannf = boltzmannf + (boltzmannfeq - boltzmannf) / tau

    # Streaming step
    for i in range(Nv):
        boltzmannf[:, :, i] = np.roll(boltzmannf[:, :, i],
                                      (veloI[i], veloJ[i]),
                                      axis=(0, 1))
    # Bouncing Back boundary condition (no-slip wall)
    boltzmannf[cylinder, :] = boltzmannf[cylinder, :][:,[0, 3, 4, 1, 2, 7, 8, 5, 6]]
    # Absorption boundary condition
    # boltzmannf[0, :, [2, 5, 6]] = boltzmannf[1, :, [2, 5, 6]]
    # boltzmannf[-1, :, [4, 7, 8]] = boltzmannf[-2, :, [4, 7, 8]]
    # boltzmannf[0, :, :] = boltzmannf[1, :, :]
    boltzmannf[-1, :, :] = boltzmannf[-2, :, :]
    # Inlet
    boltzmannf[0, :] = movingfeq

    # Visualization
    ux[cylinder] = 0
    uy[cylinder] = 0
    if it % skip == 0:
        vorticity = (np.gradient(uy, axis=0) - np.gradient(ux, axis=1)).T
        if im is None:
            im = ax.imshow(vorticity, cmap='bwr', vmin=-0.08, vmax=0.08, origin='lower')
            plt.colorbar(im, ax=ax, orientation='horizontal')
        else:
            im.set_data(vorticity)
        ax.set_title(f'Timestep {it}')
        plt.tight_layout()
        plt.draw()
        plt.savefig(f'simufigs/frame_{figlabel:04d}.png', dpi=300)
        figlabel += 1
        plt.pause(0.001)

# Physical parameters calculation (optional, requires astropy):

# import astropy.constants as const
# import astropy.units as units

# R = const.k_B * const.N_A
# M = 29 * units.g/units.mol
# T = (25 + 273.15)*units.K  # temperature in Kelvin
# xi_vel = np.sqrt(3*R*T/M).si

# # Choose dx = 0.01 m, domain 1m x 4m
# dx = 0.01 * units.m
# dt = dx / xi_vel

# # Total simulation time
# total_time = dt * Nt

# # Kinematic viscosity
# nu = ((tau - .5) * dt * R * T / M).si
