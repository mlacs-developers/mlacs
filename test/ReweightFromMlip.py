import logging
import sys

import numpy as np
from ase.units import kB
from ase.io import read
from pymbar import MBAR
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mlacs.utilities.plots import init_rcParams

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

confs = read("Trajectory.traj", index="10:")
emlip = np.loadtxt("Trajectory_potential.dat")[9:, 1]

n = len(confs)
e = np.array([at.get_potential_energy() for at in confs])
t = np.array([at.get_temperature() for at in confs])

idx = np.argsort(t)
e = e[idx]
t = t[idx]
emlip = emlip[idx]

u_kn = np.zeros((e.shape[0]*2, e.shape[0]))
for i, temp in enumerate(t):
    u_kn[i, :] = emlip / (temp * kB)
    u_kn[n+i] = e / (temp * kB)

N_k = np.append(np.ones(t.shape), np.zeros(t.shape))

mb = MBAR(u_kn, N_k, solver_protocol="robust", verbose=True)
weights = mb.weights()
neff = mb.compute_effective_sample_number()
f_k = mb.f_k

idx = np.argmin(t)
idx_max = np.argmax(t)

res = mb.compute_entropy_and_enthalpy()
delta_f = res['Delta_f'][idx]
ddelta_f = res['dDelta_f'][idx]
delta_u = res['Delta_u'][idx]
ddelta_u = res['dDelta_u'][idx]
delta_s = res['Delta_s'][idx]
ddelta_s = res['dDelta_s'][idx]

emean = np.zeros(e.shape)
estd = np.zeros(e.shape)
emean_ml = np.zeros(e.shape)
estd_ml = np.zeros(e.shape)
for i, temp in enumerate(t):
    w = weights[:, i+n]
    emean[i] = (w * e).sum()
    estd[i] = ((w * e**2).sum() - (w * e).sum()**2)  # / (kB * temp**2)
    w = weights[:, i]
    emean_ml[i] = (w * e).sum()
    estd_ml[i] = ((w * e**2).sum() - (w * e).sum()**2)  # / (kB * temp**2)

test_temp = np.linspace(200, 1200, 250)
u_ln = np.zeros((test_temp.shape[0], e.shape[0]))
for i, temp in enumerate(test_temp):
    u_ln[i] = e / (kB * temp)
res = mb.compute_perturbed_free_energies(u_ln)
ddf = res["Delta_f"][0]

ref = np.loadtxt("RefMD/Results.dat")
# ref[:, 2] /= (kB * ref[:, 0]**2)

# =================================================================#
# Plot
# =================================================================#

fig = plt.figure(figsize=(10, 30), constrained_layout=True)
init_rcParams()
gs = GridSpec(2, 2, fig)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax0.plot(t, emean / 216)
ax0.plot(t, emean_ml / 216, alpha=0.5)
ax0.plot(ref[:, 0], ref[:, 1] / 216, ls="", marker="o")
ax1.plot(t, estd / 216)
ax1.plot(t, estd_ml / 216, alpha=0.5)
ax1.plot(ref[:, 0], ref[:, 2] / 216, ls="", marker="o")
ax2.plot(t, delta_f[n:] / 216)
ax3.plot(t, neff[n:])
ax3.plot(t, neff[:n], alpha=0.5)

ax0.set_ylabel(r"E$_{\mathrm{mean}}$ [eV/at]")
ax1.set_ylabel(r"C$_{\mathrm{V}}$ [kBT/at]")
ax2.set_ylabel(r"$\Delta$f [a.u.]")
ax3.set_ylabel(r"N$_{\mathrm{eff}}$")

plt.show()
