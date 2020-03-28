# =============================================================================
# First implementation of Potts Model
#   Author : Ghislain de Labbey
#   Date : 5th March 2020
# =============================================================================

# Required for ssh execution with plots
import os
# Standard libraries
import numpy as np
import numpy.random as rd
# Fancy libraries, not necessary
from tqdm import tqdm

# Local modules
from parameters import get_parameters
import patterns
import correlations
import initialisation
import iteration

import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

rd.seed(2019)


dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind \
    = get_parameters()

# if 'ksi_i_mu'not in locals():
ksi_i_mu, delta__ksi_i_mu__k = patterns.get_correlated()

J_i_j_k_l = initialisation.hebbian_tensor(delta__ksi_i_mu__k)

r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
    dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k = initialisation.network()

print('Intégration')
tS = np.arange(0, tSim, dt)
nT = tS.shape[0]

# Plot parameters
nSnap = nT
r_i_k_plot = np.zeros((nSnap, N*(S+1)))
m_mu_plot = np.zeros((nSnap, p))
theta_i_k_plot = np.zeros((nSnap, N*S))
sig_i_k_plot = np.zeros((nSnap, N*(S+1)))

analyseTime = False
analyseDivergence = False

for iT in tqdm(range(nT)):
    iteration.iterate(J_i_j_k_l, delta__ksi_i_mu__k, tS[iT], analyseTime,
                      analyseDivergence, sig_i_k, r_i_k, r_i_S_A,
                      r_i_S_B, theta_i_k, h_i_k, m_mu, dt_r_i_S_A,
                      dt_r_i_S_B, dt_r_i_k_act, dt_theta_i_k)

    r_i_k_plot[iT, :] = r_i_k
    m_mu_plot[iT, :] = m_mu
    theta_i_k_plot[iT, :] = theta_i_k
    sig_i_k_plot[iT, :] = sig_i_k


# Plot
plt.close('all')

plt.figure(1)

ax1 = plt.subplot(3, 2, 1)
for i in range(0, N, N//10):
    for k in range(S+1):
        ax1.plot(tS[:], r_i_k_plot[:, i*(S+1)+k])

ax2 = plt.subplot(3, 2, 2)
for i in range(0, N, N//10):
    for k in range(S):
        ax2.plot(tS[:], theta_i_k_plot[:, i*S + k])

ax3 = plt.subplot(3, 2, 3)
for mu in range(p):
    ax3.plot(tS, m_mu_plot[:, mu])


ax5 = plt.subplot(3, 2, 5)
for i in range(0, N, N//10):
    for k in range(S):
        ax5.plot(tS, sig_i_k_plot[:, i*(S+1)+k])

ax6 = plt.subplot(3, 2, 4)
for i in range(0, N, N//10):
    for k in range(S):
        ax6.plot(tS, sig_i_k_plot[:, i*(S+1) + S])


ax1.set_title("r")

ax2.set_title("theta_k")

ax3.set_title("overlap")

ax5.set_title("sig_i_k")

ax6.plot("sig_i_S")
plt.figure(2)
for mu in range(p):
    plt.plot(tS, m_mu_plot[:, mu])
plt.title('overlap')

plt.show()
