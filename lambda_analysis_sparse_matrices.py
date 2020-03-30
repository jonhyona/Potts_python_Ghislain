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

dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind, \
    random_seed = get_parameters()

rd.seed(random_seed)


# if 'ksi_i_mu'not in locals():
ksi_i_mu, delta__ksi_i_mu__k = patterns.get_correlated()

J_i_j_k_l = initialisation.hebbian_tensor(delta__ksi_i_mu__k)

r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
    dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k = initialisation.network()

print('Intégration')
tS = np.arange(0, tSim, dt)
nT = tS.shape[0]

# Plot parameters
lamb = []
tTrans = []

indTrans = 0
cpt_idle = 0
nSnap = nT
# r_i_k_plot = np.zeros((nSnap, N*(S+1)))
# m_mu_plot = np.zeros((nSnap, p))
# theta_i_k_plot = np.zeros((nSnap, N*S))
# sig_i_k_plot = np.zeros((nSnap, N*(S+1)))


analyseTime = False
analyseDivergence = False

# retrieved = np.zeros(nT, dtype=int)
# max_m_mu = np.zeros(nT)
# outsider = retrieved.copy()
# max2_m_mu = max_m_mu.copy()
lamb = []
tTrans = []
retrieved_saved = []
max_m_mu_saved = []
max2_m_mu_saved = []
outsider_saved = []

indTrans = 0
cpt_idle = 0

ind_max_prev = -1

for iT in tqdm(range(nT)):
    iteration.iterate(J_i_j_k_l, delta__ksi_i_mu__k, tS[iT], analyseTime,
                      analyseDivergence, sig_i_k, r_i_k, r_i_S_A,
                      r_i_S_B, theta_i_k, h_i_k, m_mu, dt_r_i_S_A,
                      dt_r_i_S_B, dt_r_i_k_act, dt_theta_i_k)

    if tS[iT] > t_0 + tau:
        ind_max = np.argmax(m_mu)
        max_m_mu = m_mu[ind_max]

        if ind_max != ind_max_prev:

            m_mu[ind_max] = - np.inf
            outsider = np.argmax(m_mu)
            max2_m_mu = m_mu[outsider]

            tTrans.append(tS[iT])
            lamb.append(max_m_mu)
            retrieved_saved.append(ind_max)
            outsider_saved.append(outsider)
            max_m_mu_saved.append(max_m_mu)
            max2_m_mu_saved.append(max2_m_mu)

            indTrans += 1
            cpt_idle = 0

        if max_m_mu < .1:
            cpt_idle += 1
            if cpt_idle > nT/10 and nT >= 1000:
                print("latchingDied")
                break
        ind_max_prev = ind_max


# %%Plot
plt.close('all')

plt.figure(2)
ax1 = plt.subplot(211)
ax1.plot(tTrans, retrieved_saved, 'go:', label='retrieved')
ax1.plot(tTrans, outsider_saved, 'ro:', label='outsider')
ax1.set_ylabel('Retrieved index')
ax1.legend()

ax2 = plt.subplot(212)
ax2.plot(tTrans, max_m_mu_saved, 'g', linewidth=2)
ax2.plot(tTrans, max2_m_mu_saved, 'r', linewidth=2)
ax2.set_xlabel("Time")
ax2.set_ylabel("Overlap at transition")

plt.figure(3)
plt.hist(lamb, bins=20)
plt.xlim((0, 1))
# plt.ylim((0,5))
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
plt.savefig('histNit' + str(tSim))


C1C2C0 = correlations.cross_correlations(ksi_i_mu)

correlations.correlations_1D_hist(ksi_i_mu, C1C2C0)
correlations.correlations_2D_hist(ksi_i_mu, C1C2C0)

x0 = np.min(C1C2C0[:, 1])
x1 = np.max(C1C2C0[:, 1])
y0 = np.min(C1C2C0[:, 0])
y1 = np.max(C1C2C0[:, 0])

XX = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
                                    ksi_i_mu[:, outsider_saved])
YY = correlations.active_same_state(ksi_i_mu[:, retrieved_saved],
                                    ksi_i_mu[:, outsider_saved])
ZZ = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
                                  ksi_i_mu[:, outsider_saved])

C1C2C0 = correlations.cross_correlations(ksi_i_mu)
plt.figure('Correlations between transition patterns')
ax1 = plt.subplot(121)
plt.scatter(XX, YY, s=0.05)
ax1.set_ylabel('C1')
ax1.set_xlabel('C2')
ax1.set_xlim(x0, x1)
ax1.set_ylim(y0, y1)

ax2 = plt.subplot(122)
plt.hist2d(XX, YY, bins=15)
ax2.set_xlabel('C2')
ax2.set_xlim(x0, x1)
ax2.set_ylim(y0, y1)
plt.colorbar()
plt.title('Correlations between transition patterns')

plt.figure('Lambda, C')
plt.subplot(131)
plt.scatter(XX, lamb)
plt.xlabel('C1')
plt.subplot(132)
plt.scatter(YY, lamb)
plt.xlabel('C2')
plt.subplot(133)
plt.plot(ZZ, lamb)
plt.xlabel('Active Inactive')




plt.show()
