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
from parameters import get_f_russo
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

f_russo = get_f_russo()

rd.seed(random_seed)

tS = np.arange(0, tSim, dt)
nT = tS.shape[0]

analyseTime = False
analyseDivergence = False

ksi_i_mu, delta__ksi_i_mu__k = patterns.get_vijay(f_russo)

# print('Computing hebbian tensor')
J_i_j_k_l = initialisation.hebbian_tensor(delta__ksi_i_mu__k)

# print('Initial conditions')
r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
    dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k = initialisation.network()

# print('Intégration')

# Plot parameters
lamb = []
tTrans = []
retrieved_saved = []
max_m_mu_saved = []
max2_m_mu_saved = []
outsider_saved = []
indTrans = 0
cpt_idle = 0
d12 = 0
l = tSim
eta = 0
ind_max_prev = -1

for iT in tqdm(range(nT)):
    iteration.iterate(J_i_j_k_l, delta__ksi_i_mu__k, tS[iT], analyseTime,
                      analyseDivergence, sig_i_k, r_i_k, r_i_S_A,
                      r_i_S_B, theta_i_k, h_i_k, m_mu, dt_r_i_S_A,
                      dt_r_i_S_B, dt_r_i_k_act, dt_theta_i_k)

        

    if tS[iT] > t_0 + tau:
        ind_max = np.argmax(m_mu)
        max_m_mu = m_mu[ind_max]
        m_mu[ind_max] = - np.inf
        outsider = np.argmax(m_mu)
        max2_m_mu = m_mu[outsider]

        d12 += dt*(max_m_mu- max2_m_mu)

        if ind_max != ind_max_prev:
            tTrans.append(tS[iT])
            lamb.append(max_m_mu)
            retrieved_saved.append(ind_max)
            outsider_saved.append(outsider)
            max_m_mu_saved.append(max_m_mu)
            max2_m_mu_saved.append(max2_m_mu)

            indTrans += 1
            cpt_idle = 0
            eta  = 1

        if max_m_mu < .01:
            cpt_idle += 1
            if cpt_idle > nT/100 and nT >= 1000:
                print("latchingDied")
                l = tS[iT]
                break
        else:
<<<<<<< HEAD
            cpt_idle = 0    
=======
            cpt_idle = 0
>>>>>>> scan_cm
        ind_max_prev = ind_max
Q = d12*l*eta



lamb = np.array(lamb)
gap = np.logical_and(lamb > 0.25, lamb < 0.55)
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

s = 1
low_cor = lamb < 0.2
l_low_cor = r'$\lambda < 0.2$'
mid_low_cor = np.logical_and(0.2 <= lamb, lamb < 0.6)
l_mid_low_cor = r'$0.2 <= \lambda < 0.6$'
mid_high_cor = np.logical_and(0.6 <= lamb, lamb < 0.8)
l_mid_high_cor = r'$0.6 <= \lambda < 0.8$'
high_cor = 0.8 <= lamb
l_high_cor = r'$0.8 <= \lambda $'

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


plt.figure('Correlations between transition patterns')
ax1 = plt.subplot(121)
ax1.scatter(XX[low_cor], YY[low_cor], s=s, c='orange', label=l_low_cor)
ax1.scatter(XX[mid_low_cor], YY[mid_low_cor], s=s, c='cyan', label=l_mid_low_cor)
ax1.scatter(XX[mid_high_cor], YY[mid_high_cor], s=s, c='m', label=l_mid_high_cor)
ax1.scatter(XX[high_cor], YY[high_cor], s=s, c='g', label=l_high_cor)
ax1.legend()

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
plt.hist2d(XX, lamb)
plt.xlabel('C1')
plt.colorbar()
plt.subplot(132)
plt.hist2d(YY, lamb)
plt.xlabel('C2')
plt.ylabel(r'$\lambda')
plt.colorbar()
plt.subplot(133)
plt.hist2d(ZZ, lamb)
plt.xlabel('Active Inactive')
plt.colorbar()


plt.show()
