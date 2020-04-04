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


# if 'ksi_i_mu'not in locals():
ksi_i_mu, delta__ksi_i_mu__k = patterns.get_vijay(f_russo)

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

    r_i_k_plot[iT, :] = r_i_k
    m_mu_plot[iT, :] = m_mu
    theta_i_k_plot[iT, :] = theta_i_k
    sig_i_k_plot[iT, :] = sig_i_k

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
            if cpt_idle > nT/10 and nT >= 1000:
                print("latchingDied")
                l = tS[iT]
                break
        ind_max_prev = ind_max

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
# for mu in retrieved_saved[:]:
    plt.plot(tS, m_mu_plot[:, mu])
plt.title('overlap')

active = np.ones(N*(S+1), dtype='bool')
active[S::S+1] = False
print(sum(sig_i_k[active]))

plt.show()
