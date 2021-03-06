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
import initialisation
import iteration
import correlations

import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

# plt.ion()

dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind, \
    random_seed = get_parameters()

cue_ind = 0
rd.seed(random_seed)

lambdas = np.nan*np.ones((int(N*a), int(N*a)))

max_C1 = 2*int(N*a*a/S)
max_C2 = int(2*N*a*a*(S-1)/S)
n_C1 = min(max_C1+1, 25)
n_C2 = min(max_C2+1, 25)

ksi_i_mu, delta__ksi_i_mu__k = patterns.get_uncorrelated()

for C2 in np.linspace(0, 20, 21, dtype=int):
    for C1 in np.linspace(0, 10, 11, dtype=int):
        ksi_i_mu_b, delta__ksi_i_mu__k_b = patterns.get_2_patterns(C1, C2)
        ksi_i_mu[:, :2] = ksi_i_mu_b
        delta__ksi_i_mu__k[:, :2] = delta__ksi_i_mu__k_b

        print("Hebbian matrix")
        J_i_j_k_l = initialisation.hebbian_tensor(delta__ksi_i_mu__k, cm)

        print("Initialisation")
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
        ind_max_prev_saved = []
        indTrans = 0
        cpt_idle = 0
        d12 = 0
        l = tSim
        eta = False
        waiting_validation = False
        ind_max_prev = cue_ind
        t_0_base = t_0
        for iT in tqdm(range(nT)):
            iteration.iterate(J_i_j_k_l, delta__ksi_i_mu__k, tS[iT], analyseTime,
                              analyseDivergence, sig_i_k, r_i_k, r_i_S_A,
                              r_i_S_B, theta_i_k, h_i_k, m_mu, dt_r_i_S_A,
                              dt_r_i_S_B, dt_r_i_k_act, dt_theta_i_k, cue_ind, t_0)

            r_i_k_plot[iT, :] = r_i_k
            m_mu_plot[iT, :] = m_mu
            theta_i_k_plot[iT, :] = theta_i_k
            sig_i_k_plot[iT, :] = sig_i_k

            if tS[iT] > t_0 + 10*tau:
                ind_max = np.argmax(m_mu)
                max_m_mu = m_mu[ind_max]
                m_mu[ind_max] = - np.inf
                outsider = np.argmax(m_mu)
                max2_m_mu = m_mu[outsider]

                d12 += dt*(max_m_mu- max2_m_mu)

                if ind_max != ind_max_prev and not waiting_validation:
                    tmp = [tS[iT], max_m_mu, ind_max, ind_max_prev, outsider, max_m_mu, max2_m_mu]
                    waiting_validation = True
                if waiting_validation and max_m_mu > .5:
                    waiting_validation = False
                    eta = True
                    # print(tS[iT], tmp[0])
                    tTrans.append(tmp[0])
                    lamb.append(tmp[1])
                    lambdas[C1, C2] = lamb[0]
                    retrieved_saved.append(ind_max)
                    ind_max_prev_saved.append(tmp[3])
                    outsider_saved.append(tmp[4])
                    max_m_mu_saved.append(tmp[5])
                    max2_m_mu_saved.append(tmp[6])
                    indTrans += 1
                ind_max_prev = ind_max
        plt.close('all')
        plt.figure(1)
        plt.subplot(311)
        plt.plot(tS[:, None], m_mu_plot)
        plt.title('C1 = ' + str(C1) + ', C2 = ' + str(C2))
        plt.ylabel('overlap')
        plt.xlim(0, tSim)
        plt.subplot(312)
        plt.title(str(retrieved_saved))
        plt.scatter(tTrans,
                    N*a*correlations.active_same_state(
                     ksi_i_mu[:, retrieved_saved],
                     ksi_i_mu[:, outsider_saved]),
                     label = 'C1')
        plt.scatter(tTrans,
                    N*a*correlations.active_diff_state(
                     ksi_i_mu[:, retrieved_saved],
                     ksi_i_mu[:, outsider_saved]),
                        label = 'C2')
        plt.xlim(0, tSim)
        plt.ylim(-1, 21)
        plt.legend()
        plt.subplot(313)
        plt.plot(tS[:, None], m_mu_plot[:, :2])
        plt.xlim(0, tSim)
        plt.tight_layout()
        plt.savefig('scan_corr/C1_'+str(C1)+'_C2_' +str(C2))
