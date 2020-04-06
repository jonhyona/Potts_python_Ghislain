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

tS = np.arange(0, tSim, dt)
nT = tS.shape[0]

analyseTime = False
analyseDivergence = False

cm0 = 25
cm1 = 100

cm_vect = np.linspace(cm0, cm1-1,cm1-cm0, dtype=int)
Q_vect = np.zeros(cm1 - cm0)
l_vect = Q_vect.copy()
d12_vect = Q_vect.copy()

ksi_i_mu, delta__ksi_i_mu__k = patterns.get_vijay()

for ind_cm in tqdm(range(len(cm_vect))):
    cm = cm_vect[ind_cm]

    # print('Computing hebbian tensor')
    J_i_j_k_l = initialisation.hebbian_tensor(delta__ksi_i_mu__k, cm)

    # print('Initial conditions')
    r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
        dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k = initialisation.network()

    # print('Intégration')
    
    # Plot parameters
    cpt_idle = 0
    d12 = 0
    l = tSim-t_0
    eta = False
    ind_max_prev = -1

    for iT in tqdm(range(nT)):
        iteration.iterate(J_i_j_k_l, delta__ksi_i_mu__k, tS[iT], analyseTime,
                          analyseDivergence, sig_i_k, r_i_k, r_i_S_A,
                          r_i_S_B, theta_i_k, h_i_k, m_mu, dt_r_i_S_A,
                          dt_r_i_S_B, dt_r_i_k_act, dt_theta_i_k)
        if tS[iT] > t_0 + tau:
            ind_max = np.argmax(m_mu)
            max_m_mu = m_mu[ind_max]
            m_mu[ind_max] = -np.inf
            max2_m_mu = np.max(m_mu)
            d12 += dt*(max_m_mu- max2_m_mu)
            if not eta and ind_max != ind_max_prev:
                eta  = True
            if max_m_mu < .01:
                cpt_idle += 1
                if cpt_idle > nT/10 and nT >= 1000:
                    print("latchingDied")
                    l = tS[iT]-t_0
                    break
            else:
                cpt_idle = 0
            ind_max_prev = ind_max
    d12 = d12/l
    l = l/tSim
    Q_vect[ind_cm] = d12*l*eta
    l_vect[ind_cm] = l
    d12_vect[ind_cm] = d12

plt.subplot(311)
plt.plot(cm_vect, Q_vect)
plt.subplot(312)
plt.ylabel(r'\frac{d12/l}')
plt.plot(cm_vect, d12_vect)
plt.subplot(313)
plt.plot(cm_vect, l_vect/tSim)
plt.ylabel(r'\frac{l/t_Sim')
plt.show()
