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

df = 0.1
f_russo_vect = np.arange(0, 1, df)

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
            if cpt_idle > nT/10 and nT >= 1000:
                print("latchingDied")
                l = tS[iT]
                break
        ind_max_prev = ind_max
Q = d12*l*eta

lamb = np.array(lamb)
gap = np.logical_and(lamb > 0.25, lamb < 0.55)
if len(lamb[gap]) < len(lamb)/10:
    print(f_russo, Q)
    myfile.write(str(f_russo) + '\n' + str(Q) + '\n \n')
