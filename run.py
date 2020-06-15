# coding=utf-8
"""Computes the evolution of the network cued with one pattern. The
pattern to cue, g_A and tSim can be set as parameters. Elsewise, they
will be taken from parameters.py
"""

import os

# It is more efficient to use only one core per cue, but launch
# several cues in parallel. So the number of threads numpy can use is
# set to 1
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import sys
# Local modules
from parameters import dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, \
    f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, \
    t_0, g, random_seed, p_0, n_p, nSnap, russo2008_mode
import initialisation
import iteration
import file_handling

# Tqdm is usefull to monitor evolution. As it is not installed on all
# systems, this allows to use it only if available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

# The pattern to cue, g_A and tSim can be set as parameters. Elsewise,
# they will be taken from parameters.py
if len(sys.argv) >= 2:
    cue = int(sys.argv[1])
else:
    cue = p_0
if len(sys.argv) >= 3:
    g_A = float(sys.argv[2])
if len(sys.argv) >= 4:
    tSim = float(sys.argv[3])
if len(sys.argv) >= 5:
    random_seed = int(sys.argv[4])
if len(sys.argv) >= 6:
    w = float(sys.argv[5])
if len(sys.argv) >= 7:
    a_pf = float(sys.argv[6])


param = (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode)

# Some data are saved using simple text. Others are stored using the
# pkl format, which enables to store any kind of data very easily
key = file_handling.get_key(param)
ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, _ = file_handling.load_network(key)

# Time arrays
tS = np.arange(0, tSim, dt)
nT = tS.shape[0]
tSnap = np.linspace(0, tSim, nSnap)

# Debugging
analyseTime = False
analyseDivergence = False

# Plot parameters
lamb = []                       # Crossovers
transition_time = []
retrieved_saved = []
max_m_mu_saved = []             # Maximal overlap
max2_m_mu_saved = []            # Second max overlap

# Outsider is the pattern with second highest overlap
outsider_saved = []

# At low crossover, remembers the pattern with maximal overlap even if
# it doesn't comes to the overlap threshold
just_next_saved = []
previously_retrieved_saved = []
transition_counter = 0
cpt_idle = 0                    # Counts time since when the network is idle
d12 = 0                         # Latching quality metric
eta = 0                         # Did a transition occur?
previously_retrieved = -1

r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
    dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k \
    = initialisation.network(J_i_j_k_l, delta__ksi_i_mu__k, g_A, w)

r_i_k_plot = np.zeros((nSnap, N*(S+1)))
m_mu_plot = np.zeros((nSnap, p))
theta_i_k_plot = np.zeros((nSnap, N*S))
sig_i_k_plot = np.zeros((nSnap, N*(S+1)))

previously_retrieved = cue
waiting_validation = False
eta = False
cpt_idle = 0
i_snap = 0
d12 = 0
duration = 0

coact_pos = np.zeros((p, p))
coact_neg = coact_pos.copy()

r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
    dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k = initialisation.network(
        J_i_j_k_l, delta__ksi_i_mu__k, g_A, w)

for iT in tqdm(range(nT)):
    iteration.iterate(J_i_j_k_l, delta__ksi_i_mu__k, tS[iT], analyseTime,
                      analyseDivergence, sig_i_k, r_i_k, r_i_S_A,
                      r_i_S_B, theta_i_k, h_i_k, m_mu, dt_r_i_S_A,
                      dt_r_i_S_B, dt_r_i_k_act, dt_theta_i_k, cue, t_0, g_A, w)

    # Saving data for plots
    if tS[iT] >= tSnap[i_snap]:
        r_i_k_plot[i_snap, :] = r_i_k
        m_mu_plot[i_snap, :] = m_mu
        sig_i_k_plot[i_snap, :] = sig_i_k
        theta_i_k_plot[i_snap, :] = theta_i_k
        i_snap += 1

    if tS[iT] > t_0+tau_1:
        # print((np.outer(m_mu, m_mu)).shape)
        coact = np.outer(m_mu, m_mu)
        tmp_ind = coact > coact_pos
        coact_pos[tmp_ind] = coact[tmp_ind]
        tmp_ind = coact < coact_neg
        coact_neg[tmp_ind] = coact[tmp_ind]

        duration = tS[iT]-t_0
        retrieved_pattern = np.argmax(m_mu)
        max_m_mu = m_mu[retrieved_pattern]
        m_mu[retrieved_pattern] = - np.inf
        outsider = np.argmax(m_mu)
        max2_m_mu = m_mu[outsider]
        m_mu[retrieved_pattern] = max_m_mu
        d12 += dt*(max_m_mu - max2_m_mu)

        # The transition detection should be adapted. It is simpler
        # and as efficient in the C code
        if retrieved_pattern != previously_retrieved \
           and not waiting_validation:
            tmp = [tS[iT], max_m_mu, retrieved_pattern,
                   previously_retrieved, outsider, max_m_mu, max2_m_mu]
            waiting_validation = True
            previous_idle = True
            new_reached_threshold = False
        # Transitions are validated only if previous patterns dies
        if waiting_validation and not previous_idle \
           and m_mu[tmp[2]] < 0.1:
            previous_idle = True
        # Transitions are validated only if the pattern reaches an overlap
        # of 0.5. This avoid to record low-crossover transitions when
        # latching dies
        if waiting_validation and not new_reached_threshold \
           and max_m_mu > .5:
            new_reached_threshold = True
        if waiting_validation and previous_idle \
           and new_reached_threshold:
            waiting_validation = False
            eta = True
            transition_time.append(tmp[0])
            lamb.append(tmp[1])
            just_next_saved.append(tmp[2])
            retrieved_saved.append(retrieved_pattern)
            previously_retrieved_saved.append(tmp[3])
            outsider_saved.append(tmp[4])
            max_m_mu_saved.append(tmp[5])
            max2_m_mu_saved.append(tmp[6])

            transition_counter += 1
            cpt_idle = 0
            eta = True
            previously_retrieved = retrieved_pattern

        # Check that the network asn't fallen into its rest state
        if max_m_mu < .01:
            cpt_idle += 1
            if cpt_idle > dt*100 and nT >= 1000:
                print('Latching died')
                break
        else:
            cpt_idle = 0

coactivation = coactivation / duration
mean_mu = mean_mu / duration
covariance = coactivation - np.outer(mean_mu, mean_mu)
d12 = eta*d12


file_handling.save_dynamics(cue, (transition_time, lamb, just_next_saved,
                                  retrieved_saved, previously_retrieved_saved,
                                  outsider_saved, max_m_mu_saved,
                                  max2_m_mu_saved),
                            key)

file_handling.save_evolution(cue, m_mu_plot, key)

file_handling.save_metrics(cue, d12, duration, key)

# file_handling.save_coactivation(cue, coactivation, key)
# file_handling.save_covariance(cue, covariance, key)

file_handling.save_coact_pos(cue, coact_pos, key)
file_handling.save_coact_neg(cue, coact_neg, key)

