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
    t_0, g, random_seed, p_0, n_p, nSnap, russo2008_mode, kick_seed, muted_prop
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
if len(sys.argv) >= 8:
    kick_seed = int(sys.argv[7])


param = (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode, muted_prop)

# Some data are saved using simple text. Others are stored using the
# pkl format, which enables to store any kind of data very easily
key = file_handling.get_key(param)
ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, _ = file_handling.load_network(key)
cue_mask = iteration.get_units_to_cue(cue, kick_seed,
                                      delta__ksi_i_mu__k, muted_prop)

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

simult_ret = []
# Outsider is the pattern with second highest overlap
outsider_saved = []

# At low crossover, remembers the pattern with maximal overlap even if
# it doesn't comes to the overlap threshold
transition_counter = 0
cpt_idle = 0                    # Counts time since when the network is idle
d12 = 0                         # Latching quality metric
eta = 0                         # Did a transition occur?
previously_retrieved = -1

r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
    dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k \
    = initialisation.network(J_i_j_k_l, delta__ksi_i_mu__k, g_A, w, cue_mask)

r_i_k_plot = np.zeros((nSnap, N*(S+1)))
r_i_S_A_plot = np.zeros((nSnap, N))
r_i_S_B_plot = np.zeros((nSnap, N))
m_mu_plot = np.zeros((nSnap, p))
theta_i_k_plot = np.zeros((nSnap, N*S))
sig_i_k_plot = np.zeros((nSnap, N*(S+1)))
two_first_plot = np.zeros((nSnap, 2), dtype=int)
t_plot = np.zeros(nSnap)
max_m_mu_plot = t_plot.copy()
max2_m_mu_plot = t_plot.copy()

previously_retrieved = -1
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
        J_i_j_k_l, delta__ksi_i_mu__k, g_A, w, cue_mask)

for iT in tqdm(range(nT)):
    iteration.iterate(J_i_j_k_l, delta__ksi_i_mu__k, tS[iT],
                      analyseTime, analyseDivergence, sig_i_k, r_i_k,
                      r_i_S_A, r_i_S_B, theta_i_k, h_i_k, m_mu,
                      dt_r_i_S_A, dt_r_i_S_B, dt_r_i_k_act,
                      dt_theta_i_k, cue, t_0, g_A, w, cue_mask)

    # Saving data for plots

    # if tS[iT] > t_0+tau_1:
    # print((np.outer(m_mu, m_mu)).shape)
    # coact = np.outer(m_mu, m_mu)
    # tmp_ind = coact > coact_pos
    # coact_pos[tmp_ind] = coact[tmp_ind]
    # tmp_ind = coact < coact_neg
    # coact_neg[tmp_ind] = coact[tmp_ind]

    duration = tS[iT]-t_0
    retrieved_pattern = np.argmax(m_mu)
    max_m_mu = m_mu[retrieved_pattern]
    m_mu[retrieved_pattern] = - np.inf
    outsider = np.argmax(m_mu)
    max2_m_mu = m_mu[outsider]
    m_mu[retrieved_pattern] = max_m_mu
    d12 += dt*(max_m_mu - max2_m_mu)

    if tS[iT] >= tSnap[i_snap]:
        r_i_k_plot[i_snap, :] = r_i_k
        r_i_S_A_plot[i_snap, :] = r_i_S_A
        r_i_S_B_plot[i_snap, :] = r_i_S_B
        # m_mu_plot[i_snap, 0] = max_m_mu
        # m_mu_plot[i_snap, 1] = max2_m_mu
        m_mu_plot[i_snap, :] = m_mu
        sig_i_k_plot[i_snap, :] = sig_i_k
        theta_i_k_plot[i_snap, :] = theta_i_k
        two_first_plot[i_snap, 0] = retrieved_pattern
        two_first_plot[i_snap, 1] = outsider
        max_m_mu_plot[i_snap] = max_m_mu
        max2_m_mu_plot[i_snap] = max2_m_mu
        t_plot[i_snap] = tS[iT]
        i_snap += 1

    if retrieved_pattern != previously_retrieved and not waiting_validation:
        waiting_validation = True
        was_blocked = False
        crossover = max_m_mu
        trans_t = tS[iT]
        t0 = tS[iT]

    if waiting_validation and max_m_mu < crossover:
        crossover = max_m_mu
        trans_t = tS[iT]

    if retrieved_pattern != previously_retrieved and max_m_mu > 0.5 \
       and max_m_mu - max2_m_mu > 0.2:
        waiting_validation = False
        was_blocked = False
        eta = True
        t1 = tS[iT]
        transition_time.append(trans_t)
        lamb.append(crossover)
        retrieved_saved.append(retrieved_pattern)


        transition_counter += 1
        cpt_idle = 0
        eta = True
        previously_retrieved = retrieved_pattern
        last_blocker = outsider
        last_blocked = retrieved_pattern
        previously_retrieved = retrieved_pattern

    if was_blocked:
        blocked = retrieved_pattern
        blocker = outsider
        if blocker != last_blocker or blocked != last_blocked:
            # print("Blocked")
            # print(blocked, last_blocked, blocker, last_blocker)
            t1 = tS[iT]
            simult_ret.append((last_blocked, last_blocker, tS[iT], t1-t0))

    is_blocked = waiting_validation and max_m_mu > 0.5 \
        and max_m_mu - max2_m_mu <= 0.2

    if is_blocked:
        last_blocked = retrieved_pattern
        last_blocker = outsider

    was_blocked = is_blocked

    if tS[iT] > t_0+tau_1:
        # Check that the network asn't fallen into its rest state
        if max_m_mu < .01:
            cpt_idle += 1
            if cpt_idle > dt*100 and nT >= 1000:
                print('Latching died')
                break
        else:
            cpt_idle = 0

d12 = eta*d12

file_handling.save_transition_time(cue, kick_seed, transition_time, key)
file_handling.save_crossover(cue, kick_seed, lamb, key)
file_handling.save_retrieved(cue, kick_seed, retrieved_saved, key)
file_handling.save_max_m_mu(cue, kick_seed, max_m_mu_plot, key)
file_handling.save_max2_m_mu(cue, kick_seed, max2_m_mu_plot, key)
file_handling.save_simult_ret(cue, kick_seed, simult_ret, key)

file_handling.save_time(cue, kick_seed, t_plot, key)
file_handling.save_two_first(cue, kick_seed, two_first_plot, key)
file_handling.save_metrics(cue, kick_seed, d12, duration, key)

if cue == 0 and kick_seed == 0:
    file_handling.save_evolution(cue, kick_seed, m_mu_plot, key)
    file_handling.save_activation(cue, kick_seed, sig_i_k_plot, key)
    file_handling.save_thresholds(cue, kick_seed, theta_i_k_plot, key)
    file_handling.save_thresholds_B(cue, kick_seed, r_i_S_A_plot, key)
    file_handling.save_thresholds_A(cue, kick_seed, r_i_S_B_plot, key)

# file_handling.save_coact_pos(cue, kick_seed, coact_pos, key)
# file_handling.save_coact_neg(cue, kick_seed, coact_neg, key)
