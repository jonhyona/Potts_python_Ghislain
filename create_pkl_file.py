# coding=utf-8
"""Creates the network and parameters file for one set of parameters
"""
import sys
# Local modules
from parameters import dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, \
    f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, \
    t_0, g, random_seed, p_0, n_p, nSnap, russo2008_mode
import initialisation
import file_handling
import patterns

# The pattern to cue, g_A and tSim can be set as parameters. Elsewise,
# they will be taken from parameters.py. Not that this is just to have
# the proper file names, but these parameters are dynamic parameters
# and shouldn't change the network
if len(sys.argv) >= 2:
    cue = int(sys.argv[1])
else:
    cue = p_0
if len(sys.argv) >= 3:
    g_A = float(sys.argv[2])
if len(sys.argv) >= 4:
    tSim = float(sys.argv[3])

print(cue, g_A, tSim)


param = (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode)

set_name = file_handling.get_pkl_name(param)

if not file_handling.network_exists(set_name):
    ksi_i_mu, delta__ksi_i_mu__k = patterns.get_uncorrelated()
    J_i_j_k_l = initialisation.hebbian_tensor(delta__ksi_i_mu__k)

    file_handling.save_parameters_pkl(param, set_name)
    file_handling.save_network(ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l,
                               set_name,
                               param)
