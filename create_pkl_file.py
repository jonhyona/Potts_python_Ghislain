# coding=utf-8
"""Creates the network and parameters file for one set of parameters
"""
import sys
# Local modules
from parameters import dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, \
    f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, \
    t_0, g, random_seed, p_0, n_p, nSnap, russo2008_mode, muted_prop
import initialisation
import file_handling
import patterns
import errno
import os

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
if len(sys.argv) >= 5:
    random_seed = int(sys.argv[4])
if len(sys.argv) >= 6:
    w = float(sys.argv[5])
if len(sys.argv) >= 7:
    a_pf = float(sys.argv[6])
if len(sys.argv) >= 8:
    kick_seed = int(sys.argv[7])

print(cue, g_A, tSim)


param = (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode, muted_prop)

key = file_handling.get_key(param)

if not os.path.exists('data_analysis/'+key):
    try:
        os.makedirs('data_analysis/'+key)
    except OSError as exc:      # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

ksi_i_mu, delta__ksi_i_mu__k = patterns.get_from_file(
    'pattern_S%d_a%.2f_apf%.2f_pfact%d_Nfact%d_Numfact%d_zeta%.3f'
    % (S, a, a_pf, p_fact, N, num_fact, dzeta))
# ksi_i_mu, delta__ksi_i_mu__k = patterns.get_uncorrelated()
J_i_j_k_l, C_i_j = initialisation.hebbian_tensor(delta__ksi_i_mu__k,
                                                 random_seed)

file_handling.save_parameters_pkl(param, key)
file_handling.save_network(ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l,
                           C_i_j, key)
