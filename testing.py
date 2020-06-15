import sys
# Local modules
from parameters import dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, \
    f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, \
    t_0, g, random_seed, p_0, n_p, nSnap, russo2008_mode
import initialisation
import file_handling
import patterns

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


ksi_i_mub, delta__ksi_i_mu__kb, J_i_j_k_lb= file_handling.load_network(set_name)
ksi_i_muc, delta__ksi_i_mu__kc, J_i_j_k_lc= file_handling.load_network(set_name)

print((ksi_i_mu == ksi_i_mub).all())
print((ksi_i_muc==ksi_i_mub).all())
