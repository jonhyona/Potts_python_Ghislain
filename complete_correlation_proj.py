import file_handling
import sys
from tqdm import tqdm
import numpy as np
import correlations

print(sys.argv)
key = sys.argv[1]

ksi_i_mu, delta__ksi_i_mu__k, _, C_i_j = file_handling.load_network(key)
(dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
 cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
 g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
    file_handling.load_parameters(key)

correlation_proj = np.zeros((p*(p-1))//2, dtype=int)
cpt = 0

for patt1 in tqdm(range(p)):
    for patt2 in range(patt1+1, p):
        correlation_proj[cpt] = \
            correlations.correlation_patt_state(ksi_i_mu, C_i_j, patt1,
                                                patt2)
        cpt += 1

file_handling.save_correlation_proj(correlation_proj, key)
