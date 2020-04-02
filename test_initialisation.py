import scipy.sparse as spsp
from scipy import stats
import numpy as np
import numpy.random as rd
from parameters import get_parameters
from scipy.optimize import root
import patterns


dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, ind_cue, \
    random_seed = get_parameters()

rd.seed(random_seed+2)

ksi_i_mu, delta__ksi_i_mu__k = patterns.get_correlated()


mask = spsp.lil_matrix((N, N))
deck = np.linspace(0, N-1, N, dtype=int)
for i in range(N):
    rd.shuffle(deck)
    mask[i, deck[:int(cm)]] = True
kronMask = spsp.kron(mask, np.ones((S, S)))
kronMask = kronMask.tobsr(blocksize=(S, S))

J_i_j_k_l = np.dot((delta__ksi_i_mu__k-a/S),
                   np.transpose(delta__ksi_i_mu__k-a/S))
J_i_j_k_l = kronMask.multiply(J_i_j_k_l)/(cm*a*(1-a/S))
