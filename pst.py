import file_handling
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import proba_tools
from tqdm import tqdm

plt.ion()
plt.close('all')

simulations = ['f30d8a2438252005f6a9190c239c01c1']

alpha = 1
n_seeds = 11
key = simulations[0]
L = 3
p_min = 0.00001
g_min = 1e-4
alpha = 17.5
r = 1.6

(dt, tSim, N, S, p, num_fact, p_fact,
 dzeta, a_pf,
 eps,
 f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
 tau_3_B, g_A,
 beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
 russo2008_mode, muted_prop) = file_handling.load_parameters(simulations[0])

Y = 0
retrieved = file_handling.load_retrieved_several(n_seeds, key)

num_A, p_A, num_B, p_B, num_AB, p_AB, num_ABC, p_ABC, p_B_ABC, \
        p_AB_ABC = proba_tools.trio_prob_table(retrieved, key)

relevant_seq = [[] for order in range(0, L)]
relevant_seq[0] = [[Y]]

tables = proba_tools.build_trans_tables(retrieved, key, L)

for order in range(1, L):
    for chain in relevant_seq[order-1]:
        chain_numb = proba_tools.len_chain_counter(len(chain)+1,
                                                   retrieved, key)
        for X in range(p):
            Xchain = [X] + chain
            if X != chain[0]:
                p_Xchain = tables[len(Xchain)-1][tuple(Xchain)]
                if p_Xchain > p_min:
                    informative = False
                    for Z in range(p):
                        if Z != chain[-1]:
                            p_XY_Z = proba_tools.condi_prob(Z, Xchain, tables)
                            if p_XY_Z >= (1+alpha)*g_min:
                                p_Y_Z = proba_tools.condi_prob(Z,
                                                               chain,
                                                               tables)
                                print(X, Z, p_XY_Z, p_Y_Z)
                                if p_XY_Z / p_Y_Z >= r or p_XY_Z / p_Y_Z <= 1/r:
                                    informative = True
                    if informative:
                        relevant_seq[order].append(Xchain)
