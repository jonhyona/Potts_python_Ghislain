import numpy.random as rd
import numpy as np
from parameters import get_parameters

dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind, \
    random_seed = get_parameters()

rd.seed(random_seed + 1)
import numpy.random as rd
import numpy as np

""" Generates correlated patterns from the parents-children algorithm

Returns
-------
ksi_i_mu -- 2D array of int
The states of each unit is an integers between 0 and S-1
delta__ksi_i_mu__k -- 2D array of bools
Index (i*S+k, mu) is True if unit i of pattern mu is in state k
"""

# Defines parents
parents = rd.randint(0, S, ((N, num_fact)))

ind_children = np.zeros((num_fact, p_fact), dtype='int')
h_max = np.zeros(N)                        # State with maksi_mu_imul field
s_max = np.zeros(N, dtype='int')           # Maximal field value
ksi_mu_i = S*np.ones((p, N), dtype='int')  # Initialized in inactive state

# Attribute p_fact children to each parent
print('Attribute parents')
deck = list(range(0, p))
for n in range(num_fact):
    rd.shuffle(deck)
    ind_children[n, :] = deck[:p_fact]
    # m = 0
    # while m < p_fact:
    #     child_candidate = rd.randint(0, p)
    #     already_picked = False
    #     for i in range(m):
    #         if ind_children[n, i] == child_candidate:
    #             already_picked = True
    #     if not already_picked:
    #         ind_children[n, m] = child_candidate
    #         m += 1

# Compute fields
print('Compute fields')
ind_unit_vect = np.linspace(0, N-1, N, dtype=int)
ind_parents = np.linspace(0, num_fact-1, num_fact, dtype=int)
expon_vect = np.exp(-dzeta*ind_parents)

for mu in range(p):
    child_fields = np.zeros((N, S+1))
    for n in range(num_fact):
        expon = -dzeta*n
        for m in range(p_fact):
            if ind_children[n, m] == mu:
                y = rd.binomial(1, a_pf, size=N)*rd.rand(N)
                child_fields[ind_unit_vect, parents[ind_unit_vect, n]] \
                    = y*expon_vect[n]
                # for i in range(N):
                #     y = rd.rand()/a_pf
                #     if y <= 1:
                #         child_fields[i, parents[i, n]] += y*np.exp(expon)

    # Adds a small boost for sparse intput (small a_pf)
    for i in range(N):
        randState = rd.randint(0, S)
        child_fields[i, randState] += eps*rd.rand()

    # Find state with maximal field
    for i in range(N):
        s_max[i] = np.argmax(child_fields[i, :])
        h_max[i] = child_fields[i, s_max[i]]

    # Only keep the N*a units with the stronger fields
    # Sorte is in increasing order
    indSorted = np.argsort(h_max)[int(N*(1-a)):]
    ksi_mu_i[mu, indSorted] = s_max[indSorted]

# One needs ksi_i_mu
ksi_i_mu = ksi_mu_i.transpose()

print('Deltas')
# Compute patterns in a different form
delta__ksi_i_mu__k = np.zeros((N*S, p))
for i in range(N):
    for mu in range(p):
        for k in range(S):
            delta__ksi_i_mu__k[i*S+k, mu] = delta(ksi_i_mu[i, mu], k)