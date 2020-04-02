# =============================================================================
# Generate correlated patterns from the algorithm described in
#     'Boboeva, V., Brasselet, R., & Treves, A. (2018). The capacity for
#     correlated semantic memories in the cortex. Entropy, 20(11), 824.'
#
# Parameters are defined in parameters.py
# =============================================================================
import numpy.random as rd
import numpy as np
from parameters import get_parameters
from parameters import get_f_russo
from time import time

dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind, \
    random_seed = get_parameters()

rd.seed(random_seed + 1)


def delta(i, j):
    return int(i == j)


def get_uncorrelated():
    ksi_i_mu = S*np.ones((N, p), dtype=int)
    for mu in range(p):
        tmp = np.arange(N)
        rd.shuffle(tmp)
        ind_active = tmp[:int(N*a)]
        ksi_i_mu[ind_active, mu] = rd.randint(0, S, int(N*a))

    # Compute patterns in a different form
    delta__ksi_i_mu__k = np.zeros((N*S, p))
    for i in range(N):
        for mu in range(p):
            for k in range(S):
                delta__ksi_i_mu__k[i*S+k, mu] = delta(ksi_i_mu[i, mu], k)

    return ksi_i_mu, delta__ksi_i_mu__k


def get_correlated():
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
    ind_units = np.linspace(0, N-1, N, dtype=int)

    ind_children = np.zeros((num_fact, p_fact), dtype='int')
    h_max = np.zeros(N)                        # State with maksi_mu_imul field
    s_max = np.zeros(N, dtype='int')           # Maximal field value
    ksi_mu_i = S*np.ones((p, N), dtype='int')  # Initialized in inactive state

    # Attribute p_fact children to each parent
    t0 = time()
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
    t1 = time()
    print('Attribute parents : ' + str(t1-t0))

    # Compute fields
    for mu in range(p):
        child_fields = np.zeros((N, S))

        for n in range(num_fact):
            expon = -dzeta*n
            for m in range(p_fact):
                if ind_children[n, m] == mu:
                    inputs = rd.binomial(1, a_pf, N) * rd.rand(N)
                    child_fields[ind_units, parents[ind_units, n]] \
                        = inputs*np.exp(expon)

                    # for i in range(N):
                    #     y = rd.rand()/a_pf
                    #     if y <= 1:
                    #         child_fields[i, parents[i, n]] += y*np.exp(expon)

        # Adds a small boost for sparse intput (small a_pf)
        rand_states = rd.randint(0, S, N)
        child_fields[ind_units, rand_states[ind_units]]  \
            = eps*rd.rand(N)
        # for i in range(N):
        #     randState = rd.randint(0, S)
        #     child_fields[i, randState] += eps*rd.rand()

        # Find state with maximal field
        s_max = np.argmax(child_fields, axis=1)
        h_max = child_fields[ind_units, s_max[ind_units]]
        # for i in range(N):
        #     s_max[i] = np.argmax(child_fields[i, :])
        #     h_max[i] = child_fields[i, s_max[i]]
                    
        # Only keep the N*a units with the stronger fields
        # Sorte is in increasing order
        indSorted = np.argsort(h_max)[int(N*(1-a)):]
        ksi_mu_i[mu, indSorted] = s_max[indSorted]
    t2 = time()
    print('Compute fields : '+str(t2-t1))

    # One needs ksi_i_mu
    ksi_i_mu = ksi_mu_i.transpose()

    # Compute patterns in a different form
    delta__ksi_i_mu__k = np.kron(ksi_i_mu, np.ones((S, 1)))
    k_mat = np.kron(np.ones((N, p)),
                    np.reshape(np.linspace(0, S-1, S), (S, 1)))
    print(k_mat.shape)
    delta__ksi_i_mu__k = delta__ksi_i_mu__k == k_mat
    # delta__ksi_i_mu__k = np.zeros((N*S, p))
    # for i in range(N):
    #     for mu in range(p):
    #         for k in range(S):
    #             delta__ksi_i_mu__k[i*S+k, mu] = delta(ksi_i_mu[i, mu], k)

    t3 = time()
    print('Deltas : ' + str(t3-t2))

    return ksi_i_mu, delta__ksi_i_mu__k


def get_vijay(f_russo):
    factors = rd.binomial(1, f_russo, (N,num_fact))

    sMax = S*np.ones(N,dtype='int')
    hMax = np.zeros(N)
    ksi_i_mu = S*np.ones((N,p),dtype='int')
    ind_unit = np.linspace(0,N-1,N,dtype=int)

    gamma_mu_n = rd.rand(p, num_fact)*rd.binomial(1,a_pf,(p,num_fact))
    expo_fact = np.exp(-dzeta*np.linspace(0,num_fact-1, num_fact))

    gamma_mu_n = gamma_mu_n*expo_fact[None,:]

    sigma_n = rd.randint(0,S,(p,num_fact))

    for mu in range(p):
        fields = np.zeros((N,S))
        
        for n in range(num_fact):
            fields[:,sigma_n[mu,n]] += gamma_mu_n[mu,n]*factors[:,n]

        fields[ind_unit, rd.randint(0, S, N)[ind_unit]] += eps*rd.rand(N)

        sMax = np.argmax(fields, axis=1)
        hMax = np.max(fields, axis=1)
        indSorted = np.argsort(hMax)[int(N*(1-a)):]

        ksi_i_mu[indSorted, mu] = sMax[indSorted]

    # Compute patterns in a different form
    delta__ksi_i_mu__k = np.kron(ksi_i_mu, np.ones((S, 1)))
    k_mat = np.kron(np.ones((N, p)),
                    np.reshape(np.linspace(0, S-1, S), (S, 1)))
    delta__ksi_i_mu__k = delta__ksi_i_mu__k == k_mat

    return ksi_i_mu, delta__ksi_i_mu__k
