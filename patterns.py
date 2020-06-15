""" Generate correlated and uncorrelated patterns

Routines listing
----------------
get_uncorrelated()
    Creates a set of p uncorrelated patterns
get_vezha()
    Creates a set of correlated patterns using Vezha's algorithm
get_vijay()
    Creates a set of correlated patterns using Vijay's algorithm

Notes
-----
delta__ksi_i_mu__k is 1 if unit i is in state k in pattern mu. It should be a
3D-array. However, there is a convention in this project that when
required, the indices for unit i and state k are merged in one index :
ii = i*S+k. This allows to have arrays of maximal dimension 2, and use scipy
sparse module
"""
import numpy.random as rd
import numpy as np
from parameters import N, S, p, num_fact, p_fact, dzeta, a_pf, eps, a, \
    f_russo, random_seed
import pandas as pd


def get_uncorrelated(random_seed=random_seed):
    """ Generates a set of uncorrelated patterns

    Returns
    -------
    ksi_i_mu -- 2D array of int
        The states of each unit is an integers between 0 and S-1
    delta__ksi_i_mu__k -- 2D array of bools
        Index (i*S+k, mu) is True if unit i of pattern mu is in state k
    """

    rd.seed(random_seed + 1)

    # Patterns are generated in the rest state. Then, for each
    # pattern, Na units are attributed a random state
    ksi_i_mu = S*np.ones((N, p), dtype=int)
    for mu in range(p):
        deck = np.arange(N)
        rd.shuffle(deck)
        ind_active = deck[:int(N*a)]
        ksi_i_mu[ind_active, mu] = rd.randint(0, S, int(N*a))

    # Compute patterns in a different form
    delta__ksi_i_mu__k = np.zeros((N*S, p))
    for i in range(N):
        for mu in range(p):
            for k in range(S):
                delta__ksi_i_mu__k[i*S+k, mu] = ksi_i_mu[i, mu] == k

    return ksi_i_mu, delta__ksi_i_mu__k


def get_vezha(random_seed=random_seed):
    """ Generates correlated patterns from the parents-children algorithm used by
    Vezha (most recent algorithm)

    Returns
    -------
    ksi_i_mu -- 2D array of int
        The states of each unit is an integers between 0 and S-1
    delta__ksi_i_mu__k -- 2D array of bools
        Index (i*S+k, mu) is True if unit i of pattern mu is in state k

    Notes
    -----
    The algorithm implemented is described in
    'Boboeva, V., Brasselet, R., & Treves, A. (2018). The capacity for
    correlated semantic memories in the cortex. Entropy, 20(11), 824.'
    """

    rd.seed(random_seed + 1)

    ind_units = np.linspace(0, N-1, N, dtype=int)
    ind_children = np.zeros((num_fact, p_fact), dtype=int)

    parents = rd.randint(0, S, ((N, num_fact)))

    h_max = np.zeros(N)                        # Maximal field
    s_max = np.zeros(N, dtype='int')           # States with maximal field
    ksi_i_mu = S*np.ones((N, p), dtype='int')  # Generated patterns - children

    # Attribute p_fact children to each parent
    deck = list(range(0, p))
    for n in range(num_fact):
        rd.shuffle(deck)        # random permutation
        ind_children[n, :] = deck[:p_fact]

    # Compute fields, still to be optimized
    for mu in range(p):
        child_fields = np.zeros((N, S))
        for n in range(num_fact):
            expon = -dzeta*n
            for m in range(p_fact):
                if ind_children[n, m] == mu:
                    inputs = rd.binomial(1, a_pf, N) * rd.rand(N)
                    child_fields[ind_units, parents[ind_units, n]] \
                        = inputs*np.exp(expon)

        # Adds a small boost for sparse intput (small a_pf)
        rand_states = rd.randint(0, S, N)
        child_fields[ind_units, rand_states[ind_units]] = eps*rd.rand(N)

        # Find state with maximal field
        s_max = np.argmax(child_fields, axis=1)
        h_max = child_fields[ind_units, s_max[ind_units]]

        # Sort is by increasing order by default
        selected_units = np.argsort(-h_max)[:int(N*a)]
        ksi_i_mu[selected_units, mu] = s_max[selected_units]

    k_mat = np.kron(np.ones((N, p)),
                    np.reshape(np.linspace(0, S-1, S), (S, 1)))
    delta__ksi_i_mu__k = np.kron(ksi_i_mu, np.ones((S, 1)))

    delta__ksi_i_mu__k = delta__ksi_i_mu__k == k_mat

    return ksi_i_mu, delta__ksi_i_mu__k


def get_vijay(f_russo=f_russo, random_seed=random_seed):
    """ Generates correlated patterns from the parents-children algorithm used by
    Vijay in Russo2008

    Returns
    -------
    ksi_i_mu -- 2D array of int
        The states of each unit is an integers between 0 and S-1
    delta__ksi_i_mu__k -- 2D array of bools
        Index (i*S+k, mu) is True if unit i of pattern mu is in state k

    Notes
    -----
    The algorithm implemented is described in

    [1] 'Russo, E., Namboodiri, V. M., Treves, A., and Kropff, E. (2008). Free
    association transitions in models of cortical latching dynamics. New
    Journal of Physics, 10(1):015008.'

    [2] 'Treves, A. (2005). Frontal latching networks: a possible neural basis
    for infinite recursion. Cognitive neuropsychology, 22(3-4), 276-291.'
    """

    rd.seed(random_seed + 1)

    factors = np.zeros((N, num_fact))  # factors or parents
    deck = np.linspace(0, N-1, N, dtype=int)  # random permutation
    for n in range(num_fact):
        rd.shuffle(deck)
        factors[deck[:int(N*f_russo)], n] = 1
    sigma_n = rd.randint(0, S, num_fact)

    gamma_mu_n = rd.rand(p, num_fact) * rd.binomial(1, a_pf, (p, num_fact))
    expo_fact = np.exp(-dzeta*np.linspace(0, num_fact-1, num_fact))

    gamma_mu_n = gamma_mu_n*expo_fact[None, :]

    sMax = S*np.ones(N, dtype='int')  # State with maximal field
    hMax = np.zeros(N)                # Maximal field
    ksi_i_mu = S*np.ones((N, p), dtype='int')  # Patterns or children
    ind_unit = np.linspace(0, N-1, N, dtype=int)

    for mu in range(p):
        fields = np.zeros((N, S))
        for n in range(num_fact):
            fields[:, sigma_n[n]] += gamma_mu_n[mu, n]*factors[:, n]

        fields[ind_unit, rd.randint(0, S, N)[ind_unit]] += eps*rd.rand(N)

        sMax = np.argmax(fields, axis=1)
        hMax = np.max(fields, axis=1)
        indSorted = np.argsort(-hMax)[:int(N*a)]

        ksi_i_mu[indSorted, mu] = sMax[indSorted]

    delta__ksi_i_mu__k = np.kron(ksi_i_mu, np.ones((S, 1)))
    k_mat = np.kron(np.ones((N, p)),
                    np.reshape(np.linspace(0, S-1, S), (S, 1)))
    delta__ksi_i_mu__k = delta__ksi_i_mu__k == k_mat

    return ksi_i_mu, delta__ksi_i_mu__k


def get_2_patterns(C1, C2, random_seed=random_seed):
    """ Generates 2 correlated patterns with correlations C1 and C2

    Parameters
    ----------
    C1 -- int
        Number of shared active units and in the same state
    C2 -- int
        Number of shared active units but in different states

    Returns
    -------
    ksi_i_mu -- 2D array of int
        The states of each unit is an integers between 0 and S-1
    delta__ksi_i_mu__k -- 2D array of bools
        Index (i*S+k, mu) is True if unit i of pattern mu is in state k
    """
    rd.seed(random_seed + 1)


    if p != 2:
        print('Warning : p should be equal to 2')
    if S < 2:
        print('Warning : S should be at least 2')

    ksi1 = np.zeros(N, dtype=int)
    ksi2 = ksi1.copy()
    ksi1[:N*a] = 1
    ksi2[:C1] = 1
    ksi2[C1:N*a] = 2

    ksi_i_mu = np.zeros((N, 3), dtype=int)
    ksi_i_mu[:, 0] = ksi1
    ksi_i_mu[:, 1] = ksi2

    # Compute patterns in a different form
    delta__ksi_i_mu__k = np.kron(ksi_i_mu, np.ones((S, 1)))
    k_mat = np.kron(np.ones((N, p)),
                    np.reshape(np.linspace(0, S-1, S), (S, 1)))
    delta__ksi_i_mu__k = delta__ksi_i_mu__k == k_mat

    return ksi_i_mu, delta__ksi_i_mu__k


def readcsv(filename):
    data = pd.read_csv(filename, header=None)
    return(np.array(data))


def get_from_file(file_name):
    ksi_i_mu = np.zeros((N, p), dtype=int)
    ksi_i_mu[:, :] = np.transpose(readcsv(file_name))
    delta__ksi_i_mu__k = np.kron(ksi_i_mu, np.ones((S, 1)))
    k_mat = np.kron(np.ones((N, p)),
                    np.reshape(np.linspace(0, S-1, S), (S, 1)))
    delta__ksi_i_mu__k = delta__ksi_i_mu__k == k_mat
    return ksi_i_mu, delta__ksi_i_mu__k
