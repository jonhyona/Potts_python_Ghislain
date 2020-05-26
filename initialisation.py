"""Initializing the network

Routines listing
----------------
hebbian_tensor()
    Computes the hebbian tensor J_i_k_k_l, i.e. interactions between units.
network():
    Initializing the network in stationnary-rest-state
"""
import scipy.sparse as spsp
import numpy as np
import numpy.random as rd
import iteration
from parameters import N, S, p, a, U, beta, random_seed, cm, g_A, t_0

rd.seed(random_seed+2)


def hebbian_tensor(delta__ksi_i_mu__k):
    """
    Computes the hebbian tensor J_i_k_k_l, i.e. interactions between units.

    Parameters
    ----------
    delta__ksi_i_mu__k -- 2D array
        Set of patterns stored in the network

    Returns
    -------
    J_i_j_k_l -- 2D array
        Interaction tensor, given by the hebbian learning rule

    Notes
    -----
    Intuitively, J_i_j_k_l should be of dimenion 4. However, in order to use
    the sparse module from scipy, one has to have less than 2 dimensions. We
    use the convention that unit i in state k is indexed by ii = i*S + k.
    """

    # Building the connectivity matrix
    mask = spsp.lil_matrix((N, N))  # connectivity matrix
    deck = np.linspace(0, N-1, N, dtype=int)
    for i in range(N):
        rd.shuffle(deck)
        mask[i, deck[:int(cm)]] = True
        cpt = 0
        # Put diagonal coefficient to 0 keeping cm connections
        while mask[i, i]:
            mask[i, i] = False
            if int(cm) + cpt < N:
                mask[i, deck[int(cm)+cpt]] = True
            cpt += 1

    # Has to be expanded to fit the convention used in the notes
    kronMask = spsp.kron(mask, np.ones((S, S)))
    kronMask = kronMask.tobsr(blocksize=(S, S))

    J_i_j_k_l = np.dot(
        (delta__ksi_i_mu__k-a/S),
        np.transpose(delta__ksi_i_mu__k-a/S))
    J_i_j_k_l = kronMask.multiply(J_i_j_k_l)/(cm*a*(1-a/S))

    return spsp.bsr_matrix(J_i_j_k_l, blocksize=(S, S))


def network(J_i_j_k_l, delta__ksi_i_mu__k, g_A=g_A):
    """
    Initializing the network in stationnary-rest-state

    Returns
    -------
    r_i_k -- 1D array
        Input to unit i state k
    sig_i_k -- 1D array
        Activity of unit i state k
    theta_i_k -- 1D array
        Threshold for unit i state k
    r_i_S_A -- 1D array
        Activity of slow adaptive unit threshold
    r_i_S_B -- 1D array
        Activity of fast adaptive unit threshold
    h_i_k -- 1D array
        Field to unit i state k
    m_mu -- 1D array
        Overlap of the network with pattern mu
    dt_r_i_k_act -- 1D array
    dt_r_i_S_A -- 1D array
    dt_r_i_S_B -- 1D arrat
    dt_theta_i_k -- 1D array

    Notes
    -----
    Things are a bit messy here.
    Some variables contain nul states : r_i_k, sig_i_k... Some don't as
    theta_i_k. In the notes I chose to treat null states separately, which I
    hadn't in the beginning. So here one has to conventions:
        - when the array contains information about nul states, unit i state k
          is indexed by ii = i*(S+1) + k
        - when it doesn't contain information about the null state, it should
          be indexed by ii = i*S + k
    This is messy and should be changed in future version by having separate
    variables for the null state and active states

    """
    active = np.ones(N*(S+1), dtype='bool')
    inactive = active.copy()
    active[S::S+1] = False
    inactive[active] = False

    r_i_k = np.zeros(N*(S+1))
    r_i_k_act = r_i_k[active]
    r_i_S_A = g_A*r_i_k[inactive]
    r_i_S_B = (1-g_A)*r_i_k[inactive]
    sig_i_k = np.zeros(N*(S+1))

    m_mu = np.zeros(p)
    dt_r_i_k_act = np.zeros(r_i_k_act.shape)
    dt_r_i_S_A = np.zeros(r_i_S_A.shape)
    dt_r_i_S_B = np.zeros(r_i_S_B.shape)

    theta_i_k = np.zeros(N*S)
    dt_theta_i_k = np.zeros(theta_i_k.shape)
    h_i_k = np.zeros(theta_i_k.shape)

    # # Thresholds such that time-derivative are zero
    # def fun_r_i_S_A(x): g_A*S/(S+np.exp(beta*(x+U))) - x
    # r_i_S_A = (root(fun_r_i_S_A, 0).x)[0]*np.ones(len(r_i_S_A))

    # def fun_r_i_S_B(x): (1-g_A)*S/(S+np.exp(beta*(x+U))) - x
    # r_i_S_B = (root(fun_r_i_S_B, 0).x)[0]*np.ones(len(r_i_S_B))

    # theta_i_k = sig_i_k[active]
    # r_i_k[active] = r_i_k_act
    # r_i_k[inactive] = r_i_S_A+r_i_S_B

    # s[i][k]=(-2*beta-2*exp(beta*U)-2*S+sqrt(pow(2*beta+2*exp(beta*U)+2*S,2)+8*(-beta*beta-2*beta*S+2*beta*S*exp(beta*U))))/(2*(-beta*beta-2*beta*S+2*beta*S*exp(beta*U))
    sig_value = ((-2*beta-2*np.exp(beta*U)-2*S +
                  np.sqrt((2*beta+2*np.exp(beta*U)+2*S)**2
                          + 8*(-beta*beta-2*beta*S +
                               2*beta*S*np.exp(beta*U)))) /
                 (2*(-beta*beta-2*beta*S+2*beta*S*np.exp(beta*U))))
    sig_i_k[active] = sig_value
    sig_i_k[inactive] = 1 - S*sig_value
    theta_i_k[:] = sig_i_k[active]

    iteration.h_i_k_fun(h_i_k, J_i_j_k_l, sig_i_k, delta__ksi_i_mu__k, 0,
                        0, t_0)
    r_i_k[active] = h_i_k
    r_i_S_A = g_A * (1 - sig_i_k[inactive])
    r_i_S_B = (1 - g_A) * (1 - sig_i_k[inactive])
    r_i_k[inactive] = r_i_S_A + r_i_S_B

    return r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, \
        dt_r_i_S_A, dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k
