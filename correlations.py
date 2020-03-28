"""Functions to compute pattern correlation and overlap

Routines listing
----------------
active_same_state(ksi1, ksi2)
    Proportion of units that are active in the same state
active_diff_state(ksi1, ksi2)
    Proportion of units that are active in the different states
cross_correlations(ksi_i_mu)
    C1, C2 for all the pairs of patterns
"""
import numpy as np
from parameters import get_parameters
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind \
    = get_parameters()


def active_same_state(ksi1, ksi2):
    """ Proportion of units that are active in the same state

    Parameters
    ----------
    ksi1 -- array of int
        First memory
    ksi2 -- array of int
        Second memory

    Returns
    -------
    C1 : float
        Proportion of untis active in the same state in both patterns
    """

    return np.sum((ksi1 == ksi2)*(1-(ksi2 == S)), axis=0)/N/a


def active_diff_state(ksi1, ksi2):
    """ Proportion of units that are active in the different states

    Parameters
    ----------
    ksi1 -- array of int
        First memory
    ksi2 -- array of int
        Second memory

    Returns
    -------
    C2 -- float
        Proportion of units active but in different states in the two patterns
    """
    return np.sum((1-(ksi1 == ksi2))*(1-(ksi2 == S))*(1-(ksi1 == S)), axis=0)/N/a


def active_inactive(ksi1, ksi2):
    return np.sum((ksi1 == S) * (1-(ksi2 == S)), axis=0)/N/a


def cross_correlations(ksi_i_mu):
    """ C1, C2 for all the pairs of patterns

    Parameters
    ----------
    ksi_i_mu -- 2D array of ints
        First axis corresponds to the different units' state
        Second axis corresponds to the different patterns

    Returns
    -------
    m_mu -- 2D array
        First column : C1
        Second column : C2
        Pairs are organized as (ksi_i, ksi_j with i < j)
    """

    items = [(i, j) for i in range(p) for j in range(i+1, p)]

    def fun(x):
        return (active_same_state(ksi_i_mu[:, x[0]], ksi_i_mu[:, x[1]]),
                active_diff_state(ksi_i_mu[:, x[0]], ksi_i_mu[:, x[1]]),
                active_inactive(ksi_i_mu[:, x[0]], ksi_i_mu[:, x[1]]))

    corr = map(fun, items)
    return np.array(list(corr))


def overlap(ksi1, ksi2):
    tmp = np.sum((ksi1 == ksi2) * (ksi1 != S))
    return 1/(a*N*(1-a/S)) \
        * (tmp - a/S*(a*N-tmp))


def correlations_2D_hist(ksi_i_mu, C1C2C0=None):
    if C1C2C0 is None:
        C1C2C0 = cross_correlations(ksi_i_mu)
    x0 = np.min(C1C2C0[:, 1])
    x1 = np.max(C1C2C0[:, 1])
    y0 = np.min(C1C2C0[:, 0])
    y1 = np.max(C1C2C0[:, 0])

    plt.figure('correlation_2D_hist')
    plt.title('Correlations between all patterns')

    plt.subplot(121)
    plt.scatter(C1C2C0[:, 1], C1C2C0[:, 0], s=0.05)
    plt.xlim(-0.1, 0.6)
    plt.xlabel('C2')
    plt.ylabel('C1')
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)

    plt.subplot(122)
    plt.hist2d(C1C2C0[:, 1], C1C2C0[:, 0], bins=20)
    plt.colorbar()
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.xlabel('C2')
    plt.ylabel('C1')
    return C1C2C0


def correlations_1D_hist(ksi_i_mu, C1C2C0=None):
    if C1C2C0 is None:
        C1C2C0 = cross_correlations(ksi_i_mu)

    plt.figure("correlations_1D_hist")
    plt.title('Correlations between all patterns')
    plt.subplot(311)
    plt.hist(C1C2C0[:, 0], bins=20)
    plt.xlim((0, 1))
    # plt.ylim((0,5))
    plt.xlabel(r'$C_1$')
    plt.ylabel(r'$\rho(C_1)$')
    plt.subplot(312)
    plt.hist(C1C2C0[:, 1], bins=20)
    plt.xlim((0, 1))
    plt.xlabel(r'$C_2$')
    plt.ylabel(r'$\rho(C_2)$')
    plt.subplot(313)
    plt.hist(C1C2C0[:, 2], bins=20)
    plt.ylabel('Active, inactive')
    plt.xlim((0, 1))
    plt.plot()


# # ksi_i_mu, delta__ksi_i_mu__k = get_uncorrelated()

# correlations_1D_hist(ksi_i_mu)
# plt.show()
