"""Functions to compute pattern correlation and overlap

Routines listing
----------------
active_same_state(ksi1, ksi2)
    Proportion of units that are active in the same state
active_diff_state(ksi1, ksi2)
    Proportion of units that are active in the different states
cross_correlations(ksi_i_mu)
    C1, C2 for all the pairs of patterns
overlap(ksi1, ksi2):
    Overlap with one pattern if the network was fully on the other
correlations_2D_hist(ksi_i_mu, C1C2C0=None):
    Plot the cross-distribution of correlations. There is one scatter-plot
    and one 2D-histogram
correlations_1D_hist(ksi_i_mu, C1C2C0=None):
    Plot distributions of correlations independently from one-another
"""
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from parameters import N, S, p, a

# Required to plot via ssh
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


def active_same_state(ksi1, ksi2, normalized=True):
    """ Proportion of units active in the same state in both patterns"""
    if normalized:
        return np.sum((ksi1 == ksi2)*(1-(ksi2 == S)), axis=0)/N/a
    return np.sum((ksi1 == ksi2)*(1-(ksi2 == S)), axis=0)


def active_diff_state(ksi1, ksi2, normalized=True):
    """  Proportion of units active but in different states in the two patterns
    """
    if normalized:
        return np.sum((1-(ksi1 == ksi2))*(1-(ksi2 == S))*(1-(ksi1 == S)),
                      axis=0)/N/a
    return np.sum((1-(ksi1 == ksi2))*(1-(ksi2 == S))*(1-(ksi1 == S)), axis=0)


def active_inactive(ksi1, ksi2, normalized=True):
    """ Proportion of units that are active in one state and inactive in the other
    """
    if normalized:
        return np.sum((ksi1 == S) * (1-(ksi2 == S)), axis=0)
    return np.sum((ksi1 == S) * (1-(ksi2 == S)), axis=0)



def cross_correlations(ksi_i_mu, normalized=True):
    """ C1, C2 for all pairs of patterns

    Parameters
    ----------
    ksi_i_mu -- 2D array of ints
        First axis corresponds to the different units' state
        Second axis corresponds to the different patterns

    Returns
    -------
    corr -- 2D array
        First column : C1
        Second column : C2
        Pairs are organized as (ksi_i, ksi_j with i < j)
    """
    items = [(i, j) for i in range(p) for j in range(i+1, p)]

    def fun(x):
        return (active_same_state(ksi_i_mu[:, x[0]], ksi_i_mu[:, x[1]],
                                  normalized),
                active_diff_state(ksi_i_mu[:, x[0]], ksi_i_mu[:, x[1]],
                                  normalized),
                active_inactive(ksi_i_mu[:, x[0]], ksi_i_mu[:, x[1]],
                                normalized))

    corr = map(fun, items)
    return np.array(list(corr))


def overlap(ksi1, ksi2):
    """ Overlap with one pattern if the network was fully on the other"""
    tmp = np.sum((ksi1 == ksi2) * (ksi1 != S))
    return 1/(a*N*(1-a/S)) \
        * (tmp - a/S*(a*N-tmp))


def correlations_2D_hist(ksi_i_mu, C1C2C0=None):
    """ Plot the cross-distribution of correlations. There is one scatter-plot
    and one 2D-histogram"""
    if C1C2C0 is None:
        C1C2C0 = cross_correlations(ksi_i_mu)
    x0, y0, x1, y1 = (0, 0, 1, 1)

    plt.figure('correlation_2D_hist')
    plt.title('Correlations between all patterns')

    plt.subplot(121)
    plt.scatter(C1C2C0[:, 1], C1C2C0[:, 0], s=0.05)

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
    """ Plot distributions of correlations independently from one-another"""
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
