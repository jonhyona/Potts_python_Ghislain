# =============================================================================
# First implementation of Potts Model
#   Author : Ghislain de Labbey
#   Date : 5th March 2020
# =============================================================================

# Required for ssh execution with plots
import os
# Standard libraries
import numpy as np
import numpy.random as rd
# Fancy libraries, not necessary
from tqdm import tqdm

# Local modules
from parameters import get_parameters
import patterns
import correlations
import initialisation
import iteration

import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind, \
    random_seed = get_parameters()

rd.seed(random_seed)

ksi_i_mu, delta__ksi_i_mu__k = patterns.get_vijay()

print('Cross correlation computation')
C1C2C0 = correlations.cross_correlations(ksi_i_mu)

plt.figure('patterns')
plt.imshow(ksi_i_mu)

correlations.correlations_1D_hist(ksi_i_mu, C1C2C0)
correlations.correlations_2D_hist(ksi_i_mu, C1C2C0)

plt.show()