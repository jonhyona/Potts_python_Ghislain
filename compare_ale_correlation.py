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
import patterns
import correlations
import initialisation
import iteration
from scipy.spatial import ConvexHull
from parameters import dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, \
    f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, \
    t_0, g, random_seed, set_name, p_0, n_p, nSnap

import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

plt.ion()
plt.close('all')


ksi_i_mu, delta__ksi_i_mu__k \
    = patterns.get_from_file('pattern_generation_saved')
C1C2C0 = correlations.cross_correlations(ksi_i_mu, normalized=False)
data = C1C2C0[:, 0]
c_min = np.min(data)
c_max = np.max(data)
c_bins_Ale = np.arange(c_min, c_max, 1)
c_ticks_Ale = np.arange(c_min, c_max, max(1, int((c_max-c_min)/15)))

plt.figure(1)
plt.hist(C1C2C0[:, 0], bins=c_bins_Ale,  edgecolor='black', density=True)
plt.xticks(c_ticks_Ale)
plt.title(r"Ale's C algorithm, $a_{pf}=0.75$, $\zeta=0.02$")
