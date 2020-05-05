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
from parameters import get_parameters, get_f_russo
import patterns
import correlations
import initialisation
import iteration
from scipy.spatial import ConvexHull

import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

plt.ion()
plt.close('all')
dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind, \
    random_seed = get_parameters()
f_russo = get_f_russo()

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
plt.title('Ale\'s C algorithm')

n_seeds = 10
n_pairs = len(data)
data = np.zeros(n_seeds*n_pairs)

for ind_seed in range(n_seeds):
    rd.seed(ind_seed)
    ksi_i_mu, delta__ksi_i_mu__k = patterns.get_vijay()
    C1C2C0 = correlations.cross_correlations(ksi_i_mu, normalized=False)
    data[ind_seed*n_pairs: (ind_seed+1)*n_pairs] = C1C2C0[:, 0]
c_min = np.min(data)
c_max = np.max(data)
c_bins = np.arange(c_min, c_max, 1)
c_ticks = np.arange(c_min, c_max, max(1, int((c_max-c_min)/15)))

plt.figure(2)
plt.hist(C1C2C0[:, 0], bins=c_bins,  edgecolor='black', density=True)
plt.xticks(c_ticks)
plt.title('Ghislain\'s Python algorithm')

plt.figure(3)
plt.hist(C1C2C0[:, 0], bins=c_bins_Ale,  edgecolor='black', density=True)
plt.xticks(c_ticks_Ale)
plt.title('Ghislain\'s Python algorithm without highly correlated')



# rd.seed(random_seed)
# f0 = 0
# f1 = 1
# df = 1/N
# a0 = 0
# a1 = 1
# da = 1/p

# # f_vect = np.arange(f0, f1, df)
# # a_vect = np.arange(a0, a1, da)
# f_vect = np.array([50/N])
# a_vect = np.array([0.25])
# saved = np.zeros((len(f_vect), len(a_vect), 5))
# pair_number = p*(p-1)//2
# seed_number = 20
# data = np.zeros(pair_number*seed_number)
# for rd_seed in range(seed_number):
#     for ind_f in tqdm(range(len(f_vect))):
#         for ind_a_pf in range(len(a_vect)):
#             a_pf = a_vect[ind_a_pf]
#             f_russo = f_vect[ind_f]
#             ksi_i_mu, delta__ksi_i_mu__k = patterns.get_vijay(f_russo, a_pf, rd_seed)

#             # print('Cross correlation computation')
#             C1C2C0 = correlations.cross_correlations(ksi_i_mu, normalized=False)

#             data[rd_seed*pair_number:(rd_seed+1)*pair_number] = C1C2C0[:, 0]

#             n_0 = np.sum(data==0)
#             n_1 = np.sum(data==1)
#             n_2 = np.sum(data==2)

#             saved[ind_f, ind_a_pf, 0] = n_0
#             saved[ind_f, ind_a_pf, 1] = n_1
#             saved[ind_f, ind_a_pf, 2] = n_2
#             saved[ind_f, ind_a_pf, 3] = np.sum(data > 13)
#             saved[ind_f, ind_a_pf, 4] = np.mean(data)

#             if n_0 < 450 and n_0 > 250:
#                 if n_1 > 450:
#                     if n_2 > 400:
#                         saved.append((a_pf, f_russo))
#                         print(a_pf, f_russo)
# bins = np.arange(np.min(data), np.max(data), 1)
# plt.hist(data, edgecolor='black',  density=True, bins=bins, stacked=True)
# plt.show()

# plt.ion()



# plt.close('all')
# plt.figure(1)
# plt.suptitle('One parent acts on one state')


# def encircle(x, y, ax=None, **kw):
#     if not ax: ax=plt.gca()
#     p = np.c_[x,y]
#     hull = ConvexHull(p)
#     poly = plt.Polygon(p[hull.vertices,:], ec='fuchsia', fc='coral', alpha=0.5)
#     ax.add_patch(poly)

# XX, YY = np.meshgrid(f_vect, a_vect, indexing='ij')
# plt.subplot(321)
# plt.contourf(XX, YY, saved[:, :, 0])
# to_encircle = np.logical_and(saved[:, :, 0] < 450, saved[:, :, 0] > 250)
# encircle(XX[to_encircle], YY[to_encircle])
# plt.colorbar()
# plt.title('Pairs with C1=0')
# plt.xlabel(r'$f$')
# plt.ylabel(r'$a_{pf}$')

# plt.subplot(322)
# plt.contourf(XX, YY, np.log(saved[:, :, 3]))
# plt.title('log(Pairs with C1>13)')
# plt.colorbar()
# plt.xlabel(r'$f$')
# plt.ylabel(r'$a_{pf}$')

# plt.subplot(323)
# plt.contourf(XX, YY, saved[:, :, 1])
# to_encircle = saved[:, :, 1] > 450
# if to_encircle.any():
#     encircle(XX[to_encircle], YY[to_encircle])
# plt.colorbar()
# plt.ylabel(r'$a_{pf}$')
# plt.xlabel(r'$f$')
# plt.ylabel(r'$a_{pf}$')
# plt.title('Pairs with C1=1')

# plt.subplot(324)
# plt.contourf(XX, YY, saved[:, :, 4])
# plt.title('Mean of C1')
# plt.colorbar()
# plt.xlabel(r'$f$')
# plt.ylabel(r'$a_{pf}$')

# plt.subplot(325)
# plt.contourf(XX, YY, saved[:, :, 2])
# to_encircle = saved[:, :, 2] > 350
# if to_encircle.any():
#     encircle(XX[to_encircle], YY[to_encircle])
# plt.colorbar()
# plt.title('Pairs with C1=3')
# plt.xlabel(r'$f$')
# plt.ylabel(r'$a_{pf}$')

# plt.tight_layout()
# plt.show()
