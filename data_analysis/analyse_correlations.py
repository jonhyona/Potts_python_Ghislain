"""Cues the model with all possible patterns and shows the evolution of the
main parameters, as well as statistics on patterns and transitions that occured
"""
import matplotlib.pyplot as plt

# Standard libraries
import numpy as np

# Local modules
import correlations
import seaborn as sns
import file_handling

plt.ion()
plt.close('all')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16
HUGE_SIZE = 15
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=HUGE_SIZE)  # fontsize of the figure title

simulations_correlated = ['b52b35150abd42f0cf538171dad1da5b',
                          'f8d35f84b2626d8311d1cbc33708e2b4',
                          '89c0ad60ba13ea850d51d883aa43007f',
                          '98ac2ddafa16495cce9124b326ba7e30',
                          '3edaf937c6ca25f0a17c0ec12d357e70',
                          '557846333a3a3cbd71e0a7e237e02ce5',
                          '81f17a40c09f63c57166178826580fab',
                          '8c87e68c2af01f21525217a489d76e91',
                          '2805a15bc1193a493a668d717adceff2']

simulations_correlated = ['f2f842f51d5180f4eb55beb8efb61882',
                          '625546bb732bd7bf3404e8a2c9193613',
                          'fe71122c46ad4af9f201b9123f36ca42',
                          '7cfd59eb4e3d3a26ec66e4249b22cfba',
                          '001319a7dbc27bb929f6c6d00bc4f08d',
                          '8b27f66e75c7f4f4427bbe59515c6e97',
                          '7218cda81b1e89d0dfc660c0a18ff912',
                          '03771e780bda036f8f2b8160bf2d85d4',
                          'f35c969f14b35efe505be6e417c03656',
                          'd9e7392b3817a1066541daa9309950ab',
                          '4b5ccb6c6231655784281eed38749ade',
                          'a0bfb97c0c519448fe9eac86a6c52a11']

key = simulations_correlated[8]

ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, _ = file_handling.load_network(key)

(dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm,
 a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0, g,
 random_seed, p_0, n_p, nSnap, russo2008_mode, _) = \
    file_handling.load_parameters(key)

crossovers = file_handling.load_crossover(0, key)
lamb = []
for ind_cue in range(p):
    lamb += crossovers[ind_cue][1:]
lamb = np.array(lamb)

retrieved = file_handling.load_retrieved(0, key)
retrieved_saved = []
previously_saved = []
for ind_cue in range(p):
    retrieved_saved += retrieved[ind_cue][1:]
    previously_saved += retrieved[ind_cue][:-1]

tSnap = np.array(file_handling.load_time(0, key)[0])
m_mu_plot = file_handling.load_evolution(0, 0, key)


s = 2
shift = 1/N/a/5                 # In order categories to be visible in scatter
lamb = np.array(lamb)

low_cor = lamb < 0.2
l_low_cor = r'$\lambda < 0.2$'
mid_low_cor = np.logical_and(0.2 <= lamb, lamb < 0.6)
l_mid_low_cor = r'$0.2 \leq \lambda < 0.6$'
mid_high_cor = np.logical_and(0.6 <= lamb, lamb < 0.8)
l_mid_high_cor = r'$0.6 \leq \lambda < 0.8$'
high_cor = 0.8 <= lamb
l_high_cor = r'$0.8 \leq \lambda $'

C1C2C0 = correlations.cross_correlations(ksi_i_mu)

ax_order = 'xC1yC2'
if ax_order == 'xC1yC2':
    xx = correlations.active_same_state(ksi_i_mu[:, retrieved_saved],
                                        ksi_i_mu[:, previously_saved])
    yy = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
                                        ksi_i_mu[:, previously_saved])
    zz = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
                                      ksi_i_mu[:, previously_saved])
    XX = C1C2C0[:, 0]
    YY = C1C2C0[:, 1]
else:
    xx = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
                                        ksi_i_mu[:, previously_saved])
    yy = correlations.active_same_state(ksi_i_mu[:, retrieved_saved],
                                        ksi_i_mu[:, previously_saved])
    zz = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
                                      ksi_i_mu[:, previously_saved])
    XX = C1C2C0[:, 1]
    YY = C1C2C0[:, 0]

# x0 = np.min(XX) - 5*shift
# x1 = np.max(XX) + 5*shift
# y0 = np.min(YY) - 5*shift
# y1 = np.max(YY) + 5*shift
x0 = 0 - 5*shift
x1 = 0.2 + 5*shift
y0 = 0.1 - 5*shift
y1 = 0.3 + 5*shift
bins_y = np.arange(y0-shift, y1 + 1/N/a-shift, 1/N/a)
bins_x = np.arange(x0-shift, x1 + 1/N/a-shift, 1/N/a)
bins_z = np.arange(0, 1, 1/N/a)
label_x = ax_order[1:3]
label_y = ax_order[4:6]

# plt.ion()
# plt.close('all')

# plt.figure(2)
# plt.plot(tSnap[:, None], m_mu_plot)
# plt.xlabel('Time')
# plt.ylabel('Overlap')
# plt.title(r'w=' +str(w) + ', $\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')

# plt.figure(1)
# sns.distplot(lamb)
# plt.xlabel(r'$\lambda$')
# plt.ylabel('Density')
# plt.title(r'w=' +str(w) + ', $\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')


# plt.figure('2D plots')
# plt.suptitle(r'Correlations between transition patterns, w=' +str(w) + ', $\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
# ax1 = plt.subplot(221)
# ax1.scatter(xx[low_cor]+shift, yy[low_cor]+shift, s=s, c='orange',
#             label=l_low_cor)
# ax1.scatter(xx[mid_low_cor]+shift, yy[mid_low_cor]-shift, s=s, c='cyan',
#             label=l_mid_low_cor)
# ax1.scatter(xx[mid_high_cor]-shift, yy[mid_high_cor]+shift, s=s, c='m',
#             label=l_mid_high_cor)
# ax1.scatter(xx[high_cor], yy[high_cor], s=s, c='g', label=l_high_cor)
# ax1.legend()
# ax1.set_ylabel(label_y)
# ax1.set_xlabel(label_x)
# ax1.set_xlim(x0, x1)
# ax1.set_ylim(y0, y1)
# ax1.hlines(a*(S-1)/S, x0, x1, colors='k')
# ax1.vlines(a/S, y0, y1, colors='k')

# ax2 = plt.subplot(222)
# plt.hist2d(xx, yy, bins=(bins_x, bins_y))
# ax2.set_xlabel(label_x)
# ax2.set_xlim(x0, x1)
# ax2.set_ylim(y0, y1)
# ax2.hlines(a*(S-1)/S, x0, x1, colors='w')
# ax2.vlines(a/S, y0, y1, colors='w')
# plt.colorbar()

# ax3 = plt.subplot(223)
# ax3.scatter(XX, YY, s=s)
# ax3.hlines(a*(S-1)/S, x0, x1, colors='k')
# ax3.vlines(a/S, y0, y1, colors='k')
# ax3.set_xlim(x0, x1)
# ax3.set_ylim(y0, y1)

# ax4 = plt.subplot(224)
# plt.hist2d(XX, YY, bins=(bins_x, bins_y))
# plt.colorbar()
# ax4.hlines(a*(S-1)/S, x0, x1, colors='w')
# ax4.vlines(a/S, y0, y1, colors='w')
# ax4.set_xlim(x0, x1)
# ax4.set_ylim(y0, y1)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

plt.close('Corr_region_report')
plt.figure('Corr_region_report')
ax4 = plt.subplot(211)
plt.hist2d(XX, YY, bins=(bins_x, bins_y))
plt.colorbar()
ax4.hlines(a*(S-1)/S, x0, x1, colors='w')
ax4.vlines(a/S, y0, y1, colors='w')
ax4.set_xlim(0, 0.15)
ax4.set_ylim(y0, y1)
ax4.set_ylabel(r'$C_{ad}$')

ax2 = plt.subplot(212)
plt.hist2d(xx, yy, bins=(bins_x, bins_y), density=True)
ax2.set_xlabel(label_x)
ax2.set_xlim(0, 0.15)
ax2.set_ylim(y0, y1)
ax2.hlines(a*(S-1)/S, x0, x1, colors='w')
ax2.vlines(a/S, y0, y1, colors='w')
ax2.set_ylabel(r'$C_{ad}$')
ax2.set_xlabel(r'$C_{as}$')

plt.colorbar()
plt.tight_layout()
