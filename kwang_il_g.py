"""Cues the model with all possible patterns and shows the evolution of the
main parameters, as well as statistics on patterns and transitions that occured
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

# Standard libraries
import numpy as np
import numpy.random as rd

# Local modules
from parameters import set_name
import initialisation
import iteration
import correlations
import seaborn as sns
from tqdm import tqdm


# Required for ssh execution with plots
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

try:
    f = open('data_analysis/'+set_name, 'rb')
    dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm, a, U, \
        w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0, g, \
        random_seed, p_0, n_p, nSnap, \
        tS, tSnap, J_i_j_k_l, ksi_i_mu, delta__ksi_i_mu__k, \
        transition_time, lamb, retrieved_saved, previously_retrieved_saved, \
        outsider_saved, max_m_mu_saved, max2_m_mu_saved, \
        r_i_k_plot, m_mu_plot, sig_i_k_plot, theta_i_k_plot = pickle.load(f)
except IOError:
    print('Run simulation first!')
finally:
    f.close()


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
                                        ksi_i_mu[:, outsider_saved])
    yy = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
                                        ksi_i_mu[:, outsider_saved])
    zz = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
                                      ksi_i_mu[:, outsider_saved])
    XX = C1C2C0[:, 0]
    YY = C1C2C0[:, 1]
else:
    xx = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
                                        ksi_i_mu[:, outsider_saved])
    yy = correlations.active_same_state(ksi_i_mu[:, retrieved_saved],
                                        ksi_i_mu[:, outsider_saved])
    zz = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
                                      ksi_i_mu[:, outsider_saved])
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

plt.ion()
plt.close('all')

plt.figure(2)
plt.plot(tSnap[:, None], m_mu_plot)
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.title(r'w=' +str(w) + ', $\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
plt.savefig(set_name[:-4] + '_time_evolution.png')

plt.figure(1)
sns.distplot(lamb)
plt.xlabel(r'$\lambda$')
plt.ylabel('Density')
plt.title(r'w=' +str(w) + ', $\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
plt.savefig(set_name[:-4] + '_crossover_histogram.png')

plt.figure('2D plots')
plt.suptitle(r'Correlations between transition patterns, w=' +str(w) + ', $\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
ax1 = plt.subplot(221)
ax1.scatter(xx[low_cor]+shift, yy[low_cor]+shift, s=s, c='orange',
            label=l_low_cor)
ax1.scatter(xx[mid_low_cor]+shift, yy[mid_low_cor]-shift, s=s, c='cyan',
            label=l_mid_low_cor)
ax1.scatter(xx[mid_high_cor]-shift, yy[mid_high_cor]+shift, s=s, c='m',
            label=l_mid_high_cor)
ax1.scatter(xx[high_cor], yy[high_cor], s=s, c='g', label=l_high_cor)
ax1.legend()
ax1.set_ylabel(label_y)
ax1.set_xlabel(label_x)
ax1.set_xlim(x0, x1)
ax1.set_ylim(y0, y1)
ax1.hlines(a*(S-1)/S, x0, x1, colors='k')
ax1.vlines(a/S, y0, y1, colors='k')

ax2 = plt.subplot(222)
plt.hist2d(xx, yy, bins=(bins_x, bins_y))
ax2.set_xlabel(label_x)
ax2.set_xlim(x0, x1)
ax2.set_ylim(y0, y1)
ax2.hlines(a*(S-1)/S, x0, x1, colors='w')
ax2.vlines(a/S, y0, y1, colors='w')
plt.colorbar()

ax3 = plt.subplot(223)
ax3.scatter(XX, YY, s=s)
ax3.hlines(a*(S-1)/S, x0, x1, colors='k')
ax3.vlines(a/S, y0, y1, colors='k')
ax3.set_xlim(x0, x1)
ax3.set_ylim(y0, y1)

ax4 = plt.subplot(224)
plt.hist2d(XX, YY, bins=(bins_x, bins_y))
plt.colorbar()
ax4.hlines(a*(S-1)/S, x0, x1, colors='w')
ax4.vlines(a/S, y0, y1, colors='w')
ax4.set_xlim(x0, x1)
ax4.set_ylim(y0, y1)

plt.savefig(set_name[:-4] + '_2D_hist_and_scatter.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
