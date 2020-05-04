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
    f = open(set_name, 'rb')
    dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm, a, U, \
        w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0, g, \
        random_seed, p_0, n_p, nSnap \
        tS, tSnap, J_i_j_k_l, ksi_i_mu, delta__ksi_i_mu__k, \
        transition_time, lamb, retrieved_saved, previously_retrieved_saved, \
        outsider_saved, max_m_mu_saved, max2_m_mu_saved, \
        r_i_k_plot, m_mu_plot, sig_i_k_plot, theta_i_k_plot = pickle.load(f)
except IOError:
    exec(open(set_name).read())
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
x0 = np.min(C1C2C0[:, 1]) - 5*shift
x1 = np.max(C1C2C0[:, 1]) + 5*shift
y0 = np.min(C1C2C0[:, 0]) - 5*shift
y1 = np.max(C1C2C0[:, 0]) + 5*shift
bins_y = np.arange(y0-shift, y1 + 1/N/a-shift, 1/N/a)
bins_x = np.arange(x0-shift, x1 + 1/N/a-shift, 1/N/a)
bins_z = np.arange(0, 1, 1/N/a)

XX = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
                                    ksi_i_mu[:, outsider_saved])
YY = correlations.active_same_state(ksi_i_mu[:, retrieved_saved],
                                    ksi_i_mu[:, outsider_saved])
ZZ = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
                                  ksi_i_mu[:, outsider_saved])

plt.ion()

plt.figure(2)
plt.plot(tS[:, None], m_mu_plot)
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.title(r'$\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
plt.savefig('time_evolution_kwang_il_gA'+str(int(10*g_A))+'.png')

plt.figure(1)
sns.distplot(lamb)
plt.xlabel(r'$\lambda$')
plt.ylabel('Density')
plt.title(r'$\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
plt.savefig('hist_kwang_il_gA'+str(int(10*g_A))+'.png')

ax1 = plt.subplot(221)
ax1.scatter(XX[low_cor]+shift, YY[low_cor]+shift, s=s, c='orange',
            label=l_low_cor)
ax1.scatter(XX[mid_low_cor]+shift, YY[mid_low_cor]-shift, s=s, c='cyan',
            label=l_mid_low_cor)
ax1.scatter(XX[mid_high_cor]-shift, YY[mid_high_cor]+shift, s=s, c='m',
            label=l_mid_high_cor)
ax1.scatter(XX[high_cor], YY[high_cor], s=s, c='g', label=l_high_cor)
ax1.legend()
ax1.set_ylabel('C1')
ax1.set_xlabel('C2')
ax1.set_xlim(x0, x1)
ax1.set_ylim(y0, y1)
ax1.set_title('Correlations between transition patterns')

ax2 = plt.subplot(222)
plt.hist2d(XX, YY, bins=(bins_x, bins_y))
ax2.set_xlabel('C2')
ax2.set_xlim(x0, x1)
ax2.set_ylim(y0, y1)
plt.colorbar()

ax3 = plt.subplot(223)
plt.scatter(C1C2C0[:, 1], C1C2C0[:, 0], s=s)
ax3.set_xlim(x0, x1)
ax3.set_ylim(y0, y1)
ax3.set_title('Correlations between all patterns')

ax4 = plt.subplot(224)
plt.hist2d(C1C2C0[:, 1], C1C2C0[:, 0], bins=(bins_x, bins_y))
plt.colorbar()
ax4.set_xlim(x0, x1)
ax4.set_ylim(y0, y1)

plt.tight_layout()
plt.show()
