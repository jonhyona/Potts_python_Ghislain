"""Cues the model with all possible patterns and shows the evolution of the
main parameters, as well as statistics on patterns and transitions that occured
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

# Standard libraries
import numpy as np
import scipy.stats as sts
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

n_min = 1
n_max = 4*len(lamb)/p
duration_bins = np.logspace(np.log10(n_min), np.log10(n_max), 10)
duration_x = np.logspace(np.log10(n_min), np.log10(n_max), 200)

g_A_s = [0., 0.5, 1.]
min_t = min(tau_1, tau_2, tau_3_A, tau_3_B)

color_s =['blue', 'orange', 'green']
for ind_g_A in range(len(g_A_s)):
    g_A = g_A_s[ind_g_A]
    set_name = str(hash((dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps,
                     f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A,
                     beta, tau, t_0, g, random_seed, p_0, n_p, nSnap))) + '.pkl'
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
        print('Run simulation first!, g_A = '+str(g_A))
    finally:
        f.close()

    ind_t = np.linspace(
        0, len(transition_time)-1, len(transition_time), dtype=int)
    lamb = np.array(lamb)
    transition_time = np.array(transition_time)
    lambda_threshold_s = [0.2, 0.25, 0.3, 0.35]
    n_th = len(lambda_threshold_s)

    ind_cue = 0
    previous_t = -np.inf
    new_cue_ind = []
    for it in range(len(transition_time)):
        tt = transition_time[it]
        if tt < previous_t:
            ind_cue += 1
            new_cue_ind.append(it)
        previous_t = tt
    print(ind_cue)
    new_cue_ind.append(+np.inf)

    for ind_th in range(n_th):
        high_sequence_durations = []
        lambda_threshold = lambda_threshold_s[ind_th]
        high_lambda = lamb > lambda_threshold
        low_lambda = lamb < lambda_threshold
        for mu in range(ind_cue):
            cue_ind = np.logical_and(ind_t < new_cue_ind[mu+1],
                                     ind_t >= new_cue_ind[mu])
            t_last_low = transition_time[cue_ind][0]
            low_transition_time = transition_time[
                np.logical_and(low_lambda, cue_ind)]
            high_transition_time = transition_time[
                np.logical_and(high_lambda, cue_ind)]
            for ii in range(len(low_transition_time)-1):
                t = low_transition_time[ii+1]
                if np.logical_and(high_transition_time < t,
                                  high_transition_time > t_last_low).any():
                    high_sequence_durations.append(np.sum(
                        np.logical_and(high_transition_time < t,
                                       high_transition_time > t_last_low)))
                t_last_low = t

            if len(high_transition_time) == 0:
                high_sequence_durations.append(0)

            elif len(low_transition_time) == 0 \
                    or low_transition_time[-1] < high_transition_time[-1]:
                high_sequence_durations.append(len(high_transition_time))
        high_sequence_durations = np.array(high_sequence_durations)

        smooth = sts.gaussian_kde(np.log10(high_sequence_durations+dt))

        plt.subplot(n_th//2+n_th%2, 2, ind_th+1)
        smooth_data = smooth(np.log10(duration_x))
        rug_data = np.histogram(high_sequence_durations, bins=duration_bins)[0]
        plt.hist(high_sequence_durations, bins=duration_bins, alpha=0.3, density=False, color=color_s[ind_g_A])
        smooth_data = np.max(rug_data)/np.max(smooth_data)*smooth_data
        plt.plot(duration_x, smooth_data, color=color_s[ind_g_A])
        plt.xlim(n_min, n_max)
        plt.xscale('log')
for ind_th in range(n_th):
    plt.subplot(n_th//2+n_th%2, 2, ind_th+1)
    plt.xlabel('Length of highly correlated sequence')
    plt.ylabel('Density')
    plt.title(r'$\lambda_{th}$='+str(lambda_threshold_s[ind_th])+', $w$='+str(w))
plt.subplot(n_th//2+n_th%2, 2, 1)
plt.legend({r'$g_A=0.0$', r'$g_A=0.5$', r'$g_A=1.0$'})
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout()
plt.show()


# s = 2
# shift = 1/N/a/5                 # In order categories to be visible in scatter
# lamb = np.array(lamb)

# low_cor = lamb < 0.2
# l_low_cor = r'$\lambda < 0.2$'
# mid_low_cor = np.logical_and(0.2 <= lamb, lamb < 0.6)
# l_mid_low_cor = r'$0.2 \leq \lambda < 0.6$'
# mid_high_cor = np.logical_and(0.6 <= lamb, lamb < 0.8)
# l_mid_high_cor = r'$0.6 \leq \lambda < 0.8$'
# high_cor = 0.8 <= lamb
# l_high_cor = r'$0.8 \leq \lambda $'

# C1C2C0 = correlations.cross_correlations(ksi_i_mu)

# ax_order = 'xC1yC2'
# if ax_order == 'xC1yC2':
#     xx = correlations.active_same_state(ksi_i_mu[:, retrieved_saved],
#                                         ksi_i_mu[:, outsider_saved])
#     yy = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
#                                         ksi_i_mu[:, outsider_saved])
#     zz = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
#                                       ksi_i_mu[:, outsider_saved])
#     XX = C1C2C0[:, 0]
#     YY = C1C2C0[:, 1]
# else:
#     xx = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
#                                         ksi_i_mu[:, outsider_saved])
#     yy = correlations.active_same_state(ksi_i_mu[:, retrieved_saved],
#                                         ksi_i_mu[:, outsider_saved])
#     zz = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
#                                       ksi_i_mu[:, outsider_saved])
#     XX = C1C2C0[:, 1]
#     YY = C1C2C0[:, 0]

# # x0 = np.min(XX) - 5*shift
# # x1 = np.max(XX) + 5*shift
# # y0 = np.min(YY) - 5*shift
# # y1 = np.max(YY) + 5*shift
# x0 = 0 - 5*shift
# x1 = 0.2 + 5*shift
# y0 = 0.1 - 5*shift
# y1 = 0.3 + 5*shift
# bins_y = np.arange(y0-shift, y1 + 1/N/a-shift, 1/N/a)
# bins_x = np.arange(x0-shift, x1 + 1/N/a-shift, 1/N/a)
# bins_z = np.arange(0, 1, 1/N/a)
# label_x = ax_order[1:3]
# label_y = ax_order[4:6]

# plt.ion()
# plt.close('all')

# plt.figure(2)
# plt.plot(tSnap[:, None], m_mu_plot)
# plt.xlabel('Time')
# plt.ylabel('Overlap')
# plt.title(r'w=' +str(w) + ', $\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
# plt.savefig(set_name[:-4] + '_time_evolution.png')

# plt.figure(1)
# sns.distplot(lamb)
# plt.xlabel(r'$\lambda$')
# plt.ylabel('Density')
# plt.title(r'w=' +str(w) + ', $\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
# plt.savefig(set_name[:-4] + '_crossover_histogram.png')

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

# plt.savefig(set_name[:-4] + '_2D_hist_and_scatter.png')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
