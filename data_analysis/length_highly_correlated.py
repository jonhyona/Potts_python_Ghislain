# coding=utf-8
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
import initialisation
import iteration
import correlations
import seaborn as sns
from tqdm import tqdm
import file_handling
from numpy import median


# Required for ssh execution with plots
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

plt.ion()
plt.close('all')
simulations = ['f1691446ec79cc0cb9d1f3f898c30585',
               'bd725ecccb665a616415eb80b3742729',
               '779e267d7fd11b394a96bc18ac9d2261']  # w=1.4

# simulations = ['6f276611a177a98a02697e035e772a70',
#                '50a7e2e50bf9b00dff6cd257844d51f7',
#                '2a123a981c3e2871ff8ff30383ecca93']  #  w=1.3

simulations_above = ['12257f9b2af7fdeaa5ebeec24b71b13c',
                     '2999e6e4eede18f9212d8abdd146e7f4',
                     '779e267d7fd11b394a96bc18ac9d2261']  # Just above the border

simulations_above = ['f2f842f51d5180f4eb55beb8efb61882',
                     '001319a7dbc27bb929f6c6d00bc4f08d',
                     'f35c969f14b35efe505be6e417c03656']


# ryom_data = ['seq_w1.4_gA0.0', 'seq_w1.4_gA0.5', 'seq_w1.4_gA1.0']
color_s = ['blue', 'orange', 'green']
color_s_ryom = ['navy', 'peru', 'darkolivegreen']


def plot_length_highly_correlated(simulation_list):
    for ind_key in range(len(simulation_list)):
        print('ind_key = %d' % ind_key)
        simulation_key = simulation_list[ind_key]

        (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
         cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
         g, random_seed, p_0, n_p, nSnap, russo2008_mode, _) = \
            file_handling.load_parameters(simulation_key)

        lamb = file_handling.load_crossover(0, simulation_key)
        lamb_med = []
        for ind_cue in range(p):
            lamb_med += lamb[ind_cue][1:]
        lamb_med = np.array(lamb_med)
        lamb_med = median(lamb_med)
        cue_number = len(lamb)

        n_min = 1
        n_max = 400
        # n_max = 10*file_handling.event_counter(lamb, p)/p
        duration_bins = np.linspace(n_min, n_max, 100)
        # duration_bins = np.logspace(np.log10(n_min), np.log10(n_max), 10)
        # duration_x = np.logspace(np.log10(n_min), np.log10(n_max), 200)

        lambda_threshold_s = [0.2, 0.25, 0.3, 0.35]
        lambda_threshold_s = [0]
        print(lamb_med)
        n_th = len(lambda_threshold_s)

        for ind_th in range(n_th):
            threshold = lambda_threshold_s[ind_th]
            high_sequence_durations = []
            for cue_ind in range(cue_number):
                length = 0
                for ind_trans in range(len(lamb[cue_ind])):
                    if lamb[cue_ind][ind_trans] > threshold:
                        length += 1
                    elif length != 0:
                        high_sequence_durations.append(length)
                        length = 0
                if length != 0:
                    high_sequence_durations.append(length)

            high_sequence_durations = np.array(high_sequence_durations)

            # smooth = sts.gaussian_kde(np.log10(high_sequence_durations+dt))

            # plt.subplot(n_th//2+n_th % 2, 2, ind_th+1)
            # smooth_data = smooth(np.log10(duration_x))
            # rug_data = np.histogram(high_sequence_durations, bins=duration_bins)[0]
            # smooth_data = np.max(rug_data)/np.max(smooth_data)*smooth_data
            # plt.plot(duration_x, smooth_data, color=color_s[ind_key])
            # plt.hist(high_sequence_durations, alpha=0.3,
            #          density=False, color=color_s[ind_key])
            # sns.kdeplot(high_sequence_durations, cumulative=True, color=color_s[ind_key], label='g_A %.1f, w %.1f' % (g_A, w))
            sns.distplot(high_sequence_durations, color=color_s[ind_key], label='g_A %.1f' % g_A, kde_kws={'bw':1})
            plt.xlim(n_min, n_max)
            # plt.xscale('log')
    for ind_th in range(n_th):
        # plt.subplot(n_th//2+n_th % 2, 2, ind_th+1)
        plt.xlabel('Length')
        plt.ylabel('Density')
        # plt.xscale('log')
        plt.yscale('log')
        # plt.title(
        #     r'$\lambda_{th}$='+str(lambda_threshold_s[ind_th])+',$w$='+str(w))
    # plt.subplot(n_th//2+n_th % 2, 1, 1)
    plt.legend()
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()
    plt.show()

plot_length_highly_correlated(simulations_above)






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
