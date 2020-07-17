import file_handling
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy.random as rd
import correlations

plt.ion()
plt.close('all')

# simulations = ['a2cc92e57feefe09afa4b7d522648850']
# simulations = ['f30d8a2438252005f6a9190c239c01c1']
simulations = ['9e0fbd728bd38ee6eb130d85f35faa9a']
# simulations = ['b18e30bc89dbcb5bc2148fb9c6e0c51d']
# simulations = ['ff9fe40ed43a94577c1cc2fea6453bf0']


n_seeds = 1
key = simulations[0]


retrieved = file_handling.load_retrieved_several(n_seeds, key)
crossover = file_handling.load_crossover_several(n_seeds, key)
trans_time = file_handling.load_transition_time(0, key)[0]

(dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm,
 a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0, g,
 random_seed, p_0, n_p, nSnap, russo2008_mode, kick_prop) = \
            file_handling.load_parameters(key)

ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, \
    C_i_j = file_handling.load_network(key)

pair_crossovers = [[] for pair in range(p**2)]
previous = []
following = []
crossovers = []
C_as = []
C_ad = []
C_ai = []
C_0 = []


for seed in range(n_seeds):
    for ind_cue in range(p):
        for ind_trans in range(3, len(retrieved[seed][ind_cue])-1):
            pattA = retrieved[seed][ind_cue][ind_trans]
            pattB = retrieved[seed][ind_cue][ind_trans+1]
            pair_crossovers[pattA*p +
                pattB].append(crossover[seed][ind_cue][ind_trans+1])
            crossovers.append(crossover[seed][ind_cue][ind_trans+1])
            C_as.append(correlations.active_same_state(
                ksi_i_mu[:, pattA],
                ksi_i_mu[:, pattB]))
            C_ad.append(correlations.active_diff_state(
                ksi_i_mu[:, pattA],
                ksi_i_mu[:, pattB]))
            C_ai.append(correlations.active_inactive(
                ksi_i_mu[:, pattA],
                ksi_i_mu[:, pattB]))
            previous.append(pattA)
            following.append(pattB)

# plt.figure('Crossover histo')
# plt.hist(crossovers, bins=100)

# prop_high = np.zeros(p**2)
# prop_low = np.zeros(p**2)
# means = np.zeros(p**2)
# stds = np.zeros(p**2)
# length = np.zeros(p**2)
# # thres = np.median(crossovers)
# thres = 0.5
# max_len = 0

# joy_list = []
# joy_means = []
# for pair in range(p**2):
#     if pair_crossovers[pair] == []:
#         prop_high[pair] = np.nan
#         prop_low[pair] = np.nan
#     else:
#         lamb = np.array(pair_crossovers[pair])
#         prop_high[pair] = np.sum(lamb >= thres)/len(lamb)
#         prop_low[pair] = np.sum(lamb < thres)/len(lamb)
#         length[pair] = len(lamb)
#         means[pair] = np.mean(lamb)
#         stds[pair] = np.std(lamb)
#         max_len = max(length[pair], max_len)
#         if length[pair] > 100:
#             if rd.rand() > 0.5:
#                 plt.figure('lambda')
#                 sns.distplot(lamb, kde_kws={'bw':
#                                             0.4}, hist=False)
#                 plt.xlim(0, 1)
#                 plt.ylabel('Proportion of transitions')
#                 plt.xlabel('Crossover')

# exist = length > 4
# prop_high = prop_high[exist]
# prop_low = prop_low[exist]
# means = means[exist]
# stds = stds[exist]
# length = length[exist]

# indices = np.argsort(prop_high)

# plt.close("high_low_crossover_repartition")
# plt.figure("high_low_crossover_repartition")
# plt.plot(prop_high[indices], label='High')
# plt.plot(prop_low[indices], label='Low')
# # plt.plot(means[indices])
# # plt.plot(stds[indices])
# # plt.plot(length[indices]/max_len, label='Number of transitions')
# plt.legend()
# plt.title(r"High and low crossover transitions, by pair, $\lambda_{th}$ = %.2f" % thres)
# plt.xlabel("Pair index, sorted by increasing high-crossover-proportion")
# plt.ylabel("Proportion of transitions")


# overlaps = file_handling.load_evolution(0, 0, key)
# tS = np.array(file_handling.load_time(0, key)[0])
# recorded = tS > 0.
# trans_time = file_handling.load_transition_time(0, key)[0]
# plt.figure('Evolution')
# plt.plot(tS[recorded], overlaps[recorded])

# plt.scatter(trans_time, crossover[0][0])

# C1C2C0 = correlations.cross_correlations(ksi_i_mu)

# plt.close('scatter_cas')
# fig = plt.figure('scatter_cas')
# ax1 = plt.subplot2grid((5, 1), (0, 0))
# ax2 = plt.subplot2grid((5, 1), (1, 0))
# ax3 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)

# sns.distplot(C1C2C0[:, 0], norm_hist=True, kde=False, color='tab:red', ax=ax1)
# ax1.set_title(r'Histogram of $C_{as}$ in all pairs')
# ax1.set_xlim(0, 0.1)

# sns.distplot(C_as, norm_hist=True, kde=False, color='tab:blue', ax=ax2)
# ax2.set_title(r'Histogram of $C_{as}$ in pairs with transitions')
# ax2.set_xlim(0, 0.1)

# jitterx = 0.004*(rd.rand(len(C_as))-0.5)
# jittery = 0.01*(rd.rand(len(crossovers))-0.5)
# ax3.scatter(C_as+jitterx, crossovers+jittery, color='tab:blue', s=1, alpha=0.2, label='jittered')
# ax3.scatter(C_as, crossovers, color='tab:purple', s=1, label='original')
# ax3.set_xlim(0, 0.1)
# ax3.set_title(r'$C_{as}$ and crossover')
# ax3.set_xlabel(r'$C_{as}$')
# ax3.set_ylabel(r'$\lambda$')
# ax3.legend()
# plt.tight_layout()


# plt.close('scatter_cad')
# fig = plt.figure('scatter_cad')
# ax1 = plt.subplot2grid((5, 1), (0, 0))
# ax2 = plt.subplot2grid((5, 1), (1, 0))
# ax3 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)

# sns.distplot(C1C2C0[:, 1], norm_hist=True, kde=False, color='tab:red', ax=ax1)
# ax1.set_title(r'Histogram of $C_{ad}$ in all pairs')
# ax1.set_xlim(0.1, 0.3)

# sns.distplot(C_ad, norm_hist=True, kde=False, color='tab:blue', ax=ax2)
# ax2.set_title(r'Histogram of $C_{ad}$ in pairs with transitions')
# ax2.set_xlim(0.1, 0.3)
# # ax2.set_xlim(0, 0.1)

# jitterx = 0.004*(rd.rand(len(C_ad))-0.5)
# jittery = 0.01*(rd.rand(len(crossovers))-0.5)
# ax3.scatter(C_ad+jitterx, crossovers+jittery, color='tab:blue', s=1, alpha=0.2, label='jittered')
# ax3.scatter(C_ad, crossovers, color='tab:purple', s=1, label='original')
# # ax3.set_xlim(0, 0.1)
# ax3.set_title(r'$C_{ad}$ and crossover')
# ax3.set_xlabel(r'$C_{ad}$')
# ax3.set_ylabel(r'$\lambda$')
# ax3.legend()
# ax3.set_xlim(0.1, 0.3)

# plt.tight_layout()


# plt.close('scatter_cai')
# fig = plt.figure('scatter_cai')
# ax1 = plt.subplot2grid((5, 1), (0, 0))
# ax2 = plt.subplot2grid((5, 1), (1, 0))
# ax3 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)

# sns.distplot(C1C2C0[:, 2], norm_hist=True, kde=False, color='tab:red', ax=ax1)
# ax1.set_title(r'Histogram of $C_{ai}$ in all pairs')
# ax1.set_xlim(0.6, 0.9)

# sns.distplot(C_ai, norm_hist=True, kde=False, color='tab:blue', ax=ax2)
# ax2.set_title(r'Histogram of $C_{ai}$ in pairs with transitions')
# ax2.set_xlim(0.6, 0.9)
# # ax2.set_xlim(0, 0.1)

# jitterx = 0.004*(rd.rand(len(C_ai))-0.5)
# jittery = 0.01*(rd.rand(len(crossovers))-0.5)
# ax3.scatter(C_ai+jitterx, crossovers+jittery, color='tab:blue', s=1, alpha=0.2, label='jittered')
# ax3.scatter(C_ai, crossovers, color='tab:purple', s=1, label='original')
# # ax3.set_xlim(0, 0.1)
# ax3.set_title(r'$C_{ai}$ and crossover')
# ax3.set_xlabel(r'$C_{ai}$')
# ax3.set_ylabel(r'$\lambda$')
# ax3.legend()
# ax3.set_xlim(0.6, 0.9)
# plt.tight_layout()


s = 2
shift = 1/N/a/5                 # In order categories to be visible in scatter
lamb = np.array(crossovers)

low_cor = lamb < 0.3
l_low_cor = r'$\lambda < 0.3$'
mid_low_cor = np.logical_and(0.3 <= lamb, lamb < 0.5)
l_mid_low_cor = r'$0.3 \leq \lambda < 0.5$'
mid_high_cor = np.logical_and(0.5 <= lamb, lamb < 0.7)
l_mid_high_cor = r'$0.5 \leq \lambda < 0.7$'
high_cor = 0.7 <= lamb
l_high_cor = r'$0.7 \leq \lambda $'

C1C2C0 = correlations.cross_correlations(ksi_i_mu)

ax_order = 'xC1yC2'
if ax_order == 'xC1yC2':
    xx = correlations.active_same_state(ksi_i_mu[:, previous],
                                        ksi_i_mu[:, following])
    yy = correlations.active_diff_state(ksi_i_mu[:, previous],
                                        ksi_i_mu[:, following])
    zz = correlations.active_inactive(ksi_i_mu[:, previous],
                                      ksi_i_mu[:, following])
    XX = C1C2C0[:, 0]
    YY = C1C2C0[:, 1]
else:
    xx = correlations.active_diff_state(ksi_i_mu[:, previous],
                                        ksi_i_mu[:, following])
    yy = correlations.active_same_state(ksi_i_mu[:, previous],
                                        ksi_i_mu[:, following])
    zz = correlations.active_inactive(ksi_i_mu[:, previous],
                                      ksi_i_mu[:, following])
    XX = C1C2C0[:, 1]
    YY = C1C2C0[:, 0]

x0 = 0 - 5*shift
x1 = 0.2 + 5*shift
y0 = 0.1 - 5*shift
y1 = 0.3 + 5*shift
bins_y = np.arange(y0-shift, y1 + 1/N/a-shift, 1/N/a)
bins_x = np.arange(x0-shift, x1 + 1/N/a-shift, 1/N/a)
bins_z = np.arange(0, 1, 1/N/a)
label_x = ax_order[1:3]
label_y = ax_order[4:6]


plt.figure('2D plots')
# plt.suptitle(r'Correlations between transition patterns, w=' +str(w) + ', $\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
ax1 = plt.subplot(321)
ax1.scatter(xx[low_cor]+shift, yy[low_cor]+shift, s=s, c='orange',
            label=l_low_cor)
ax1.scatter(xx[mid_low_cor]+shift, yy[mid_low_cor]-shift, s=s, c='cyan',
            label=l_mid_low_cor)
ax1.scatter(xx[mid_high_cor]-shift, yy[mid_high_cor]+shift, s=s, c='m',
            label=l_mid_high_cor)
ax1.scatter(xx[high_cor], yy[high_cor], s=s, c='g', label=l_high_cor)
ax1.legend()
ax1.set_ylabel(label_y)
ax1.set_xlim(x0, x1)
ax1.set_ylim(y0, y1)
ax1.hlines(a*(S-1)/S, x0, x1, colors='k')
ax1.vlines(a/S, y0, y1, colors='k')

ax2 = plt.subplot(322)
plt.hist2d(xx[low_cor], yy[low_cor], bins=(bins_x, bins_y), vmin=0, vmax=400)
ax2.set_xlim(x0, x1)
ax2.set_ylim(y0, y1)
ax2.hlines(a*(S-1)/S, x0, x1, colors='w')
ax2.vlines(a/S, y0, y1, colors='w')
plt.colorbar()
plt.title(l_low_cor)


ax3 = plt.subplot(323)
plt.hist2d(xx[mid_low_cor], yy[mid_low_cor], bins=(bins_x, bins_y), vmin=0, vmax=400)
ax3.set_xlim(x0, x1)
ax3.set_ylim(y0, y1)
ax3.hlines(a*(S-1)/S, x0, x1, colors='w')
ax3.vlines(a/S, y0, y1, colors='w')
ax3.set_ylabel(label_y)
plt.colorbar()
plt.title(l_mid_low_cor)

ax4 = plt.subplot(324)
plt.hist2d(xx[mid_high_cor], yy[mid_high_cor], bins=(bins_x, bins_y), vmin=0, vmax=400)
ax4.set_xlim(x0, x1)
ax4.set_ylim(y0, y1)
ax4.hlines(a*(S-1)/S, x0, x1, colors='w')
ax4.vlines(a/S, y0, y1, colors='w')
plt.colorbar()
plt.title(l_mid_high_cor)


ax5 = plt.subplot(325)
ax5.scatter(XX, YY, s=s)
ax5.hlines(a*(S-1)/S, x0, x1, colors='k')
ax5.vlines(a/S, y0, y1, colors='k')
ax5.set_xlim(x0, x1)
ax5.set_ylim(y0, y1)
ax5.set_ylabel(label_y)
ax5.set_xlabel(label_x)
ax5.set_title('All patterns')

ax6 = plt.subplot(326)
plt.hist2d(XX, YY, bins=(bins_x, bins_y))
plt.colorbar()
ax6.hlines(a*(S-1)/S, x0, x1, colors='w')
ax6.vlines(a/S, y0, y1, colors='w')
ax6.set_xlim(x0, x1)
ax6.set_ylim(y0, y1)
ax6.set_xlabel(label_x)
ax6.set_title('All patterns')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout()
plt.show()
