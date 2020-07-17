import file_handling
import matplotlib.pyplot as plt
import numpy as np
import correlations
import seaborn as sns
import numpy.random as rd

plt.ion()
plt.close('all')

# simulations = ['a2cc92e57feefe09afa4b7d522648850']
# simulations = ['f30d8a2438252005f6a9190c239c01c1']
simulations = ['9e0fbd728bd38ee6eb130d85f35faa9a']
# simulations = ['b18e30bc89dbcb5bc2148fb9c6e0c51d']
# simulations = ['ff9fe40ed43a94577c1cc2fea6453bf0']

n_seeds = 1
s = 8
alpha = 1
key = simulations[0]
key = '127820d0fdd3a25cf40f2b05dc08460e'

(dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm,
 a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0, g,
 random_seed, p_0, n_p, nSnap, russo2008_mode, kick_prop) = \
            file_handling.load_parameters(key)

print("Loading")
retrieved = file_handling.load_retrieved_several(n_seeds, key)
crossover = file_handling.load_crossover_several(n_seeds, key)
trans_times = file_handling.load_transition_time(0,  key)
ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, \
    C_i_j = file_handling.load_network(key)

times = file_handling.load_time(0, key)

simult_ret = file_handling.load_simult_ret(0, key)[0]
pair_crossovers = [[] for pair in range(p**2)]

previous = []
following = []
marriage_time = []
C_as = []
C_ad = []
C_ai = []
C_0 = []


for ind_marriage in range(len(simult_ret)):
    pattA = int(simult_ret[ind_marriage][0])
    pattB = int(simult_ret[ind_marriage][1])
    marriage_time.append(simult_ret[ind_marriage][3])
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


overlaps = file_handling.load_evolution(0, 0, key)
tS = np.array(file_handling.load_time(0, key)[0])
recorded = tS > 0.
trans_time = file_handling.load_transition_time(0, key)[0]

C1C2C0 = correlations.cross_correlations(ksi_i_mu)

plt.close('scatter_cas')
fig = plt.figure('scatter_cas')
ax1 = plt.subplot2grid((5, 1), (0, 0))
ax2 = plt.subplot2grid((5, 1), (1, 0))
ax3 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)

sns.distplot(C1C2C0[:, 0], norm_hist=True, kde=False, color='tab:red', ax=ax1)
ax1.set_title(r'Histogram of $C_{as}$ in all pairs')
ax1.set_xlim(0, 0.1)

sns.distplot(C_as, norm_hist=True, kde=False, color='tab:blue', ax=ax2)
ax2.set_title(r'Histogram of $C_{as}$ in pairs with simultaneous retrieval')
ax2.set_xlim(0, 0.1)

jitterx = 0.004*(rd.rand(len(C_as))-0.5)
jittery = 0.01*(rd.rand(len(marriage_time))-0.5)
ax3.scatter(C_as+jitterx, marriage_time+jittery, color='tab:blue', s=s, alpha=alpha, label='jittered')
# ax3.scatter(C_as, marriage_time, color='tab:purple', s=s, label='original')
ax3.set_xlim(0, 0.1)
ax3.set_title(r'$C_{as}$ and marriage duration')
ax3.set_xlabel(r'$C_{as}$')
ax3.set_ylabel(r'Marriage duration')
ax3.legend()
plt.tight_layout()


plt.close('scatter_cad')
fig = plt.figure('scatter_cad')
ax1 = plt.subplot2grid((5, 1), (0, 0))
ax2 = plt.subplot2grid((5, 1), (1, 0))
ax3 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)

sns.distplot(C1C2C0[:, 1], norm_hist=True, kde=False, color='tab:red', ax=ax1)
ax1.set_title(r'Histogram of $C_{ad}$ in all pairs')
ax1.set_xlim(0.1, 0.3)

sns.distplot(C_ad, norm_hist=True, kde=False, color='tab:blue', ax=ax2)
ax2.set_title(r'Histogram of $C_{ad}$ in pairs with simultaneous retrieval')
ax2.set_xlim(0.1, 0.3)
# ax2.set_xlim(0, 0.1)

jitterx = 0.004*(rd.rand(len(C_ad))-0.5)
jittery = 0.01*(rd.rand(len(marriage_time))-0.5)
ax3.scatter(C_ad+jitterx, marriage_time+jittery, color='tab:blue', s=s, alpha=alpha, label='jittered')
# ax3.scatter(C_ad, marriage_time, color='tab:purple', s=s, label='original')
# ax3.set_xlim(0, 0.1)
ax3.set_title(r'$C_{ad}$ and marriage duration')
ax3.set_xlabel(r'$C_{ad}$')
ax3.set_ylabel(r'Marriage duration')
ax3.legend()
ax3.set_xlim(0.1, 0.3)

plt.tight_layout()


plt.close('scatter_cai')
fig = plt.figure('scatter_cai')
ax1 = plt.subplot2grid((5, 1), (0, 0))
ax2 = plt.subplot2grid((5, 1), (1, 0))
ax3 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)

sns.distplot(C1C2C0[:, 2], norm_hist=True, kde=False, color='tab:red', ax=ax1)
ax1.set_title(r'Histogram of $C_{ai}$ in all pairs')
ax1.set_xlim(0.6, 0.9)

sns.distplot(C_ai, norm_hist=True, kde=False, color='tab:blue', ax=ax2)
ax2.set_title(r'Histogram of $C_{ai}$ in pairs with simultaneous retrieval')
ax2.set_xlim(0.6, 0.9)
# ax2.set_xlim(0, 0.1)

jitterx = 0.004*(rd.rand(len(C_ai))-0.5)
jittery = 0.01*(rd.rand(len(marriage_time))-0.5)
ax3.scatter(C_ai+jitterx, marriage_time+jittery, color='tab:blue', s=s, alpha=alpha, label='jittered')
# ax3.scatter(C_ai, marriage_time, color='tab:purple', s=s, label='original')
# ax3.set_xlim(0, 0.1)
ax3.set_title(r'$C_{ai}$ and marriage duration')
ax3.set_xlabel(r'$C_{ai}$')
ax3.set_ylabel(r'Marriage duration')
ax3.legend()
ax3.set_xlim(0.6, 0.9)
plt.tight_layout()

