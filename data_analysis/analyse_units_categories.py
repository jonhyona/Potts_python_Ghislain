import file_handling
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy.random as rd
from iteration import spread_active_states, sum_active_states, active

plt.ion()
# plt.close('all')


key = '6a9b5fece80fcc9f74ec85417b1e3003'
key = '7d8e6641617fb92a5cd9e0f442525ec8'

n_seeds = 1


retrieved = file_handling.load_retrieved_several(n_seeds, key)
crossover = file_handling.load_crossover_several(n_seeds, key)
trans_time = file_handling.load_transition_time(0, key)[0]
tS = file_handling.load_time_cue(0, 0, key)

(dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm,
 a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0, g,
 random_seed, p_0, n_p, nSnap, russo2008_mode, kick_prop) = \
            file_handling.load_parameters(key)

ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, \
    C_i_j = file_handling.load_network(key)


def field(sig_i_k):
    sig_i_k_act = sig_i_k[active]
    h_i_k = J_i_j_k_l.dot(sig_i_k_act)
    h_i_k += w*sig_i_k_act
    if not russo2008_mode:
        h_i_k -= w/S*spread_active_states.dot(
            sum_active_states.dot(sig_i_k_act))
    return h_i_k


cue = 0
sig_i_k = file_handling.load_activation_cue(0, 0, key)
r_i_k = file_handling.load_thresholds_cue(0, 0, key)
r_i_S_A = file_handling.load_thresholds_S_A_cue(0, 0, key)
r_i_S_B = file_handling.load_thresholds_S_B_cue(0, 0, key)

for ii in range(N):
    for kk in range(S):
        r_i_k[:, ii*S + kk] += r_i_S_B[:, ii]

h_i_k = np.zeros((len(tS), N*S))
for tt in range(len(tS)):
    h_i_k[tt, :] = field(sig_i_k[tt, :])

m_mu = np.array(file_handling.load_evolution(cue, 0, key))

recorded = tS > 0.
tS = tS[recorded]
m_mu = m_mu[recorded, :]
sig_i_k = sig_i_k[recorded, :]
r_i_k = r_i_k[recorded, :]
h_i_k = h_i_k[recorded, :]

ind_trans = 5
# mu = retrieved[0][0][ind_trans]
# nu = retrieved[0][0][ind_trans+1]
mu = 11
nu = 39
color_dict = {181:'tab:blue', 35:'tab:orange', 97:'tab:blue', 39:'tab:orange'}
trans_num = 3


def plot_fun(mu, nu):
    colorstring = color_dict[nu]

    tmp0 = np.logical_and(ksi_i_mu[:, mu] == ksi_i_mu[:, nu],
                          ksi_i_mu[:, mu] != S)
    tmp1 = np.array(range(N))[tmp0]
    active_same_state = (S+1)*tmp1 + ksi_i_mu[tmp0, mu]

    tmp0 = np.logical_and(ksi_i_mu[:, mu] != ksi_i_mu[:, nu],
                          np.logical_and(ksi_i_mu[:, mu] != S, ksi_i_mu[:,
                                                                        nu] != S))
    tmp1 = np.array(range(N))[tmp0]
    active_diff_state_mu = (S+1)*tmp1 + ksi_i_mu[tmp0, mu]
    active_diff_state_nu = (S+1)*tmp1 + ksi_i_mu[tmp0, nu]

    tmp0 = np.logical_and(ksi_i_mu[:, mu] != S, ksi_i_mu[:, nu] == S)
    tmp1 = np.array(range(N))[tmp0]
    active_mu_inactive = (S+1)*tmp1 + ksi_i_mu[tmp0, mu]

    tmp0 = np.logical_and(ksi_i_mu[:, mu] == S, ksi_i_mu[:, nu] != S)
    tmp1 = np.array(range(N))[tmp0]
    active_nu_inactive = (S+1)*tmp1 + ksi_i_mu[tmp0, nu]

    tmp0 = np.logical_and(ksi_i_mu[:, mu] == ksi_i_mu[:, nu],
                          ksi_i_mu[:, mu] != S)
    tmp1 = np.array(range(N))[tmp0]
    active_same_state2 = S*tmp1 + ksi_i_mu[tmp0, mu]

    tmp0 = np.logical_and(ksi_i_mu[:, mu] != ksi_i_mu[:, nu],
                          np.logical_and(ksi_i_mu[:, mu] != S, ksi_i_mu[:,
                                                                        nu] != S))
    tmp1 = np.array(range(N))[tmp0]
    active_diff_state_mu2 = S*tmp1 + ksi_i_mu[tmp0, mu]
    active_diff_state_nu2 = S*tmp1 + ksi_i_mu[tmp0, nu]

    tmp0 = np.logical_and(ksi_i_mu[:, mu] != S, ksi_i_mu[:, nu] == S)
    tmp1 = np.array(range(N))[tmp0]
    active_mu_inactive2 = S*tmp1 + ksi_i_mu[tmp0, mu]

    tmp0 = np.logical_and(ksi_i_mu[:, mu] == S, ksi_i_mu[:, nu] != S)
    tmp1 = np.array(range(N))[tmp0]
    active_nu_inactive2 = S*tmp1 + ksi_i_mu[tmp0, nu]

    plt.figure('Overlap')
    alpha = 0.1
    plt.subplot(2, 2, 1)
    plt.plot(tS, m_mu[:, mu], 'tab:blue')
    plt.plot(tS, m_mu[:, nu], 'tab:red')
    plt.plot(tS, m_mu, ':k')
    plt.xlim(300, 1300)
    for ind_trans in range(1, len(retrieved[0][cue])):
        if trans_time[ind_trans] > 300 and trans_time[ind_trans] < 1300:
            plt.text(trans_time[ind_trans], 0.9,
                     str(retrieved[0][cue][ind_trans]))
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.title('Overlap')


    plt.subplot(2, 2, 2)
    plt.plot(tS, sig_i_k[:, active_same_state], color=colorstring, alpha=alpha)
    plt.plot(tS, np.mean(sig_i_k[:, active_same_state], axis=1), color=colorstring, linewidth=4)
    plt.title('Active in the same state')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)

    plt.subplot(2, 2, 3)
    plt.plot(tS, sig_i_k[:, active_diff_state_nu], color=colorstring, alpha=alpha)
    plt.plot(tS, np.mean(sig_i_k[:, active_diff_state_nu], axis=1), color=colorstring, linewidth=4)
    plt.title('Active in different states')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)

    plt.subplot(2, 2, 4)
    plt.plot(tS, sig_i_k[:, active_nu_inactive], color=colorstring, alpha=alpha)
    plt.plot(tS, np.mean(sig_i_k[:, active_nu_inactive], axis=1), color=colorstring, linewidth=4)
    plt.title('Active-inactive')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)

    plt.suptitle('Activation of units based on their category')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    alpha = 0.01
    plt.figure('Thresholds')
    plt.subplot(2, 2, 1)
    plt.plot(tS, m_mu[:, nu], colorstring)
    for ind_trans in range(1, len(retrieved[0][cue])):
            if trans_time[ind_trans] > 300 and trans_time[ind_trans] < 1300:
                plt.text(trans_time[ind_trans], 0.9,
                         str(retrieved[0][cue][ind_trans]))
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.title('Overlap')
    plt.xlim(300, 1300)


    plt.subplot(2, 2, 2)
    plt.plot(tS, r_i_k[:, active_same_state2], colorstring, alpha=alpha)
    plt.plot(tS, np.mean(r_i_k[:, active_same_state2], axis=1), colorstring, linewidth=2)
    plt.title('Active in the same state')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)

    plt.subplot(2, 2, 3)
    plt.plot(tS, r_i_k[:, active_diff_state_nu2], colorstring, alpha=alpha)
    plt.plot(tS, np.mean(r_i_k[:, active_diff_state_nu2], axis=1), colorstring, linewidth=2)
    plt.title('Active in different states')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)

    plt.subplot(2, 2, 4)
    plt.plot(tS, r_i_k[:, active_nu_inactive2], colorstring, alpha=alpha)
    plt.plot(tS, np.mean(r_i_k[:, active_nu_inactive2], axis=1), colorstring, linewidth=2)
    plt.title('Active-inactive')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)

    plt.suptitle('Thresholds of units based on their category')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    alpha = 0.05
    plt.figure('Field')
    plt.subplot(2, 2, 1)
    plt.plot(tS, m_mu[:, nu], colorstring)
    # for ind_trans in range(1, len(retrieved[0][cue])):
    #     plt.text(trans_time[ind_trans], 0.9,
    #              str(retrieved[0][cue][ind_trans]))
    plt.title('Overlap')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)


    plt.subplot(2, 2, 2)
    plt.plot(tS, h_i_k[:, active_same_state2], colorstring, alpha=alpha)
    plt.plot(tS, np.mean(h_i_k[:, active_same_state2], axis=1), colorstring, linewidth=2)
    plt.title('Active in the same state')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)


    plt.subplot(2, 2, 3)
    plt.plot(tS, h_i_k[:, active_diff_state_nu2], colorstring, alpha=alpha)
    plt.plot(tS, np.mean(h_i_k[:, active_diff_state_nu2], axis=1), colorstring, linewidth=2)
    plt.title('Active in different states')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)

    plt.subplot(2, 2, 4)
    plt.plot(tS, h_i_k[:, active_nu_inactive2], colorstring, alpha=alpha)
    plt.plot(tS, np.mean(h_i_k[:, active_nu_inactive2], axis=1), colorstring, linewidth=2)
    plt.title('Active-inactive')
    plt.axvline(trans_time[trans_num], color='k')
    plt.axvline(trans_time[trans_num+1], color='k')
    plt.axvline(trans_time[trans_num+2], color='k')
    plt.xlim(300, 1300)


    plt.suptitle('Fields to units based on their category')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.figure('snake')
    plt.plot(m_mu[:, mu], m_mu[:, nu], color=colorstring)

plt.close('all')
# plot_fun(133, 181)
# plot_fun(133, 35)
plot_fun(11, 97)
plot_fun(11, 39)
