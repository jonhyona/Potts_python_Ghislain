import file_handling
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy.random as rd
import joypy

plt.ion()
plt.close('all')


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

cue = 0
sig_i_k = file_handling.load_activation_cue(0, 0, key)
m_mu = np.array(file_handling.load_evolution(cue, 0, key))

recorded = tS > 0.
tS = tS[recorded]
m_mu = m_mu[recorded, :]
sig_i_k = sig_i_k[recorded, :]

ind_trans = 5
mu = retrieved[0][0][ind_trans]
nu = retrieved[0][0][ind_trans+1]

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

alpha = 0.1
plt.subplot(2, 2, 1)
plt.plot(tS, m_mu[:, mu], 'tab:blue')
plt.plot(tS, m_mu[:, nu], 'tab:red')
plt.plot(tS, m_mu, ':k')
for ind_trans in range(1, len(retrieved[0][cue])):
    plt.text(trans_time[ind_trans], 0.9,
             str(retrieved[0][cue][ind_trans]))
plt.title('Overlap')


plt.subplot(2, 2, 2)
plt.plot(tS, sig_i_k[:, active_same_state], 'tab:purple', alpha=alpha)
plt.plot(tS, np.mean(sig_i_k[:, active_same_state], axis=1), 'tab:purple', linewidth=4)
plt.title('Active in the same state')

plt.subplot(2, 2, 3)
plt.plot(tS, sig_i_k[:, active_diff_state_mu], 'tab:blue', alpha=alpha)
plt.plot(tS, sig_i_k[:, active_diff_state_nu], 'tab:red', alpha=alpha)
plt.plot(tS, np.mean(sig_i_k[:, active_diff_state_mu], axis=1), 'tab:blue', linewidth=4)
plt.plot(tS, np.mean(sig_i_k[:, active_diff_state_nu], axis=1), 'tab:red', linewidth=4)
plt.title('Active in different states')

plt.subplot(2, 2, 4)
plt.plot(tS, sig_i_k[:, active_mu_inactive], 'tab:blue', alpha=alpha)
plt.plot(tS, sig_i_k[:, active_nu_inactive], 'tab:red', alpha=alpha)
plt.plot(tS, np.mean(sig_i_k[:, active_mu_inactive], axis=1), 'tab:blue', linewidth=4)
plt.plot(tS, np.mean(sig_i_k[:, active_nu_inactive], axis=1), 'tab:red', linewidth=4)
plt.title('Active-inactive')

plt.suptitle('Activation of units based on their category')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
