import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import scipy.stats as sts
# Local modules
import file_handling

plt.ion()
plt.close('all')


simulations = ['61df59f111b4dc5091f743d8e2dafabb']

color_s = ['blue', 'orange', 'green']


def plot_overlap(cue, key):
    overlaps = file_handling.load_evolution(cue, 0, key)

    print(overlaps)
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode, _) = \
        file_handling.load_parameters(key)

    tS = np.linspace(0, tSim, nSnap)
    # plt.title(r"$a_{pf}$=%.2f, $w$=%.1f, $g_A$=%.1f" % (a_pf, w, g_A))
    ax.set_title(r"$w$=%.1f, $a_{pf}$=%.2f" % (w, a_pf))

    # retrieved = file_handling.load_retrieved(0, key)[0]
    # trans_time = file_handling.load_transition_time(0, key)[0]
    # for ind_trans in range(len(trans_time)-1):
    #     ax.text((trans_time[ind_trans+1]+trans_time[ind_trans])/2, 0.9, str(retrieved[ind_trans]), fontsize=15)

    ax.plot(tS, overlaps)


n_sim = len(simulations)
g_A_s = [0., 0.5]
apf_s = [0., 0.05, 0.1, 0.2]
n_gA = len(g_A_s)
n_apf = len(apf_s)
fig = plt.figure('time_evolution_only_w0', constrained_layout=True)
gs = fig.add_gridspec(1, 1)
x0_ind = [0, 0, 1]
x1_ind = [1, 1, 2]
y0_ind = [0, 2, 1]
y1_ind = [2, 4, 3]
for ii in range(n_sim):
    key = simulations[ii]
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode, _) = \
        file_handling.load_parameters(key)
    ind_gA = [i for i in range(len(g_A_s)) if g_A_s[i] == g_A][0]
    ind_apf = [i for i in range(len(apf_s)) if apf_s[i] == a_pf][0]
    ax = plt.subplot(len(g_A_s), len(apf_s), ind_gA*len(apf_s) +
                     ind_apf + 1)

    # x0 = x0_ind[ii]
    # x1 = x1_ind[ii]
    # y0 = y0_ind[ii]
    # y1 = y1_ind[ii]
    # ax = fig.add_subplot(gs[x0:x1, y0:y1])
    ax.set_xlim(0, 4000)
    # ax.set_ylabel(r'Overlap $m_\mu$')
    # ax.set_xlabel('Time (ms)')
    plot_overlap(0, key)
# plt.tight_layout()
# plt.show()
