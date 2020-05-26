import matplotlib.pyplot as plt
import matplotlib.colors as colors
import file_handling
import numpy as np
import numpy.random as rd

dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm, a, U, \
    w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0, g, \
    random_seed, p_0, n_p, nSnap, \
    tS, tSnap, russo2008_mode, \
    J_i_j_k_l, ksi_i_mu, delta__ksi_i_mu__k, \
    transition_time, lamb, retrieved_saved, previously_retrieved_saved, \
    outsider_saved, max_m_mu_saved, max2_m_mu_saved, just_next_saved, \
    r_i_k_plot, m_mu_plot, sig_i_k_plot, theta_i_k_plot\
    = file_handling.load_data()

g_A_s = [0., 0.5, 1.]
color_s = ['blue', 'orange', 'green']

for ind_g_A in range(len(g_A_s)):
    g_A = g_A_s[ind_g_A]
    set_name = file_handling.get_set_name(dt, tSim, N, S, p, num_fact, p_fact,
                                          dzeta, a_pf, eps,
                                          f_russo, cm, a, U, w, tau_1, tau_2,
                                          tau_3_A, tau_3_B, g_A,
                                          beta, tau, t_0, g, random_seed, p_0,
                                          n_p, nSnap, russo2008_mode)

    dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm, a, U, \
        w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0, g, \
        random_seed, p_0, n_p, nSnap, \
        tS, tSnap, russo2008_mode, \
        J_i_j_k_l, ksi_i_mu, delta__ksi_i_mu__k, \
        transition_time, lamb, retrieved_saved, previously_retrieved_saved, \
        outsider_saved, max_m_mu_saved, max2_m_mu_saved, just_next_saved, \
        r_i_k_plot, m_mu_plot, sig_i_k_plot, theta_i_k_plot\
        = file_handling.load_data(set_name)

    retrieved_saved = np.array(retrieved_saved)
    previously_retrieved_saved = np.array(previously_retrieved_saved)
    random_patterns = rd.randint(0, p, len(lamb))
    cue_number = 0
    previous_t = -np.inf
    new_cue_ind = [0]
    for it in range(len(transition_time)):
        tt = transition_time[it]
        if tt < previous_t:
            cue_number += 1
            new_cue_ind.append(it)
        previous_t = tt
    # print(cue_number)
    new_cue_ind.append(len(lamb)+1)

    max_cycle = 7
    cycle_count = {}
    random_cycle_count = {}
    max_count = 0
    for size_cycle in range(1, max_cycle+1):
        for cue_ind in range(cue_number):
            (trans_0, trans_1) = (new_cue_ind[cue_ind], new_cue_ind[cue_ind+1])
            for ind_trans in range(trans_0, min(trans_1, len(lamb))-size_cycle+1):
                cycle = [previously_retrieved_saved[ind_trans]] \
                    + retrieved_saved[ind_trans:ind_trans+size_cycle-1].tolist()
                cycle = tuple(cycle)
                if cycle in cycle_count:
                    cycle_count[cycle] += 1
                else:
                    cycle_count[cycle] = 1

                cycle = random_patterns[ind_trans:ind_trans+size_cycle]
                cycle = tuple(cycle)
                if cycle in random_cycle_count:
                    random_cycle_count[cycle] += 1
                else:
                    random_cycle_count[cycle] = 1

    for cycle in list(cycle_count):
        if cycle_count[cycle] <= 1:
            cycle_count.pop(cycle)
    for cycle in list(random_cycle_count):
        if random_cycle_count[cycle] <= 1:
            random_cycle_count.pop(cycle)
    for cycle in cycle_count:
        max_count = max(cycle_count[cycle], max_count)
    bins = np.arange(1, max_count, 1, dtype=int)
    data = np.zeros((max_count+1, max_cycle+1))
    random_data = data.copy()
    xx = np.arange(0, max_cycle+1, 1)
    yy = np.arange(0, max_count+1, 1)
    XX, YY = np.meshgrid(xx, yy)
    for cycle in cycle_count:
        data[cycle_count[cycle], len(cycle)] += 1
    for cycle in random_cycle_count:
        random_data[random_cycle_count[cycle], len(cycle)] += 1
    data = data
    plt.subplot(211)
    plt.pcolor(XX, YY, data, norm=colors.LogNorm(vmin=1, vmax=5e3))
    plt.xlim(1, 6)
    plt.ylim(1, 400)
    cbar = plt.colorbar()
    plt.xlabel('Cycle length')
    plt.ylabel('Number of cycle repetition')
    
    plt.yscale('log')
    plt.title("Latching sequence")
    plt.subplot(212)
    plt.pcolor(XX, YY, random_data, norm=colors.LogNorm(vmin=1, vmax=5e3))
    plt.xlim(1, 6)
    plt.ylim(1, 400)
    cbar = plt.colorbar()
    plt.xlabel('Cycle length')
    plt.ylabel('Number of cycle repetition')
    plt.title("Random sequence")
    plt.yscale('log')
    plt.suptitle(r"Repartion of cycles, $w$=%.2f, $g_A$=%.2f, %d transitions" % (w, g_A, len(lamb)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
