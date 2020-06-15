import matplotlib.pyplot as plt
import matplotlib.colors as colors
import file_handling
import numpy as np
import numpy.random as rd

plt.ion()
simulations = ['4406010109634386043', '8276126441040930693',
               '2037770147520999148']

ryom_data = ['seq_w1.4_gA0.0', 'seq_w1.4_gA0.5', 'seq_w1.4_gA1.0']
color_s = ['blue', 'orange', 'green']
color_s_ryom = ['navy', 'peru', 'darkolivegreen']

def event_counter(retrieved):
    res = 0
    for cue_ind in range(p):
        res += len(retrieved[cue_ind])
    return res


for ind_key in range(len(simulations)):
    print('ind_key = %d' % ind_key)
    simulation_key = simulations[ind_key]
    ryom_name = ryom_data[ind_key]
    (dt, tSim, N, S, p, num_fact, p_fact,
     dzeta, a_pf,
     eps,
     f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
     tau_3_B, g_A,
     beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
     russo2008_mode) = file_handling.load_parameters(simulation_key+'.pkl')

    retrieved_saved = file_handling.load_retrieved(simulation_key+'.txt')
    # retrieved_saved = file_handling.load_ryom_retrieved(ryom_name)
    cue_number = len(retrieved_saved)

    max_cycle = 7
    cycle_count = {}
    random_cycle_count = {}
    max_count = 0
    for size_cycle in range(1, max_cycle+1):
        for cue_ind in range(cue_number):
            sequence = []
            if cue_ind != retrieved_saved[cue_ind][0]:
                sequence.append(cue_ind)
            sequence += retrieved_saved[cue_ind]
            random_patterns = rd.randint(0, p, len(sequence))

            for ind_trans in range(len(sequence)-size_cycle):
                cycle = sequence[ind_trans: ind_trans+size_cycle]
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
    plt.xlim(1, 8)
    plt.ylim(1, 1000)
    cbar = plt.colorbar()
    plt.xlabel('Cycle length')
    plt.ylabel('Number of cycle repetition')

    plt.yscale('log')
    plt.title("Latching sequence")
    plt.subplot(212)
    plt.pcolor(XX, YY, random_data, norm=colors.LogNorm(vmin=1, vmax=5e3))
    plt.xlim(1, 8)
    plt.ylim(1, 1000)
    cbar = plt.colorbar()
    plt.xlabel('Cycle length')
    plt.ylabel('Number of cycle repetition')
    plt.title("Random sequence")
    plt.yscale('log')
    plt.suptitle(r"Repartion of cycles, $w$=%.2f, $g_A$=%.2f, %d transitions, Gsln" % (w, g_A, event_counter(retrieved_saved)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../Notes/.img/Repartition_of_cycles_w=%.2f_g_A=%.2f_%d_transitions_Gsln.png" % (w, g_A, event_counter(retrieved_saved)))
    plt.show()
    plt.close('all')
