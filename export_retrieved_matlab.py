import file_handling
import csv

simulations = ['f30d8a2438252005f6a9190c239c01c1']
n_seeds = [11]
for ind_key in range(len(simulations)):
    key = simulations[ind_key]
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode, muted_prop) \
     = file_handling.load_parameters(key)

    with open('data_analysis/' + key + '/matlab_retrieved.csv', mode='w') as f:
        writer = csv.writer(f)
        for kick_seed in range(n_seeds[ind_key]):
            retrieved = file_handling.load_retrieved(kick_seed, key)
            for cue_ind in range(p):
                eq_string = ''
                for ind_trans in range(len(retrieved[cue_ind])):
                    eq_string += chr(retrieved[cue_ind][ind_trans] + 8704)
                writer.writerow([eq_string])
