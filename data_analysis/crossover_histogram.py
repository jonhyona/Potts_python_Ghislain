# coding=utf-8
"""Cues the model with all possible patterns and shows the evolution of the
main parameters, as well as statistics on patterns and transitions that occured
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Standard libraries
import numpy as np

# Local modules
import file_handling

plt.ion()
plt.close('all')

simulations = ['12257f9b2af7fdeaa5ebeec24b71b13c',
               '2999e6e4eede18f9212d8abdd146e7f4',
               '779e267d7fd11b394a96bc18ac9d2261']  # Just above the border
title_str = 'Just above the latching border'

simulations = ['644eba9257be9f7f7fef6c9a4b872b1e',
               'b48aa9b95e3182d7c2e0e20522a28bd2',
               'ff7857f17a0abd6792f5795bb97de966']  # Just below the border
title_str = 'Just below the latching border'



def plot_lamb_hist_gsln(simulation_list):
    labels = []
    # def plot_lamb_hist(simulation_list):
    for ind_key in range(len(simulation_list)):
        print('ind_key = %d' % ind_key)
        simulation_key = simulation_list[ind_key]
        (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
         cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
         g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
            file_handling.load_parameters(simulation_key)

        lamb = file_handling.load_overlap(simulation_key)
        labels.append(r"$g_A$=%.2f, w=%.1f, %d transitions" % (g_A, w, file_handling.event_counter(lamb, p)))

        lamb_1D = np.zeros(file_handling.event_counter(lamb, p))
        cpt = 0
        for ind_cue in range(len(lamb)):
            for ind_trans in range(len(lamb[ind_cue])):
                lamb_1D[cpt] = lamb[ind_cue][ind_trans]
                cpt += 1
        sns.distplot(lamb_1D)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Density')
    plt.title(title_str)
    plt.legend(labels)


def plot_lamb_hist_kwang_il(simulation_list):
    labels = []
    # def plot_lamb_hist(simulation_list):
    for ind_key in range(len(simulations)):
        print('ind_key = %d' % ind_key)
        simulation_key = simulations[ind_key]

        (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
         cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
         g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
            file_handling.load_parameters(simulation_key+'.pkl')

        labels.append(r"$g_A$=%.2f" % g_A)
        lamb = file_handling.load_overlap(simulation_key+'.txt')

        lamb_1D = np.zeros(file_handling.event_counter(lamb, p))
        cpt = 0
        for ind_cue in range(len(lamb)):
            for ind_trans in range(len(lamb[ind_cue])):
                lamb_1D[cpt] = lamb[ind_cue][ind_trans]
                cpt += 1
        sns.distplot(lamb_1D)
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Density')
        plt.title(r'w=' +str(w) + ', Gsln')
        plt.legend(labels)

plt.show()
