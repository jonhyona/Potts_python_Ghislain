import matplotlib.pyplot as plt
import matplotlib.colors as colors
import file_handling
import numpy as np
import numpy.random as rd
import copy
import proba_tools

plt.ion()
plt.close('all')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16
HUGE_SIZE = 15
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=HUGE_SIZE)  # fontsize of the figure title


simulations = ['83058b3d6f4ce563cecce654468e59ec',
               '5fde28fc139c3252f8d52b513c7b2364',
               '6211c3984769aa0bde863c1fa97be8ef',
               '3ae4c5af2e17c42b644210bae0c6c88b',
               'f7fbf477d959473b676fd5e53a243e51',
               '0235947fe67f095afdce360a44932aa7',
               '3f9349f4d58a0590c6575920163dbd45',
               '252c00a8ee9a6dbb553e7166d460b4fe',
               '06381599d71447f5c7c667e3e3106f88',
               'e668a8d56be4b85ea0fe063c0511c617',
               '494a1f3016417558967fc452e83604b0',
               '5c135a6e2604d153a91e4fd757218b49',
               '12a26501a9dd07618c85bd6f324237ed',
               '1d13122682b8d57568e86741055d953b',
               'f61c95aad79795bbe476c2a6692025d5']

simulations = ['f2f842f51d5180f4eb55beb8efb61882',
               '625546bb732bd7bf3404e8a2c9193613',
               'fe71122c46ad4af9f201b9123f36ca42',
               '7cfd59eb4e3d3a26ec66e4249b22cfba',
               '001319a7dbc27bb929f6c6d00bc4f08d',
               '8b27f66e75c7f4f4427bbe59515c6e97',
               '7218cda81b1e89d0dfc660c0a18ff912',
               '03771e780bda036f8f2b8160bf2d85d4',
               'f35c969f14b35efe505be6e417c03656',
               'd9e7392b3817a1066541daa9309950ab',
               '4b5ccb6c6231655784281eed38749ade',
               'a0bfb97c0c519448fe9eac86a6c52a11']

simulations = ['f35c969f14b35efe505be6e417c03656']

ryom_data = ['seq_w1.4_gA0.0', 'seq_w1.4_gA0.5', 'seq_w1.4_gA1.0']
color_s = ['blue', 'orange', 'green']
color_s_ryom = ['navy', 'peru', 'darkolivegreen']


def event_counter(retrieved):
    res = 0
    for cue_ind in range(p):
        res += len(retrieved[cue_ind])
    return res


def get_eq_markov(retrieved, key):
    num_A, p_A, num_B, p_B, num_AB, p_AB, num_ABC, p_ABC, p_B_ABC, p_AB_ABC = \
        proba_tools.trio_prob_table(retrieved, key)
    proba_table = np.zeros((p, p))
    num_A = np.sum(num_AB, axis=1)
    occuring_A = num_A != 0
    proba_table[occuring_A, :] = num_AB[occuring_A, :] / num_A[occuring_A, None]
    retrieved_markov = copy.deepcopy(retrieved)

    for kick_seed in range(len(retrieved)):
        for cue_ind in range(p):
            retrieved_markov[kick_seed][cue_ind] = []
            if isinstance(retrieved[kick_seed][cue_ind], list) \
               and len(retrieved[kick_seed][cue_ind]) >= 3:
                # print(len(retrieved[kick_seed][cue_ind]))
                duration = len(retrieved[kick_seed][cue_ind])
                if cue_ind != retrieved[kick_seed][cue_ind][0]:
                    duration += 1
                retrieved_markov[kick_seed][cue_ind].append(cue_ind)
            if occuring_A[cue_ind]:
                prev_mu = cue_ind
                for ind_trans in range(duration-1):
                    try:
                        prev_mu = rd.choice(np.array(range(p)), 1,
                                            p=proba_table[prev_mu,
                                                          :].ravel())[0]
                    except ValueError:
                        print(prev_mu, a_pf)
                        break
                
                    retrieved_markov[kick_seed][cue_ind].append(prev_mu)
    return retrieved_markov


def get_eq_random(retrieved, key):
    retrieved_random = copy.deepcopy(retrieved)

    num_A, p_A, num_B, p_B, num_AB, p_AB, num_ABC, p_ABC, p_B_ABC, p_AB_ABC = \
        proba_tools.trio_prob_table(retrieved, key)
    num_A = np.sum(num_AB, axis=1)
    occuring_A = num_A != 0

    for kick_seed in range(len(retrieved)):
        for cue_ind in range(p):
            retrieved_random[kick_seed][cue_ind] = []
            if isinstance(retrieved[kick_seed][cue_ind], list) \
               and len(retrieved[kick_seed][cue_ind]) >= 3:
                # print(len(retrieved[kick_seed][cue_ind]))
                duration = len(retrieved[kick_seed][cue_ind])
                if cue_ind != retrieved[kick_seed][cue_ind][0]:
                    duration += 1
                retrieved_random[kick_seed][cue_ind].append(cue_ind)
            if occuring_A[cue_ind]:
                prev_mu = cue_ind
                for ind_trans in range(duration-1):
                    try:
                        prev_mu = rd.randint(0, p)
                    except ValueError:
                        print(prev_mu, a_pf)
                        break

                    retrieved_random[kick_seed][cue_ind].append(prev_mu)
    return retrieved_random


def plot_cycles(retrieved, simulation_key):
    (dt, tSim, N, S, p, num_fact, p_fact,
     dzeta, a_pf,
     eps,
     f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
     tau_3_B, g_A,
     beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
     russo2008_mode, _) = file_handling.load_parameters(simulation_key)

    # retrieved_saved = file_handling.load_ryom_retrieved(ryom_name)
    kick_seed = 0
    retrieved_saved = retrieved[kick_seed]
    cue_number = len(retrieved_saved)
    max_cycle = 11
    cycle_count = {}
    random_cycle_count = {}
    max_count = 0
    for size_cycle in range(1, max_cycle+1):
        for cue_ind in range(cue_number):
            sequence = []
            if cue_ind != retrieved_saved[cue_ind][0]:
                sequence.append(cue_ind)
            sequence += retrieved_saved[cue_ind]

            for ind_trans in range(len(sequence)-size_cycle):
                cycle = sequence[ind_trans: ind_trans+size_cycle]
                cycle = tuple(cycle)
                # print(cycle)
                if cycle in cycle_count:
                    cycle_count[cycle] += 1
                else:
                    cycle_count[cycle] = 1

    for cycle in list(cycle_count):
        if cycle_count[cycle] <= 1:
            cycle_count.pop(cycle)
    for cycle in cycle_count:
        max_count = max(cycle_count[cycle], max_count)
    bins = np.arange(1, max_count, 1, dtype=int)
    data = np.zeros((max_count+1, max_cycle+1))
    xx = np.arange(0, max_cycle+1, 1)
    yy = np.arange(0, max_count+1, 1)
    XX, YY = np.meshgrid(xx, yy)
    for cycle in cycle_count:
        data[cycle_count[cycle], len(cycle)] += 1
    plt.pcolor(XX, YY, data, norm=colors.LogNorm(vmin=1, vmax=5e3))
    plt.xlim(1, max_cycle)
    plt.ylim(1, 1000)
    cbar = plt.colorbar()
    plt.yscale('log')


for ind_key in range(1):
    print('ind_key = %d' % ind_key)
    simulation_key = simulations[ind_key]
    ryom_name = ryom_data[ind_key]
    retrieved = file_handling.load_retrieved_several(1, simulation_key)
    plt.subplot(311)
    plot_cycles(retrieved, simulation_key)
    plt.title("Latching sequence")

    retrieved_random = get_eq_random(retrieved, simulation_key)
    retrieved_markov = get_eq_markov(retrieved, simulation_key)

    plt.subplot(312)
    plt.title("Markov sequence")
    plot_cycles(retrieved_markov, simulation_key)
    plt.ylabel('Repetitions')

    plt.subplot(313)
    plt.title("Random sequence")
    plot_cycles(retrieved_random, simulation_key)
    plt.xlabel('Subsequence length')

    plt.tight_layout()

